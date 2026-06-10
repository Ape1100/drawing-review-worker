import abc
import io
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from dotenv import load_dotenv
from supabase import create_client, Client as SupabaseClient

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import camelot
    HAS_CAMELOT = True
except ImportError:
    HAS_CAMELOT = False

load_dotenv()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    supabase_url: str
    supabase_service_role_key: str
    ai_provider: str = "openai"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    org_id: Optional[str] = None
    poll_interval_seconds: int = 3
    max_pages: int = 80
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-haiku-20241022"
    jobs_table: str = "drawing_review_jobs"
    findings_table: str = "drawing_review_findings"
    drawings_bucket: str = "drawings-original"
    marked_bucket: str = "drawings-marked"
    worker_id: str = ""
    max_retries: int = 3
    retry_backoff_base: int = 60
    confidence_threshold: float = 0.3
    job_timeout_minutes: int = 30


def get_config() -> Config:
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

    provider = os.getenv("AI_PROVIDER", "openai").strip().lower()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    if provider == "openai" and not openai_key:
        raise RuntimeError("OPENAI_API_KEY is required when AI_PROVIDER=openai")
    if provider == "anthropic" and not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required when AI_PROVIDER=anthropic")

    return Config(
        supabase_url=url,
        supabase_service_role_key=key,
        ai_provider=provider,
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        org_id=os.getenv("ORG_ID", "").strip() or None,
        worker_id=os.getenv("WORKER_ID", "").strip() or f"worker-{uuid.uuid4().hex[:8]}",
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.3")),
        job_timeout_minutes=int(os.getenv("JOB_TIMEOUT_MINUTES", "30")),
    )


# ---------------------------------------------------------------------------
# AIProvider abstraction
# ---------------------------------------------------------------------------

class AIProvider(abc.ABC):
    @abc.abstractmethod
    def complete(self, system_prompt: str, user_content: str, temperature: float = 0.2) -> str:
        ...


class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, system_prompt: str, user_content: str, temperature: float = 0.2) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        return resp.choices[0].message.content or ""


class AnthropicProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-20241022"):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, system_prompt: str, user_content: str, temperature: float = 0.2) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        return resp.content[0].text if resp.content else ""


def make_provider(cfg: Config) -> AIProvider:
    if cfg.ai_provider == "anthropic":
        return AnthropicProvider(cfg.anthropic_api_key, cfg.anthropic_model)
    return OpenAIProvider(cfg.openai_api_key, cfg.openai_model)


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def supabase_client(cfg: Config) -> SupabaseClient:
    return create_client(cfg.supabase_url, cfg.supabase_service_role_key)


def update_job(sb: SupabaseClient, cfg: Config, job_id: str, patch: Dict[str, Any]) -> None:
    sb.table(cfg.jobs_table).update(patch).eq("id", job_id).execute()


def download_pdf_bytes(sb: SupabaseClient, cfg: Config, pdf_path: str) -> bytes:
    res = sb.storage.from_(cfg.drawings_bucket).download(pdf_path)
    if isinstance(res, (bytes, bytearray)):
        return bytes(res)
    if hasattr(res, "data") and isinstance(res.data, (bytes, bytearray)):
        return bytes(res.data)
    if hasattr(res, "read"):
        return res.read()
    raise RuntimeError("Unexpected storage download response type")


def insert_findings(
    sb: SupabaseClient, cfg: Config, job_id: str, org_id: str,
    findings: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not findings:
        return []
    rows = []
    for f in findings:
        rows.append({
            "job_id": job_id,
            "organization_id": org_id,
            "severity": f.get("severity", "info"),
            "category": f.get("category", "general"),
            "matched_text": (f.get("title") or f.get("matched_text") or "")[:500],
            "context_text": f.get("description") or f.get("context_text") or "",
            "page_number": f.get("page_number", 1),
            "confidence": float(f.get("confidence", 0.5)),
            "bbox": f.get("bbox"),
            "finding_status": "unreviewed",
            "source": f.get("source", "ai"),
        })
    res = sb.table(cfg.findings_table).insert(rows).execute()
    return getattr(res, "data", None) or []


# ---------------------------------------------------------------------------
# Stage 1: PDF extraction
# ---------------------------------------------------------------------------

def extract_pages(pdf_bytes: bytes, max_pages: int) -> List[Tuple[int, str]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = min(len(doc), max_pages)
    pages: List[Tuple[int, str]] = []
    for i in range(count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
        pages.append((i + 1, text))
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# PDF Type Detection (vector / hybrid / scanned)
# ---------------------------------------------------------------------------

def detect_pdf_type(pdf_bytes: bytes, max_pages: int = 10) -> Dict[str, Any]:
    """
    Analyze a PDF to determine if it is vector-based, scanned (image-only),
    or a hybrid (mix of text and images).
    Returns: {pdf_type, has_extractable_text, has_images, image_page_count, text_page_count}
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = min(len(doc), max_pages)
    text_pages = 0
    image_pages = 0

    for i in range(count):
        page = doc.load_page(i)
        text = (page.get_text("text") or "").strip()
        images = page.get_images(full=True)
        has_text = len(text) > 20
        has_imgs = len(images) > 0

        if has_text:
            text_pages += 1
        if has_imgs:
            image_pages += 1

    doc.close()

    has_extractable_text = text_pages > 0
    has_images = image_pages > 0

    if has_extractable_text and not has_images:
        pdf_type = "vector"
    elif has_extractable_text and has_images:
        pdf_type = "hybrid"
    elif not has_extractable_text and has_images:
        pdf_type = "scanned"
    else:
        pdf_type = "unknown"

    return {
        "pdf_type": pdf_type,
        "has_extractable_text": has_extractable_text,
        "has_images": has_images,
        "image_page_count": image_pages,
        "text_page_count": text_pages,
    }


# ---------------------------------------------------------------------------
# Schedule / Table extraction (pdfplumber + Camelot)
# ---------------------------------------------------------------------------

def extract_tables_pdfplumber(pdf_bytes: bytes, max_pages: int = 80) -> List[Dict[str, Any]]:
    """
    Use pdfplumber to extract tables from a PDF. Returns a list of
    {page_number, table_index, headers, rows} dicts.
    Falls back gracefully if pdfplumber is not installed.
    """
    if not HAS_PDFPLUMBER:
        return []

    tables: List[Dict[str, Any]] = []
    try:
        pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
        for i, page in enumerate(pdf.pages[:max_pages]):
            page_tables = page.extract_tables() or []
            for t_idx, table in enumerate(page_tables):
                if not table or len(table) < 2:
                    continue
                headers = [str(c).strip() if c else "" for c in table[0]]
                rows = []
                for row in table[1:]:
                    rows.append([str(c).strip() if c else "" for c in row])
                tables.append({
                    "page_number": i + 1,
                    "table_index": t_idx,
                    "headers": headers,
                    "rows": rows,
                })
        pdf.close()
    except Exception as e:
        print(f"  [pdfplumber] Table extraction error (non-fatal): {e}")

    return tables


def extract_tables_camelot(pdf_path: str, max_pages: int = 20) -> List[Dict[str, Any]]:
    """
    Use Camelot for high-fidelity table extraction from vector PDFs.
    Requires a file path on disk. Falls back gracefully if Camelot not installed.
    """
    if not HAS_CAMELOT:
        return []

    tables: List[Dict[str, Any]] = []
    try:
        page_range = f"1-{max_pages}"
        camelot_tables = camelot.read_pdf(pdf_path, pages=page_range, flavor="lattice")
        for t in camelot_tables:
            df = t.df
            if df.empty or len(df) < 2:
                continue
            headers = [str(c).strip() for c in df.iloc[0].tolist()]
            rows = []
            for _, row in df.iloc[1:].iterrows():
                rows.append([str(c).strip() for c in row.tolist()])
            tables.append({
                "page_number": t.page,
                "table_index": t.order,
                "headers": headers,
                "rows": rows,
                "accuracy": round(t.accuracy, 1),
            })
    except Exception as e:
        print(f"  [camelot] Table extraction error (non-fatal): {e}")

    return tables


def format_tables_for_analysis(tables: List[Dict[str, Any]]) -> str:
    """Convert extracted tables into a text representation for AI analysis."""
    if not tables:
        return ""
    parts = []
    for t in tables[:20]:
        header_str = " | ".join(t["headers"])
        row_strs = [" | ".join(r) for r in t["rows"][:50]]
        parts.append(
            f"[Table on Page {t['page_number']}]\n"
            f"{header_str}\n"
            + "\n".join(row_strs)
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Confidence threshold filtering
# ---------------------------------------------------------------------------

def filter_findings_by_confidence(
    findings: List[Dict[str, Any]],
    threshold: float,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Filter findings below the confidence threshold.
    Critical findings are never filtered regardless of confidence.
    Returns (kept_findings, filtered_count).
    """
    if threshold <= 0:
        return findings, 0

    kept = []
    filtered = 0
    for f in findings:
        conf = float(f.get("confidence", 0.5))
        sev = f.get("severity", "info")
        if sev == "critical" or conf >= threshold:
            kept.append(f)
        else:
            filtered += 1

    return kept, filtered


# ---------------------------------------------------------------------------
# Worker metrics recording
# ---------------------------------------------------------------------------

class MetricsRecorder:
    """Records worker-level metrics to the database for monitoring."""

    def __init__(self, sb: SupabaseClient, worker_id: str):
        self._sb = sb
        self._worker_id = worker_id
        self._last_heartbeat = 0.0

    def record(
        self,
        metric_type: str,
        org_id: Optional[str] = None,
        job_id: Optional[str] = None,
        value_int: Optional[int] = None,
        value_json: Optional[Dict] = None,
    ) -> None:
        try:
            row: Dict[str, Any] = {
                "worker_id": self._worker_id,
                "metric_type": metric_type,
                "recorded_at": now_iso(),
            }
            if org_id:
                row["organization_id"] = org_id
            if job_id:
                row["job_id"] = job_id
            if value_int is not None:
                row["value_int"] = value_int
            if value_json is not None:
                row["value_json"] = json.dumps(value_json)
            self._sb.table("drawing_review_worker_metrics").insert(row).execute()
        except Exception as e:
            print(f"  [metrics] Failed to record {metric_type}: {e}")

    def heartbeat(self, queue_depth: int = 0) -> None:
        now = time.time()
        if now - self._last_heartbeat < 60:
            return
        self._last_heartbeat = now
        self.record("heartbeat", value_int=queue_depth)

    def job_completed(self, org_id: str, job_id: str, duration_ms: int, findings_count: int) -> None:
        self.record("job_completed", org_id, job_id, duration_ms, {
            "findings_count": findings_count,
            "duration_ms": duration_ms,
        })

    def job_failed(self, org_id: str, job_id: str, error: str, retry_count: int) -> None:
        self.record("job_failed", org_id, job_id, retry_count, {
            "error": error[:500],
            "retry_count": retry_count,
        })

    def job_retried(self, org_id: str, job_id: str, retry_count: int) -> None:
        self.record("job_retried", org_id, job_id, retry_count)


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

def should_retry(job: Dict[str, Any], max_retries: int) -> bool:
    """Determine if a failed job is eligible for automatic retry."""
    retry_count = int(job.get("retry_count", 0))
    if retry_count >= max_retries:
        return False
    error = job.get("error_message") or ""
    non_retryable = [
        "Missing original_file_path",
        "Missing organization_id",
        "No extractable text found",
    ]
    return not any(msg in error for msg in non_retryable)


def schedule_retry(
    sb: SupabaseClient, cfg: Config, job: Dict[str, Any]
) -> bool:
    """
    Schedule a failed job for retry with exponential backoff.
    Returns True if retry was scheduled, False if not eligible.
    """
    retry_count = int(job.get("retry_count", 0))
    if retry_count >= cfg.max_retries:
        return False

    backoff_seconds = cfg.retry_backoff_base * (2 ** retry_count)
    retry_at = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)

    update_job(sb, cfg, str(job["id"]), {
        "retry_count": retry_count + 1,
        "retry_after": retry_at.isoformat(),
        "last_error_at": now_iso(),
    })
    print(f"[job:{str(job['id'])[:8]}] Scheduled retry #{retry_count + 1} after {backoff_seconds}s")
    return True


def fetch_retryable_jobs(sb: SupabaseClient, cfg: Config) -> List[Dict[str, Any]]:
    """Fetch failed jobs whose retry_after timestamp has passed."""
    q = (
        sb.table(cfg.jobs_table)
        .select("*")
        .eq("status", "failed")
        .lt("retry_count", cfg.max_retries)
        .lte("retry_after", now_iso())
        .is_("deleted_at", None)
        .order("retry_after", desc=False)
        .limit(5)
    )
    if cfg.org_id:
        q = q.eq("organization_id", cfg.org_id)
    res = q.execute()
    return getattr(res, "data", None) or []


def recover_stuck_jobs(sb: SupabaseClient, cfg: Config) -> int:
    """
    Find jobs stuck in 'running' state longer than the timeout and mark them failed.
    Returns the number of jobs recovered.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=cfg.job_timeout_minutes)).isoformat()
    q = (
        sb.table(cfg.jobs_table)
        .select("id,organization_id,retry_count,started_at")
        .eq("status", "running")
        .lt("started_at", cutoff)
        .is_("deleted_at", None)
        .limit(10)
    )
    if cfg.org_id:
        q = q.eq("organization_id", cfg.org_id)
    res = q.execute()
    stuck = getattr(res, "data", None) or []

    for job in stuck:
        job_id = str(job["id"])
        print(f"[job:{job_id[:8]}] Recovering stuck job (running > {cfg.job_timeout_minutes}m)")
        update_job(sb, cfg, job_id, {
            "status": "failed",
            "error_message": f"Job timed out after {cfg.job_timeout_minutes} minutes",
            "failure_reason": "timeout",
            "last_error_at": now_iso(),
        })
        schedule_retry(sb, cfg, {**job, "error_message": "timeout"})

    return len(stuck)


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------

def get_queue_depth(sb: SupabaseClient, cfg: Config) -> int:
    """Return the number of jobs currently queued."""
    q = (
        sb.table(cfg.jobs_table)
        .select("id", count="exact")
        .eq("status", "queued")
        .is_("deleted_at", None)
    )
    if cfg.org_id:
        q = q.eq("organization_id", cfg.org_id)
    res = q.execute()
    return getattr(res, "count", 0) or 0


# ---------------------------------------------------------------------------
# Stage 2: Title block extraction
# ---------------------------------------------------------------------------

_TITLE_BLOCK_SYSTEM = (
    "You are a technical document parser. Extract title block fields from construction drawing text. "
    "Return ONLY valid JSON with these optional keys: "
    "sheet_number, sheet_title, revision, scale, drawn_by, checked_by, approved_by, "
    "issue_date, ifc_status (boolean), engineer_of_record. "
    "If a field is absent, omit it. Do not include explanatory text."
)


def extract_title_block(provider: AIProvider, first_page_text: str) -> Dict[str, Any]:
    if not first_page_text.strip():
        return {}
    try:
        raw = provider.complete(
            _TITLE_BLOCK_SYSTEM,
            f"Drawing text (first page):\n\n{first_page_text[:6000]}",
            temperature=0.1,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        print(f"  [title_block] extraction error (non-fatal): {e}")
        return {}


# ---------------------------------------------------------------------------
# Stage 3: Discipline classification
# ---------------------------------------------------------------------------

_DISCIPLINE_SYSTEM = (
    "You are a construction document classifier. Identify the engineering discipline of a drawing. "
    "Return ONLY valid JSON: {\"discipline\": \"<value>\", \"confidence\": <0-1>}. "
    "Allowed discipline values: structural, architectural, mechanical, shop_drawing, erection, "
    "connection, civil, unknown."
)

_DISCIPLINE_KEYWORDS: Dict[str, List[str]] = {
    "structural": ["ASTM", "AISC", "MOMENT FRAME", "SHEAR WALL", "COLUMN SCHEDULE", "BEAM SCHEDULE"],
    "shop_drawing": ["SHOP DETAIL", "FABRICATION", "BILL OF MATERIAL", "B.O.M", "SHOP DRAWING"],
    "erection": ["ERECTION PLAN", "ANCHOR BOLT", "ERECTION SEQUENCE"],
    "connection": ["CONNECTION DETAIL", "END PLATE", "BOLTED CONNECTION", "WELD SYMBOL"],
    "architectural": ["FLOOR PLAN", "ELEVATION", "FINISH SCHEDULE", "DOOR SCHEDULE"],
    "mechanical": ["HVAC", "DUCTWORK", "MECHANICAL PLAN", "PLUMBING"],
    "civil": ["GRADING PLAN", "UTILITY", "SITE PLAN", "DRAINAGE"],
}


def _keyword_discipline(text: str) -> Optional[str]:
    upper = text.upper()
    for disc, kws in _DISCIPLINE_KEYWORDS.items():
        if any(kw in upper for kw in kws):
            return disc
    return None


def classify_discipline(
    provider: AIProvider, title_block: Dict[str, Any], first_page_text: str
) -> Tuple[str, float]:
    keyword_hint = _keyword_discipline(first_page_text)
    try:
        context = (
            f"Title block: {json.dumps(title_block)}\n\n"
            f"First page text snippet:\n{first_page_text[:4000]}"
        )
        if keyword_hint:
            context += f"\n\nKeyword hint: {keyword_hint}"
        raw = provider.complete(_DISCIPLINE_SYSTEM, context, temperature=0.1)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        return str(result.get("discipline", "unknown")), float(result.get("confidence", 0.5))
    except Exception as e:
        print(f"  [discipline] classification error (non-fatal): {e}")
        return keyword_hint or "unknown", 0.4


# ---------------------------------------------------------------------------
# Stage 4: Discipline-specific analysis
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS: Dict[str, str] = {
    "structural": (
        "You are a senior structural steel QC reviewer. "
        "Review drawing text and identify findings related to:\n"
        "- Charpy V-notch / toughness requirements (temperature, energy, applicability)\n"
        "- Welding requirements (AWS D1.1/D1.8, WPS, preheat/interpass, process restrictions)\n"
        "- Material callouts and inconsistencies (ASTM grades, unusual specs)\n"
        "- Joint types (CJP, PJP, fillet, V-groove, J-groove, bevel, single/double)\n"
        "- NDT requirements (UT, RT, MT, PT, VT) and acceptance criteria\n"
        "- AISC 303 erection and fabrication requirements\n"
        "Return ONLY valid JSON."
    ),
    "shop_drawing": (
        "You are a senior shop drawing reviewer for structural steel fabrication. "
        "Review the drawing text and identify findings related to:\n"
        "- Bill of Material accuracy (member sizes, lengths, quantities)\n"
        "- Weld symbols and callouts (size, type, location)\n"
        "- Material grade callouts and substitutions\n"
        "- Connection details vs. contract drawings\n"
        "- Coating, surface prep, and galvanizing requirements\n"
        "- AWS D1.1 / AISC fabrication tolerances\n"
        "Return ONLY valid JSON."
    ),
    "connection": (
        "You are a structural connection design reviewer. "
        "Review the drawing text and identify findings related to:\n"
        "- Bolt specifications (grade, diameter, torque, installation method)\n"
        "- Weld details (joint type, access holes, backing bars)\n"
        "- Plate dimensions and material grades\n"
        "- Pre-qualified vs. demand-critical welds\n"
        "- AISC 358 seismic connection requirements\n"
        "Return ONLY valid JSON."
    ),
    "architectural": (
        "You are a construction document reviewer. "
        "Review the drawing text and identify findings related to:\n"
        "- Specification conflicts or missing references\n"
        "- Dimension and tolerance inconsistencies\n"
        "- Material callouts requiring clarification\n"
        "- Code compliance notes (CBC, IBC)\n"
        "Return ONLY valid JSON."
    ),
}

_DEFAULT_SYSTEM = (
    "You are a senior construction document QC reviewer. "
    "Review the drawing text and produce a JSON list of findings "
    "covering code compliance, specification conflicts, missing data, and safety-critical items. "
    "Return ONLY valid JSON."
)

# Review profile additional prompts — injected as project_specific_instructions
_PROFILE_PROMPTS: Dict[str, str] = {
    "general": "",
    "structural_steel": (
        "PROFILE: Structural Steel Review. "
        "Prioritize: member sizes (W-shapes, HSS, angles), connection details (bolts, welds), "
        "material grades (A36, A572, A992, A500, A53), camber and pre-fabrication notes, "
        "erection marks and piece marks, bearing plates, anchor rod patterns. "
        "Flag missing weld sizes, missing bolt grades, and missing or unclear connection details."
    ),
    "shop_drawing": (
        "PROFILE: Shop Drawing Review. "
        "Prioritize: fabrication dimensions vs. contract drawing dimensions, "
        "piece mark consistency, weld procedure references, NDT requirements, "
        "surface preparation and coating specs, shipping split marks, "
        "field vs. shop weld callouts. Flag any dimension conflicts with structural drawings."
    ),
    "aws_d1_1": (
        "PROFILE: AWS D1.1 Structural Welding Code Review. "
        "Apply Chapter 2 (Design), Chapter 3 (Prequalification), and Chapter 4 (Qualification). "
        "Flag: missing weld size, missing weld length, incomplete weld symbols (per AWS A2.4), "
        "missing WPS reference, missing preheat/interpass temperature for steels with CE > 0.40, "
        "missing backing bar specification for CJP groove welds, "
        "missing NDT callout per Table 6.1 inspection categories, "
        "missing Charpy CVN for demand-critical applications, "
        "joint detail not matching a prequalified joint (Table 3.2-3.7). "
        "Severity critical for items that affect structural integrity or code compliance."
    ),
    "aws_d1_8": (
        "PROFILE: AWS D1.8 Seismic Supplemental Welding Provisions Review. "
        "Apply alongside AWS D1.1 rules. "
        "Flag: missing demand-critical weld designation on beam-to-column connections in SMF/IMF, "
        "missing CVN Charpy requirements for demand-critical welds (27 ft-lb at -20°F), "
        "missing protected zone callout per AISC 341 Section D1.3, "
        "SCWB check: missing strong-column/weak-beam ratio verification note, "
        "continuity plate requirements not addressed for SMF connections, "
        "missing supplemental RBS (reduced beam section) dimensional tolerances, "
        "E70T-1 or E71T-1 electrodes specified without requiring low-hydrogen designation. "
        "Cross-reference AISC 341 Sections E3, E4, E6 for SMF, IMF, and EBF requirements."
    ),
    "aisc_certification": (
        "PROFILE: AISC 303 Code of Standard Practice Review. "
        "Flag: missing piece marks or erection mark scheme, "
        "missing anchor rod templates or setting plans, "
        "missing camber schedule or notation on beams over 40ft, "
        "field bolt installation method not specified (TC bolts vs. A325/F3125), "
        "mixed steel grades in a single member without explicit callout, "
        "missing surface preparation specification (SSPC-SP6, SP10, etc.), "
        "HDG (hot-dip galvanizing) not referenced where exposed-to-weather members are present, "
        "missing bolting lot callout for pre-installation verification."
    ),
}

_ANALYSIS_OUTPUT_SCHEMA = {
    "summary": "string - brief overview",
    "findings": [
        {
            "severity": "critical|warning|info",
            "category": "string",
            "title": "string",
            "description": "string",
            "page_number": "integer",
            "detail_ref": "string|null",
            "evidence": "string - REQUIRED: short VERBATIM snippet (3-15 words) copied character-for-character from the page text where the issue appears",
            "confidence": "number 0-1",
            "bbox": "object|null - {x0, y0, x1, y1} in PDF points (origin top-left) ONLY if actual coordinates are known, else null"
        }
    ]
}

# Appended to every analysis system prompt so findings carry locatable evidence.
_LOCATION_INSTRUCTION = (
    "\n\nLOCATION DATA (required for annotation placement):\n"
    "- For EVERY finding, set 'evidence' to a short snippet (3-15 words) copied VERBATIM, "
    "character-for-character, from the page text where the issue appears. Do not paraphrase, "
    "do not change capitalization or punctuation — the snippet is used to search the PDF page "
    "and draw a highlight box at the exact location.\n"
    "- Prefer distinctive snippets (callouts, dimension strings, note numbers) over generic words.\n"
    "- Set 'bbox' to {x0, y0, x1, y1} in PDF points (origin top-left) only when you can identify "
    "actual coordinates; otherwise set it to null and rely on the 'evidence' snippet."
)


def analyze_drawing(
    provider: AIProvider,
    discipline: str,
    pages: List[Tuple[int, str]],
    contract_prompt: str = "",
) -> Dict[str, Any]:
    system_prompt = _SYSTEM_PROMPTS.get(discipline, _DEFAULT_SYSTEM) + _LOCATION_INSTRUCTION

    page_packets = [
        {"page_number": pno, "text": txt[:12000]}
        for pno, txt in pages
        if txt.strip()
    ]

    user_payload: Dict[str, Any] = {
        "task": "Review this drawing package and produce QC findings.",
        "output_schema": _ANALYSIS_OUTPUT_SCHEMA,
        "pages": page_packets,
    }
    if contract_prompt:
        user_payload["project_specific_instructions"] = contract_prompt

    try:
        raw = provider.complete(system_prompt, json.dumps(user_payload), temperature=0.2)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "summary": "Model returned non-JSON output.",
            "findings": [{
                "severity": "warning",
                "category": "parse_error",
                "title": "AI response could not be parsed",
                "description": raw[:2000] if 'raw' in dir() else "Unknown error",
                "page_number": 1,
                "detail_ref": None,
                "evidence": None,
                "confidence": 0.2,
            }]
        }


# ---------------------------------------------------------------------------
# Deterministic Rule Engine (AWS D1.1 / D1.8 / AISC 303)
# Supplements AI findings with high-confidence pattern-matched checks.
# ---------------------------------------------------------------------------

import re as _re

_ALL_PROFILES = frozenset(["general", "structural_steel", "shop_drawing", "aws_d1_1", "aws_d1_8", "aisc_certification"])

@dataclass
class _Rule:
    rule_id: str
    category: str
    severity: str
    title: str
    description: str
    confidence: float
    # Pattern that must match somewhere in a page's text (case-insensitive)
    trigger_pattern: str
    # Pattern whose ABSENCE on the same page triggers the finding (optional)
    absence_pattern: Optional[str] = None
    # Additional context pattern that must also be present to trigger (optional)
    context_pattern: Optional[str] = None
    # Review profiles that activate this rule (empty = all profiles)
    profiles: Optional[List[str]] = None


# ── AWS D1.1 Structural Welding Code ────────────────────────────────────────
# ── AWS D1.8 Seismic Supplement ─────────────────────────────────────────────
# ── AISC 303 Code of Standard Practice ──────────────────────────────────────

_RULES: List[_Rule] = [
    # --- AWS D1.1: Weld completeness ---
    _Rule(
        rule_id="D1.1-FILLET-NO-SIZE",
        category="AWS D1.1 — Weld Callout",
        severity="warning",
        title="Fillet weld referenced without explicit size",
        description=(
            "A fillet weld is referenced on this page but no weld size (e.g. '3/8\"', '5/16\"', '1/4\"') "
            "appears in close proximity. Per AWS D1.1 Section 2.4, the weld size shall be shown on drawings."
        ),
        confidence=0.80,
        trigger_pattern=r"\bfillet\s+weld\b",
        absence_pattern=r'\b(?:\d+/\d+|[0-9]+(?:\.[0-9]+)?)\s*(?:"|in\b|inch)',
        profiles=["general", "structural_steel", "aws_d1_1", "aws_d1_8"],
    ),
    _Rule(
        rule_id="D1.1-CJP-NO-BACKING",
        category="AWS D1.1 — Joint",
        severity="warning",
        title="CJP groove weld without backing bar requirement",
        description=(
            "A Complete Joint Penetration (CJP) weld is called out, but no backing bar or back-gouging "
            "requirement is indicated. Per AWS D1.1 Section 3.12, CJP welds made from one side typically "
            "require a backing bar unless back-gouging is specified."
        ),
        confidence=0.75,
        trigger_pattern=r"\bCJP\b",
        absence_pattern=r"\b(?:backing\s+bar|back.goug|BU\s+bar|ceramic\s+backing)\b",
        profiles=["general", "structural_steel", "aws_d1_1", "aws_d1_8"],
    ),
    _Rule(
        rule_id="D1.1-WPS-MISSING",
        category="AWS D1.1 — WPS",
        severity="warning",
        title="Welding process specified without WPS reference",
        description=(
            "A welding process (SMAW, FCAW, GMAW, GTAW, SAW) is identified on this page but no "
            "Welding Procedure Specification (WPS) number is referenced. Per AWS D1.1 Section 6.4, "
            "all production welding shall be performed in accordance with a qualified WPS."
        ),
        confidence=0.72,
        trigger_pattern=r"\b(?:SMAW|FCAW|GMAW|GTAW|SAW|EH70T|E70|E71)\b",
        absence_pattern=r"\bWPS\b",
        profiles=["general", "structural_steel", "aws_d1_1", "aws_d1_8"],
    ),
    _Rule(
        rule_id="D1.1-NDT-NO-CRITERIA",
        category="AWS D1.1 — NDT",
        severity="warning",
        title="NDT requirement without acceptance criteria reference",
        description=(
            "An NDT requirement (UT, RT, MT, PT) is called out but no acceptance criteria table or "
            "code section is referenced (e.g., AWS D1.1 Table 6.1 for UT). Acceptance criteria are "
            "mandatory per AWS D1.1 Section 6."
        ),
        confidence=0.70,
        trigger_pattern=r"\b(?:UT|RT|MT|PT|VT|NDE|NDT|ultrasonic|radiograph|magnetic\s+particle)\b",
        absence_pattern=r"\b(?:Table\s+6\.|Section\s+6\.|acceptance\s+criteria|AWS\s+D1\.1)\b",
        profiles=["general", "structural_steel", "aws_d1_1", "aws_d1_8"],
    ),
    _Rule(
        rule_id="D1.1-PREHEAT-A514",
        category="AWS D1.1 — Preheat",
        severity="critical",
        title="High-strength steel (A514/A709) without preheat requirement",
        description=(
            "ASTM A514 or A709 Grade 100/HPS steel is referenced, but no preheat temperature is "
            "specified. AWS D1.1 Table 3.2 requires minimum preheat of 125°F (52°C) for A514/A709 "
            "HPS steel regardless of thickness."
        ),
        confidence=0.85,
        trigger_pattern=r"\bA(?:514|709\s+(?:Grade\s+100|HPS))\b",
        absence_pattern=r"\b(?:preheat|pre-heat|125.?F|52.?C)\b",
        profiles=["general", "structural_steel", "aws_d1_1", "aws_d1_8"],
    ),
    _Rule(
        rule_id="D1.1-INTERMITTENT-NO-LENGTH",
        category="AWS D1.1 — Weld Callout",
        severity="warning",
        title="Intermittent weld callout without length and pitch",
        description=(
            "An intermittent weld is referenced but no weld length and pitch (e.g. '2-6') are "
            "shown. AWS D1.1 Section 2.4.1.5 requires that intermittent welds show both the "
            "length of each weld and the center-to-center spacing."
        ),
        confidence=0.72,
        trigger_pattern=r"\bintermittent\s+weld\b",
        absence_pattern=r"\b\d+[-–]\d+\b",
        profiles=["general", "structural_steel", "aws_d1_1", "aws_d1_8"],
    ),

    # --- AWS D1.8: Seismic ---
    _Rule(
        rule_id="D1.8-DEMAND-CRITICAL-NO-CVN",
        category="AWS D1.8 — Seismic",
        severity="critical",
        title="Demand-critical weld without CVN toughness requirement",
        description=(
            "A demand-critical weld designation is present but Charpy V-Notch (CVN) toughness "
            "requirements are not specified. Per AWS D1.8 Section 6.3, filler metals for demand-critical "
            "welds must meet a minimum CVN of 20 ft-lbs at −20°F."
        ),
        confidence=0.88,
        trigger_pattern=r"\bdemand.critical\b",
        absence_pattern=r"\b(?:CVN|Charpy|ft.?lb|J\s+at|toughness)\b",
        profiles=["general", "structural_steel", "aws_d1_8"],
    ),
    _Rule(
        rule_id="D1.8-SMF-NO-CONTINUITY",
        category="AWS D1.8 — Seismic",
        severity="warning",
        title="Special Moment Frame (SMF) connection without continuity plate detail",
        description=(
            "An SMF or IMF connection is referenced, but continuity plates (column stiffeners) are "
            "not shown or referenced. AISC 358 Section 2.4 and AWS D1.8 require continuity plates "
            "to be shown explicitly in seismic moment frame connections."
        ),
        confidence=0.78,
        trigger_pattern=r"\b(?:SMF|IMF|Special\s+Moment\s+Frame|Intermediate\s+Moment\s+Frame)\b",
        absence_pattern=r"\b(?:continuity\s+plate|column\s+stiffener|transverse\s+stiffener)\b",
        profiles=["general", "structural_steel", "aws_d1_8"],
    ),
    _Rule(
        rule_id="D1.8-PROTECTED-ZONE",
        category="AWS D1.8 — Seismic",
        severity="info",
        title="Protected zone referenced — verify no attachments",
        description=(
            "A protected zone is identified on this page. Per AWS D1.8 Section 4.2, shear studs, "
            "decking attachments, and other non-structural welds are prohibited in protected zones. "
            "Verify that no attachments are shown or implied within the protected zone boundaries."
        ),
        confidence=0.82,
        trigger_pattern=r"\bprotected\s+zone\b",
        profiles=["general", "structural_steel", "aws_d1_8"],
    ),
    _Rule(
        rule_id="D1.8-SCWB-NO-CALCS",
        category="AWS D1.8 — Seismic",
        severity="info",
        title="Strong-Column-Weak-Beam (SCWB) ratio referenced",
        description=(
            "SCWB or strong-column-weak-beam is referenced on this sheet. Verify that the design "
            "calculations confirming AISC 358 Eq. (E3-1) compliance are on file."
        ),
        confidence=0.75,
        trigger_pattern=r"\b(?:SCWB|strong.column|weak.beam)\b",
        profiles=["general", "structural_steel", "aws_d1_8"],
    ),

    # --- AISC 303: Fabrication / Erection ---
    _Rule(
        rule_id="AISC303-PIECE-MARK",
        category="AISC 303 — Identification",
        severity="warning",
        title="Member shown without piece mark identification",
        description=(
            "Structural members appear to be referenced on this page without explicit piece marks "
            "(e.g., B1, C2, G4). Per AISC 303 Section 6.1, each fabricated piece shall be "
            "identified with a unique piece mark prior to shipment."
        ),
        confidence=0.65,
        trigger_pattern=r"\b(?:W\d+x\d+|HSS\d|WT\d|L\d+x\d|PL\s*\d+|MC\d)\b",
        absence_pattern=r"\b[A-Z]\d+\b",
        profiles=["general", "structural_steel", "shop_drawing", "aisc_certification"],
    ),
    _Rule(
        rule_id="AISC303-ANCHOR-NO-TEMPLATE",
        category="AISC 303 — Erection",
        severity="info",
        title="Anchor rod pattern without setting template reference",
        description=(
            "Anchor rods or anchor bolts are shown but no setting template or anchor bolt plan "
            "is referenced. AISC 303 Section 7.5 recommends anchor rod setting templates to "
            "control location tolerances during concrete placement."
        ),
        confidence=0.70,
        trigger_pattern=r"\b(?:anchor\s+(?:rod|bolt)|AB\s+PLAN)\b",
        absence_pattern=r"\b(?:template|anchor\s+bolt\s+plan|setting\s+plan)\b",
        profiles=["general", "structural_steel", "shop_drawing", "aisc_certification"],
    ),
    _Rule(
        rule_id="AISC303-FIELD-BOLT-NO-METHOD",
        category="AISC 303 — Erection",
        severity="warning",
        title="Field bolt specified without installation method",
        description=(
            "Field bolts are called out but no installation/tightening method is specified "
            "(snug-tight, pretensioned, or slip-critical). Per AISC 303 and RCSC, the required "
            "bolt installation category must be shown on the contract documents."
        ),
        confidence=0.78,
        trigger_pattern=r"\bfield\s+bolt\b",
        absence_pattern=r"\b(?:snug.tight|pretension|slip.critical|A325|A490|F3125|TC\s+bolt)\b",
        profiles=["general", "structural_steel", "shop_drawing", "aisc_certification"],
    ),

    # --- Material callouts ---
    _Rule(
        rule_id="MAT-A36-SEISMIC",
        category="Material — Specification",
        severity="info",
        title="A36 steel used in seismic lateral system",
        description=(
            "ASTM A36 is referenced alongside seismic design requirements. AISC 341 Section A3.1 "
            "requires A36 shapes to include a supplemental Ry factor (1.5 for A36) for seismic "
            "capacity design — verify this is accounted for in connection design."
        ),
        confidence=0.70,
        trigger_pattern=r"\bA36\b",
        context_pattern=r"\b(?:seismic|SDC|SCWB|SMF|IMF|EBF|demand.critical)\b",
    ),
    _Rule(
        rule_id="MAT-MIXED-GRADES",
        category="Material — Specification",
        severity="warning",
        title="Multiple steel grades referenced — verify compatibility",
        description=(
            "More than one ASTM steel grade appears on this page. Verify that mixed-grade "
            "connections account for matching filler metal strength requirements per AWS D1.1 "
            "Table 3.1 and that weaker material governs the connection design."
        ),
        confidence=0.65,
        trigger_pattern=r"\bA36\b",
        context_pattern=r"\b(?:A572|A992|A709|A913|A514)\b",
    ),
]


def run_rule_checks(pages: List[Tuple[int, str]], review_profile: str = "general") -> List[Dict[str, Any]]:
    """
    Run deterministic rules against each page's text, filtered by review_profile.
    Returns a list of findings in the same schema as AI analysis output.
    Deduplicates: each rule fires at most once per drawing (first page wins).
    """
    fired: set = set()
    findings: List[Dict[str, Any]] = []

    for pno, text in pages:
        if not text.strip():
            continue

        for rule in _RULES:
            if rule.rule_id in fired:
                continue

            if rule.profiles and review_profile not in rule.profiles:
                continue

            # Check trigger pattern
            match = _re.search(rule.trigger_pattern, text, _re.IGNORECASE)
            if not match:
                continue

            # Check required context pattern (must ALSO be present)
            if rule.context_pattern and not _re.search(rule.context_pattern, text, _re.IGNORECASE):
                continue

            # Check absence pattern (must NOT be present to fire)
            if rule.absence_pattern and _re.search(rule.absence_pattern, text, _re.IGNORECASE):
                continue

            # Capture the full text line containing the match so the bbox
            # resolver can locate it on the page via search_for().
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            if line_end == -1:
                line_end = len(text)
            matched_line = text[line_start:line_end].strip()

            fired.add(rule.rule_id)
            findings.append({
                "severity": rule.severity,
                "category": rule.category,
                "title": rule.title,
                "description": rule.description,
                "page_number": pno,
                "confidence": rule.confidence,
                "evidence": matched_line or match.group(0),
                "match_text": match.group(0),
                "rule_id": rule.rule_id,
                "bbox": None,
                "source": "rule_engine",
            })

    return findings


# ---------------------------------------------------------------------------
# Weld Callout Inventory + Steel Spec Summary
# Deterministic text scan of weld callout types and steel/material specs,
# plus an AI pass that builds a structured fabrication-spec summary.
# Results are stored in the job's summary JSONB under 'weld_inventory'
# and 'steel_specs'.
# ---------------------------------------------------------------------------

# Textual weld callout patterns (AWS A2.4 symbols are vector graphics and not
# detectable in the text layer; these match the accompanying text callouts).
_WELD_CALLOUT_PATTERNS: List[Tuple[str, str]] = [
    ("fillet",            r"\bfillet(?:\s+weld)?\b"),
    ("cjp",               r"\bCJP\b|\bcomplete\s+joint\s+penetration\b"),
    ("pjp",               r"\bPJP\b|\bpartial\s+joint\s+penetration\b"),
    ("groove",            r"\bgroove\s+weld\b"),
    ("v_groove",          r"\b(?:single|double)?\s*-?\s*V[\s-]*groove\b"),
    ("bevel_groove",      r"\bbevel[\s-]*(?:groove|weld)\b"),
    ("j_groove",          r"\bJ[\s-]*groove\b"),
    ("u_groove",          r"\bU[\s-]*groove\b"),
    ("flare_bevel",       r"\bflare[\s-]*(?:bevel|V)[\s-]*(?:groove|weld)?\b"),
    ("plug",              r"\bplug\s+weld\b"),
    ("slot",              r"\bslot\s+weld\b"),
    ("stud",              r"\b(?:weld(?:ed)?|shear|headed)\s+studs?\b|\bstud\s+weld(?:ing)?\b"),
    ("seal",              r"\bseal\s+weld\b"),
    ("tack",              r"\btack\s+weld\b"),
    ("spot",              r"\bspot\s+weld\b"),
    ("butt",              r"\bbutt\s+weld\b"),
    ("intermittent",      r"\b(?:intermittent|stitch)\s+weld\b"),
    ("field_weld",        r"\bfield\s+weld\b"),
    ("all_around",        r"\b(?:weld\s+)?all[\s-]*around\b"),
]

# Steel material grade patterns
_STEEL_GRADE_PATTERN = (
    r"\b(?:ASTM\s+)?(A(?:36|53|108|123|153|307|325|354|449|490|500|501|514|529|563|"
    r"572|588|618|653|709|913|992|1011|1043|1085)|F(?:1554|3125|436))"
    r"(?:[\s,]*(?:Gr(?:ade)?\.?\s*)?(50|55|60|65|42|46|36|105|[A-C])\b)?"
)

# Filler metal / electrode designations (AWS A5.x classifications)
_FILLER_METAL_PATTERN = (
    # SMAW: E7018, E10018-H4R — restricted to real AWS A5.1/A5.5 strength
    # series so sheet numbers like E1001 don't false-positive
    r"\bE(?:60|70|80|90|100|110|120)\d{2}(?:-[A-Z0-9]+)?\b"
    r"|\bE\d{2}T-?\d+[A-Z]?\b"               # FCAW: E71T-1, E70T-6
    r"|\bER\d{2}S-?\d\b"                     # GMAW/GTAW: ER70S-6
    r"|\bF\d[A-Z]\d-E[A-Z0-9]+\b"            # SAW flux-electrode: F7A2-EM12K
    r"|\bE(?:60|70|80|90|100|110|120)(?:XX|xx)\b"  # generic: E70XX
    r"|\blow[\s-]*hydrogen\b"
)

_CVN_LINE_PATTERN = r"(?:CVN|[Cc]harpy)"


def _grab_line(text: str, start: int, end: int) -> str:
    ls = text.rfind("\n", 0, start) + 1
    le = text.find("\n", end)
    if le == -1:
        le = len(text)
    return " ".join(text[ls:le].split())


def scan_weld_callouts(pages: List[Tuple[int, str]]) -> Dict[str, Any]:
    """Count textual weld callout types across all pages."""
    by_type: Dict[str, Dict[str, Any]] = {}
    for pno, text in pages:
        if not text.strip():
            continue
        for weld_type, pattern in _WELD_CALLOUT_PATTERNS:
            count = len(_re.findall(pattern, text, _re.IGNORECASE))
            if count:
                entry = by_type.setdefault(weld_type, {"count": 0, "pages": []})
                entry["count"] += count
                if pno not in entry["pages"]:
                    entry["pages"].append(pno)
    return {
        "total_callouts": sum(e["count"] for e in by_type.values()),
        "by_type": by_type,
        "note": "Counts are based on text callouts; graphical AWS A2.4 weld symbols are not text-detectable.",
    }


def scan_steel_specs(pages: List[Tuple[int, str]]) -> Dict[str, Any]:
    """Collect steel grades, CVN requirement lines, and filler metal mentions."""
    grades: Dict[str, Dict[str, Any]] = {}
    cvn_lines: List[Dict[str, Any]] = []
    filler: Dict[str, Dict[str, Any]] = {}

    for pno, text in pages:
        if not text.strip():
            continue

        for m in _re.finditer(_STEEL_GRADE_PATTERN, text, _re.IGNORECASE):
            spec = m.group(1).upper()
            grade_suffix = (m.group(2) or "").upper()
            key = f"{spec} Gr.{grade_suffix}" if grade_suffix else spec
            entry = grades.setdefault(key, {"count": 0, "pages": []})
            entry["count"] += 1
            if pno not in entry["pages"]:
                entry["pages"].append(pno)

        for m in _re.finditer(_CVN_LINE_PATTERN, text):
            line = _grab_line(text, m.start(), m.end())
            if line and not any(c["text"] == line for c in cvn_lines):
                cvn_lines.append({"page": pno, "text": line[:300]})

        for m in _re.finditer(_FILLER_METAL_PATTERN, text, _re.IGNORECASE):
            key = " ".join(m.group(0).upper().split())
            entry = filler.setdefault(key, {"count": 0, "pages": [], "example_line": ""})
            entry["count"] += 1
            if pno not in entry["pages"]:
                entry["pages"].append(pno)
            if not entry["example_line"]:
                entry["example_line"] = _grab_line(text, m.start(), m.end())[:300]

    return {
        "grades": grades,
        "cvn_lines": cvn_lines[:30],
        "filler_metal": filler,
    }


_STEEL_SPEC_SYSTEM = (
    "You are a structural steel fabrication specification analyst. "
    "You are given (a) raw scanner hits for steel grades, CVN/Charpy lines, and filler metal "
    "designations found in a drawing package, and (b) excerpts of the drawing text related to "
    "steel and steel fabrication. Produce a concise structured summary for a QC reviewer.\n"
    "Return ONLY valid JSON with this structure:\n"
    "{\n"
    "  \"steel_grades\": [{\"grade\": str, \"application\": str}],\n"
    "  \"cvn_requirements\": [str],\n"
    "  \"filler_metal_requirements\": [str],\n"
    "  \"fabrication_notes\": [str]\n"
    "}\n"
    "For steel_grades, state what each grade is used for when the text makes it clear "
    "(e.g. 'A992 - wide flange shapes'). For cvn_requirements, state temperature/energy values "
    "verbatim when present. Keep every list item to one sentence. Use empty lists when no data exists."
)

_SPEC_KEYWORD_PATTERN = (
    r"(?:steel|weld|CVN|charpy|electrode|filler|ASTM|fabricat|galvaniz|bolt|AISC|AWS)"
)


def summarize_steel_specs(
    provider: AIProvider,
    pages: List[Tuple[int, str]],
    spec_scan: Dict[str, Any],
) -> Dict[str, Any]:
    """One AI call that turns scanner hits + relevant text into a structured spec summary."""
    # Pull only the lines that mention steel/welding topics, capped for token budget
    relevant_lines: List[str] = []
    budget = 30000
    used = 0
    for pno, text in pages:
        if used >= budget:
            break
        for line in text.splitlines():
            line = line.strip()
            if len(line) >= 8 and _re.search(_SPEC_KEYWORD_PATTERN, line, _re.IGNORECASE):
                tagged = f"[p{pno}] {line[:300]}"
                relevant_lines.append(tagged)
                used += len(tagged)
                if used >= budget:
                    break

    payload = {
        "scanner_hits": spec_scan,
        "relevant_text_lines": relevant_lines,
    }
    raw = provider.complete(_STEEL_SPEC_SYSTEM, json.dumps(payload), temperature=0.1)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def build_steel_inventory(
    provider: AIProvider, pages: List[Tuple[int, str]]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run the weld callout scan and the steel spec scan + AI summary.
    Returns (weld_inventory, steel_specs); failures degrade gracefully.
    """
    weld_inventory: Dict[str, Any] = {}
    steel_specs: Dict[str, Any] = {}
    try:
        weld_inventory = scan_weld_callouts(pages)
    except Exception as e:
        print(f"  [inventory] weld callout scan failed (non-fatal): {e}")

    try:
        spec_scan = scan_steel_specs(pages)
        steel_specs = {"scanned": spec_scan}
        try:
            steel_specs["summary"] = summarize_steel_specs(provider, pages, spec_scan)
        except Exception as e:
            print(f"  [inventory] AI spec summary failed (non-fatal): {e}")
    except Exception as e:
        print(f"  [inventory] steel spec scan failed (non-fatal): {e}")

    return weld_inventory, steel_specs


# ---------------------------------------------------------------------------
# Revision Comparison (AI-powered diff between two drawing text corpora)
# ---------------------------------------------------------------------------

_REVISION_COMPARE_SYSTEM = (
    "You are a senior structural engineer performing a revision comparison between two versions "
    "of a construction drawing. Identify precisely what changed between revision A and revision B.\n"
    "Focus on:\n"
    "- Member sizes, dimensions, and material grades\n"
    "- Weld callouts, sizes, and types\n"
    "- Connection details and bolt specifications\n"
    "- NDT and inspection requirements\n"
    "- Notes, specifications, and general notes\n"
    "- Added or removed details and sections\n"
    "Return ONLY valid JSON with this structure:\n"
    "{\n"
    "  \"summary\": \"one-paragraph overview of changes\",\n"
    "  \"significance\": \"critical|significant|minor\",\n"
    "  \"added\": [{\"description\": str, \"significance\": \"critical|warning|info\", \"location\": str}],\n"
    "  \"removed\": [{\"description\": str, \"significance\": \"critical|warning|info\", \"location\": str}],\n"
    "  \"modified\": [{\"description\": str, \"from\": str, \"to\": str, \"significance\": \"critical|warning|info\", \"location\": str}],\n"
    "  \"unchanged_notes\": [str]\n"
    "}"
)


def compare_revisions(
    provider: AIProvider,
    pages_a: List[Tuple[int, str]],
    pages_b: List[Tuple[int, str]],
    label_a: str = "Rev A",
    label_b: str = "Rev B",
) -> Dict[str, Any]:
    """
    AI-powered comparison of two drawing text corpora.
    Returns structured diff: added, removed, modified.
    """
    def _format_pages(pages: List[Tuple[int, str]], max_chars: int = 20000) -> str:
        parts = []
        total = 0
        for pno, txt in pages:
            if not txt.strip():
                continue
            snippet = txt[:3000]
            parts.append(f"[Page {pno}]\n{snippet}")
            total += len(snippet)
            if total >= max_chars:
                break
        return "\n\n".join(parts)

    text_a = _format_pages(pages_a)
    text_b = _format_pages(pages_b)

    user_payload = {
        "task": f"Compare revision '{label_a}' (Drawing A) against revision '{label_b}' (Drawing B). "
                "Identify all changes — added items, removed items, and modifications.",
        "drawing_a": {"label": label_a, "text": text_a},
        "drawing_b": {"label": label_b, "text": text_b},
    }

    try:
        raw = provider.complete(_REVISION_COMPARE_SYSTEM, json.dumps(user_payload), temperature=0.1)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        return {
            "summary": f"Comparison failed: {e}",
            "significance": "minor",
            "added": [],
            "removed": [],
            "modified": [],
            "unchanged_notes": [],
            "error": str(e),
        }


def process_revision_comparison(
    sb: SupabaseClient, cfg: Config, provider: AIProvider, comp: Dict[str, Any]
) -> None:
    """Process a single queued revision comparison job."""
    comp_id: str = str(comp["id"])
    job_id_a: str = str(comp.get("job_id_a", ""))
    job_id_b: str = str(comp.get("job_id_b", ""))
    label_a: str = comp.get("label_a") or "Version A"
    label_b: str = comp.get("label_b") or "Version B"

    print(f"[rev_comp:{comp_id[:8]}] Comparing {label_a} vs {label_b}")

    try:
        sb.table("drawing_revision_comparisons").update({
            "status": "running",
            "started_at": now_iso(),
        }).eq("id", comp_id).execute()

        # Fetch both jobs
        def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
            res = sb.table(cfg.jobs_table).select("*").eq("id", job_id).single().execute()
            return (getattr(res, "data", None) or {}) if res else None

        job_a = _get_job(job_id_a)
        job_b = _get_job(job_id_b)

        if not job_a or not job_b:
            raise RuntimeError("One or both drawing jobs not found")

        # Download and extract text from both PDFs
        pdf_a = download_pdf_bytes(sb, cfg, job_a["original_file_path"])
        pdf_b = download_pdf_bytes(sb, cfg, job_b["original_file_path"])

        pages_a = extract_pages(pdf_a, cfg.max_pages)
        pages_b = extract_pages(pdf_b, cfg.max_pages)

        # Run AI comparison
        result = compare_revisions(provider, pages_a, pages_b, label_a, label_b)

        sb.table("drawing_revision_comparisons").update({
            "status": "completed",
            "result": json.dumps(result),
            "completed_at": now_iso(),
        }).eq("id", comp_id).execute()

        total_changes = len(result.get("added", [])) + len(result.get("removed", [])) + len(result.get("modified", []))
        print(f"[rev_comp:{comp_id[:8]}] Completed — {total_changes} changes found")

    except Exception as exc:
        error_msg = str(exc)[:2000]
        print(f"[rev_comp:{comp_id[:8]}] FAILED: {error_msg}")
        sb.table("drawing_revision_comparisons").update({
            "status": "failed",
            "error_message": error_msg,
        }).eq("id", comp_id).execute()


# ---------------------------------------------------------------------------
# Specification document processing + conflict detection
# ---------------------------------------------------------------------------

_SPEC_EXTRACT_SYSTEM = (
    "You are a senior structural engineer extracting requirements from a construction specification document.\n"
    "Extract all testable requirements and organize them into categories.\n"
    "Return ONLY valid JSON with this structure:\n"
    "{\n"
    '  "materials": [{"requirement": str, "section": str, "values": [str]}],\n'
    '  "ndt": [{"requirement": str, "section": str, "values": [str]}],\n'
    '  "coatings": [{"requirement": str, "section": str, "values": [str]}],\n'
    '  "weld_processes": [{"requirement": str, "section": str, "values": [str]}],\n'
    '  "bolt_grades": [{"requirement": str, "section": str, "values": [str]}],\n'
    '  "other": [{"requirement": str, "section": str, "values": [str]}],\n'
    '  "raw_summary": "one-paragraph summary of key requirements"\n'
    "}\n"
    "Focus on ASTM material designations, NDT methods and acceptance criteria, coating/surface prep specs, "
    "welding process and filler metal requirements, bolt grades and installation methods, and any other "
    "quantifiable or verifiable specification requirement."
)

_SPEC_CONFLICT_SYSTEM = (
    "You are a senior QC engineer comparing project specification requirements against drawing content.\n"
    "Identify conflicts where the drawing references a material, process, or standard that differs from "
    "what the specification requires.\n"
    "Return ONLY valid JSON array of conflicts:\n"
    "[\n"
    '  {"finding_category": str, "severity": "info|warning|critical", '
    '   "spec_requirement": str, "spec_section": str, "drawing_value": str, '
    '   "conflict_description": str, "page_number": int|null}\n'
    "]\n"
    "Return an empty array [] if no conflicts are found. Focus on material grade mismatches, "
    "missing NDT requirements, bolt grade discrepancies, coating specification gaps, and weld process conflicts."
)


def extract_spec_requirements(provider: AIProvider, pages: List[Tuple[int, str]]) -> Dict[str, Any]:
    text_parts = []
    total = 0
    for pno, txt in pages:
        if not txt.strip():
            continue
        snippet = txt[:4000]
        text_parts.append(f"[Page {pno}]\n{snippet}")
        total += len(snippet)
        if total >= 30000:
            break
    spec_text = "\n\n".join(text_parts)

    try:
        raw = provider.complete(_SPEC_EXTRACT_SYSTEM, json.dumps({"spec_text": spec_text}), temperature=0.1)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        return {"materials": [], "ndt": [], "coatings": [], "weld_processes": [],
                "bolt_grades": [], "other": [], "raw_summary": f"Extraction failed: {e}"}


def detect_spec_conflicts(
    provider: AIProvider,
    spec_requirements: Dict[str, Any],
    drawing_pages: List[Tuple[int, str]],
) -> List[Dict[str, Any]]:
    drawing_parts = []
    total = 0
    for pno, txt in drawing_pages:
        if not txt.strip():
            continue
        snippet = txt[:3000]
        drawing_parts.append(f"[Page {pno}]\n{snippet}")
        total += len(snippet)
        if total >= 20000:
            break
    drawing_text = "\n\n".join(drawing_parts)

    user_payload = {
        "spec_requirements": spec_requirements,
        "drawing_text": drawing_text,
    }

    try:
        raw = provider.complete(_SPEC_CONFLICT_SYSTEM, json.dumps(user_payload), temperature=0.1)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def process_spec_document(
    sb: SupabaseClient, cfg: Config, provider: AIProvider, doc: Dict[str, Any]
) -> None:
    doc_id = str(doc["id"])
    file_path = doc.get("file_path", "")
    org_id = doc.get("organization_id", "")
    print(f"[spec:{doc_id[:8]}] Processing spec: {doc.get('name', '')}")

    try:
        sb.table("drawing_spec_documents").update({
            "status": "processing",
        }).eq("id", doc_id).execute()

        pdf_bytes = download_pdf_bytes(sb, cfg, file_path)
        pages = extract_pages(pdf_bytes, cfg.max_pages)
        nonempty = sum(1 for _, t in pages if t.strip())

        if nonempty == 0:
            sb.table("drawing_spec_documents").update({
                "status": "failed",
                "error_message": "No extractable text found in specification PDF.",
            }).eq("id", doc_id).execute()
            return

        requirements = extract_spec_requirements(provider, pages)

        sb.table("drawing_spec_documents").update({
            "status": "completed",
            "extracted_requirements": json.dumps(requirements),
        }).eq("id", doc_id).execute()

        print(f"[spec:{doc_id[:8]}] Extracted requirements successfully")

        # Event-driven: check completed drawing jobs for conflicts
        _run_spec_conflicts_for_doc(sb, cfg, provider, doc, requirements)

    except Exception as exc:
        error_msg = str(exc)[:2000]
        print(f"[spec:{doc_id[:8]}] FAILED: {error_msg}")
        sb.table("drawing_spec_documents").update({
            "status": "failed",
            "error_message": error_msg,
        }).eq("id", doc_id).execute()


def _run_spec_conflicts_for_doc(
    sb: SupabaseClient, cfg: Config, provider: AIProvider,
    doc: Dict[str, Any], requirements: Dict[str, Any],
) -> None:
    org_id = doc.get("organization_id", "")
    drawing_set_id = doc.get("drawing_set_id")
    job_id = doc.get("job_id")

    query = sb.table(cfg.jobs_table).select("id,original_file_path,organization_id").eq("status", "completed").eq("organization_id", org_id)
    if drawing_set_id:
        query = query.eq("drawing_set_id", drawing_set_id)
    elif job_id:
        query = query.eq("id", job_id)
    else:
        query = query.limit(20)

    res = query.execute()
    jobs = getattr(res, "data", None) or []

    for dj in jobs:
        try:
            pdf_bytes = download_pdf_bytes(sb, cfg, dj["original_file_path"])
            pages = extract_pages(pdf_bytes, cfg.max_pages)
            conflicts = detect_spec_conflicts(provider, requirements, pages)

            for c in conflicts:
                sb.table("drawing_spec_conflicts").insert({
                    "organization_id": org_id,
                    "job_id": dj["id"],
                    "spec_document_id": doc["id"],
                    "finding_category": c.get("finding_category", "material"),
                    "severity": c.get("severity", "warning"),
                    "spec_requirement": c.get("spec_requirement", ""),
                    "spec_section": c.get("spec_section"),
                    "drawing_value": c.get("drawing_value", ""),
                    "conflict_description": c.get("conflict_description", ""),
                    "page_number": c.get("page_number"),
                    "status": "open",
                }).execute()

            if conflicts:
                print(f"[spec:{doc['id'][:8]}] Found {len(conflicts)} conflicts with job {dj['id'][:8]}")
        except Exception as e:
            print(f"[spec:{doc['id'][:8]}] Conflict detection failed for job {dj['id'][:8]}: {e}")


def _run_spec_conflicts_for_job(
    sb: SupabaseClient, cfg: Config, provider: AIProvider, job: Dict[str, Any]
) -> None:
    """Event-driven: when a drawing job completes, check against all completed specs."""
    org_id = job.get("organization_id", "")
    drawing_set_id = job.get("drawing_set_id")

    query = sb.table("drawing_spec_documents").select("*").eq("organization_id", org_id).eq("status", "completed")
    if drawing_set_id:
        query = query.eq("drawing_set_id", drawing_set_id)

    res = query.execute()
    specs = getattr(res, "data", None) or []

    if not specs:
        return

    try:
        pdf_bytes = download_pdf_bytes(sb, cfg, job["original_file_path"])
        pages = extract_pages(pdf_bytes, cfg.max_pages)
    except Exception:
        return

    for spec in specs:
        try:
            requirements = spec.get("extracted_requirements")
            if isinstance(requirements, str):
                requirements = json.loads(requirements)
            if not requirements:
                continue

            conflicts = detect_spec_conflicts(provider, requirements, pages)
            for c in conflicts:
                sb.table("drawing_spec_conflicts").insert({
                    "organization_id": org_id,
                    "job_id": job["id"],
                    "spec_document_id": spec["id"],
                    "finding_category": c.get("finding_category", "material"),
                    "severity": c.get("severity", "warning"),
                    "spec_requirement": c.get("spec_requirement", ""),
                    "spec_section": c.get("spec_section"),
                    "drawing_value": c.get("drawing_value", ""),
                    "conflict_description": c.get("conflict_description", ""),
                    "page_number": c.get("page_number"),
                    "status": "open",
                }).execute()

            if conflicts:
                print(f"[job:{job['id'][:8]}] Found {len(conflicts)} spec conflicts from spec {spec['id'][:8]}")
        except Exception as e:
            print(f"[job:{job['id'][:8]}] Spec conflict check failed: {e}")


def fetch_next_spec_document(sb: SupabaseClient, cfg: Config) -> Optional[Dict[str, Any]]:
    q = (
        sb.table("drawing_spec_documents")
        .select("*")
        .eq("status", "queued")
        .order("created_at", desc=False)
        .limit(1)
    )
    if cfg.org_id:
        q = q.eq("organization_id", cfg.org_id)
    res = q.execute()
    data = getattr(res, "data", None) or []
    return data[0] if data else None


# ---------------------------------------------------------------------------
# Stage 5: Marked PDF generation
# ---------------------------------------------------------------------------

# Severity → (stroke_color, fill_color, label_bg) in RGB 0-1 floats
_SEVERITY_COLORS = {
    "critical": (
        (0.85, 0.15, 0.15),   # red stroke
        (0.98, 0.90, 0.90),   # light red fill
        (0.85, 0.15, 0.15),   # label background
    ),
    "warning": (
        (0.90, 0.50, 0.05),   # orange stroke
        (0.99, 0.95, 0.88),   # light orange fill
        (0.90, 0.50, 0.05),
    ),
    "info": (
        (0.10, 0.45, 0.85),   # blue stroke
        (0.90, 0.95, 1.00),   # light blue fill
        (0.10, 0.45, 0.85),
    ),
}

_DEFAULT_COLOR = (
    (0.40, 0.40, 0.40),
    (0.95, 0.95, 0.95),
    (0.40, 0.40, 0.40),
)

# Margin around a text-search hit when no bbox is provided by the AI
_HIT_MARGIN = 4  # pts


def _find_text_rect(page: fitz.Page, text: str) -> Optional[fitz.Rect]:
    """Search for the first occurrence of text on a page, return its rect."""
    if not text or len(text.strip()) < 4:
        return None
    text = " ".join(text.split())  # collapse whitespace/newlines
    # Try the full text first, then progressively shorter prefixes
    for candidate in [text, text[:80], text[:40]]:
        hits = page.search_for(candidate.strip(), quads=False)
        if hits:
            r = hits[0]
            return fitz.Rect(r.x0 - _HIT_MARGIN, r.y0 - _HIT_MARGIN,
                             r.x1 + _HIT_MARGIN, r.y1 + _HIT_MARGIN)
    return None


def _candidate_snippets(finding: Dict[str, Any]) -> List[str]:
    """
    Ordered list of text snippets worth searching for on the page:
    verbatim evidence first (most precise), then matched_text/title,
    then key phrases pulled from the longer context/description.
    """
    snippets: List[str] = []

    def add(s: Any) -> None:
        if isinstance(s, str):
            s = s.strip()
            if len(s) >= 4 and s not in snippets:
                snippets.append(s)

    add(finding.get("evidence"))
    add(finding.get("match_text"))
    add(finding.get("matched_text"))
    add(finding.get("title"))
    add(finding.get("detail_ref"))

    # Quoted fragments in the description often cite drawing text verbatim
    context = finding.get("context_text") or finding.get("description") or ""
    for quoted in _re.findall(r"[\"\u201c']([^\"\u201d']{4,80})[\"\u201d']", context):
        add(quoted)

    # Fall back to short phrases from the context, longest first
    phrases = [p.strip() for p in _re.split(r"[.;\n]", context)]
    for p in sorted(phrases, key=len, reverse=True)[:3]:
        if 8 <= len(p) <= 80:
            add(p)

    return snippets[:8]


def _validated_bbox_rect(raw_bbox: Any, page_rect: fitz.Rect) -> Optional[fitz.Rect]:
    """
    Convert a stored/AI-provided bbox dict to a Rect, accepting it only if it
    is a plausible region: positive area, inside the page, and not covering
    most of the sheet (a common AI-hallucination pattern).
    """
    if not raw_bbox or not isinstance(raw_bbox, dict):
        return None
    try:
        rect = fitz.Rect(
            float(raw_bbox.get("x0", 0)), float(raw_bbox.get("y0", 0)),
            float(raw_bbox.get("x1", 0)), float(raw_bbox.get("y1", 0)),
        )
    except (TypeError, ValueError):
        return None
    if rect.is_empty or rect.x1 <= rect.x0 or rect.y1 <= rect.y0:
        return None
    if not page_rect.contains(rect):
        return None
    if rect.get_area() > 0.5 * page_rect.get_area():
        return None
    return rect


def _resolve_finding_rect(page: fitz.Page, finding: Dict[str, Any]) -> Optional[fitz.Rect]:
    """
    Best-effort location for a finding on its page:
    1. a provided bbox that passes validation,
    2. text search over candidate snippets.
    """
    rect = _validated_bbox_rect(finding.get("bbox"), page.rect)
    if rect is not None:
        return rect
    for snippet in _candidate_snippets(finding):
        rect = _find_text_rect(page, snippet)
        if rect is not None:
            return rect
    return None


def resolve_finding_bboxes(pdf_bytes: bytes, findings: List[Dict[str, Any]]) -> int:
    """
    Fill in finding['bbox'] (PDF-point coordinates) for every finding that can
    be located on its page, replacing invalid AI-provided boxes. Mutates the
    findings in place and returns how many ended up with a bbox.
    """
    if not findings:
        return 0
    located = 0
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for f in findings:
            pno = int(f.get("page_number") or 1)
            if not 1 <= pno <= len(doc):
                f["bbox"] = None
                continue
            page = doc.load_page(pno - 1)
            rect = _resolve_finding_rect(page, f)
            if rect is not None:
                f["bbox"] = {
                    "x0": round(rect.x0, 2), "y0": round(rect.y0, 2),
                    "x1": round(rect.x1, 2), "y1": round(rect.y1, 2),
                }
                located += 1
            else:
                f["bbox"] = None
    finally:
        doc.close()
    return located


def _place_finding_annotation(
    page: fitz.Page,
    finding_index: int,
    finding: Dict[str, Any],
    page_height: float,
) -> None:
    """
    Draw a numbered callout box for a single finding on a PDF page.
    Uses the AI-provided bbox if present; falls back to text search;
    falls back to a margin annotation strip on the right edge.
    """
    severity = finding.get("severity", "info")
    stroke, fill, label_bg = _SEVERITY_COLORS.get(severity, _DEFAULT_COLOR)
    num = finding_index + 1

    # --- Determine location: validated bbox, then multi-snippet text search ---
    rect = _resolve_finding_rect(page, finding)

    # Fallback: place a numbered badge in the right margin at an evenly-spaced Y position
    if rect is None:
        pw = page.rect.width
        ph = page_height
        badge_size = 18
        margin_x = pw - badge_size - 6
        # spread badges evenly down the right margin
        y_step = max(badge_size + 4, ph / max(20, 1))
        y_top = 30 + finding_index * y_step
        y_top = min(y_top, ph - badge_size - 4)
        rect = fitz.Rect(margin_x, y_top, margin_x + badge_size, y_top + badge_size)
        _draw_badge_only(page, rect, num, label_bg, stroke)
        return

    # --- Draw highlight box ---
    shape = page.new_shape()
    shape.draw_rect(rect)
    shape.finish(color=stroke, fill=fill, fill_opacity=0.25, width=1.5)
    shape.commit()

    # --- Draw numbered badge in top-left corner of the rect ---
    badge_r = 8
    badge_cx = rect.x0 + badge_r
    badge_cy = rect.y0 - badge_r / 2
    # keep badge inside page
    badge_cy = max(badge_cy, badge_r + 1)

    shape2 = page.new_shape()
    shape2.draw_circle(fitz.Point(badge_cx, badge_cy), badge_r)
    shape2.finish(color=label_bg, fill=label_bg, fill_opacity=1.0, width=0)
    shape2.commit()

    page.insert_text(
        fitz.Point(badge_cx - (5 if num < 10 else 7), badge_cy + 4),
        str(num),
        fontsize=9,
        color=(1, 1, 1),
        fontname="Helv",
    )


def _draw_badge_only(
    page: fitz.Page,
    rect: fitz.Rect,
    num: int,
    fill_color: Tuple,
    stroke_color: Tuple,
) -> None:
    shape = page.new_shape()
    shape.draw_circle(rect.tl + fitz.Point(rect.width / 2, rect.height / 2), rect.width / 2)
    shape.finish(color=stroke_color, fill=fill_color, fill_opacity=1.0, width=1.0)
    shape.commit()
    cx = rect.x0 + rect.width / 2
    cy = rect.y0 + rect.height / 2
    page.insert_text(
        fitz.Point(cx - (4 if num < 10 else 6), cy + 4),
        str(num),
        fontsize=8,
        color=(1, 1, 1),
        fontname="Helv",
    )


def _draw_legend_page(doc: fitz.Document, findings: List[Dict[str, Any]]) -> None:
    """Append a summary legend page listing all numbered findings."""
    page = doc.new_page(width=612, height=792)  # letter
    y = 50

    page.insert_text(fitz.Point(50, y), "QC Review — Findings Legend", fontsize=16,
                     color=(0.1, 0.1, 0.1), fontname="hebo")
    y += 30
    page.insert_text(fitz.Point(50, y),
                     f"Total findings: {len(findings)}", fontsize=10,
                     color=(0.4, 0.4, 0.4), fontname="Helv")
    y += 20

    # Column headers
    page.insert_text(fitz.Point(50, y), "#", fontsize=9, color=(0.2, 0.2, 0.2), fontname="hebo")
    page.insert_text(fitz.Point(70, y), "Sev", fontsize=9, color=(0.2, 0.2, 0.2), fontname="hebo")
    page.insert_text(fitz.Point(110, y), "Pg", fontsize=9, color=(0.2, 0.2, 0.2), fontname="hebo")
    page.insert_text(fitz.Point(135, y), "Category", fontsize=9, color=(0.2, 0.2, 0.2), fontname="hebo")
    page.insert_text(fitz.Point(230, y), "Title", fontsize=9, color=(0.2, 0.2, 0.2), fontname="hebo")
    y += 14

    # Divider
    shape = page.new_shape()
    shape.draw_line(fitz.Point(50, y), fitz.Point(560, y))
    shape.finish(color=(0.7, 0.7, 0.7), width=0.5)
    shape.commit()
    y += 8

    sev_colors_text = {"critical": (0.7, 0.1, 0.1), "warning": (0.7, 0.35, 0.0), "info": (0.1, 0.35, 0.7)}

    for i, f in enumerate(findings):
        if y > 750:  # start new page if needed
            page = doc.new_page(width=612, height=792)
            y = 50

        sev = f.get("severity", "info")
        tc = sev_colors_text.get(sev, (0.3, 0.3, 0.3))
        pno = f.get("page_number", "?")
        cat = (f.get("category") or "")[:18]
        title = (f.get("matched_text") or f.get("title") or "")[:55]

        page.insert_text(fitz.Point(50, y), str(i + 1), fontsize=8, color=(0.1, 0.1, 0.1), fontname="Helv")
        page.insert_text(fitz.Point(70, y), sev[:4].upper(), fontsize=8, color=tc, fontname="hebo")
        page.insert_text(fitz.Point(110, y), str(pno), fontsize=8, color=(0.1, 0.1, 0.1), fontname="Helv")
        page.insert_text(fitz.Point(135, y), cat, fontsize=8, color=(0.1, 0.1, 0.1), fontname="Helv")
        page.insert_text(fitz.Point(230, y), title, fontsize=8, color=(0.1, 0.1, 0.1), fontname="Helv")
        y += 13


def generate_marked_pdf(
    pdf_bytes: bytes,
    findings: List[Dict[str, Any]],
) -> bytes:
    """
    Overlay finding annotations on the original PDF pages, then append
    a legend page. Returns the annotated PDF as bytes.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)

    # Group findings by page (1-indexed)
    by_page: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for idx, f in enumerate(findings):
        pno = int(f.get("page_number") or 1)
        pno = max(1, min(pno, page_count))
        by_page.setdefault(pno, []).append((idx, f))

    for pno, page_findings in by_page.items():
        page = doc.load_page(pno - 1)
        ph = page.rect.height
        for idx, f in page_findings:
            try:
                _place_finding_annotation(page, idx, f, ph)
            except Exception as e:
                print(f"  [markup] annotation error on page {pno}, finding {idx}: {e}")

    # Append legend
    if findings:
        _draw_legend_page(doc, findings)

    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True)
    doc.close()
    return buf.getvalue()


def upload_marked_pdf(
    sb: SupabaseClient,
    cfg: Config,
    org_id: str,
    job_id: str,
    marked_bytes: bytes,
    original_filename: str,
) -> str:
    """Upload marked PDF to storage and return its path."""
    stem = original_filename.rsplit(".", 1)[0] if "." in original_filename else original_filename
    marked_path = f"{org_id}/{job_id}/marked_{stem}.pdf"
    sb.storage.from_(cfg.marked_bucket).upload(
        marked_path,
        marked_bytes,
        {"content-type": "application/pdf", "upsert": "true"},
    )
    return marked_path


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def fetch_next_job(sb: SupabaseClient, cfg: Config) -> Optional[Dict[str, Any]]:
    q = (
        sb.table(cfg.jobs_table)
        .select("*")
        .eq("status", "queued")
        .is_("deleted_at", None)
        .order("queue_priority", desc=True)
        .order("created_at", desc=False)
        .limit(1)
    )
    if cfg.org_id:
        q = q.eq("organization_id", cfg.org_id)
    res = q.execute()
    data = getattr(res, "data", None) or []
    return data[0] if data else None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ms_since(start: float) -> int:
    return int((time.time() - start) * 1000)


def process_job(
    sb: SupabaseClient, cfg: Config, provider: AIProvider, job: Dict[str, Any],
    metrics: Optional[MetricsRecorder] = None,
) -> None:
    job_id: str = str(job["id"])
    pdf_path: str = job.get("original_file_path", "")
    org_id: str = job.get("organization_id", "")
    contract_prompt: str = (job.get("contract_prompt") or "").strip()
    original_filename: str = job.get("original_filename", "drawing.pdf")
    review_profile: str = (job.get("review_profile") or "general").strip()
    conf_threshold: float = float(job.get("confidence_threshold") or cfg.confidence_threshold)

    if not pdf_path:
        update_job(sb, cfg, job_id, {"status": "failed", "error_message": "Missing original_file_path", "failure_reason": "validation"})
        return
    if not org_id:
        update_job(sb, cfg, job_id, {"status": "failed", "error_message": "Missing organization_id", "failure_reason": "validation"})
        return

    is_retry = int(job.get("retry_count", 0)) > 0
    print(f"[job:{job_id[:8]}] Starting{'(retry #' + str(job.get('retry_count', 0)) + ')' if is_retry else ''} — file: {pdf_path}")

    job_start = time.time()
    extraction_ms = 0
    analysis_ms = 0
    marking_ms = 0

    try:
        update_job(sb, cfg, job_id, {
            "status": "running",
            "stage": "extracting",
            "progress": 5,
            "error_message": None,
            "started_at": now_iso(),
            "worker_id": cfg.worker_id,
        })

        sb.table(cfg.findings_table).delete().eq("job_id", job_id).execute()

        # Stage 1: Download + extract text
        ext_start = time.time()
        pdf_bytes = download_pdf_bytes(sb, cfg, pdf_path)

        # PDF type detection
        pdf_info = detect_pdf_type(pdf_bytes, min(cfg.max_pages, 10))
        print(f"[job:{job_id[:8]}] PDF type: {pdf_info['pdf_type']} (text:{pdf_info['text_page_count']}, img:{pdf_info['image_page_count']})")

        update_job(sb, cfg, job_id, {
            "pdf_type": pdf_info["pdf_type"],
            "has_extractable_text": pdf_info["has_extractable_text"],
            "has_images": pdf_info["has_images"],
            "image_page_count": pdf_info["image_page_count"],
        })

        pages = extract_pages(pdf_bytes, cfg.max_pages)
        nonempty = sum(1 for _, t in pages if t.strip())

        # Extract tables if pdfplumber is available
        extracted_tables: List[Dict[str, Any]] = []
        if HAS_PDFPLUMBER:
            extracted_tables = extract_tables_pdfplumber(pdf_bytes, cfg.max_pages)
            if extracted_tables:
                print(f"[job:{job_id[:8]}] Extracted {len(extracted_tables)} tables via pdfplumber")

        extraction_ms = _ms_since(ext_start)
        print(f"[job:{job_id[:8]}] Extracted {len(pages)} pages ({nonempty} non-empty) in {extraction_ms}ms")

        update_job(sb, cfg, job_id, {
            "stage": "classifying",
            "progress": 20,
            "total_pages": len(pages),
            "extraction_duration_ms": extraction_ms,
        })

        if nonempty == 0:
            update_job(sb, cfg, job_id, {
                "status": "failed",
                "error_message": f"No extractable text found in PDF (type: {pdf_info['pdf_type']}). "
                    + ("This appears to be a scanned document — OCR support coming soon." if pdf_info["pdf_type"] == "scanned" else ""),
                "failure_reason": "no_text",
            })
            return

        first_page_text = pages[0][1] if pages else ""

        # Stage 2: Title block extraction
        analysis_start = time.time()
        title_block = extract_title_block(provider, first_page_text)
        print(f"[job:{job_id[:8]}] Title block: {title_block}")

        update_job(sb, cfg, job_id, {
            "title_block": json.dumps(title_block) if title_block else None,
            "progress": 30,
        })

        # Stage 3: Discipline classification
        discipline, _ = classify_discipline(provider, title_block, first_page_text)
        print(f"[job:{job_id[:8]}] Discipline: {discipline}")

        update_job(sb, cfg, job_id, {
            "discipline": discipline,
            "stage": "analyzing",
            "progress": 40,
        })

        # Stage 4a: Deterministic rule checks
        rule_findings = run_rule_checks(pages, review_profile)
        print(f"[job:{job_id[:8]}] Rule engine: {len(rule_findings)} findings")

        # Stage 4b: AI analysis — include table data when available
        profile_prompt = _PROFILE_PROMPTS.get(review_profile, "")
        combined_contract = "\n\n".join(filter(None, [profile_prompt, contract_prompt]))

        table_context = format_tables_for_analysis(extracted_tables)
        if table_context:
            combined_contract += f"\n\n--- EXTRACTED SCHEDULE/TABLE DATA ---\n{table_context}"

        result = analyze_drawing(provider, discipline, pages, contract_prompt=combined_contract)

        # Stage 4c: Weld callout inventory + steel spec summary
        weld_inventory, steel_specs = build_steel_inventory(provider, pages)
        if weld_inventory.get("total_callouts"):
            print(f"[job:{job_id[:8]}] Weld callouts: {weld_inventory['total_callouts']} "
                  f"across {len(weld_inventory.get('by_type', {}))} types")
        grades_found = list((steel_specs.get("scanned") or {}).get("grades") or {})
        if grades_found:
            print(f"[job:{job_id[:8]}] Steel grades: {', '.join(grades_found[:10])}")

        analysis_ms = _ms_since(analysis_start)
        update_job(sb, cfg, job_id, {"progress": 75, "analysis_duration_ms": analysis_ms})

        # Tag and deduplicate
        ai_findings = [dict(f, source="ai") for f in (result.get("findings") or [])]
        deduped_ai: List[Dict[str, Any]] = []
        for af in ai_findings:
            rule_hit = next(
                (r for r in rule_findings
                 if r["category"] == af.get("category") and r["page_number"] == af.get("page_number")),
                None
            )
            if rule_hit is None:
                deduped_ai.append(af)
        all_findings = rule_findings + deduped_ai

        # Confidence threshold filtering
        findings, filtered_count = filter_findings_by_confidence(all_findings, conf_threshold)
        if filtered_count > 0:
            print(f"[job:{job_id[:8]}] Filtered {filtered_count} low-confidence findings (threshold={conf_threshold})")

        # Resolve bbox coordinates (validated AI bbox or text-search) before insert
        try:
            located = resolve_finding_bboxes(pdf_bytes, findings)
            print(f"[job:{job_id[:8]}] Located {located}/{len(findings)} findings on-page (bbox resolved)")
        except Exception as bbox_err:
            print(f"[job:{job_id[:8]}] bbox resolution failed (non-fatal): {bbox_err}")

        inserted = insert_findings(sb, cfg, job_id, org_id, findings)
        print(f"[job:{job_id[:8]}] Inserted {len(findings)} findings ({len(rule_findings)} rules + {len(deduped_ai)} AI, {filtered_count} filtered)")

        # Stage 5: Generate marked PDF
        mark_start = time.time()
        marked_path: Optional[str] = None
        if findings:
            update_job(sb, cfg, job_id, {"stage": "marking", "progress": 85})
            try:
                print(f"[job:{job_id[:8]}] Generating marked PDF...")
                marked_bytes = generate_marked_pdf(pdf_bytes, findings)
                marked_path = upload_marked_pdf(
                    sb, cfg, org_id, job_id, marked_bytes, original_filename
                )
                print(f"[job:{job_id[:8]}] Marked PDF uploaded: {marked_path}")
            except Exception as mark_err:
                print(f"[job:{job_id[:8]}] Marked PDF generation failed (non-fatal): {mark_err}")
        marking_ms = _ms_since(mark_start)

        # Build summary
        total_duration_ms = _ms_since(job_start)
        summary: Dict[str, Any] = {
            "text": result.get("summary", ""),
            "total_findings": len(findings),
            "by_severity": {
                "critical": sum(1 for f in findings if f.get("severity") == "critical"),
                "warning": sum(1 for f in findings if f.get("severity") == "warning"),
                "info": sum(1 for f in findings if f.get("severity") == "info"),
            },
            "by_category": {},
            "pages_processed": len(pages),
            "tables_extracted": len(extracted_tables),
            "findings_filtered": filtered_count,
            "pdf_type": pdf_info["pdf_type"],
            "weld_inventory": weld_inventory,
            "steel_specs": steel_specs,
        }
        for f in findings:
            cat = f.get("category", "other")
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1

        patch: Dict[str, Any] = {
            "status": "completed",
            "stage": "complete",
            "progress": 100,
            "summary": json.dumps(summary),
            "completed_at": now_iso(),
            "processing_duration_ms": total_duration_ms,
            "extraction_duration_ms": extraction_ms,
            "analysis_duration_ms": analysis_ms,
            "marking_duration_ms": marking_ms,
            "findings_filtered_count": filtered_count,
        }
        if marked_path:
            patch["marked_file_path"] = marked_path

        update_job(sb, cfg, job_id, patch)
        print(f"[job:{job_id[:8]}] Completed — {len(findings)} findings, discipline={discipline}, {total_duration_ms}ms")

        if metrics:
            metrics.job_completed(org_id, job_id, total_duration_ms, len(findings))

        # Stage 6: Event-driven spec conflict detection
        try:
            _run_spec_conflicts_for_job(sb, cfg, provider, job)
        except Exception as spec_err:
            print(f"[job:{job_id[:8]}] Spec conflict check failed (non-fatal): {spec_err}")

    except Exception as exc:
        error_msg = str(exc)[:4000]
        print(f"[job:{job_id[:8]}] FAILED: {error_msg}")
        update_job(sb, cfg, job_id, {
            "status": "failed",
            "error_message": error_msg,
            "failure_reason": "processing_error",
            "last_error_at": now_iso(),
            "processing_duration_ms": _ms_since(job_start),
        })
        if metrics:
            metrics.job_failed(org_id, job_id, error_msg, int(job.get("retry_count", 0)))
        if should_retry({"error_message": error_msg, **job}, cfg.max_retries):
            schedule_retry(sb, cfg, {**job, "error_message": error_msg})


def fetch_next_comparison(sb: SupabaseClient, cfg: Config) -> Optional[Dict[str, Any]]:
    q = (
        sb.table("drawing_revision_comparisons")
        .select("*")
        .eq("status", "queued")
        .order("created_at", desc=False)
        .limit(1)
    )
    if cfg.org_id:
        q = q.eq("organization_id", cfg.org_id)
    res = q.execute()
    data = getattr(res, "data", None) or []
    return data[0] if data else None


def main() -> None:
    cfg = get_config()
    sb = supabase_client(cfg)
    provider = make_provider(cfg)
    metrics = MetricsRecorder(sb, cfg.worker_id)

    print(f"Worker started | id={cfg.worker_id} | provider={cfg.ai_provider} | supabase={cfg.supabase_url}")
    print(f"  max_retries={cfg.max_retries} | confidence_threshold={cfg.confidence_threshold} | timeout={cfg.job_timeout_minutes}m")
    if cfg.org_id:
        print(f"  ORG_ID filter: {cfg.org_id}")
    print(f"  pdfplumber={'available' if HAS_PDFPLUMBER else 'not installed'} | camelot={'available' if HAS_CAMELOT else 'not installed'}")

    loop_count = 0
    last_recovery_check = 0.0
    consecutive_errors = 0

    while True:
        # Transient network/API errors (sleep/wake, Wi-Fi drops, Supabase
        # hiccups) must not kill the worker — back off and keep polling.
        try:
            loop_count += 1

            # Periodic heartbeat and stuck job recovery (every 60s)
            now = time.time()
            if now - last_recovery_check > 60:
                last_recovery_check = now
                queue_depth = get_queue_depth(sb, cfg)
                metrics.heartbeat(queue_depth)
                recovered = recover_stuck_jobs(sb, cfg)
                if recovered > 0:
                    print(f"[main] Recovered {recovered} stuck jobs")

            # Drawing review jobs take priority
            job = fetch_next_job(sb, cfg)
            if job:
                process_job(sb, cfg, provider, job, metrics)
                consecutive_errors = 0
                time.sleep(1)
                continue

            # Check for retryable failed jobs
            retryable = fetch_retryable_jobs(sb, cfg)
            if retryable:
                rjob = retryable[0]
                rjob_id = str(rjob["id"])
                print(f"[job:{rjob_id[:8]}] Auto-retrying (attempt #{rjob.get('retry_count', 0)})")
                metrics.job_retried(rjob.get("organization_id", ""), rjob_id, int(rjob.get("retry_count", 0)))
                update_job(sb, cfg, rjob_id, {"status": "queued", "retry_after": None})
                time.sleep(1)
                continue

            # Then revision comparisons
            comp = fetch_next_comparison(sb, cfg)
            if comp:
                process_revision_comparison(sb, cfg, provider, comp)
                time.sleep(1)
                continue

            # Then spec document processing
            spec_doc = fetch_next_spec_document(sb, cfg)
            if spec_doc:
                process_spec_document(sb, cfg, provider, spec_doc)
                time.sleep(1)
                continue

            consecutive_errors = 0
            time.sleep(cfg.poll_interval_seconds)

        except KeyboardInterrupt:
            print("[main] Interrupted — shutting down")
            raise
        except Exception as loop_err:
            consecutive_errors += 1
            backoff = min(300, cfg.poll_interval_seconds * (2 ** min(consecutive_errors, 6)))
            print(f"[main] Poll loop error (#{consecutive_errors}, retrying in {backoff}s): "
                  f"{type(loop_err).__name__}: {loop_err}")
            time.sleep(backoff)
            # Rebuild the Supabase client after repeated failures in case the
            # underlying HTTP connection pool is wedged.
            if consecutive_errors >= 3:
                try:
                    sb = supabase_client(cfg)
                    metrics = MetricsRecorder(sb, cfg.worker_id)
                    print("[main] Rebuilt Supabase client after repeated errors")
                except Exception as rebuild_err:
                    print(f"[main] Client rebuild failed: {rebuild_err}")


if __name__ == "__main__":
    main()
