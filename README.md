# Drawing Review Worker

Python worker for the Steel QC Hub app. Polls Supabase for queued drawing
review jobs, downloads PDFs, runs a deterministic rule engine plus AI analysis
(AWS D1.1, AWS D1.8, AISC 303), writes findings back to the database, and
uploads an annotated (marked) PDF.

## What it processes

1. Drawing review jobs (`drawing_review_jobs`)
2. Retryable failed jobs (exponential backoff)
3. Revision comparisons (`drawing_revision_comparisons`)
4. Spec document extraction (`drawing_spec_documents`)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your real keys
```

Required environment variables (see `.env.example`):

| Variable | Purpose |
|---|---|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role key (bypasses RLS) |
| `OPENAI_API_KEY` | OpenAI API key (default provider) |
| `ANTHROPIC_API_KEY` | Anthropic API key (if `AI_PROVIDER=anthropic`) |
| `AI_PROVIDER` | `openai` (default) or `anthropic` |
| `ORG_ID` | Optional: only process jobs for one organization |

## Run

```bash
python worker.py
```

The worker loops forever: it claims queued jobs, processes them through the
extraction → classification → analysis → marking pipeline, sends heartbeat
metrics every 60 seconds, and recovers stuck jobs.
