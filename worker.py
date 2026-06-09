import time

def get_next_job():
    # placeholder (we’ll connect this later to your app/database)
    return None

def process_job(job):
    print("Processing job:", job)

def main():
    print("Worker is running")

    while True:
        job = get_next_job()

        if job:
            process_job(job)
        else:
            print("No jobs found. Waiting...")
            time.sleep(5)

if __name__ == "__main__":
    main()
