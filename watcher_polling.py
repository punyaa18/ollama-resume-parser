import os
import time
from ingest import ingest_resumes
from evaluator import evaluate_candidate

RESUME_DIR = "resumes"
seen_files = set()

print("👀 Polling resumes folder every 5 seconds...\n")

while True:
    current_files = set(
        f for f in os.listdir(RESUME_DIR)
        if f.endswith((".pdf", ".txt"))
    )

    new_files = current_files - seen_files

    if new_files:
        for file in new_files:
            print(f"📄 New resume detected: {file}")
            ingest_resumes()
            evaluate_candidate()

        seen_files = current_files

    time.sleep(5)
