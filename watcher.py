import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ingest import ingest_resumes
from evaluator import evaluate_candidate

class ResumeHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith((".pdf", ".txt")):
            print("\n📄 New resume detected")
            ingest_resumes()
            evaluate_candidate()

if __name__ == "__main__":
    print("👀 Watching 'resumes' folder...\n")

    event_handler = ResumeHandler()
    observer = Observer()
    observer.schedule(event_handler, path="resumes", recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
