import sys
import os

# ensure project root (parent of realtime_anomaly_project) is on sys.path so package imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from apscheduler.schedulers.background import BackgroundScheduler
from realtime_anomaly_project.data_ingestion.yahoo import run_once as fetch_data
from realtime_anomaly_project.data_ingestion.news_ingest import run_once as fetch_news
from realtime_anomaly_project.stats.simple import compute_anomalies
from realtime_anomaly_project.config.settings import FETCH_MIN, STATS_MIN

def start():
    # Create a BackgroundScheduler instance
    scheduler = BackgroundScheduler()

    # schedule fetch_data and fetch_news to run every FETCH_MIN minutes (use seconds for precise control)
    scheduler.add_job(fetch_data, 'interval', seconds=FETCH_MIN * 60, id='fetch_data', next_run_time=None)
    scheduler.add_job(fetch_news, 'interval', seconds=FETCH_MIN * 60, id='fetch_news', next_run_time=None)

    # schedule anomaly detection job
    scheduler.add_job(compute_anomalies, 'interval', seconds=STATS_MIN * 60, id='compute_anomalies', next_run_time=None)

    scheduler.start()
    print(f"Scheduler started: fetching every {FETCH_MIN} minutes, computing stats every {STATS_MIN} minutes.")
    return scheduler

if __name__ == "__main__":
    start()
