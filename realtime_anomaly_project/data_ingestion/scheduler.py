import sys
import os
# ensure project root is on sys.path when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from apscheduler.schedulers.background import BackgroundScheduler
from data_ingestion.yahoo import run_once as fetch_data
from data_ingestion.news_ingest import run_once as fetch_news
from stats.simple import compute_anomalies
from config.settings import FETCH_MIN, STATS_MIN

def start():
    # Create a BackgroundScheduler instance
    scheduler = BackgroundScheduler()
    scheduler.add_job(fetch_data, 'interval', minutes=FETCH_MIN, id='fetch_data')
    scheduler.add_job(fetch_news, 'interval', minutes=FETCH_MIN, id='fetch_news')
    scheduler.add_job(compute_anomalies, 'interval', minutes=STATS_MIN, id='compute_anomalies')
    scheduler.start()
    print(f"Scheduler started: Fetching data every {FETCH_MIN} minutes, anomaly detection every {STATS_MIN} minutes.")
    return scheduler

if __name__ == "__main__":
    start()
