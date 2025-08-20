import sys
import os
import time
import argparse
import subprocess
from signal import SIGINT, SIGTERM

# ensure project root is on sys.path so package imports work when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def init_database():
    # create DB tables
    from realtime_anomaly_project.database.db_setup import setup_database
    setup_database()
    print("Database initialized (tables created if missing).")

def run_once_fetch_and_compute():
    # run a single fetch + compute cycle and exit
    from realtime_anomaly_project.data_ingestion.yahoo import run_once as fetch_once
    from realtime_anomaly_project.stats.simple import compute_anomalies
    from realtime_anomaly_project.data_ingestion.news_ingest import run_once as fetch_news_once

    print("Running one-shot data fetch (stocks + news)...")
    fetch_once()
    fetch_news_once()
    print("Running anomaly computations...")
    try:
        compute_anomalies()
    except Exception as e:
        print(f"compute_anomalies error: {e}")
    print("One-shot run complete.")

def start_scheduler():
    from realtime_anomaly_project.data_ingestion.scheduler import start as start_scheduler
    print("Starting scheduler (background jobs)...")
    sched = start_scheduler()
    return sched

def start_dashboard_subprocess(port=None):
    # Launch Streamlit using the same Python interpreter (avoids missing 'streamlit' on PATH)
    cmd = [sys.executable, "-m", "streamlit", "run", "realtime_anomaly_project/app/dashboard.py"]
    if port:
        cmd += ["--server.port", str(port)]
    try:
        proc = subprocess.Popen(cmd)
        print(f"Started Streamlit dashboard (pid={proc.pid}).")
        return proc
    except Exception as e:
        print(f"Failed to start Streamlit dashboard: {e}")
        return None

def main():
    p = argparse.ArgumentParser(description="Run the realtime_anomaly_project from one entrypoint.")
    p.add_argument("--one-shot", "-1", action="store_true", help="Run a single fetch+compute and exit.")
    p.add_argument("--no-scheduler", action="store_true", help="Do not start the scheduler.")
    p.add_argument("--dashboard", "-d", action="store_true", help="Also launch Streamlit dashboard in a subprocess.")
    p.add_argument("--dashboard-port", type=int, default=None, help="Port for Streamlit dashboard.")
    args = p.parse_args()

    init_database()

    if args.one_shot:
        run_once_fetch_and_compute()
        return

    dashboard_proc = None
    scheduler = None
    try:
        if not args.no_scheduler:
            scheduler = start_scheduler()

        if args.dashboard:
            dashboard_proc = start_dashboard_subprocess(port=args.dashboard_port)

        # keep main thread alive and respond to signals
        print("Entrypoint running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        print("Shutdown requested, stopping services...")
    finally:
        if scheduler:
            try:
                scheduler.shutdown(wait=False)
                print("Scheduler stopped.")
            except Exception as e:
                print(f"Scheduler shutdown error: {e}")
        if dashboard_proc:
            try:
                dashboard_proc.send_signal(SIGTERM)
                dashboard_proc.wait(timeout=5)
                print("Dashboard subprocess terminated.")
            except Exception:
                dashboard_proc.kill()
                print("Dashboard subprocess killed.")
        print("Exiting.")

if __name__ == "__main__":
    main()