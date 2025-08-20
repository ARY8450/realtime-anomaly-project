from data_ingestion.scheduler import start
from utils.logging import setup_logging
from config.settings import FETCH_MIN, STATS_MIN

logger = setup_logging()

def run_pipeline():
    """ Start the pipeline for data ingestion, anomaly detection, and more """
    try:
        logger.info(f"Starting the Real-Time Anomaly Detection System...")
        
        # Start the scheduler to run tasks periodically
        scheduler = start()

        logger.info(f"System started: Fetching data every {FETCH_MIN} minutes and computing anomalies every {STATS_MIN} minutes.")

        # Keep the system running indefinitely
        while True:
            pass  # This keeps the main process running so the scheduler can keep working

    except Exception as e:
        logger.error(f"An error occurred in the pipeline: {e}")

if __name__ == "__main__":
    run_pipeline()
