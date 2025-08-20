import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from realtime_anomaly_project.config import settings
from realtime_anomaly_project.database.db_setup import setup_database, StockData
from realtime_anomaly_project.data_ingestion import yahoo
import pandas as pd

Session = setup_database()

def tickers_in_db():
    with Session() as s:
        rows = s.query(StockData.ticker).distinct().all()
        return {r[0] for r in rows}

def fetch_with_retries(ticker: str, retries=3, backoff_base=2, per_try_sleep=1):
    """
    Try fetch_intraday up to `retries` times with exponential backoff.
    Returns normalized DataFrame on success, None on failure.
    """
    for attempt in range(1, retries + 1):
        try:
            df = yahoo.fetch_intraday(ticker)
            # fetch_intraday stores to data_storage and should persist, but ensure persistence here:
            if isinstance(df, pd.DataFrame) and not df.empty:
                try:
                    # persist/upsert helper in yahoo.py
                    rows = yahoo.persist_stock_data_upsert(ticker, df)
                    print(f"[{ticker}] persisted {rows} rows (attempt {attempt})")
                except Exception as e:
                    print(f"[{ticker}] persist error on attempt {attempt}: {e}")
                return df
            else:
                print(f"[{ticker}] empty fetch (attempt {attempt})")
        except Exception as e:
            print(f"[{ticker}] fetch exception (attempt {attempt}): {e}")
        # backoff
        sleep_for = backoff_base ** (attempt - 1) * per_try_sleep
        time.sleep(sleep_for)
    return None

def fetch_missing_all(batch_sleep=3, retries=3):
    present = tickers_in_db()
    to_fetch = [t for t in settings.TICKERS if t not in present]
    print("Tickers missing in DB:", to_fetch)
    results = {"fetched": [], "failed": []}

    for idx, t in enumerate(to_fetch, 1):
        print(f"Fetching ({idx}/{len(to_fetch)}): {t}")
        df = fetch_with_retries(ticker=t, retries=retries)
        if isinstance(df, pd.DataFrame) and not df.empty:
            results["fetched"].append(t)
        else:
            results["failed"].append(t)
        # small pause between tickers to be polite
        time.sleep(batch_sleep)

    print("Fetch summary:")
    print("  fetched:", results["fetched"])
    print("  failed:", results["failed"])
    return results

if __name__ == "__main__":
    fetch_missing_all()