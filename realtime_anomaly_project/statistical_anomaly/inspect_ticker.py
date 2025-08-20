import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from realtime_anomaly_project.data_ingestion.yahoo import fetch_intraday, data_storage
from realtime_anomaly_project.database.db_setup import setup_database, StockData
import pandas as pd

def inspect(ticker):
    df = fetch_intraday(ticker)
    print("fetch returned type:", type(df))
    if isinstance(df, pd.DataFrame):
        print("shape:", df.shape)
        print(df.head(5).to_string())
    Session = setup_database()
    with Session() as s:
        cnt = s.query(StockData).filter(StockData.ticker==ticker).count()
        print(f"DB rows for {ticker}:", cnt)

    # safe in-memory row count (avoid ambiguous DataFrame truth-value)
    df_mem = data_storage.get(ticker)
    if df_mem is None:
        in_memory_rows = 0
    else:
        try:
            in_memory_rows = len(df_mem)
        except Exception:
            # fallback for unexpected types
            in_memory_rows = 1

    print("in-memory rows:", in_memory_rows)

if __name__ == '__main__':
    import sys
    t = sys.argv[1] if len(sys.argv)>1 else "RELIANCE.NS"
    inspect(t)