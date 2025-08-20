import sys, os
# ensure project root is importable when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import collections
from realtime_anomaly_project.statistical_anomaly.unsupervised import compute_unsupervised_anomalies

# Option A: use in-memory store populated by yahoo.fetch (fast, for immediate runs)
def df_map_from_storage():
    from realtime_anomaly_project.data_ingestion.yahoo import data_storage
    df_map = {}
    for ticker, df in (data_storage or {}).items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        # ensure DatetimeIndex tz-aware UTC
        if 'datetime' in df.columns:
            df = df.copy()
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.dropna(subset=['datetime']).set_index('datetime')
        if not isinstance(df.index, pd.DatetimeIndex):
            continue
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        # ensure lowercase 'close'
        if 'close' not in df.columns and 'Close' in df.columns:
            df = df.rename(columns={'Close': 'close'})
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])
        if not df.empty:
            df_map[ticker] = df.sort_index()
    return df_map

# Option B: build map from persisted DB (recommended for historical runs)
def df_map_from_db():
    from realtime_anomaly_project.database.db_setup import setup_database, StockData
    Session = setup_database()
    df_map = {}
    with Session() as session:
        rows = session.query(StockData).order_by(StockData.timestamp).all()
        if not rows:
            return {}
        buckets = collections.defaultdict(list)
        for r in rows:
            buckets[r.ticker].append({
                "datetime": r.timestamp,
                "open": r.open_price,
                "high": r.high_price,
                "low": r.low_price,
                "close": r.close_price,
                "volume": r.volume
            })
        for ticker, recs in buckets.items():
            df = pd.DataFrame(recs)
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
            df = df.dropna(subset=['datetime']).set_index('datetime').sort_index()
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df = df.dropna(subset=['close'])
            if not df.empty:
                df_map[ticker] = df
    return df_map

def inspect_fetch(ticker="RELIANCE.NS"):
    """
    Run the fetcher for a single ticker and print diagnostics.
    Usage (from project root):
      python -m realtime_anomaly_project.statistical_anomaly.data_storage RELIANCE.NS
    """
    from realtime_anomaly_project.data_ingestion.yahoo import fetch_intraday, data_storage
    import pandas as pd

    df = fetch_intraday(ticker)
    print("fetch returned:", type(df))
    if isinstance(df, pd.DataFrame):
        print("shape:", df.shape)
        print("index sample:", df.index[:5])
        print("index tz:", getattr(df.index, "tz", None))
        print("columns:", list(df.columns))
        print("head:\n", df.head().to_string())
    else:
        print("fetch returned non-DataFrame or None:", df)
    print("data_storage contains key:", ticker, data_storage.get(ticker) is not None)

if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    inspect_fetch(t)

    # prefer in-memory store for immediate results; fall back to DB
    df_map = df_map_from_storage()
    if not df_map:
        print("In-memory storage empty — trying DB...")
        df_map = df_map_from_db()

    if not df_map:
        print("No data available to run unsupervised anomaly detection. Run the fetcher first.")
        print("One-shot fetch: python -m realtime_anomaly_project.data_ingestion.yahoo")
        raise SystemExit(1)

    results = compute_unsupervised_anomalies(df_map)
    print("Unsupervised anomaly results (per-ticker):")
    for t, scores in results.items():
        print(t, scores)