import sys, traceback, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import sklearn
from realtime_anomaly_project.statistical_anomaly.unsupervised import compute_unsupervised_anomalies

print("Python:", sys.executable)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)

def df_map_from_storage():
    from realtime_anomaly_project.data_ingestion.yahoo import data_storage
    df_map = {}
    for ticker, df in (data_storage or {}).items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        df = df.copy()
        # ensure DatetimeIndex
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.dropna(subset=['datetime']).set_index('datetime')
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors='coerce')
                df = df[~df.index.isna()]
            except Exception:
                continue
        if df.index.tz is None:
            try:
                df.index = df.index.tz_localize('UTC')
            except Exception:
                df.index = df.index.tz_localize('UTC', ambiguous='NaT', nonexistent='shift_forward')
        # ensure lowercase 'close'
        if 'close' not in df.columns and 'Close' in df.columns:
            df = df.rename(columns={'Close': 'close'})
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])
        if not df.empty:
            df_map[ticker] = df.sort_index()
    return df_map

def df_map_from_db():
    from realtime_anomaly_project.database.db_setup import setup_database, StockData
    Session = setup_database()
    import collections
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

# try in-memory first then DB
df_map = df_map_from_storage()
if not df_map:
    print("In-memory store empty — trying DB")
    df_map = df_map_from_db()

print("Tickers available for anomaly run:", list(df_map.keys()))
for t, df in df_map.items():
    print(f" - {t}: shape={getattr(df,'shape',None)}, index.tz={getattr(df.index,'tz',None)}")
    try:
        print(df[['close']].head(3).to_string())
    except Exception as e:
        print("  preview error:", e)

try:
    results = compute_unsupervised_anomalies(df_map)
    print("Results:", results)
except Exception:
    print("Exception while running compute_unsupervised_anomalies:")
    traceback.print_exc()