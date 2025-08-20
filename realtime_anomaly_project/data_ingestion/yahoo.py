import sys
import os
import time
from datetime import datetime, timedelta, timezone

# ensure project root (parent of realtime_anomaly_project) is on sys.path so package imports resolve
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sqlalchemy.orm import Session
from realtime_anomaly_project.database.db_setup import StockData, setup_database

# lightweight in-memory store
data_storage = {}

# Initialize database session
SessionLocal = setup_database()
session = SessionLocal()

def _parse_lookback_days(lookback: str) -> int:
    """Parse strings like '60d' -> 60. Fallback to 60."""
    try:
        if isinstance(lookback, str) and lookback.endswith('d'):
            return int(lookback[:-1])
        return int(lookback)
    except Exception:
        return 60

def fetch_2m_for_range(ticker: str, start: datetime, end: datetime, max_chunk_days: int = 7):
    """
    Best-effort: fetch 2m bars by chunking into <= max_chunk_days windows and concatenating.
    Yahoo may still refuse older 2m data; check returned DataFrame for gaps.
    """
    import pandas as pd
    import yfinance as yf

    cur = start
    dfs = []
    while cur < end:
        chunk_end = min(cur + timedelta(days=max_chunk_days), end)
        try:
            df = yf.download(
                ticker,
                start=cur.strftime("%Y-%m-%d"),
                end=(chunk_end + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="2m",
                progress=False,
                auto_adjust=True,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                dfs.append(df)
        except Exception as exc:
            print(f"Chunk fetch error for {ticker} {cur.date()} - {chunk_end.date()}: {exc}")
        cur = chunk_end
        time.sleep(1)  # polite pause between requests

    if not dfs:
        return None

    full = pd.concat(dfs).drop_duplicates().sort_index()
    # normalize to common schema used across project
    full = full.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
    })
    full = full.reset_index().rename(columns={"index": "datetime"})
    full['ts'] = (pd.to_datetime(full['datetime']).astype('int64') // 10**9)
    cols = ['ts', 'datetime', 'open', 'high', 'low', 'close', 'volume']
    return full[[c for c in cols if c in full.columns]]

def persist_stock_data(ticker: str, df):
    """
    Persist fetched stock data into the database.
    """
    if df is None or df.empty:
        print(f"No data to persist for {ticker}.")
        return

    try:
        for _, row in df.iterrows():
            stock_entry = StockData(
                ticker=ticker,
                timestamp=row['datetime'],
                open_price=row['open'],
                high_price=row['high'],
                low_price=row['low'],
                close_price=row['close'],
                volume=row['volume']
            )
            session.merge(stock_entry)  # Use merge to avoid duplicates
        session.commit()
        print(f"Persisted {len(df)} rows for {ticker}.")
    except Exception as e:
        session.rollback()
        print(f"Error persisting data for {ticker}: {e}")

def fetch_intraday(ticker: str):
    # import settings lazily and heavy libs inside function
    from realtime_anomaly_project.config import settings
    import pandas as pd
    import yfinance as yf

    LOOKBACK = settings.LOOKBACK
    INTERVAL = settings.INTERVAL

    days = _parse_lookback_days(LOOKBACK)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)

    print(f"Fetching data for {ticker} (interval={INTERVAL}, lookback={LOOKBACK})...")

    # If user requests 1m/2m and days > 7, attempt chunked fetch
    if INTERVAL in ("1m", "2m") and days > 7:
        df = fetch_2m_for_range(ticker, start, now, max_chunk_days=7)
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            data_storage[ticker] = df
            persist_stock_data(ticker, df)  # Persist data to DB
            return df
        else:
            print(f"Warning: chunked 2m fetch returned no data for {ticker}. Falling back to period fetch.")

    # Default fallback: try yf.download with period
    try:
        df = yf.download(ticker, period=LOOKBACK, interval=INTERVAL, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

    # explicit emptiness checks to avoid ambiguous truth value
    if df is None or (hasattr(df, "empty") and df.empty):
        print(f"Warning: No data fetched for {ticker} with period={LOOKBACK}, interval={INTERVAL}")
        return None

    # normalize
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
    })
    df = df.reset_index().rename(columns={"index": "datetime"})
    try:
        df['ts'] = (pd.to_datetime(df['datetime']).astype('int64') // 10**9)
    except Exception:
        pass

    data_storage[ticker] = df
    persist_stock_data(ticker, df)  # Persist data to DB
    return df

def run_once():
    from realtime_anomaly_project.config import settings
    tickers = settings.TICKERS
    for t in tickers:
        try:
            fetch_intraday(t)
            rows = 0
            v = data_storage.get(t)
            if hasattr(v, "__len__"):
                rows = len(v)
            print(f"Fetched and stored data for {t} (rows: {rows})")
        except Exception as e:
            print(f"Error fetching {t}: {e}")

if __name__ == "__main__":
    run_once()