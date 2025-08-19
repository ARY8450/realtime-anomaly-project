import sys
import os

# keep fast: only adjust sys.path, no heavy imports at module-level
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# lightweight in-memory store
data_storage = {}

def fetch_intraday(ticker: str):
    """Fetch OHLCV data for a ticker from Yahoo Finance (lazy imports)."""
    # Lazy imports to avoid heavy startup cost
    import yfinance as yf
    import pandas as pd
    from config import settings

    LOOKBACK = settings.LOOKBACK
    INTERVAL = settings.INTERVAL

    print(f"Fetching data for {ticker} (period={LOOKBACK}, interval={INTERVAL})...")
    df = yf.download(ticker, period=LOOKBACK, interval=INTERVAL, progress=False, auto_adjust=True)

    if df.empty:
        print(f"Warning: No data fetched for {ticker}")
        return pd.DataFrame()

    # convert index to epoch seconds (works with pandas datetime index)
    df = df.copy()
    df['ts'] = df.index.astype(int) // 10**9
    df = df[['ts', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    return df

def run_once():
    """Fetch data for each ticker and store in-memory (lazy settings import)."""
    from config import settings
    for ticker in settings.TICKERS:
        t = ticker.strip()
        try:
            df = fetch_intraday(t)
            if not df.empty:
                data_storage[t] = df
                print(f"Saved {len(df)} rows for {t}")
            else:
                print(f"Skipping {t} (no data).")
        except Exception as e:
            print(f"Error fetching {t}: {e}")

if __name__ == "__main__":
    run_once()