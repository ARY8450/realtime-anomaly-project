import sys
import os

# ensure package imports work when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
import traceback
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
from realtime_anomaly_project.config import settings

# lightweight in-memory store
data_storage = {}

# DB imports
from realtime_anomaly_project.database.db_setup import setup_database, StockData, get_engine
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

# ensure DB and Session factory exist and ENGINE is set
SessionLocal = setup_database()
ENGINE = get_engine()

def persist_stock_data_upsert(ticker: str, df: pd.DataFrame):
    """
    Bulk upsert normalized DataFrame rows into stock_data using SQLite ON CONFLICT.
    Returns number of rows attempted.
    """
    if df is None or df.empty:
        return 0

    rows = []
    for ts, row in df.iterrows():
        # ts is a pandas.Timestamp with tz-aware UTC usually; convert to naive UTC for SQLite
        try:
            dt = pd.to_datetime(ts)
            if getattr(dt, "tzinfo", None) is not None:
                dt = dt.tz_convert("UTC").tz_localize(None)
            # final fallback: ensure python datetime
            dt = dt.to_pydatetime()
        except Exception:
            continue

        rows.append({
            "ticker": ticker,
            "timestamp": dt,
            "open_price": float(row.get('open', row.get('Open', 0) or 0)),
            "high_price": float(row.get('high', row.get('High', 0) or 0)),
            "low_price": float(row.get('low', row.get('Low', 0) or 0)),
            "close_price": float(row.get('close', row.get('Close', 0) or 0)),
            "volume": float(row.get('volume', row.get('Volume', 0) or 0))
        })

    if not rows:
        return 0

    # Build upsert statement using SQLite dialect
    table = StockData.__table__
    stmt = sqlite_insert(table).values(rows)
    update_dict = {
        "open_price": stmt.excluded.open_price,
        "high_price": stmt.excluded.high_price,
        "low_price": stmt.excluded.low_price,
        "close_price": stmt.excluded.close_price,
        "volume": stmt.excluded.volume
    }
    stmt = stmt.on_conflict_do_update(index_elements=["ticker", "timestamp"], set_=update_dict)

    # Execute using engine
    with ENGINE.begin() as conn:
        conn.execute(stmt)

    return len(rows)

def _parse_lookback_days(lookback: str) -> int:
    try:
        if isinstance(lookback, str) and lookback.endswith('d'):
            return int(lookback[:-1])
        return int(lookback)
    except Exception:
        return 60

def _normalize_df_datetime_and_close(df):
    """
    Ensure DataFrame has:
      - a tz-aware UTC DatetimeIndex
      - a lowercase 'close' 1-D Series
      - a 'ts' unix seconds integer column
    Handles MultiIndex columns by flattening them.
    """
    import pandas as pd
    if df is None:
        return None

    # coerce to DataFrame if possible
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return None

    # --- datetime/index normalization (unchanged) ---
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        df = df.set_index('datetime')
    else:
        try:
            df.index = pd.to_datetime(df.index, errors='coerce')
            if df.index.hasnans:
                df = df[~df.index.isna()]
        except Exception:
            for col in ['date', 'timestamp', 'ts']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df = df.dropna(subset=[col])
                    df = df.set_index(col)
                    break

    if not isinstance(df.index, pd.DatetimeIndex) or df.index.empty:
        print("DEBUG: normalization failed — no valid datetime index or empty after drop.")
        return None

    # make index tz-aware UTC
    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
    except Exception:
        try:
            df.index = df.index.tz_localize('UTC', ambiguous='NaT', nonexistent='shift_forward')
        except Exception:
            pass

    # --- handle MultiIndex columns by flattening ---
    if isinstance(df.columns, pd.MultiIndex):
        flat = []
        for col in df.columns:
            parts = [str(p) for p in col if p is not None and str(p) != '']
            flat.append("_".join(parts))
        df.columns = flat

    # Normalize close column to 'close' (handle variants)
    if 'close' not in df.columns:
        for alt in ['Close', 'close_price', 'ClosePrice', 'adj_close', 'Adj Close']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'close'})
                break

    # If still not present, try to find any column name containing 'close' (case-insensitive)
    if 'close' not in df.columns:
        candidates = [c for c in df.columns if 'close' in c.lower()]
        if candidates:
            df = df.rename(columns={candidates[0]: 'close'})

    # If still missing, try to pick a numeric column (fallback)
    if 'close' not in df.columns:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            print(f"DEBUG: no explicit 'close' found, using numeric column '{numeric_cols[0]}' as close")
            df = df.rename(columns={numeric_cols[0]: 'close'})
        else:
            print("DEBUG: normalization failed — no 'close' or numeric candidate found. columns:", list(df.columns))
            return None

    # Extract 1-D close series and coerce numeric
    try:
        col = df['close']
        if isinstance(col, pd.DataFrame):
            # collapse to first numeric column if needed
            numeric_subcols = [c for c in col.columns if pd.api.types.is_numeric_dtype(col[c])]
            if numeric_subcols:
                series = col[numeric_subcols[0]]
            else:
                series = col.iloc[:, 0]
            print(f"DEBUG: collapsed df['close'] from DataFrame to Series using column '{series.name}'")
        else:
            series = col
        series = pd.to_numeric(series, errors='coerce')
    except Exception as exc:
        print("DEBUG: failed extracting 'close' series:", exc)
        return None

    df = df.copy()
    df['close'] = series
    df = df.dropna(subset=['close'])
    if df.empty:
        print("DEBUG: normalization resulted in empty df after dropping non-numeric 'close'.")
        return None

    # compute ts unix seconds safely
    try:
        df = df.sort_index()
        df['ts'] = (df.index.view('int64') // 10**9).astype('int64')
    except Exception:
        try:
            df['ts'] = (pd.to_datetime(df.index).astype('int64') // 10**9).astype('int64')
        except Exception:
            pass

    return df

def fetch_2m_for_range(ticker: str, start: datetime, end: datetime, max_chunk_days: int = 7):
    import pandas as pd
    import yfinance as yf
    import traceback

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
                # normalize immediately
                norm = _normalize_df_datetime_and_close(df)
                if norm is not None and not norm.empty:
                    dfs.append(norm)
            else:
                # try to coerce other payloads into DataFrame
                try:
                    tmp = pd.DataFrame(df)
                    if not tmp.empty:
                        norm = _normalize_df_datetime_and_close(tmp)
                        if norm is not None and not norm.empty:
                            dfs.append(norm)
                except Exception:
                    # skip silently but print debug
                    print(f"DEBUG: skipped non-DataFrame chunk for {ticker} {cur.date()} - {chunk_end.date()}, type={type(df)}")
        except Exception as exc:
            print(f"Chunk fetch error for {ticker} {cur.date()} - {chunk_end.date()}: {exc}")
            traceback.print_exc()
        cur = chunk_end
        time.sleep(1)  # polite pause between requests

    if not dfs:
        return None

    # ensure only DataFrames remain
    valid_dfs = [d for d in dfs if hasattr(d, "shape")]
    if not valid_dfs:
        print("DEBUG: no valid DataFrame chunks to concat for", ticker)
        return None

    try:
        full = pd.concat(valid_dfs, axis=0, ignore_index=False).drop_duplicates().sort_index()
    except Exception as e:
        print("DEBUG: pd.concat failed for", ticker, "->", e)
        import traceback
        traceback.print_exc()
        return None

    # ensure final normalization (recompute ts etc)
    full = _normalize_df_datetime_and_close(full)
    return full

def persist_stock_data(ticker: str, df):
    """
    Persist the normalized DataFrame into the StockData table.
    Uses simple existence-check per (ticker, timestamp) to avoid duplicates.
    """
    import pandas as pd
    if df is None or df.empty:
        return 0
    # ensure expected columns
    if 'close' not in df.columns or not isinstance(df.index, pd.DatetimeIndex):
        return 0

    count = 0
    with SessionLocal() as session:
        for ts, row in df.iterrows():
            # normalize timestamp to naive UTC datetime for DB (SQLAlchemy/SQLite)
            try:
                dt = pd.to_datetime(ts)
                if getattr(dt, "tzinfo", None) is not None:
                    dt = dt.tz_convert("UTC").tz_localize(None)
            except Exception:
                continue

            # simple existence check to avoid duplicates
            exists = session.query(StockData).filter(
                StockData.ticker == ticker,
                StockData.timestamp == dt
            ).first()
            if exists:
                # optionally update fields
                exists.open_price = float(row.get('open', getattr(row, 'Open', None) or 0))
                exists.high_price = float(row.get('high', getattr(row, 'High', None) or 0))
                exists.low_price = float(row.get('low', getattr(row, 'Low', None) or 0))
                exists.close_price = float(row.get('close', getattr(row, 'Close', None) or 0))
                exists.volume = float(row.get('volume', getattr(row, 'Volume', None) or 0))
            else:
                entry = StockData(
                    ticker=ticker,
                    timestamp=dt,
                    open_price=float(row.get('open', getattr(row, 'Open', None) or 0)),
                    high_price=float(row.get('high', getattr(row, 'High', None) or 0)),
                    low_price=float(row.get('low', getattr(row, 'Low', None) or 0)),
                    close_price=float(row.get('close', getattr(row, 'Close', None) or 0)),
                    volume=float(row.get('volume', getattr(row, 'Volume', None) or 0))
                )
                session.add(entry)
            count += 1
        try:
            session.commit()
        except Exception:
            session.rollback()
            raise
    return count

def _sanitize_ticker(ticker: str) -> str:
    """Trim whitespace/newlines from ticker and ensure string type."""
    if ticker is None:
        return ticker
    return str(ticker).strip()

def _symbol_has_prices_quick(ticker: str, period="5d", interval="1d"):
    """
    Quick check to see if yfinance returns any price rows for this ticker.
    Uses a short period/interval to be fast. Returns:
      True  -> has data
      False -> no data (likely delisted or unsupported)
      None  -> indeterminate (error)
    """
    # Build list of symbols to try: alias first (if configured) then the ticker
    try_symbols = []
    alias = getattr(settings, "SYMBOL_ALIASES", {}).get(ticker)
    if alias:
        try_symbols.append(alias)
    try_symbols.append(ticker)

    for sym in try_symbols:
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(period=period, interval=interval, progress=False, auto_adjust=True)
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                return True
            # empty but no exception — treat as no data for this symbol and continue trying others
        except Exception as exc:
            msg = str(exc).lower()
            if "delist" in msg or "no data found" in msg or "possibly delisted" in msg:
                # explicit delisted/no-data for this symbol -> try next symbol in list
                continue
            # other errors: return None so caller can decide to probe further
            print(f"_symbol_has_prices_quick: transient error for {sym}: {exc}")
            return None
    # nothing found among tried symbols
    return False

def _try_symbol_variants(ticker: str, period: str, interval: str):
    """
    Try alternate yfinance fetch strategies and symbol variants for tickers that return no data.
    Returns a DataFrame or None.
    """
    import pandas as pd
    import traceback

    variants = []
    # try configured alias first
    alias = getattr(settings, "SYMBOL_ALIASES", {}).get(ticker)
    if alias:
        variants.append(alias)

    # existing variant candidates
    variants.append(ticker)
    if ticker.endswith(".NS"):
        base = ticker[:-3]
        variants.extend([base + ".NS", base, base + "-NS", ticker.replace(".", "-")])
    else:
        variants.extend([ticker + ".NS", ticker + "-NS", ticker.upper()])

    # deduplicate while preserving order
    seen = set()
    variants = [v for v in variants if not (v in seen or seen.add(v))]

    for sym in variants:
        try:
            print(f"VARIANT: trying yfinance.Ticker.history for symbol={sym}, period={period}, interval={interval}")
            tk = yf.Ticker(sym)
            hist = tk.history(period=period, interval=interval, auto_adjust=True)
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                print(f"VARIANT: history() success for {sym}")
                return hist
        except Exception as exc:
            print(f"VARIANT: history() error for {sym}: {exc}")
            traceback.print_exc()
    return None

# Ensure fetch_intraday uses _symbol_has_prices_quick before heavy probes (existing code block)
def _probe_alternate_fetch(ticker: str, lookback: str, interval: str):
    """
    Attempt alternate fetch strategies when the primary yf.download call returns no data or un-normalizable data.
    Tries:
      - _try_symbol_variants (history)
      - fetch_daily_fallback (1d bars)
      - shorter lookbacks with yf.download
    Returns a normalized DataFrame or None.
    """
    import traceback
    import yfinance as yf

    print(f"PROBE: attempting alternate fetches for {ticker} (lookback={lookback}, interval={interval})")

    # 1) Try symbol variants/history()
    try:
        alt = _try_symbol_variants(ticker, period=lookback, interval=interval)
        if isinstance(alt, pd.DataFrame) and not alt.empty:
            norm = _normalize_df_datetime_and_close(alt)
            if norm is not None and not norm.empty:
                print(f"PROBE: _try_symbol_variants returned usable data for {ticker}")
                return norm
    except Exception as exc:
        print(f"PROBE: _try_symbol_variants error for {ticker}: {exc}")
        traceback.print_exc()

    # 2) Try daily fallback as a less granular but often-available source
    try:
        daily = fetch_daily_fallback(ticker, lookback)
        if daily is not None and not daily.empty:
            print(f"PROBE: fetch_daily_fallback returned data for {ticker}")
            return daily
    except Exception as exc:
        print(f"PROBE: fetch_daily_fallback error for {ticker}: {exc}")

    # 3) Try reduced lookbacks (shorter periods) with the requested interval
    for lb in ("7d", "30d", "1d"):
        if lb == lookback:
            continue
        try:
            print(f"PROBE: attempting yf.download for {ticker} period={lb} interval={interval}")
            raw = yf.download(ticker, period=lb, interval=interval, progress=False, auto_adjust=True)
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                norm = _normalize_df_datetime_and_close(raw)
                if norm is not None and not norm.empty:
                    print(f"PROBE: yf.download succeeded for {ticker} period={lb}")
                    return norm
        except Exception as exc:
            print(f"PROBE: yf.download error for {ticker} period={lb}: {exc}")

    print(f"PROBE: no alternate fetch succeeded for {ticker}")
    return None

def fetch_intraday(ticker: str):
    ticker = _sanitize_ticker(ticker)
    # quick existence check to avoid long probes for delisted tickers
    quick = _symbol_has_prices_quick(ticker, period="5d", interval="1d")
    if quick is False:
        print(f"{ticker}: quick check indicates no price data (possibly delisted). Skipping.")
        return None
    # if quick is None (transient error) or True, continue with normal fetch/probe

    from realtime_anomaly_project.config import settings

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
            try:
                persist_stock_data_upsert(ticker, df)
            except Exception as e:
                print(f"Error persisting data for {ticker}: {e}")
            return df
        else:
            print(f"Warning: chunked 2m fetch returned no data for {ticker}. Falling back to period fetch.")

    # Default fallback: try yf.download with period
    try:
        import yfinance as yf
        raw = yf.download(ticker, period=LOOKBACK, interval=INTERVAL, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        raw = None

    # explicit emptiness checks to avoid ambiguous truth value
    if raw is None or (hasattr(raw, "empty") and raw.empty):
        print(f"Warning: No data fetched for {ticker} with period={LOOKBACK}, interval={INTERVAL}. Will probe alternates.")
        # try alternate strategies (shorter lookback / different intervals)
        probed = _probe_alternate_fetch(ticker, LOOKBACK, INTERVAL)
        if probed is not None:
            data_storage[ticker] = probed
            return probed
        return None

    # Normalize before storing
    norm = _normalize_df_datetime_and_close(raw)
    if norm is None or norm.empty:
        print(f"Warning: fetched data could not be normalized for {ticker}. Attempting alternate probes.")
        probed = _probe_alternate_fetch(ticker, LOOKBACK, INTERVAL)
        if probed is not None:
            data_storage[ticker] = probed
            return probed
        print(f"Warning: normalization failed for {ticker} after probe attempts.")
        return None

    # store in-memory and persist
    data_storage[ticker] = norm
    try:
        rows = persist_stock_data_upsert(ticker, norm)
        print(f"Persisted {rows} rows for {ticker} to DB.")
    except Exception as e:
        print(f"Error persisting data for {ticker}: {e}")

    return norm

def run_once(tickers=None):
    """
    Fetch all tickers (from settings.TICKERS by default) in batches.
    This replaces the older per-ticker run_once and respects BATCH_SIZE / BATCH_SLEEP_SECONDS.
    """
    from realtime_anomaly_project.config import settings
    tickers = tickers or settings.TICKERS
    batch_size = getattr(settings, "BATCH_SIZE", 10)
    sleep_between = getattr(settings, "BATCH_SLEEP_SECONDS", 5)

    total = len(tickers or [])
    print(f"Starting one-shot fetch for {total} tickers (batch_size={batch_size}, sleep={sleep_between}s)")

    for start in range(0, total, batch_size):
        batch = tickers[start:start + batch_size]
        print(f"Processing batch {start//batch_size + 1} ({len(batch)} tickers): {batch}")
        for t in batch:
            try:
                norm = fetch_intraday(t)  # fetch_intraday handles normalization + persistence
                rows = 0
                v = data_storage.get(t)
                if hasattr(v, "__len__"):
                    rows = len(v)
                print(f"Fetched and stored data for {t} (rows: {rows})")
            except Exception as e:
                print(f"Error fetching {t}: {e}")
        # don't sleep after the final batch
        if start + batch_size < total:
            print(f"Batch complete — sleeping {sleep_between} seconds before next batch...")
            time.sleep(sleep_between)

    print("One-shot fetch complete.")

def fetch_daily_fallback(ticker: str, lookback: str | None = None):
    """
    Fallback fetch that requests daily bars (1d) for lookback (defaults to settings.LOOKBACK).
    Normalizes and persists rows using existing persist helper.
    Returns normalized DataFrame or None.
    """
    from realtime_anomaly_project.config import settings
    lb = lookback or getattr(settings, "LOOKBACK", "30d")
    ticker = _sanitize_ticker(ticker)
    try:
        print(f"DAILY-FALLBACK: fetching {ticker} interval=1d period={lb}")
        raw = yf.download(ticker, period=lb, interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        print(f"DAILY-FALLBACK: yf.download error for {ticker}: {e}")
        raw = None

    if raw is None or (hasattr(raw, "empty") and raw.empty):
        # try symbol variants via history() as last resort
        try:
            alt = _try_symbol_variants(ticker, period=lb, interval="1d")
            if isinstance(alt, pd.DataFrame) and not alt.empty:
                raw = alt
        except Exception:
            raw = None

    if raw is None or (hasattr(raw, "empty") and raw.empty):
        print(f"DAILY-FALLBACK: no daily data for {ticker}")
        return None

    # Normalize and persist
    try:
        norm = _normalize_df_datetime_and_close(raw)
        if norm is None or norm.empty:
            print(f"DAILY-FALLBACK: normalization produced empty for {ticker}")
            return None
        # Persist using bulk upsert if available
        try:
            rows = persist_stock_data_upsert(ticker, norm)
            print(f"DAILY-FALLBACK: persisted {rows} daily rows for {ticker}")
        except Exception as e:
            print(f"DAILY-FALLBACK: persist failed for {ticker}: {e}")
        return norm
    except Exception as e:
        print(f"DAILY-FALLBACK: unexpected error for {ticker}: {e}")
        return None