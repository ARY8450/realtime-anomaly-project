import sys, os, time
# ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from realtime_anomaly_project.config import settings
from realtime_anomaly_project.data_ingestion import yahoo
from realtime_anomaly_project.database.db_setup import setup_database, StockData
import pandas as pd

Session = setup_database()

def in_db(ticker: str) -> bool:
    with Session() as s:
        return s.query(StockData).filter(StockData.ticker == ticker).count() > 0

def force_persist(ticker: str, df: pd.DataFrame) -> int:
    """Persist using yahoo.persist_stock_data_upsert if available, else try session insert."""
    if df is None or getattr(df, "empty", True):
        return 0
    if hasattr(yahoo, "persist_stock_data_upsert"):
        try:
            return yahoo.persist_stock_data_upsert(ticker, df)
        except Exception as e:
            print(f"[{ticker}] persist helper failed: {e}")
    # fallback: simple per-row insert using Session
    rows = 0
    with Session() as s:
        for ts, row in df.iterrows():
            try:
                from realtime_anomaly_project.database.db_setup import StockData as SD
                dt = pd.to_datetime(ts)
                if getattr(dt, "tzinfo", None) is not None:
                    dt = dt.tz_convert("UTC").tz_localize(None)
                entry = SD(
                    ticker=ticker,
                    timestamp=dt,
                    open_price=float(row.get("open", row.get("Open", 0) or 0)),
                    high_price=float(row.get("high", row.get("High", 0) or 0)),
                    low_price=float(row.get("low", row.get("Low", 0) or 0)),
                    close_price=float(row.get("close", row.get("Close", 0) or 0)),
                    volume=float(row.get("volume", row.get("Volume", 0) or 0))
                )
                s.add(entry)
                rows += 1
            except Exception:
                s.rollback()
        try:
            s.commit()
        except Exception:
            s.rollback()
    return rows

def try_fetch_and_persist(ticker: str, retries=2, pause=2) -> bool:
    t = ticker.strip()
    # Try primary fetch_intraday (has probes & persistence normally)
    for attempt in range(1, retries + 1):
        print(f"[{t}] attempt {attempt} fetch_intraday()")
        try:
            df = yahoo.fetch_intraday(t)
        except Exception as e:
            print(f"[{t}] fetch_intraday exception: {e}")
            df = None
        if isinstance(df, pd.DataFrame) and not df.empty:
            persisted = force_persist(t, df)
            print(f"[{t}] fetched {len(df)} rows, persisted {persisted}")
            return True
        # Try probe alternate fetch if available
        if hasattr(yahoo, "_probe_alternate_fetch"):
            print(f"[{t}] trying _probe_alternate_fetch")
            try:
                df = yahoo._probe_alternate_fetch(t, settings.LOOKBACK, settings.INTERVAL)
            except Exception as e:
                print(f"[{t}] probe exception: {e}")
                df = None
            if isinstance(df, pd.DataFrame) and not df.empty:
                persisted = force_persist(t, df)
                print(f"[{t}] probe fetched {len(df)} rows, persisted {persisted}")
                return True
        # Try symbol variant history helper
        if hasattr(yahoo, "_try_symbol_variants"):
            print(f"[{t}] trying _try_symbol_variants")
            try:
                alt = yahoo._try_symbol_variants(t, period=settings.LOOKBACK, interval=settings.INTERVAL)
            except Exception as e:
                print(f"[{t}] variant-history exception: {e}")
                alt = None
            if isinstance(alt, pd.DataFrame) and not alt.empty:
                # normalize if normalizer exists
                if hasattr(yahoo, "_normalize_df_datetime_and_close"):
                    try:
                        norm = yahoo._normalize_df_datetime_and_close(alt)
                    except Exception:
                        norm = alt
                else:
                    norm = alt
                persisted = force_persist(t, norm)
                print(f"[{t}] variant fetched {len(norm)} rows, persisted {persisted}")
                return True
        time.sleep(pause * attempt)
    # After primary attempts failed, try daily fallback once
    print(f"[{t}] primary intraday/probe attempts failed — trying daily fallback (1d)")
    try:
        df_daily = None
        if hasattr(yahoo, "fetch_daily_fallback"):
            df_daily = yahoo.fetch_daily_fallback(t, lookback=None)
        if isinstance(df_daily, pd.DataFrame) and not df_daily.empty:
            persisted = force_persist(t, df_daily)
            print(f"[{t}] daily fallback fetched {len(df_daily)} rows, persisted {persisted}")
            return True
    except Exception as e:
        print(f"[{t}] daily fallback exception: {e}")
    print(f"[{t}] all attempts including daily fallback failed")
    return False

def bulk_fetch_all(sleep_between=1):
    desired = [t.strip() for t in settings.TICKERS]
    print(f"Bulk fetching {len(desired)} tickers (DB already has some).")
    results = {"fetched": [], "skipped": [], "failed": []}
    for idx, t in enumerate(desired, 1):
        print(f"\n[{idx}/{len(desired)}] {t}")
        if in_db(t):
            print(f"[{t}] already in DB, skipping")
            results["skipped"].append(t)
            continue
        ok = try_fetch_and_persist(t, retries=3, pause=2)
        if ok:
            results["fetched"].append(t)
        else:
            results["failed"].append(t)
        time.sleep(sleep_between)
    # final DB listing
    with Session() as s:
        rows = s.query(StockData.ticker).distinct().all()
        tickers_in_db = sorted([r[0] for r in rows])
    print("\nFetch summary:", results)
    print("Tickers in DB now:", tickers_in_db)
    return results

if __name__ == "__main__":
    bulk_fetch_all()