import sys
import os
import argparse
import subprocess

# ensure project root is importable when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
from realtime_anomaly_project.statistical_anomaly.unsupervised import compute_unsupervised_anomalies
from realtime_anomaly_project.config import settings

# DB helper imports
from realtime_anomaly_project.database.db_setup import setup_database, StockData
Session = setup_database()

# in-memory store (may be empty)
from realtime_anomaly_project.data_ingestion.yahoo import data_storage

# new imports: bulk fetch helper
try:
    from realtime_anomaly_project.statistical_anomaly.bulk_fetch_all import bulk_fetch_all
except Exception:
    bulk_fetch_all = None


def load_ticker_df(ticker: str) -> pd.DataFrame | None:
    """Load ticker DataFrame from in-memory store first, then DB fallback.
    Tries aliases and common symbol variants if the exact ticker has no rows.
    Returns a tz-aware DataFrame with a numeric 'close' column or None."""
    # try in-memory
    tkey = ticker.strip()
    df = data_storage.get(tkey)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df_local = df.copy()
        # normalize same as DB path below
    else:
        # DB lookup helpers
        def _rows_for_symbol(sym: str):
            with Session() as s:
                return s.query(StockData).filter(StockData.ticker == sym).order_by(StockData.timestamp).all()

        # try exact ticker first
        rows = _rows_for_symbol(tkey)

        # try configured alias
        alias = getattr(settings, "SYMBOL_ALIASES", {}).get(tkey)
        if (not rows) and alias:
            rows = _rows_for_symbol(alias)

        # try common variants if still nothing
        if not rows:
            variants = []
            if tkey.endswith(".NS"):
                base = tkey[:-3]
            else:
                base = tkey
            variants.extend([base + ".NS", base, base + "-NS", tkey.replace(".", "-")])
            # dedupe and exclude original
            seen = set([tkey])
            variants = [v for v in variants if v not in seen]
            for v in variants:
                rows = _rows_for_symbol(v)
                if rows:
                    break

        if not rows:
            return None

        # build DataFrame from DB rows
        recs = []
        for r in rows:
            recs.append({
                "datetime": r.timestamp,
                "open": r.open_price,
                "high": r.high_price,
                "low": r.low_price,
                "close": r.close_price,
                "volume": r.volume
            })
        df_local = pd.DataFrame(recs)

    # normalize to DatetimeIndex UTC and numeric close
    try:
        if "datetime" in df_local.columns:
            df_local["datetime"] = pd.to_datetime(df_local["datetime"], errors="coerce", utc=True)
            df_local = df_local.dropna(subset=["datetime"]).set_index("datetime").sort_index()
        else:
            df_local.index = pd.to_datetime(df_local.index, errors="coerce", utc=True)
            df_local = df_local[~df_local.index.isna()].sort_index()
        # Operate on an explicit DatetimeIndex object so type-checkers (Pylance)
        # know the object has datetime-specific attributes like .tz/.tz_localize.
        # Use pd.to_datetime which returns a DatetimeIndex; this makes the
        # type explicit to static checkers and avoids Index[int] inference.
        idx = pd.to_datetime(df_local.index, errors="coerce", utc=False)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        df_local.index = idx
    except Exception:
        return None

    if "close" not in df_local.columns and "Close" in df_local.columns:
        df_local = df_local.rename(columns={"Close": "close"})
    if "close" not in df_local.columns:
        numeric_cols = [c for c in df_local.columns if pd.api.types.is_numeric_dtype(df_local[c])]
        if numeric_cols:
            df_local = df_local.rename(columns={numeric_cols[0]: "close"})
        else:
            return None

    df_local["close"] = pd.to_numeric(df_local["close"], errors="coerce")
    df_local = df_local.dropna(subset=["close"])
    if df_local.empty:
        return None
    return df_local


def to_dataframe_full(results: dict, tickers_master):
    """
    Build a DataFrame of anomaly results for every ticker in tickers_master.
    Adds: rows (count), last_ts (UTC), last_close
    """
    # collect columns from results
    cols = set()
    for v in (results or {}).values():
        if isinstance(v, dict):
            cols.update(v.keys())
    cols = sorted(cols)
    # additional meta columns
    meta_cols = ["rows", "last_ts", "last_close"]
    all_cols = cols + meta_cols

    # initialize DataFrame with NaNs / None
    df = pd.DataFrame(index=tickers_master, columns=all_cols, dtype="object")

    # fill anomaly scores if present
    for ticker, scores in (results or {}).items():
        if isinstance(scores, dict):
            for k, val in scores.items():
                df.at[ticker, k] = val

    # fill meta from data sources
    for ticker in tickers_master:
        tdf = load_ticker_df(ticker)
        if tdf is None:
            df.at[ticker, "rows"] = 0
            df.at[ticker, "last_ts"] = None
            df.at[ticker, "last_close"] = None
        else:
            df.at[ticker, "rows"] = int(len(tdf))
            last_ts = tdf.index[-1]
            df.at[ticker, "last_ts"] = pd.to_datetime(last_ts).tz_convert("UTC")
            df.at[ticker, "last_close"] = float(tdf["close"].iloc[-1])

    # coerce numeric anomaly columns to floats where applicable
    for c in cols:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            pass

    return df


def write_csv(df: pd.DataFrame, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "sql_db")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"anomaly_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = os.path.abspath(os.path.join(out_dir, fname))
    df.to_csv(path, index=True)
    return path


def open_in_vscode(path: str) -> bool:
    try:
        subprocess.run(["code", path], check=False)
        return True
    except FileNotFoundError:
        return False


def main():
    p = argparse.ArgumentParser(description="Compute unsupervised statistical anomaly scores and show as table/CSV")
    p.add_argument("--open", "-o", action="store_true", help="Try to open CSV in VS Code using 'code' CLI")
    p.add_argument("--csv", help="Write CSV to this path instead of default folder")
    p.add_argument("--no-print", action="store_true", help="Do not print table to console")
    p.add_argument("--only-with-data", action="store_true", help="Show only tickers that have anomaly results or data")
    p.add_argument("--fetch-missing", action="store_true", help="Fetch tickers missing from DB before computing results")
    p.add_argument("--fetch-all", action="store_true", help="Attempt to fetch all tickers (will skip ones already in DB by default)")
    args = p.parse_args()

    # optionally fetch data first
    if args.fetch_missing or args.fetch_all:
        if bulk_fetch_all is None:
            print("Bulk fetch helper not available. Ensure statistical_anomaly.bulk_fetch_all.py exists.")
        else:
            print("Starting fetch step (this may take several minutes)...")
            # bulk_fetch_all already skips tickers present in DB; fetch-all will still attempt probe for each
            bulk_fetch_all()
            print("Fetch step completed. Proceeding to compute anomaly results.")

    # compute anomaly scores (uses in-memory or DB as implemented)
    results = compute_unsupervised_anomalies()

    tickers_master = settings.TICKERS
    df = to_dataframe_full(results, tickers_master)

    if args.only_with_data:
        # keep rows that have either anomaly values or rows>0
        df = df[
            df["rows"].astype(int) > 0
        ].copy()
        if df.empty:
            print("No tickers with data found.")

    if not args.no_print:
        if df.empty:
            print("No anomaly results available.")
        else:
            disp = df.copy()
            # format last_ts and last_close nicely
            if "last_ts" in disp.columns:
                disp["last_ts"] = disp["last_ts"].astype(str).replace("NaT", "")
            if "last_close" in disp.columns:
                disp["last_close"] = pd.to_numeric(disp["last_close"], errors="coerce").round(4)
            # fill empty anomaly values with "No data"
            anomaly_cols = [c for c in disp.columns if c not in ("rows", "last_ts", "last_close")]
            for c in anomaly_cols:
                disp[c] = disp[c].apply(lambda x: "" if pd.isna(x) else round(float(x), 6) if isinstance(x, (int, float)) else x)
            print(disp.fillna("No data").to_string())

    out_path = args.csv if args.csv else write_csv(df)
    print(f"Wrote results CSV: {out_path}")

    if args.open:
        ok = open_in_vscode(out_path)
        if ok:
            print("Opened CSV in VS Code (via 'code' CLI).")
        else:
            print("VS Code 'code' CLI not found on PATH. Open the CSV manually.")


if __name__ == "__main__":
    main()