import sys
import os
# ensure project root is on sys.path when running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import subprocess
from datetime import datetime

import pandas as pd
from realtime_anomaly_project.database.db_setup import StockData, setup_database

def fetch_rows(ticker=None, limit=None):
    Session = setup_database()
    with Session() as session:
        q = session.query(StockData)
        if ticker:
            q = q.filter(StockData.ticker == ticker)
        q = q.order_by(StockData.timestamp.desc())
        if limit:
            q = q.limit(limit)
        rows = q.all()

    return rows

def rows_to_df(rows):
    if not rows:
        return pd.DataFrame()
    data = []
    for r in rows:
        data.append({
            "id": r.id,
            "ticker": r.ticker,
            "timestamp": r.timestamp,
            "open": r.open_price,
            "high": r.high_price,
            "low": r.low_price,
            "close": r.close_price,
            "volume": r.volume
        })
    return pd.DataFrame(data)

def write_csv(df, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "sql_db")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"stock_data_view_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = os.path.abspath(os.path.join(out_dir, fname))
    df.to_csv(path, index=False)
    return path

def open_in_vscode(path):
    # Attempts to open the file in VS Code using the `code` CLI if available
    try:
        subprocess.run(["code", path], check=False)
        return True
    except FileNotFoundError:
        return False

def main():
    p = argparse.ArgumentParser(description="View persisted stock data and open in VS Code")
    p.add_argument("--ticker", "-t", help="Ticker to filter (e.g. RELIANCE.NS)")
    p.add_argument("--limit", "-n", type=int, help="Limit number of rows")
    p.add_argument("--open", "-o", action="store_true", help="Try to open the CSV in VS Code using 'code' CLI")
    args = p.parse_args()

    rows = fetch_rows(ticker=args.ticker, limit=args.limit)
    df = rows_to_df(rows)

    if df.empty:
        print("No rows found for the query.")
        return

    csv_path = write_csv(df)
    print(f"Wrote {len(df)} rows to: {csv_path}")

    # Print a quick preview in console
    print(df.head(20).to_string(index=False))

    if args.open:
        ok = open_in_vscode(csv_path)
        if ok:
            print("Tried to open CSV in VS Code (using 'code' CLI).")
        else:
            print("VS Code 'code' CLI not found on PATH. Open the CSV manually in VS Code.")

if __name__ == "__main__":
    main()