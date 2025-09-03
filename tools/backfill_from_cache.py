#!/usr/bin/env python3
"""Backfill latest DB row per ticker from the JSON live cache.
Usage:
  python tools/backfill_from_cache.py --dry-run
"""
import json, sqlite3, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true', help='Print actions only')
parser.add_argument('--db', default='realtime_anomaly_project/sql_db/realtime_anomaly.db')
parser.add_argument('--cache', default='realtime_anomaly_project/sql_db/realtime_live_cache.json')
parser.add_argument('--force', action='store_true', help='Apply cache values even if DB already has non-zero values')
parser.add_argument('--apply-to-all-zeros', action='store_true', help='Update all rows with zero/null OHLCV (not just latest)')
args = parser.parse_args()

if not os.path.exists(args.cache):
    raise SystemExit('Cache not found: ' + args.cache)
if not os.path.exists(args.db):
    raise SystemExit('DB not found: ' + args.db)

with open(args.cache, 'r', encoding='utf8') as fh:
    data = json.load(fh).get('data', {})
con = sqlite3.connect(args.db)
cur = con.cursor()

count_updates = 0
for ticker, payload in data.items():
    op = payload.get('open')
    hi = payload.get('high')
    lo = payload.get('low')
    vol = payload.get('volume')
    if all(v is None for v in (op, hi, lo, vol)):
        continue
    if args.apply_to_all_zeros:
        cur.execute("SELECT id, timestamp, open_price, high_price, low_price, volume FROM stock_data WHERE ticker=? AND (open_price IS NULL OR open_price=0.0 OR high_price IS NULL OR high_price=0.0 OR low_price IS NULL OR low_price=0.0 OR volume IS NULL OR volume=0.0) ORDER BY timestamp ASC", (ticker,))
        rows = cur.fetchall()
    else:
        cur.execute("SELECT id, timestamp, open_price, high_price, low_price, volume FROM stock_data WHERE ticker=? ORDER BY timestamp DESC LIMIT 1", (ticker,))
        r = cur.fetchone()
        rows = [r] if r else []

    if not rows:
        continue
    for row in rows:
        row_id, ts, cur_op, cur_hi, cur_lo, cur_vol = row
        updates = {}
        if op is not None and (args.force or cur_op is None or cur_op == 0.0):
            try:
                updates['open_price'] = float(op)
            except Exception:
                pass
        if hi is not None and (args.force or cur_hi is None or cur_hi == 0.0):
            try:
                updates['high_price'] = float(hi)
            except Exception:
                pass
        if lo is not None and (args.force or cur_lo is None or cur_lo == 0.0):
            try:
                updates['low_price'] = float(lo)
            except Exception:
                pass
        if vol is not None and (args.force or cur_vol is None or cur_vol == 0.0):
            try:
                updates['volume'] = float(vol)
            except Exception:
                pass

        if not updates:
            continue

        if args.dry_run:
            print(f"DRY: Would update {ticker} id={row_id} ts={ts} set {updates}")
        else:
            set_clause = ",".join([f"{k}=?" for k in updates.keys()])
            params = list(updates.values()) + [row_id]
            cur.execute(f"UPDATE stock_data SET {set_clause} WHERE id=?", params)
            count_updates += 1

if not args.dry_run:
    con.commit()
con.close()
print('Done. updates_applied=' + str(count_updates))
