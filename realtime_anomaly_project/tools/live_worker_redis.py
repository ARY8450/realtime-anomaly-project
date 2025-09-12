"""Simple live worker: fetch features for all tickers and write into a cache.
Will attempt to write to Redis if available (REDIS_URL env var). Always writes a
fallback JSON file at sql_db/realtime_live_cache.json so the dashboard can read it.

Run with:
    python tools/live_worker_redis.py --once
"""
import os
import sys
import time
import json
from typing import Dict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

try:
    from realtime_anomaly_project.app._dashboard_helpers import fetch_live_features
except Exception:
    # fallback import path
    try:
        from app._dashboard_helpers import fetch_live_features
    except Exception:
        fetch_live_features = None

try:
    import realtime_anomaly_project.config.settings as settings
except Exception:
    try:
        import config.settings as settings
    except Exception:
        settings = None

TICKERS = (getattr(settings, 'TICKERS', None) or []) if settings is not None else []

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'realtime_anomaly_project', 'sql_db')
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, 'realtime_live_cache.json')

def write_json_cache(data: Dict):
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as fh:
            json.dump({'ts': int(time.time()), 'data': data}, fh, indent=2, ensure_ascii=False)
    except Exception:
        pass

def try_redis_publish(data: Dict):
    try:
        import redis
    except Exception:
        return False
    url = os.getenv('REDIS_URL') or os.getenv('REDIS_URI') or 'redis://localhost:6379/0'
    try:
        r = redis.from_url(url)
        # write each ticker as a JSON string under key realtime:<ticker>
        pipe = r.pipeline()
        for tk, vals in data.items():
            key = f'realtime:{tk}'
            pipe.set(key, json.dumps({'ts': int(time.time()), 'v': vals}))
        pipe.execute()
        return True
    except Exception:
        return False

def fetch_all_once():
    out = {}
    if not TICKERS:
        print('No TICKERS configured in settings; aborting')
        return out
    if fetch_live_features is None:
        print('fetch_live_features helper not available; aborting')
        return out
    total = len(TICKERS)
    for i, t in enumerate(TICKERS, start=1):
        try:
            feat = fetch_live_features(t)
            out[t] = feat or {}
            print(f"{i}/{total}: {t} -> open={out[t].get('open')} high={out[t].get('high')} low={out[t].get('low')} vol={out[t].get('volume')}")
        except Exception as e:
            print(f"{i}/{total}: {t} -> failed: {e}")
        # small sleep to avoid bursting Yahoo
        time.sleep(0.2)
    return out

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--once', action='store_true')
    p.add_argument('--interval', type=int, default=30, help='Polling interval in seconds for continuous mode')
    args = p.parse_args()
    if args.once:
        print('Running one-shot fetch for all tickers...')
        data = fetch_all_once()
        write_json_cache(data)
        ok = try_redis_publish(data)
        print('Wrote JSON cache to', CACHE_FILE, 'redis_ok=', ok)
        return
    # continuous mode
    print('Starting continuous worker (press Ctrl+C to stop)')
    try:
        while True:
            data = fetch_all_once()
            write_json_cache(data)
            try_redis_publish(data)
            # cadence is controllable via --interval
            time.sleep(int(args.interval or 30))
    except KeyboardInterrupt:
        print('Worker stopped')

if __name__ == '__main__':
    main()

"""Backfill from cache: read cached features and update SQLite DB.
Useful to backfill missing data points in the DB from the feature cache.

Run with:
    python tools/backfill_from_cache.py --dry-run
"""
import json
import sqlite3
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true', help='Print actions only')
parser.add_argument('--db', default='realtime_anomaly_project/sql_db/realtime_anomaly.db')
parser.add_argument('--cache', default='realtime_anomaly_project/sql_db/realtime_live_cache.json')
args = parser.parse_args()

if not os.path.exists(args.cache):
    raise SystemExit('Cache not found: ' + args.cache)
if not os.path.exists(args.db):
    raise SystemExit('DB not found: ' + args.db)

data = json.load(open(args.cache)).get('data', {})
con = sqlite3.connect(args.db)
cur = con.cursor()

count_updates = 0
for ticker, payload in data.items():
    # prefer numeric values if present
    op = payload.get('open')
    hi = payload.get('high')
    lo = payload.get('low')
    vol = payload.get('volume')
    # skip if nothing to apply
    if all(v is None for v in (op, hi, lo, vol)):
        continue

    # find latest row for this ticker
    cur.execute("SELECT id, timestamp, open_price, high_price, low_price, volume FROM stock_data WHERE ticker=? ORDER BY timestamp DESC LIMIT 1", (ticker,))
    row = cur.fetchone()
    if not row:
        # no row to update (you could INSERT instead if desired)
        continue
    row_id, ts, cur_op, cur_hi, cur_lo, cur_vol = row

    # decide whether to update: update fields that are NULL or 0.0 or explicitly missing
    updates = {}
    if op is not None and (cur_op is None or cur_op == 0.0):
        updates['open_price'] = float(op)
    if hi is not None and (cur_hi is None or cur_hi == 0.0):
        updates['high_price'] = float(hi)
    if lo is not None and (cur_lo is None or cur_lo == 0.0):
        updates['low_price'] = float(lo)
    if vol is not None and (cur_vol is None or cur_vol == 0.0):
        updates['volume'] = float(vol)

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
