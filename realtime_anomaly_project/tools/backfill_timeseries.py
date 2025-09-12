#!/usr/bin/env python3
r"""Backfill OHLCV for historical DB rows using targeted fetches per timestamp.

This script finds DB rows with missing OHLCV (NULL or 0.0) and calls
`fetch_ohlcv_for_ts(ticker, timestamp)` to try to backfill open/high/low/volume.

Usage examples:
    # dry-run for single ticker, 10 rows
    .venv\Scripts\python.exe tools\backfill_timeseries.py --ticker ADANIPORTS.NS --dry-run --limit 10

    # run for all tickers, commit changes
    .venv\Scripts\python.exe tools\backfill_timeseries.py --all

Options:
    --dry-run     : print planned updates but don't modify DB
    --ticker T    : specify a ticker (can be repeated)
    --all         : run for all tickers present in DB
    --limit N     : limit rows probed per ticker (default: 0 -> unlimited)
    --sleep S     : seconds to sleep between probes (default: 0.5)
"""

import sys
import os
# ensure package imports work when running this script from the tools/ directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import sqlite3
import os
import time
from realtime_anomaly_project.data_ingestion.yahoo import fetch_ohlcv_for_ts

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--ticker', action='append', help='Ticker to process (can repeat)')
parser.add_argument('--all', action='store_true', help='Process all tickers in DB')
parser.add_argument('--limit', type=int, default=0, help='Limit rows probed per ticker (0=unlimited)')
parser.add_argument('--sleep', type=float, default=0.5, help='Seconds to sleep between target probes')
parser.add_argument('--aggressive', action='store_true', help='Try multiple intervals/lookbacks when probing')
parser.add_argument('--use-cache', action='store_true', help='Fall back to realtime_live_cache.json when probes fail')
parser.add_argument('--force-cache', action='store_true', help='When using cache, apply cache values regardless of timestamp distance')
parser.add_argument('--cache-file', default='realtime_anomaly_project/sql_db/realtime_live_cache.json', help='Path to live cache JSON file')
parser.add_argument('--db', default='realtime_anomaly_project/sql_db/realtime_anomaly.db')
args = parser.parse_args()

if not os.path.exists(args.db):
    raise SystemExit('DB not found: ' + args.db)

con = sqlite3.connect(args.db)
cur = con.cursor()

# determine tickers
tickers = []
if args.all:
    cur.execute("SELECT DISTINCT ticker FROM stock_data")
    tickers = [r[0] for r in cur.fetchall()]
elif args.ticker:
    tickers = args.ticker
else:
    raise SystemExit('Specify --ticker or --all')

print('Tickers to process:', tickers)

# load live cache optionally
live_cache = None
if args.use_cache and os.path.exists(args.cache_file):
    try:
        import json
        live_cache = json.load(open(args.cache_file, 'r', encoding='utf-8'))
        # normalize structure: expect { 'ts': <int>, 'data': { ticker: { ... } } }
        if not isinstance(live_cache, dict) or 'data' not in live_cache:
            live_cache = None
    except Exception:
        live_cache = None

for ticker in tickers:
    print('\nProcessing', ticker)
    # select rows with missing or zero OHLCV
    q = ("SELECT id, timestamp, open_price, high_price, low_price, close_price, volume "
         "FROM stock_data WHERE ticker=? AND (open_price IS NULL OR open_price=0.0 OR high_price IS NULL OR high_price=0.0 "
         "OR low_price IS NULL OR low_price=0.0 OR volume IS NULL OR volume=0.0) ORDER BY timestamp ASC")
    params = (ticker,)
    if args.limit and args.limit > 0:
        q = q + ' LIMIT ?'
        params = (ticker, args.limit)
    cur.execute(q, params)
    rows = cur.fetchall()
    print('Rows with missing OHLCV to probe:', len(rows))
    updates_for_ticker = 0
    for row in rows:
        row_id, ts, cur_op, cur_hi, cur_lo, cur_cl, cur_vol = row
        print('\nRow id', row_id, 'ts', ts, 'current', {'open': cur_op, 'high': cur_hi, 'low': cur_lo, 'vol': cur_vol})
        used_cache = False
        probe = None
        try:
            # primary probe
            probe = fetch_ohlcv_for_ts(ticker, ts)
        except Exception as e:
            print('Probe error:', e)
            probe = None

        # if aggressive flag set and primary probe failed, try several interval/lookback combos
        if not probe and args.aggressive:
            probes_tried = []
            intervals = ['1h', '30m', '15m', '5m', '1d']
            lookbacks = [2, 7, 14]
            for lb in lookbacks:
                for interval in intervals:
                    try:
                        probes_tried.append((lb, interval))
                        p = fetch_ohlcv_for_ts(ticker, ts, lookback_days=lb, interval=interval)
                        if p:
                            probe = p
                            print(f"Aggressive probe success: lookback={lb} interval={interval}")
                            break
                    except Exception as e:
                        print('Aggressive probe error:', e)
                if probe:
                    break
            if not probe:
                print('Aggressive probes tried:', probes_tried)

        if not probe:
            print('Probe returned no data for', ts)
            # fallback to live cache if requested
            if args.use_cache and live_cache:
                try:
                    cache_data = live_cache.get('data', {}).get(ticker)
                    if cache_data:
                        cache_ts = live_cache.get('ts')
                        # accept cache if close in time (within 24h) or if force-cache
                        apply_from_cache = False
                        if args.force_cache:
                            apply_from_cache = True
                        else:
                            try:
                                import datetime
                                import pandas as pd
                                if cache_ts is not None:
                                    cache_dt = datetime.datetime.fromtimestamp(int(cache_ts), tz=None)
                                    # row ts is likely stored as a string; try parse
                                    try:
                                        row_dt = pd.to_datetime(ts)
                                        # ensure naive datetime for comparison
                                        if getattr(row_dt, 'tzinfo', None) is not None:
                                            row_dt = row_dt.tz_convert('UTC').tz_localize(None)
                                        row_dt = row_dt.to_pydatetime()
                                    except Exception:
                                        row_dt = None
                                    if row_dt is not None:
                                        delta = abs((cache_dt - row_dt).total_seconds())
                                        if delta <= 24 * 3600:
                                            apply_from_cache = True
                            except Exception:
                                apply_from_cache = False
                        if apply_from_cache:
                            # map expected keys
                            probe = {
                                'open': cache_data.get('open'),
                                'high': cache_data.get('high'),
                                'low': cache_data.get('low'),
                                'close': cache_data.get('close') or cache_data.get('last'),
                                'volume': cache_data.get('volume')
                            }
                            used_cache = True
                            print('Using live cache values for', ticker, 'id', row_id)
                except Exception as e:
                    print('Cache fallback error:', e)

            if not probe:
                time.sleep(args.sleep)
                continue
        # prepare updates only where current is None or 0.0 and probe has values
        updates = {}
        v = probe.get('open') if isinstance(probe, dict) else None
        if v is not None and (cur_op is None or cur_op == 0.0):
            try:
                updates['open_price'] = float(v)
            except Exception:
                pass
        v = probe.get('high') if isinstance(probe, dict) else None
        if v is not None and (cur_hi is None or cur_hi == 0.0):
            try:
                updates['high_price'] = float(v)
            except Exception:
                pass
        v = probe.get('low') if isinstance(probe, dict) else None
        if v is not None and (cur_lo is None or cur_lo == 0.0):
            try:
                updates['low_price'] = float(v)
            except Exception:
                pass
        v = probe.get('volume') if isinstance(probe, dict) else None
        if v is not None and (cur_vol is None or cur_vol == 0.0):
            try:
                updates['volume'] = float(v)
            except Exception:
                pass

        if not updates:
            print('No update required (probe had nothing new or DB already filled)')
            time.sleep(args.sleep)
            continue

        if args.dry_run:
            note = ' (from cache)' if used_cache else ''
            print('DRY: Would update id', row_id, 'set', updates, note)
        else:
            set_clause = ",".join([f"{k}=?" for k in updates.keys()])
            params = list(updates.values()) + [row_id]
            cur.execute(f"UPDATE stock_data SET {set_clause} WHERE id=?", params)
            updates_for_ticker += 1
            # commit periodically
            if updates_for_ticker % 50 == 0:
                con.commit()
            print('Updated id', row_id, 'set', updates)
        time.sleep(args.sleep)

    if not args.dry_run:
        con.commit()
    print('Ticker', ticker, 'updates applied:', updates_for_ticker)

con.close()
print('\nAll done.')
