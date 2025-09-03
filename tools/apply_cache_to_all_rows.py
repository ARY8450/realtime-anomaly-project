import os, json, sqlite3, time

ROOT = os.path.dirname(os.path.dirname(__file__))
DB = os.path.join(ROOT, 'realtime_anomaly_project', 'sql_db', 'realtime_anomaly.db')
CACHE = os.path.join(ROOT, 'realtime_anomaly_project', 'sql_db', 'realtime_live_cache.json')

if not os.path.exists(DB):
    raise SystemExit('DB not found: ' + DB)
if not os.path.exists(CACHE):
    raise SystemExit('Cache not found: ' + CACHE)

with open(CACHE, 'r', encoding='utf8') as fh:
    data = json.load(fh).get('data', {})

con = sqlite3.connect(DB)
cur = con.cursor()

start_changes = con.total_changes
updated = 0
for ticker, payload in data.items():
    op = payload.get('open')
    hi = payload.get('high')
    lo = payload.get('low')
    vol = payload.get('volume')
    # Skip tickers with all-None payload
    if all(v is None for v in (op, hi, lo, vol)):
        continue
    try:
        cur.execute('UPDATE stock_data SET open_price=?, high_price=?, low_price=?, volume=? WHERE ticker=?', (op, hi, lo, vol, ticker))
        updated += cur.rowcount if cur.rowcount>0 else 0
    except Exception as e:
        print('ERROR updating', ticker, e)

con.commit()
end_changes = con.total_changes
print('Done. updates_applied=', updated, ' (total_changes delta=', end_changes - start_changes, ')')

# Print a small sample
sample = list(data.keys())[:3]
for t in sample:
    cur.execute('SELECT id,timestamp,open_price,high_price,low_price,volume FROM stock_data WHERE ticker=? ORDER BY timestamp DESC LIMIT 3', (t,))
    rows = cur.fetchall()
    print('\n==', t)
    for r in rows:
        print(r)

con.close()
