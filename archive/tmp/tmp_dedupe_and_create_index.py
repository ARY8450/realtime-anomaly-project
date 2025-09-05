import sqlite3, os, sys

db = os.path.abspath('realtime_anomaly_project/sql_db/realtime_anomaly.db')
print('DB:', db)
if not os.path.exists(db):
    print('DB not found')
    sys.exit(1)
con = sqlite3.connect(db)
cur = con.cursor()
# find duplicates
print('\nFinding duplicate (ticker, timestamp) counts...')
dup_sql = "SELECT ticker, timestamp, COUNT(*) c FROM stock_data GROUP BY ticker, timestamp HAVING c>1;"
dups = cur.execute(dup_sql).fetchall()
print('Found', len(dups), 'duplicate groups')
if not dups:
    print('No duplicates, creating index...')
else:
    # For each duplicate group, keep the row with the max(id) and delete others
    for ticker, ts, c in dups:
        # get ids ordered desc
        cur.execute("SELECT id FROM stock_data WHERE ticker=? AND timestamp=? ORDER BY id DESC;", (ticker, ts))
        ids = [r[0] for r in cur.fetchall()]
        keep = ids[0]
        remove = ids[1:]
        if remove:
            q = f"DELETE FROM stock_data WHERE id IN ({','.join(['?']*len(remove))});"
            cur.execute(q, remove)
            print(f"Deduped {ticker} {ts}: kept {keep}, removed {len(remove)} rows")
    con.commit()
# create unique index
idx_name = 'uix_stock_data_ticker_timestamp'
print(f"\nCreating UNIQUE INDEX {idx_name} on (ticker, timestamp) if not exists...")
try:
    cur.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {idx_name} ON stock_data(ticker, timestamp);")
    con.commit()
    print('Index created or already exists.')
except Exception as e:
    print('Error creating index:', e)
    con.close()
    sys.exit(3)
# list indexes
print('\nPRAGMA index_list(stock_data);')
for row in cur.execute("PRAGMA index_list('stock_data');").fetchall():
    print(row)
print('\nDone.')
con.close()
