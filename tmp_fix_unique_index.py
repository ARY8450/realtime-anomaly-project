import sqlite3, os, sys

db = os.path.abspath('realtime_anomaly_project/sql_db/realtime_anomaly.db')
print('DB:', db)
if not os.path.exists(db):
    print('DB not found')
    sys.exit(1)
con = sqlite3.connect(db)
cur = con.cursor()
# Check duplicates
print('\nChecking for duplicate (ticker, timestamp) rows...')
dup_sql = "SELECT ticker, timestamp, COUNT(*) c FROM stock_data GROUP BY ticker, timestamp HAVING c>1;"
dups = cur.execute(dup_sql).fetchall()
if dups:
    print('Found duplicates (first 20):')
    for r in dups[:20]:
        print(r)
    print('\nPlease deduplicate these rows before creating a UNIQUE index.')
    con.close()
    sys.exit(2)
else:
    print('No duplicates found.')
# Create unique index if not exists
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
# Show indexes
print('\nPRAGMA index_list(stock_data);')
for row in cur.execute("PRAGMA index_list('stock_data');").fetchall():
    print(row)
# Verify index_info
print('\nPRAGMA index_info(%s);' % idx_name)
for r in cur.execute(f"PRAGMA index_info('{idx_name}');").fetchall():
    print(r)
# Show CREATE TABLE
print('\nCREATE TABLE SQL:')
for r in cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='stock_data';").fetchall():
    print(r[0])
con.close()
print('\nDone.')
