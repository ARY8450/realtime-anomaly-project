import sqlite3, os

db = os.path.abspath('realtime_anomaly_project/sql_db/realtime_anomaly.db')
print('DB:', db)
if not os.path.exists(db):
    print('DB not found')
    raise SystemExit(1)
con = sqlite3.connect(db)
cur = con.cursor()
print('\nPRAGMA table_info(stock_data);')
for row in cur.execute("PRAGMA table_info('stock_data');").fetchall():
    print(row)
print('\nPRAGMA index_list(stock_data);')
for row in cur.execute("PRAGMA index_list('stock_data');").fetchall():
    print(row)
idxs = [r[1] for r in cur.execute("PRAGMA index_list('stock_data');").fetchall()]
for name in idxs:
    print(f"\nPRAGMA index_info({name});")
    try:
        for r in cur.execute(f"PRAGMA index_info('{name}');").fetchall():
            print(r)
    except Exception as e:
        print('error reading index_info', e)
print('\nCREATE TABLE SQL:')
for r in cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='stock_data';").fetchall():
    print(r[0])
con.close()
