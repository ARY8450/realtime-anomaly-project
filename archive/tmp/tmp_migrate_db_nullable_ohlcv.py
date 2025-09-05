import sqlite3, os, shutil

db = os.path.abspath('realtime_anomaly_project/sql_db/realtime_anomaly.db')
bak = db + '.bak'
print('DB:', db)
if not os.path.exists(db):
    print('DB not found')
    raise SystemExit(1)
shutil.copy2(db, bak)
print('Backed up to', bak)
con = sqlite3.connect(db)
cur = con.cursor()
# create new table with nullable OHLCV and unique constraint
cur.executescript('''
BEGIN;
CREATE TABLE IF NOT EXISTS stock_data_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker VARCHAR NOT NULL,
    timestamp DATETIME NOT NULL,
    open_price FLOAT,
    high_price FLOAT,
    low_price FLOAT,
    close_price FLOAT,
    volume FLOAT
);
-- copy data (allow NULLs)
INSERT INTO stock_data_new (id, ticker, timestamp, open_price, high_price, low_price, close_price, volume)
SELECT id, ticker, timestamp, open_price, high_price, low_price, close_price, volume FROM stock_data;
DROP TABLE stock_data;
ALTER TABLE stock_data_new RENAME TO stock_data;
CREATE UNIQUE INDEX IF NOT EXISTS uix_stock_data_ticker_timestamp ON stock_data(ticker, timestamp);
CREATE INDEX IF NOT EXISTS ix_stock_data_ticker ON stock_data(ticker);
CREATE INDEX IF NOT EXISTS ix_stock_data_timestamp ON stock_data(timestamp);
COMMIT;
''')
print('Migration script applied.')
con.close()
print('Done.')
