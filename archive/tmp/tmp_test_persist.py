import pandas as pd
from realtime_anomaly_project.data_ingestion.yahoo import persist_stock_data_upsert, fetch_ohlcv_for_ts
import sqlite3, os

ticker = 'ADANIPORTS.NS'
# choose a timestamp likely present in cache/history
ts = pd.to_datetime('2025-09-03 03:45:00')
idx = pd.DatetimeIndex([ts])
df = pd.DataFrame({'close': [1338.699951171875]}, index=idx)
print('Calling persist_stock_data_upsert for', ticker, 'at', ts)
rows = persist_stock_data_upsert(ticker, df)
print('persist returned rows:', rows)

# Query the DB for this ticker/timestamp
db = os.path.abspath('realtime_anomaly_project/sql_db/realtime_anomaly.db')
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("SELECT ticker, timestamp, open_price, high_price, low_price, close_price, volume FROM stock_data WHERE ticker=? ORDER BY timestamp DESC LIMIT 5", (ticker,))
for r in cur.fetchall():
    print(r)
con.close()

# Also try direct probe for the timestamp
print('Direct probe result:', fetch_ohlcv_for_ts(ticker, ts))
