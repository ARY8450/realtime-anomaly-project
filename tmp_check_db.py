import sqlite3

db='realtime_anomaly_project/sql_db/realtime_anomaly.db'
con=sqlite3.connect(db)
cur=con.cursor()
for t in ['ADANIPORTS.NS','HDFC.NS','INFY.NS']:
    cur.execute("SELECT id,timestamp,open_price,high_price,low_price,volume FROM stock_data WHERE ticker=? ORDER BY timestamp DESC LIMIT 1", (t,))
    r=cur.fetchone()
    print(t, '->', r)
con.close()
