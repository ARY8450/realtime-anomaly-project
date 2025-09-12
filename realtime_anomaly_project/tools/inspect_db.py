import sys
import os

# Ensure repo root is importable when running this script from tools/
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from realtime_anomaly_project.database.db_setup import setup_database, StockData
Session = setup_database()
with Session() as s:
    res = s.query(StockData.ticker).distinct().all()
    tickers = [r[0] for r in res]
    print('DB tickers count:', len(tickers))
    print(tickers[:200])
    if tickers:
        for t in tickers[:20]:
            latest = s.query(StockData).filter(StockData.ticker==t).order_by(StockData.timestamp.desc()).first()
            if latest is None:
                print(t, 'latest: None')
            else:
                # print ISO timestamp for clarity
                try:
                    print(t, 'latest:', getattr(latest, 'timestamp'))
                except Exception:
                    print(t, 'latest: <unprintable timestamp>')
