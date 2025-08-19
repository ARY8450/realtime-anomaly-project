import sys
import os

# ensure project root is on sys.path so package imports resolve when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from realtime_anomaly_project.database.db_setup import StockData, NewsData, AnomalyResults, setup_database

# Set up session
Session = setup_database()
session = Session()

# Query StockData table for a specific ticker
stocks = session.query(StockData).filter(StockData.ticker == 'RELIANCE.NS').all()
for stock in stocks:
    print(stock)

# Query NewsData table for a specific ticker
news = session.query(NewsData).filter(NewsData.ticker == 'RELIANCE.NS').all()
for article in news:
    print(article)

# Query AnomalyResults table for a specific ticker
anomalies = session.query(AnomalyResults).filter(AnomalyResults.ticker == 'RELIANCE.NS').all()
for anomaly in anomalies:
    print(anomaly)

# Close the session after queries
session.close()
