import sys
import os

# ensure project root is on sys.path so package imports work when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from realtime_anomaly_project.data_ingestion.yahoo import fetch_intraday
from realtime_anomaly_project.database.io import insert_stock_data, insert_news_data, insert_anomaly_results, close_session

# Fetch stock data
ticker = "RELIANCE.NS"
stock_data = fetch_intraday(ticker)
insert_stock_data(ticker, stock_data)

# Insert news data (example: this would come from your news ingestion process)
news = [
    {
        "news_id": "12345",
        "ticker": ticker,
        "timestamp": 1630434000,
        "title": "Reliance Industries Quarterly Results",
        "summary": "Reliance Industries reported a strong quarterly performance...",
        "sentiment": "positive",
        "sentiment_score": 0.8,
        "category": "earnings"
    }
]
insert_news_data(news)

# Insert anomaly results (example: this would come from your anomaly detection process)
anomaly_results = [
    {
        "timestamp": 1630434000,
        "z_score": 2.5,
        "rsi": 75,
        "fusion_score": 0.85,
        "decision": "Anomalous"
    }
]
insert_anomaly_results(ticker, anomaly_results)

# Close session
close_session()
