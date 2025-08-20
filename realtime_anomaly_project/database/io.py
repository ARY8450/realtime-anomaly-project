import sys
import os
from datetime import datetime
from sqlalchemy.orm import sessionmaker

# ensure project root parent is on sys.path so package imports resolve when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from realtime_anomaly_project.database.db_setup import StockData, NewsData, AnomalyResults, setup_database

# Setup database session
Session = setup_database()
session = Session()

# Function to insert stock data into the database
def insert_stock_data(ticker, data):
    for index, row in data.iterrows():
        stock_data = StockData(
            ticker=ticker,
            timestamp=row['ts'],
            open_price=row['open'],
            high_price=row['high'],
            low_price=row['low'],
            close_price=row['close'],
            volume=row['volume']
        )
        session.add(stock_data)
    session.commit()

# Function to insert news data into the database
def insert_news_data(news):
    for article in news:
        news_data = NewsData(
            news_id=article['news_id'],
            ticker=article['ticker'],
            timestamp=article['timestamp'],
            title=article['title'],
            summary=article['summary'],
            sentiment=article['sentiment'],
            sentiment_score=article['sentiment_score'],
            category=article['category']
        )
        session.add(news_data)
    session.commit()

# Function to insert anomaly results into the database
def insert_anomaly_results(ticker, anomaly_results):
    for result in anomaly_results:
        anomaly_data = AnomalyResults(
            ticker=ticker,
            timestamp=datetime.fromtimestamp(result['timestamp']),
            z_score=result['z_score'],
            rsi=result['rsi'],
            fusion_score=result['fusion_score'],
            decision=result['decision']
        )
        session.add(anomaly_data)
    session.commit()

# Close the session after the insert operations
def close_session():
    session.close()