from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

# Define the database model
Base = declarative_base()

class StockData(Base):
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    
    def __repr__(self):
        return f"<StockData(ticker={self.ticker}, timestamp={self.timestamp}, close_price={self.close_price})>"

class NewsData(Base):
    __tablename__ = 'news_data'

    id = Column(Integer, primary_key=True)
    news_id = Column(String, nullable=False)
    ticker = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    title = Column(String, nullable=False)
    summary = Column(Text, nullable=False)
    sentiment = Column(String, nullable=False)
    sentiment_score = Column(Float, nullable=False)
    category = Column(String, nullable=False)
    
    def __repr__(self):
        return f"<NewsData(news_id={self.news_id}, ticker={self.ticker}, sentiment={self.sentiment}, category={self.category})>"

class AnomalyResults(Base):
    __tablename__ = 'anomaly_results'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    z_score = Column(Float, nullable=False)
    rsi = Column(Float, nullable=False)
    fusion_score = Column(Float, nullable=False)
    decision = Column(String, nullable=False)  # Anomalous or Normal
    
    def __repr__(self):
        return f"<AnomalyResults(ticker={self.ticker}, timestamp={self.timestamp}, decision={self.decision})>"

# Database setup function
def setup_database(db_url="sqlite:///realtime_anomaly.db"):
    # Create an SQLite database (or use an existing one)
    engine = create_engine(db_url)
    
    # Create the tables in the database
    Base.metadata.create_all(engine)
    
    # Create a session factory
    Session = sessionmaker(bind=engine)
    return Session