import sys
import os

# Ensure project root is on sys.path so "config" and other local packages can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import feedparser
import hashlib
from datetime import datetime
from config.settings import TICKERS
from sentiment_module.wcn_classifier import classify_category  # Assuming you have WCN for category classification
import pandas as pd

# In-memory storage for news articles
news_storage = {}

# Function to get Yahoo Finance RSS URLs for each ticker
def rss_urls_for(ticker: str):
    return [f"https://finance.yahoo.com/rss/{ticker}"]

# Function to normalize timestamp for RSS feed entries
def normalize_ts(entry):
    if "published_parsed" in entry and entry.published_parsed:
        return int(datetime(*entry.published_parsed[:6]).timestamp())
    return int(datetime.now().timestamp())

# Function to fetch and store news articles
def fetch_news(ticker: str):
    rows = []
    for url in rss_urls_for(ticker):
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            ts = normalize_ts(entry)

            # Creating a unique ID for each news article
            news_id = hashlib.md5((title + str(ts) + ticker).encode()).hexdigest()[:24]

            # Classify news into categories
            category = classify_category(title + " " + summary)

            # Store the news
            rows.append([news_id, ticker, ts, title, summary, category])
    
    # Save to the in-memory storage
    for row in rows:
        ticker = row[1]
        if ticker not in news_storage:
            news_storage[ticker] = []
        news_storage[ticker].append(row)
    
    print(f"Fetched {len(rows)} news articles for {ticker}")

# Function to run the news ingestion for each ticker
def run_once():
    for ticker in TICKERS:
        fetch_news(ticker)

if __name__ == "__main__":
    run_once()