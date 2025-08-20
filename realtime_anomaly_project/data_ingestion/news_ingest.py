import sys
import os

# Ensure project root is on sys.path so "config" and other local packages can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import feedparser
import hashlib
from datetime import datetime
from config.settings import TICKERS
from sentiment_module.wcn_classifier import classify_category
from sentiment_module.finbert_sentiment import analyze_sentiment
import pandas as pd
import requests
from bs4 import BeautifulSoup
import threading

# progress tracker for background fetch
news_fetch_progress = {
    "running": False,
    "total": 0,
    "fetched": 0,
    "failed": []
}

# In-memory storage for news articles
news_storage = {}
# simple sentiment cache to avoid re-running FinBERT on the same text
sentiment_cache = {}

# Function to get Yahoo Finance RSS URLs for each ticker
def rss_urls_for(ticker: str):
    """Return a list of RSS/feed endpoints to try for a ticker.

    We include Yahoo Finance, MoneyControl (if available), Mint, and Economic Times.
    If a site lacks an RSS feed for tickers, we'll fall back to site-level RSS or later an HTML scraper.
    """
    urls = []
    # Yahoo finance ticker feed (may be limited)
    urls.append(f"https://finance.yahoo.com/rss/{ticker}")
    # MoneyControl - many pages do not provide per-ticker RSS; try site search feed
    urls.append(f"https://www.moneycontrol.com/rss/MCtopnews.xml")
    # Mint - site top stories RSS
    urls.append("https://www.livemint.com/rss/homepage")
    # Economic Times - general markets RSS
    urls.append("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms")
    return urls

# Function to normalize timestamp for RSS feed entries
def normalize_ts(entry):
    if "published_parsed" in entry and entry.published_parsed:
        return int(datetime(*entry.published_parsed[:6]).timestamp())
    return int(datetime.now().timestamp())

# Function to fetch and store news articles
def fetch_news(ticker: str):
    rows = []
    for url in rss_urls_for(ticker):
        try:
            feed = feedparser.parse(url)
        except Exception:
            feed = None
        if feed and getattr(feed, 'entries', None):
            for entry in feed.entries:
                title = str(entry.get("title", ""))
                summary = str(entry.get("summary", entry.get('description', '')))
                ts = normalize_ts(entry)

                # Creating a unique ID for each news article
                news_id = hashlib.md5((title + str(ts) + ticker).encode()).hexdigest()[:24]

                # Classify news into categories
                category = classify_category(title + " " + summary)

                # Sentiment (FinBERT) - use cache when possible
                txt_key = hashlib.md5((title + summary).encode()).hexdigest()
                if txt_key in sentiment_cache:
                    sentiment_label, sentiment_score = sentiment_cache[txt_key]
                else:
                    try:
                        sentiment_label, sentiment_score = analyze_sentiment(title + " " + summary)
                    except Exception:
                        sentiment_label, sentiment_score = ("neutral", 0.0)
                    sentiment_cache[txt_key] = (sentiment_label, float(sentiment_score))

                # Store the news
                rows.append([news_id, ticker, ts, title, summary, category, sentiment_label, float(sentiment_score)])
        else:
            # fallback: try HTML scraping for site-level headlines (simple approach)
            try:
                r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200:
                    html = r.text
                    # site-specific scraping heuristics
                    titles = []
                    if 'livemint' in url:
                        # look for headline anchors
                        soup = BeautifulSoup(html, "html.parser")
                        for a in soup.select('a.headline, a.story-title, h2 a'):
                            ttxt = a.get_text(strip=True)
                            if ttxt:
                                titles.append(ttxt)
                    elif 'moneycontrol' in url:
                        soup = BeautifulSoup(html, "html.parser")
                        for a in soup.select('a, h2, h3'):
                            ttxt = a.get_text(strip=True)
                            if ttxt and len(ttxt) > 20:
                                titles.append(ttxt)
                    elif 'indiatimes' in url or 'economictimes' in url:
                        soup = BeautifulSoup(html, "html.parser")
                        for a in soup.select('.eachStory a, h2 a, h3 a'):
                            ttxt = a.get_text(strip=True)
                            if ttxt:
                                titles.append(ttxt)
                    else:
                        soup = BeautifulSoup(html, "html.parser")
                        for a in soup.find_all('a')[:30]:
                            ttxt = a.get_text(strip=True)
                            if ttxt and len(ttxt) > 30:
                                titles.append(ttxt)

                    for title in list(dict.fromkeys(titles))[:10]:
                        summary = ''
                        ts = int(datetime.now().timestamp())
                        news_id = hashlib.md5((title + str(ts) + ticker).encode()).hexdigest()[:24]
                        category = classify_category(title + " " + summary)
                        txt_key = hashlib.md5((title + summary).encode()).hexdigest()
                        if txt_key in sentiment_cache:
                            sentiment_label, sentiment_score = sentiment_cache[txt_key]
                        else:
                            try:
                                sentiment_label, sentiment_score = analyze_sentiment(title + " " + summary)
                            except Exception:
                                sentiment_label, sentiment_score = ("neutral", 0.0)
                            sentiment_cache[txt_key] = (sentiment_label, float(sentiment_score))
                        rows.append([news_id, ticker, ts, title, summary, category, sentiment_label, float(sentiment_score)])
            except Exception:
                continue

    # Save to the in-memory storage
    for row in rows:
        tk = row[1]
        if tk not in news_storage:
            news_storage[tk] = []
        news_storage[tk].append(row)

    print(f"Fetched {len(rows)} news articles for {ticker}")

# Function to run the news ingestion for each ticker
def run_once():
    for ticker in TICKERS:
        fetch_news(ticker)


def fetch_news_background(tickers: list, progress_callback=None):
    """Fetch news for a list of tickers in background and update news_fetch_progress."""
    news_fetch_progress["running"] = True
    news_fetch_progress["total"] = len(tickers)
    news_fetch_progress["fetched"] = 0
    news_fetch_progress["failed"] = []

    for t in tickers:
        try:
            fetch_news(t)
            news_fetch_progress["fetched"] += 1
        except Exception as e:
            news_fetch_progress["failed"].append((t, str(e)))
        if callable(progress_callback):
            try:
                progress_callback(news_fetch_progress)
            except Exception:
                pass

    news_fetch_progress["running"] = False
    return news_fetch_progress


def start_background_fetch(tickers: list, progress_callback=None):
    thread = threading.Thread(target=fetch_news_background, args=(tickers, progress_callback), daemon=True)
    thread.start()
    return thread

if __name__ == "__main__":
    run_once()