"""
Configuration settings for the Real-Time Anomaly Detection System
"""

# Nifty-Fifty Tickers
TICKERS = [
    "ADANIPORTS.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS",
    "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS",
    "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
    "HCLTECH.NS", "HDFC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS",
    "IOC.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS",
    "SUNPHARMA.NS", "TATASTEEL.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS", "TECHM.NS", "TCS.NS"
]

# Technical Analysis Parameters
Z_ROLL = 20  # Rolling window for Z-score calculation
Z_K = 2.5    # Z-score threshold for anomaly detection
RSI_N = 14   # RSI period

# System Configuration
DEFAULT_LOOKBACK = "1y"
DEFAULT_UPDATE_INTERVAL = 30
MAX_TICKERS = 50

# News Sources
NEWS_SOURCES = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline",
    "https://www.moneycontrol.com/rss/business.xml",
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "https://www.financialexpress.com/market/rss",
    "https://rss.cnn.com/rss/money_news_international.rss",
    "https://www.ndtvprofit.com/rss",
    "https://www.business-standard.com/rss"
]

# Fusion Weights
FUSION_WEIGHTS = {
    'anomaly_weight': 0.30,
    'sentiment_weight': 0.25,
    'trend_weight': 0.25,
    'seasonality_weight': 0.20
}

# Anomaly Detection Thresholds
ANOMALY_THRESHOLDS = {
    'sentiment_threshold': 0.5,
    'z_score_threshold': 2.5,
    'rsi_threshold': 70
}
