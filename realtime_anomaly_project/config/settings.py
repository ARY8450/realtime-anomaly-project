import os
import yaml
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# Load weights from the weights.yaml file (look relative to this settings.py file)
def load_weights():
    config_dir = os.path.dirname(__file__)
    weights_path = os.path.join(config_dir, "weights.yaml")
    if not os.path.exists(weights_path):
        # return safe defaults if no weights file present
        return {
            "fusion_weights": {"alpha": 0.33, "beta": 0.33, "gamma": 0.34},
            "anomaly_thresholds": {"sentiment_threshold": 0.5, "z_k": 2.5, "rsi_threshold": 70},
            "deep_anomaly_params": {"reconstruction_error_threshold": 0.8}
        }
    try:
        with open(weights_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {
            "fusion_weights": {"alpha": 0.33, "beta": 0.33, "gamma": 0.34},
            "anomaly_thresholds": {"sentiment_threshold": 0.5, "z_k": 2.5, "rsi_threshold": 70},
            "deep_anomaly_params": {"reconstruction_error_threshold": 0.8}
        }

# Load the weights into a variable
weights = load_weights()

def fetch_nifty50_from_nse(timeout=10):
    # try official NSE csv (archive endpoint). May be blocked; this is a best-effort attempt.
    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
        "https://www1.nseindia.com/content/indices/ind_nifty50list.csv"
    ]
    headers = {"User-Agent": "python-requests/1.0"}
    for u in urls:
        try:
            r = requests.get(u, headers=headers, timeout=timeout)
            if r.status_code == 200 and r.text:
                df = pd.read_csv(pd.compat.StringIO(r.text))
                if "Symbol" in df.columns or "SYMBOL" in df.columns:
                    col = "Symbol" if "Symbol" in df.columns else "SYMBOL"
                    symbols = df[col].astype(str).str.strip().tolist()
                    return [s if s.endswith(".NS") else f"{s}.NS" for s in symbols]
        except Exception:
            continue
    return None

def fetch_nifty50_from_wikipedia(timeout=10):
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/NIFTY_50")
        for t in tables:
            if "Symbol" in t.columns or "Company" in t.columns:
                # try to find a Symbol-like column
                for col in ["Symbol", "Ticker", "SYMBOL"]:
                    if col in t.columns:
                        syms = t[col].astype(str).str.strip().tolist()
                        return [s if s.endswith(".NS") else f"{s}.NS" for s in syms]
    except Exception:
        pass
    return None

# Stock tickers (Yahoo Finance style). Prefer setting TICKERS in .env as a comma-separated list.
# If not provided in .env, the code will try to read config/nifty50.txt (one ticker per line).
# Fallback default is a small sample so code still runs.
def default_tickers():
    # attempt to fetch live list; fall back to a small sample
    syms = fetch_nifty50_from_nse()
    if syms:
        return syms
    syms = fetch_nifty50_from_wikipedia()
    if syms:
        return syms
    # fallback (minimal sample) — encourage filling .env or nifty50.txt with the canonical list
    return ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]

TICKERS_ENV = os.getenv("TICKERS")
if TICKERS_ENV:
    TICKERS = [t.strip() if t.strip().endswith(".NS") else f"{t.strip()}.NS" for t in TICKERS_ENV.split(",") if t.strip()]
else:
    TICKERS = default_tickers()

# Request 60 days of history and 2-minute bars
LOOKBACK = os.getenv("LOOKBACK", "60d")
INTERVAL = os.getenv("INTERVAL", "2m")

# Scheduler cadence (in minutes)
FETCH_MIN = int(os.getenv("FETCH_MIN", "2"))
STATS_MIN = int(os.getenv("STATS_MIN", "2"))

# Anomaly detection params
Z_ROLL = int(os.getenv("Z_ROLL", "20"))
Z_K = float(os.getenv("Z_K", str(weights.get("anomaly_thresholds", {}).get("z_k", 2.5))))
RSI_N = int(os.getenv("RSI_N", "14"))

# Fusion & thresholds
FUSION_WEIGHTS = weights.get("fusion_weights", {"alpha": 0.33, "beta": 0.33, "gamma": 0.34})
ANOMALY_THRESHOLDS = weights.get("anomaly_thresholds", {"sentiment_threshold": 0.5, "z_k": Z_K})
DEEP_ANOMALY_PARAMS = weights.get("deep_anomaly_params", {})

# Dashboard refresh (seconds)
REFRESH_SEC = int(os.getenv("REFRESH_SEC", "60"))

# Test printing loaded settings and weights (optional)
print(f"TICKERS: {TICKERS[:10]}{'...' if len(TICKERS)>10 else ''}")
print(f"LOOKBACK: {LOOKBACK}")
print(f"INTERVAL: {INTERVAL}")
print(f"Fusion Weights: {FUSION_WEIGHTS}")
print(f"Anomaly Thresholds: {ANOMALY_THRESHOLDS}")
print(f"Deep Anomaly Params: {DEEP_ANOMALY_PARAMS}")
