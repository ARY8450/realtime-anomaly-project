import os
from dotenv import load_dotenv
import yaml

# Load environment variables from .env file (if present)
load_dotenv()

# Load weights from the weights.yaml file (look relative to this settings.py file)
def load_weights():
    config_dir = os.path.dirname(__file__)
    weights_path = os.path.join(config_dir, "weights.yaml")
    if not os.path.exists(weights_path):
        print(f"Warning: weights.yaml not found at {weights_path}. Using default weights.")
        return {
            "fusion_weights": {"z": 0.5, "rsi": 0.5},
            "anomaly_thresholds": {"z_k": 2.5},
            "deep_anomaly_params": {},
            "sentiment_model_weights": {}
        }
    try:
        with open(weights_path, "r") as file:
            return yaml.safe_load(file) or {}
    except Exception as e:
        print(f"Warning: failed to load weights.yaml ({e}). Using default weights.")
        return {
            "fusion_weights": {"z": 0.5, "rsi": 0.5},
            "anomaly_thresholds": {"z_k": 2.5},
            "deep_anomaly_params": {},
            "sentiment_model_weights": {}
        }

# Load the weights into a variable
weights = load_weights()

# Stock tickers (Yahoo Finance style). You can edit this in the .env file.
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

# Map problematic tickers to alternate symbols to try when Yahoo returns no data.
# Example: HDFC.NS merged/renamed — try HDFCBANK.NS as an alias (adjust if you know a better mapping).
SYMBOL_ALIASES = {
    "HDFC.NS": "HDFCBANK.NS"
}

# batch control to avoid hitting provider limits
BATCH_SIZE = 10
BATCH_SLEEP_SECONDS = 5

# default fetch/scheduler settings (if present earlier)
LOOKBACK = globals().get("LOOKBACK", "30d")
INTERVAL = globals().get("INTERVAL", "1h")
FETCH_MIN = globals().get("FETCH_MIN", 15)
STATS_MIN = globals().get("STATS_MIN", 15)

# Anomaly detection parameters
Z_ROLL = int(os.getenv("Z_ROLL", "20"))  # rolling window size for z-score
Z_K = float(os.getenv("Z_K", str(weights.get("anomaly_thresholds", {}).get("z_k", 2.5))))
RSI_N = int(os.getenv("RSI_N", "14"))    # window size for RSI (Relative Strength Index)

# Dashboard auto-refresh interval (in seconds)
REFRESH_SEC = int(os.getenv("REFRESH_SEC", "60"))  # e.g., 60 seconds

# Fusion Weights for combining different anomaly scores
FUSION_WEIGHTS = weights.get("fusion_weights", {"z": 0.5, "rsi": 0.5})

# Anomaly detection thresholds from weights.yaml
ANOMALY_THRESHOLDS = weights.get("anomaly_thresholds", {"z_k": Z_K})

# Deep Anomaly detection parameters
DEEP_ANOMALY_PARAMS = weights.get("deep_anomaly_params", {})

# Sentiment model weights from weights.yaml
SENTIMENT_MODEL_WEIGHTS = weights.get("sentiment_model_weights", {})

# Test printing loaded settings and weights (optional)
print(f"TICKERS: {TICKERS}")
print(f"LOOKBACK: {LOOKBACK}")
print(f"INTERVAL: {INTERVAL}")
print(f"Fusion Weights: {FUSION_WEIGHTS}")
print(f"Anomaly Thresholds: {ANOMALY_THRESHOLDS}")
print(f"Deep Anomaly Params: {DEEP_ANOMALY_PARAMS}")

# Optional mapping of tickers to sectors/domains to support dashboard grouping and filtering
try:
    import yaml
    tickers_yaml = os.path.join(os.path.dirname(__file__), "ticker_sectors.yaml")
    if os.path.exists(tickers_yaml):
        with open(tickers_yaml, "r") as f:
            TICKER_SECTORS = yaml.safe_load(f) or {}
    else:
        TICKER_SECTORS = {}
except Exception:
    # fallback to empty mapping if yaml isn't available
    TICKER_SECTORS = {}
