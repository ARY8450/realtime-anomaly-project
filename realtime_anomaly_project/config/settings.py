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
TICKERS = os.getenv("TICKERS", "RELIANCE.NS,TCS.NS,HDFCBANK.NS").split(",")

# Lookback period and interval for Yahoo Finance API
LOOKBACK = os.getenv("LOOKBACK", "30d")    # e.g., 7d, 30d, 60d
INTERVAL = os.getenv("INTERVAL", "1h")     # valid: 1m, 5m, 15m, 1h, 1d, etc.

# Scheduler cadence (in minutes)
FETCH_MIN = int(os.getenv("FETCH_MIN", "10"))  # e.g., fetch price data every 10 minutes
STATS_MIN = int(os.getenv("STATS_MIN", "10"))  # e.g., run anomaly detection every 10 minutes

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
