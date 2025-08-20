import sys
import os
# ensure project root parent is on sys.path so package imports resolve when running file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from realtime_anomaly_project.config.settings import FUSION_WEIGHTS, ANOMALY_THRESHOLDS, TICKERS
import numpy as np
import pandas as pd
from realtime_anomaly_project.sentiment_module.finbert_sentiment import analyze_sentiment
from realtime_anomaly_project.statistical_anomaly.classical_methods import compute_anomalies
from realtime_anomaly_project.statistical_anomaly.unsupervised import compute_unsupervised_anomalies
from realtime_anomaly_project.deep_anomaly.transformer_ae import compute_deep_anomalies

# Function to combine anomaly scores using the fusion weights
def fuse_anomaly_scores(stat_score, sentiment_score, unsupervised_score, deep_score):
    """
    Combine anomaly scores using fusion weights
    """
    fusion_score = (
        FUSION_WEIGHTS.get('alpha', 0.33) * sentiment_score + 
        FUSION_WEIGHTS.get('beta', 0.33) * stat_score + 
        FUSION_WEIGHTS.get('gamma', 0.34) * deep_score
    )
    return fusion_score

# Function to generate anomaly score and decision
def get_anomaly_decision(ticker, stat_dict, sent_dict, unsup_dict, deep_dict):
    """
    Combine different anomaly scores for the final decision.
    Takes precomputed maps to avoid recomputing inside the loop.
    """
    # defensive defaults if any of the compute_* returned None
    stat_dict = stat_dict or {}
    sent_dict = sent_dict or {}
    unsup_dict = unsup_dict or {}
    deep_dict = deep_dict or {}

    # stat score
    stat_score = float(stat_dict.get(ticker, {}).get('stat_score', 0))

    # sentiment: expect (label, score) or a single score; default 0
    sent_val = sent_dict.get(ticker, 0)
    if isinstance(sent_val, (list, tuple)) and len(sent_val) > 1:
        sentiment_score = float(sent_val[1])
    else:
        try:
            sentiment_score = float(sent_val)
        except Exception:
            sentiment_score = 0.0

    # unsupervised score (array-like -> mean) or scalar
    unsup_val = unsup_dict.get(ticker, 0)
    try:
        unsupervised_score = float(np.mean(unsup_val)) if hasattr(unsup_val, "__iter__") else float(unsup_val)
    except Exception:
        unsupervised_score = 0.0

    # deep score (array-like -> mean) or scalar
    deep_val = deep_dict.get(ticker, 0)
    try:
        deep_score = float(np.mean(deep_val)) if hasattr(deep_val, "__iter__") else float(deep_val)
    except Exception:
        deep_score = 0.0

    fusion_score = fuse_anomaly_scores(stat_score, sentiment_score, unsupervised_score, deep_score)

    decision = "Anomalous" if fusion_score > ANOMALY_THRESHOLDS.get('sentiment_threshold', 0.5) else "Normal"
    
    return {
        "fusion_score": fusion_score,
        "decision": decision
    }

# Function to run anomaly detection and get fusion scores for all tickers
def run_fusion():
    """
    Run the fusion process for all tickers and print out the anomaly decision.
    Compute each expensive detector once and reuse results.
    """
    # compute everything once (some may return None)
    stat = compute_anomalies() or {}
    unsup = compute_unsupervised_anomalies() or {}
    deep = compute_deep_anomalies() or {}

    # compute sentiment per ticker and store
    sent = {}
    for ticker in TICKERS:
        try:
            sent[ticker] = analyze_sentiment(ticker)
        except Exception:
            sent[ticker] = 0

    fusion_results = {}
    for ticker in TICKERS:
        result = get_anomaly_decision(ticker, stat, sent, unsup, deep)
        fusion_results[ticker] = result
    
    for ticker, result in fusion_results.items():
        print(f"{ticker}: Fusion Score = {result['fusion_score']:.2f}, Decision: {result['decision']}")

    return fusion_results

if __name__ == "__main__":
    run_fusion()
