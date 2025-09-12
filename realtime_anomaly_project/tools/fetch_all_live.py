"""Fetch live features for all configured tickers and print a compact table.
This script avoids importing `dashboard.py` to prevent Streamlit runtime warnings.
"""
import time
from realtime_anomaly_project.config import settings
from realtime_anomaly_project.app._dashboard_helpers import fetch_live_features

TICKERS = getattr(settings, 'TICKERS', []) or []
print(f"Tickers to fetch: {len(TICKERS)}")
rows = []
for i, t in enumerate(TICKERS, 1):
    try:
        feat = fetch_live_features(t)
    except Exception as e:
        feat = {}
    rows.append((t, feat))
    print(f"{i}/{len(TICKERS)}: {t} -> open={feat.get('open')} high={feat.get('high')} low={feat.get('low')} vol={feat.get('volume')}")
    # be polite to yfinance
    time.sleep(0.3)

print('\nSummary:')
for t, f in rows:
    try:
        print(t, '\t', f.get('open'), f.get('high'), f.get('low'), f.get('volume'), f.get('marketCap'))
    except Exception:
        print(t, '\t', 'error')
