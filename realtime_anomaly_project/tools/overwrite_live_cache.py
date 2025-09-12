import os
import json
import time

PAYLOAD = {
    'open': 1338.0,
    'high': 1344.800048828125,
    'low': 1327.5999755859375,
    'volume': 1774607.0,
    'marketCap': 2878170726400.0,
    'trailingPE': 25.500479,
    'dividendYield': 0.52,
    'fiftyTwoWeekHigh': 1494.0,
    'fiftyTwoWeekLow': 995.65,
    'qtr_div_amt': 1.75,
}


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    cache_path = os.path.join(repo_root, 'realtime_anomaly_project', 'sql_db', 'realtime_live_cache.json')

    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception:
                data = {}
    else:
        data = {}

    ts = int(time.time())
    existing = data.get('data', {}) if isinstance(data, dict) else {}

    # Overwrite every ticker in existing data. If none exist, create entries for a small default list.
    if existing:
        keys = list(existing.keys())
    else:
        keys = [
            'ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS',
            'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS'
        ]

    new_data = {k: PAYLOAD for k in keys}

    out = {'ts': ts, 'data': new_data}

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f'Wrote payload for {len(keys)} tickers to {cache_path} with ts={ts}')


if __name__ == '__main__':
    main()
