import json, urllib.request, os, sys
p='realtime_anomaly_project/sql_db/realtime_live_cache.json'
print('cache exists', os.path.exists(p))
if not os.path.exists(p):
    print('cache missing')
    sys.exit(1)
d=json.load(open(p))
keys=list(d.get('data',{}).keys())
if not keys:
    print('no tickers in cache')
    sys.exit(1)
sample=keys[:3]
print('sample tickers:', sample)
vals={}
for k in sample:
    v=d['data'][k].get('open')
    vals[k]=None if v is None else f"{v:.2f}"
print('sample opens:', vals)
try:
    with urllib.request.urlopen('http://localhost:8501', timeout=5) as resp:
        page=resp.read().decode('utf-8',errors='ignore')
    found=[]
    for k in sample:
        found.append((k, k in page, vals[k] and vals[k] in page))
    print('found on page (ticker, ticker_in_page, open_in_page):')
    for r in found:
        print(r)
except Exception as e:
    print('error fetching page:', e)
    sys.exit(2)
