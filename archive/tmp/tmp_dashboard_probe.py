import json, sqlite3, urllib.request, re, sys

cache_path = 'realtime_anomaly_project/sql_db/realtime_live_cache.json'
print('=== LIVE CACHE CHECK ===')
try:
    with open(cache_path, 'r', encoding='utf-8') as f:
        j = json.load(f)
    print('cache ts:', j.get('ts'))
    tickers = ['ADANIPORTS.NS', 'RELIANCE.NS', 'TCS.NS']
    for t in tickers:
        d = j.get('data', {}).get(t)
        print(f'  {t}:', d)
except Exception as e:
    print('Failed to read live cache:', e)

print('\n=== STREAMLIT HTTP CHECK ===')
try:
    resp = urllib.request.urlopen('http://localhost:8501', timeout=6)
    html = resp.read(8192).decode('utf-8', errors='replace')
    m = re.search(r'<title>(.*?)</title>', html, re.I|re.S)
    title = m.group(1).strip() if m else '<no title found>'
    print('HTTP status:', resp.getcode(), 'title:', title)
except Exception as e:
    print('HTTP check failed:', e)

print('\n=== SQLITE DB CHECK ===')
try:
    db_path = 'realtime_anomaly_project/sql_db/realtime_anomaly.db'
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print('tables:', tables)
    found = False
    for tbl in tables:
        try:
            cur.execute(f"PRAGMA table_info('{tbl}')")
            cols = [r[1] for r in cur.fetchall()]
            if 'ticker' in cols and 'ts' in cols:
                cur.execute(f"SELECT * FROM {tbl} WHERE ticker=? ORDER BY ts DESC LIMIT 1", ('ADANIPORTS.NS',))
                row = cur.fetchone()
                print('table', tbl, 'latest ADANIPORTS.NS row:', row)
                found = True
                break
        except Exception:
            continue
    if not found:
        print('No table with ticker+ts columns found or no rows for ADANIPORTS.NS')
    conn.close()
except Exception as e:
    print('DB check failed:', e)

print('\n=== DONE ===')
