import urllib.request, re, time

for i in range(6):
    try:
        resp = urllib.request.urlopen('http://localhost:8501', timeout=4)
        html = resp.read(8192).decode('utf-8', errors='replace')
        m = re.search(r'<title>(.*?)</title>', html, re.I|re.S)
        title = m.group(1).strip() if m else '<no title>'
        print('OK', resp.getcode(), title)
        break
    except Exception as e:
        print('attempt', i, 'failed:', e)
        time.sleep(1)
else:
    print('streamlit not responding after retries')
