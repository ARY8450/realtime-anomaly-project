import psutil

# check if port 8501 is listening
listening=False
for c in psutil.net_connections():
    try:
        if c.status=='LISTEN' and c.laddr.port==8501:
            listening=True
            break
    except Exception:
        continue
print('8501 listening:', listening)
# list python processes that look like streamlit
for p in psutil.process_iter(['pid','name','cmdline']):
    try:
        name=(p.info.get('name') or '').lower()
        cmd=p.info.get('cmdline') or []
        if 'python' in name or any('streamlit' in (c or '').lower() for c in cmd):
            print(p.info)
    except Exception:
        pass
