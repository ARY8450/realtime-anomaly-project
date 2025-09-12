try:
    import psutil
except ImportError:
    raise ImportError("psutil not found. Install it with: pip install psutil") from None

# check if port 8501 is listening
listening=False
for c in psutil.net_connections():
    try:
        if c.status == 'LISTEN':
            lport = None
            if hasattr(c, 'laddr'):
                laddr = c.laddr
                # prefer attribute access but use getattr to avoid attribute errors on tuples
                lport = getattr(laddr, 'port', None)
                if lport is None:
                    if isinstance(laddr, (tuple, list)) and len(laddr) >= 2:
                        lport = laddr[1]
            if lport == 8501:
                listening = True
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
