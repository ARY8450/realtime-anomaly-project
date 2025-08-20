import yfinance as yf, pandas as pd
t="ADANIPORTS.NS"
for period in ("30d","60d"):
    for interval in ("1h","1d","30m"):
        try:
            raw = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=True)
            print(period, interval, "shape:", getattr(raw,'shape',None))
        except Exception as e:
            print("error", period, interval, e)