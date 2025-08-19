import os
from dotenv import load_dotenv

load_dotenv()

TICKERS = os.getenv("TICKERS", "RELIANCE.NS,TCS.NS,HDFCBANK.NS").split(",")

LOOKBACK = os.getenv("LOOKBACK", "30d")  
INTERVAL = os.getenv("INTERVAL", "1h")   

FETCH_MIN = int(os.getenv("FETCH_MIN", "10"))   
STATS_MIN = int(os.getenv("STATS_MIN", "10"))  

Z_ROLL = int(os.getenv("Z_ROLL", "20")) 
Z_K = float(os.getenv("Z_K", "2.5"))     
RSI_N = int(os.getenv("RSI_N", "14"))     

REFRESH_SEC = int(os.getenv("REFRESH_SEC", "60"))
