import sys
import os

# Ensure realtime_anomaly_project folder is on sys.path so "config", "data_ingestion", etc. can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from config.settings import TICKERS, TICKER_SECTORS
import config.settings as settings
import yaml

# load named portfolios from YAML
PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'portfolios.yaml')
try:
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, 'r') as pf:
            SAVED_PORTFOLIOS = yaml.safe_load(pf) or {}
    else:
        SAVED_PORTFOLIOS = {}
except Exception:
    SAVED_PORTFOLIOS = {}
from data_ingestion.yahoo import data_storage
from statistical_anomaly.classical_methods import compute_anomalies
from statistical_anomaly.unsupervised import compute_unsupervised_anomalies
from statistical_anomaly.bulk_fetch_all import bulk_fetch_all
import threading
import time
from data_ingestion import news_ingest
import numpy as np
from fusion import recommender
from data_ingestion import news_ingest

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")

st.title("Real-Time Anomaly Detection Dashboard")

# --- Portfolio UI ---
if "portfolio_selected" not in st.session_state:
    st.session_state["portfolio_selected"] = []
if "saved_portfolio" not in st.session_state:
    st.session_state["saved_portfolio"] = []

st.sidebar.subheader("Portfolio")
# Sector filter
all_sectors = sorted(set(TICKER_SECTORS.values())) if isinstance(TICKER_SECTORS, dict) else []
selected_sectors = st.sidebar.multiselect("Filter by sector", options=["All"] + all_sectors, default=["All"]) if all_sectors else ["All"]

available_tickers = TICKERS
if selected_sectors and "All" not in selected_sectors:
    allowed = [t for t in TICKERS if TICKER_SECTORS.get(t) in selected_sectors]
    available_tickers = allowed

st.session_state["portfolio_selected"] = st.sidebar.multiselect(
    "Select tickers for your portfolio", options=available_tickers, default=st.session_state.get("portfolio_selected", [])
)
# path to ticker sectors YAML (admin editor will write here)
TICKER_SECTORS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'ticker_sectors.yaml')

# Small admin panel to edit ticker_sectors.yaml
with st.sidebar.expander("Admin: Edit ticker sectors YAML", expanded=False):
    try:
        if os.path.exists(TICKER_SECTORS_FILE):
            with open(TICKER_SECTORS_FILE, 'r', encoding='utf-8') as f:
                existing = f.read()
        else:
            existing = yaml.safe_dump(TICKER_SECTORS or {})
    except Exception:
        existing = yaml.safe_dump(TICKER_SECTORS or {})

    edited = st.text_area("Edit ticker -> sector mapping (YAML)", value=existing, height=300, key="admin_sectors_text")
    if st.button("Save sectors YAML"):
        try:
            parsed = yaml.safe_load(edited) or {}
            # ensure mapping
            if not isinstance(parsed, dict):
                st.error("YAML must contain a mapping of ticker -> sector")
            else:
                with open(TICKER_SECTORS_FILE, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(parsed, f, sort_keys=True)
                # update in-memory settings so the rest of the dashboard picks up changes
                settings.TICKER_SECTORS = parsed
                TICKER_SECTORS = parsed
                st.success("Saved ticker sectors YAML and reloaded mapping")
        except Exception as e:
            st.error(f"Failed to save YAML: {e}")
# Named portfolio selector
st.sidebar.markdown("---")
saved_names = list(SAVED_PORTFOLIOS.keys())
selected_saved = st.sidebar.selectbox("Load saved portfolio", options=["(none)"] + saved_names)
if selected_saved and selected_saved != "(none)":
    if st.sidebar.button("Load selected saved portfolio"):
        payload = SAVED_PORTFOLIOS.get(selected_saved)
        # support old format (list of tickers) and new format (dict with tickers+holdings)
        if isinstance(payload, dict) and 'tickers' in payload:
            st.session_state["portfolio_selected"] = list(payload.get('tickers', []))
            # restore holdings if present
            holdings = payload.get('holdings', {}) or {}
            for t, hh in holdings.items():
                q_key = f'hold_qty_{t}'
                p_key = f'hold_price_{t}'
                try:
                    st.session_state[q_key] = float(hh.get('qty', 0.0))
                    st.session_state[p_key] = float(hh.get('entry', 0.0))
                except Exception:
                    pass
        elif isinstance(payload, list):
            st.session_state["portfolio_selected"] = list(payload)
        else:
            st.session_state["portfolio_selected"] = list(payload or [])
        st.sidebar.success(f"Loaded portfolio: {selected_saved}")

new_port_name = st.sidebar.text_input("Save current selection as", value="")
if st.sidebar.button("Save current selection as named portfolio") and new_port_name:
    # capture holdings currently stored in session_state
    holdings_snapshot = {}
    # session_state keys may not be typed as str for static analyzers; coerce explicitly
    def _safe_float(x, default=0.0):
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    for k in list(map(str, st.session_state.keys())):
        v = st.session_state.get(k)
        if k.startswith('hold_qty_'):
            t = k[len('hold_qty_'):]
            holdings_snapshot.setdefault(t, {})['qty'] = _safe_float(v, 0.0)
        if k.startswith('hold_price_'):
            t = k[len('hold_price_'):]
            holdings_snapshot.setdefault(t, {})['entry'] = _safe_float(v, 0.0)
    SAVED_PORTFOLIOS[new_port_name] = {
        'tickers': list(st.session_state.get("portfolio_selected", [])),
        'holdings': holdings_snapshot,
    }
    try:
        with open(PORTFOLIO_FILE, 'w') as pf:
            yaml.safe_dump(SAVED_PORTFOLIOS, pf)
        st.sidebar.success(f"Saved portfolio {new_port_name}")
    except Exception as e:
        st.sidebar.error(f"Failed to save: {e}")

if saved_names and st.sidebar.button("Delete selected saved portfolio") and selected_saved and selected_saved != "(none)":
    SAVED_PORTFOLIOS.pop(selected_saved, None)
    try:
        with open(PORTFOLIO_FILE, 'w') as pf:
            yaml.safe_dump(SAVED_PORTFOLIOS, pf)
        st.sidebar.success(f"Deleted portfolio {selected_saved}")
    except Exception as e:
        st.sidebar.error(f"Failed to delete: {e}")
if st.sidebar.button("Save portfolio as default"):
    st.session_state["saved_portfolio"] = list(st.session_state.get("portfolio_selected", []))
    st.sidebar.success("Portfolio saved as default")
if st.sidebar.button("Load saved portfolio") and st.session_state.get("saved_portfolio"):
    st.session_state["portfolio_selected"] = list(st.session_state.get("saved_portfolio") or [])
    st.sidebar.success("Loaded saved portfolio")

# Ticker selector (single ticker view)
ticker = st.sidebar.selectbox("Select Ticker", TICKERS)

# --- Model training controls ---
st.sidebar.markdown("---")
st.sidebar.subheader("Prediction model")
train_horizon = st.sidebar.number_input("Training horizon (steps)", min_value=1, max_value=60, value=1)
train_thresh = st.sidebar.number_input("Train threshold (fraction)", min_value=0.0, max_value=0.1, value=0.001, step=0.001)
train_on = st.sidebar.selectbox("Train on", ["Selected portfolio", "All tickers"], index=0)
if st.sidebar.button("Train prediction model"):
    # gather ticker dfs
    with st.spinner("Preparing training data and training model..."):
        ticker_dfs = {}
        train_list = st.session_state.get("portfolio_selected") or [] if train_on == "Selected portfolio" else TICKERS
        for t in train_list:
            df_t = data_storage.get(t)
            if df_t is None or df_t.empty:
                try:
                    from realtime_anomaly_project.statistical_anomaly.show_results import load_ticker_df as _load
                    df_t = _load(t)
                except Exception:
                    df_t = None
            if df_t is not None and not df_t.empty:
                ticker_dfs[t] = df_t
        try:
            model, report, acc = recommender.train_model(ticker_dfs, horizon=int(train_horizon), thresh=float(train_thresh))
            st.sidebar.success(f"Model trained — accuracy {acc:.3f}")
            st.session_state['recommender_trained'] = True
            st.session_state['recommender_report'] = report
        except Exception as e:
            st.sidebar.error(f"Training failed: {e}")

use_model_predictions = st.sidebar.checkbox("Use model predictions in portfolio table", value=True)

st.header(f"Stock Data for {ticker}")
df = data_storage.get(ticker)
if not (isinstance(df, pd.DataFrame) and not df.empty):
    try:
        from realtime_anomaly_project.statistical_anomaly.show_results import load_ticker_df
        df = load_ticker_df(ticker)
        # cache into in-memory store for faster subsequent access (lazy loader)
        if isinstance(df, pd.DataFrame) and not df.empty:
            try:
                data_storage[ticker] = df
                st.sidebar.info(f"Loaded {ticker} from DB into in-memory cache")
            except Exception:
                # if data_storage isn't writable (unlikely), ignore caching
                pass
    except Exception:
        df = None

    # Fetch-all control
    st.sidebar.markdown("---")
    if st.sidebar.button("Fetch all tickers to DB"):
        with st.spinner("Fetching all tickers (this may take several minutes)..."):
            try:
                results = bulk_fetch_all()
                st.sidebar.success(f"Fetch completed. Fetched: {len(results.get('fetched',[]))}, Failed: {len(results.get('failed',[]))}")
            except Exception as e:
                st.sidebar.error(f"Fetch-all failed: {e}")
    # Load-all control: populate in-memory cache from DB for fast UI browsing
    if 'load_all_progress' not in st.session_state:
        st.session_state['load_all_progress'] = 0
        st.session_state['load_all_loaded'] = 0
        st.session_state['load_all_failed'] = []
        st.session_state['load_all_running'] = False
        st.session_state['load_all_done'] = False

    def _load_all_background():
        try:
            from realtime_anomaly_project.statistical_anomaly.show_results import load_ticker_df
        except Exception:
            load_ticker_df = None
        total = len(TICKERS)
        loaded = 0
        failed = []
        for i, t in enumerate(TICKERS, 1):
            try:
                df_t = load_ticker_df(t) if load_ticker_df is not None else None
            except Exception:
                df_t = None
            if isinstance(df_t, pd.DataFrame) and not df_t.empty:
                try:
                    data_storage[t] = df_t
                    loaded += 1
                except Exception:
                    failed.append(t)
            else:
                failed.append(t)
            # update progress
            st.session_state['load_all_progress'] = int(i / total * 100)
            st.session_state['load_all_loaded'] = loaded
            st.session_state['load_all_failed'] = failed[:50]
            # small sleep to yield
            time.sleep(0.01)
        st.session_state['load_all_running'] = False
        st.session_state['load_all_done'] = True

    if st.sidebar.button("Load all DB tickers into memory (background)"):
        if st.session_state.get('load_all_running'):
            st.sidebar.info("Load-all already running")
        else:
            st.session_state['load_all_running'] = True
            st.session_state['load_all_done'] = False
            st.session_state['load_all_progress'] = 0
            st.session_state['load_all_loaded'] = 0
            st.session_state['load_all_failed'] = []
            thread = threading.Thread(target=_load_all_background, daemon=True)
            thread.start()
            st.sidebar.success("Started background load; progress will appear below")

    # show progress / status
    if st.session_state.get('load_all_running'):
        st.sidebar.write(f"Loading... {st.session_state.get('load_all_progress',0)}%")
        st.sidebar.progress(st.session_state.get('load_all_progress', 0))
        st.sidebar.write(f"Loaded: {st.session_state.get('load_all_loaded', 0)}")
    elif st.session_state.get('load_all_done'):
        st.sidebar.success(f"Load complete: {st.session_state.get('load_all_loaded', 0)} tickers loaded")
        failed_list = st.session_state.get('load_all_failed') or []
        if failed_list:
            st.sidebar.warning(f"Failed: {len(failed_list)} tickers (showing up to 10): {failed_list[:10]}")

    # Clear in-memory cache button
    if st.sidebar.button("Clear in-memory cache"):
        data_storage.clear()
        # reset session state progress
        st.session_state['load_all_progress'] = 0
        st.session_state['load_all_loaded'] = 0
        st.session_state['load_all_failed'] = []
        st.session_state['load_all_running'] = False
        st.session_state['load_all_done'] = False
        st.sidebar.success("In-memory cache cleared")
if df is not None and not df.empty:
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    df = df[~df.index.isna()]
    df = df.loc[~df.index.duplicated(keep="first")]

    lookback_opt = st.sidebar.selectbox("Lookback", ["1d", "7d", "30d", "90d", "All"], index=2)
    z_thresh = st.sidebar.slider("Z-score threshold", 0.5, 10.0, 2.5, 0.1)
    rsi_period = st.sidebar.number_input("RSI period", min_value=5, max_value=50, value=14)
    rsi_thresh = st.sidebar.slider("RSI threshold", 50, 90, 70)

    if lookback_opt != "All":
        n = int(lookback_opt.rstrip("d"))
        start = df.index.max() - pd.Timedelta(days=n)
        df = df.loc[df.index >= start]

    close = df["close"].astype(float)
    roll_mean = close.rolling(window=20, min_periods=1).mean()
    roll_std = close.rolling(window=20, min_periods=1).std(ddof=0).replace(0, 1e-9)
    zscore = (close - roll_mean) / roll_std

    # RSI computation (standard Wilder's smoothing approximation)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(rsi_period, min_periods=1).mean().replace(0, 1e-9)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Build anomaly flags
    z_anomaly = zscore.abs() > float(z_thresh)
    rsi_anomaly = rsi > float(rsi_thresh)
    anomaly_flag = pd.Series("none", index=df.index)
    anomaly_flag.loc[z_anomaly & ~rsi_anomaly] = "zscore"
    anomaly_flag.loc[~z_anomaly & rsi_anomaly] = "rsi"
    anomaly_flag.loc[z_anomaly & rsi_anomaly] = "both"

    # Attach computed columns to df for display
    df["_zscore"] = zscore
    df["_rsi"] = rsi
    df["_anomaly"] = anomaly_flag

    # Optionally call project anomaly functions (kept for compatibility; they may update shared state)
    try:
        compute_anomalies(df)
    except Exception:
        # keep silent if compute_anomalies has different signature
        try:
            compute_anomalies()
        except Exception:
            pass
    try:
        compute_unsupervised_anomalies(df)
    except Exception:
        try:
            compute_unsupervised_anomalies()
        except Exception:
            pass

    # Top metrics
    st.metric("Latest Price", f"{close.iloc[-1]:.2f}", delta=f"{(close.pct_change().iloc[-1] * 100):+.2f}%")

    # Show interactive data table (tail) and full expandable table
    st.subheader(f"Latest data for {ticker}")
    st.dataframe(df.tail(50), use_container_width=True)
    with st.expander("Show full table"):
        st.dataframe(df, use_container_width=True)

    # Show anomaly markers in the table (tail)
    st.subheader("Anomaly markers (recent)")
    marker_table = df[[c for c in df.columns if c in ["close", "_zscore", "_rsi", "_anomaly"]]].tail(50)
    st.dataframe(marker_table, use_container_width=True)

    # Plot price with anomaly markers
    # Market regime & trend prediction
    # moving averages for regime detection
    short_w = 20
    long_w = 50
    df["_ma_short"] = close.rolling(window=short_w, min_periods=1).mean()
    df["_ma_long"] = close.rolling(window=long_w, min_periods=1).mean()
    df["_regime"] = np.where(df["_ma_short"] > df["_ma_long"], "bull", "bear")

    # simple linear trend prediction: fit last `pred_window` points and predict next value
    pred_window = int(st.sidebar.number_input("Prediction window (points)", min_value=8, max_value=200, value=24))
    def predict_next(series, window):
        y = series.dropna().values
        if y.size < 3:
            return np.nan
        y = y[-window:]
        x = np.arange(len(y))
        a, b = np.polyfit(x, y, 1)
        return a * (len(y)) + b
    df["_trend_pred_next"] = np.nan
    try:
        try:
            val = predict_next(close, pred_window)
            # assign to last index safely
            if not df.index.empty:
                df.at[df.index[-1], "_trend_pred_next"] = val
        except Exception:
            try:
                df.at[df.index[-1], "_trend_pred_next"] = np.nan
            except Exception:
                pass
    except Exception:
        try:
            df.at[df.index[-1], "_trend_pred_next"] = np.nan
        except Exception:
            pass

    # Time series plot with MAs, anomalies, regime shading, and next-step prediction
    st.subheader(f"Time series and regime for {ticker}")
    ts_fig = go.Figure()
    ts_fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close", line=dict(color="black")))
    ts_fig.add_trace(go.Scatter(x=df.index, y=df["_ma_short"], name=f"MA{short_w}", line=dict(color="blue")))
    ts_fig.add_trace(go.Scatter(x=df.index, y=df["_ma_long"], name=f"MA{long_w}", line=dict(color="orange")))
    # anomaly markers
    markers = df[df["_anomaly"] != "none"]
    if not markers.empty:
        ts_fig.add_trace(go.Scatter(x=markers.index, y=markers["close"], mode="markers",
                                    marker=dict(size=8, color=markers["_anomaly"].map({"zscore":"orange","rsi":"purple","both":"red"})),
                                    name="Anomalies"))
    # add predicted next point (draw a short dashed line from last close to predicted)
    pred_val = df["_trend_pred_next"].iloc[-1]
    last_idx = None
    next_idx = None
    if not np.isnan(pred_val):
        last_idx = df.index[-1]
        # infer freq - pd.infer_freq may return None; fall back to last delta or 1 minute
        freq = pd.infer_freq(df.index)
        if isinstance(freq, str) and freq:
            try:
                next_idx = last_idx + pd.Timedelta(freq)
            except Exception:
                next_idx = last_idx + (df.index[-1] - df.index[-2]) if len(df.index) > 1 else last_idx + pd.Timedelta(minutes=1)
        else:
            next_idx = last_idx + (df.index[-1] - df.index[-2]) if len(df.index) > 1 else last_idx + pd.Timedelta(minutes=1)
        if next_idx is not None:
            ts_fig.add_trace(go.Scatter(x=[last_idx, next_idx], y=[df["close"].iloc[-1], pred_val],
                                        name="Trend prediction", mode="lines+markers",
                                        line=dict(dash="dash", color="green")))
    # add regime background shading: find contiguous segments
    regimes = df["_regime"].fillna(method="ffill")
    seg_start = None
    current = None
    for idx, val in regimes.items():
        if current is None:
            current = val
            seg_start = idx
            continue
        if val != current:
            ts_fig.add_vrect(x0=seg_start, x1=idx, fillcolor="rgba(0,255,0,0.06)" if current == "bull" else "rgba(255,0,0,0.04)",
                             line_width=0)
            seg_start = idx
            current = val
    # last segment
    if seg_start is not None:
        ts_fig.add_vrect(x0=seg_start, x1=df.index[-1], fillcolor="rgba(0,255,0,0.06)" if current == "bull" else "rgba(255,0,0,0.04)",
                         line_width=0)

    ts_fig.update_layout(legend=dict(orientation="h"), height=450, margin=dict(t=40))
    st.plotly_chart(ts_fig, use_container_width=True)

    # Candlestick chart (if OHLC present) with the same overlays
    if {"open", "high", "low", "close"}.issubset(df.columns):
        st.subheader(f"Candlestick with regime & trend for {ticker}")
        candle_fig = go.Figure(data=[go.Candlestick(x=df.index,
                                                     open=df["open"],
                                                     high=df["high"],
                                                     low=df["low"],
                                                     close=df["close"],
                                                     name="OHLC")])
        candle_fig.add_trace(go.Scatter(x=df.index, y=df["_ma_short"], name=f"MA{short_w}", line=dict(color="blue")))
        candle_fig.add_trace(go.Scatter(x=df.index, y=df["_ma_long"], name=f"MA{long_w}", line=dict(color="orange")))
        # anomalies on candlestick
        if not markers.empty:
            candle_fig.add_trace(go.Scatter(x=markers.index, y=markers["high"].fillna(markers["close"]),
                                            mode="markers", marker=dict(size=8, color=markers["_anomaly"].map({"zscore":"orange","rsi":"purple","both":"red"})),
                                            name="Anomalies"))
        # predicted point line (reuse last_idx/next_idx computed earlier if available)
        if not np.isnan(pred_val) and last_idx is not None and next_idx is not None:
            candle_fig.add_trace(go.Scatter(x=[last_idx, next_idx], y=[df["close"].iloc[-1], pred_val],
                                            name="Trend prediction", mode="lines+markers",
                                            line=dict(dash="dash", color="green")))
        # regime shading (reuse same technique)
        regimes = df["_regime"].fillna(method="ffill")
        seg_start = None
        current = None
        for idx, val in regimes.items():
            if current is None:
                current = val
                seg_start = idx
                continue
            if val != current:
                candle_fig.add_vrect(x0=seg_start, x1=idx, fillcolor="rgba(0,255,0,0.06)" if current == "bull" else "rgba(255,0,0,0.04)",
                                     line_width=0)
                seg_start = idx
                current = val
        if seg_start is not None:
            candle_fig.add_vrect(x0=seg_start, x1=df.index[-1], fillcolor="rgba(0,255,0,0.06)" if current == "bull" else "rgba(255,0,0,0.04)",
                                 line_width=0)
        candle_fig.update_layout(legend=dict(orientation="h"), height=600, margin=dict(t=40))
        st.plotly_chart(candle_fig, use_container_width=True)
    else:
        st.info("OHLC columns not present — candlestick chart unavailable.")

    # Show computed anomaly table and regime/trend cols
    st.subheader("Computed anomaly & regime indicators")
    show_cols = ["open","high","low","close", "_ma_short", "_ma_long", "_regime", "_zscore", "_rsi", "_anomaly"]
    present_cols = [c for c in show_cols if c in df.columns]
    st.write(df[present_cols].tail(50))

else:
    st.warning(f"No data available for {ticker}")

# Summary Section (e.g., news impacts, anomaly correlation)
st.header(f"Summary for {ticker}")
st.write("Summarize the stock trends, detected anomalies, and their impact here.")

# Latest news summary (uses news_ingest.news_storage)
st.subheader("Latest news summary")
news_for_t = news_ingest.news_storage.get(ticker, [])
if news_for_t:
    # show top 5 latest articles
    for nid, tck, ts, title, summary, category in sorted(news_for_t, key=lambda r: r[2], reverse=True)[:5]:
        st.markdown(f"**{title}**  \n{summary[:300]}...")
        st.caption(f"Category: {category} — {pd.to_datetime(ts, unit='s')}" )
else:
    st.info("No recent news articles found for this ticker (run news ingestion to populate).")

# Portfolio-level summary and prediction
st.header("Portfolio Summary")
portfolio = st.session_state.get("portfolio_selected") or []
if not portfolio:
    st.info("No portfolio selected. Use the sidebar to select tickers for your portfolio.")
else:
    # gather last close, regime, and anomaly for each ticker in portfolio
    rows = []
    for t in portfolio:
        # initialize defaults to keep static analyzers happy
        last_close_val = None
        regime = "unknown"
        pred = "n/a"
        last_anom = "none"
        sec = TICKER_SECTORS.get(t, "Unknown")

        try:
            df_t = data_storage.get(t) or (lambda: None)()
        except Exception:
            df_t = None
        if df_t is None or df_t.empty:
            # try DB loader
            try:
                from realtime_anomaly_project.statistical_anomaly.show_results import load_ticker_df
                df_t = load_ticker_df(t)
            except Exception:
                df_t = None
        if df_t is None or df_t.empty:
            rows.append({"ticker": t, "last_close": last_close_val, "regime": regime, "anomaly": last_anom, "prediction": pred, "sector": sec})
            continue

        # compute simple regime and trend
        close_t = pd.to_numeric(df_t["close"], errors="coerce").dropna()
        if not close_t.empty:
            last_close_val = float(close_t.iloc[-1])
        try:
            ma_short = close_t.rolling(window=20, min_periods=1).mean().iloc[-1]
            ma_long = close_t.rolling(window=50, min_periods=1).mean().iloc[-1]
            regime = "bull" if ma_short > ma_long else "bear"
        except Exception:
            regime = "unknown"

        # trend: slope of last 24 points
        a = 0.0
        if len(close_t) >= 6:
            y_series = close_t.dropna().iloc[-24:]
            y = np.asarray(y_series.values, dtype=float)
            x = np.arange(len(y))
            if len(y) >= 2:
                try:
                    a = float(np.polyfit(x, y, 1)[0])
                except Exception:
                    a = 0.0

        # simple rule-based prediction
        if regime == "bull" and a > 0:
            pred = "buy"
        elif regime == "bear" and a < 0:
            pred = "short"
        elif abs(a) < 1e-6:
            pred = "hold"
        else:
            pred = "hold"

        # anomaly flag: check last value
        if "_anomaly" in df_t.columns and not df_t["_anomaly"].empty:
            last_anom = df_t["_anomaly"].iloc[-1]

        rows.append({"ticker": t, "last_close": last_close_val, "regime": regime, "anomaly": last_anom, "prediction": pred, "sector": sec})

    port_df = pd.DataFrame(rows).set_index("ticker")
    st.dataframe(port_df, use_container_width=True)

    # action buttons: bulk action suggestion
    st.subheader("Portfolio Actions")
    buys = port_df[port_df["prediction"] == "buy"].index.tolist()
    shorts = port_df[port_df["prediction"] == "short"].index.tolist()
    holds = port_df[port_df["prediction"] == "hold"].index.tolist()
    st.write(f"Suggested BUY: {buys}")
    st.write(f"Suggested SHORT: {shorts}")
    st.write(f"Suggested HOLD: {holds}")

    # --- Portfolio aggregated view (multi-stock) ---
    if len(portfolio) > 0:
        st.subheader("Portfolio aggregated view")
        st.write("Create an aggregated portfolio value time series from selected tickers. Set per-ticker weights in the sidebar (default equal weight).")

        # Sidebar weight inputs for each selected ticker
        st.sidebar.markdown("---")
        st.sidebar.write("Portfolio weights (relative)")
        weights = {}
        for t in portfolio:
            key = f"weight_{t}"
            # default to 1.0 if not present — pass as value to widget and don't reassign session_state after creation
            default_w = st.session_state.get(key, 1.0)
            w = st.sidebar.number_input(f"Weight: {t}", min_value=0.0, value=float(default_w), step=0.1, key=key)
            weights[t] = float(w)

        # Build aligned close price DataFrame for portfolio tickers
        series_list = []
        series_names = []
        for t in portfolio:
            try:
                df_t = data_storage.get(t)
            except Exception:
                df_t = None
            if df_t is None or df_t.empty:
                try:
                    from realtime_anomaly_project.statistical_anomaly.show_results import load_ticker_df
                    df_t = load_ticker_df(t)
                except Exception:
                    df_t = None
            if df_t is None or df_t.empty:
                continue
            s = pd.to_numeric(df_t["close"], errors="coerce").rename(t)
            s.index = pd.to_datetime(s.index, errors="coerce")
            s = s.sort_index().ffill()
            series_list.append(s)
            series_names.append(t)

        if not series_list:
            st.info("No valid time series available for selected portfolio tickers.")
        else:
            df_port = pd.concat(series_list, axis=1)
            # forward-fill and drop rows where all NaN
            df_port = df_port.sort_index().ffill().dropna(how="all")
            # apply weights (default equal if all zeros)
            total_w = sum(weights.get(t, 1.0) for t in series_names)
            if total_w == 0:
                total_w = len(series_names)
                weights = {t: 1.0 for t in series_names}
            weight_arr = pd.Series({t: weights.get(t, 1.0) / total_w for t in series_names})
            # compute weighted portfolio value (sum of weighted closes)
            port_vals = (df_port[list(series_names)] * weight_arr).sum(axis=1)

            # aggregated anomaly counts across tickers (aligned)
            anom_frames = []
            for t in series_names:
                try:
                    df_t = data_storage.get(t)
                except Exception:
                    try:
                        from realtime_anomaly_project.statistical_anomaly.show_results import load_ticker_df as _load
                        df_t = _load(t)
                    except Exception:
                        df_t = None
                if df_t is None or df_t.empty or "_anomaly" not in df_t.columns:
                    continue
                s_an = (df_t["_anomaly"] != "none").astype(int).rename(t)
                s_an.index = pd.to_datetime(s_an.index, errors="coerce")
                anom_frames.append(s_an)
            if anom_frames:
                df_anom = pd.concat(anom_frames, axis=1).sort_index().fillna(0).astype(int)
                anom_count = df_anom.sum(axis=1)
            else:
                anom_count = pd.Series(0, index=port_vals.index)

            # Plot portfolio value and anomaly count on secondary axis
            pfig = go.Figure()
            pfig.add_trace(go.Scatter(x=port_vals.index, y=port_vals.values, name="Portfolio Value", line=dict(color="navy")))
            if anom_count is not None and not anom_count.empty:
                # align anomaly series to portfolio index
                anom_aligned = anom_count.reindex(port_vals.index).fillna(0)
                pfig.add_trace(go.Bar(x=anom_aligned.index, y=anom_aligned.values, name="Anomaly count", marker_color="red", opacity=0.4, yaxis="y2"))
                pfig.update_layout(yaxis2=dict(overlaying="y", side="right", title_text="# anomalies"))

            pfig.update_layout(title="Portfolio aggregated value and anomaly counts", legend=dict(orientation="h"), height=480)
            st.plotly_chart(pfig, use_container_width=True)

            # show a small per-ticker status table with holdings and P/L
            st.subheader("Per-ticker latest status")
            latest_rows = []
            # holdings inputs: quantity and entry price per ticker
            holdings = {}
            for t in series_names:
                q_key = f"hold_qty_{t}"
                p_key = f"hold_price_{t}"
                default_q = float(st.session_state.get(q_key, 0.0))
                default_p = float(st.session_state.get(p_key, 0.0))
                col1, col2 = st.columns([1,1])
                with col1:
                    qty = st.number_input(f"Qty {t}", min_value=0.0, value=default_q, step=1.0, key=q_key)
                with col2:
                    entry = st.number_input(f"Entry price {t}", min_value=0.0, value=default_p, step=0.01, key=p_key)
                holdings[t] = {"qty": float(qty), "entry": float(entry)}

            for t in series_names:
                last_price = float(df_port[t].dropna().iloc[-1]) if df_port[t].dropna().size else None
                pred = port_df.loc[t, "prediction"] if ("prediction" in port_df.columns and t in port_df.index) else "n/a"
                h = holdings.get(t, {"qty": 0.0, "entry": 0.0})
                qty = h.get("qty", 0.0)
                entry = h.get("entry", 0.0)
                # compute P/L: absolute and percent
                if last_price is not None and qty and entry:
                    abs_pl = (last_price - entry) * qty
                    pct_pl = (last_price - entry) / entry if entry != 0 else 0.0
                else:
                    abs_pl = None
                    pct_pl = None
                latest_rows.append({"ticker": t, "last_price": last_price, "prediction": pred, "weight": weights.get(t, 0.0), "sector": TICKER_SECTORS.get(t, "Unknown"), "qty": qty, "entry": entry, "abs_pl": abs_pl, "pct_pl": pct_pl})
            latest_df = pd.DataFrame(latest_rows).set_index("ticker")

            # add color-coded badge column based on anomaly or sentiment thresholds
            def badge_for_row(t):
                # priority: anomaly (red if both), sentiment if available, else green
                try:
                    df_t = data_storage.get(t) or (lambda: None)()
                except Exception:
                    df_t = None
                an = None
                if isinstance(df_t, pd.DataFrame) and '_anomaly' in df_t.columns and not df_t.empty:
                    an = df_t['_anomaly'].iloc[-1]
                # sentiment check from news storage: look for latest sentiment score
                sent_score = None
                news_list = news_ingest.news_storage.get(t, [])
                if news_list:
                    # news entries: (nid, tck, ts, title, summary, category, sentiment_label, sentiment_score)
                    last = news_list[-1]
                    if len(last) >= 8:
                        sent_score = last[7]
                # decide badge
                if an == 'both' or an == 'zscore' or an == 'rsi' and an is not None:
                    # if both or specific anomalies, amber/red depending on severity
                    if an == 'both':
                        return 'red'
                    return 'orange'
                if sent_score is not None:
                    try:
                        if float(sent_score) > 0.6:
                            return 'green'
                        if float(sent_score) < -0.4:
                            return 'red'
                        return 'orange'
                    except Exception:
                        return 'grey'
                return 'green'

            latest_df['badge'] = [badge_for_row(t) for t in latest_df.index]
            st.dataframe(latest_df, use_container_width=True)

            # compute real portfolio P/L using holdings quantities
            total_abs_pl = 0.0
            total_cost = 0.0
            for t, h in holdings.items():
                qty = h.get('qty', 0.0)
                entry = h.get('entry', 0.0)
                last_price = float(df_port[t].dropna().iloc[-1]) if df_port[t].dropna().size else None
                if qty and entry and last_price is not None:
                    total_abs_pl += (last_price - entry) * qty
                    total_cost += entry * qty
            portfolio_pl_pct = (total_abs_pl / total_cost) if total_cost else None
            st.metric("Portfolio P/L (abs)", f"{total_abs_pl:.2f}" if total_cost else "n/a", delta=f"{(portfolio_pl_pct*100):+.2f}%" if portfolio_pl_pct is not None else "n/a")

            # grouped summary by sector
            st.subheader("Portfolio by sector")
            sector_counts = latest_df.groupby("sector").size().rename("count").to_frame()
            st.table(sector_counts)

            # Sector aggregated metrics: average anomaly (simple count of anomaly flags) and sector P/L
            st.subheader("Sector aggregated metrics")
            sector_metrics = []
            # To compute sector P/L aligned to portfolio weights and comparable periods, resample each ticker series to daily and compute pct returns, then weight by current portfolio weights.
            for sector, group in latest_df.reset_index().groupby('sector'):
                tickers_in_sector = group['ticker'].tolist()
                anom_props = []
                weighted_returns = []
                weight_sum = 0.0
                # prepare a DataFrame of daily pct_returns per ticker
                per_ticker_returns = {}
                for t in tickers_in_sector:
                    try:
                        df_t = data_storage.get(t)
                    except Exception:
                        df_t = None
                    if df_t is None or df_t.empty:
                        try:
                            from realtime_anomaly_project.statistical_anomaly.show_results import load_ticker_df as _load
                            df_t = _load(t)
                        except Exception:
                            df_t = None
                    if df_t is None or df_t.empty:
                        continue
                    if '_anomaly' in df_t.columns:
                        anom_props.append((df_t['_anomaly'] != 'none').mean())
                    s = pd.to_numeric(df_t['close'], errors='coerce').rename(t)
                    s.index = pd.to_datetime(s.index, errors='coerce')
                    # resample to daily close and compute pct change
                    try:
                        daily = s.sort_index().resample('1D').last().ffill()
                        pct = daily.pct_change().fillna(0)
                        per_ticker_returns[t] = pct
                    except Exception:
                        continue
                if not per_ticker_returns:
                    sector_metrics.append({'sector': sector, 'avg_anomaly': 0.0, 'avg_pl': 0.0, 'n': len(tickers_in_sector)})
                    continue
                # align by index
                returns_df = pd.concat(per_ticker_returns.values(), axis=1, keys=per_ticker_returns.keys()).fillna(0)
                # apply portfolio weights normalized across the sector tickers
                sector_weights = {t: weights.get(t, 0.0) for t in returns_df.columns}
                wsum = sum(sector_weights.values())
                if wsum == 0:
                    # equal weight across tickers
                    sector_wseries = pd.Series({t: 1.0/len(returns_df.columns) for t in returns_df.columns})
                else:
                    sector_wseries = pd.Series({t: sector_weights.get(t, 0.0)/wsum for t in returns_df.columns})
                # compute weighted daily returns and cumulative return
                weighted_daily = (returns_df * sector_wseries).sum(axis=1)
                cumulative = (1 + weighted_daily).cumprod()
                sector_pl = cumulative.iloc[-1] - 1.0 if not cumulative.empty else 0.0
                avg_anom = float(np.mean(anom_props)) if anom_props else 0.0
                sector_metrics.append({'sector': sector, 'avg_anomaly': avg_anom, 'avg_pl': float(sector_pl), 'n': len(tickers_in_sector)})

            sm_df = pd.DataFrame(sector_metrics).set_index('sector')
            if not sm_df.empty:
                st.table(sm_df)
                # bar chart for avg anomaly and avg pl
                fig_sec = go.Figure()
                fig_sec.add_trace(go.Bar(x=sm_df.index, y=sm_df['avg_anomaly'], name='Avg anomaly', marker_color='red'))
                fig_sec.add_trace(go.Bar(x=sm_df.index, y=sm_df['avg_pl'], name='Avg P/L', marker_color='green', yaxis='y2'))
                fig_sec.update_layout(yaxis2=dict(overlaying='y', side='right', title_text='Avg P/L'))
                fig_sec.update_layout(title='Sector metrics', barmode='group', height=400)
                st.plotly_chart(fig_sec, use_container_width=True)

            # Button: Show full data for each ticker in portfolio
            if st.button("Show data for all portfolio tickers"):
                for t in series_names:
                    st.subheader(f"Recent data for {t}")
                    try:
                        df_t = data_storage.get(t)
                    except Exception:
                        df_t = None
                    if df_t is None or df_t.empty:
                        try:
                            from realtime_anomaly_project.statistical_anomaly.show_results import load_ticker_df as _load
                            df_t = _load(t)
                        except Exception:
                            df_t = None
                    if df_t is None or df_t.empty:
                        st.info(f"No data available for {t}")
                        continue
                    st.dataframe(df_t.tail(50), use_container_width=True)
                    with st.expander(f"Show full table for {t}"):
                        st.dataframe(df_t, use_container_width=True)

            # Button: Fetch latest news for portfolio tickers
            if st.button("Fetch latest news for portfolio (background)"):
                tickers_to_fetch = series_names
                # start background fetch and show progress
                news_ingest.start_background_fetch(tickers_to_fetch)
                st.sidebar.info("Started background news fetch. Progress will appear below.")

            # show background news fetch progress
            if news_ingest.news_fetch_progress.get("running"):
                total = news_ingest.news_fetch_progress.get("total", 0)
                fetched = news_ingest.news_fetch_progress.get("fetched", 0)
                p = int((fetched / total) * 100) if total else 0
                st.sidebar.write(f"News fetch progress: {fetched}/{total} ({p}%)")
                st.sidebar.progress(p)
            elif news_ingest.news_fetch_progress.get("total", 0) > 0:
                # show final status
                fetched = news_ingest.news_fetch_progress.get("fetched", 0)
                failed = news_ingest.news_fetch_progress.get("failed", [])
                st.sidebar.success(f"News fetch finished: {fetched} fetched")
                if failed:
                    st.sidebar.warning(f"Failed: {len(failed)} tickers")
