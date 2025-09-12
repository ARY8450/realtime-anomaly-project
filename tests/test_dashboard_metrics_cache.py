import time
import pandas as pd
import numpy as np
import streamlit as st

from realtime_anomaly_project.app.dashboard import compute_metrics_for_df


def _make_minimal_df():
    idx = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
    df = pd.DataFrame({'close': np.linspace(100, 110, len(idx)),
                       'open': np.linspace(99, 109, len(idx)),
                       'high': np.linspace(101, 111, len(idx)),
                       'low': np.linspace(98, 108, len(idx)),
                       'volume': np.random.randint(1000, 2000, len(idx))},
                      index=idx)
    return df


def test_compute_metrics_returns_expected_keys():
    df = _make_minimal_df()
    out = compute_metrics_for_df(df, rsi_period_local=14)
    assert isinstance(out, dict)
    # required keys
    for k in ('metrics', 'summary_df', 'det_series', 'seasonal_viz', 'period'):
        assert k in out
    assert 'metrics' in out and isinstance(out['metrics'], dict)
    assert 'summary_df' in out and (isinstance(out['summary_df'], pd.DataFrame) or out['summary_df'] is None)


def test_metrics_cache_and_recompute(monkeypatch):
    # use a synthetic ticker name
    ticker = 'TESTTICKER'
    df = _make_minimal_df()
    # ensure session_state keys exist
    if 'metrics_cache' not in st.session_state:
        st.session_state['metrics_cache'] = {}
    if 'metrics_cache_meta' not in st.session_state:
        st.session_state['metrics_cache_meta'] = {}

    # compute and store
    st.session_state['metrics_cache'][ticker] = compute_metrics_for_df(df, rsi_period_local=14)
    st.session_state['metrics_cache_meta'][ticker] = time.time()
    ts1 = st.session_state['metrics_cache_meta'][ticker]
    assert ts1 > 0

    # recompute after sleeping to ensure timestamp change
    time.sleep(0.1)
    st.session_state['metrics_cache'][ticker] = compute_metrics_for_df(df * 1.01, rsi_period_local=14)
    st.session_state['metrics_cache_meta'][ticker] = time.time()
    ts2 = st.session_state['metrics_cache_meta'][ticker]
    assert ts2 >= ts1
