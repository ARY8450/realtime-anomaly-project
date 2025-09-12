import pandas as pd
import numpy as np

# make synthetic tests deterministic
np.random.seed(0)
from pathlib import Path


def test_technical_indicators_synthetic():
    from realtime_anomaly_project.advanced_statistics.technical_indicators import functions as ti

    rng = pd.date_range("2021-01-01", periods=200, freq="D")
    prices = pd.Series(np.cumsum(np.random.randn(200) * 0.5) + 100, index=rng)
    high = prices + np.random.rand(200) * 0.5
    low = prices - np.random.rand(200) * 0.5
    macd = ti.macd(prices)
    boll = ti.bollinger(prices)
    at = ti.atr(high, low, prices)
    skd = ti.stochastic_kd(high, low, prices)
    fish = ti.fisher_transform(prices)

    assert isinstance(macd, pd.DataFrame)
    assert 'hist' in macd.columns
    assert isinstance(boll, pd.DataFrame)
    assert 'percent_b' in boll.columns
    assert isinstance(at, pd.Series)
    assert isinstance(skd, pd.DataFrame)
    assert isinstance(fish, pd.Series)


def test_model_residuals_synthetic():
    from realtime_anomaly_project.advanced_statistics.model_residuals import functions as mr

    rng = pd.date_range("2021-01-01", periods=200, freq="D")
    df = pd.DataFrame({
        'a': np.random.randn(200),
        'b': np.random.randn(200) * 0.5 + 0.2,
    }, index=rng)
    md = mr.mahalanobis_distance(df)
    iso = mr.isolation_forest_score(df)
    lof = mr.lof_score(df)
    assert isinstance(md, pd.Series)
    assert isinstance(iso, pd.Series)
    assert isinstance(lof, pd.Series)

    # pinball loss
    y = pd.Series(np.random.randn(100))
    q = y.copy() + np.random.randn(100) * 0.1
    loss = mr.pinball_loss(y, q, q=0.5)
    assert isinstance(loss, float)

    # optional: if cached API CSV exists, run functions on that
    cached = Path('realtime_anomaly_project/sql_db/realtime_live_cache.json')
    if cached.exists():
        # try to load and apply basic functions (best-effort)
        try:
            import json
            from realtime_anomaly_project.advanced_statistics.technical_indicators import functions as ti
            j = json.loads(cached.read_text())
            # pick first ticker timeseries
            first = next(iter(j.values()))
            # expect OHLC keys
            if isinstance(first, dict) and 'close' in first:
                close = pd.Series(first['close'])
                _ = ti.rsi(close)
        except Exception:
            pass
