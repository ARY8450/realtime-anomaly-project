import pandas as pd
import numpy as np

# deterministic
np.random.seed(0)


def test_smoke_basic():
    # quick smoke test for importable modules and RSI behavior
    from realtime_anomaly_project.advanced_statistics.technical_indicators import functions as ti

    rng = pd.date_range("2020-01-01", periods=50, freq="D")
    prices = pd.Series(np.linspace(100, 110, 50) + np.random.randn(50) * 0.5, index=rng)
    r = ti.rsi(prices)
    assert isinstance(r, pd.Series)
    # RSI should be between 0 and 100 for valid values
    vals = r.dropna()
    if not vals.empty:
        assert vals.between(0, 100).all()
