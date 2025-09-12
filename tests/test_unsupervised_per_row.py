import numpy as np
import pandas as pd

from realtime_anomaly_project.statistical_anomaly.unsupervised import compute_unsupervised_anomalies


def make_synthetic_ts(n=200, seed=0):
    rng = np.random.RandomState(seed)
    t = pd.date_range('2020-01-01', periods=n, freq='H')
    # baseline + occasional spikes
    vals = np.cumsum(rng.normal(scale=0.1, size=n))
    spikes = (rng.rand(n) < 0.02) * rng.normal(loc=3.0, scale=1.0, size=n)
    vals = vals + spikes
    df = pd.DataFrame({'close': vals}, index=t)
    return df


def test_compute_unsupervised_per_row():
    df = make_synthetic_ts()
    res = compute_unsupervised_anomalies(df, per_row=True, unsup_threshold=1.0)

    # results should include the synthetic ticker key
    assert isinstance(res, dict)
    assert '_local' in res

    # The dataframe should now have per-row score columns
    assert '_unsup_iso' in df.columns or '_unsup_lof' in df.columns
    # _unsup column should exist
    assert '_unsup' in df.columns

    # summary keys present
    s = res['_local']
    assert 'isolation_forest' in s
    assert 'local_outlier_factor' in s
