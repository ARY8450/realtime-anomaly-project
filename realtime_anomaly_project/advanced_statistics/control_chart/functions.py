"""Control chart statistics and change detectors.

Implemented:
 - cusum_statistic: one-sided CUSUM (positive and negative)

Each function accepts a pandas Series and window/period parameters.

"""
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


def cusum_statistic(series: pd.Series, target: Optional[float] = None, k: float = 0.5, h: float = 5.0) -> pd.DataFrame:
    """Compute one-sided CUSUM statistics (positive and negative) for a series.

    Parameters
    ----------
    series : pd.Series
        Input time series (numeric). Index will be preserved.
    target : float, optional
        Reference value (mean); if None, use series.mean().
    k : float
        Slack/allowance parameter (often half the shift magnitude to detect).
    h : float
        Decision threshold. When C+ or C- exceed h, a shift is signaled.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['cusum_pos', 'cusum_neg', 'signal_pos', 'signal_neg']

    Example
    -------
    >>> df = cusum_statistic(series, k=0.5, h=4)

    """
    if target is None:
        target = float(series.mean())

    s = series.ffill().bfill().astype(float)
    cpos = np.zeros(len(s), dtype=float)
    cneg = np.zeros(len(s), dtype=float)

    for i in range(1, len(s)):
        diff = s.iloc[i] - target - k
        cpos[i] = max(0.0, cpos[i - 1] + diff)
        diffn = target - s.iloc[i] - k
        cneg[i] = max(0.0, cneg[i - 1] + diffn)

    df = pd.DataFrame(index=s.index)
    df['cusum_pos'] = cpos
    df['cusum_neg'] = cneg
    df['signal_pos'] = df['cusum_pos'] > h
    df['signal_neg'] = df['cusum_neg'] > h
    return df


def page_hinkley(series: pd.Series, delta: float = 0.005, threshold: float = 5.0, alpha: float = 1.0) -> pd.DataFrame:
    """Page–Hinkley test statistic (cumulative difference) to detect mean shifts.

    Parameters
    ----------
    series : pd.Series
    delta : float
        Small positive constant to control sensitivity.
    threshold : float
        Threshold to trigger a change detection.
    alpha : float
        Weight for cumulative mean (1 = standard PH)

    Returns
    -------
    pd.DataFrame with columns ['ph', 'alarm']
    """
    x = series.ffill().bfill().astype(float)
    m = 0.0
    ph = np.zeros(len(x), dtype=float)
    min_ph = 0.0
    alarms = np.zeros(len(x), dtype=bool)
    for i, xi in enumerate(x):
        m = m + (xi - m) / (i + 1) * alpha
        ph[i] = ph[i - 1] + (xi - m - delta) if i > 0 else max(0.0, xi - m - delta)
        if ph[i] < min_ph:
            min_ph = ph[i]
        alarms[i] = (ph[i] - min_ph) > threshold
    return pd.DataFrame({'ph': ph, 'alarm': alarms}, index=x.index)


def ewma_control_distance(series: pd.Series, span: int = 20, nsigma: float = 3.0) -> pd.Series:
    """Distance of each point from EWMA control limits in units of EWMA std.

    Returns a Series where values > nsigma indicate out-of-control points.
    """
    x = series.astype(float)
    mu = x.ewm(span=span, adjust=False).mean()
    sigma = (x - mu).ewm(span=span, adjust=False).std()
    upper = mu + nsigma * sigma
    lower = mu - nsigma * sigma
    dist = pd.Series(index=x.index, dtype=float)
    dist[:] = 0.0
    dist[x > upper] = (x - upper)[x > upper] / sigma[x > upper]
    dist[x < lower] = (lower - x)[x < lower] / sigma[x < lower]
    return dist


def rolling_variance_shift(series: pd.Series, window: int = 30, lookback: int = 30) -> pd.Series:
    """Detect shifts in variance by comparing current rolling variance to previous window variance.

    Returns ratio of current var / previous_var (values >>1 indicate increase).
    """
    cur_var = series.rolling(window=window, min_periods=3).var()
    prev_var = series.shift(window).rolling(window=window, min_periods=3).var()
    return cur_var / prev_var.replace(0, np.nan)


def levene_proxy(series: pd.Series, window: int = 60, split: float = 0.5) -> pd.Series:
    """Proxy for Levene/Brown–Forsythe: compute absolute-deviation test statistic over rolling window.

    Splits the window into two groups at fraction `split` and returns p-value of Levene test.
    """
    pvals = pd.Series(index=series.index, dtype=float)
    for i in range(len(series)):
        window_slice = series.iloc[max(0, i - window + 1):i + 1]
        if len(window_slice) < 6:
            pvals.iloc[i] = np.nan
            continue
        n = len(window_slice)
        split_idx = int(n * split)
        g1 = window_slice.iloc[:split_idx]
        g2 = window_slice.iloc[split_idx:]
        try:
            stat, p = stats.levene(g1, g2, center='median')
        except Exception:
            p = np.nan
        pvals.iloc[i] = p
    return pvals
