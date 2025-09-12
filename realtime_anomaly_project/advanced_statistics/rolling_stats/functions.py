"""Rolling statistics and simple anomaly scores.

Implemented functions:
  - z_score
  - robust_z_score (MAD)
  - ewma_z_score
  - iqr_tukey_fence_score
  - winsorized_z
  - rolling_skew_z
  - rolling_kurtosis_z
  - level_trend_accel
  - roc (rate of change)
  - drawdown_depth

Each function accepts a pandas Series and window/period arguments where relevant.
"""
from typing import Tuple
import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import median_abs_deviation as _scipy_mad


def z_score(series: pd.Series, window: int = 30, min_periods: Optional[int] = None) -> pd.Series:
    """Rolling z-score: (x - rolling_mean)/rolling_std.

    Parameters
    ----------
    series : pd.Series
    window : int
    min_periods : int

    Returns
    -------
    pd.Series

    Example:
        >>> zs = z_score(close, window=30)
    """
    if min_periods is None:
        min_periods = max(3, window // 3)
    roll_mean = series.rolling(window=window, min_periods=min_periods).mean()
    roll_std = series.rolling(window=window, min_periods=min_periods).std()
    return (series - roll_mean) / roll_std


def robust_z_score(series: pd.Series, window: int = 30) -> pd.Series:
    """Robust z-score using MAD: (x - median)/ (1.4826*MAD).

    Uses rolling median and rolling MAD approximation.
    """
    def _mad(x: pd.Series) -> float:
        # prefer numpy-based MAD for typing compatibility; fall back to scipy if present
        arr = x.dropna().astype(float).to_numpy()
        if arr is None or len(arr) == 0:
            return float('nan')
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        # scale to be comparable with std for normal dist
        return float(mad * 1.4826)

    roll_med = series.rolling(window=window, min_periods=3).median()
    roll_mad = series.rolling(window=window, min_periods=3).apply(_mad, raw=False)
    return (series - roll_med) / roll_mad


def ewma_z_score(series: pd.Series, span: int = 20) -> pd.Series:
    """EWMA z-score: (x - ewma_mean)/ewm_std using exponential weighting.

    Example:
        >>> ez = ewma_z_score(close, span=20)
    """
    x = series.astype(float)
    ewma = x.ewm(span=span, adjust=False).mean()
    # compute EWM variance via (E[x^2]-E[x]^2)
    ewma2 = (x ** 2).ewm(span=span, adjust=False).mean()
    ew_var = ewma2 - ewma ** 2
    ew_std = ew_var.clip(lower=1e-12) ** 0.5
    return (x - ewma) / ew_std


def iqr_tukey_fence_score(series: pd.Series, window: int = 30) -> pd.Series:
    """Compute distance from Tukey fences normalized by IQR.

    Score positive when above upper fence, negative when below lower fence.
    """
    q1 = series.rolling(window=window, min_periods=3).quantile(0.25)
    q3 = series.rolling(window=window, min_periods=3).quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr
    score = pd.Series(index=series.index, dtype=float)
    score[:] = 0.0
    score[series > upper] = (series - upper)[series > upper] / iqr[series > upper]
    score[series < lower] = (series - lower)[series < lower] / iqr[series < lower]
    return score


def winsorized_z(series: pd.Series, window: int = 30, limits: Tuple[float, float] = (0.05, 0.95)) -> pd.Series:
    """Compute winsorized z: winsorize values then compute z-score.

    limits are the lower and upper quantiles to clamp to.
    """
    loq = series.rolling(window=window, min_periods=3).quantile(limits[0])
    hiq = series.rolling(window=window, min_periods=3).quantile(limits[1])
    def _wins(x, lo, hi):
        return np.clip(x, lo, hi)

    out = pd.Series(index=series.index, dtype=float)
    for idx in series.index:
        window_slice = series.loc[:idx].tail(window)
        if len(window_slice) < 3:
            out.at[idx] = np.nan
            continue
        lo = window_slice.quantile(limits[0])
        hi = window_slice.quantile(limits[1])
        w = np.clip(series.at[idx], lo, hi)
        m = window_slice.mean()
        s = window_slice.std()
        out.at[idx] = (w - m) / (s if s > 0 else np.nan)
    return out


def rolling_skew_z(series: pd.Series, window: int = 30) -> pd.Series:
    """Rolling skewness normalized z (skew / rolling skew std).
    """
    roll_skew = series.rolling(window=window, min_periods=3).skew()
    mu = roll_skew.rolling(window=window, min_periods=3).mean()
    sigma = roll_skew.rolling(window=window, min_periods=3).std()
    return (roll_skew - mu) / sigma


def rolling_kurtosis_z(series: pd.Series, window: int = 30) -> pd.Series:
    """Rolling kurtosis normalized z.
    """
    roll_kurt = series.rolling(window=window, min_periods=3).kurt()
    mu = roll_kurt.rolling(window=window, min_periods=3).mean()
    sigma = roll_kurt.rolling(window=window, min_periods=3).std()
    return (roll_kurt - mu) / sigma


def level_trend_accel(series: pd.Series, window: int = 30) -> pd.DataFrame:
    """Estimate level, trend (slope), and acceleration (second derivative) using rolling linear fits.

    Returns DataFrame with columns ['level','trend','accel'] aligned to input index.
    """
    idx = series.index
    lev = pd.Series(index=idx, dtype=float)
    trend = pd.Series(index=idx, dtype=float)
    accel = pd.Series(index=idx, dtype=float)
    for i in range(len(series)):
        window_slice = series.iloc[max(0, i - window + 1):i + 1]
        if len(window_slice) < 3:
            lev.iloc[i] = np.nan
            trend.iloc[i] = np.nan
            accel.iloc[i] = np.nan
            continue
        x = np.arange(len(window_slice)).astype(float)
        y = window_slice.values.astype(float)
        # fit quadratic y = a + b*x + c*x^2
        coeffs = np.polyfit(x, y, 2)
        a, b, c = coeffs[-1], coeffs[-2], coeffs[-3]
        lev.iloc[i] = a
        trend.iloc[i] = b
        accel.iloc[i] = 2 * c
    return pd.DataFrame({'level': lev, 'trend': trend, 'accel': accel})


def roc(series: pd.Series, period: int = 1) -> pd.Series:
    """Rate of change: (x_t - x_{t-period}) / x_{t-period}.
    """
    return series.pct_change(periods=period)


def drawdown_depth(series: pd.Series, window: Optional[int] = None) -> pd.Series:
    """Compute drawdown depth relative to rolling (or global) maximum.

    If window is None use global maximum up to each point.
    """
    if window is None:
        cummax = series.cummax()
    else:
        cummax = series.rolling(window=window, min_periods=1).max()
    dd = (series - cummax) / cummax
    return dd
