"""Seasonality and autocorrelation based statistics.

Implemented:
 - acf_spike_score: measure prominent spikes in ACF indicating periodicity

"""
from typing import Callable, Optional, Any, cast, TYPE_CHECKING
import math
import numpy as np
import pandas as pd

# Lazy-imported handles for statsmodels functions/classes
_acf: Optional[Callable] = None
_pacf: Optional[Callable] = None
_STL: Optional[Callable] = None


def _lazy_import_statsmodels() -> None:
    """Import statsmodels components on first use. Raises informative ImportError if missing."""
    global _acf, _pacf, _STL
    if _acf is not None and _pacf is not None and _STL is not None:
        return
    try:
        # import here to keep statsmodels an optional runtime dependency
        from statsmodels.tsa.stattools import acf as _acf_local, pacf as _pacf_local  # type: ignore[import]
        from statsmodels.tsa.seasonal import STL as _STL_local  # type: ignore[import]
    except Exception as e:
        raise ImportError("statsmodels is required for seasonality_autocorr functions; install it to use these features") from e
    _acf = _acf_local
    _pacf = _pacf_local
    _STL = _STL_local


def acf_spike_score(series: pd.Series, nlags: int = 40) -> pd.Series:
    """Compute normalized absolute ACF values for lags 1..nlags.

    Returns
    -------
    pd.Series
        Absolute ACF values indexed by lag (1..nlags).
    """
    s = series.dropna().astype(float)
    if len(s) < 3:
        return pd.Series(dtype=float)
    _lazy_import_statsmodels()
    # statsmodels acf accepts array-like
    acf_fn = cast(Callable[..., Any], _acf)
    ac = acf_fn(s.to_numpy(), nlags=nlags, fft=True, missing='conservative')
    spikes = np.abs(ac[1:])
    idx = list(range(1, len(spikes) + 1))
    return pd.Series(spikes, index=idx)


def pacf_spike_score(series: pd.Series, nlags: int = 40) -> pd.Series:
    """Compute normalized absolute PACF values for lags 1..nlags.
    """
    s = series.dropna().astype(float)
    if len(s) < 3:
        return pd.Series(dtype=float)
    _lazy_import_statsmodels()
    pacf_fn = cast(Callable[..., Any], _pacf)
    pk = pacf_fn(s.to_numpy(), nlags=nlags, method='ywunbiased')
    spikes = np.abs(pk[1:])
    idx = list(range(1, len(spikes) + 1))
    return pd.Series(spikes, index=idx)


def stl_seasonal_strength(series: pd.Series, period: int) -> float:
    """Estimate seasonal strength using STL decomposition.

    Returns fraction of variance explained by the seasonal component:
        var(seasonal) / (var(seasonal) + var(resid)).
    """
    s = series.dropna().astype(float)
    if len(s) < max(3 * period, 10):
        return float('nan')
    _lazy_import_statsmodels()
    STL_cls = cast(Callable[..., Any], _STL)
    stl = STL_cls(s.to_numpy(), period=period, robust=True)
    res = stl.fit()
    var_s = float(np.nanvar(res.seasonal))
    var_resid = float(np.nanvar(res.resid))
    return float(var_s / (var_s + var_resid)) if (var_s + var_resid) > 0 else float('nan')


def seasonal_mismatch_residual(series: pd.Series, period: int) -> float:
    """Measure how consistent the seasonal pattern is across cycles.

    Computes the STL seasonal component, then computes the mean seasonal profile
    for each phase in the period and reconstructs a repeating seasonal pattern
    from that mean profile. The metric is the mean absolute deviation between
    the STL seasonal component and the reconstructed repeating profile,
    normalized by the seasonal component's standard deviation to be unitless.
    """
    s = series.dropna().astype(float)
    if len(s) < max(3 * period, 10):
        return float('nan')
    _lazy_import_statsmodels()
    STL_cls = cast(Callable[..., Any], _STL)
    stl = STL_cls(s.to_numpy(), period=period, robust=True)
    res = stl.fit()
    seasonal = np.asarray(res.seasonal)
    # build mean profile across cycles
    n = len(seasonal)
    full_cycles = n // period
    if full_cycles < 2:
        return float('nan')
    # truncate to full cycles for easy reshaping
    truncated = seasonal[: full_cycles * period]
    mat = truncated.reshape((full_cycles, period))
    mean_profile = np.nanmean(mat, axis=0)
    # reconstruct repeating profile
    recon = np.tile(mean_profile, full_cycles)
    # compare only the truncated part
    resid = truncated - recon
    mean_abs = float(np.nanmean(np.abs(resid)))
    denom = float(np.nanstd(seasonal))
    return mean_abs / (denom + 1e-12)


def seasonal_peak_timing_error(series: pd.Series, period: int) -> float:
    """Compute the average peak timing error (in fraction of period).

    For each full cycle, find the index within the period where the seasonal
    component attains its maximum (peak). Compute the circular absolute
    deviation of those peak positions around their median, then normalize by
    the period to return a fraction (0..0.5 typically).
    """
    s = series.dropna().astype(float)
    if len(s) < max(3 * period, 10):
        return float('nan')
    _lazy_import_statsmodels()
    STL_cls = cast(Callable[..., Any], _STL)
    stl = STL_cls(s.to_numpy(), period=period, robust=True)
    res = stl.fit()
    seasonal = np.asarray(res.seasonal)
    n = len(seasonal)
    full_cycles = n // period
    if full_cycles < 2:
        return float('nan')
    truncated = seasonal[: full_cycles * period]
    mat = truncated.reshape((full_cycles, period))
    peaks = np.nanargmax(mat, axis=1)
    # median peak position
    median_peak = int(np.nanmedian(peaks))
    # compute circular distance
    diffs = np.abs(peaks - median_peak)
    circ = np.minimum(diffs, period - diffs)
    mean_circ = float(np.nanmean(circ))
    return mean_circ / float(period)
