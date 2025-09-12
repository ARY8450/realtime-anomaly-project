"""Frequency-domain and complexity measures.

Implemented:
 - spectral_entropy: normalized Shannon entropy of power spectral density

"""
import math
from typing import Optional
import numpy as np
import pandas as pd
from scipy.signal import welch

def spectral_entropy(series: pd.Series, fs: float = 1.0, nperseg: Optional[int] = None) -> float:
    """Compute spectral entropy of a time series using Welch PSD.

    Parameters
    ----------
    series : pd.Series
        Input signal.
    fs : float
        Sampling frequency (default 1.0)
    nperseg : int, optional
        Length of each segment for Welch; default None lets scipy choose.

    Returns
    -------
    float
        Normalized spectral entropy between 0 and 1.

    Example
    -------
    >>> se = spectral_entropy(series, fs=1.0)

    """
    x = series.dropna().astype(float).to_numpy()
    if x is None or len(x) == 0:
        return float('nan')
    f, Pxx = welch(x, fs=fs, nperseg=nperseg)
    P = Pxx / np.sum(Pxx)
    # small eps to avoid log(0)
    eps = 1e-12
    H = -np.sum(P * np.log2(P + eps))
    Hmax = np.log2(len(P))
    return float(H / Hmax) if Hmax > 0 else 0.0

def band_power_ratios(series: pd.Series, bands=None, fs: float = 1.0, nperseg: Optional[int] = None) -> dict:
    """Compute relative band power ratios for specified frequency bands.

    Parameters
    ----------
    series : pd.Series
    bands : list of (low, high) tuples (Hz)
    fs : sampling frequency

    Returns
    -------
    dict mapping band tuple -> fraction of total power
    """
    if bands is None:
        bands = [(0.0, 0.1 * fs), (0.1 * fs, 0.3 * fs), (0.3 * fs, 0.5 * fs)]
    x = series.dropna().astype(float).to_numpy()
    if x is None or len(x) == 0:
        return {b: float('nan') for b in bands}
    f, Pxx = welch(x, fs=fs, nperseg=nperseg)
    total = float(np.trapz(Pxx, f))
    out = {}
    for (lo, hi) in bands:
        mask = (f >= lo) & (f <= hi)
        p = float(np.trapz(Pxx[mask], f[mask])) if np.any(mask) else 0.0
        out[(lo, hi)] = float(p / total) if total > 0 else float('nan')
    return out

def hjorth_params(series: pd.Series) -> dict:
    """Return Hjorth activity, mobility, complexity for a signal.

    activity = var(x)
    mobility = sqrt(var(dx)/var(x))
    complexity = mobility(dx)/mobility(x)
    """
    x = series.dropna().astype(float).to_numpy()
    if x is None or len(x) < 3:
        return {'activity': float('nan'), 'mobility': float('nan'), 'complexity': float('nan')}
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = np.var(x, ddof=1)
    var_dx = np.var(dx, ddof=1)
    var_ddx = np.var(ddx, ddof=1) if len(ddx) > 0 else 0.0
    activity = float(var_x)
    mobility = float(np.sqrt(var_dx / var_x)) if var_x > 0 else float('nan')
    mobility_dx = float(np.sqrt(var_ddx / var_dx)) if var_dx > 0 else float('nan')
    complexity = float(mobility_dx / mobility) if mobility and not np.isnan(mobility) else float('nan')
    return {'activity': activity, 'mobility': mobility, 'complexity': complexity}


def hurst_exponent(series: pd.Series, min_window: int = 10, max_window: int = 100) -> float:
    """Estimate Hurst exponent using aggregated variance method.

    Returns H where 0.5 ~ random walk, >0.5 persistent.
    """
    x = series.dropna().astype(float).to_numpy()
    n = 0 if x is None else len(x)
    if n < min_window:
        return float('nan')
    ws = np.unique(np.floor(np.logspace(np.log10(min_window), np.log10(min(max_window, n//2)), num=10)).astype(int))
    vars_ = []
    for w in ws:
        if w < 2:
            continue
        # aggregate by taking means of blocks of size w
        k = n // w
        if k < 2:
            continue
        agg = np.array([x[i * w:(i + 1) * w].mean() for i in range(k)])
        vars_.append(np.var(agg, ddof=1))
    if len(vars_) < 2:
        return float('nan')
    coeffs = np.polyfit(np.log(ws[:len(vars_)]), np.log(vars_), 1)
    slope = coeffs[0]
    H = -slope / 2.0
    return float(H)


def higuchi_fd(series: pd.Series, kmax: int = 10) -> float:
    """Estimate Higuchi fractal dimension.

    Reference: Higuchi, 1988.
    """
    x = series.dropna().astype(float).to_numpy()
    N = 0 if x is None else len(x)
    if N < 3:
        return float('nan')
    Lk = []
    for k in range(1, kmax + 1):
        Lm = []
        for m in range(k):
            idx = np.arange(1, int(np.floor((N - m) / k)) + 1)
            if idx is None or idx.shape[0] == 0:
                continue
            lm = (np.sum(np.abs(x[m + idx * k - 1] - x[m + (idx - 1) * k - 1])) * (N - 1) / (int(np.floor((N - m) / k)) * k)) / k
            Lm.append(lm)
        if len(Lm) > 0:
            Lk.append(np.mean(Lm))
    if len(Lk) < 2:
        return float('nan')
    ln_k = np.log(1.0 / np.arange(1, len(Lk) + 1))
    ln_Lk = np.log(Lk)
    coeffs = np.polyfit(ln_k, ln_Lk, 1)
    return float(coeffs[0]) * -1.0


def permutation_entropy(series: pd.Series, m: int = 3, delay: int = 1) -> float:
    """Permutation entropy (Bandt-Pompe) normalized to [0,1].
    """
    x = np.asarray(series.dropna().astype(float).values)
    n = len(x)
    if n < m * delay:
        return float('nan')
    perms = {}
    for i in range(n - (m - 1) * delay):
        window = x[i:(i + m * delay):delay]
        order = tuple(np.argsort(np.asarray(window)))
        perms[order] = perms.get(order, 0) + 1
    counts = np.array(list(perms.values()), dtype=float)
    p = counts / counts.sum()
    H = -np.sum(p * np.log2(p))
    # use math.factorial to avoid numpy.math typing issues
    Hmax = np.log2(math.factorial(m))
    return float(H / Hmax) if Hmax > 0 else 0.0
