"""Technical indicator implementations (RSI, MACD, Bollinger etc.)

Implemented:
 - rsi: standard Wilder's RSI with optional divergence simple helper

"""
import numpy as np
import pandas as pd


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder's RSI for a series.

    Parameters
    ----------
    series : pd.Series
        Price series (close prices recommended).
    period : int
        Lookback period (default 14).

    Returns
    -------
    pd.Series
        RSI values aligned with input index.

    Example
    -------
    >>> r = rsi(close_series, period=14)
    # Divergence helper: compare r[-1] with price trend separately.

    """
    x = series.dropna().astype(float)
    if x.empty:
        return pd.Series(index=series.index, dtype=float)

    delta = x.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # Wilder's smoothing
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    # reindex to original
    rsi = rsi.reindex(series.index)
    return rsi


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD line, signal line, and MACD histogram.

    Returns DataFrame with columns ['macd','signal','hist']
    """
    x = series.astype(float)
    ema_fast = x.ewm(span=fast, adjust=False).mean()
    ema_slow = x.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig
    return pd.DataFrame({'macd': macd_line, 'signal': sig, 'hist': hist})


def bollinger(series: pd.Series, window: int = 20, nstd: float = 2.0) -> pd.DataFrame:
    """Compute Bollinger bands, %B and bandwidth.

    Returns DataFrame with ['mid','upper','lower','percent_b','bandwidth']
    """
    mid = series.rolling(window=window, min_periods=3).mean()
    std = series.rolling(window=window, min_periods=3).std()
    upper = mid + nstd * std
    lower = mid - nstd * std
    percent_b = (series - lower) / (upper - lower)
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    return pd.DataFrame({'mid': mid, 'upper': upper, 'lower': lower, 'percent_b': percent_b, 'bandwidth': bandwidth})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (ATR) normalized by price (ATR / close).

    Accepts high, low, close series aligned to same index.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(window=window, min_periods=3).mean()
    return atr_val / close.replace(0, np.nan)


def stochastic_kd(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
    """Compute Stochastic %K and %D.

    %K = (close - lowest_low) / (highest_high - lowest_low)
    %D = SMA of %K
    """
    lowest = low.rolling(window=k_window, min_periods=3).min()
    highest = high.rolling(window=k_window, min_periods=3).max()
    k = 100 * (close - lowest) / (highest - lowest)
    d = k.rolling(window=d_window, min_periods=1).mean()
    return pd.DataFrame({'%K': k, '%D': d})


def fisher_transform(series: pd.Series, period: int = 10) -> pd.Series:
    """Fisher transform of normalized returns over a rolling window.

    This computes a transformed signal that emphasizes turning points.
    """
    returns = series.pct_change().fillna(0)
    # normalize series to (-0.999,0.999)
    roll_min = returns.rolling(window=period, min_periods=1).min()
    roll_max = returns.rolling(window=period, min_periods=1).max()
    x = 2 * (returns - roll_min) / (roll_max - roll_min).replace(0, np.nan) - 1
    x = x.clip(-0.999, 0.999).fillna(0)
    fish = 0.5 * np.log((1 + x) / (1 - x))
    return pd.Series(fish, index=series.index, name=series.name)
