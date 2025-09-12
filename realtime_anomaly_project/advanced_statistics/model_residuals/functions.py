"""Model-residual based diagnostics.

Implemented:
 - one_step_residual_z: compute z-score of one-step-ahead residuals given predictions

"""
import numpy as np
import pandas as pd
 
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from typing import Optional


def one_step_residual_z(observed: pd.Series, predicted: pd.Series) -> pd.Series:
    """Compute z-score of one-step-ahead residuals.

    Parameters
    ----------
    observed : pd.Series
        Observed values
    predicted : pd.Series
        Predicted values (aligned index)

    Returns
    -------
    pd.Series
        z-scored residuals (residual / rolling std of residuals with window=30)

    Example
    -------
    >>> z = one_step_residual_z(obs, pred)

    """
    res = (observed - predicted).astype(float)
    rolling_std = res.rolling(window=30, min_periods=5).std().replace(0, np.nan)
    z = res / rolling_std
    return z


def mahalanobis_distance(df: pd.DataFrame, features: Optional[list] = None) -> pd.Series:
    """Compute Mahalanobis distance for each row in df using selected features.

    Returns a Series aligned to df.index with distances.
    """
    if features is None:
        features = df.columns.tolist()
    X = df[features].dropna()
    if X.empty:
        return pd.Series(index=df.index, dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cov = EmpiricalCovariance().fit(Xs)
    md = cov.mahalanobis(Xs)
    s = pd.Series(index=X.index, data=np.sqrt(md))
    # reindex to full df index, fill NaN where features were NA
    return s.reindex(df.index)


def isolation_forest_score(df: pd.DataFrame, features: Optional[list] = None, random_state: int = 0) -> pd.Series:
    """Return IsolationForest anomaly score (higher -> more anomalous).
    """
    if features is None:
        features = df.columns.tolist()
    X = df[features].dropna()
    if X.empty:
        return pd.Series(index=df.index, dtype=float)
    iso = IsolationForest(random_state=random_state)
    iso.fit(X)
    score = -iso.decision_function(X)  # invert so larger = more anomalous
    s = pd.Series(index=X.index, data=score)
    return s.reindex(df.index)


def lof_score(df: pd.DataFrame, features: Optional[list] = None, n_neighbors: int = 20) -> pd.Series:
    """Local Outlier Factor scores (negative_outlier_factor_ inverted).
    """
    if features is None:
        features = df.columns.tolist()
    X = df[features].dropna()
    if X.empty:
        return pd.Series(index=df.index, dtype=float)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, metric='minkowski', novelty=False)
    labels = lof.fit_predict(X)
    scores = -lof.negative_outlier_factor_
    s = pd.Series(index=X.index, data=scores)
    return s.reindex(df.index)


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, q: float = 0.5) -> float:
    """Pinball (quantile) loss aggregated over series.
    """
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def garch_residuals_placeholder(returns: pd.Series):
    """Placeholder for GARCH residuals computation.

    To implement fully, install `arch` and fit a GARCH model, then return standardized residuals.
    This placeholder returns simple z-scored returns.
    """
    return (returns - returns.mean()) / returns.std()
