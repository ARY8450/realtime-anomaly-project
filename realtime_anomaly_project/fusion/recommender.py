import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import randint, uniform

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for ML from a ticker DataFrame.

    Features:
    - pct_change (1,2,5)
    - rolling mean/std (5,10,20)
    - RSI-like using simple rolling gains/losses
    - z-score of returns
    - anomaly flag if present
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    close = pd.to_numeric(df["close"], errors="coerce").ffill()
    returns = close.pct_change().fillna(0)

    feats = pd.DataFrame(index=df.index)
    feats["ret_1"] = returns
    feats["ret_2"] = returns.rolling(2).sum().fillna(0)
    feats["ret_5"] = returns.rolling(5).sum().fillna(0)

    feats["ma5"] = close.rolling(5).mean().ffill()
    feats["ma10"] = close.rolling(10).mean().ffill()
    feats["ma20"] = close.rolling(20).mean().ffill()

    feats["std5"] = close.rolling(5).std().fillna(0)
    feats["std10"] = close.rolling(10).std().fillna(0)

    # EMA indicators
    feats["ema8"] = close.ewm(span=8, adjust=False).mean()
    feats["ema21"] = close.ewm(span=21, adjust=False).mean()
    feats["macd"] = feats["ema8"] - feats["ema21"]

    # Momentum
    feats["mom_3"] = close.diff(3).fillna(0)
    feats["mom_7"] = close.diff(7).fillna(0)

    # ATR-like volatility using high/low if available
    if "high" in df.columns and "low" in df.columns and "close" in df.columns:
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        feats["atr14"] = tr.rolling(14).mean().fillna(0)
    else:
        feats["atr14"] = feats["std10"]

    # RSI-ish
    delta = close.diff().fillna(0)
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rsi = 100 - 100 / (1 + (up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9)))
    feats["rsi"] = rsi.fillna(50)

    # zscore of returns
    feats["zret"] = (returns - returns.rolling(20).mean()).fillna(0) / (returns.rolling(20).std().replace(0, 1e-9)).fillna(0)

    # anomaly flag if available
    if "_anomaly" in df.columns:
        feats["anom"] = (df["_anomaly"] != "none").astype(int).reindex(feats.index).fillna(0).astype(int)
    else:
        feats["anom"] = 0

    # last close
    feats["close"] = close

    # drop rows with any NaN (should be minimal after fills)
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return feats


def _make_target(df: pd.DataFrame, horizon: int = 1, thresh: float = 0.001) -> pd.Series:
    """Create a multiclass target for next-step movement.

    - 1 -> buy (next return > thresh)
    - -1 -> short (next return < -thresh)
    - 0 -> hold/neutral
    """
    close = pd.to_numeric(df["close"], errors="coerce").ffill()
    fut = close.shift(-horizon)
    ret = (fut - close) / close
    tgt = pd.Series(0, index=close.index)
    tgt[ret > thresh] = 1
    tgt[ret < -thresh] = -1
    return tgt


def prepare_dataset(ticker_dfs: dict, horizon: int = 1, thresh: float = 0.001):
    """Prepare X, y from a dict of ticker->DataFrame.
    Returns X, y (aligned)"""
    X_parts = []
    y_parts = []
    for t, df in ticker_dfs.items():
        try:
            feats = _make_features(df)
            tgt = _make_target(df, horizon=horizon, thresh=thresh)
            # align
            common = feats.index.intersection(tgt.index)
            if len(common) < 50:
                continue
            X_parts.append(feats.loc[common])
            y_parts.append(tgt.loc[common])
        except Exception:
            continue
    if not X_parts:
        return None, None
    X = pd.concat(X_parts, axis=0, ignore_index=False)
    y = pd.concat(y_parts, axis=0, ignore_index=False).reindex(X.index)
    # drop any rows with nan target
    mask = y.notna()
    # align mask index to X
    mask_bool = mask.reindex(X.index).fillna(False).astype(bool).to_numpy()
    X = X.iloc[mask_bool]
    y = y.iloc[mask_bool]
    return X, y


def train_model(ticker_dfs: dict, horizon: int = 1, thresh: float = 0.001, test_size: float = 0.2, random_state: int = 0):
    """Train a RandomForest classifier on prepared data and save to disk. Returns (model, report)."""
    X, y = prepare_dataset(ticker_dfs, horizon=horizon, thresh=thresh)
    if X is None or y is None or len(X) < 200:
        raise ValueError("Not enough training data. Provide more tickers or longer history.")
    # convert multiclass -1,0,1 to labels 0,1,2
    y_enc = y.replace({-1: 0, 0: 1, 1: 2}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc)

    model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    acc = accuracy_score(y_test, preds)

    # persist model and column order
    obj = {"model": model, "columns": list(X.columns)}
    joblib.dump(obj, MODEL_PATH)

    return model, report, acc


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    obj = joblib.load(MODEL_PATH)
    return obj


def predict_from_df(df: pd.DataFrame):
    """Load model from disk and predict the class for the last row of df."""
    obj = load_model()
    if obj is None:
        return None
    model = obj["model"]
    cols = obj["columns"]
    feats = _make_features(df)
    if feats.empty:
        return None
    x = feats.iloc[[-1]][cols].fillna(0)
    pred = model.predict(x)[0]
    prob = model.predict_proba(x)[0]
    # decode label back
    label = {0: "short", 1: "hold", 2: "buy"}.get(int(pred), "hold")
    return {"label": label, "prob": prob.tolist()}
