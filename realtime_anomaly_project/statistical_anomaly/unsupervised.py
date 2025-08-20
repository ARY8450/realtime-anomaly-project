import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def _features_from_df(df):
    """Simple feature extraction: rolling returns and volatility from close price."""
    if df is None or df.empty or 'close' not in df.columns:
        return None
    s = df['close'].astype(float).sort_index()
    returns = s.pct_change().fillna(0)
    vol = returns.rolling(5, min_periods=1).std().fillna(0)
    feat = pd.concat([returns, vol], axis=1).fillna(0)
    feat.columns = ['ret', 'vol']
    return feat

def _build_df_map_from_storage_or_db():
    """Try in-memory data_storage first, then fall back to DB."""
    df_map = {}
    try:
        from realtime_anomaly_project.data_ingestion.yahoo import data_storage
        for ticker, df in (data_storage or {}).items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            df_local = df.copy()
            # ensure datetime index and tz-aware
            if 'datetime' in df_local.columns:
                df_local['datetime'] = pd.to_datetime(df_local['datetime'], errors='coerce')
                df_local = df_local.dropna(subset=['datetime']).set_index('datetime')
            if not isinstance(df_local.index, pd.DatetimeIndex):
                try:
                    df_local.index = pd.to_datetime(df_local.index, errors='coerce')
                    df_local = df_local[~df_local.index.isna()]
                except Exception:
                    continue
            if df_local.index.tz is None:
                try:
                    df_local.index = df_local.index.tz_localize('UTC')
                except Exception:
                    df_local.index = pd.to_datetime(df_local.index, errors='coerce').tz_localize('UTC')
            if 'close' not in df_local.columns and 'Close' in df_local.columns:
                df_local = df_local.rename(columns={'Close': 'close'})
            df_local['close'] = pd.to_numeric(df_local['close'], errors='coerce')
            df_local = df_local.dropna(subset=['close'])
            if not df_local.empty:
                df_map[ticker] = df_local.sort_index()
        if df_map:
            return df_map
    except Exception:
        pass

    # ensure DB is initialised before reading
    try:
        from realtime_anomaly_project.database.db_setup import setup_database
        setup_database()   # creates DB file/tables and sets ENGINE
    except Exception:
        pass

    # fallback: load from DB (existing logic)
    try:
        from realtime_anomaly_project.database.db_setup import setup_database, StockData
        Session = setup_database()
        import collections
        with Session() as session:
            rows = session.query(StockData).order_by(StockData.timestamp).all()
            if not rows:
                return {}
            buckets = collections.defaultdict(list)
            for r in rows:
                buckets[r.ticker].append({
                    "datetime": r.timestamp,
                    "open": r.open_price,
                    "high": r.high_price,
                    "low": r.low_price,
                    "close": r.close_price,
                    "volume": r.volume
                })
            for ticker, recs in buckets.items():
                df = pd.DataFrame(recs)
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
                df = df.dropna(subset=['datetime']).set_index('datetime').sort_index()
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df = df.dropna(subset=['close'])
                if not df.empty:
                    df_map[ticker] = df
    except Exception:
        pass

    return df_map

def compute_unsupervised_anomalies(df_map=None, n_neighbors=20):
    """
    Compute unsupervised anomaly scores for each ticker using multiple methods.
    If df_map is None, build it from in-memory storage or DB.
    Returns dict[ticker] -> dict of method -> score (higher = more anomalous).
    """
    results = {}

    if df_map is None:
        df_map = _build_df_map_from_storage_or_db()

    if not df_map:
        return results

    for ticker, df in df_map.items():
        feat = _features_from_df(df)
        if feat is None or feat.shape[0] < 5:
            results[ticker] = None
            continue

        X = feat.values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        scores = {}

        # Isolation Forest
        try:
            iso = IsolationForest(random_state=42, contamination='auto').fit(Xs)
            iso_scores = -iso.decision_function(Xs)
            scores['isolation_forest'] = float(np.nanmax(iso_scores))
        except Exception:
            scores['isolation_forest'] = None

        # Local Outlier Factor
        try:
            lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, max(5, Xs.shape[0]-1)), novelty=False)
            lof_pred = lof.fit_predict(Xs)
            lof_scores = -lof.negative_outlier_factor_
            scores['local_outlier_factor'] = float(np.nanmax(lof_scores))
        except Exception:
            scores['local_outlier_factor'] = None

        # One-Class SVM
        try:
            oc = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05).fit(Xs)
            oc_scores = -oc.decision_function(Xs)
            scores['one_class_svm'] = float(np.nanmax(oc_scores))
        except Exception:
            scores['one_class_svm'] = None

        # PCA reconstruction error
        try:
            pca = PCA(n_components=min(2, Xs.shape[1]))
            comps = pca.fit_transform(Xs)
            recon = pca.inverse_transform(comps)
            recon_err = np.mean((Xs - recon)**2, axis=1)
            scores['pca_recon_error'] = float(np.nanmax(recon_err))
        except Exception:
            scores['pca_recon_error'] = None

        results[ticker] = scores

    return results
