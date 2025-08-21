"""Train classifier and regressor on all tickers listed in settings.TICKERS.

Usage: run from project root with the same venv: python tools/train_all_tickers.py

The script loads each ticker via the existing loader `realtime_anomaly_project.statistical_anomaly.show_results.load_ticker_df`,
skips tickers without sufficient data, then calls `realtime_anomaly_project.fusion.recommender.train_model` and
`train_regression_model` and prints summaries.
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from realtime_anomaly_project.config import settings
from realtime_anomaly_project.statistical_anomaly.show_results import load_ticker_df
from realtime_anomaly_project.fusion import recommender

def main():
    tickers = getattr(settings, 'TICKERS', []) or []
    print(f"Tickers to consider: {len(tickers)}")
    ticker_map = {}
    for t in tickers:
        df = load_ticker_df(t)
        if df is None:
            print(f"[SKIP] {t}: no data")
            continue
        # need at least 60 rows to be useful
        if len(df) < 60:
            print(f"[SKIP] {t}: only {len(df)} rows")
            continue
        ticker_map[t] = df
        print(f"[LOAD] {t}: {len(df)} rows")

    if not ticker_map:
        print("No tickers with sufficient data found. Aborting.")
        return

    print("Starting classifier training (this may take a while)...")
    try:
        # disable LightGBM here to avoid runtime hangs on some datasets/environments
        try:
            recommender.LGBM_AVAILABLE = False
        except Exception:
            pass
        # use a small search budget and single-job to keep runtime reasonable
        model, report, acc = recommender.train_model(ticker_map, n_iter=8, n_jobs=1)
        print("Classifier trained. Accuracy on holdout:", acc)
        print("Classification report summary (per-class precision/recall/f1):")
        if isinstance(report, dict):
            for k, v in report.items():
                if k in ('accuracy', 'macro avg', 'weighted avg'):
                    continue
                print(f"Class {k}: precision={v.get('precision')}, recall={v.get('recall')}, f1={v.get('f1-score')}")
        else:
            # report was not a dict (static analyzer thought it might be a string); print raw report
            print("Classification report (raw):")
            print(report)
    except Exception as e:
        print("Classifier training failed:", e)

    print("Starting regression training (predict fractional return)...")
    try:
        reg, metrics = recommender.train_regression_model(ticker_map)
        print("Regressor trained. MSE:", metrics.get('mse'), "MAE:", metrics.get('mae'))
    except Exception as e:
        print("Regression training failed:", e)

    print("Done. Models (if trained) persisted to the recommender MODEL_PATH.")

if __name__ == '__main__':
    main()
