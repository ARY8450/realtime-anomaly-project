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
        model, report, acc = recommender.train_model(ticker_map)
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

    print("Starting enhanced classifier training with feature selection...")
    try:
        model_fs, report_fs, acc_fs, selected_features = recommender.train_model_with_feature_selection(ticker_map)
        print(f"Enhanced classifier trained. Accuracy: {acc_fs:.4f}")
        print(f"Selected {len(selected_features)} features: {selected_features[:5]}...")
    except Exception as e:
        print("Enhanced classifier training failed:", e)

    print("Starting ensemble classifier training...")
    try:
        ensemble_model, ensemble_report, ensemble_acc = recommender.train_ensemble_model(ticker_map)
        print(f"Ensemble classifier trained. Accuracy: {ensemble_acc:.4f}")
    except Exception as e:
        print("Ensemble classifier training failed:", e)

    print("Starting enhanced regression training...")
    try:
        ensemble_reg, reg_metrics = recommender.train_ensemble_regressor(ticker_map)
        print(f"Ensemble regressor trained. MSE: {reg_metrics.get('mse'):.6f}, MAE: {reg_metrics.get('mae'):.6f}, R²: {reg_metrics.get('r2'):.4f}")
    except Exception as e:
        print("Ensemble regression training failed:", e)

    print("Done. Models (if trained) persisted to the recommender MODEL_PATH.")

if __name__ == '__main__':
    main()
