"""Fetch live data for all configured tickers, compute metric summaries, and print to stdout.

Usage: python -m realtime_anomaly_project.tools.print_all_scores
"""
from __future__ import annotations
import sys
import time
from typing import Any
import importlib
import numpy as np
import pandas as pd

from realtime_anomaly_project.config import settings
from realtime_anomaly_project.data_ingestion.yahoo import fetch_intraday, data_storage


def _summarize(v: Any):
    if v is None:
        return 'n/a'
    if isinstance(v, (int, float)) and (pd.notna(v)):
        return float(v)
    if isinstance(v, str):
        return v
    try:
        if isinstance(v, pd.DataFrame):
            last = v.tail(1)
            if last.empty:
                return 'n/a'
            return dict(last.to_dict(orient='records')[0])
        if isinstance(v, pd.Series):
            s = v.dropna()
            return float(s.iloc[-1]) if not s.empty else 'n/a'
        arr = np.asarray(v)
        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 1:
            return float(arr[-1]) if arr.size else 'n/a'
    except Exception:
        try:
            return str(v)
        except Exception:
            return 'n/a'


def compute_and_print_all():
    tickers = getattr(settings, 'TICKERS', []) or []
    if not tickers:
        print('No tickers configured in settings.TICKERS')
        return 1

    # lazy import advanced stats modules (they are optional)
    def _safe(m: str):
        try:
            return importlib.import_module(m)
        except Exception:
            return None

    rs = _safe('realtime_anomaly_project.advanced_statistics.rolling_stats.functions')
    cc = _safe('realtime_anomaly_project.advanced_statistics.control_chart.functions')
    sa = _safe('realtime_anomaly_project.advanced_statistics.seasonality_autocorr.functions')
    fc = _safe('realtime_anomaly_project.advanced_statistics.frequency_complexity.functions')
    ti = _safe('realtime_anomaly_project.advanced_statistics.technical_indicators.functions')
    mr = _safe('realtime_anomaly_project.advanced_statistics.model_residuals.functions')

    # header for CSV-like output
    metrics_order = [
        'z_score','robust_z_score','ewma_z_score','iqr_tukey_fence_score','winsorized_z',
        'rolling_skew_z','rolling_kurtosis_z','level_trend_accel','roc','drawdown_depth',
        'cusum_statistic','page_hinkley','ewma_control_distance','rolling_variance_shift','levene_proxy',
        'acf_spike_score','pacf_spike_score','stl_seasonal_strength','seasonal_mismatch_residual','seasonal_peak_timing_error',
        'spectral_entropy','band_power_ratios','hjorth_params','hurst_exponent','higuchi_fd','permutation_entropy',
        'rsi','bollinger','macd','atr','stochastic_kd','fisher_transform',
        'one_step_residual_z','pinball_loss','garch_residuals','mahalanobis_distance','isolation_forest_score','lof_score'
    ]

    rows = []

    for t in tickers:
        try:
            # fetch fresh data
            df = fetch_intraday(t)
            if df is None or df.empty:
                print(f"{t}," + ','.join(['n/a'] * len(metrics_order)))
                continue

            # try to build close series
            close = df.get('close') if 'close' in df.columns else df.iloc[:, 0]
            if close is None:
                print(f"{t}," + ','.join(['n/a'] * len(metrics_order)))
                continue
            close = pd.Series(close).astype(float).dropna()

            # infer period for seasonality (ensure index is a DatetimeIndex)
            try:
                idx = pd.to_datetime(df.index, errors='coerce')
                inferred = pd.infer_freq(pd.DatetimeIndex(idx.dropna())) or None
            except Exception:
                inferred = None
            period = None
            if inferred and 'D' in inferred:
                period = 7
            elif inferred and 'H' in inferred:
                period = 24

            vals = {}
            # Rolling & robust
            try:
                vals['z_score'] = _summarize(rs.z_score(close)) if rs is not None else 'n/a'
            except Exception:
                vals['z_score'] = 'n/a'
            try:
                vals['robust_z_score'] = _summarize(rs.robust_z_score(close)) if rs is not None else 'n/a'
            except Exception:
                vals['robust_z_score'] = 'n/a'
            try:
                vals['ewma_z_score'] = _summarize(rs.ewma_z_score(close)) if rs is not None else 'n/a'
            except Exception:
                vals['ewma_z_score'] = 'n/a'
            try:
                vals['iqr_tukey_fence_score'] = _summarize(rs.iqr_tukey_fence_score(close)) if rs is not None else 'n/a'
            except Exception:
                vals['iqr_tukey_fence_score'] = 'n/a'
            try:
                vals['winsorized_z'] = _summarize(rs.winsorized_z(close)) if rs is not None else 'n/a'
            except Exception:
                vals['winsorized_z'] = 'n/a'
            try:
                vals['rolling_skew_z'] = _summarize(rs.rolling_skew_z(close)) if rs is not None else 'n/a'
            except Exception:
                vals['rolling_skew_z'] = 'n/a'
            try:
                vals['rolling_kurtosis_z'] = _summarize(rs.rolling_kurtosis_z(close)) if rs is not None else 'n/a'
            except Exception:
                vals['rolling_kurtosis_z'] = 'n/a'
            try:
                vals['level_trend_accel'] = _summarize(rs.level_trend_accel(close)) if rs is not None else 'n/a'
            except Exception:
                vals['level_trend_accel'] = 'n/a'
            try:
                vals['roc'] = _summarize(rs.roc(close)) if rs is not None else 'n/a'
            except Exception:
                vals['roc'] = 'n/a'
            try:
                vals['drawdown_depth'] = _summarize(rs.drawdown_depth(close)) if rs is not None else 'n/a'
            except Exception:
                vals['drawdown_depth'] = 'n/a'

            # Control chart
            try:
                vals['cusum_statistic'] = _summarize(cc.cusum_statistic(close)) if cc is not None else 'n/a'
            except Exception:
                vals['cusum_statistic'] = 'n/a'
            try:
                vals['page_hinkley'] = _summarize(cc.page_hinkley(close)) if cc is not None else 'n/a'
            except Exception:
                vals['page_hinkley'] = 'n/a'
            try:
                vals['ewma_control_distance'] = _summarize(cc.ewma_control_distance(close)) if cc is not None else 'n/a'
            except Exception:
                vals['ewma_control_distance'] = 'n/a'
            try:
                vals['rolling_variance_shift'] = _summarize(cc.rolling_variance_shift(close)) if cc is not None else 'n/a'
            except Exception:
                vals['rolling_variance_shift'] = 'n/a'
            try:
                vals['levene_proxy'] = _summarize(cc.levene_proxy(close)) if cc is not None else 'n/a'
            except Exception:
                vals['levene_proxy'] = 'n/a'

            # Seasonality
            try:
                vals['acf_spike_score'] = _summarize(sa.acf_spike_score(close)) if sa is not None else 'n/a'
            except Exception:
                vals['acf_spike_score'] = 'n/a'
            try:
                vals['pacf_spike_score'] = _summarize(sa.pacf_spike_score(close)) if sa is not None else 'n/a'
            except Exception:
                vals['pacf_spike_score'] = 'n/a'
            try:
                if period is None:
                    vals['stl_seasonal_strength'] = 'n/a'
                else:
                    vals['stl_seasonal_strength'] = _summarize(sa.stl_seasonal_strength(close, period=period)) if sa is not None else 'n/a'
            except Exception:
                vals['stl_seasonal_strength'] = 'n/a'
            try:
                vals['seasonal_mismatch_residual'] = _summarize(sa.seasonal_mismatch_residual(close, period=period)) if sa is not None else 'n/a'
            except Exception:
                vals['seasonal_mismatch_residual'] = 'n/a'
            try:
                vals['seasonal_peak_timing_error'] = _summarize(sa.seasonal_peak_timing_error(close, period=period)) if sa is not None else 'n/a'
            except Exception:
                vals['seasonal_peak_timing_error'] = 'n/a'

            # Frequency & complexity
            try:
                vals['spectral_entropy'] = _summarize(fc.spectral_entropy(close)) if fc is not None else 'n/a'
            except Exception:
                vals['spectral_entropy'] = 'n/a'
            try:
                vals['band_power_ratios'] = _summarize(fc.band_power_ratios(close)) if fc is not None else 'n/a'
            except Exception:
                vals['band_power_ratios'] = 'n/a'
            try:
                vals['hjorth_params'] = _summarize(fc.hjorth_params(close)) if fc is not None else 'n/a'
            except Exception:
                vals['hjorth_params'] = 'n/a'
            try:
                vals['hurst_exponent'] = _summarize(fc.hurst_exponent(close)) if fc is not None else 'n/a'
            except Exception:
                vals['hurst_exponent'] = 'n/a'
            try:
                vals['higuchi_fd'] = _summarize(fc.higuchi_fd(close)) if fc is not None else 'n/a'
            except Exception:
                vals['higuchi_fd'] = 'n/a'
            try:
                vals['permutation_entropy'] = _summarize(fc.permutation_entropy(close)) if fc is not None else 'n/a'
            except Exception:
                vals['permutation_entropy'] = 'n/a'

            # Technical indicators
            try:
                vals['rsi'] = _summarize(ti.rsi(close, period=14)) if ti is not None else 'n/a'
            except Exception:
                vals['rsi'] = 'n/a'
            try:
                vals['bollinger'] = _summarize(ti.bollinger(close)) if ti is not None else 'n/a'
            except Exception:
                vals['bollinger'] = 'n/a'
            try:
                vals['macd'] = _summarize(ti.macd(close)) if ti is not None else 'n/a'
            except Exception:
                vals['macd'] = 'n/a'
            try:
                if ti is not None and 'high' in df.columns and 'low' in df.columns:
                    vals['atr'] = _summarize(ti.atr(df['high'].astype(float), df['low'].astype(float), close))
                else:
                    vals['atr'] = 'n/a'
            except Exception:
                vals['atr'] = 'n/a'
            try:
                if ti is not None and 'high' in df.columns and 'low' in df.columns:
                    vals['stochastic_kd'] = _summarize(ti.stochastic_kd(df['high'].astype(float), df['low'].astype(float), close))
                else:
                    vals['stochastic_kd'] = 'n/a'
            except Exception:
                vals['stochastic_kd'] = 'n/a'
            try:
                vals['fisher_transform'] = _summarize(ti.fisher_transform(close)) if ti is not None else 'n/a'
            except Exception:
                vals['fisher_transform'] = 'n/a'

            # Model-based / multivariate
            try:
                pred = close.shift(1).reindex(close.index)
                vals['one_step_residual_z'] = _summarize(mr.one_step_residual_z(close, pred)) if mr is not None else 'n/a'
            except Exception:
                vals['one_step_residual_z'] = 'n/a'
            try:
                pred_q = close.shift(1)
                vals['pinball_loss'] = _summarize(mr.pinball_loss(close.dropna(), pred_q.dropna(), q=0.5)) if mr is not None else 'n/a'
            except Exception:
                vals['pinball_loss'] = 'n/a'
            try:
                vals['garch_residuals'] = _summarize(mr.garch_residuals_placeholder(close.pct_change().dropna())) if mr is not None else 'n/a'
            except Exception:
                vals['garch_residuals'] = 'n/a'
            try:
                mv_feats = [c for c in ['close','open','high','low','volume'] if c in df.columns]
                if mv_feats and mr is not None:
                    vals['mahalanobis_distance'] = _summarize(mr.mahalanobis_distance(df[mv_feats]))
                else:
                    vals['mahalanobis_distance'] = 'n/a'
            except Exception:
                vals['mahalanobis_distance'] = 'n/a'
            try:
                mv = df[[c for c in ['close','open','high','low','volume'] if c in df.columns]]
                vals['isolation_forest_score'] = _summarize(mr.isolation_forest_score(mv)) if mr is not None else 'n/a'
            except Exception:
                vals['isolation_forest_score'] = 'n/a'
            try:
                mv = df[[c for c in ['close','open','high','low','volume'] if c in df.columns]]
                vals['lof_score'] = _summarize(mr.lof_score(mv)) if mr is not None else 'n/a'
            except Exception:
                vals['lof_score'] = 'n/a'

            # collect row
            row = {'ticker': t}
            for m in metrics_order:
                v = vals.get(m, 'n/a')
                # flatten small dict values for scalar cell
                if isinstance(v, dict):
                    v = '|'.join([f"{k}:{round(float(x),4) if isinstance(x,(int,float)) else x}" for k,x in v.items()])
                row[m] = v
            rows.append(row)
            # be kind to APIs
            time.sleep(0.2)

        except Exception as e:
            print(f"{t},error,{e}")

    # build DataFrame and print a compact table
    if rows:
        all_df = pd.DataFrame(rows).set_index('ticker')
        # pretty print (truncate wide columns sensibly)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
            print('\nComputed metrics (summary) for all tickers:\n')
            print(all_df.to_string())
        # save CSV
        out_path = 'realtime_anomaly_project/tools/all_tickers_metrics.csv'
        try:
            all_df.to_csv(out_path)
            print(f"\nSaved CSV: {out_path}")
        except Exception:
            pass
    else:
        print('No results to show')

    return 0


if __name__ == '__main__':
    sys.exit(compute_and_print_all())
