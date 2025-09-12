# advanced_statistics

This package provides a scaffold of advanced time-series statistics and anomaly detectors.

Subpackages:
- rolling_stats: rolling z-scores, MAD, EWMA, IQR, winsorized, skew/kurtosis z, ROC, drawdown
- control_chart: CUSUM, Page–Hinkley, EWMA control, variance-shifts
- seasonality_autocorr: ACF/PACF spikes, STL seasonal strength
- frequency_complexity: spectral entropy, band-power, Hjorth, Hurst, fractal dims
- technical_indicators: RSI, Bollinger, MACD, ATR, Stochastic, Fisher transform
- model_residuals: residual z, quantile residuals, GARCH residuals, Mahalanobis, LOF

Currently scaffolded functions implement one initial method per subpackage and provide a template for adding the rest.
