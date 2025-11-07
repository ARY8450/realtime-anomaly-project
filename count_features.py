"""Count how many features are calculated vs used"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from realtime_anomaly_project.performance_optimizer import PerformanceOptimizer

# Create test data
optimizer = PerformanceOptimizer()
test_data = pd.DataFrame({
    'close': np.random.randn(500).cumsum() + 100,
    'high': np.random.randn(500).cumsum() + 102,
    'low': np.random.randn(500).cumsum() + 98,
    'volume': np.random.randint(1000000, 5000000, 500)
})

# Calculate features
features = optimizer.create_advanced_features(test_data)

print(f"=" * 80)
print(f"FEATURE ANALYSIS")
print(f"=" * 80)
print(f"\n📊 Total features CALCULATED: {len(features.columns)}")
print(f"\n✅ All calculated features:")
print(f"{'-' * 80}")

# Group features by category
categories = {
    'Price-Based': ['returns', 'log_returns', 'returns_squared', 'volatility', 'vol_ratio'],
    'Momentum': ['rsi', 'macd', 'stoch', 'williams'],
    'Trend (MA)': ['sma', 'ema', 'price_to'],
    'Volume': ['volume', 'obv', 'vpt', 'mfi'],
    'Bollinger Bands': ['bb_'],
    'Volatility': ['atr', 'natr'],
    'Rate of Change': ['roc_', 'momentum', 'acceleration'],
    'Price Patterns': ['hl_', 'close_position', 'donchian'],
    'Statistical': ['zscore', 'percentile'],
    'Lag Features': ['lag_'],
    'Interaction': ['rsi_volatility', 'volume_price_corr']
}

for category, keywords in categories.items():
    matching = [col for col in features.columns if any(kw in col for kw in keywords)]
    if matching:
        print(f"\n{category} ({len(matching)} features):")
        for col in matching:
            print(f"  • {col}")

print(f"\n{'-' * 80}")
print(f"\n📋 Complete feature list ({len(features.columns)} total):")
for i, col in enumerate(features.columns, 1):
    print(f"{i:3d}. {col}")
