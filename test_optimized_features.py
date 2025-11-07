"""
Test AdvancedTrendPredictor with Optimized Feature Set
Uses user-requested features + high-impact complementary indicators
Target: 75-85% accuracy
"""

import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realtime_anomaly_project.advanced_trend_predictor import AdvancedTrendPredictor
from realtime_anomaly_project.performance_optimizer import PerformanceOptimizer

def test_optimized_predictor():
    """Test with optimized feature set and more data"""
    
    print("=" * 80)
    print("OPTIMIZED TREND PREDICTOR TEST - FOCUSED FEATURES")
    print("=" * 80)
    print("\n📋 USER-REQUESTED FEATURES:")
    print("   ✓ Volume Rate of Change (VROC)")
    print("   ✓ Cumulative Return")
    print("   ✓ Exponential Moving Average (EMA)")
    print("   ✓ On-Balance Volume (OBV)")
    print("   ✓ Bid-Ask Spread (High-Low proxy)")
    print("\n🎯 Target Accuracy: 75-85%")
    print("=" * 80)
    
    # Download MORE data for better training
    print("\n1. Downloading 5 YEARS of data for RELIANCE.NS...")
    ticker = "RELIANCE.NS"
    data = yf.download(ticker, period="5y", progress=False)
    
    if data.empty:
        print("❌ Failed to download data!")
        return
    
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Normalize column names to lowercase
    data.columns = data.columns.str.lower()
    
    print(f"✅ Downloaded {len(data)} days of data (5 years)")
    
    # Prepare features
    print("\n2. Creating OPTIMIZED features...")
    optimizer = PerformanceOptimizer()
    features_df = optimizer.create_advanced_features(data)
    
    if features_df is None or features_df.empty:
        print("❌ Failed to create features!")
        return
    
    # Remove NaN values
    features_df = features_df.dropna()
    
    # Create target (1 if price goes up next day, 0 otherwise)
    features_df['target'] = (data['close'].pct_change().shift(-1) > 0).astype(int)
    features_df = features_df.dropna()
    
    # Separate features and target
    feature_cols = [col for col in features_df.columns if col != 'target']
    X = features_df[feature_cols].values
    y = features_df['target'].values
    
    print(f"✅ Created {len(feature_cols)} features for {len(X)} samples")
    print(f"   Ratio: {len(X) / len(feature_cols):.1f} samples per feature")
    
    print(f"\n📊 Feature List:")
    print("-" * 80)
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    # Split data (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    print(f"\n3. Data split:")
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Testing set:  {len(X_test)} samples")
    
    # Train predictor with MORE trials for better optimization
    print(f"\n4. Training AdvancedTrendPredictor...")
    print(f"   • Using 50 Optuna trials per model (increased from 30)")
    print(f"   • Models: XGBoost, LightGBM, CatBoost, Ensemble")
    print(f"   • Target accuracy: 85%")
    print("-" * 80)
    
    predictor = AdvancedTrendPredictor(
        target_accuracy=0.85,
        use_gpu=False
    )
    
    # Train and evaluate
    results = predictor.train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # Display results
    print(f"\n{'=' * 80}")
    print(f"FINAL RESULTS - OPTIMIZED FEATURE SET")
    print(f"{'=' * 80}")
    
    if not results or 'all_results' not in results:
        print("❌ No models trained successfully!")
        return
    
    # Get all model results
    all_results = results['all_results']
    best_acc = results.get('best_accuracy', 0)
    best_name = results.get('best_model_name', 'UNKNOWN')
    
    # Display best model
    if best_acc > 0:
        print(f"\n✨ BEST MODEL: {best_name.upper()}")
        
        if best_name in all_results:
            best_metrics = all_results[best_name]['metrics']
            accuracy = best_metrics.get('accuracy', 0)
            f1 = best_metrics.get('f1', 0)
            precision = best_metrics.get('precision', 0)
            recall = best_metrics.get('recall', 0)
            
            print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"   F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
            print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
            
            # Check if target achieved
            target_met = 0.75 <= accuracy <= 0.85
            print(f"\n🎯 Target (75-85%) Achieved: {'✅ YES!' if target_met else '❌ NO'}")
            
            if target_met:
                print(f"\n🎉 SUCCESS! Accuracy is in target range!")
            elif accuracy > 0.85:
                print(f"\n🚀 EXCEEDED TARGET! Accuracy is above 85%!")
            else:
                print(f"\n⚠️  Need {(0.75 - accuracy)*100:.2f}% more to reach target")
    
    print(f"\n{'=' * 80}")
    print("ALL MODELS PERFORMANCE:")
    print(f"{'=' * 80}")
    print(f"{'Model':<15} {'Accuracy':<20} {'F1 Score':<20} {'Precision':<20}")
    print("-" * 80)
    
    for model_name, model_data in sorted(all_results.items()):
        if isinstance(model_data, dict) and 'metrics' in model_data:
            m = model_data['metrics']
            acc = m.get('accuracy', 0)
            f1_score = m.get('f1', 0)
            prec = m.get('precision', 0)
            print(f"{model_name.upper():<15} {acc:.4f} ({acc*100:.2f}%){' '*5} "
                  f"{f1_score:.4f} ({f1_score*100:.2f}%){' '*5} "
                  f"{prec:.4f} ({prec*100:.2f}%)")
    
    print("\n" + "=" * 80)
    print("COMPARISON WITH PREVIOUS RESULTS:")
    print("-" * 80)
    print("89 features (2 years): 59.60% accuracy ❌")
    print("23 features (2 years): 60.61% accuracy ⚠️")
    print(f"{len(feature_cols)} features (5 years): {best_acc*100:.2f}% accuracy {'✅' if best_acc >= 0.75 else '❌'}")
    print("=" * 80)

if __name__ == "__main__":
    # Set encoding for Windows terminal
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    test_optimized_predictor()
