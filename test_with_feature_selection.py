"""
Test AdvancedTrendPredictor with Feature Selection
Tests multiple feature selection strategies to improve accuracy to 75-85% range
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from realtime_anomaly_project.advanced_trend_predictor import AdvancedTrendPredictor
from realtime_anomaly_project.performance_optimizer import PerformanceOptimizer

def test_with_feature_selection():
    """Test with different feature selection strategies"""
    
    print("=" * 80)
    print("ADVANCED TREND PREDICTOR TEST - WITH FEATURE SELECTION")
    print("=" * 80)
    
    # Download data
    print("\n1. Downloading data for RELIANCE.NS...")
    ticker = "RELIANCE.NS"
    data = yf.download(ticker, period="2y", progress=False)
    
    if data.empty:
        print("❌ Failed to download data!")
        return
    
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Normalize column names to lowercase
    data.columns = data.columns.str.lower()
    
    print(f"✅ Downloaded {len(data)} days of data")
    
    # Prepare features
    print("\n2. Creating features...")
    optimizer = PerformanceOptimizer()
    features_df = optimizer.create_advanced_features(data)
    
    if features_df is None or features_df.empty:
        print("❌ Failed to create features!")
        return
    
    # Remove NaN values
    features_df = features_df.dropna()
    
    # Create target (1 if price goes up next day, 0 otherwise)
    features_df['target'] = (data['Close'].pct_change().shift(-1) > 0).astype(int)
    features_df = features_df.dropna()
    
    # Separate features and target
    feature_cols = [col for col in features_df.columns if col != 'target']
    X = features_df[feature_cols].values
    y = features_df['target'].values
    
    print(f"✅ Created {len(feature_cols)} features for {len(X)} samples")
    
    # Test different feature selection strategies
    strategies = {
        'Top 30 (SelectKBest)': 30,
        'Top 40 (SelectKBest)': 40,
        'Top 50 (SelectKBest)': 50,
    }
    
    results = []
    
    for strategy_name, k_features in strategies.items():
        print(f"\n{'=' * 80}")
        print(f"TESTING: {strategy_name}")
        print(f"{'=' * 80}")
        
        # Feature selection
        print(f"\n3. Selecting top {k_features} features...")
        selector = SelectKBest(f_classif, k=k_features)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_cols[i] for i in selected_indices]
        
        print(f"✅ Selected features: {', '.join(selected_features[:5])}...")
        
        # Train predictor with more trials
        print(f"\n4. Training AdvancedTrendPredictor (50 trials per model)...")
        predictor = AdvancedTrendPredictor(
            target_accuracy=0.85,  # 85% target
            use_gpu=False,
            n_trials=50  # Increased from 30 to 50
        )
        
        # Train and evaluate
        metrics = predictor.train_and_evaluate(X_selected, y)
        
        # Display results
        print(f"\n{'=' * 80}")
        print(f"RESULTS FOR {strategy_name}")
        print(f"{'=' * 80}")
        
        best_model = max(metrics.items(), key=lambda x: x[1].get('accuracy', 0))
        best_name = best_model[0]
        best_metrics = best_model[1]
        
        accuracy = best_metrics.get('accuracy', 0)
        f1 = best_metrics.get('f1', 0)
        precision = best_metrics.get('precision', 0)
        recall = best_metrics.get('recall', 0)
        
        print(f"\n✨ BEST MODEL: {best_name}")
        print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"   Target (85%) Achieved: {'✅ YES' if accuracy >= 0.85 else '❌ NO'}")
        
        # Store results
        results.append({
            'strategy': strategy_name,
            'features': k_features,
            'accuracy': accuracy,
            'f1': f1,
            'model': best_name
        })
        
        print(f"\nALL MODELS PERFORMANCE:")
        print(f"{'Model':<15} {'Accuracy':<20} {'F1 Score':<20} {'Precision':<20}")
        print("-" * 80)
        
        for model_name, model_metrics in sorted(metrics.items()):
            acc = model_metrics.get('accuracy', 0)
            f1_score = model_metrics.get('f1', 0)
            prec = model_metrics.get('precision', 0)
            print(f"{model_name:<15} {acc:.4f} ({acc*100:.2f}%){' '*5} "
                  f"{f1_score:.4f} ({f1_score*100:.2f}%){' '*5} "
                  f"{prec:.4f} ({prec*100:.2f}%)")
    
    # Final comparison
    print(f"\n{'=' * 80}")
    print("FINAL COMPARISON - ALL STRATEGIES")
    print(f"{'=' * 80}")
    print(f"\n{'Strategy':<30} {'Features':<12} {'Best Model':<15} {'Accuracy':<20}")
    print("-" * 80)
    
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        acc = result['accuracy']
        print(f"{result['strategy']:<30} {result['features']:<12} "
              f"{result['model']:<15} {acc:.4f} ({acc*100:.2f}%)")
    
    # Best overall
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\n🏆 BEST OVERALL: {best_result['strategy']}")
    print(f"   Accuracy: {best_result['accuracy']*100:.2f}%")
    print(f"   Target (75-85%) Achieved: {'✅ YES' if 0.75 <= best_result['accuracy'] <= 0.85 else '❌ NO'}")

if __name__ == "__main__":
    # Set encoding for Windows terminal
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    test_with_feature_selection()
