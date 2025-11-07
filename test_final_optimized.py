"""
Final Optimized Test - 41 Features + SMOTE + Better Hyperparameters
Target: 75-85% accuracy
"""

import yfinance as yf
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_anomaly_project.performance_optimizer import PerformanceOptimizer

def test_final_optimized():
    """Test with balanced data and optimized parameters"""
    
    print("=" * 80)
    print("FINAL OPTIMIZED TEST - 41 FOCUSED FEATURES")
    print("=" * 80)
    print("\n📋 Key Features:")
    print("   ✓ VROC, Cumulative Return, EMA, OBV, Bid-Ask Spread")
    print("   ✓ RSI, MACD, Bollinger Bands, ATR, ROC")
    print("   ✓ 5 years of data (1237+ samples)")
    print("   ✓ SMOTE for class balance")
    print("=" * 80)
    
    # Download data
    print("\n1. Downloading data...")
    data = yf.download("RELIANCE.NS", period="5y", progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = data.columns.str.lower()
    
    print(f"✅ Downloaded {len(data)} days")
    
    # Create features
    print("\n2. Creating features...")
    optimizer = PerformanceOptimizer()
    features_df = optimizer.create_advanced_features(data).dropna()
    
    # Create target
    features_df['target'] = (data['close'].pct_change().shift(-1) > 0).astype(int)
    features_df = features_df.dropna()
    
    # Replace inf values with NaN, then fill
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(0)
    
    feature_cols = [col for col in features_df.columns if col != 'target']
    X = features_df[feature_cols].values
    y = features_df['target'].values
    
    print(f"✅ Created {len(feature_cols)} features, {len(X)} samples")
    print(f"   Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply SMOTE to balance training data
    print("\n3. Applying SMOTE for class balance...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"✅ Balanced: {np.bincount(y_train_balanced)}")
    
    # Train models
    print("\n4. Training models...")
    print("-" * 80)
    
    results = {}
    
    # XGBoost
    print("\n📊 Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_balanced, y_train_balanced, verbose=False)
    y_pred = xgb_model.predict(X_test)
    results['XGBoost'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted')
    }
    print(f"✅ Accuracy: {results['XGBoost']['accuracy']*100:.2f}%")
    
    # LightGBM
    print("\n📊 Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train_balanced, y_train_balanced)
    y_pred = lgb_model.predict(X_test)
    results['LightGBM'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted')
    }
    print(f"✅ Accuracy: {results['LightGBM']['accuracy']*100:.2f}%")
    
    # CatBoost
    print("\n📊 Training CatBoost...")
    cat_model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=False
    )
    cat_model.fit(X_train_balanced, y_train_balanced)
    y_pred = cat_model.predict(X_test)
    results['CatBoost'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted')
    }
    print(f"✅ Accuracy: {results['CatBoost']['accuracy']*100:.2f}%")
    
    # Ensemble (Voting)
    print("\n📊 Creating Ensemble...")
    xgb_pred = xgb_model.predict(X_test)
    lgb_pred = lgb_model.predict(X_test)
    cat_pred = cat_model.predict(X_test)
    ensemble_pred = np.round((xgb_pred + lgb_pred + cat_pred) / 3).astype(int)
    results['Ensemble'] = {
        'accuracy': accuracy_score(y_test, ensemble_pred),
        'f1': f1_score(y_test, ensemble_pred, average='weighted'),
        'precision': precision_score(y_test, ensemble_pred, average='weighted'),
        'recall': recall_score(y_test, ensemble_pred, average='weighted')
    }
    print(f"✅ Accuracy: {results['Ensemble']['accuracy']*100:.2f}%")
    
    # Results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    best_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_metrics = results[best_name]
    
    print(f"\n✨ BEST MODEL: {best_name}")
    print(f"   Accuracy:  {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
    print(f"   F1 Score:  {best_metrics['f1']:.4f} ({best_metrics['f1']*100:.2f}%)")
    print(f"   Precision: {best_metrics['precision']:.4f} ({best_metrics['precision']*100:.2f}%)")
    print(f"   Recall:    {best_metrics['recall']:.4f} ({best_metrics['recall']*100:.2f}%)")
    
    target_met = 0.75 <= best_metrics['accuracy'] <= 0.85
    print(f"\n🎯 Target (75-85%) Achieved: {'✅ YES!' if target_met else '❌ NO'}")
    
    if target_met:
        print("\n🎉 SUCCESS! Accuracy in target range!")
    elif best_metrics['accuracy'] > 0.85:
        print("\n🚀 EXCEEDED TARGET!")
    else:
        gap = 0.75 - best_metrics['accuracy']
        print(f"\n⚠️  Need {gap*100:.2f}% more to reach 75%")
    
    print(f"\n{'=' * 80}")
    print("ALL MODELS:")
    print(f"{'Model':<15} {'Accuracy':<15} {'F1':<15} {'Precision':<15}")
    print("-" * 80)
    for name, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{name:<15} {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)  "
              f"{metrics['f1']:.4f}        {metrics['precision']:.4f}")
    
    print("\n" + "=" * 80)
    print("COMPARISON:")
    print("-" * 80)
    print(f"89 features (2y):  59.60% ❌")
    print(f"23 features (2y):  60.61% ⚠️")
    print(f"41 features (5y):  {best_metrics['accuracy']*100:.2f}% {'✅' if best_metrics['accuracy'] >= 0.75 else '❌'}")
    print("=" * 80)

if __name__ == "__main__":
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    test_final_optimized()
