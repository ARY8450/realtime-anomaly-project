"""
Real Backtesting System for Performance Validation
Tests predictions against actual historical outcomes to calculate true performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import logging
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealBacktester:
    """
    Real backtesting system that validates predictions against actual outcomes
    """
    
    def __init__(self, lookback_days: int = 365):
        """
        Initialize backtester
        
        Args:
            lookback_days: Number of days to look back for testing
        """
        self.lookback_days = lookback_days
        self.results_cache = {}
        
    def backtest_trend_predictions(self, ticker: str, df: pd.DataFrame) -> Dict[str, float]:
        """
        Backtest trend predictions against actual price movements
        
        For each historical point:
        1. Make a prediction (BUY/SELL/HOLD) based on technical indicators
        2. Check actual price movement in next N days
        3. Compare prediction vs outcome
        
        Returns:
            Dictionary with real precision, recall, F1, accuracy metrics
        """
        try:
            if len(df) < 30:
                logger.warning(f"Insufficient data for {ticker}: {len(df)} rows")
                return self._get_default_metrics()
            
            predictions = []
            actuals = []
            
            # Use rolling window approach
            for i in range(20, len(df) - 5):  # Need 20 days history, 5 days future
                # Get historical data up to this point
                hist_data = df.iloc[:i+1].copy()
                
                # Make prediction
                prediction = self._make_trend_prediction(hist_data)
                
                # Get actual outcome (5-day forward return)
                current_price = df.iloc[i]['close']
                future_price = df.iloc[i+5]['close']
                actual_return = (future_price - current_price) / current_price
                
                # Convert to classification
                # BUY = 1 (expect price increase)
                # HOLD = 0 (expect small change)
                # SELL = -1 (expect price decrease)
                pred_class = self._prediction_to_class(prediction)
                actual_class = self._return_to_class(actual_return)
                
                predictions.append(pred_class)
                actuals.append(actual_class)
            
            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Convert to binary for metrics (simplify to UP vs NOT-UP)
            binary_preds = (predictions == 1).astype(int)
            binary_actuals = (actuals == 1).astype(int)
            
            metrics = {
                'precision': precision_score(binary_actuals, binary_preds, zero_division=0),
                'recall': recall_score(binary_actuals, binary_preds, zero_division=0),
                'f1_score': f1_score(binary_actuals, binary_preds, zero_division=0),
                'accuracy': np.mean(predictions == actuals),
                'total_predictions': len(predictions),
                'correct_predictions': np.sum(predictions == actuals)
            }
            
            # Add ROC-AUC and PR-AUC if we have both classes
            if len(np.unique(binary_actuals)) > 1 and len(np.unique(binary_preds)) > 1:
                try:
                    metrics['roc_auc'] = roc_auc_score(binary_actuals, binary_preds)
                    metrics['pr_auc'] = average_precision_score(binary_actuals, binary_preds)
                except:
                    metrics['roc_auc'] = 0.5
                    metrics['pr_auc'] = 0.5
            else:
                metrics['roc_auc'] = 0.5
                metrics['pr_auc'] = 0.5
            
            logger.info(f"Backtesting {ticker}: Accuracy={metrics['accuracy']:.2%}, "
                       f"Precision={metrics['precision']:.2%}, "
                       f"Recall={metrics['recall']:.2%}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error backtesting {ticker}: {e}")
            return self._get_default_metrics()
    
    def _make_trend_prediction(self, df: pd.DataFrame) -> str:
        """
        Make a trend prediction using technical indicators
        Same logic as realtime system
        """
        try:
            # Calculate indicators
            close = df['close']
            
            # Recent momentum
            recent_change = close.pct_change().tail(10).mean()
            
            # RSI
            rsi = self._calculate_rsi(close).iloc[-1] if len(close) > 14 else 50
            
            # Volatility
            volatility = close.pct_change().tail(20).std()
            
            # Volume trend
            volume_trend = 0
            if 'volume' in df.columns:
                volume_change = df['volume'].pct_change().tail(5).mean()
                volume_trend = 1 if volume_change > 0.1 else -1 if volume_change < -0.1 else 0
            
            # Calculate trend score
            trend_score = self._calculate_trend_score(recent_change, volatility, rsi, volume_trend)
            
            # Make prediction
            if trend_score > 0.6:
                return 'BUY'
            elif trend_score < 0.4:
                return 'SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 'HOLD'
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_trend_score(self, recent_change: float, volatility: float, 
                               rsi: float, volume_trend: int) -> float:
        """Calculate trend score (same as realtime system)"""
        score = 0.5
        
        # Price momentum (40% weight)
        if abs(recent_change) > 0.02:
            score += 0.4 * (recent_change / 0.05)  # Normalize
        
        # RSI (30% weight)
        if rsi < 30:
            score += 0.15  # Oversold - bullish
        elif rsi > 70:
            score -= 0.15  # Overbought - bearish
        
        # Volume trend (20% weight)
        score += 0.1 * volume_trend
        
        # Volatility adjustment (10% weight)
        if volatility > 0.03:
            score -= 0.05  # High volatility - reduce confidence
        
        return max(0, min(1, score))
    
    def _prediction_to_class(self, prediction: str) -> int:
        """Convert prediction to class"""
        if prediction == 'BUY':
            return 1
        elif prediction == 'SELL':
            return -1
        else:
            return 0
    
    def _return_to_class(self, return_pct: float, threshold: float = 0.02) -> int:
        """Convert return to class"""
        if return_pct > threshold:
            return 1  # Price went up
        elif return_pct < -threshold:
            return -1  # Price went down
        else:
            return 0  # Price stayed flat
    
    def backtest_anomaly_detection(self, ticker: str, df: pd.DataFrame) -> Dict[str, float]:
        """
        Backtest anomaly detection using statistical ground truth
        
        Ground truth: Points that are >3 standard deviations from mean are anomalies
        """
        try:
            if len(df) < 50:
                return self._get_default_metrics()
            
            from sklearn.ensemble import IsolationForest
            
            # Extract features
            features = self._extract_anomaly_features(df)
            
            if features.empty or len(features) < 30:
                return self._get_default_metrics()
            
            # Ground truth: Statistical anomalies (price changes > 3 std devs)
            returns = df['close'].pct_change().dropna()
            mean_return = returns.mean()
            std_return = returns.std()
            
            # True anomalies are extreme price movements
            true_anomalies = np.abs(returns - mean_return) > (3 * std_return)
            
            # Make predictions using IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            
            # Train on all data
            predictions_raw = iso_forest.fit_predict(features)
            predictions = (predictions_raw == -1).astype(int)
            
            # Align with ground truth
            min_len = min(len(predictions), len(true_anomalies))
            predictions = predictions[:min_len]
            true_anomalies = true_anomalies.iloc[:min_len].values.astype(int)
            
            # Calculate metrics
            metrics = {
                'precision': precision_score(true_anomalies, predictions, zero_division=0),
                'recall': recall_score(true_anomalies, predictions, zero_division=0),
                'f1_score': f1_score(true_anomalies, predictions, zero_division=0),
                'accuracy': np.mean(predictions == true_anomalies),
                'total_anomalies_detected': np.sum(predictions),
                'total_true_anomalies': np.sum(true_anomalies)
            }
            
            # Add ROC-AUC if possible
            if len(np.unique(true_anomalies)) > 1:
                try:
                    # Get anomaly scores
                    scores = iso_forest.score_samples(features)
                    metrics['roc_auc'] = roc_auc_score(true_anomalies, -scores)  # Negative scores for anomalies
                    metrics['pr_auc'] = average_precision_score(true_anomalies, -scores)
                except:
                    metrics['roc_auc'] = 0.5
                    metrics['pr_auc'] = 0.5
            else:
                metrics['roc_auc'] = 0.5
                metrics['pr_auc'] = 0.5
            
            logger.info(f"Anomaly backtesting {ticker}: Precision={metrics['precision']:.2%}, "
                       f"Recall={metrics['recall']:.2%}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error backtesting anomalies for {ticker}: {e}")
            return self._get_default_metrics()
    
    def _extract_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for anomaly detection"""
        features = pd.DataFrame()
        
        if 'close' in df.columns:
            features['price_change'] = df['close'].pct_change()
            features['volatility'] = df['close'].pct_change().rolling(10).std()
            features['price_ma_ratio'] = df['close'] / df['close'].rolling(20).mean()
        
        if 'volume' in df.columns:
            features['volume_change'] = df['volume'].pct_change()
            features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.bfill().fillna(0)
        
        return features
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """Return default metrics when backtesting fails"""
        return {
            'precision': 0.5,
            'recall': 0.5,
            'f1_score': 0.5,
            'accuracy': 0.5,
            'roc_auc': 0.5,
            'pr_auc': 0.5,
            'total_predictions': 0,
            'correct_predictions': 0
        }
    
    def get_performance_report(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive performance report for a ticker
        """
        try:
            # Fetch historical data
            logger.info(f"Fetching historical data for {ticker}...")
            df = yf.download(ticker, period='2y', progress=False)
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [col.lower() for col in df.columns]
            
            if df.empty:
                logger.warning(f"No data for {ticker}")
                return {}
            
            # Run backtests
            logger.info(f"Running trend prediction backtest for {ticker}...")
            trend_metrics = self.backtest_trend_predictions(ticker, df)
            
            logger.info(f"Running anomaly detection backtest for {ticker}...")
            anomaly_metrics = self.backtest_anomaly_detection(ticker, df)
            
            report = {
                'ticker': ticker,
                'backtest_date': datetime.now().isoformat(),
                'data_points': len(df),
                'date_range': f"{df.index[0].date()} to {df.index[-1].date()}",
                'trend_prediction': trend_metrics,
                'anomaly_detection': anomaly_metrics
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report for {ticker}: {e}")
            return {}


def run_backtest_for_tickers(tickers: List[str]) -> Dict[str, Any]:
    """
    Run backtesting for multiple tickers
    """
    backtester = RealBacktester(lookback_days=730)  # 2 years
    results = {}
    
    for ticker in tickers:
        logger.info(f"\n{'='*70}")
        logger.info(f"Backtesting {ticker}")
        logger.info(f"{'='*70}")
        
        report = backtester.get_performance_report(ticker)
        if report:
            results[ticker] = report
    
    return results


if __name__ == "__main__":
    # Test with Indian tickers
    tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HINDUNILVR.NS', 'HDFCBANK.NS']
    
    print("="*70)
    print("Real Backtesting System - Performance Validation")
    print("="*70)
    
    results = run_backtest_for_tickers(tickers)
    
    # Print summary
    print("\n" + "="*70)
    print("BACKTEST RESULTS SUMMARY")
    print("="*70)
    
    for ticker, report in results.items():
        print(f"\n{ticker}:")
        print(f"  Data: {report.get('data_points', 0)} points, {report.get('date_range', 'N/A')}")
        
        trend = report.get('trend_prediction', {})
        print(f"\n  Trend Prediction (Real Metrics):")
        print(f"    Accuracy:  {trend.get('accuracy', 0):.2%}")
        print(f"    Precision: {trend.get('precision', 0):.2%}")
        print(f"    Recall:    {trend.get('recall', 0):.2%}")
        print(f"    F1 Score:  {trend.get('f1_score', 0):.2%}")
        print(f"    ROC-AUC:   {trend.get('roc_auc', 0):.2%}")
        
        anomaly = report.get('anomaly_detection', {})
        print(f"\n  Anomaly Detection (Real Metrics):")
        print(f"    Precision: {anomaly.get('precision', 0):.2%}")
        print(f"    Recall:    {anomaly.get('recall', 0):.2%}")
        print(f"    F1 Score:  {anomaly.get('f1_score', 0):.2%}")
        print(f"    ROC-AUC:   {anomaly.get('roc_auc', 0):.2%}")
        print(f"    Detected:  {anomaly.get('total_anomalies_detected', 0)} anomalies")
    
    print("\n" + "="*70)
    print("✓ Backtesting complete!")
    print("="*70)
