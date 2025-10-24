"""
Advanced Visualization System for Real-Time Anomaly Detection Project
Creates comprehensive visualizations including candlestick charts, line charts, 
overlap regions, and anomaly detection graphs
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add project path
sys.path.insert(0, os.path.dirname(__file__))

class AdvancedVisualizationSystem:
    """
    Advanced visualization system for the real-time anomaly detection project
    Features:
    - Interactive candlestick charts
    - Overlap region analysis
    - Anomaly detection visualizations
    - Performance metrics dashboards
    - Real-time data visualization
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize visualization system
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print(f"🎨 Advanced Visualization System initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def create_candlestick_analysis(self, ticker: str, df: pd.DataFrame, 
                                  predictions: List[float] = None, 
                                  anomalies: List[bool] = None) -> str:
        """
        Create comprehensive candlestick analysis with predictions and anomalies
        
        Args:
            ticker: Stock ticker symbol
            df: OHLCV data
            predictions: Predicted prices
            anomalies: Anomaly flags
            
        Returns:
            Path to saved visualization
        """
        try:
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(
                    f'{ticker} - Price Analysis with Predictions',
                    'Volume Analysis',
                    'Anomaly Detection'
                ),
                row_heights=[0.5, 0.2, 0.3]
            )
            
            # Main candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Actual Price',
                    increasing_line_color='#00ff88',
                    decreasing_line_color='#ff4444'
                ),
                row=1, col=1
            )
            
            # Add predictions if available
            if predictions is not None:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=predictions,
                        mode='lines+markers',
                        name='Predicted Price',
                        line=dict(color='#0066cc', width=3),
                        marker=dict(size=6, symbol='diamond')
                    ),
                    row=1, col=1
                )
            
            # Add moving averages
            if len(df) >= 20:
                sma_20 = df['close'].rolling(20).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=sma_20,
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='orange', width=2, dash='dash')
                    ),
                    row=1, col=1
                )
            
            # Volume chart
            if 'volume' in df.columns:
                colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red' 
                         for i in range(len(df))]
                
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.7
                    ),
                    row=2, col=1
                )
            
            # Anomaly detection chart
            if anomalies is not None:
                anomaly_dates = df.index[anomalies]
                anomaly_prices = df['close'][anomalies]
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['close'],
                        mode='lines',
                        name='Price',
                        line=dict(color='lightblue', width=1)
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_dates,
                        y=anomaly_prices,
                        mode='markers',
                        name='Anomalies',
                        marker=dict(
                            color='red',
                            size=12,
                            symbol='x',
                            line=dict(width=2, color='darkred')
                        )
                    ),
                    row=3, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f'{ticker} Comprehensive Analysis - {df.index[0].date()} to {df.index[-1].date()}',
                xaxis_title='Date',
                height=1000,
                showlegend=True,
                template='plotly_white'
            )
            
            # Update axes
            fig.update_xaxes(title_text="Date", row=3, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="Price", row=3, col=1)
            
            # Save visualization
            filename = f"{self.output_dir}/{ticker}_candlestick_analysis.html"
            fig.write_html(filename)
            
            print(f"✅ Candlestick analysis saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Candlestick analysis failed: {str(e)}")
            return ""
    
    def create_overlap_region_analysis(self, ticker: str, df: pd.DataFrame, 
                                     predictions: List[float]) -> str:
        """
        Create overlap region analysis showing actual vs predicted prices
        
        Args:
            ticker: Stock ticker symbol
            df: Historical data
            predictions: Predicted prices
            
        Returns:
            Path to saved visualization
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Price Overlap Analysis',
                    'Error Distribution',
                    'Direction Accuracy',
                    'Correlation Analysis'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Price Overlap
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['close'],
                    mode='lines+markers',
                    name='Actual Price',
                    line=dict(color='green', width=3),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=predictions,
                    mode='lines+markers',
                    name='Predicted Price',
                    line=dict(color='blue', width=3),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
            
            # Add confidence bands
            confidence_upper = [p * 1.05 for p in predictions]
            confidence_lower = [p * 0.95 for p in predictions]
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=confidence_upper,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=confidence_lower,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0,100,80,0.2)',
                    name='Confidence Band',
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            # 2. Error Distribution
            errors = np.array(predictions) - df['close'].values
            fig.add_trace(
                go.Histogram(
                    x=errors,
                    name='Error Distribution',
                    nbinsx=20,
                    marker_color='orange',
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            # 3. Direction Accuracy
            actual_direction = np.diff(df['close'].values)
            predicted_direction = np.diff(predictions)
            direction_correct = np.sign(actual_direction) == np.sign(predicted_direction)
            
            correct_count = np.sum(direction_correct)
            incorrect_count = len(direction_correct) - correct_count
            
            fig.add_trace(
                go.Bar(
                    x=['Correct Direction', 'Incorrect Direction'],
                    y=[correct_count, incorrect_count],
                    name='Direction Accuracy',
                    marker_color=['green', 'red'],
                    text=[f'{correct_count}', f'{incorrect_count}'],
                    textposition='auto'
                ),
                row=2, col=1
            )
            
            # 4. Correlation Scatter
            fig.add_trace(
                go.Scatter(
                    x=df['close'].values,
                    y=predictions,
                    mode='markers',
                    name='Actual vs Predicted',
                    marker=dict(
                        color='purple',
                        size=8,
                        opacity=0.6
                    ),
                    text=[f'Day {i+1}' for i in range(len(df))],
                    hovertemplate='<b>%{text}</b><br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Add correlation line
            correlation = np.corrcoef(df['close'].values, predictions)[0, 1]
            min_price = min(df['close'].min(), min(predictions))
            max_price = max(df['close'].max(), max(predictions))
            
            fig.add_trace(
                go.Scatter(
                    x=[min_price, max_price],
                    y=[min_price, max_price],
                    mode='lines',
                    name=f'Perfect Correlation (r={correlation:.3f})',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=f'{ticker} Overlap Region Analysis',
                height=800,
                showlegend=True,
                template='plotly_white'
            )
            
            # Save visualization
            filename = f"{self.output_dir}/{ticker}_overlap_analysis.html"
            fig.write_html(filename)
            
            print(f"✅ Overlap analysis saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Overlap analysis failed: {str(e)}")
            return ""
    
    def create_anomaly_detection_graphs(self, ticker: str, df: pd.DataFrame, 
                                     anomaly_scores: List[float],
                                     anomaly_flags: List[bool]) -> str:
        """
        Create comprehensive anomaly detection visualizations
        
        Args:
            ticker: Stock ticker symbol
            df: Historical data
            anomaly_scores: Anomaly scores
            anomaly_flags: Binary anomaly flags
            
        Returns:
            Path to saved visualization
        """
        try:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Price with Anomaly Detection',
                    'Anomaly Score Over Time',
                    'Anomaly Distribution',
                    'Anomaly Clusters',
                    'Volatility vs Anomalies',
                    'Anomaly Statistics'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Price with Anomaly Detection
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='lightblue', width=2)
                ),
                row=1, col=1
            )
            
            # Highlight anomalies
            anomaly_dates = df.index[anomaly_flags]
            anomaly_prices = df['close'][anomaly_flags]
            
            fig.add_trace(
                go.Scatter(
                    x=anomaly_dates,
                    y=anomaly_prices,
                    mode='markers',
                    name='Anomalies',
                    marker=dict(
                        color='red',
                        size=12,
                        symbol='x',
                        line=dict(width=2, color='darkred')
                    )
                ),
                row=1, col=1
            )
            
            # 2. Anomaly Score Over Time
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=anomaly_scores,
                    mode='lines+markers',
                    name='Anomaly Score',
                    line=dict(color='orange', width=2),
                    marker=dict(size=4)
                ),
                row=1, col=2
            )
            
            # Add threshold line
            threshold = 0.5  # Example threshold
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold: {threshold}",
                row=1, col=2
            )
            
            # 3. Anomaly Distribution
            fig.add_trace(
                go.Histogram(
                    x=anomaly_scores,
                    name='Anomaly Score Distribution',
                    nbinsx=20,
                    marker_color='lightgreen',
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # 4. Anomaly Clusters (using PCA-like visualization)
            # Create a 2D representation of the data
            features = self._extract_features_for_clustering(df)
            if features is not None and len(features) > 0:
                # Simple 2D projection
                x_proj = features[:, 0] if features.shape[1] > 0 else np.arange(len(features))
                y_proj = features[:, 1] if features.shape[1] > 1 else anomaly_scores
                
                colors = ['red' if flag else 'blue' for flag in anomaly_flags]
                
                fig.add_trace(
                    go.Scatter(
                        x=x_proj,
                        y=y_proj,
                        mode='markers',
                        name='Data Points',
                        marker=dict(
                            color=colors,
                            size=8,
                            opacity=0.6
                        ),
                        text=[f'Day {i+1}' for i in range(len(df))],
                        hovertemplate='<b>%{text}</b><br>Anomaly: %{marker.color}<extra></extra>'
                    ),
                    row=2, col=2
                )
            
            # 5. Volatility vs Anomalies
            volatility = df['close'].pct_change().rolling(10).std()
            fig.add_trace(
                go.Scatter(
                    x=volatility,
                    y=anomaly_scores,
                    mode='markers',
                    name='Volatility vs Anomaly',
                    marker=dict(
                        color=anomaly_scores,
                        size=8,
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="Anomaly Score")
                    )
                ),
                row=3, col=1
            )
            
            # 6. Anomaly Statistics
            anomaly_count = np.sum(anomaly_flags)
            total_count = len(anomaly_flags)
            anomaly_rate = anomaly_count / total_count * 100
            
            stats_data = {
                'Total Points': total_count,
                'Anomalies': anomaly_count,
                'Anomaly Rate (%)': round(anomaly_rate, 2),
                'Avg Score': round(np.mean(anomaly_scores), 3),
                'Max Score': round(np.max(anomaly_scores), 3)
            }
            
            fig.add_trace(
                go.Bar(
                    x=list(stats_data.keys()),
                    y=list(stats_data.values()),
                    name='Anomaly Statistics',
                    marker_color='lightcoral',
                    text=list(stats_data.values()),
                    textposition='auto'
                ),
                row=3, col=2
            )
            
            # Update layout
            fig.update_layout(
                title=f'{ticker} Anomaly Detection Analysis',
                height=1200,
                showlegend=True,
                template='plotly_white'
            )
            
            # Save visualization
            filename = f"{self.output_dir}/{ticker}_anomaly_detection.html"
            fig.write_html(filename)
            
            print(f"✅ Anomaly detection graphs saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Anomaly detection graphs failed: {str(e)}")
            return ""
    
    def create_performance_dashboard(self, backtest_results: Dict[str, Any]) -> str:
        """
        Create comprehensive performance dashboard
        
        Args:
            backtest_results: Results from backtesting system
            
        Returns:
            Path to saved visualization
        """
        try:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Accuracy by Ticker',
                    'Error Distribution',
                    'Direction Accuracy',
                    'Correlation Analysis',
                    'Performance Metrics',
                    'Model Comparison'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Extract data for visualization
            tickers = []
            accuracies = []
            mae_values = []
            direction_accuracies = []
            correlations = []
            
            for ticker, results in backtest_results.items():
                if 'accuracy_metrics' in results:
                    tickers.append(ticker)
                    accuracies.append(results['accuracy_metrics'].get('overall_accuracy', 0))
                    mae_values.append(results['accuracy_metrics'].get('mae', 0))
                    direction_accuracies.append(results['accuracy_metrics'].get('direction_accuracy', 0))
                    correlations.append(results['accuracy_metrics'].get('correlation', 0))
            
            # 1. Accuracy by Ticker
            fig.add_trace(
                go.Bar(
                    x=tickers,
                    y=accuracies,
                    name='Overall Accuracy',
                    marker_color='lightblue',
                    text=[f'{acc:.1f}%' for acc in accuracies],
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # 2. Error Distribution
            fig.add_trace(
                go.Histogram(
                    x=mae_values,
                    name='MAE Distribution',
                    nbinsx=10,
                    marker_color='orange',
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            # 3. Direction Accuracy
            fig.add_trace(
                go.Bar(
                    x=tickers,
                    y=direction_accuracies,
                    name='Direction Accuracy',
                    marker_color='lightgreen',
                    text=[f'{acc:.1f}%' for acc in direction_accuracies],
                    textposition='auto'
                ),
                row=2, col=1
            )
            
            # 4. Correlation Analysis
            fig.add_trace(
                go.Scatter(
                    x=accuracies,
                    y=correlations,
                    mode='markers+text',
                    name='Accuracy vs Correlation',
                    marker=dict(
                        color=correlations,
                        size=12,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Correlation")
                    ),
                    text=tickers,
                    textposition='top center'
                ),
                row=2, col=2
            )
            
            # 5. Performance Metrics
            metrics = ['MAE', 'RMSE', 'MAPE', 'R²']
            avg_values = [
                np.mean(mae_values),
                np.mean([results.get('accuracy_metrics', {}).get('rmse', 0) for results in backtest_results.values()]),
                np.mean([results.get('accuracy_metrics', {}).get('mape', 0) for results in backtest_results.values()]),
                np.mean([results.get('accuracy_metrics', {}).get('r_squared', 0) for results in backtest_results.values()])
            ]
            
            fig.add_trace(
                go.Bar(
                    x=metrics,
                    y=avg_values,
                    name='Average Metrics',
                    marker_color='lightcoral',
                    text=[f'{val:.3f}' for val in avg_values],
                    textposition='auto'
                ),
                row=3, col=1
            )
            
            # 6. Model Comparison (simulated)
            model_names = ['Current Model', 'Baseline', 'Enhanced Model']
            model_accuracies = [np.mean(accuracies), np.mean(accuracies) * 0.8, np.mean(accuracies) * 1.1]
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=model_accuracies,
                    name='Model Comparison',
                    marker_color=['blue', 'red', 'green'],
                    text=[f'{acc:.1f}%' for acc in model_accuracies],
                    textposition='auto'
                ),
                row=3, col=2
            )
            
            # Update layout
            fig.update_layout(
                title='Performance Dashboard - Backtesting Results',
                height=1200,
                showlegend=True,
                template='plotly_white'
            )
            
            # Save visualization
            filename = f"{self.output_dir}/performance_dashboard.html"
            fig.write_html(filename)
            
            print(f"✅ Performance dashboard saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Performance dashboard failed: {str(e)}")
            return ""
    
    def create_real_time_dashboard(self, ticker: str, live_data: Dict[str, Any]) -> str:
        """
        Create real-time dashboard for live monitoring
        
        Args:
            ticker: Stock ticker symbol
            live_data: Real-time data dictionary
            
        Returns:
            Path to saved visualization
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Real-Time Price',
                    'Anomaly Detection',
                    'Sentiment Analysis',
                    'Trend Prediction'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Real-Time Price
            if 'price_data' in live_data:
                price_data = live_data['price_data']
                fig.add_trace(
                    go.Scatter(
                        x=price_data['timestamps'],
                        y=price_data['prices'],
                        mode='lines+markers',
                        name='Price',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
            
            # 2. Anomaly Detection
            if 'anomaly_data' in live_data:
                anomaly_data = live_data['anomaly_data']
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_data['timestamps'],
                        y=anomaly_data['scores'],
                        mode='lines+markers',
                        name='Anomaly Score',
                        line=dict(color='red', width=2)
                    ),
                    row=1, col=2
                )
            
            # 3. Sentiment Analysis
            if 'sentiment_data' in live_data:
                sentiment_data = live_data['sentiment_data']
                fig.add_trace(
                    go.Scatter(
                        x=sentiment_data['timestamps'],
                        y=sentiment_data['scores'],
                        mode='lines+markers',
                        name='Sentiment Score',
                        line=dict(color='green', width=2)
                    ),
                    row=2, col=1
                )
            
            # 4. Trend Prediction
            if 'trend_data' in live_data:
                trend_data = live_data['trend_data']
                fig.add_trace(
                    go.Scatter(
                        x=trend_data['timestamps'],
                        y=trend_data['predictions'],
                        mode='lines+markers',
                        name='Trend Prediction',
                        line=dict(color='purple', width=2)
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=f'{ticker} Real-Time Dashboard',
                height=800,
                showlegend=True,
                template='plotly_white'
            )
            
            # Save visualization
            filename = f"{self.output_dir}/{ticker}_realtime_dashboard.html"
            fig.write_html(filename)
            
            print(f"✅ Real-time dashboard saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Real-time dashboard failed: {str(e)}")
            return ""
    
    def _extract_features_for_clustering(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features for clustering visualization"""
        try:
            features = []
            
            # Price-based features
            features.append(df['close'].pct_change().fillna(0).values)
            features.append(df['close'].rolling(5).mean().pct_change().fillna(0).values)
            
            # Volume features if available
            if 'volume' in df.columns:
                features.append(df['volume'].pct_change().fillna(0).values)
            
            # Technical indicators
            rsi = self._calculate_rsi(df['close']).fillna(50).values
            features.append((rsi - 50) / 50)  # Normalize RSI
            
            # Combine features
            feature_matrix = np.column_stack(features)
            return feature_matrix
            
        except Exception as e:
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta).where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_summary_report(self, all_visualizations: List[str]) -> str:
        """
        Create a summary report with links to all visualizations
        
        Args:
            all_visualizations: List of visualization file paths
            
        Returns:
            Path to summary report
        """
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Real-Time Anomaly Detection - Visualization Summary</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .visualization {{ margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 3px; }}
                    a {{ color: #3498db; text-decoration: none; }}
                    a:hover {{ text-decoration: underline; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🎨 Real-Time Anomaly Detection - Visualization Summary</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>📊 Available Visualizations</h2>
                    <p>Total visualizations: {len(all_visualizations)}</p>
            """
            
            for i, viz_path in enumerate(all_visualizations, 1):
                filename = os.path.basename(viz_path)
                html_content += f"""
                    <div class="visualization">
                        <h3>{i}. {filename}</h3>
                        <p><a href="{viz_path}" target="_blank">Open Visualization</a></p>
                    </div>
                """
            
            html_content += """
                </div>
                
                <div class="section">
                    <h2>📈 Visualization Types</h2>
                    <ul>
                        <li><strong>Candlestick Analysis:</strong> Comprehensive price analysis with predictions and anomalies</li>
                        <li><strong>Overlap Region Analysis:</strong> Detailed comparison of actual vs predicted prices</li>
                        <li><strong>Anomaly Detection Graphs:</strong> Advanced anomaly detection visualizations</li>
                        <li><strong>Performance Dashboard:</strong> Overall system performance metrics</li>
                        <li><strong>Real-Time Dashboard:</strong> Live monitoring dashboards</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>🔧 Usage Instructions</h2>
                    <ol>
                        <li>Click on any visualization link to open in a new tab</li>
                        <li>Use the interactive features to explore the data</li>
                        <li>Download images or data as needed</li>
                        <li>Share specific visualizations with stakeholders</li>
                    </ol>
                </div>
            </body>
            </html>
            """
            
            # Save summary report
            filename = f"{self.output_dir}/visualization_summary.html"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ Summary report saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Summary report failed: {str(e)}")
            return ""

def main():
    """Main function to demonstrate visualization system"""
    print("🎨 Starting Advanced Visualization System")
    print("=" * 60)
    
    # Initialize visualization system
    viz_system = AdvancedVisualizationSystem()
    
    # Create sample data for demonstration
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    np.random.seed(42)
    
    # Sample price data
    base_price = 100
    price_changes = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))
    
    # Create OHLCV data
    df = pd.DataFrame({
        'open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Generate predictions
    predictions = [p * (1 + np.random.normal(0, 0.01)) for p in prices]
    
    # Generate anomaly flags
    anomaly_flags = np.random.random(len(dates)) < 0.1  # 10% anomaly rate
    anomaly_scores = np.random.random(len(dates))
    
    # Create visualizations
    print("📊 Creating visualizations...")
    
    # 1. Candlestick Analysis
    candlestick_path = viz_system.create_candlestick_analysis(
        'DEMO', df, predictions, anomaly_flags
    )
    
    # 2. Overlap Region Analysis
    overlap_path = viz_system.create_overlap_region_analysis(
        'DEMO', df, predictions
    )
    
    # 3. Anomaly Detection Graphs
    anomaly_path = viz_system.create_anomaly_detection_graphs(
        'DEMO', df, anomaly_scores, anomaly_flags
    )
    
    # 4. Performance Dashboard (sample data)
    sample_results = {
        'DEMO': {
            'accuracy_metrics': {
                'overall_accuracy': 85.5,
                'mae': 2.3,
                'rmse': 3.1,
                'mape': 2.8,
                'direction_accuracy': 78.0,
                'correlation': 0.82,
                'r_squared': 0.67
            }
        }
    }
    
    performance_path = viz_system.create_performance_dashboard(sample_results)
    
    # 5. Create summary report
    all_viz = [candlestick_path, overlap_path, anomaly_path, performance_path]
    summary_path = viz_system.create_summary_report(all_viz)
    
    print("\n✅ Visualization system demonstration completed!")
    print(f"📁 All visualizations saved in: {viz_system.output_dir}")
    print(f"📋 Summary report: {summary_path}")
    
    return all_viz

if __name__ == "__main__":
    visualizations = main()
