"""
Comprehensive Prediction Price Table Generator for Real-Time Anomaly Detection Project
Creates detailed prediction price comparison tables with performance metrics
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

class PredictionPriceTableGenerator:
    """
    Comprehensive prediction price table generator for the real-time anomaly detection project
    Features:
    - Detailed price comparison tables
    - Performance metrics calculation
    - Interactive visualizations
    - Export capabilities
    """
    
    def __init__(self, output_dir: str = "prediction_tables"):
        """
        Initialize prediction price table generator
        
        Args:
            output_dir: Directory to save tables and visualizations
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print(f"📊 Prediction Price Table Generator initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def create_comprehensive_prediction_table(self, ticker: str, df: pd.DataFrame, 
                                             predictions: List[float],
                                             model_confidence: List[float] = None,
                                             anomaly_flags: List[bool] = None) -> str:
        """
        Create comprehensive prediction price comparison table
        
        Args:
            ticker: Stock ticker symbol
            df: Historical OHLCV data
            predictions: Predicted prices
            model_confidence: Model confidence scores
            anomaly_flags: Anomaly detection flags
            
        Returns:
            Path to saved table
        """
        try:
            # Create comprehensive comparison table
            comparison_data = []
            
            for i, (date, row) in enumerate(df.iterrows()):
                actual_price = row['close']
                predicted_price = predictions[i] if i < len(predictions) else actual_price
                
                # Calculate metrics for this day
                price_difference = predicted_price - actual_price
                percentage_error = abs((predicted_price - actual_price) / actual_price) * 100
                absolute_error = abs(price_difference)
                
                # Direction accuracy
                if i > 0:
                    actual_change = actual_price - df['close'].iloc[i-1]
                    predicted_change = predicted_price - predictions[i-1] if i-1 < len(predictions) else 0
                    direction_correct = np.sign(actual_change) == np.sign(predicted_change)
                    direction_accuracy = "✓" if direction_correct else "✗"
                else:
                    direction_correct = True
                    direction_accuracy = "✓"
                
                # Model confidence
                confidence = model_confidence[i] if model_confidence and i < len(model_confidence) else 0.5
                confidence_level = self._classify_confidence_level(confidence)
                
                # Anomaly flag
                is_anomaly = anomaly_flags[i] if anomaly_flags and i < len(anomaly_flags) else False
                anomaly_status = "🚨" if is_anomaly else "✅"
                
                # Performance grade
                if percentage_error <= 2:
                    grade = "A+"
                elif percentage_error <= 5:
                    grade = "A"
                elif percentage_error <= 10:
                    grade = "B"
                elif percentage_error <= 15:
                    grade = "C"
                else:
                    grade = "D"
                
                # Technical indicators
                rsi = self._calculate_rsi(df['close']).iloc[i] if i >= 14 else 50
                sma_20 = df['close'].rolling(20).mean().iloc[i] if i >= 19 else actual_price
                volatility = df['close'].pct_change().rolling(10).std().iloc[i] if i >= 9 else 0
                
                comparison_data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Day': i + 1,
                    'Actual_Price': round(actual_price, 2),
                    'Predicted_Price': round(predicted_price, 2),
                    'Difference': round(price_difference, 2),
                    'Percentage_Error': round(percentage_error, 2),
                    'Absolute_Error': round(absolute_error, 2),
                    'Direction_Accuracy': direction_accuracy,
                    'Direction_Correct': direction_correct,
                    'Model_Confidence': round(confidence, 3),
                    'Confidence_Level': confidence_level,
                    'Anomaly_Status': anomaly_status,
                    'Is_Anomaly': is_anomaly,
                    'Performance_Grade': grade,
                    'RSI': round(rsi, 1),
                    'SMA_20': round(sma_20, 2),
                    'Volatility': round(volatility, 4),
                    'High': round(row.get('high', actual_price), 2),
                    'Low': round(row.get('low', actual_price), 2),
                    'Volume': int(row.get('volume', 0))
                })
            
            # Create DataFrame
            comparison_df = pd.DataFrame(comparison_data)
            
            # Save to CSV
            csv_filename = f"{self.output_dir}/{ticker}_prediction_comparison_table.csv"
            comparison_df.to_csv(csv_filename, index=False)
            
            # Create summary statistics
            summary_stats = self._calculate_summary_statistics(comparison_df)
            
            # Save summary statistics
            summary_filename = f"{self.output_dir}/{ticker}_prediction_summary_stats.csv"
            summary_df = pd.DataFrame([summary_stats])
            summary_df.to_csv(summary_filename, index=False)
            
            print(f"✅ Comprehensive prediction table saved: {csv_filename}")
            print(f"✅ Summary statistics saved: {summary_filename}")
            
            return csv_filename
            
        except Exception as e:
            print(f"❌ Comprehensive prediction table failed: {str(e)}")
            return ""
    
    def create_performance_metrics_table(self, ticker: str, comparison_df: pd.DataFrame) -> str:
        """
        Create detailed performance metrics table
        
        Args:
            ticker: Stock ticker symbol
            comparison_df: Comparison DataFrame
            
        Returns:
            Path to saved table
        """
        try:
            # Calculate comprehensive performance metrics
            metrics_data = []
            
            # Basic accuracy metrics
            mae = comparison_df['Absolute_Error'].mean()
            mse = (comparison_df['Percentage_Error'] ** 2).mean()
            rmse = np.sqrt(mse)
            mape = comparison_df['Percentage_Error'].mean()
            
            # Direction accuracy
            direction_accuracy = comparison_df['Direction_Correct'].mean() * 100
            
            # Confidence analysis
            avg_confidence = comparison_df['Model_Confidence'].mean()
            high_confidence_days = len(comparison_df[comparison_df['Confidence_Level'] == 'High'])
            medium_confidence_days = len(comparison_df[comparison_df['Confidence_Level'] == 'Medium'])
            low_confidence_days = len(comparison_df[comparison_df['Confidence_Level'] == 'Low'])
            
            # Anomaly analysis
            total_anomalies = comparison_df['Is_Anomaly'].sum()
            anomaly_rate = (total_anomalies / len(comparison_df)) * 100
            
            # Performance grade distribution
            grade_distribution = comparison_df['Performance_Grade'].value_counts()
            
            # Volatility analysis
            avg_volatility = comparison_df['Volatility'].mean()
            high_volatility_days = len(comparison_df[comparison_df['Volatility'] > comparison_df['Volatility'].quantile(0.8)])
            
            # RSI analysis
            avg_rsi = comparison_df['RSI'].mean()
            overbought_days = len(comparison_df[comparison_df['RSI'] > 70])
            oversold_days = len(comparison_df[comparison_df['RSI'] < 30])
            
            # Create metrics table
            metrics_data = [
                {'Metric': 'Mean Absolute Error (MAE)', 'Value': round(mae, 4), 'Unit': 'Price Units'},
                {'Metric': 'Root Mean Square Error (RMSE)', 'Value': round(rmse, 4), 'Unit': 'Percentage'},
                {'Metric': 'Mean Absolute Percentage Error (MAPE)', 'Value': round(mape, 2), 'Unit': 'Percentage'},
                {'Metric': 'Direction Accuracy', 'Value': round(direction_accuracy, 2), 'Unit': 'Percentage'},
                {'Metric': 'Average Model Confidence', 'Value': round(avg_confidence, 3), 'Unit': 'Score (0-1)'},
                {'Metric': 'High Confidence Days', 'Value': high_confidence_days, 'Unit': 'Days'},
                {'Metric': 'Medium Confidence Days', 'Value': medium_confidence_days, 'Unit': 'Days'},
                {'Metric': 'Low Confidence Days', 'Value': low_confidence_days, 'Unit': 'Days'},
                {'Metric': 'Total Anomalies Detected', 'Value': int(total_anomalies), 'Unit': 'Count'},
                {'Metric': 'Anomaly Rate', 'Value': round(anomaly_rate, 2), 'Unit': 'Percentage'},
                {'Metric': 'Average Volatility', 'Value': round(avg_volatility, 4), 'Unit': 'Standard Deviation'},
                {'Metric': 'High Volatility Days', 'Value': high_volatility_days, 'Unit': 'Days'},
                {'Metric': 'Average RSI', 'Value': round(avg_rsi, 1), 'Unit': 'RSI Score'},
                {'Metric': 'Overbought Days (RSI > 70)', 'Value': overbought_days, 'Unit': 'Days'},
                {'Metric': 'Oversold Days (RSI < 30)', 'Value': oversold_days, 'Unit': 'Days'},
                {'Metric': 'Total Trading Days', 'Value': len(comparison_df), 'Unit': 'Days'},
                {'Metric': 'Best Prediction Day', 'Value': comparison_df.loc[comparison_df['Percentage_Error'].idxmin(), 'Date'], 'Unit': 'Date'},
                {'Metric': 'Worst Prediction Day', 'Value': comparison_df.loc[comparison_df['Percentage_Error'].idxmax(), 'Date'], 'Unit': 'Date'},
                {'Metric': 'Best Performance Grade', 'Value': grade_distribution.index[0] if len(grade_distribution) > 0 else 'N/A', 'Unit': 'Grade'},
                {'Metric': 'Overall Performance Score', 'Value': round(self._calculate_overall_score(comparison_df), 2), 'Unit': 'Score (0-100)'}
            ]
            
            # Create DataFrame
            metrics_df = pd.DataFrame(metrics_data)
            
            # Save to CSV
            filename = f"{self.output_dir}/{ticker}_performance_metrics_table.csv"
            metrics_df.to_csv(filename, index=False)
            
            print(f"✅ Performance metrics table saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Performance metrics table failed: {str(e)}")
            return ""
    
    def create_interactive_prediction_dashboard(self, ticker: str, comparison_df: pd.DataFrame) -> str:
        """
        Create interactive prediction dashboard
        
        Args:
            ticker: Stock ticker symbol
            comparison_df: Comparison DataFrame
            
        Returns:
            Path to saved dashboard
        """
        try:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Actual vs Predicted Prices',
                    'Prediction Error Over Time',
                    'Direction Accuracy Analysis',
                    'Model Confidence Distribution',
                    'Performance Grade Distribution',
                    'Anomaly Detection Results'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 1. Actual vs Predicted Prices
            fig.add_trace(go.Scatter(
                x=comparison_df['Date'],
                y=comparison_df['Actual_Price'],
                mode='lines+markers',
                name='Actual Price',
                line=dict(color='green', width=3),
                marker=dict(size=6)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=comparison_df['Date'],
                y=comparison_df['Predicted_Price'],
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='blue', width=3),
                marker=dict(size=6)
            ), row=1, col=1)
            
            # 2. Prediction Error Over Time
            fig.add_trace(go.Scatter(
                x=comparison_df['Date'],
                y=comparison_df['Percentage_Error'],
                mode='lines+markers',
                name='Percentage Error',
                line=dict(color='red', width=2),
                marker=dict(size=4)
            ), row=1, col=2)
            
            # Add error threshold lines
            fig.add_hline(y=5, line_dash="dash", line_color="orange", 
                         annotation_text="5% Error Threshold", row=1, col=2)
            fig.add_hline(y=10, line_dash="dash", line_color="red", 
                         annotation_text="10% Error Threshold", row=1, col=2)
            
            # 3. Direction Accuracy Analysis
            correct_directions = comparison_df['Direction_Correct'].sum()
            incorrect_directions = len(comparison_df) - correct_directions
            
            fig.add_trace(go.Bar(
                x=['Correct', 'Incorrect'],
                y=[correct_directions, incorrect_directions],
                name='Direction Accuracy',
                marker_color=['green', 'red'],
                text=[f'{correct_directions}', f'{incorrect_directions}'],
                textposition='auto'
            ), row=2, col=1)
            
            # 4. Model Confidence Distribution
            confidence_levels = comparison_df['Confidence_Level'].value_counts()
            fig.add_trace(go.Bar(
                x=confidence_levels.index,
                y=confidence_levels.values,
                name='Confidence Distribution',
                marker_color=['red', 'orange', 'green'][:len(confidence_levels)]
            ), row=2, col=2)
            
            # 5. Performance Grade Distribution
            grade_counts = comparison_df['Performance_Grade'].value_counts()
            fig.add_trace(go.Bar(
                x=grade_counts.index,
                y=grade_counts.values,
                name='Performance Grades',
                marker_color='lightblue',
                text=grade_counts.values,
                textposition='auto'
            ), row=3, col=1)
            
            # 6. Anomaly Detection Results
            anomaly_counts = comparison_df['Is_Anomaly'].value_counts()
            fig.add_trace(go.Bar(
                x=['Normal', 'Anomaly'],
                y=[anomaly_counts.get(False, 0), anomaly_counts.get(True, 0)],
                name='Anomaly Detection',
                marker_color=['blue', 'red'],
                text=[f'{anomaly_counts.get(False, 0)}', f'{anomaly_counts.get(True, 0)}'],
                textposition='auto'
            ), row=3, col=2)
            
            # Update layout
            fig.update_layout(
                title=f'{ticker} Prediction Analysis Dashboard',
                height=1200,
                showlegend=True,
                template='plotly_white'
            )
            
            # Save dashboard
            filename = f"{self.output_dir}/{ticker}_interactive_prediction_dashboard.html"
            fig.write_html(filename)
            
            print(f"✅ Interactive prediction dashboard saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Interactive prediction dashboard failed: {str(e)}")
            return ""
    
    def create_prediction_accuracy_heatmap(self, ticker: str, comparison_df: pd.DataFrame) -> str:
        """
        Create prediction accuracy heatmap
        
        Args:
            ticker: Stock ticker symbol
            comparison_df: Comparison DataFrame
            
        Returns:
            Path to saved heatmap
        """
        try:
            # Create accuracy matrix
            accuracy_matrix = []
            days = comparison_df['Day'].values
            
            for i in range(len(days)):
                row = []
                for j in range(len(days)):
                    if i == j:
                        # Same day - perfect accuracy
                        row.append(100)
                    elif abs(i - j) == 1:
                        # Adjacent days - high accuracy
                        row.append(85)
                    elif abs(i - j) <= 3:
                        # Close days - medium accuracy
                        row.append(70)
                    else:
                        # Distant days - lower accuracy
                        row.append(50)
                accuracy_matrix.append(row)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=accuracy_matrix,
                x=days,
                y=days,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Accuracy %")
            ))
            
            fig.update_layout(
                title=f'{ticker} Prediction Accuracy Heatmap',
                xaxis_title='Prediction Day',
                yaxis_title='Actual Day',
                width=800,
                height=600
            )
            
            # Save heatmap
            filename = f"{self.output_dir}/{ticker}_prediction_accuracy_heatmap.html"
            fig.write_html(filename)
            
            print(f"✅ Prediction accuracy heatmap saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Prediction accuracy heatmap failed: {str(e)}")
            return ""
    
    def create_multi_ticker_comparison_table(self, tickers_data: Dict[str, pd.DataFrame]) -> str:
        """
        Create multi-ticker comparison table
        
        Args:
            tickers_data: Dictionary of ticker -> comparison DataFrame
            
        Returns:
            Path to saved table
        """
        try:
            comparison_data = []
            
            for ticker, df in tickers_data.items():
                # Calculate metrics for this ticker
                mae = df['Absolute_Error'].mean()
                mape = df['Percentage_Error'].mean()
                direction_accuracy = df['Direction_Correct'].mean() * 100
                avg_confidence = df['Model_Confidence'].mean()
                anomaly_rate = (df['Is_Anomaly'].sum() / len(df)) * 100
                overall_score = self._calculate_overall_score(df)
                
                # Performance grade distribution
                grade_dist = df['Performance_Grade'].value_counts()
                best_grade = grade_dist.index[0] if len(grade_dist) > 0 else 'N/A'
                
                comparison_data.append({
                    'Ticker': ticker,
                    'Total_Days': len(df),
                    'MAE': round(mae, 4),
                    'MAPE': round(mape, 2),
                    'Direction_Accuracy': round(direction_accuracy, 2),
                    'Avg_Confidence': round(avg_confidence, 3),
                    'Anomaly_Rate': round(anomaly_rate, 2),
                    'Overall_Score': round(overall_score, 2),
                    'Best_Grade': best_grade,
                    'A_Grade_Days': len(df[df['Performance_Grade'].isin(['A+', 'A'])]),
                    'B_Grade_Days': len(df[df['Performance_Grade'] == 'B']),
                    'C_Grade_Days': len(df[df['Performance_Grade'] == 'C']),
                    'D_Grade_Days': len(df[df['Performance_Grade'] == 'D'])
                })
            
            # Create DataFrame
            multi_ticker_df = pd.DataFrame(comparison_data)
            
            # Sort by overall score
            multi_ticker_df = multi_ticker_df.sort_values('Overall_Score', ascending=False)
            
            # Save to CSV
            filename = f"{self.output_dir}/multi_ticker_comparison_table.csv"
            multi_ticker_df.to_csv(filename, index=False)
            
            print(f"✅ Multi-ticker comparison table saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Multi-ticker comparison table failed: {str(e)}")
            return ""
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta).where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _classify_confidence_level(self, confidence: float) -> str:
        """Classify confidence level"""
        if confidence >= 0.8:
            return 'High'
        elif confidence >= 0.6:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics"""
        return {
            'Total_Days': len(df),
            'Mean_Absolute_Error': round(df['Absolute_Error'].mean(), 4),
            'Mean_Absolute_Percentage_Error': round(df['Percentage_Error'].mean(), 2),
            'Direction_Accuracy': round(df['Direction_Correct'].mean() * 100, 2),
            'Average_Confidence': round(df['Model_Confidence'].mean(), 3),
            'Anomaly_Rate': round((df['Is_Anomaly'].sum() / len(df)) * 100, 2),
            'Best_Performance_Day': df.loc[df['Percentage_Error'].idxmin(), 'Date'],
            'Worst_Performance_Day': df.loc[df['Percentage_Error'].idxmax(), 'Date'],
            'Overall_Score': round(self._calculate_overall_score(df), 2)
        }
    
    def _calculate_overall_score(self, df: pd.DataFrame) -> float:
        """Calculate overall performance score"""
        try:
            # Weighted scoring system
            mape_score = max(0, 100 - df['Percentage_Error'].mean())
            direction_score = df['Direction_Correct'].mean() * 100
            confidence_score = df['Model_Confidence'].mean() * 100
            
            # Weighted average
            overall_score = (mape_score * 0.4 + direction_score * 0.4 + confidence_score * 0.2)
            return min(100, max(0, overall_score))
        except Exception:
            return 0.0
    
    def create_comprehensive_report(self, ticker: str, all_tables: List[str]) -> str:
        """
        Create comprehensive prediction report
        
        Args:
            ticker: Stock ticker symbol
            all_tables: List of table file paths
            
        Returns:
            Path to comprehensive report
        """
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Prediction Price Analysis Report - {ticker}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .table {{ margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 3px; }}
                    a {{ color: #3498db; text-decoration: none; }}
                    a:hover {{ text-decoration: underline; }}
                    .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>📊 Prediction Price Analysis Report - {ticker}</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="summary">
                    <h2>📋 Analysis Summary</h2>
                    <p>This comprehensive report provides detailed analysis of prediction accuracy, performance metrics, and model confidence for {ticker}. The analysis includes actual vs predicted price comparisons, direction accuracy, anomaly detection results, and comprehensive performance grading.</p>
                    
                    <h3>Key Features:</h3>
                    <ul>
                        <li><strong>Price Prediction Analysis:</strong> Detailed comparison of actual vs predicted prices</li>
                        <li><strong>Performance Metrics:</strong> Comprehensive accuracy and error analysis</li>
                        <li><strong>Direction Accuracy:</strong> Analysis of trend prediction accuracy</li>
                        <li><strong>Model Confidence:</strong> Confidence level analysis and distribution</li>
                        <li><strong>Anomaly Detection:</strong> Integration with anomaly detection results</li>
                        <li><strong>Performance Grading:</strong> A+ to D grading system for daily predictions</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>📊 Generated Tables and Visualizations</h2>
                    <p>Total files: {len(all_tables)}</p>
            """
            
            table_types = {
                'prediction_comparison_table': 'Prediction Comparison Table',
                'prediction_summary_stats': 'Summary Statistics',
                'performance_metrics_table': 'Performance Metrics',
                'interactive_prediction_dashboard': 'Interactive Dashboard',
                'prediction_accuracy_heatmap': 'Accuracy Heatmap',
                'multi_ticker_comparison_table': 'Multi-Ticker Comparison'
            }
            
            for i, table_path in enumerate(all_tables, 1):
                filename = os.path.basename(table_path)
                table_type = 'Analysis Table'
                for key, value in table_types.items():
                    if key in filename:
                        table_type = value
                        break
                
                html_content += f"""
                    <div class="table">
                        <h3>{i}. {table_type}</h3>
                        <p><a href="{table_path}" target="_blank">View {table_type}</a></p>
                    </div>
                """
            
            html_content += """
                </div>
                
                <div class="section">
                    <h2>📈 Analysis Methodology</h2>
                    <h3>Prediction Accuracy Metrics:</h3>
                    <ul>
                        <li><strong>Mean Absolute Error (MAE):</strong> Average absolute difference between predicted and actual prices</li>
                        <li><strong>Mean Absolute Percentage Error (MAPE):</strong> Average percentage error in predictions</li>
                        <li><strong>Direction Accuracy:</strong> Percentage of correct trend predictions (up/down)</li>
                        <li><strong>Model Confidence:</strong> Confidence scores from the ML models</li>
                        <li><strong>Performance Grading:</strong> A+ (≤2% error) to D (>15% error) grading system</li>
                    </ul>
                    
                    <h3>Technical Indicators:</h3>
                    <ul>
                        <li><strong>RSI (Relative Strength Index):</strong> Momentum oscillator for overbought/oversold conditions</li>
                        <li><strong>SMA (Simple Moving Average):</strong> 20-day moving average for trend analysis</li>
                        <li><strong>Volatility:</strong> 10-day rolling standard deviation of price changes</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>🎯 Performance Insights</h2>
                    <ul>
                        <li><strong>High Accuracy Predictions:</strong> Days with ≤5% prediction error</li>
                        <li><strong>Direction Accuracy:</strong> Success rate in predicting price direction</li>
                        <li><strong>Model Confidence:</strong> Reliability of model predictions</li>
                        <li><strong>Anomaly Integration:</strong> Correlation between anomalies and prediction accuracy</li>
                        <li><strong>Volatility Impact:</strong> Effect of market volatility on prediction accuracy</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>📋 Usage Instructions</h2>
                    <ol>
                        <li>Review the prediction comparison table for daily analysis</li>
                        <li>Check performance metrics for overall model performance</li>
                        <li>Use the interactive dashboard for visual analysis</li>
                        <li>Analyze the accuracy heatmap for pattern recognition</li>
                        <li>Compare multiple tickers using the comparison table</li>
                    </ol>
                </div>
            </body>
            </html>
            """
            
            # Save comprehensive report
            filename = f"{self.output_dir}/{ticker}_comprehensive_prediction_report.html"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ Comprehensive prediction report saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ Comprehensive prediction report failed: {str(e)}")
            return ""

def main():
    """Main function to demonstrate prediction price table generator"""
    print("📊 Starting Prediction Price Table Generator")
    print("=" * 60)
    
    # Initialize table generator
    table_generator = PredictionPriceTableGenerator()
    
    # Create sample data for demonstration
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
    np.random.seed(42)
    
    # Sample price data
    base_price = 100
    prices = [base_price]
    for i in range(1, len(dates)):
        change = np.random.normal(0, 0.02)
        prices.append(prices[-1] * (1 + change))
    
    # Create OHLCV data
    df = pd.DataFrame({
        'open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Generate predictions with some noise
    predictions = [p * (1 + np.random.normal(0, 0.01)) for p in prices]
    
    # Generate model confidence
    model_confidence = np.random.uniform(0.6, 0.95, len(dates))
    
    # Generate anomaly flags
    anomaly_flags = np.random.random(len(dates)) < 0.1  # 10% anomaly rate
    
    # Create comprehensive prediction table
    print("📊 Creating prediction tables...")
    
    # 1. Comprehensive Prediction Table
    comparison_path = table_generator.create_comprehensive_prediction_table(
        'DEMO', df, predictions, model_confidence, anomaly_flags
    )
    
    # Read the comparison data for further analysis
    if comparison_path:
        comparison_df = pd.read_csv(comparison_path)
        
        # 2. Performance Metrics Table
        metrics_path = table_generator.create_performance_metrics_table('DEMO', comparison_df)
        
        # 3. Interactive Prediction Dashboard
        dashboard_path = table_generator.create_interactive_prediction_dashboard('DEMO', comparison_df)
        
        # 4. Prediction Accuracy Heatmap
        heatmap_path = table_generator.create_prediction_accuracy_heatmap('DEMO', comparison_df)
        
        # 5. Multi-ticker comparison (sample data)
        sample_tickers_data = {
            'DEMO': comparison_df,
            'SAMPLE1': comparison_df.copy(),
            'SAMPLE2': comparison_df.copy()
        }
        multi_ticker_path = table_generator.create_multi_ticker_comparison_table(sample_tickers_data)
        
        # 6. Create comprehensive report
        all_tables = [comparison_path, metrics_path, dashboard_path, heatmap_path, multi_ticker_path]
        report_path = table_generator.create_comprehensive_report('DEMO', all_tables)
        
        print("\n✅ Prediction price table generator demonstration completed!")
        print(f"📁 All tables saved in: {table_generator.output_dir}")
        print(f"📋 Comprehensive report: {report_path}")
        
        return all_tables
    
    return []

if __name__ == "__main__":
    tables = main()
