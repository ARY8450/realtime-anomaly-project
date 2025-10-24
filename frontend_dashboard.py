"""
Frontend Dashboard for Real-Time Anomaly Detection System
Creates a comprehensive web dashboard to display all analysis components
"""

import os
import sys
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class FrontendDashboard:
    """Frontend Dashboard for displaying all analysis components"""
    
    def __init__(self, data_dir: str = "comprehensive_analysis"):
        self.data_dir = data_dir
        self.output_dir = "frontend_dashboard"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_analysis_data(self):
        """Load all analysis data from CSV files"""
        try:
            # Load backtesting results
            backtesting_path = os.path.join(self.data_dir, "backtesting_results", "RELIANCE_NS_prediction_comparison_table.csv")
            if os.path.exists(backtesting_path):
                self.backtesting_df = pd.read_csv(backtesting_path)
            else:
                self.backtesting_df = None
                
            # Load summary stats
            summary_path = os.path.join(self.data_dir, "backtesting_results", "RELIANCE_NS_prediction_summary_stats.csv")
            if os.path.exists(summary_path):
                self.summary_df = pd.read_csv(summary_path)
            else:
                self.summary_df = None
                
            # Load performance metrics
            metrics_path = os.path.join(self.data_dir, "prediction_tables", "RELIANCE_NS_performance_metrics_table.csv")
            if os.path.exists(metrics_path):
                self.metrics_df = pd.read_csv(metrics_path)
            else:
                self.metrics_df = None
                
            print("✅ Analysis data loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading analysis data: {str(e)}")
            return False
    
    def create_main_dashboard(self):
        """Create the main dashboard HTML page"""
        print("Creating Main Dashboard...")
        
        # Load data
        if not self.load_analysis_data():
            return None
            
        # Create comprehensive dashboard HTML
        dashboard_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-Time Anomaly Detection Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header p {{
            color: #7f8c8d;
            font-size: 1.2em;
            margin-bottom: 20px;
        }}
        
        .timestamp {{
            background: #3498db;
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            display: inline-block;
            font-weight: 600;
        }}
        
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        }}
        
        .card-header {{
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #ecf0f1;
        }}
        
        .card-icon {{
            font-size: 2em;
            margin-right: 15px;
            color: #3498db;
        }}
        
        .card-title {{
            font-size: 1.5em;
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: scale(1.05);
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .chart-container {{
            height: 400px;
            margin: 20px 0;
            border-radius: 10px;
            overflow: hidden;
        }}
        
        .table-container {{
            max-height: 400px;
            overflow-y: auto;
            border-radius: 10px;
            border: 1px solid #ecf0f1;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
            position: sticky;
            top: 0;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .status-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .status-excellent {{ background: #d4edda; color: #155724; }}
        .status-good {{ background: #d1ecf1; color: #0c5460; }}
        .status-warning {{ background: #fff3cd; color: #856404; }}
        .status-danger {{ background: #f8d7da; color: #721c24; }}
        
        .full-width {{
            grid-column: 1 / -1;
        }}
        
        .tabs {{
            display: flex;
            margin-bottom: 20px;
            border-bottom: 2px solid #ecf0f1;
        }}
        
        .tab {{
            padding: 12px 24px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
            font-weight: 600;
        }}
        
        .tab:hover {{
            background: #f8f9fa;
        }}
        
        .tab.active {{
            border-bottom-color: #3498db;
            color: #3498db;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .loading {{
            text-align: center;
            padding: 40px;
            color: #7f8c8d;
        }}
        
        .error {{
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        
        @media (max-width: 768px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .metrics-grid {{
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-chart-line"></i> Real-Time Anomaly Detection Dashboard</h1>
            <p>Comprehensive Analysis & Visualization Platform</p>
            <div class="timestamp">
                <i class="fas fa-clock"></i> Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        
        <div class="dashboard-grid">
            <!-- Key Metrics Overview -->
            <div class="card full-width">
                <div class="card-header">
                    <i class="fas fa-tachometer-alt card-icon"></i>
                    <h2 class="card-title">Key Performance Metrics</h2>
                </div>
                <div class="metrics-grid" id="metricsGrid">
                    <!-- Metrics will be populated by JavaScript -->
                </div>
            </div>
            
            <!-- Backtesting Results -->
            <div class="card">
                <div class="card-header">
                    <i class="fas fa-chart-bar card-icon"></i>
                    <h2 class="card-title">Backtesting Results</h2>
                </div>
                <div class="tabs">
                    <div class="tab active" onclick="showTab('backtesting-chart')">Chart</div>
                    <div class="tab" onclick="showTab('backtesting-table')">Data Table</div>
                </div>
                <div id="backtesting-chart" class="tab-content active">
                    <div class="chart-container" id="backtestingChart"></div>
                </div>
                <div id="backtesting-table" class="tab-content">
                    <div class="table-container">
                        <table id="backtestingTable">
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Actual Price</th>
                                    <th>Predicted Price</th>
                                    <th>Difference</th>
                                    <th>Error %</th>
                                    <th>Grade</th>
                                    <th>Anomaly</th>
                                </tr>
                            </thead>
                            <tbody id="backtestingTableBody">
                                <!-- Table data will be populated by JavaScript -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <!-- Anomaly Detection -->
            <div class="card">
                <div class="card-header">
                    <i class="fas fa-exclamation-triangle card-icon"></i>
                    <h2 class="card-title">Anomaly Detection</h2>
                </div>
                <div class="chart-container" id="anomalyChart"></div>
            </div>
            
            <!-- Price Prediction Visualization -->
            <div class="card">
                <div class="card-header">
                    <i class="fas fa-candlestick-chart card-icon"></i>
                    <h2 class="card-title">Price Prediction Analysis</h2>
                </div>
                <div class="tabs">
                    <div class="tab active" onclick="showTab('candlestick-chart')">Candlestick</div>
                    <div class="tab" onclick="showTab('line-chart')">Line Chart</div>
                </div>
                <div id="candlestick-chart" class="tab-content active">
                    <div class="chart-container" id="candlestickChart"></div>
                </div>
                <div id="line-chart" class="tab-content">
                    <div class="chart-container" id="lineChart"></div>
                </div>
            </div>
            
            <!-- Performance Analysis -->
            <div class="card full-width">
                <div class="card-header">
                    <i class="fas fa-analytics card-icon"></i>
                    <h2 class="card-title">Performance Analysis</h2>
                </div>
                <div class="tabs">
                    <div class="tab active" onclick="showTab('performance-metrics')">Metrics</div>
                    <div class="tab" onclick="showTab('performance-chart')">Trend Analysis</div>
                </div>
                <div id="performance-metrics" class="tab-content active">
                    <div class="table-container">
                        <table id="performanceTable">
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th>Value</th>
                                    <th>Unit</th>
                                </tr>
                            </thead>
                            <tbody id="performanceTableBody">
                                <!-- Performance data will be populated by JavaScript -->
                            </tbody>
                        </table>
                    </div>
                </div>
                <div id="performance-chart" class="tab-content">
                    <div class="chart-container" id="performanceChart"></div>
                </div>
            </div>
            
            <!-- Model Architecture -->
            <div class="card full-width">
                <div class="card-header">
                    <i class="fas fa-sitemap card-icon"></i>
                    <h2 class="card-title">System Architecture</h2>
                </div>
                <div class="chart-container" id="architectureChart"></div>
            </div>
        </div>
    </div>

    <script>
        // Global data storage
        let analysisData = {{
            backtesting: null,
            summary: null,
            metrics: null
        }};
        
        // Tab switching functionality
        function showTab(tabId) {{
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            
            // Show selected tab content
            document.getElementById(tabId).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }}
        
        // Load analysis data
        async function loadAnalysisData() {{
            try {{
                // Load backtesting data
                const backtestingResponse = await fetch('data/backtesting_data.json');
                if (backtestingResponse.ok) {{
                    analysisData.backtesting = await backtestingResponse.json();
                }}
                
                // Load summary data
                const summaryResponse = await fetch('data/summary_data.json');
                if (summaryResponse.ok) {{
                    analysisData.summary = await summaryResponse.json();
                }}
                
                // Load metrics data
                const metricsResponse = await fetch('data/metrics_data.json');
                if (metricsResponse.ok) {{
                    analysisData.metrics = await metricsResponse.json();
                }}
                
                // Initialize dashboard
                initializeDashboard();
                
            }} catch (error) {{
                console.error('Error loading data:', error);
                showError('Failed to load analysis data');
            }}
        }}
        
        // Initialize dashboard components
        function initializeDashboard() {{
            populateMetrics();
            createBacktestingChart();
            createAnomalyChart();
            createCandlestickChart();
            createLineChart();
            createPerformanceChart();
            createArchitectureChart();
            populateTables();
        }}
        
        // Populate key metrics
        function populateMetrics() {{
            const metricsGrid = document.getElementById('metricsGrid');
            if (!analysisData.summary) {{
                metricsGrid.innerHTML = '<div class="error">No summary data available</div>';
                return;
            }}
            
            const summary = analysisData.summary[0];
            metricsGrid.innerHTML = `
                <div class="metric-card">
                    <div class="metric-value">${{summary.Total_Days || 'N/A'}}</div>
                    <div class="metric-label">Trading Days</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${{summary.Mean_Absolute_Percentage_Error || 'N/A'}}%</div>
                    <div class="metric-label">Mean Error</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${{summary.Direction_Accuracy || 'N/A'}}%</div>
                    <div class="metric-label">Direction Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${{summary.Average_Confidence || 'N/A'}}</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${{summary.Anomaly_Rate || 'N/A'}}%</div>
                    <div class="metric-label">Anomaly Rate</div>
                </div>
            `;
        }}
        
        // Create backtesting chart
        function createBacktestingChart() {{
            if (!analysisData.backtesting) {{
                document.getElementById('backtestingChart').innerHTML = '<div class="error">No backtesting data available</div>';
                return;
            }}
            
            const data = analysisData.backtesting;
            const dates = data.map(d => d.Date);
            const actual = data.map(d => d.Actual_Price);
            const predicted = data.map(d => d.Predicted_Price);
            
            const trace1 = {{
                x: dates,
                y: actual,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Actual Price',
                line: {{color: '#2ecc71', width: 3}},
                marker: {{size: 6}}
            }};
            
            const trace2 = {{
                x: dates,
                y: predicted,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Predicted Price',
                line: {{color: '#3498db', width: 3}},
                marker: {{size: 6}}
            }};
            
            const layout = {{
                title: 'Actual vs Predicted Prices - 30 Day Backtesting',
                xaxis: {{title: 'Date'}},
                yaxis: {{title: 'Price'}},
                hovermode: 'closest',
                showlegend: true
            }};
            
            Plotly.newPlot('backtestingChart', [trace1, trace2], layout);
        }}
        
        // Create anomaly detection chart
        function createAnomalyChart() {{
            if (!analysisData.backtesting) {{
                document.getElementById('anomalyChart').innerHTML = '<div class="error">No anomaly data available</div>';
                return;
            }}
            
            const data = analysisData.backtesting;
            const dates = data.map(d => d.Date);
            const prices = data.map(d => d.Actual_Price);
            const anomalies = data.map(d => d.Is_Anomaly);
            
            // Separate normal and anomaly points
            const normalDates = [];
            const normalPrices = [];
            const anomalyDates = [];
            const anomalyPrices = [];
            
            dates.forEach((date, i) => {{
                if (anomalies[i]) {{
                    anomalyDates.push(date);
                    anomalyPrices.push(prices[i]);
                }} else {{
                    normalDates.push(date);
                    normalPrices.push(prices[i]);
                }}
            }});
            
            const trace1 = {{
                x: normalDates,
                y: normalPrices,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Normal',
                line: {{color: '#95a5a6', width: 2}},
                marker: {{size: 4}}
            }};
            
            const trace2 = {{
                x: anomalyDates,
                y: anomalyPrices,
                type: 'scatter',
                mode: 'markers',
                name: 'Anomaly',
                marker: {{
                    color: '#e74c3c',
                    size: 12,
                    symbol: 'x',
                    line: {{width: 2, color: '#c0392b'}}
                }}
            }};
            
            const layout = {{
                title: 'Anomaly Detection Results',
                xaxis: {{title: 'Date'}},
                yaxis: {{title: 'Price'}},
                hovermode: 'closest',
                showlegend: true
            }};
            
            Plotly.newPlot('anomalyChart', [trace1, trace2], layout);
        }}
        
        // Create candlestick chart
        function createCandlestickChart() {{
            if (!analysisData.backtesting) {{
                document.getElementById('candlestickChart').innerHTML = '<div class="error">No candlestick data available</div>';
                return;
            }}
            
            const data = analysisData.backtesting;
            const dates = data.map(d => d.Date);
            const open = data.map(d => d.Actual_Price * 0.99); // Simulate open prices
            const high = data.map(d => d.High);
            const low = data.map(d => d.Low);
            const close = data.map(d => d.Actual_Price);
            const predicted = data.map(d => d.Predicted_Price);
            
            const candlestick = {{
                x: dates,
                open: open,
                high: high,
                low: low,
                close: close,
                type: 'candlestick',
                name: 'Actual Price',
                increasing_line_color: '#2ecc71',
                decreasing_line_color: '#e74c3c'
            }};
            
            const prediction = {{
                x: dates,
                y: predicted,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Predicted Price',
                line: {{color: '#3498db', width: 3}},
                marker: {{size: 6}}
            }};
            
            const layout = {{
                title: 'Candlestick Chart with Predictions',
                xaxis: {{title: 'Date'}},
                yaxis: {{title: 'Price'}},
                hovermode: 'closest',
                showlegend: true
            }};
            
            Plotly.newPlot('candlestickChart', [candlestick, prediction], layout);
        }}
        
        // Create line chart
        function createLineChart() {{
            if (!analysisData.backtesting) {{
                document.getElementById('lineChart').innerHTML = '<div class="error">No line chart data available</div>';
                return;
            }}
            
            const data = analysisData.backtesting;
            const dates = data.map(d => d.Date);
            const actual = data.map(d => d.Actual_Price);
            const predicted = data.map(d => d.Predicted_Price);
            
            // Create confidence bands
            const upperBand = predicted.map(p => p * 1.05);
            const lowerBand = predicted.map(p => p * 0.95);
            
            const trace1 = {{
                x: dates,
                y: actual,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Actual Price',
                line: {{color: '#2ecc71', width: 3}},
                marker: {{size: 6}}
            }};
            
            const trace2 = {{
                x: dates,
                y: predicted,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Predicted Price',
                line: {{color: '#3498db', width: 3}},
                marker: {{size: 6}}
            }};
            
            const trace3 = {{
                x: dates,
                y: upperBand,
                type: 'scatter',
                mode: 'lines',
                line: {{width: 0}},
                showlegend: false,
                hoverinfo: 'skip'
            }};
            
            const trace4 = {{
                x: dates,
                y: lowerBand,
                type: 'scatter',
                mode: 'lines',
                line: {{width: 0}},
                fill: 'tonexty',
                fillcolor: 'rgba(52, 152, 219, 0.2)',
                name: 'Confidence Band',
                hoverinfo: 'skip'
            }};
            
            const layout = {{
                title: 'Price Prediction Comparison with Confidence Bands',
                xaxis: {{title: 'Date'}},
                yaxis: {{title: 'Price'}},
                hovermode: 'closest',
                showlegend: true
            }};
            
            Plotly.newPlot('lineChart', [trace1, trace2, trace3, trace4], layout);
        }}
        
        // Create performance chart
        function createPerformanceChart() {{
            if (!analysisData.backtesting) {{
                document.getElementById('performanceChart').innerHTML = '<div class="error">No performance data available</div>';
                return;
            }}
            
            const data = analysisData.backtesting;
            const dates = data.map(d => d.Date);
            const errors = data.map(d => d.Percentage_Error);
            const confidence = data.map(d => d.Model_Confidence);
            
            const trace1 = {{
                x: dates,
                y: errors,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Prediction Error %',
                line: {{color: '#e74c3c', width: 2}},
                marker: {{size: 6}},
                yaxis: 'y'
            }};
            
            const trace2 = {{
                x: dates,
                y: confidence,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Model Confidence',
                line: {{color: '#3498db', width: 2}},
                marker: {{size: 6}},
                yaxis: 'y2'
            }};
            
            const layout = {{
                title: 'Performance Metrics Over Time',
                xaxis: {{title: 'Date'}},
                yaxis: {{title: 'Error Percentage', side: 'left'}},
                yaxis2: {{
                    title: 'Confidence Score',
                    side: 'right',
                    overlaying: 'y',
                    range: [0, 1]
                }},
                hovermode: 'closest',
                showlegend: true
            }};
            
            Plotly.newPlot('performanceChart', [trace1, trace2], layout);
        }}
        
        // Create architecture chart
        function createArchitectureChart() {{
            // Create a simple architecture diagram using Plotly
            const nodes = [
                {{x: 1, y: 8, text: 'Data Sources', color: '#e1f5fe'}},
                {{x: 3, y: 8, text: 'Data Ingestion', color: '#f3e5f5'}},
                {{x: 5, y: 8, text: 'Data Processing', color: '#e8f5e8'}},
                {{x: 7, y: 8, text: 'ML Models', color: '#fff3e0'}},
                {{x: 1, y: 6, text: 'Anomaly Detection', color: '#ffebee'}},
                {{x: 3, y: 6, text: 'Sentiment Analysis', color: '#f1f8e9'}},
                {{x: 5, y: 6, text: 'Trend Prediction', color: '#e3f2fd'}},
                {{x: 7, y: 6, text: 'Portfolio Analysis', color: '#fce4ec'}},
                {{x: 4, y: 4, text: 'Fusion Engine', color: '#f9fbe7'}},
                {{x: 4, y: 2, text: 'Decision Engine', color: '#e0f2f1'}},
                {{x: 1, y: 0, text: 'Dashboard', color: '#e8eaf6'}},
                {{x: 3, y: 0, text: 'Alerts', color: '#fff8e1'}},
                {{x: 5, y: 0, text: 'API', color: '#f3e5f5'}},
                {{x: 7, y: 0, text: 'Database', color: '#e0f7fa'}}
            ];
            
            const trace = {{
                x: nodes.map(n => n.x),
                y: nodes.map(n => n.y),
                mode: 'markers+text',
                type: 'scatter',
                text: nodes.map(n => n.text),
                textposition: 'middle center',
                marker: {{
                    size: 50,
                    color: nodes.map(n => n.color),
                    line: {{width: 2, color: '#2c3e50'}}
                }},
                hovertemplate: '<b>%{{text}}</b><extra></extra>'
            }};
            
            const layout = {{
                title: 'System Architecture Overview',
                xaxis: {{showgrid: false, zeroline: false, showticklabels: false}},
                yaxis: {{showgrid: false, zeroline: false, showticklabels: false}},
                hovermode: 'closest',
                showlegend: false,
                width: 800,
                height: 400
            }};
            
            Plotly.newPlot('architectureChart', [trace], layout);
        }}
        
        // Populate data tables
        function populateTables() {{
            // Populate backtesting table
            if (analysisData.backtesting) {{
                const tbody = document.getElementById('backtestingTableBody');
                tbody.innerHTML = analysisData.backtesting.map(row => `
                    <tr>
                        <td>${{row.Date}}</td>
                        <td>${{row.Actual_Price}}</td>
                        <td>${{row.Predicted_Price}}</td>
                        <td>${{row.Difference}}</td>
                        <td>${{row.Percentage_Error}}%</td>
                        <td><span class="status-badge status-${{getStatusClass(row.Performance_Grade)}}">${{row.Performance_Grade}}</span></td>
                        <td><span class="status-badge ${{row.Is_Anomaly ? 'status-danger' : 'status-good'}}">${{row.Is_Anomaly ? 'Yes' : 'No'}}</span></td>
                    </tr>
                `).join('');
            }}
            
            // Populate performance table
            if (analysisData.metrics) {{
                const tbody = document.getElementById('performanceTableBody');
                tbody.innerHTML = analysisData.metrics.map(row => `
                    <tr>
                        <td>${{row.Metric}}</td>
                        <td>${{row.Value}}</td>
                        <td>${{row.Unit}}</td>
                    </tr>
                `).join('');
            }}
        }}
        
        // Helper function to get status class
        function getStatusClass(grade) {{
            if (grade === 'A+') return 'excellent';
            if (grade === 'A') return 'good';
            if (grade === 'B') return 'warning';
            return 'danger';
        }}
        
        // Show error message
        function showError(message) {{
            const container = document.querySelector('.container');
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = message;
            container.insertBefore(errorDiv, container.firstChild);
        }}
        
        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            loadAnalysisData();
        }});
    </script>
</body>
</html>
        """
        
        # Save main dashboard
        dashboard_path = os.path.join(self.output_dir, "index.html")
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        print(f"✅ Main dashboard created: {dashboard_path}")
        return dashboard_path
    
    def create_data_files(self):
        """Create JSON data files for the dashboard"""
        print("Creating data files...")
        
        # Create data directory
        data_dir = os.path.join(self.output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        try:
            # Convert backtesting data to JSON
            if self.backtesting_df is not None:
                backtesting_data = self.backtesting_df.to_dict('records')
                backtesting_path = os.path.join(data_dir, "backtesting_data.json")
                with open(backtesting_path, 'w', encoding='utf-8') as f:
                    json.dump(backtesting_data, f, indent=2, default=str)
                print(f"Backtesting data saved: {backtesting_path}")
            
            # Convert summary data to JSON
            if self.summary_df is not None:
                summary_data = self.summary_df.to_dict('records')
                summary_path = os.path.join(data_dir, "summary_data.json")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary_data, f, indent=2, default=str)
                print(f"Summary data saved: {summary_path}")
            
            # Convert metrics data to JSON
            if self.metrics_df is not None:
                metrics_data = self.metrics_df.to_dict('records')
                metrics_path = os.path.join(data_dir, "metrics_data.json")
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics_data, f, indent=2, default=str)
                print(f"Metrics data saved: {metrics_path}")
            
            return True
            
        except Exception as e:
            print(f"Error creating data files: {str(e)}")
            return False
    
    def create_dashboard_runner(self):
        """Create a simple dashboard runner script"""
        runner_script = """
import os
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import time

def start_server():
    '''Start a simple HTTP server to serve the dashboard'''
    port = 8080
    os.chdir('frontend_dashboard')
    
    try:
        server = HTTPServer(('localhost', port), SimpleHTTPRequestHandler)
        print(f"🚀 Dashboard server starting on http://localhost:{port}")
        print("📊 Open your browser and navigate to the URL above")
        print("⏹️  Press Ctrl+C to stop the server")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    start_server()
        """
        
        runner_path = "run_dashboard.py"
        with open(runner_path, 'w', encoding='utf-8') as f:
            f.write(runner_script)
        
        print(f"Dashboard runner created: {runner_path}")
        return runner_path
    
    def run_dashboard(self):
        """Run the complete dashboard creation process"""
        print("Creating Frontend Dashboard...")
        print("=" * 60)
        
        try:
            # Create main dashboard
            dashboard_path = self.create_main_dashboard()
            if not dashboard_path:
                return False
            
            # Create data files
            if not self.create_data_files():
                return False
            
            # Create dashboard runner
            runner_path = self.create_dashboard_runner()
            
            print("\nFRONTEND DASHBOARD CREATED SUCCESSFULLY!")
            print("=" * 60)
            
            print(f"\nDashboard Location: {self.output_dir}/")
            print(f"Main Dashboard: {self.output_dir}/index.html")
            print(f"Data Files: {self.output_dir}/data/")
            print(f"Runner Script: {runner_path}")
            
            print(f"\nHOW TO VIEW YOUR DASHBOARD:")
            print(f"1. Run: python {runner_path}")
            print(f"2. Open: http://localhost:8080")
            print(f"3. Or directly open: {self.output_dir}/index.html")
            
            print(f"\nFEATURES INCLUDED:")
            print(f"• Interactive charts and visualizations")
            print(f"• Real-time data tables")
            print(f"• Responsive design for all devices")
            print(f"• Tabbed interface for easy navigation")
            print(f"• Performance metrics and analysis")
            print(f"• System architecture visualization")
            
            return True
            
        except Exception as e:
            print(f"❌ Dashboard creation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function to create and run the dashboard"""
    print("Starting Frontend Dashboard Creation")
    print("=" * 60)
    
    # Create dashboard instance
    dashboard = FrontendDashboard()
    
    # Run dashboard creation
    success = dashboard.run_dashboard()
    
    if success:
        print(f"\n🎉 Dashboard ready! Run 'python run_dashboard.py' to start the server.")
    else:
        print(f"\n❌ Dashboard creation failed. Please check the errors above.")

if __name__ == "__main__":
    main()
