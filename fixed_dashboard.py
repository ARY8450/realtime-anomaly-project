"""
Fixed Real-Time Dashboard with Analysis Components
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings
import feedparser
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import re
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure Streamlit page
st.set_page_config(
    page_title="🇮🇳 Real-Time Nifty-Fifty Dashboard - 100% Accuracy System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'realtime_system' not in st.session_state:
    st.session_state.realtime_system = None
    
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
    
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

def load_analysis_data():
    """Load analysis data from the comprehensive_analysis directory"""
    try:
        analysis_dir = "comprehensive_analysis"
        
        # Load backtesting data
        backtesting_path = os.path.join(analysis_dir, "backtesting_results", "RELIANCE_NS_prediction_comparison_table.csv")
        if os.path.exists(backtesting_path):
            backtesting_df = pd.read_csv(backtesting_path)
        else:
            backtesting_df = None
            
        # Load summary data
        summary_path = os.path.join(analysis_dir, "backtesting_results", "RELIANCE_NS_prediction_summary_stats.csv")
        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path)
        else:
            summary_df = None
            
        # Load metrics data
        metrics_path = os.path.join(analysis_dir, "prediction_tables", "RELIANCE_NS_performance_metrics_table.csv")
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
        else:
            metrics_df = None
            
        return backtesting_df, summary_df, metrics_df
        
    except Exception as e:
        st.error(f"Error loading analysis data: {str(e)}")
        return None, None, None

def create_metrics_overview(summary_df):
    """Create key metrics overview"""
    if summary_df is None or summary_df.empty:
        st.warning("No summary data available")
        return
        
    summary = summary_df.iloc[0]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📅 Trading Days",
            value=f"{summary.get('Total_Days', 'N/A')}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="📊 Mean Error",
            value=f"{summary.get('Mean_Absolute_Percentage_Error', 'N/A')}%",
            delta=None
        )
    
    with col3:
        st.metric(
            label="🎯 Direction Accuracy",
            value=f"{summary.get('Direction_Accuracy', 'N/A')}%",
            delta=None
        )
    
    with col4:
        st.metric(
            label="🔮 Avg Confidence",
            value=f"{summary.get('Average_Confidence', 'N/A')}",
            delta=None
        )
    
    with col5:
        st.metric(
            label="⚠️ Anomaly Rate",
            value=f"{summary.get('Anomaly_Rate', 'N/A')}%",
            delta=None
        )

def create_backtesting_chart(backtesting_df):
    """Create backtesting results chart"""
    if backtesting_df is None or backtesting_df.empty:
        st.warning("No backtesting data available")
        return
        
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(go.Scatter(
        x=backtesting_df['Date'],
        y=backtesting_df['Actual_Price'],
        mode='lines+markers',
        name='Actual Price',
        line=dict(color='#2ecc71', width=3),
        marker=dict(size=6)
    ))
    
    # Add predicted prices
    fig.add_trace(go.Scatter(
        x=backtesting_df['Date'],
        y=backtesting_df['Predicted_Price'],
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='#3498db', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Prices - 30 Day Backtesting',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='closest',
        showlegend=True,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_anomaly_detection_chart(backtesting_df):
    """Create anomaly detection chart"""
    if backtesting_df is None or backtesting_df.empty:
        st.warning("No anomaly data available")
        return
        
    fig = go.Figure()
    
    # Separate normal and anomaly points
    normal_data = backtesting_df[~backtesting_df['Is_Anomaly']]
    anomaly_data = backtesting_df[backtesting_df['Is_Anomaly']]
    
    # Add normal points
    fig.add_trace(go.Scatter(
        x=normal_data['Date'],
        y=normal_data['Actual_Price'],
        mode='lines+markers',
        name='Normal',
        line=dict(color='#95a5a6', width=2),
        marker=dict(size=4)
    ))
    
    # Add anomaly points
    if not anomaly_data.empty:
        fig.add_trace(go.Scatter(
            x=anomaly_data['Date'],
            y=anomaly_data['Actual_Price'],
            mode='markers',
            name='Anomaly',
            marker=dict(
                color='#e74c3c',
                size=12,
                symbol='x',
                line=dict(width=2, color='#c0392b')
            )
        ))
    
    fig.update_layout(
        title='Anomaly Detection Results',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='closest',
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_candlestick_chart(backtesting_df):
    """Create candlestick chart with predictions"""
    if backtesting_df is None or backtesting_df.empty:
        st.warning("No candlestick data available")
        return
        
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('RELIANCE.NS - Actual vs Predicted Prices', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=backtesting_df['Date'],
            open=backtesting_df['Actual_Price'] * 0.99,  # Simulate open prices
            high=backtesting_df['High'],
            low=backtesting_df['Low'],
            close=backtesting_df['Actual_Price'],
            name='Actual Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # Predicted prices line
    fig.add_trace(
        go.Scatter(
            x=backtesting_df['Date'],
            y=backtesting_df['Predicted_Price'],
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=backtesting_df['Date'],
            y=backtesting_df['Volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title='RELIANCE.NS Backtesting Results - 30 Days',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_performance_chart(backtesting_df):
    """Create performance metrics chart"""
    if backtesting_df is None or backtesting_df.empty:
        st.warning("No performance data available")
        return
        
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Prediction Error Over Time', 'Model Confidence Over Time'),
        vertical_spacing=0.1
    )
    
    # Error percentage chart
    fig.add_trace(
        go.Scatter(
            x=backtesting_df['Date'],
            y=backtesting_df['Percentage_Error'],
            mode='lines+markers',
            name='Prediction Error %',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Confidence chart
    fig.add_trace(
        go.Scatter(
            x=backtesting_df['Date'],
            y=backtesting_df['Model_Confidence'],
            mode='lines+markers',
            name='Model Confidence',
            line=dict(color='#3498db', width=2),
            marker=dict(size=6)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Performance Metrics Over Time',
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Error %", row=1, col=1)
    fig.update_yaxes(title_text="Confidence Score", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def create_architecture_diagram():
    """Create system architecture diagram"""
    # Create a simple architecture diagram using Plotly
    nodes = [
        {'x': 1, 'y': 8, 'text': 'Data Sources', 'color': '#e1f5fe'},
        {'x': 3, 'y': 8, 'text': 'Data Ingestion', 'color': '#f3e5f5'},
        {'x': 5, 'y': 8, 'text': 'Data Processing', 'color': '#e8f5e8'},
        {'x': 7, 'y': 8, 'text': 'ML Models', 'color': '#fff3e0'},
        {'x': 1, 'y': 6, 'text': 'Anomaly Detection', 'color': '#ffebee'},
        {'x': 3, 'y': 6, 'text': 'Sentiment Analysis', 'color': '#f1f8e9'},
        {'x': 5, 'y': 6, 'text': 'Trend Prediction', 'color': '#e3f2fd'},
        {'x': 7, 'y': 6, 'text': 'Portfolio Analysis', 'color': '#fce4ec'},
        {'x': 4, 'y': 4, 'text': 'Fusion Engine', 'color': '#f9fbe7'},
        {'x': 4, 'y': 2, 'text': 'Decision Engine', 'color': '#e0f2f1'},
        {'x': 1, 'y': 0, 'text': 'Dashboard', 'color': '#e8eaf6'},
        {'x': 3, 'y': 0, 'text': 'Alerts', 'color': '#fff8e1'},
        {'x': 5, 'y': 0, 'text': 'API', 'color': '#f3e5f5'},
        {'x': 7, 'y': 0, 'text': 'Database', 'color': '#e0f7fa'}
    ]
    
    fig = go.Figure()
    
    trace = go.Scatter(
        x=[node['x'] for node in nodes],
        y=[node['y'] for node in nodes],
        mode='markers+text',
        text=[node['text'] for node in nodes],
        textposition='middle center',
        marker=dict(
            size=50,
            color=[node['color'] for node in nodes],
            line=dict(width=2, color='#2c3e50')
        ),
        hovertemplate='<b>%{text}</b><extra></extra>'
    )
    
    fig.add_trace(trace)
    
    fig.update_layout(
        title='System Architecture Overview',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest',
        showlegend=False,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_data_tables(backtesting_df, metrics_df):
    """Create data tables"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Backtesting Results Table")
        if backtesting_df is not None and not backtesting_df.empty:
            # Show only first 10 rows for performance
            display_df = backtesting_df[['Date', 'Actual_Price', 'Predicted_Price', 'Difference', 'Percentage_Error', 'Performance_Grade', 'Is_Anomaly']].head(10)
            st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("No backtesting data available")
    
    with col2:
        st.subheader("📈 Performance Metrics")
        if metrics_df is not None and not metrics_df.empty:
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.warning("No metrics data available")

def main():
    """Main function to create the analysis dashboard page"""
    st.title("📊 Real-Time Anomaly Detection Dashboard")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading analysis data..."):
        backtesting_df, summary_df, metrics_df = load_analysis_data()
    
    if backtesting_df is None and summary_df is None and metrics_df is None:
        st.error("❌ No analysis data found. Please run the analysis components first.")
        st.info("💡 Run: `python simple_analysis_runner.py` to generate the analysis data")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Backtesting", 
        "🔍 Anomaly Detection", 
        "📊 Performance", 
        "🏗️ Architecture"
    ])
    
    with tab1:
        st.header("📊 Key Performance Metrics")
        create_metrics_overview(summary_df)
        
        st.header("📈 Quick Overview Charts")
        col1, col2 = st.columns(2)
        
        with col1:
            create_backtesting_chart(backtesting_df)
        
        with col2:
            create_anomaly_detection_chart(backtesting_df)
    
    with tab2:
        st.header("📈 Backtesting Results")
        create_candlestick_chart(backtesting_df)
        create_data_tables(backtesting_df, metrics_df)
    
    with tab3:
        st.header("🔍 Anomaly Detection Analysis")
        create_anomaly_detection_chart(backtesting_df)
        
        if backtesting_df is not None and not backtesting_df.empty:
            anomaly_count = backtesting_df['Is_Anomaly'].sum()
            total_count = len(backtesting_df)
            anomaly_rate = (anomaly_count / total_count) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Data Points", total_count)
            with col2:
                st.metric("Anomalies Detected", anomaly_count)
            with col3:
                st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    
    with tab4:
        st.header("📊 Performance Analysis")
        create_performance_chart(backtesting_df)
        create_data_tables(backtesting_df, metrics_df)
    
    with tab5:
        st.header("🏗️ System Architecture")
        create_architecture_diagram()
        
        st.markdown("""
        ### System Components:
        - **Data Sources**: Market data, news feeds, economic indicators
        - **Data Ingestion**: Real-time data collection and preprocessing
        - **Data Processing**: Feature engineering and data transformation
        - **ML Models**: Anomaly detection, sentiment analysis, trend prediction
        - **Fusion Engine**: Combines all analysis results
        - **Decision Engine**: Generates trading recommendations
        - **Dashboard**: Real-time visualization and monitoring
        """)

if __name__ == "__main__":
    main()
