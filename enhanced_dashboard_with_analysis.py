"""
Enhanced Real-Time Dashboard with Analysis Components
Adds the analysis dashboard as a new tab in your existing Streamlit application
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

# Import the analysis dashboard page
from analysis_dashboard_page import (
    load_analysis_data, 
    create_metrics_overview, 
    create_backtesting_chart, 
    create_anomaly_detection_chart,
    create_candlestick_chart,
    create_performance_chart,
    create_architecture_diagram,
    create_data_tables
)

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

# Import real-time system
@st.cache_resource
def get_realtime_system():
    """Get the real-time system instance"""
    try:
        from realtime_anomaly_project.realtime_enhanced_system_100_accuracy import RealTimeEnhancedDataSystemFor100Accuracy
        return RealTimeEnhancedDataSystemFor100Accuracy()
    except Exception as e:
        st.error(f"Error initializing real-time system: {str(e)}")
        return None

# News fetching functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_ticker_news(ticker: str, max_articles: int = 10) -> List[Dict[str, Any]]:
    """Fetch news articles related to a specific ticker"""
    try:
        # Get company info
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        company_name = info.get('longName', ticker.replace('.NS', ''))
        
        # RSS news sources focused on Indian markets
        news_sources = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://www.moneycontrol.com/rss/business.xml",
            "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
            "https://www.financialexpress.com/market/rss",
        ]
        
        news_articles = []
        search_terms = [company_name.lower(), ticker.replace('.NS', '').lower(), 'nifty', 'indian stock', 'india', 'market']
        
        for source_url in news_sources:
            try:
                feed = feedparser.parse(source_url)
                entries = getattr(feed, 'entries', [])
                
                for entry in entries[:5]:  # Limit per source
                    if len(news_articles) >= max_articles:
                        break
                        
                    title = entry.get('title', '') or ''
                    description = entry.get('description', '') or ''
                    link = entry.get('link', '') or ''
                    
                    # Check if article is relevant - more lenient matching
                    content_text = (title + ' ' + description).lower()
                    is_relevant = any(term in content_text for term in search_terms)
                    
                    # If no specific match, include general market news for Indian stocks
                    if not is_relevant and ticker.endswith('.NS'):
                        general_terms = ['stock', 'market', 'trading', 'nifty', 'sensex', 'bse', 'nse']
                        is_relevant = any(term in content_text for term in general_terms)
                    
                    if is_relevant:
                        # Try to get image
                        image_url = get_article_image(entry, link)
                        
                        news_articles.append({
                            'title': title,
                            'description': description,
                            'link': link,
                            'published': entry.get('published', ''),
                            'source': source_url,
                            'image_url': image_url
                        })
                        
            except Exception as e:
                continue
        
        return news_articles[:max_articles]
        
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {str(e)}")
        return []

def get_article_image(entry, link):
    """Try to extract image URL from article entry"""
    try:
        # Check for media content
        if hasattr(entry, 'media_content'):
            for media in entry.media_content:
                if media.get('type', '').startswith('image'):
                    return media.get('url', '')
        
        # Check for enclosures
        if hasattr(entry, 'enclosures'):
            for enclosure in entry.enclosures:
                if enclosure.get('type', '').startswith('image'):
                    return enclosure.get('href', '')
        
        # Try to scrape image from link
        if link:
            try:
                response = requests.get(link, timeout=5)
                soup = BeautifulSoup(response.content, 'html.parser')
                img_tag = soup.find('img')
                if img_tag and img_tag.get('src'):
                    return img_tag.get('src')
            except:
                pass
                
    except Exception:
        pass
    
    return None

def display_anomaly_analysis(analysis_results):
    """Display anomaly detection results"""
    st.header("🔍 Real-Time Anomaly Detection")
    
    for ticker, results in analysis_results.items():
        with st.expander(f"📊 {ticker} - Anomaly Analysis", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Anomaly Score",
                    value=f"{results.get('anomaly_score', 0):.3f}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    label="Is Anomaly",
                    value="⚠️ YES" if results.get('is_anomaly', False) else "✅ NO",
                    delta=None
                )
            
            with col3:
                st.metric(
                    label="Confidence",
                    value=f"{results.get('confidence', 0):.1%}",
                    delta=None
                )
            
            # Anomaly chart
            if 'price_data' in results and results['price_data'] is not None:
                df = results['price_data']
                if not df.empty:
                    fig = go.Figure()
                    
                    # Add price line
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['close'],
                        mode='lines',
                        name='Price',
                        line=dict(color='lightblue', width=2)
                    ))
                    
                    # Highlight anomalies
                    if 'anomaly_flags' in results and results['anomaly_flags'] is not None:
                        anomaly_dates = df.index[results['anomaly_flags']]
                        anomaly_prices = df['close'][results['anomaly_flags']]
                        
                        fig.add_trace(go.Scatter(
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
                        ))
                    
                    fig.update_layout(
                        title=f"{ticker} - Price with Anomaly Detection",
                        xaxis_title="Time",
                        yaxis_title="Price",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

def display_sentiment_analysis(analysis_results, selected_ticker):
    """Display sentiment analysis results"""
    st.header("💭 Real-Time Sentiment Analysis")
    
    if selected_ticker in analysis_results:
        results = analysis_results[selected_ticker]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_score = results.get('sentiment_score', 0)
            st.metric(
                label="Sentiment Score",
                value=f"{sentiment_score:.3f}",
                delta=None
            )
        
        with col2:
            sentiment_label = results.get('sentiment_label', 'Neutral')
            st.metric(
                label="Sentiment",
                value=sentiment_label,
                delta=None
            )
        
        with col3:
            confidence = results.get('sentiment_confidence', 0)
            st.metric(
                label="Confidence",
                value=f"{confidence:.1%}",
                delta=None
            )
        
        # Display news articles
        st.subheader(f"📰 Latest News for {selected_ticker}")
        news_articles = fetch_ticker_news(selected_ticker, max_articles=5)
        
        if news_articles:
            for i, article in enumerate(news_articles):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**{article['title']}**")
                        st.write(article['description'][:200] + "...")
                        st.write(f"[Read more]({article['link']})")
                    
                    with col2:
                        if article.get('image_url'):
                            st.image(article['image_url'], width=100)
                    
                    if i < len(news_articles) - 1:
                        st.divider()
        else:
            st.info("No recent news articles found for this ticker.")

def display_trend_prediction(analysis_results):
    """Display trend prediction results"""
    st.header("📊 Real-Time Trend Prediction")
    
    for ticker, results in analysis_results.items():
        with st.expander(f"📈 {ticker} - Trend Prediction", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend_score = results.get('trend_score', 0)
                st.metric(
                    label="Trend Score",
                    value=f"{trend_score:.3f}",
                    delta=None
                )
            
            with col2:
                trend_direction = results.get('trend_direction', 'Neutral')
                st.metric(
                    label="Trend Direction",
                    value=trend_direction,
                    delta=None
                )
            
            with col3:
                confidence = results.get('trend_confidence', 0)
                st.metric(
                    label="Confidence",
                    value=f"{confidence:.1%}",
                    delta=None
                )
            
            # Price prediction chart
            if 'price_data' in results and results['price_data'] is not None:
                df = results['price_data']
                if not df.empty and len(df) > 1:
                    fig = go.Figure()
                    
                    # Add actual prices
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['close'],
                        mode='lines+markers',
                        name='Actual Price',
                        line=dict(color='green', width=3)
                    ))
                    
                    # Add predicted prices if available
                    if 'predicted_prices' in results and results['predicted_prices'] is not None:
                        pred_prices = results['predicted_prices']
                        if len(pred_prices) > 0:
                            pred_dates = df.index[-len(pred_prices):]
                            fig.add_trace(go.Scatter(
                                x=pred_dates,
                                y=pred_prices,
                                mode='lines+markers',
                                name='Predicted Price',
                                line=dict(color='blue', width=3)
                            ))
                    
                    fig.update_layout(
                        title=f"{ticker} - Price Trend Analysis",
                        xaxis_title="Time",
                        yaxis_title="Price",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

def display_seasonality_analysis(analysis_results):
    """Display seasonality analysis results"""
    st.header("🗓️ Seasonality Analysis")
    
    for ticker, results in analysis_results.items():
        with st.expander(f"📅 {ticker} - Seasonality Analysis", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                seasonality_score = results.get('seasonality_score', 0)
                st.metric(
                    label="Seasonality Score",
                    value=f"{seasonality_score:.3f}",
                    delta=None
                )
            
            with col2:
                seasonal_pattern = results.get('seasonal_pattern', 'None')
                st.metric(
                    label="Seasonal Pattern",
                    value=seasonal_pattern,
                    delta=None
                )
            
            with col3:
                confidence = results.get('seasonality_confidence', 0)
                st.metric(
                    label="Confidence",
                    value=f"{confidence:.1%}",
                    delta=None
                )

def display_fusion_scores(analysis_results):
    """Display fusion scores"""
    st.header("🔮 Fusion Scores & Recommendations")
    
    for ticker, results in analysis_results.items():
        with st.expander(f"🎯 {ticker} - Fusion Analysis", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fusion_score = results.get('fusion_score', 0)
                st.metric(
                    label="Fusion Score",
                    value=f"{fusion_score:.3f}",
                    delta=None
                )
            
            with col2:
                recommendation = results.get('recommendation', 'HOLD')
                st.metric(
                    label="Recommendation",
                    value=recommendation,
                    delta=None
                )
            
            with col3:
                confidence = results.get('fusion_confidence', 0)
                st.metric(
                    label="Confidence",
                    value=f"{confidence:.1%}",
                    delta=None
                )
            
            # Fusion score breakdown
            st.subheader("📊 Score Breakdown")
            breakdown_cols = st.columns(4)
            
            with breakdown_cols[0]:
                st.metric("Anomaly", f"{results.get('anomaly_score', 0):.3f}")
            with breakdown_cols[1]:
                st.metric("Sentiment", f"{results.get('sentiment_score', 0):.3f}")
            with breakdown_cols[2]:
                st.metric("Trend", f"{results.get('trend_score', 0):.3f}")
            with breakdown_cols[3]:
                st.metric("Seasonality", f"{results.get('seasonality_score', 0):.3f}")

def display_portfolio_analysis(analysis_results, portfolio):
    """Display portfolio-specific analysis"""
    st.header("📂 Portfolio Analysis")
    
    if not portfolio:
        st.info("No portfolio configured. Please add stocks to your portfolio in the sidebar.")
        return
    
    # Calculate portfolio metrics
    total_value = 0
    portfolio_metrics = {}
    
    for ticker, quantity in portfolio.items():
        if quantity > 0 and ticker in analysis_results:
            results = analysis_results[ticker]
            current_price = results.get('current_price', 0)
            value = quantity * current_price
            total_value += value
            
            portfolio_metrics[ticker] = {
                'quantity': quantity,
                'price': current_price,
                'value': value,
                'fusion_score': results.get('fusion_score', 0),
                'recommendation': results.get('recommendation', 'HOLD')
            }
    
    # Portfolio overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Portfolio Value", f"₹{total_value:,.2f}")
    
    with col2:
        avg_fusion_score = np.mean([metrics['fusion_score'] for metrics in portfolio_metrics.values()])
        st.metric("Average Fusion Score", f"{avg_fusion_score:.3f}")
    
    with col3:
        buy_signals = sum(1 for metrics in portfolio_metrics.values() if 'BUY' in metrics['recommendation'])
        st.metric("Buy Signals", buy_signals)
    
    # Individual stock analysis
    st.subheader("📊 Individual Stock Analysis")
    
    for ticker, metrics in portfolio_metrics.items():
        with st.expander(f"📈 {ticker} - Portfolio Analysis", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Quantity", f"{metrics['quantity']:.0f}")
            with col2:
                st.metric("Current Price", f"₹{metrics['price']:.2f}")
            with col3:
                st.metric("Total Value", f"₹{metrics['value']:,.2f}")
            with col4:
                st.metric("Recommendation", metrics['recommendation'])

def main():
    """Main dashboard function"""
    st.title("🇮🇳 Real-Time Nifty-Fifty Dashboard - 100% Accuracy System")
    st.subheader("Live Accuracy System for Anomaly Detection, Sentiment Analysis, Trend Prediction & Portfolio Analytics")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Nifty-Fifty ticker selection
    nifty_fifty_tickers = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS',
        'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'AXISBANK.NS', 'NESTLEIND.NS',
        'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS', 'TECHM.NS', 'TCS.NS'
    ]
    
    selected_tickers = st.sidebar.multiselect(
        "Select Nifty-Fifty Stocks for Real-Time Analysis",
        options=nifty_fifty_tickers,
        default=['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS'],
        key="selected_tickers"
    )
    
    # Portfolio configuration for Nifty stocks
    st.sidebar.subheader("📊 Nifty Portfolio Configuration")
    portfolio = {}
    for ticker in selected_tickers:
        quantity = st.sidebar.number_input(
            f"{ticker} Shares",
            min_value=0.0,
            value=0.0,
            step=1.0,
            key=f"portfolio_{ticker}"
        )
        if quantity > 0:
            portfolio[ticker] = quantity
    
    # Auto-refresh settings
    st.sidebar.subheader("🔄 Update Settings")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 300, 30)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Initialize system
    if st.session_state.realtime_system is None:
        st.session_state.realtime_system = get_realtime_system()
    
    if st.session_state.realtime_system is None:
        st.error("❌ Failed to initialize real-time system. Please check your configuration.")
        return
    
    # Run analysis
    if not selected_tickers:
        st.warning("⚠️ Please select at least one ticker for analysis.")
        return
    
    with st.spinner("🔄 Running real-time analysis..."):
        analysis_results = {}
        for ticker in selected_tickers:
            try:
                results = st.session_state.realtime_system.run_comprehensive_analysis(ticker)
                analysis_results[ticker] = results
            except Exception as e:
                st.error(f"Error analyzing {ticker}: {str(e)}")
                continue
    
    if not analysis_results:
        st.error("❌ No analysis results available. Please check your ticker selection and try again.")
        return
    
    # Create tabs for different views - INCLUDING THE NEW ANALYSIS TAB
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔍 Anomaly Detection", 
        "💭 Sentiment Analysis", 
        "📊 Trend Prediction", 
        "🗓️ Seasonality", 
        "🔮 Fusion Scores", 
        "📂 Portfolio Specific",
        "📈 Analysis Dashboard"  # NEW TAB
    ])
    
    with tab1:
        display_anomaly_analysis(analysis_results)
    
    with tab2:
        # Get selected ticker for news display
        tickers_list = list(analysis_results.keys())
        if tickers_list:
            selected_ticker = st.selectbox("Select ticker for news analysis:", tickers_list)
            display_sentiment_analysis(analysis_results, selected_ticker)
        else:
            st.warning("No tickers available for sentiment analysis.")
    
    with tab3:
        display_trend_prediction(analysis_results)
    
    with tab4:
        display_seasonality_analysis(analysis_results)
    
    with tab5:
        display_fusion_scores(analysis_results)
    
    with tab6:
        display_portfolio_analysis(analysis_results, portfolio)
    
    # NEW ANALYSIS DASHBOARD TAB
    with tab7:
        st.header("📈 Analysis Dashboard")
        st.markdown("---")
        
        # Load analysis data
        with st.spinner("Loading analysis data..."):
            backtesting_df, summary_df, metrics_df = load_analysis_data()
        
        if backtesting_df is None and summary_df is None and metrics_df is None:
            st.error("❌ No analysis data found. Please run the analysis components first.")
            st.info("💡 Run: `python simple_analysis_runner.py` to generate the analysis data")
        else:
            # Create sub-tabs for different analysis views
            analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4, analysis_tab5 = st.tabs([
                "📊 Overview", 
                "📈 Backtesting", 
                "🔍 Anomaly Detection", 
                "📊 Performance", 
                "🏗️ Architecture"
            ])
            
            with analysis_tab1:
                st.header("📊 Key Performance Metrics")
                create_metrics_overview(summary_df)
                
                st.header("📈 Quick Overview Charts")
                col1, col2 = st.columns(2)
                
                with col1:
                    create_backtesting_chart(backtesting_df)
                
                with col2:
                    create_anomaly_detection_chart(backtesting_df)
            
            with analysis_tab2:
                st.header("📈 Backtesting Results")
                create_candlestick_chart(backtesting_df)
                create_data_tables(backtesting_df, metrics_df)
            
            with analysis_tab3:
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
            
            with analysis_tab4:
                st.header("📊 Performance Analysis")
                create_performance_chart(backtesting_df)
                create_data_tables(backtesting_df, metrics_df)
            
            with analysis_tab5:
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
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
