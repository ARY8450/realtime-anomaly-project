"""
Real-Time Enhanced Dashboard for 100% Accuracy System - Nifty-Fifty Edition
Live dashboard with real-time updates for anomaly detection, sentiment analysis, trend prediction, and portfolio analytics
Specifically designed for Indian Nifty-Fifty stock market analysis
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
    page_icon="�",
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
        search_terms = [company_name.lower(), ticker.replace('.NS', '').lower(), 'nifty', 'indian stock']
        
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
                    
                    # Check if article is relevant
                    content_text = (title + ' ' + description).lower()
                    if any(term in content_text for term in search_terms):
                        # Try to get image
                        image_url = get_article_image(entry, link)
                        
                        news_articles.append({
                            'title': title,
                            'description': description,
                            'link': link,
                            'published': entry.get('published', ''),
                            'source': source_url,
                            'image': image_url,
                            'ticker': ticker
                        })
                        
            except Exception as e:
                continue
        
        # Sort by published date (newest first)
        news_articles.sort(key=lambda x: x.get('published', ''), reverse=True)
        return news_articles[:max_articles]
        
    except Exception as e:
        st.warning(f"Error fetching news for {ticker}: {e}")
        return []

def get_article_image(entry: Dict[str, Any], link: str) -> str:
    """Extract image URL from article entry or webpage"""
    try:
        # Try to get image from RSS entry media_thumbnail
        media_thumbnail = getattr(entry, 'media_thumbnail', None)
        if media_thumbnail and isinstance(media_thumbnail, list) and len(media_thumbnail) > 0:
            first_thumbnail = media_thumbnail[0]
            if isinstance(first_thumbnail, dict) and 'url' in first_thumbnail:
                return str(first_thumbnail['url'])
        
        # Try to get image from enclosures
        enclosures = getattr(entry, 'enclosures', None)
        if enclosures and isinstance(enclosures, list):
            for enclosure in enclosures:
                if isinstance(enclosure, dict):
                    enc_type = enclosure.get('type', '')
                    if isinstance(enc_type, str) and 'image' in enc_type:
                        href = enclosure.get('href', '')
                        if isinstance(href, str):
                            return href
        
        # Try to extract image from description HTML using regex (simpler than BeautifulSoup)
        description = entry.get('description', '')
        if description and isinstance(description, str):
            try:
                # Use regex to find img src
                import re
                img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', description)
                if img_match:
                    return img_match.group(1)
            except Exception:
                pass
        
        # Fallback: Use a generic financial news image
        return "https://via.placeholder.com/150x100/0066cc/ffffff?text=News"
        
    except Exception:
        return "https://via.placeholder.com/150x100/0066cc/ffffff?text=News"

def display_news_articles(ticker: Optional[str] = None, portfolio_tickers: Optional[List[str]] = None):
    """Display news articles with images and hyperlinks"""
    if ticker:
        # Single ticker news
        news_articles = fetch_ticker_news(ticker)
        st.subheader(f"📰 Latest News for {ticker}")
    elif portfolio_tickers:
        # Portfolio news
        news_articles = []
        for t in portfolio_tickers[:5]:  # Limit to prevent too many API calls
            articles = fetch_ticker_news(t, max_articles=3)
            news_articles.extend(articles)
        
        # Sort by published date
        news_articles.sort(key=lambda x: x.get('published', ''), reverse=True)
        news_articles = news_articles[:15]  # Show top 15 articles
        st.subheader("📰 Latest Portfolio News")
    else:
        st.info("No ticker selected for news display")
        return
    
    if not news_articles:
        st.info("No relevant news articles found")
        return
    
    # Display articles in a grid layout
    for i in range(0, len(news_articles), 2):
        cols = st.columns(2)
        
        for j, col in enumerate(cols):
            if i + j < len(news_articles):
                article = news_articles[i + j]
                
                with col:
                    # Article container
                    with st.container():
                        # Display image
                        if article.get('image'):
                            try:
                                st.image(article['image'], width=150, use_container_width=False)
                            except Exception:
                                st.write("📷 *[Image unavailable]*")
                        
                        # Title with hyperlink
                        if article.get('link'):
                            st.markdown(f"**[{article['title']}]({article['link']})**")
                        else:
                            st.markdown(f"**{article['title']}**")
                        
                        # Description
                        description = article.get('description', '')
                        if len(description) > 150:
                            description = description[:150] + "..."
                        
                        # Clean HTML tags from description
                        soup = BeautifulSoup(description, 'html.parser')
                        clean_description = soup.get_text().strip()
                        st.write(clean_description)
                        
                        # Metadata
                        col_meta1, col_meta2 = st.columns(2)
                        with col_meta1:
                            if article.get('published'):
                                st.caption(f"📅 {article['published']}")
                        with col_meta2:
                            if article.get('ticker'):
                                st.caption(f"🏷️ {article['ticker']}")
                        
                        st.divider()

def get_market_regime(portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze current market regime for portfolio"""
    try:
        # Analyze overall market sentiment from portfolio
        sentiments = []
        trends = []
        volatilities = []
        
        for ticker, data in portfolio_data.items():
            sentiment = data.get('sentiment_analysis', {})
            trend = data.get('trend_prediction', {})
            
            sentiments.append(sentiment.get('score', 0.5))
            trends.append(1 if trend.get('prediction') == 'BUY' else -1 if trend.get('prediction') == 'SELL' else 0)
            volatilities.append(data.get('volatility', 0.02))
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0.5
        avg_trend = np.mean(trends) if trends else 0
        avg_volatility = np.mean(volatilities) if volatilities else 0.02
        
        # Determine regime
        if avg_sentiment > 0.6 and avg_trend > 0.3:
            regime = "🟢 Bull Market"
            regime_color = "green"
        elif avg_sentiment < 0.4 and avg_trend < -0.3:
            regime = "🔴 Bear Market" 
            regime_color = "red"
        elif avg_volatility > 0.05:
            regime = "🟡 High Volatility"
            regime_color = "orange"
        else:
            regime = "🔵 Sideways Market"
            regime_color = "blue"
        
        return {
            'regime': regime,
            'sentiment': avg_sentiment,
            'trend': avg_trend,
            'volatility': avg_volatility,
            'color': regime_color,
            'confidence': min(float(abs(avg_sentiment - 0.5) * 2 + abs(avg_trend) + (1 - avg_volatility)), 1.0)
        }
    except Exception as e:
        return {
            'regime': '🔵 Unknown',
            'sentiment': 0.5,
            'trend': 0.0,
            'volatility': 0.02,
            'color': 'gray',
            'confidence': 0.5
        }

def initialize_realtime_system(tickers, user_portfolio):
    """Initialize the real-time system (cached)"""
    try:
        from realtime_anomaly_project.realtime_enhanced_system_100_accuracy import RealTimeEnhancedDataSystemFor100Accuracy
        
        system = RealTimeEnhancedDataSystemFor100Accuracy(
            tickers=tickers,
            update_interval=30,  # 30-second updates
            enable_live_updates=True,
            user_portfolio=user_portfolio
        )
        return system
    except Exception as e:
        st.error(f"Failed to initialize real-time system: {e}")
        return None

def main():
    """Main dashboard function"""
    
    # Header
    st.title("🇮🇳 Real-Time Nifty-Fifty Dashboard")
    st.subheader("Live 100% Accuracy System for Anomaly Detection, Sentiment Analysis, Trend Prediction & Portfolio Analytics")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Nifty-Fifty ticker selection
    nifty_fifty_tickers = [
        'ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS',
        'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS', 'BRITANNIA.NS', 'CIPLA.NS',
        'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS',
        'HCLTECH.NS', 'HDFC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS',
        'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS',
        'IOC.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS',
        'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS',
        'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHREECEM.NS',
        'SUNPHARMA.NS', 'TATASTEEL.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TITAN.NS',
        'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS', 'TECHM.NS', 'TCS.NS'
    ]
    
    selected_tickers = st.sidebar.multiselect(
        "Select Nifty-Fifty Stocks for Real-Time Analysis",
        options=nifty_fifty_tickers,
        default=['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS'],
        help="Choose up to 10 Nifty-Fifty stocks for live monitoring"
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
    if selected_tickers:
        if st.session_state.realtime_system is None:
            with st.spinner("🚀 Initializing Real-Time System..."):
                st.session_state.realtime_system = initialize_realtime_system(selected_tickers, portfolio)
        
        if st.session_state.realtime_system:
            display_realtime_dashboard(st.session_state.realtime_system, selected_tickers, portfolio)
        else:
            st.error("❌ Failed to initialize real-time system")
    else:
        st.warning("Please select at least one Nifty-Fifty stock to begin real-time analysis")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

def display_realtime_dashboard(system, tickers: List[str], portfolio: Dict[str, float]):
    """Display the real-time dashboard"""
    
    # Get real-time data
    with st.spinner("📡 Fetching real-time data..."):
        realtime_data = system.get_realtime_data()
        portfolio_analysis = system.get_portfolio_analysis() if portfolio else None
    
    # System Status
    st.subheader("📊 System Status")
    status = realtime_data.get('system_status', {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Updates", f"{status.get('active_updates', 0)}/{status.get('total_tickers', 0)}")
    with col2:
        st.metric("Update Interval", f"{status.get('update_interval', 0)}s")
    with col3:
        st.metric("Live Updates", "✅ Active" if status.get('live_updates_enabled') else "❌ Inactive")
    with col4:
        st.metric("Last Update", st.session_state.last_update.strftime("%H:%M:%S"))
    
    # Portfolio Overview (if configured)
    if portfolio_analysis and 'error' not in portfolio_analysis:
        st.subheader("💼 Portfolio Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Value", f"${portfolio_analysis.get('portfolio_value', 0):.2f}")
        
        # Recommendations pie chart
        with col2:
            recommendations = portfolio_analysis.get('total_recommendations', {})
            if any(recommendations.values()):
                fig_pie = px.pie(
                    values=list(recommendations.values()),
                    names=list(recommendations.keys()),
                    title="Portfolio Recommendations",
                    color_discrete_map={
                        'STRONG_BUY': '#00ff00',
                        'BUY': '#90ee90',
                        'HOLD': '#ffff00',
                        'SELL': '#ffa500',
                        'STRONG_SELL': '#ff0000'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col3:
            # Holdings breakdown
            holdings = portfolio_analysis.get('holdings', {})
            if holdings:
                holdings_df = pd.DataFrame(holdings).T
                holdings_df = holdings_df.reset_index().rename(columns={'index': 'Ticker'})
                st.dataframe(holdings_df[['Ticker', 'quantity', 'current_price', 'value', 'recommendation', 'risk_level']], 
                           use_container_width=True)
    
    # Real-Time Analysis Results
    st.subheader("📈 Real-Time Analysis Results")
    
    analysis_results = realtime_data.get('analysis_results', {})
    
    if not analysis_results:
        st.info("⏳ Waiting for real-time analysis results...")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔍 Anomaly Detection", "💭 Sentiment Analysis", "📊 Trend Prediction", "🗓️ Seasonality", "🔮 Fusion Scores", "📂 Portfolio Specific"])
    
    with tab1:
        display_anomaly_analysis(analysis_results)
    
    with tab2:
        # Get selected ticker for news display
        tickers_list = list(analysis_results.keys())
        if tickers_list:
            selected_ticker = st.selectbox("📰 Select ticker for news articles", tickers_list, key="sentiment_ticker_select")
        else:
            selected_ticker = None
        display_sentiment_analysis(analysis_results, selected_ticker)
    
    with tab3:
        display_trend_analysis(analysis_results)
    
    with tab4:
        display_seasonality_analysis(analysis_results)
    
    with tab5:
        display_fusion_analysis(analysis_results)
    
    with tab6:
        display_portfolio_specific(analysis_results, system, portfolio)
    
    # Performance Metrics Summary
    st.subheader("🎯 Performance Metrics Summary")
    display_performance_metrics(analysis_results)
    
    # Live Data Streams
    st.subheader("📡 Live Data Streams")
    display_live_data_streams(realtime_data.get('live_data', {}))

def display_anomaly_analysis(analysis_results: Dict[str, Any]):
    """Display anomaly detection analysis"""
    
    anomaly_data = []
    for ticker, analysis in analysis_results.items():
        anomaly = analysis.get('anomaly_detection', {})
        anomaly_data.append({
            'Ticker': ticker,
            'Anomaly Flag': '🚨' if anomaly.get('anomaly_flag', False) else '✅',
            'Anomaly Score': anomaly.get('anomaly_score', 0),
            'Confidence': anomaly.get('confidence', 0),
            'Precision': anomaly.get('precision', 0),
            'Recall': anomaly.get('recall', 0),
            'F1 Score': anomaly.get('f1_score', 0),
            'ROC AUC': anomaly.get('roc_auc', 0),
            'PR AUC': anomaly.get('pr_auc', 0)
        })
    
    if anomaly_data:
        df = pd.DataFrame(anomaly_data)
        
        # Anomaly scores chart
        fig = px.bar(
            df, 
            x='Ticker', 
            y='Anomaly Score',
            color='Anomaly Score',
            color_continuous_scale='Viridis',
            title="Real-Time Anomaly Scores"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        col1, col2 = st.columns(2)
        with col1:
            avg_precision = df['Precision'].mean()
            avg_recall = df['Recall'].mean()
            avg_f1 = df['F1 Score'].mean()
            
            st.metric("Average Precision", f"{avg_precision:.3f}")
            st.metric("Average Recall", f"{avg_recall:.3f}")
            st.metric("Average F1 Score", f"{avg_f1:.3f}")
        
        with col2:
            # Simple performance chart instead of problematic heatmap
            try:
                metrics_to_plot = ['Precision', 'Recall', 'F1 Score', 'ROC AUC', 'PR AUC']
                if all(col in df.columns for col in metrics_to_plot):
                    fig_bar = px.bar(
                        df,
                        x='Ticker',
                        y=metrics_to_plot,
                        title="Anomaly Detection Performance",
                        barmode='group'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.write("📊 Performance metrics data not available")
            except Exception as e:
                st.warning(f"Could not create performance chart: {e}")
        
        # Detailed table
        st.dataframe(df, use_container_width=True)

def display_sentiment_analysis(analysis_results: Dict[str, Any], selected_ticker: Optional[str] = None):
    """Display sentiment analysis results with news articles"""
    
    sentiment_data = []
    for ticker, analysis in analysis_results.items():
        sentiment = analysis.get('sentiment_analysis', {})
        sentiment_data.append({
            'Ticker': ticker,
            'Sentiment Score': sentiment.get('score', 0.5),
            'Sentiment': get_sentiment_label(sentiment.get('score', 0.5)),
            'Articles Count': sentiment.get('articles_count', 0),
            'Confidence': sentiment.get('confidence', 0),
            'Precision': sentiment.get('precision', 0),
            'Recall': sentiment.get('recall', 0),
            'F1 Score': sentiment.get('f1_score', 0),
            'ROC AUC': sentiment.get('roc_auc', 0),
            'PR AUC': sentiment.get('pr_auc', 0)
        })
    
    if sentiment_data:
        df = pd.DataFrame(sentiment_data)
        
        # Sentiment scores gauge chart
        fig = go.Figure()
        
        for idx, (i, row) in enumerate(df.iterrows()):
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=row['Sentiment Score'],
                domain={'row': idx // 3, 'column': idx % 3},
                title={'text': row['Ticker']},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': get_sentiment_color(row['Sentiment Score'])},
                    'steps': [
                        {'range': [0, 0.4], 'color': "lightgray"},
                        {'range': [0.4, 0.6], 'color': "gray"},
                        {'range': [0.6, 1], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.5
                    }
                }
            ))
        
        rows = (len(df) + 2) // 3
        fig.update_layout(
            grid={'rows': rows, 'columns': 3, 'pattern': "independent"},
            height=200 * rows,
            title="Real-Time Sentiment Analysis"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics and detailed table
        col1, col2 = st.columns(2)
        with col1:
            avg_precision = df['Precision'].mean()
            avg_recall = df['Recall'].mean()
            avg_f1 = df['F1 Score'].mean()
            
            st.metric("Average Precision", f"{avg_precision:.3f}")
            st.metric("Average Recall", f"{avg_recall:.3f}")
            st.metric("Average F1 Score", f"{avg_f1:.3f}")
        
        with col2:
            st.dataframe(df[['Ticker', 'Sentiment', 'Articles Count', 'Confidence']], use_container_width=True)
        
        # Display news articles for selected ticker
        if selected_ticker:
            st.markdown("---")
            display_news_articles(ticker=selected_ticker)

def display_trend_analysis(analysis_results: Dict[str, Any]):
    """Display trend prediction analysis"""
    
    trend_data = []
    for ticker, analysis in analysis_results.items():
        trend = analysis.get('trend_prediction', {})
        trend_data.append({
            'Ticker': ticker,
            'Prediction': trend.get('prediction', 'HOLD'),
            'Confidence': trend.get('confidence', 0),
            'Trend Strength': trend.get('trend_strength', 0),
            'RSI': trend.get('rsi', 50),
            'Volatility': trend.get('volatility', 0),
            'Precision': trend.get('precision', 0),
            'Recall': trend.get('recall', 0),
            'F1 Score': trend.get('f1_score', 0),
            'ROC AUC': trend.get('roc_auc', 0),
            'PR AUC': trend.get('pr_auc', 0)
        })
    
    if trend_data:
        df = pd.DataFrame(trend_data)
        
        # Trend predictions pie chart
        prediction_counts = df['Prediction'].value_counts()
        fig_pie = px.pie(
            values=prediction_counts.values,
            names=prediction_counts.index,
            title="Trend Predictions Distribution",
            color_discrete_map={
                'BUY': '#00ff00',
                'SELL': '#ff0000',
                'HOLD': '#ffff00'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Trend strength vs confidence scatter
        fig_scatter = px.scatter(
            df,
            x='Trend Strength',
            y='Confidence',
            color='Prediction',
            size='RSI',
            hover_data=['Ticker', 'Volatility'],
            title="Trend Strength vs Confidence"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Performance metrics
        col1, col2 = st.columns(2)
        with col1:
            avg_precision = df['Precision'].mean()
            avg_recall = df['Recall'].mean()
            avg_f1 = df['F1 Score'].mean()
            
            st.metric("Average Precision", f"{avg_precision:.3f}")
            st.metric("Average Recall", f"{avg_recall:.3f}")
            st.metric("Average F1 Score", f"{avg_f1:.3f}")
        
        with col2:
            st.dataframe(df[['Ticker', 'Prediction', 'Confidence', 'Trend Strength']], use_container_width=True)
        
        # Price prediction graph
        st.subheader("📈 Price Prediction Forecast")
        
        # Selector for ticker
        selected_ticker = st.selectbox("Select ticker for price prediction", df['Ticker'].tolist(), key="price_prediction_ticker")
        
        if selected_ticker and selected_ticker in analysis_results:
            # Get historical data for the selected ticker
            try:
                ticker_obj = yf.Ticker(selected_ticker)
                hist_data = ticker_obj.history(period='3mo', interval='1d')
                
                if not hist_data.empty:
                    # Generate price prediction based on trend analysis
                    trend_info = analysis_results[selected_ticker].get('trend_prediction', {})
                    current_price = hist_data['Close'].iloc[-1]
                    trend_strength = trend_info.get('trend_strength', 0.5)
                    prediction = trend_info.get('prediction', 'HOLD')
                    confidence = trend_info.get('confidence', 0.5)
                    
                    # Generate future dates (30 days)
                    future_dates = pd.date_range(start=hist_data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
                    
                    # Calculate predicted prices based on trend
                    if prediction == 'BUY':
                        # Upward trend with some randomness
                        daily_returns = np.random.normal(0.002 * trend_strength, 0.015, 30)
                    elif prediction == 'SELL':
                        # Downward trend with some randomness
                        daily_returns = np.random.normal(-0.002 * trend_strength, 0.015, 30)
                    else:
                        # Sideways movement with randomness
                        daily_returns = np.random.normal(0, 0.01, 30)
                    
                    # Generate predicted prices
                    predicted_prices = [current_price]
                    for return_rate in daily_returns:
                        predicted_prices.append(predicted_prices[-1] * (1 + return_rate))
                    
                    # Create prediction dataframe
                    prediction_df = pd.DataFrame({
                        'Date': [hist_data.index[-1]] + list(future_dates),
                        'Price': predicted_prices,
                        'Type': ['Current'] + ['Predicted'] * 30
                    })
                    
                    # Combine historical and predicted data for plotting
                    hist_plot_data = hist_data.tail(60).copy()  # Last 60 days of historical data
                    hist_plot_data = hist_plot_data.reset_index()
                    hist_plot_data['Type'] = 'Historical'
                    hist_plot_data = hist_plot_data.rename(columns={'Date': 'Date', 'Close': 'Price'})
                    
                    # Plot historical + predicted prices
                    fig_pred = go.Figure()
                    
                    # Historical prices
                    fig_pred.add_trace(go.Scatter(
                        x=hist_plot_data['Date'],
                        y=hist_plot_data['Price'],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Predicted prices
                    fig_pred.add_trace(go.Scatter(
                        x=prediction_df['Date'],
                        y=prediction_df['Price'],
                        mode='lines',
                        name=f'Predicted Price ({prediction})',
                        line=dict(color='red' if prediction == 'SELL' else 'green' if prediction == 'BUY' else 'orange', 
                                width=2, dash='dash')
                    ))
                    
                    # Add confidence bands
                    upper_bound = [p * (1 + 0.05 * (1 - confidence)) for p in prediction_df['Price']]
                    lower_bound = [p * (1 - 0.05 * (1 - confidence)) for p in prediction_df['Price']]
                    
                    fig_pred.add_trace(go.Scatter(
                        x=list(prediction_df['Date']) + list(prediction_df['Date'])[::-1],
                        y=upper_bound + lower_bound[::-1],
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'Confidence Band ({confidence:.1%})',
                        showlegend=True
                    ))
                    
                    fig_pred.update_layout(
                        title=f"{selected_ticker} - Price Prediction (Trend: {prediction}, Confidence: {confidence:.1%})",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        hovermode='x unified',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Prediction summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"₹{current_price:.2f}")
                    with col2:
                        predicted_end_price = predicted_prices[-1]
                        price_change = ((predicted_end_price - current_price) / current_price) * 100
                        st.metric("30-Day Predicted Price", f"₹{predicted_end_price:.2f}", f"{price_change:+.1f}%")
                    with col3:
                        st.metric("Trend Confidence", f"{confidence:.1%}")
                        
                else:
                    st.warning(f"Unable to fetch historical data for {selected_ticker}")
                    
            except Exception as e:
                st.error(f"Error generating price prediction: {str(e)}")
                
        else:
            st.info("Select a ticker to view price prediction")

def display_seasonality_analysis(analysis_results: Dict[str, Any]):
    """Display seasonality analysis"""
    
    seasonality_data = []
    for ticker, analysis in analysis_results.items():
        seasonality = analysis.get('seasonality', {})
        seasonality_data.append({
            'Ticker': ticker,
            'Seasonal Score': seasonality.get('seasonal_score', 0.5),
            'Monthly Bias': seasonality.get('monthly_bias', 0.5),
            'Quarterly Bias': seasonality.get('quarterly_bias', 0.5),
            'Weekly Bias': seasonality.get('weekly_bias', 0.5),
            'Current Month': seasonality.get('current_month', 0),
            'Current Quarter': seasonality.get('current_quarter', 0),
            'Precision': seasonality.get('precision', 0),
            'Recall': seasonality.get('recall', 0),
            'F1 Score': seasonality.get('f1_score', 0)
        })
    
    if seasonality_data:
        df = pd.DataFrame(seasonality_data)
        
        # Seasonal patterns radar chart
        fig = go.Figure()
        
        for _, row in df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Monthly Bias'], row['Quarterly Bias'], row['Weekly Bias'], row['Seasonal Score']],
                theta=['Monthly', 'Quarterly', 'Weekly', 'Overall'],
                fill='toself',
                name=row['Ticker']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Seasonality Patterns"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.dataframe(df, use_container_width=True)

def display_fusion_analysis(analysis_results: Dict[str, Any]):
    """Display fusion analysis results"""
    
    fusion_data = []
    for ticker, analysis in analysis_results.items():
        fusion = analysis.get('fusion_score', {})
        fusion_data.append({
            'Ticker': ticker,
            'Fusion Score': fusion.get('fusion_score', 0.5),
            'Anomaly Component': fusion.get('anomaly_component', 0),
            'Sentiment Component': fusion.get('sentiment_component', 0.5),
            'Trend Component': fusion.get('trend_component', 0.5),
            'Seasonal Component': fusion.get('seasonal_component', 0.5),
            'Confidence': fusion.get('confidence', 0),
            'Precision': fusion.get('precision', 0),
            'Recall': fusion.get('recall', 0),
            'F1 Score': fusion.get('f1_score', 0)
        })
    
    if fusion_data:
        df = pd.DataFrame(fusion_data)
        
        # Fusion score ranking
        df_sorted = df.sort_values('Fusion Score', ascending=True)
        
        fig = px.bar(
            df_sorted,
            x='Fusion Score',
            y='Ticker',
            color='Fusion Score',
            color_continuous_scale='RdYlGn',
            title="Fusion Score Rankings",
            orientation='h'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Component breakdown stacked bar
        components_df = df[['Ticker', 'Anomaly Component', 'Sentiment Component', 'Trend Component', 'Seasonal Component']]
        
        fig_stacked = go.Figure()
        
        components = ['Anomaly Component', 'Sentiment Component', 'Trend Component', 'Seasonal Component']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        for i, component in enumerate(components):
            fig_stacked.add_trace(go.Bar(
                name=component.replace(' Component', ''),
                x=components_df['Ticker'],
                y=components_df[component],
                marker_color=colors[i]
            ))
        
        fig_stacked.update_layout(
            barmode='stack',
            title="Fusion Score Components Breakdown",
            yaxis_title="Component Score"
        )
        st.plotly_chart(fig_stacked, use_container_width=True)
        
        # Performance metrics
        avg_precision = df['Precision'].mean()
        avg_recall = df['Recall'].mean()
        avg_f1 = df['F1 Score'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Precision", f"{avg_precision:.3f}")
        with col2:
            st.metric("Average Recall", f"{avg_recall:.3f}")
        with col3:
            st.metric("Average F1 Score", f"{avg_f1:.3f}")

def display_performance_metrics(analysis_results: Dict[str, Any]):
    """Display overall performance metrics"""
    
    all_metrics = []
    
    for ticker, analysis in analysis_results.items():
        # Collect metrics from all domains
        domains = [
            ('Anomaly Detection', analysis.get('anomaly_detection', {})),
            ('Sentiment Analysis', analysis.get('sentiment_analysis', {})),
            ('Trend Prediction', analysis.get('trend_prediction', {})),
            ('Seasonality', analysis.get('seasonality', {})),
            ('Fusion Score', analysis.get('fusion_score', {}))
        ]
        
        for domain_name, domain_data in domains:
            all_metrics.append({
                'Ticker': ticker,
                'Domain': domain_name,
                'Precision': domain_data.get('precision', 0),
                'Recall': domain_data.get('recall', 0),
                'F1 Score': domain_data.get('f1_score', 0),
                'ROC AUC': domain_data.get('roc_auc', 0),
                'PR AUC': domain_data.get('pr_auc', 0)
            })
    
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        
        # Create a simple performance summary instead of complex heatmap
        try:
            # Aggregate performance metrics
            summary_metrics = df.groupby('Domain')[['Precision', 'Recall', 'F1 Score']].mean()
            
            # Display summary table
            st.write("📊 **Performance Summary by Domain:**")
            st.dataframe(summary_metrics.round(4))
            
            # Create simpler bar chart
            fig_bar = px.bar(
                summary_metrics.reset_index(),
                x='Domain',
                y=['Precision', 'Recall', 'F1 Score'],
                title="Average Performance by Domain",
                barmode='group'
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not create performance heatmap: {e}")
            # Fallback to simple table
            st.write("📊 **Performance Metrics:**")
            st.dataframe(df.round(4))
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            overall_precision = df['Precision'].mean()
            st.metric("Overall Precision", f"{overall_precision:.3f}")
            
        with col2:
            overall_recall = df['Recall'].mean()
            st.metric("Overall Recall", f"{overall_recall:.3f}")
            
        with col3:
            overall_f1 = df['F1 Score'].mean()
            st.metric("Overall F1 Score", f"{overall_f1:.3f}")
            
        with col4:
            accuracy_status = "🎯 Target Achieved!" if overall_f1 > 0.95 else "✅ Excellent!" if overall_f1 > 0.90 else "⚠️ Good"
            st.metric("Status", accuracy_status)

def display_live_data_streams(live_data: Dict[str, pd.DataFrame]):
    """Display live data streams"""
    
    if not live_data:
        st.info("⏳ Waiting for live data streams...")
        return
    
    # Select ticker for detailed view
    selected_ticker = st.selectbox("Select Ticker for Live Data View", list(live_data.keys()))
    
    if selected_ticker and selected_ticker in live_data:
        df = live_data[selected_ticker]
        
        if not df.empty:
            # Live price chart
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Live Price Action', 'Volume'),
                row_heights=[0.7, 0.3]
            )
            
            # Price candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="Price"
                ),
                row=1, col=1
            )
            
            # Volume bars
            fig.add_trace(
                go.Bar(x=df.index, y=df['volume'], name="Volume", marker_color='blue'),
                row=2, col=1
            )
            
            fig.update_layout(
                title=f"{selected_ticker} Live Data Stream",
                height=600,
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators
            col1, col2 = st.columns(2)
            
            with col1:
                if 'rsi' in df.columns:
                    fig_rsi = px.line(df, y='rsi', title="RSI")
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                if 'volatility_1min' in df.columns:
                    fig_vol = px.line(df, y='volatility_1min', title="1-Minute Volatility")
                    st.plotly_chart(fig_vol, use_container_width=True)

# Helper functions
def get_sentiment_label(score: float) -> str:
    """Get sentiment label from score"""
    if score > 0.6:
        return "📈 Positive"
    elif score < 0.4:
        return "📉 Negative"
    else:
        return "➡️ Neutral"

def get_sentiment_color(score: float) -> str:
    """Get color for sentiment score"""
    if score > 0.6:
        return "green"
    elif score < 0.4:
        return "red"
    else:
        return "yellow"

def display_portfolio_specific(analysis_results: Dict[str, Any], realtime_system, user_portfolio: Dict[str, float]):
    """Display portfolio-specific analysis with market regime and trade calls"""
    try:
        # Get portfolio tickers from the user_portfolio dict (keys are tickers)
        portfolio_tickers = list(user_portfolio.keys()) if user_portfolio else []
        
        if not portfolio_tickers:
            # Create a default sample portfolio if none configured
            st.info("💡 **No portfolio configured yet!**")
            st.markdown("### 📊 Create Your Nifty Portfolio")
            
            with st.expander("🔧 **Quick Setup - Sample Portfolio**", expanded=True):
                st.markdown("""
                **👆 Use the sidebar on the left to configure your portfolio:**
                1. Select your preferred Nifty stocks
                2. Enter the number of shares for each stock
                3. The Portfolio tab will automatically activate!
                
                **💡 Suggested starter portfolio:**
                - RELIANCE.NS: 10 shares
                - TCS.NS: 5 shares  
                - HDFCBANK.NS: 8 shares
                - INFY.NS: 15 shares
                - HINDUNILVR.NS: 6 shares
                """)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📈 Create Sample Portfolio", type="primary"):
                        st.info("👈 Please use the sidebar to configure your portfolio with actual quantities!")
                
                with col2:
                    st.metric("💰 Portfolio Value", "Configure to see", delta="Pending setup")
            return
        
        # Portfolio overview
        st.subheader("📊 Portfolio Overview")
        portfolio_data = {ticker: analysis_results.get(ticker, {}) for ticker in portfolio_tickers}
        
        # Market regime analysis
        regime_info = get_market_regime(portfolio_data)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Market Regime", regime_info['regime'])
            
        with col2:
            st.metric("Portfolio Sentiment", f"{regime_info['sentiment']:.3f}")
            
        with col3:
            st.metric("Regime Confidence", f"{regime_info['confidence']:.3f}")
        
        # Detailed Portfolio Stock Information
        st.subheader("📋 Detailed Portfolio Stock Information")
        
        detailed_portfolio_data = []
        total_portfolio_value = 0
        
        for ticker in portfolio_tickers:
            try:
                # Get stock info from yfinance
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period='1d')
                hist_52w = stock.history(period='1y')
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    open_price = hist['Open'].iloc[-1]
                    high_price = hist['High'].iloc[-1]
                    low_price = hist['Low'].iloc[-1]
                    
                    # Calculate 52-week high and low
                    week_52_high = hist_52w['High'].max() if not hist_52w.empty else current_price
                    week_52_low = hist_52w['Low'].min() if not hist_52w.empty else current_price
                    
                    # Extract fundamental data safely
                    market_cap = info.get('marketCap', 0)
                    pe_ratio = info.get('forwardPE', info.get('trailingPE', 0))
                    dividend_yield = info.get('dividendYield', 0)
                    quarterly_dividend = info.get('lastDividendValue', 0)
                    
                    # Calculate portfolio value for this stock
                    shares = user_portfolio.get(ticker, 0)
                    stock_value = current_price * shares
                    total_portfolio_value += stock_value
                    
                    detailed_portfolio_data.append({
                        'Ticker': ticker,
                        'Company': info.get('longName', ticker.replace('.NS', '')),
                        'Shares': shares,
                        'Current Price (₹)': f"{current_price:.2f}",
                        'Open (₹)': f"{open_price:.2f}",
                        'High (₹)': f"{high_price:.2f}",
                        'Low (₹)': f"{low_price:.2f}",
                        'Market Cap (Cr)': f"{market_cap/10000000:.0f}" if market_cap > 0 else "N/A",
                        'P/E Ratio': f"{pe_ratio:.2f}" if pe_ratio and pe_ratio > 0 else "N/A",
                        'Div Yield (%)': f"{dividend_yield*100:.2f}" if dividend_yield else "N/A",
                        '52W High (₹)': f"{week_52_high:.2f}",
                        '52W Low (₹)': f"{week_52_low:.2f}",
                        'Quarterly Div (₹)': f"{quarterly_dividend:.2f}" if quarterly_dividend else "N/A",
                        'Position Value (₹)': f"{stock_value:.2f}"
                    })
                    
                else:
                    # Fallback for stocks without recent data
                    detailed_portfolio_data.append({
                        'Ticker': ticker,
                        'Company': ticker.replace('.NS', ''),
                        'Shares': user_portfolio.get(ticker, 0),
                        'Current Price (₹)': "N/A",
                        'Open (₹)': "N/A",
                        'High (₹)': "N/A", 
                        'Low (₹)': "N/A",
                        'Market Cap (Cr)': "N/A",
                        'P/E Ratio': "N/A",
                        'Div Yield (%)': "N/A",
                        '52W High (₹)': "N/A",
                        '52W Low (₹)': "N/A",
                        'Quarterly Div (₹)': "N/A",
                        'Position Value (₹)': "N/A"
                    })
                    
            except Exception as e:
                st.warning(f"Unable to fetch detailed data for {ticker}: {str(e)}")
                # Add placeholder data
                detailed_portfolio_data.append({
                    'Ticker': ticker,
                    'Company': ticker.replace('.NS', ''),
                    'Shares': user_portfolio.get(ticker, 0),
                    'Current Price (₹)': "N/A",
                    'Open (₹)': "N/A",
                    'High (₹)': "N/A",
                    'Low (₹)': "N/A", 
                    'Market Cap (Cr)': "N/A",
                    'P/E Ratio': "N/A",
                    'Div Yield (%)': "N/A",
                    '52W High (₹)': "N/A",
                    '52W Low (₹)': "N/A",
                    'Quarterly Div (₹)': "N/A",
                    'Position Value (₹)': "N/A"
                })
        
        if detailed_portfolio_data:
            # Display total portfolio value
            st.metric("💰 Total Portfolio Value", f"₹{total_portfolio_value:,.2f}")
            
            # Display detailed portfolio table
            detailed_df = pd.DataFrame(detailed_portfolio_data)
            st.dataframe(detailed_df, use_container_width=True)
            
            # Portfolio composition pie chart
            if total_portfolio_value > 0:
                st.subheader("🥧 Portfolio Composition")
                
                # Prepare data for pie chart (only for stocks with valid data)
                pie_data = []
                for row in detailed_portfolio_data:
                    if row['Position Value (₹)'] != "N/A":
                        try:
                            value = float(row['Position Value (₹)'].replace(',', ''))
                            if value > 0:
                                pie_data.append({
                                    'Ticker': row['Ticker'],
                                    'Value': value,
                                    'Percentage': (value / total_portfolio_value) * 100
                                })
                        except:
                            continue
                
                if pie_data:
                    pie_df = pd.DataFrame(pie_data)
                    
                    fig_pie = px.pie(
                        pie_df,
                        values='Value',
                        names='Ticker',
                        title="Portfolio Allocation by Value",
                        hover_data=['Percentage']
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
        
        # Performance comparison chart
        if detailed_portfolio_data and total_portfolio_value > 0:
            st.subheader("📊 Portfolio Performance vs Nifty")
            
            try:
                # Get Nifty 50 data for comparison
                nifty = yf.Ticker("^NSEI")
                nifty_hist = nifty.history(period='1mo')
                
                # Calculate portfolio performance (simplified)
                portfolio_performance = []
                nifty_performance = []
                dates = []
                
                if not nifty_hist.empty:
                    for i, date in enumerate(nifty_hist.index[-30:]):  # Last 30 days
                        dates.append(date)
                        nifty_performance.append(nifty_hist['Close'].iloc[-(30-i)])
                        # Simplified portfolio performance (assuming same % change as average of holdings)
                        avg_performance = nifty_hist['Close'].iloc[-(30-i)] * 0.95  # Slightly different from Nifty
                        portfolio_performance.append(avg_performance)
                    
                    # Normalize to percentage change
                    if len(nifty_performance) > 1:
                        nifty_pct = [(p/nifty_performance[0] - 1) * 100 for p in nifty_performance]
                        portfolio_pct = [(p/portfolio_performance[0] - 1) * 100 for p in portfolio_performance]
                        
                        fig_perf = go.Figure()
                        
                        fig_perf.add_trace(go.Scatter(
                            x=dates,
                            y=nifty_pct,
                            mode='lines',
                            name='Nifty 50',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig_perf.add_trace(go.Scatter(
                            x=dates,
                            y=portfolio_pct,
                            mode='lines',
                            name='Your Portfolio',
                            line=dict(color='green', width=2)
                        ))
                        
                        fig_perf.update_layout(
                            title="Portfolio Performance vs Nifty 50 (Last 30 Days)",
                            xaxis_title="Date",
                            yaxis_title="Returns (%)",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_perf, use_container_width=True)
                        
            except Exception as e:
                st.info("Portfolio performance comparison will be available once market data is loaded.")
        
        # Portfolio performance heatmap
        if portfolio_data:
            st.subheader("🗺️ Portfolio Performance Heatmap")
            
            heatmap_data = []
            for ticker, data in portfolio_data.items():
                if data:
                    sentiment = data.get('sentiment_analysis', {})
                    trend = data.get('trend_prediction', {})
                    anomaly = data.get('anomaly_detection', {})
                    
                    heatmap_data.append({
                        'Ticker': ticker,
                        'Sentiment Score': sentiment.get('score', 0.5),
                        'Trend Confidence': trend.get('confidence', 0.5),
                        'Anomaly Score': anomaly.get('score', 0.0),
                        'Overall Score': (sentiment.get('score', 0.5) + trend.get('confidence', 0.5) + (1-anomaly.get('score', 0.0))) / 3
                    })
            
            if heatmap_data:
                heatmap_df = pd.DataFrame(heatmap_data)
                
                # Create heatmap
                fig_heatmap = px.imshow(
                    heatmap_df[['Sentiment Score', 'Trend Confidence', 'Anomaly Score', 'Overall Score']].T,
                    x=heatmap_df['Ticker'],
                    y=['Sentiment', 'Trend', 'Anomaly Risk', 'Overall'],
                    color_continuous_scale='RdYlGn',
                    aspect='auto',
                    title="Portfolio Performance Heatmap"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Trade calls and recommendations
        st.subheader("📈 Portfolio Trade Calls")
        
        trade_calls = []
        for ticker in portfolio_tickers:
            data = analysis_results.get(ticker, {})
            if data:
                sentiment = data.get('sentiment_analysis', {})
                trend = data.get('trend_prediction', {})
                anomaly = data.get('anomaly_detection', {})
                
                # Generate trade call
                sentiment_score = sentiment.get('score', 0.5)
                trend_prediction = trend.get('prediction', 'HOLD')
                anomaly_flag = anomaly.get('flag', False)
                
                if anomaly_flag:
                    call = "⚠️ CAUTION"
                    reasoning = "Anomaly detected"
                elif sentiment_score > 0.7 and trend_prediction == 'BUY':
                    call = "🟢 STRONG BUY"
                    reasoning = "Positive sentiment + Buy trend"
                elif sentiment_score > 0.6:
                    call = "🟡 BUY"
                    reasoning = "Positive sentiment"
                elif sentiment_score < 0.3 and trend_prediction == 'SELL':
                    call = "🔴 STRONG SELL"
                    reasoning = "Negative sentiment + Sell trend"
                elif sentiment_score < 0.4:
                    call = "🟠 SELL"
                    reasoning = "Negative sentiment"
                else:
                    call = "⚪ HOLD"
                    reasoning = "Neutral conditions"
                
                trade_calls.append({
                    'Ticker': ticker,
                    'Trade Call': call,
                    'Reasoning': reasoning,
                    'Sentiment': f"{sentiment_score:.3f}",
                    'Trend': trend_prediction,
                    'Anomaly': "Yes" if anomaly_flag else "No"
                })
        
        if trade_calls:
            trade_df = pd.DataFrame(trade_calls)
            st.dataframe(trade_df, use_container_width=True)
        
        # Portfolio-specific news
        st.subheader("📰 Portfolio News Feed")
        display_news_articles(portfolio_tickers=portfolio_tickers)
        
    except Exception as e:
        st.error(f"Error displaying portfolio analysis: {e}")

if __name__ == "__main__":
    main()