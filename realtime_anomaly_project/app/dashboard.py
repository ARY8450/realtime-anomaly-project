import sys
import os

# Ensure realtime_anomaly_project folder is on sys.path so "config", "data_ingestion", etc. can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
from config.settings import TICKERS
from data_ingestion.yahoo import data_storage  # In-memory storage for stock data
from statistical_anomaly.classical_methods import compute_anomalies
from statistical_anomaly.unsupervised import compute_unsupervised_anomalies

# Set up Streamlit page configuration
st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")

# Title of the dashboard
st.title("Real-Time Anomaly Detection Dashboard")

# Sidebar to select the ticker for analysis
ticker = st.sidebar.selectbox("Select Ticker", TICKERS)

# Display Stock Data
st.header(f"Stock Data for {ticker}")
df = data_storage.get(ticker)

if df is not None and not df.empty:
    st.write(df.tail(10))  # Display the latest 10 rows

    # Plot the stock price over time
    st.subheader(f"Stock Price Over Time for {ticker}")
    fig = px.line(df, x=df.index, y="close", title=f"Stock Price for {ticker}")
    st.plotly_chart(fig)

    # Add Z-score and RSI to the plot
    compute_anomalies()
    compute_unsupervised_anomalies()

    # Show the Z-score and RSI anomaly results
    st.subheader("Anomaly Detection Results")

    # Fetching the latest anomalies
    anomalies = {
        "Z-Score Anomalies": df["close"].pct_change().abs() > 2.5,  # Example threshold
        "RSI Anomalies": df["close"].pct_change().abs() > 0.05  # Example RSI-based threshold
    }

    # Display anomalies
    anomaly_data = pd.DataFrame(anomalies)
    anomaly_data.index = df.index

    st.write(anomaly_data.tail(10))

    # Plot the anomalies on a graph
    st.subheader(f"Anomalies Detected in {ticker}")

    anomaly_fig = px.scatter(df, x=df.index, y="close", color=anomaly_data["Z-Score Anomalies"].astype(str), 
                             title=f"Anomalies Detected in {ticker}")
    st.plotly_chart(anomaly_fig)

else:
    st.warning(f"No data available for {ticker}")

# Summary Section (e.g., news impacts, anomaly correlation)
st.header(f"Summary for {ticker}")
st.write("Summarize the stock trends, detected anomalies, and their impact here.")
