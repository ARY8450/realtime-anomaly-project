"""
Script to add analysis tab to existing dashboard
"""

import os
import shutil

def add_analysis_tab():
    """Add analysis tab to the existing dashboard"""
    
    # Read the existing dashboard file
    dashboard_file = "06_RealTime_Dashboard_100_Accuracy.py"
    
    if not os.path.exists(dashboard_file):
        print(f"Error: {dashboard_file} not found")
        return False
    
    # Create backup
    backup_file = f"{dashboard_file}.backup"
    shutil.copy2(dashboard_file, backup_file)
    print(f"Backup created: {backup_file}")
    
    # Read the file
    with open(dashboard_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the tabs line and modify it
    old_tabs_line = 'tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔍 Anomaly Detection", "💭 Sentiment Analysis", "📊 Trend Prediction", "🗓️ Seasonality", "🔮 Fusion Scores", "📂 Portfolio Specific"])'
    new_tabs_line = 'tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🔍 Anomaly Detection", "💭 Sentiment Analysis", "📊 Trend Prediction", "🗓️ Seasonality", "🔮 Fusion Scores", "📂 Portfolio Specific", "📈 Analysis Dashboard"])'
    
    if old_tabs_line in content:
        content = content.replace(old_tabs_line, new_tabs_line)
        
        # Add the analysis tab content
        analysis_tab_content = '''
    with tab7:
        st.header("📈 Analysis Dashboard")
        st.markdown("---")
        
        # Check if analysis data exists
        analysis_dir = "comprehensive_analysis"
        if not os.path.exists(analysis_dir):
            st.error("❌ No analysis data found. Please run the analysis components first.")
            st.info("💡 Run: `python simple_analysis_runner.py` to generate the analysis data")
        else:
            # Load analysis data
            try:
                import pandas as pd
                
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
                
                if backtesting_df is not None or summary_df is not None or metrics_df is not None:
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
                        if summary_df is not None and not summary_df.empty:
                            summary = summary_df.iloc[0]
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                st.metric("Trading Days", f"{summary.get('Total_Days', 'N/A')}")
                            with col2:
                                st.metric("Mean Error", f"{summary.get('Mean_Absolute_Percentage_Error', 'N/A')}%")
                            with col3:
                                st.metric("Direction Accuracy", f"{summary.get('Direction_Accuracy', 'N/A')}%")
                            with col4:
                                st.metric("Avg Confidence", f"{summary.get('Average_Confidence', 'N/A')}")
                            with col5:
                                st.metric("Anomaly Rate", f"{summary.get('Anomaly_Rate', 'N/A')}%")
                        else:
                            st.warning("No summary data available")
                        
                        st.header("📈 Quick Overview Charts")
                        if backtesting_df is not None and not backtesting_df.empty:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Backtesting chart
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=backtesting_df['Date'],
                                    y=backtesting_df['Actual_Price'],
                                    mode='lines+markers',
                                    name='Actual Price',
                                    line=dict(color='#2ecc71', width=3)
                                ))
                                fig.add_trace(go.Scatter(
                                    x=backtesting_df['Date'],
                                    y=backtesting_df['Predicted_Price'],
                                    mode='lines+markers',
                                    name='Predicted Price',
                                    line=dict(color='#3498db', width=3)
                                ))
                                fig.update_layout(
                                    title='Actual vs Predicted Prices',
                                    xaxis_title='Date',
                                    yaxis_title='Price',
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Anomaly chart
                                fig = go.Figure()
                                normal_data = backtesting_df[~backtesting_df['Is_Anomaly']]
                                anomaly_data = backtesting_df[backtesting_df['Is_Anomaly']]
                                
                                fig.add_trace(go.Scatter(
                                    x=normal_data['Date'],
                                    y=normal_data['Actual_Price'],
                                    mode='lines+markers',
                                    name='Normal',
                                    line=dict(color='#95a5a6', width=2)
                                ))
                                
                                if not anomaly_data.empty:
                                    fig.add_trace(go.Scatter(
                                        x=anomaly_data['Date'],
                                        y=anomaly_data['Actual_Price'],
                                        mode='markers',
                                        name='Anomaly',
                                        marker=dict(color='#e74c3c', size=12, symbol='x')
                                    ))
                                
                                fig.update_layout(
                                    title='Anomaly Detection Results',
                                    xaxis_title='Date',
                                    yaxis_title='Price',
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No backtesting data available")
                    
                    with analysis_tab2:
                        st.header("📈 Backtesting Results")
                        if backtesting_df is not None and not backtesting_df.empty:
                            # Candlestick chart
                            fig = make_subplots(
                                rows=2, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.1,
                                subplot_titles=('RELIANCE.NS - Actual vs Predicted Prices', 'Volume'),
                                row_heights=[0.7, 0.3]
                            )
                            
                            fig.add_trace(
                                go.Candlestick(
                                    x=backtesting_df['Date'],
                                    open=backtesting_df['Actual_Price'] * 0.99,
                                    high=backtesting_df['High'],
                                    low=backtesting_df['Low'],
                                    close=backtesting_df['Actual_Price'],
                                    name='Actual Price',
                                    increasing_line_color='green',
                                    decreasing_line_color='red'
                                ),
                                row=1, col=1
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=backtesting_df['Date'],
                                    y=backtesting_df['Predicted_Price'],
                                    mode='lines+markers',
                                    name='Predicted Price',
                                    line=dict(color='blue', width=2)
                                ),
                                row=1, col=1
                            )
                            
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
                                height=600
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Data table
                            st.subheader("📊 Backtesting Results Table")
                            display_df = backtesting_df[['Date', 'Actual_Price', 'Predicted_Price', 'Difference', 'Percentage_Error', 'Performance_Grade', 'Is_Anomaly']].head(10)
                            st.dataframe(display_df, use_container_width=True)
                        else:
                            st.warning("No backtesting data available")
                    
                    with analysis_tab3:
                        st.header("🔍 Anomaly Detection Analysis")
                        if backtesting_df is not None and not backtesting_df.empty:
                            # Anomaly detection chart
                            fig = go.Figure()
                            normal_data = backtesting_df[~backtesting_df['Is_Anomaly']]
                            anomaly_data = backtesting_df[backtesting_df['Is_Anomaly']]
                            
                            fig.add_trace(go.Scatter(
                                x=normal_data['Date'],
                                y=normal_data['Actual_Price'],
                                mode='lines+markers',
                                name='Normal',
                                line=dict(color='#95a5a6', width=2)
                            ))
                            
                            if not anomaly_data.empty:
                                fig.add_trace(go.Scatter(
                                    x=anomaly_data['Date'],
                                    y=anomaly_data['Actual_Price'],
                                    mode='markers',
                                    name='Anomaly',
                                    marker=dict(color='#e74c3c', size=12, symbol='x')
                                ))
                            
                            fig.update_layout(
                                title='Anomaly Detection Results',
                                xaxis_title='Date',
                                yaxis_title='Price',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Anomaly statistics
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
                        else:
                            st.warning("No anomaly data available")
                    
                    with analysis_tab4:
                        st.header("📊 Performance Analysis")
                        if backtesting_df is not None and not backtesting_df.empty:
                            # Performance chart
                            fig = make_subplots(
                                rows=2, cols=1,
                                subplot_titles=('Prediction Error Over Time', 'Model Confidence Over Time'),
                                vertical_spacing=0.1
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=backtesting_df['Date'],
                                    y=backtesting_df['Percentage_Error'],
                                    mode='lines+markers',
                                    name='Prediction Error %',
                                    line=dict(color='#e74c3c', width=2)
                                ),
                                row=1, col=1
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=backtesting_df['Date'],
                                    y=backtesting_df['Model_Confidence'],
                                    mode='lines+markers',
                                    name='Model Confidence',
                                    line=dict(color='#3498db', width=2)
                                ),
                                row=2, col=1
                            )
                            
                            fig.update_layout(
                                title='Performance Metrics Over Time',
                                height=600
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Performance metrics table
                            if metrics_df is not None and not metrics_df.empty:
                                st.subheader("📈 Performance Metrics")
                                st.dataframe(metrics_df, use_container_width=True)
                        else:
                            st.warning("No performance data available")
                    
                    with analysis_tab5:
                        st.header("🏗️ System Architecture")
                        # Simple architecture diagram
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
                        
                        # Architecture diagram placeholder
                        st.info("Architecture diagram will be displayed here")
                else:
                    st.warning("No analysis data available")
            except Exception as e:
                st.error(f"Error loading analysis data: {str(e)}")
'''
        
        # Find where to insert the analysis tab content
        # Look for the end of the existing tabs
        if 'with tab6:' in content:
            # Find the end of tab6 content
            tab6_end = content.find('with tab6:')
            if tab6_end != -1:
                # Find the end of the tab6 content
                lines = content[tab6_end:].split('\n')
                tab6_content_end = 0
                indent_level = None
                
                for i, line in enumerate(lines):
                    if line.strip().startswith('with tab6:'):
                        # Find the indentation level
                        indent_level = len(line) - len(line.lstrip())
                        continue
                    
                    if indent_level is not None and line.strip() and not line.startswith(' ' * indent_level) and not line.startswith('\t'):
                        tab6_content_end = tab6_end + len('\n'.join(lines[:i]))
                        break
                
                if tab6_content_end > 0:
                    # Insert the analysis tab content
                    content = content[:tab6_content_end] + analysis_tab_content + content[tab6_content_end:]
                else:
                    # Fallback: add at the end
                    content += analysis_tab_content
        
        # Write the modified content back
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Analysis tab added to {dashboard_file}")
        return True
    else:
        print("Could not find the tabs line to modify")
        return False

def main():
    """Main function"""
    print("Adding Analysis Tab to Existing Dashboard")
    print("=" * 50)
    
    success = add_analysis_tab()
    
    if success:
        print("\nAnalysis tab added successfully!")
        print("You can now run your existing dashboard and it will include the analysis tab.")
        print("Run: streamlit run 06_RealTime_Dashboard_100_Accuracy.py")
    else:
        print("\nFailed to add analysis tab")

if __name__ == "__main__":
    main()
