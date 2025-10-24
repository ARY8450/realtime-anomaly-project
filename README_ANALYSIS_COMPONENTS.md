# Real-Time Anomaly Detection Project - Analysis Components

This document provides a comprehensive overview of all the analysis components created for the Real-Time Anomaly Detection Project.

## 🚀 Overview

The project now includes comprehensive analysis components that provide:
- **Backtesting Results**: 30-day backtesting with actual vs predicted price comparisons
- **Advanced Visualizations**: Candlestick charts, line charts, and overlap region analysis
- **Anomaly Detection Graphs**: Real-time anomaly detection and pattern analysis
- **Model Architecture Diagrams**: Complete system architecture and data flow
- **Prediction Price Tables**: Detailed prediction comparison tables and metrics

## 📊 Components

### 1. Backtesting System (`backtesting_system.py`)

**Purpose**: Compare actual vs predicted prices for 30-day periods with detailed analysis.

**Features**:
- 30-day backtesting periods
- Actual vs predicted price comparison
- Candlestick and line chart visualizations
- Performance metrics calculation
- Overlap region analysis
- Export capabilities (CSV, JSON)

**Usage**:
```python
from backtesting_system import ComprehensiveBacktestingSystem

# Initialize system
backtest_system = ComprehensiveBacktestingSystem(
    tickers=['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS'],
    lookback_period=30
)

# Run backtesting
results = backtest_system.run_comprehensive_backtest()

# Export results
export_path = backtest_system.export_results()
```

### 2. Advanced Visualization System (`visualization_system.py`)

**Purpose**: Create comprehensive visualizations including candlestick charts, line charts, and overlap regions.

**Features**:
- Interactive candlestick analysis
- Overlap region analysis
- Anomaly detection visualizations
- Performance dashboards
- Real-time monitoring charts
- Export to HTML

**Usage**:
```python
from visualization_system import AdvancedVisualizationSystem

# Initialize visualizer
viz_system = AdvancedVisualizationSystem()

# Create candlestick analysis
candlestick_path = viz_system.create_candlestick_analysis(
    ticker, df, predictions, anomalies
)

# Create overlap region analysis
overlap_path = viz_system.create_overlap_region_analysis(
    ticker, df, predictions
)
```

### 3. Anomaly Detection Visualizer (`anomaly_detection_visualizer.py`)

**Purpose**: Generate comprehensive anomaly detection graphs and visualizations.

**Features**:
- Real-time anomaly monitoring
- Anomaly pattern analysis
- Clustering and severity analysis
- Performance metrics visualization
- Interactive anomaly exploration
- Export to HTML

**Usage**:
```python
from anomaly_detection_visualizer import AnomalyDetectionVisualizer

# Initialize visualizer
anomaly_viz = AnomalyDetectionVisualizer()

# Create comprehensive anomaly analysis
comprehensive_path = anomaly_viz.create_comprehensive_anomaly_analysis(
    ticker, df, anomaly_scores, anomaly_flags, anomaly_types
)

# Create real-time anomaly monitor
monitor_path = anomaly_viz.create_real_time_anomaly_monitor(ticker, live_data)
```

### 4. Model Architecture Diagram Generator (`model_architecture_diagram.py`)

**Purpose**: Create comprehensive system architecture diagrams showing the complete data flow.

**Features**:
- System architecture diagrams
- Data flow visualizations
- Component interaction diagrams
- Performance metrics visualization
- Network topology diagrams
- Deployment architecture

**Usage**:
```python
from model_architecture_diagram import ModelArchitectureDiagramGenerator

# Initialize generator
arch_generator = ModelArchitectureDiagramGenerator()

# Create system architecture diagram
system_arch_path = arch_generator.create_system_architecture_diagram()

# Create data flow diagram
data_flow_path = arch_generator.create_data_flow_diagram()
```

### 5. Prediction Price Table Generator (`prediction_price_table_generator.py`)

**Purpose**: Create comprehensive prediction price comparison tables with performance metrics.

**Features**:
- Detailed price comparison tables
- Performance metrics calculation
- Interactive visualizations
- Export capabilities (CSV, HTML)
- Multi-ticker comparison
- Comprehensive reporting

**Usage**:
```python
from prediction_price_table_generator import PredictionPriceTableGenerator

# Initialize generator
table_generator = PredictionPriceTableGenerator()

# Create comprehensive prediction table
comparison_path = table_generator.create_comprehensive_prediction_table(
    ticker, df, predictions, model_confidence, anomaly_flags
)

# Create performance metrics table
metrics_path = table_generator.create_performance_metrics_table(ticker, comparison_df)
```

### 6. Comprehensive Analysis System (`comprehensive_analysis_system.py`)

**Purpose**: Integrate all components for complete analysis.

**Features**:
- Complete backtesting analysis
- Advanced visualizations
- Anomaly detection analysis
- Model architecture diagrams
- Prediction price tables
- Comprehensive reporting

**Usage**:
```python
from comprehensive_analysis_system import ComprehensiveAnalysisSystem

# Initialize comprehensive system
analysis_system = ComprehensiveAnalysisSystem()

# Run complete analysis
results = analysis_system.run_complete_analysis()
```

## 🎯 Key Features

### Backtesting Results
- **30-day backtesting periods** with actual vs predicted price comparisons
- **Candlestick graphs** showing price action with predictions overlay
- **Line charts** comparing actual vs predicted prices
- **Overlap region analysis** showing prediction accuracy
- **Performance metrics** including MAE, RMSE, MAPE, direction accuracy
- **Export capabilities** to CSV and JSON formats

### Visualizations
- **Interactive candlestick charts** with predictions and anomalies
- **Overlap region visualizations** showing prediction accuracy
- **Performance dashboards** with comprehensive metrics
- **Real-time monitoring charts** for live analysis
- **Export to HTML** for easy sharing and presentation

### Anomaly Detection Graphs
- **Real-time anomaly monitoring** with live updates
- **Anomaly pattern analysis** showing temporal patterns
- **Clustering analysis** for anomaly grouping
- **Severity analysis** with risk assessment
- **Performance metrics** for anomaly detection accuracy

### Model Architecture
- **System architecture diagrams** showing complete data flow
- **Data flow visualizations** with component interactions
- **ML models architecture** with detailed component breakdown
- **Network topology diagrams** for infrastructure understanding
- **Deployment architecture** for production setup

### Prediction Price Tables
- **Daily prediction comparison** with actual vs predicted prices
- **Performance metrics tables** with comprehensive statistics
- **Interactive dashboards** for visual analysis
- **Multi-ticker comparison** for portfolio analysis
- **Export capabilities** to CSV and HTML formats

## 📁 Output Structure

```
comprehensive_analysis/
├── backtesting_results/
│   ├── backtesting_results_YYYYMMDD_HHMMSS.json
│   └── backtesting_results_YYYYMMDD_HHMMSS_TICKER_comparison.csv
├── visualizations/
│   ├── TICKER_candlestick_analysis.html
│   ├── TICKER_overlap_analysis.html
│   └── performance_dashboard.html
├── anomaly_visualizations/
│   ├── TICKER_comprehensive_anomaly_analysis.html
│   ├── TICKER_realtime_anomaly_monitor.html
│   └── TICKER_anomaly_pattern_analysis.html
├── architecture_diagrams/
│   ├── system_architecture_diagram.png
│   ├── data_flow_diagram.html
│   ├── ml_models_architecture.html
│   └── performance_metrics_diagram.html
├── prediction_tables/
│   ├── TICKER_prediction_comparison_table.csv
│   ├── TICKER_performance_metrics_table.csv
│   └── TICKER_interactive_prediction_dashboard.html
└── comprehensive_analysis_report.html
```

## 🚀 Quick Start

1. **Run Individual Components**:
   ```bash
   python backtesting_system.py
   python visualization_system.py
   python anomaly_detection_visualizer.py
   python model_architecture_diagram.py
   python prediction_price_table_generator.py
   ```

2. **Run Comprehensive Analysis**:
   ```bash
   python comprehensive_analysis_system.py
   ```

3. **View Results**:
   - Open the generated HTML files in your browser
   - Review CSV files for detailed data
   - Check the comprehensive analysis report

## 📊 Sample Data

All components include sample data generation for demonstration purposes. The sample data includes:
- 30 days of OHLCV data
- Generated predictions with realistic noise
- Anomaly scores and flags
- Model confidence scores
- Technical indicators (RSI, SMA, Volatility)

## 🔧 Customization

Each component can be customized for your specific needs:

- **Tickers**: Modify the ticker list for your analysis
- **Time Periods**: Adjust backtesting periods and lookback windows
- **Visualizations**: Customize chart styles and colors
- **Metrics**: Add or modify performance metrics
- **Export Formats**: Change output formats and locations

## 📈 Performance Metrics

The system provides comprehensive performance metrics:

- **Accuracy Metrics**: MAE, RMSE, MAPE, Direction Accuracy
- **Confidence Analysis**: Model confidence scores and distribution
- **Anomaly Detection**: Anomaly rates and severity analysis
- **Technical Indicators**: RSI, SMA, Volatility analysis
- **Performance Grading**: A+ to D grading system

## 🎯 Use Cases

1. **Trading Analysis**: Use backtesting results for trading strategy validation
2. **Risk Management**: Monitor anomaly detection for risk assessment
3. **Model Validation**: Evaluate model performance with comprehensive metrics
4. **System Architecture**: Understand system design and data flow
5. **Performance Monitoring**: Track system performance and accuracy

## 📋 Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- plotly, networkx
- yfinance (for data fetching)
- scikit-learn (for ML models)

## 🤝 Contributing

To add new features or improve existing components:

1. Follow the existing code structure
2. Add comprehensive documentation
3. Include error handling
4. Provide sample data for testing
5. Update this README with new features

## 📞 Support

For questions or issues with the analysis components:

1. Check the individual component documentation
2. Review the sample data and examples
3. Examine the generated output files
4. Refer to the comprehensive analysis report

---

**Note**: This analysis system is designed for the Real-Time Anomaly Detection Project and provides comprehensive tools for financial market analysis, model validation, and system architecture understanding.
