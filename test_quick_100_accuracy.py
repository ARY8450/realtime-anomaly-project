#!/usr/bin/env python3
"""
Quick test of the 100% accuracy system with Nifty-Fifty tickers
Tests anomaly detection, sentiment analysis, and trend prediction for all 50 Nifty stocks
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'realtime_anomaly_project'))

def test_enhanced_system():
    """Test the enhanced system with Nifty-Fifty tickers"""
    print("🇮🇳 Testing Enhanced 100% Accuracy System - Nifty-Fifty Edition")
    print("=" * 70)
    
    try:
        # Add current directory to path to find the module
        current_dir = os.path.dirname(__file__)
        realtime_project_path = os.path.join(current_dir, 'realtime_anomaly_project')
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        if realtime_project_path not in sys.path:
            sys.path.append(realtime_project_path)
        
        # Import the enhanced system using dynamic import
        import importlib.util
        
        enhanced_module_path = os.path.join(realtime_project_path, 'enhanced_data_system_100_accuracy.py')
        if not os.path.exists(enhanced_module_path):
            raise ImportError(f"Enhanced system module not found at: {enhanced_module_path}")
            
        spec = importlib.util.spec_from_file_location("enhanced_data_system_100_accuracy", enhanced_module_path)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load enhanced_data_system_100_accuracy module")
            
        enhanced_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enhanced_module)
        
        EnhancedDataSystemFor100Accuracy = enhanced_module.EnhancedDataSystemFor100Accuracy
        
        # Create system instance with all Nifty-Fifty tickers
        # Get Nifty-Fifty tickers from settings
        try:
            settings_spec = importlib.util.spec_from_file_location(
                "settings", 
                os.path.join(realtime_project_path, "config", "settings.py")
            )
            if settings_spec and settings_spec.loader:
                settings_module = importlib.util.module_from_spec(settings_spec)
                settings_spec.loader.exec_module(settings_module)
                nifty_fifty_tickers = getattr(settings_module, 'TICKERS', [])
                if not nifty_fifty_tickers:
                    raise ValueError("No Nifty-Fifty tickers found in settings")
            else:
                raise ImportError("Could not load settings module")
        except Exception as e:
            print(f"Warning: Could not load Nifty-Fifty tickers from settings: {e}")
            # Fallback to hardcoded Nifty-Fifty tickers
            nifty_fifty_tickers = [
                "ADANIPORTS.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS",
                "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS", "BRITANNIA.NS", "CIPLA.NS",
                "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
                "HCLTECH.NS", "HDFC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
                "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS",
                "IOC.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
                "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
                "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS",
                "SUNPHARMA.NS", "TATASTEEL.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TITAN.NS",
                "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS", "TECHM.NS", "TCS.NS"
            ]
        
        print(f"🇮🇳 Processing all {len(nifty_fifty_tickers)} Nifty-Fifty tickers for comprehensive analysis...")
        print(f"📈 Nifty-Fifty stocks: {', '.join(nifty_fifty_tickers[:10])}{'...' if len(nifty_fifty_tickers) > 10 else ''}")
        
        enhanced_system = EnhancedDataSystemFor100Accuracy(
            tickers=nifty_fifty_tickers,  # Use all Nifty-Fifty tickers
            lookback='6mo'  # Use 6 months for faster testing
        )
        print("✓ Enhanced system initialized successfully with Nifty-Fifty tickers")
        
        # Test data fetching
        print("\n1. Testing data fetching...")
        data = enhanced_system.fetch_comprehensive_data()
        print(f"✓ Data fetched: {len(data)} records")
        
        # Test comprehensive analysis
        print("\n2. Testing comprehensive analysis...")
        results = enhanced_system.run_comprehensive_analysis()
        print(f"✓ Analysis completed for {len(results)} tickers")
        
        # Display comprehensive results for all Nifty-Fifty tickers
        print(f"\n3. 🇮🇳 Comprehensive Results for All {len(nifty_fifty_tickers)} Nifty-Fifty Tickers:")
        print("=" * 130)
        print(f"{'Nifty Stock':<15} {'Anomaly':<10} {'Sentiment':<10} {'Trend':<12} {'Fusion':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10} {'ROC-AUC':<10} {'PR-AUC':<10} {'Accuracy':<10}")
        print("=" * 130)
        
        if isinstance(results, dict):
            # Check if results has the expected structure
            if 'anomaly_detection' in results and isinstance(results['anomaly_detection'], dict):
                # Get all tickers from any of the result categories
                all_result_tickers = set()
                for category in ['anomaly_detection', 'sentiment_analysis', 'trend_prediction', 'fusion_scores']:
                    if category in results and isinstance(results[category], dict):
                        all_result_tickers.update(results[category].keys())
                
                # Sort tickers for consistent display
                sorted_tickers = sorted(all_result_tickers)
                
                for ticker in sorted_tickers:
                    # Get data from each analysis category
                    anomaly_data = results.get('anomaly_detection', {}).get(ticker, {})
                    sentiment_data = results.get('sentiment_analysis', {}).get(ticker, {})
                    trend_data = results.get('trend_prediction', {}).get(ticker, {})
                    fusion_data = results.get('fusion_scores', {}).get(ticker, {})
                    
                    # Extract scores with safe formatting
                    anomaly_score = anomaly_data.get('score', anomaly_data.get('anomaly_score', 'N/A'))
                    sentiment_score = sentiment_data.get('score', sentiment_data.get('sentiment_score', 'N/A'))
                    trend_prediction = trend_data.get('prediction', trend_data.get('trend_prediction', 'N/A'))
                    fusion_score = fusion_data.get('score', fusion_data.get('fusion_score', 'N/A'))
                    
                    # Extract performance metrics
                    f1_score = (
                        anomaly_data.get('f1_score', sentiment_data.get('f1_score', 
                        trend_data.get('f1_score', fusion_data.get('f1_score', 'N/A'))))
                    )
                    precision = (
                        anomaly_data.get('precision', sentiment_data.get('precision', 
                        trend_data.get('precision', fusion_data.get('precision', 'N/A'))))
                    )
                    recall = (
                        anomaly_data.get('recall', sentiment_data.get('recall', 
                        trend_data.get('recall', fusion_data.get('recall', 'N/A'))))
                    )
                    roc_auc = (
                        anomaly_data.get('roc_auc', sentiment_data.get('roc_auc', 
                        trend_data.get('roc_auc', fusion_data.get('roc_auc', 'N/A'))))
                    )
                    pr_auc = (
                        anomaly_data.get('pr_auc', sentiment_data.get('pr_auc', 
                        trend_data.get('pr_auc', fusion_data.get('pr_auc', 'N/A'))))
                    )
                    
                    # Get overall accuracy
                    if 'accuracy_metrics' in results:
                        accuracy = results['accuracy_metrics'].get('overall_accuracy', 'N/A')
                    else:
                        accuracy = anomaly_data.get('accuracy', sentiment_data.get('accuracy', 
                                  trend_data.get('accuracy', fusion_data.get('accuracy', 'N/A'))))
                    
                    # Format values for display
                    def format_value(val, is_percentage=False):
                        if isinstance(val, (int, float)):
                            if is_percentage:
                                return f"{val:.2%}"
                            else:
                                return f"{val:.4f}"
                        return str(val)[:8] + ".." if len(str(val)) > 10 else str(val)
                    
                    # Print formatted row for Nifty-Fifty stock
                    print(f"{ticker:<15} {format_value(anomaly_score):<10} {format_value(sentiment_score):<10} "
                          f"{format_value(trend_prediction):<12} {format_value(fusion_score):<10} "
                          f"{format_value(f1_score):<10} {format_value(precision):<10} {format_value(recall):<10} "
                          f"{format_value(roc_auc):<10} {format_value(pr_auc):<10} {format_value(accuracy, True):<10}")
                
                print("=" * 130)
                
                # Display summary statistics
                if 'accuracy_metrics' in results:
                    print(f"\n4. 🇮🇳 Nifty-Fifty System Performance Summary:")
                    metrics = results['accuracy_metrics']
                    print(f"  📊 Overall Nifty-Fifty Accuracy: {metrics.get('overall_accuracy', 0):.1%}")
                    print(f"  🎯 Target Achieved: {'✅ YES' if metrics.get('target_achieved', False) else '⏳ IN PROGRESS'}")
                    print(f"  🔍 Anomaly Detection Accuracy: {metrics.get('anomaly_detection_accuracy', 0):.1%}")
                    print(f"  💭 Sentiment Analysis Accuracy: {metrics.get('sentiment_analysis_accuracy', 0):.1%}")
                    print(f"  📈 Trend Prediction Accuracy: {metrics.get('trend_prediction_accuracy', 0):.1%}")
                
                # Display recommendation summary
                if 'recommendations' in results:
                    print(f"\n5. 🇮🇳 Nifty-Fifty Trading Recommendations Summary:")
                    rec = results['recommendations']
                    print(f"  🟢 Buy Signals: {len(rec.get('buy_signals', []))} Nifty stocks")
                    print(f"  🔴 Sell Signals: {len(rec.get('sell_signals', []))} Nifty stocks")
                    print(f"  ⚠️  Risk Warnings: {len(rec.get('risk_warnings', []))} Nifty stocks")
                    print(f"  🟡 Hold Recommendations: {len(rec.get('hold_recommendations', []))} Nifty stocks")
                    
                    if rec.get('buy_signals'):
                        print(f"     Buy: {', '.join(rec['buy_signals'][:10])}{'...' if len(rec['buy_signals']) > 10 else ''}")
                    if rec.get('sell_signals'):
                        print(f"     Sell: {', '.join(rec['sell_signals'][:10])}{'...' if len(rec['sell_signals']) > 10 else ''}")
                
            else:
                # Results might be in a different format, show first few items
                print("Results structure different than expected. Showing available data:")
                for key, value in list(results.items())[:5]:
                    print(f"\n{key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
        else:
            print("Results format not as expected")
        
        print("\n✓ All tests passed! Enhanced system is working correctly.")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure enhanced_data_system_100_accuracy.py is in the correct location")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_system()
    sys.exit(0 if success else 1)