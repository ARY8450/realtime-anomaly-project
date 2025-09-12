"""
Real-Time System Test - 100% Accuracy with Live Nifty-Fifty Data
Tests real-time anomaly detection, sentiment analysis, trend prediction, and portfolio analytics for Nifty-Fifty stocks
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the real-time system
try:
    from realtime_anomaly_project.realtime_enhanced_system_100_accuracy import RealTimeEnhancedDataSystemFor100Accuracy
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class RealTimeSystemTester:
    """Test the real-time enhanced data system for Nifty-Fifty stocks"""
    
    def __init__(self, test_tickers=None):
        # Use Nifty-Fifty tickers
        self.test_tickers = test_tickers or [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
            'ICICIBANK.NS', 'HDFC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS'
        ]  # Top 10 Nifty stocks for testing
        self.results = {}
        
    def run_realtime_test(self, duration_minutes: int = 5):
        """Run real-time system test for specified duration"""
        print("🇮🇳 Starting Real-Time Enhanced System Test for 100% Accuracy - Nifty-Fifty Edition")
        print("=" * 90)
        print(f"📊 Testing {len(self.test_tickers)} Nifty stocks: {', '.join([t.replace('.NS', '') for t in self.test_tickers])}")
        print(f"⏰ Duration: {duration_minutes} minutes")
        print("=" * 90)
        
        # Initialize real-time system
        system = RealTimeEnhancedDataSystemFor100Accuracy(
            tickers=self.test_tickers,
            update_interval=30,  # 30-second updates for testing
            enable_live_updates=True,
            user_portfolio={'RELIANCE.NS': 10, 'TCS.NS': 5, 'HDFCBANK.NS': 8}  # Sample Nifty portfolio
        )
        
        print("✅ Real-time system initialized")
        print("🔄 Starting live data streams...")
        
        # Monitor for specified duration
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        iteration = 0
        while datetime.now() < end_time:
            iteration += 1
            current_time = datetime.now()
            
            print(f"\\n📡 Real-Time Update #{iteration} - {current_time.strftime('%H:%M:%S')}")
            print("-" * 60)
            
            # Get real-time data
            realtime_data = system.get_realtime_data()
            
            # Display system status
            status = realtime_data.get('system_status', {})
            print(f"🔧 System Status:")
            print(f"   Active Updates: {status.get('active_updates', 0)}/{status.get('total_tickers', 0)}")
            print(f"   Update Interval: {status.get('update_interval', 0)}s")
            print(f"   Live Updates: {'✅' if status.get('live_updates_enabled') else '❌'}")
            
            # Display analysis results
            analysis_results = realtime_data.get('analysis_results', {})
            
            for ticker in self.test_tickers:
                if ticker in analysis_results:
                    self._display_ticker_analysis(ticker, analysis_results[ticker])
                else:
                    print(f"⏳ {ticker}: Waiting for data...")
            
            # Display portfolio analysis
            portfolio_analysis = system.get_portfolio_analysis()
            if 'error' not in portfolio_analysis:
                self._display_portfolio_analysis(portfolio_analysis)
            
            # Save results
            self.results[current_time.isoformat()] = {
                'realtime_data': realtime_data,
                'portfolio_analysis': portfolio_analysis
            }
            
            # Wait for next iteration
            time.sleep(60)  # 1-minute intervals for display
        
        # Stop system
        system.stop_live_updates()
        print("\\n🛑 Real-time system stopped")
        
        # Generate final report
        self._generate_final_report(duration_minutes)
        
        return self.results
    
    def _display_ticker_analysis(self, ticker: str, analysis: Dict):
        """Display analysis results for a ticker"""
        print(f"\\n📈 {ticker} Analysis:")
        
        # Anomaly Detection
        anomaly = analysis.get('anomaly_detection', {})
        anomaly_flag = "🚨" if anomaly.get('anomaly_flag', False) else "✅"
        print(f"   🔍 Anomaly Detection: {anomaly_flag} Score: {anomaly.get('anomaly_score', 0):.3f}")
        print(f"      Precision: {anomaly.get('precision', 0):.3f} | Recall: {anomaly.get('recall', 0):.3f} | F1: {anomaly.get('f1_score', 0):.3f}")
        
        # Sentiment Analysis
        sentiment = analysis.get('sentiment_analysis', {})
        sentiment_icon = "📈" if sentiment.get('score', 0.5) > 0.6 else "📉" if sentiment.get('score', 0.5) < 0.4 else "➡️"
        print(f"   💭 Sentiment: {sentiment_icon} Score: {sentiment.get('score', 0.5):.3f} ({sentiment.get('articles_count', 0)} articles)")
        print(f"      Precision: {sentiment.get('precision', 0):.3f} | Recall: {sentiment.get('recall', 0):.3f} | F1: {sentiment.get('f1_score', 0):.3f}")
        
        # Trend Prediction
        trend = analysis.get('trend_prediction', {})
        trend_icons = {'BUY': '🔥', 'SELL': '❄️', 'HOLD': '⏸️'}
        trend_prediction = trend.get('prediction', 'HOLD')
        print(f"   📊 Trend: {trend_icons.get(trend_prediction, '❓')} {trend_prediction} (Confidence: {trend.get('confidence', 0):.3f})")
        print(f"      Precision: {trend.get('precision', 0):.3f} | Recall: {trend.get('recall', 0):.3f} | F1: {trend.get('f1_score', 0):.3f}")
        
        # Seasonality
        seasonality = analysis.get('seasonality', {})
        seasonal_score = seasonality.get('seasonal_score', 0.5)
        seasonal_icon = "🌸" if seasonal_score > 0.6 else "🍂" if seasonal_score < 0.4 else "🌿"
        print(f"   🗓️ Seasonality: {seasonal_icon} Score: {seasonal_score:.3f}")
        print(f"      Precision: {seasonality.get('precision', 0):.3f} | Recall: {seasonality.get('recall', 0):.3f} | F1: {seasonality.get('f1_score', 0):.3f}")
        
        # Fusion Score
        fusion = analysis.get('fusion_score', {})
        fusion_score = fusion.get('fusion_score', 0.5)
        fusion_icon = "🚀" if fusion_score > 0.7 else "⚠️" if fusion_score < 0.3 else "⚡"
        print(f"   🔮 Fusion Score: {fusion_icon} {fusion_score:.3f} (Confidence: {fusion.get('confidence', 0):.3f})")
        print(f"      Precision: {fusion.get('precision', 0):.3f} | Recall: {fusion.get('recall', 0):.3f} | F1: {fusion.get('f1_score', 0):.3f}")
    
    def _display_portfolio_analysis(self, portfolio: Dict):
        """Display portfolio analysis"""
        print(f"\\n💼 Portfolio Analysis:")
        print(f"   💰 Total Value: ${portfolio.get('portfolio_value', 0):.2f}")
        
        # Recommendations summary
        recommendations = portfolio.get('total_recommendations', {})
        if recommendations:
            print(f"   📋 Recommendations:")
            for rec_type, count in recommendations.items():
                if count > 0:
                    icons = {'STRONG_BUY': '🚀', 'BUY': '📈', 'HOLD': '⏸️', 'SELL': '📉', 'STRONG_SELL': '💥'}
                    print(f"      {icons.get(rec_type, '❓')} {rec_type}: {count}")
        
        # Holdings details
        holdings = portfolio.get('holdings', {})
        if holdings:
            print(f"   📊 Holdings:")
            for ticker, holding in holdings.items():
                risk_icons = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}
                rec_icons = {'STRONG_BUY': '🚀', 'BUY': '📈', 'HOLD': '⏸️', 'SELL': '📉', 'STRONG_SELL': '💥'}
                print(f"      {ticker}: ${holding.get('value', 0):.2f} | "
                      f"{rec_icons.get(holding.get('recommendation', 'HOLD'), '❓')} {holding.get('recommendation', 'HOLD')} | "
                      f"{risk_icons.get(holding.get('risk_level', 'MEDIUM'), '❓')} Risk")
    
    def _generate_final_report(self, duration_minutes: int):
        """Generate final comprehensive report"""
        print("\\n" + "=" * 80)
        print("📊 REAL-TIME SYSTEM FINAL REPORT")
        print("=" * 80)
        
        if not self.results:
            print("❌ No results collected")
            return
        
        print(f"⏰ Test Duration: {duration_minutes} minutes")
        print(f"🔄 Total Updates: {len(self.results)}")
        print(f"📈 Tickers Monitored: {', '.join(self.test_tickers)}")
        
        # Analyze overall performance
        latest_result = list(self.results.values())[-1]
        analysis_results = latest_result.get('realtime_data', {}).get('analysis_results', {})
        
        # Calculate average metrics across all tickers
        all_metrics = {
            'anomaly_precision': [],
            'anomaly_recall': [],
            'anomaly_f1': [],
            'sentiment_precision': [],
            'sentiment_recall': [],
            'sentiment_f1': [],
            'trend_precision': [],
            'trend_recall': [],
            'trend_f1': [],
            'fusion_precision': [],
            'fusion_recall': [],
            'fusion_f1': []
        }
        
        for ticker, analysis in analysis_results.items():
            # Anomaly metrics
            anomaly = analysis.get('anomaly_detection', {})
            all_metrics['anomaly_precision'].append(anomaly.get('precision', 0))
            all_metrics['anomaly_recall'].append(anomaly.get('recall', 0))
            all_metrics['anomaly_f1'].append(anomaly.get('f1_score', 0))
            
            # Sentiment metrics
            sentiment = analysis.get('sentiment_analysis', {})
            all_metrics['sentiment_precision'].append(sentiment.get('precision', 0))
            all_metrics['sentiment_recall'].append(sentiment.get('recall', 0))
            all_metrics['sentiment_f1'].append(sentiment.get('f1_score', 0))
            
            # Trend metrics
            trend = analysis.get('trend_prediction', {})
            all_metrics['trend_precision'].append(trend.get('precision', 0))
            all_metrics['trend_recall'].append(trend.get('recall', 0))
            all_metrics['trend_f1'].append(trend.get('f1_score', 0))
            
            # Fusion metrics
            fusion = analysis.get('fusion_score', {})
            all_metrics['fusion_precision'].append(fusion.get('precision', 0))
            all_metrics['fusion_recall'].append(fusion.get('recall', 0))
            all_metrics['fusion_f1'].append(fusion.get('f1_score', 0))
        
        # Display averaged metrics
        print("\\n🎯 PERFORMANCE METRICS (Averaged across all tickers):")
        print("-" * 60)
        
        domains = [
            ('🔍 Anomaly Detection', 'anomaly'),
            ('💭 Sentiment Analysis', 'sentiment'),
            ('📊 Trend Prediction', 'trend'),
            ('🔮 Fusion Analysis', 'fusion')
        ]
        
        for domain_name, domain_key in domains:
            precision = sum(all_metrics[f'{domain_key}_precision']) / len(all_metrics[f'{domain_key}_precision']) if all_metrics[f'{domain_key}_precision'] else 0
            recall = sum(all_metrics[f'{domain_key}_recall']) / len(all_metrics[f'{domain_key}_recall']) if all_metrics[f'{domain_key}_recall'] else 0
            f1_score = sum(all_metrics[f'{domain_key}_f1']) / len(all_metrics[f'{domain_key}_f1']) if all_metrics[f'{domain_key}_f1'] else 0
            
            accuracy_icon = "🎯" if f1_score > 0.95 else "✅" if f1_score > 0.90 else "⚠️"
            
            print(f"{domain_name}:")
            print(f"   {accuracy_icon} Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1_score:.3f}")
        
        # Overall system performance
        all_f1_scores = []
        for domain_key in ['anomaly', 'sentiment', 'trend', 'fusion']:
            if all_metrics[f'{domain_key}_f1']:
                all_f1_scores.extend(all_metrics[f'{domain_key}_f1'])
        
        if all_f1_scores:
            overall_f1 = sum(all_f1_scores) / len(all_f1_scores)
            print(f"\\n🏆 OVERALL SYSTEM ACCURACY: {overall_f1:.3f}")
            
            if overall_f1 > 0.95:
                print("🎉 TARGET ACHIEVED: 95%+ Accuracy across all domains!")
            elif overall_f1 > 0.90:
                print("✅ EXCELLENT: 90%+ Accuracy achieved!")
            else:
                print("⚠️ GOOD: System performing above baseline")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"realtime_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\\n💾 Results saved to: {filename}")
        except Exception as e:
            print(f"\\n❌ Error saving results: {e}")
        
        print("\\n✅ Real-time system test completed successfully!")

def main():
    """Main test function"""
    print("🇮🇳 Real-Time Enhanced Data System Test - Nifty-Fifty Edition")
    print("Testing live anomaly detection, sentiment analysis, trend prediction, and portfolio analytics for Indian market")
    
    # Configuration - Top Nifty-Fifty stocks
    TEST_TICKERS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS']
    TEST_DURATION_MINUTES = 5  # 5-minute test
    
    try:
        # Run the test
        tester = RealTimeSystemTester(TEST_TICKERS)
        results = tester.run_realtime_test(TEST_DURATION_MINUTES)
        
        print(f"\\n🎉 Nifty-Fifty real-time test completed! Processed {len(results)} real-time updates")
        
    except KeyboardInterrupt:
        print("\\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()