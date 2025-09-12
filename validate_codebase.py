"""
Quick validation script for the cleaned codebase
Verifies that all core components are working correctly
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing imports...")
    
    try:
        # Core libraries
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        import plotly.express as px
        import yfinance as yf
        import feedparser
        from textblob import TextBlob # type: ignore
        
        # Project modules  
        from realtime_anomaly_project.realtime_enhanced_system_100_accuracy import RealTimeEnhancedDataSystemFor100Accuracy
        from realtime_anomaly_project.sentiment_module.sentiment_analyzer import SentimentAnalyzer
        
        print("✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_system_initialization():
    """Test system initialization"""
    print("🔧 Testing system initialization...")
    
    try:
        # Import here to avoid scope issues
        from realtime_anomaly_project.realtime_enhanced_system_100_accuracy import RealTimeEnhancedDataSystemFor100Accuracy
        from realtime_anomaly_project.sentiment_module.sentiment_analyzer import SentimentAnalyzer
        
        # Initialize main system
        system = RealTimeEnhancedDataSystemFor100Accuracy()
        print("✅ Main system initialized!")
        
        # Initialize sentiment analyzer
        analyzer = SentimentAnalyzer()
        print("✅ Sentiment analyzer initialized!")
        
        return True
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("⚡ Testing basic functionality...")
    
    try:
        # Import here to avoid scope issues
        from realtime_anomaly_project.sentiment_module.sentiment_analyzer import SentimentAnalyzer
        
        # Test sentiment analysis
        analyzer = SentimentAnalyzer()
        sentiment_result = analyzer.analyze_sentiment("This is a positive test message")
        
        if 'sentiment_score' in sentiment_result and 'sentiment_label' in sentiment_result:
            print("✅ Sentiment analysis working!")
        else:
            print("❌ Sentiment analysis result invalid")
            return False
            
        # Test data fetching
        import yfinance as yf
        ticker = yf.Ticker("RELIANCE.NS")
        info = ticker.info
        
        if 'longName' in info or 'shortName' in info:
            print("✅ Data fetching working!")
        else:
            print("⚠️ Data fetching may have issues (could be network-related)")
            
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        traceback.print_exc()
        return False

def check_file_structure():
    """Check if all essential files are present"""
    print("📁 Checking file structure...")
    
    essential_files = [
        "README.md",
        "requirements.txt", 
        "06_RealTime_Dashboard_100_Accuracy.py",
        "realtime_anomaly_project/__init__.py",
        "realtime_anomaly_project/realtime_enhanced_system_100_accuracy.py",
        "realtime_anomaly_project/sentiment_module/sentiment_analyzer.py"
    ]
    
    missing_files = []
    for file_path in essential_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing essential files: {missing_files}")
        return False
    else:
        print("✅ All essential files present!")
        return True

def main():
    """Run all validation tests"""
    print("🧪 CODEBASE VALIDATION")
    print("=" * 50)
    
    tests = [
        ("File Structure", check_file_structure),
        ("Imports", test_imports),
        ("System Initialization", test_system_initialization), 
        ("Basic Functionality", test_basic_functionality)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Codebase is ready for deployment.")
        print("\n📋 Next steps:")
        print("1. Run: streamlit run 06_RealTime_Dashboard_100_Accuracy.py")
        print("2. Access dashboard at: http://localhost:8501")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)