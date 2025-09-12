#!/usr/bin/env python3
"""
Setup script for Real-Time Anomaly Detection System
Run this script to set up the system for first use.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def verify_installation():
    """Verify that key components can be imported"""
    print("🔍 Verifying installation...")
    
    try:
        import streamlit
        import pandas as pd
        import numpy as np
        import yfinance as yf
        from realtime_anomaly_project.realtime_enhanced_system_100_accuracy import RealTimeEnhancedDataSystemFor100Accuracy
        print("✅ All core components verified!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    dirs_to_create = [
        "data",
        "logs",
        "realtime_anomaly_project/logs"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created!")

def main():
    """Main setup function"""
    print("🚀 Setting up Real-Time Anomaly Detection System")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("realtime_anomaly_project"):
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Setup failed during verification")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the dashboard: streamlit run 06_RealTime_Dashboard_100_Accuracy.py")
    print("2. Test the system: python test_quick_100_accuracy.py")
    print("3. Check the README.md for detailed usage instructions")
    print("\n🌐 Dashboard will be available at: http://localhost:8501")

if __name__ == "__main__":
    main()