"""
Simple script to run the existing dashboard with analysis components
"""

import subprocess
import sys
import os

def main():
    """Run the existing dashboard"""
    print("Starting Real-Time Dashboard")
    print("=" * 50)
    
    # Check if analysis data exists
    analysis_dir = "comprehensive_analysis"
    if not os.path.exists(analysis_dir):
        print("Analysis data not found. Generating analysis components first...")
        print("Running: python simple_analysis_runner.py")
        
        try:
            result = subprocess.run([sys.executable, "simple_analysis_runner.py"], 
                                 capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("Analysis components generated successfully!")
            else:
                print(f"Error generating analysis components: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("Analysis generation timed out")
            return False
        except Exception as e:
            print(f"Error running analysis: {str(e)}")
            return False
    
    # Run the existing dashboard
    print("\nStarting Dashboard...")
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    print("=" * 50)
    
    try:
        # Run streamlit with the existing dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "06_RealTime_Dashboard_100_Accuracy.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error running dashboard: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nFailed to start dashboard")
        sys.exit(1)
    else:
        print("\nDashboard stopped successfully")
