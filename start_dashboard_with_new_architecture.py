"""
Script to start the dashboard with the new architecture diagram
"""

import subprocess
import sys
import os

def main():
    """Start the dashboard with new architecture"""
    print("Starting Real-Time Dashboard with New Architecture")
    print("=" * 60)
    
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
                print("Continuing without analysis data...")
        except subprocess.TimeoutExpired:
            print("Analysis generation timed out. Continuing without analysis data...")
        except Exception as e:
            print(f"Error running analysis: {str(e)}")
            print("Continuing without analysis data...")
    else:
        print("Analysis data found. Starting dashboard...")
    
    # Start the dashboard
    print("\nStarting Real-Time Dashboard with New Architecture...")
    print("Dashboard will be available at: http://localhost:8501")
    print("You now have 8 tabs with the updated architecture diagram!")
    print("Press Ctrl+C to stop the dashboard")
    print("=" * 60)
    
    try:
        # Run streamlit
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
