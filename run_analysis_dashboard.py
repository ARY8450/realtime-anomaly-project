"""
Simple script to run the analysis dashboard
"""

import subprocess
import sys
import os

def main():
    """Run the analysis dashboard"""
    print("Starting Analysis Dashboard")
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
                print("Continuing without analysis data...")
        except subprocess.TimeoutExpired:
            print("Analysis generation timed out. Continuing without analysis data...")
        except Exception as e:
            print(f"Error running analysis: {str(e)}")
            print("Continuing without analysis data...")
    
    # Start the analysis dashboard
    print("\nStarting Analysis Dashboard...")
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    print("=" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "fixed_dashboard.py",
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
