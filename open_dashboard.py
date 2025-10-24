"""
Simple script to open the dashboard directly in the browser
No server required - opens the HTML file directly
"""

import os
import webbrowser
import sys
from pathlib import Path

def open_dashboard():
    """Open the dashboard directly in the browser"""
    try:
        # Get the current directory
        current_dir = Path.cwd()
        dashboard_path = current_dir / "frontend_dashboard" / "index.html"
        
        # Check if the dashboard exists
        if not dashboard_path.exists():
            print(f"Dashboard not found at: {dashboard_path}")
            print("Please run 'python simple_frontend_dashboard.py' first to create the dashboard")
            return False
        
        # Convert to file URL
        file_url = dashboard_path.as_uri()
        
        print(f"Opening dashboard: {file_url}")
        print(f"Dashboard location: {dashboard_path}")
        
        # Open in default browser
        webbrowser.open(file_url)
        
        print("Dashboard opened in your default browser!")
        print("\nIf the dashboard doesn't load properly:")
        print("1. Make sure all files are in the frontend_dashboard/ folder")
        print("2. Try opening frontend_dashboard/index.html directly")
        print("3. Check that your browser allows local file access")
        
        return True
        
    except Exception as e:
        print(f"Error opening dashboard: {str(e)}")
        return False

def main():
    """Main function"""
    print("Opening Real-Time Anomaly Detection Dashboard")
    print("=" * 60)
    
    success = open_dashboard()
    
    if success:
        print("\nDashboard should now be open in your browser!")
    else:
        print("\nFailed to open dashboard. Please check the error above.")

if __name__ == "__main__":
    main()
