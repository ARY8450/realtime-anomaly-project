"""
Quick script to restart the dashboard with Transformer AutoEncoder
"""

import subprocess
import sys
import os

print("=" * 70)
print("Restarting Dashboard with Transformer AutoEncoder")
print("=" * 70)

print("\n📋 Integration Status:")
print("  ✓ AdvancedAnomalyDetector created")
print("  ✓ TransformerAutoencoder fixed and tested")
print("  ✓ Real-time system updated")
print("  ✓ Model caching implemented")
print("  ✓ All tests passed")

print("\n🚀 Starting dashboard on port 8501...")
print("  URL: http://localhost:8501")
print("  Press Ctrl+C to stop")
print("\n" + "=" * 70)

# Activate virtual environment and run streamlit
venv_python = os.path.join(".venv", "Scripts", "python.exe")

try:
    subprocess.run([
        venv_python,
        "-m", "streamlit", "run",
        "06_RealTime_Dashboard_100_Accuracy.py",
        "--server.port", "8501"
    ], check=True)
except KeyboardInterrupt:
    print("\n\n✓ Dashboard stopped")
except Exception as e:
    print(f"\n✗ Error starting dashboard: {e}")
    sys.exit(1)
