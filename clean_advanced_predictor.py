#!/usr/bin/env python3
import re

# Read the file
with open('realtime_anomaly_project/advanced_trend_predictor.py', 'r') as f:
    lines = f.readlines()

# Find the problematic section and clean it up
cleaned_lines = []
in_lstm_method = False
skip_lines = False

for i, line in enumerate(lines):
    if 'def create_lstm_model(' in line:
        in_lstm_method = True
        cleaned_lines.append(line)
    elif in_lstm_method and line.strip() == 'return None':
        cleaned_lines.append(line)
        # Skip until the next method definition
        skip_lines = True
    elif skip_lines and line.strip().startswith('def '):
        skip_lines = False
        in_lstm_method = False
        cleaned_lines.append(line)
    elif not skip_lines:
        cleaned_lines.append(line)

# Write the cleaned version
with open('realtime_anomaly_project/advanced_trend_predictor.py', 'w') as f:
    f.writelines(cleaned_lines)

print("Cleaned up the advanced_trend_predictor.py file")