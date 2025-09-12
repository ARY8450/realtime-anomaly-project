import re

# Read the file
with open('realtime_anomaly_project/enhanced_data_system.py', 'r') as f:
    content = f.read()

# Replace all talib function calls to use numpy arrays
replacements = [
    (r'talib\.(.*?)\(close,', r'talib.\1(close_np,'),
    (r'talib\.(.*?)\(high, low, close,', r'talib.\1(high_np, low_np, close_np,'),
    (r'talib\.(.*?)\(high, low,', r'talib.\1(high_np, low_np,'),
    (r'talib\.(.*?)\(open_price,', r'talib.\1(open_np,'),
    (r'talib\.(.*?)\(volume,', r'talib.\1(volume_np,'),
]

for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

# Write back
with open('realtime_anomaly_project/enhanced_data_system.py', 'w') as f:
    f.write(content)

print('Fixed talib function calls')