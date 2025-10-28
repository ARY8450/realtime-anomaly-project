import re

# Read the file
with open('06_RealTime_Dashboard_100_Accuracy.py', 'r', encoding='utf-8') as f:
    content = f.read()

# List of line numbers and their corresponding unique keys
replacements = [
    (1106, 'sentiment_trend_chart'),
    (1153, 'sentiment_score_chart'),
    (1194, 'seasonality_chart'),
    (1219, 'seasonality_bar_chart'),
    (1281, 'fusion_score_chart'),
    (1335, 'fusion_scatter_chart'),
    (1447, 'trend_prediction_chart'),
    (1511, 'portfolio_performance_chart'),
    (1550, 'portfolio_allocation_chart'),
    (1573, 'portfolio_stacked_chart'),
    (1635, 'portfolio_bar_chart'),
    (1715, 'portfolio_technical_chart'),
    (1725, 'portfolio_rsi_chart'),
    (1730, 'portfolio_volume_chart')
]

lines = content.split('\n')

for line_num, key_name in replacements:
    if line_num - 1 < len(lines):
        line = lines[line_num - 1]
        if 'st.plotly_chart' in line and 'key=' not in line:
            # Replace the line to add the key parameter
            lines[line_num - 1] = line.replace(')', f', key="{key_name}")')

# Write back
with open('06_RealTime_Dashboard_100_Accuracy.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print('Fixed all remaining plotly charts')