from realtime_anomaly_project.statistical_anomaly.classical_methods import compute_anomalies
import json

A = compute_anomalies(plot_each=False)
keys = list(A.keys())[:6]
out = {k: {'anomaly_flag': (A[k]['anomaly_flag'] if A[k] is not None else None), 'pca_mse': (A[k].get('pca_mse') if isinstance(A[k], dict) else None)} for k in keys}
print(json.dumps(out, default=str, indent=2))
