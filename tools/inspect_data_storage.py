import sys
import os

# Ensure repo root is on sys.path so `realtime_anomaly_project` can be imported
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from realtime_anomaly_project.data_ingestion import yahoo as y
import pandas as pd

ds = getattr(y, 'data_storage', {})
print('data_storage size:', len(ds))
print(list(ds.keys())[:200])
if ds:
    k = list(ds.keys())[0]
    v = ds[k]
    print('\nSample key:', k, 'type=', type(v))
    try:
        if hasattr(v, 'head'):
            print(v.head().to_string())
        else:
            print(repr(v)[:1000])
    except Exception as e:
        print('cannot show head:', e)
