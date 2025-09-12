import sys, traceback, os
# ensure repo root on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
print('sys.path[0]=', sys.path[0])
try:
    import realtime_anomaly_project
    print('IMPORT_OK')
except Exception:
    traceback.print_exc()
    print('IMPORT_FAIL')
