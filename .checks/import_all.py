import pkgutil
import importlib
import sys
import traceback
import os

# ensure repo root on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

pkg_name = 'realtime_anomaly_project'

print('Scanning package:', pkg_name)
failures = []
success = []

try:
    pkg = importlib.import_module(pkg_name)
except Exception:
    traceback.print_exc()
    print('FAILED to import package', pkg_name)
    sys.exit(2)

for finder, name, ispkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + '.'):
    try:
        module = importlib.import_module(name)
        print('OK  ', name)
        success.append(name)
    except Exception:
        print('ERR ', name)
        traceback.print_exc()
        failures.append((name, traceback.format_exc()))

print('\nSummary:')
print('Imported:', len(success))
print('Failed:  ', len(failures))
if failures:
    print('\nFailures:')
    for n, tb in failures:
        print('---', n)
        print(tb)

if failures:
    sys.exit(3)
else:
    sys.exit(0)
