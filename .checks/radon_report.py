import json
import os
from radon.complexity import cc_visit, cc_rank
from radon.metrics import mi_visit

root = os.path.abspath('realtime_anomaly_project')
report = {'files': {}}

for dirpath, dirs, files in os.walk(root):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(dirpath, f)
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    src = fh.read()
            except Exception as e:
                continue
            try:
                blocks = cc_visit(src)
                mi = mi_visit(src, True)
            except Exception as e:
                continue
            max_cc = 0
            funcs = []
            for b in blocks:
                max_cc = max(max_cc, b.complexity)
                funcs.append({'name': b.name, 'cc': b.complexity, 'lineno': b.lineno})
            report['files'][os.path.relpath(path)] = {
                'max_cc': max_cc,
                'functions': funcs,
                'mi': mi
            }

with open('.checks/radon_report.json', 'w', encoding='utf-8') as outf:
    json.dump(report, outf, indent=2)
print('wrote .checks/radon_report.json')
