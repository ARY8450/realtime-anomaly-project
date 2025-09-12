import json
from pathlib import Path
from radon.complexity import cc_visit
files = list(Path('realtime_anomaly_project').rglob('*.py'))
report = {}
for f in files:
    try:
        src = f.read_text(encoding='utf-8')
        blocks = cc_visit(src)
        report[str(f)] = [{'name':b.name,'complexity':b.complexity,'lineno':b.lineno} for b in blocks]
    except Exception as e:
        report[str(f)] = {'error':str(e)}
Path('.checks').mkdir(exist_ok=True)
Path('.checks/radon_report.json').write_text(json.dumps(report, indent=2))
print('radon report updated, files scanned:', len(files))
