import json
p='realtime_anomaly_project/sql_db/realtime_live_cache.json'
with open(p,'r',encoding='utf8') as f:
    d=json.load(f)
if 'ADANIPORTS.NS' in d.get('data',{}):
    d['data']['ADANIPORTS.NS']['open']=99999.0
    d['data']['ADANIPORTS.NS']['high']=99999.0
    d['data']['ADANIPORTS.NS']['low']=99999.0
    d['data']['ADANIPORTS.NS']['volume']=99999999.0
with open(p,'w',encoding='utf8') as f:
    json.dump(d,f,indent=2)
print('patched ADANIPORTS.NS')
