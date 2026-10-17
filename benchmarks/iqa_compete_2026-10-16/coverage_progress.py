#!/usr/bin/env python3
"""Count observed coverage, including explicit pending ledgers and partial writes."""
import argparse,datetime,hashlib,itertools,json,pathlib
from native_run import ADAPTERS
from run import LEARNED,PYTHON_METRICS
P=pathlib.Path
SIZES=['cid22','aic-small','aic-medium','aic-large','2k','4k','8k']
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('inventory',type=P);p.add_argument('roots',nargs='+',type=P)
p.add_argument('--freeze-dir',type=P)
p.add_argument('--baseline',action='append',type=P,default=[]);p.add_argument('--save',type=P)
a=p.parse_args();inventory=json.loads(a.inventory.read_text());selected={};sources=[];pending=[]
if a.freeze_dir:a.freeze_dir.mkdir(parents=True)
paths=list(a.baseline)
for root in a.roots:
 found=sorted(root.rglob('i265.jsonl'))
 if not found:pending.append(str(root))
 paths+=found
for path in paths:
 data=path.read_bytes();complete,_,fragment=data.rpartition(b'\n')
 if fragment:pending.append(str(path)+': incomplete final JSONL line')
 lines=complete.splitlines() if b'\n' in data else []
 source=dict(path=str(path),complete_rows=len(lines))
 if a.freeze_dir:
  prefix=(complete+b'\n') if b'\n' in data else b''
  target=a.freeze_dir/f'source-{len(sources):02}.jsonl'
  with target.open('xb') as f:f.write(prefix)
  source.update(snapshot=str(target.resolve()),snapshot_sha256=hashlib.sha256(prefix).hexdigest())
 sources.append(source)
 for line in lines:
  row=json.loads(line);key=tuple(row[k] for k in ['impl','metric_name','device','size_name','threads','mode'])
  selected[key]=row
expected={'rust':set(),'competitors':set()}
for item in inventory:
 devices=['cuda'] if item['gpu'] else ['cpu']
 if item['metric_name'] in LEARNED:devices+=['cuda']
 for device,size,n,mode in itertools.product(devices,SIZES,[1,20],['score','score+map']):expected['rust'].add(('zenmetrics',item['metric_name'],device,size,n,mode))
for impl,names in ({k:[x[0] for x in v] for k,v in ADAPTERS.items()}|PYTHON_METRICS).items():
 for name,size,n,mode in itertools.product(names,SIZES,[1,20],['score','score+map']):expected['competitors'].add((impl,name,'cpu',size,n,mode))
counts={}
for size in SIZES:
 counts[size]={}
 for group,keys in expected.items():
  wanted={k for k in keys if k[3]==size};rows=[selected[k] for k in wanted&selected.keys()]
  counts[size][group]=dict(requested=len(wanted),present=len(rows),timed=sum(r['status'] in ['ok','variant_mismatch'] for r in rows),unsupported=sum(r['status']=='unsupported' for r in rows),refused=sum(r['status'].startswith('skipped:') for r in rows),partial=sum(r['status']=='partial' for r in rows))
scopes={}
for row in selected.values():
 if row.get('coverage_first') and row.get('resource'):
  scopes[row.get('log_path') or row.get('log')]=row['resource']
resources=dict(observed_unique_scopes=len(scopes))
if scopes:resources.update(peak_rss_gib=max(r['peak_rss_gib'] for r in scopes.values()),min_available_mib=min(r['min_available_mib'] for r in scopes.values()),peak_load=max(r['peak_load'] for r in scopes.values()),method='Actual run-heavy receipts; maximum/minimum over unique observed scopes, not an extrapolation')
report=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_scope_resources=resources,per_size=counts,sources=sources,pending_ledgers=pending,contract='Observed rows only; existing status/contamination labels retained; no metric calls or parity qualification')
if a.save:
 with a.save.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,sort_keys=True))
