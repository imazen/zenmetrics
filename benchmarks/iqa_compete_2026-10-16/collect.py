#!/usr/bin/env python3
"""Choose explicitly ordered qualified attempts; never discard source ledgers."""
import argparse,hashlib,json,pathlib,collections
from run import family
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('ledgers',nargs='+',type=P);p.add_argument('--output',type=P,required=True);a=p.parse_args()
assert not a.output.exists(),'Choose a new output or preserve the prior collection first'
selected={};sources=[]
for ledger in a.ledgers:
 digest=hashlib.sha256(ledger.read_bytes()).hexdigest();sources.append(dict(path=str(ledger),sha256=digest))
 for index,line in enumerate(ledger.read_text().splitlines(),1):
  row=json.loads(line);key=tuple(row.get(k) for k in ['host','metric_name','impl','device','threads','mode','size_name'])
  row.update(metric_family=family(row['metric_name']),source_ledger=str(ledger),source_ledger_sha256=digest,source_line=index)
  if row['impl']=='zenmetrics' and row['metric_name'].startswith('vmaf') and row['status']=='ok' and not row.get('model_preload_adapter'):
   row['status']='skipped:diagnostic_model_parsing_inside_timer'
  selected[key]=row
with a.output.open('x') as f:
 for key,row in sorted(selected.items()):f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
receipt=dict(sources=sources,selection='Last supplied ledger wins per cell; last row wins within ledger. Source attempts remain retained.',rows=len(selected),statuses=dict(collections.Counter(r['status'].split(':')[0] for r in selected.values())),output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())
a.output.with_suffix('.meta.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
