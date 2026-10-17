#!/usr/bin/env python3
"""Require actual ten-round map repairs and exact pre-repair CPU score bits."""
import argparse,hashlib,itertools,json,pathlib,struct
P=pathlib.Path
KEY=['host','os','metric_name','size_name','threads','pair_sha256']
METRICS=['fsim-y','haarpsi-y','iwssim-piq']

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['raw','baseline','output']:p.add_argument(name,type=P)
 a=p.parse_args();baseline={};rows=[]
 for line in a.baseline.read_text().splitlines():
  r=json.loads(line)
  if r['impl']=='zenmetrics' and r['device']=='cpu' and r['mode']=='score' and r['status']=='ok' and r.get('n',0)>=10:
   baseline[tuple(r[k] for k in KEY)]=r
 expected=set(itertools.product(METRICS,['cid22','2k'],[1,20],['score','score+map']))
 seen=set();logs={}
 for line in a.raw.read_text().splitlines():
  r=json.loads(line);cell=tuple(r[k] for k in ['metric_name','size_name','threads','mode'])
  assert cell in expected and cell not in seen,'Unexpected/duplicate repair cell'
  assert r['impl']=='zenmetrics' and r['device']=='cpu' and r['status']=='ok' and r['n']>=10,'Repair not qualified'
  assert r.get('map_score_matches_scalar_bits') is True,'Native map score differs from scalar'
  previous=baseline[tuple(r[k] for k in KEY)]
  bits=struct.pack('>d',r['score']).hex()
  assert bits==struct.pack('>d',previous['score']).hex(),'Pre-repair CPU score bits changed'
  seen.add(cell);rows.append(dict(cell=list(cell),score_bits=bits,n=r['n'],binary_sha256=r['impl_binary_sha']))
  logs[r['log_path']]=dict(sha256=r['log_sha256'],resource=r['resource'])
 assert seen==expected,'Missing repair cells'
 receipt=dict(raw=str(a.raw.resolve()),raw_sha256=hashlib.sha256(a.raw.read_bytes()).hexdigest(),
  baseline=str(a.baseline.resolve()),baseline_sha256=hashlib.sha256(a.baseline.read_bytes()).hexdigest(),
  exact_rows=len(rows),rows=rows,logs=logs,qualification='Actual ten-round scalar/map pairs; scores exactly match the prior CPU binary; no threshold changed')
 with a.output.open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(dict(exact_rows=len(rows),cell_scopes=len(logs))))
if __name__=='__main__':main()
