#!/usr/bin/env python3
"""Refresh saved GPU comparisons and audit one testbed without executing metrics."""
import argparse,hashlib,json,pathlib,subprocess,sys
P=pathlib.Path
BASE=P(__file__).resolve().parent

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['raw','pairs','output']:p.add_argument(name,type=P)
 p.add_argument('--inventory',type=P,default=BASE/'frozen-inventory.json')
 p.add_argument('--sizes',default='cid22,2k');p.add_argument('--threads',default='1,20')
 p.add_argument('--gpu-runtime',choices=['cuda','wgpu']);p.add_argument('--nn-cuda',action='store_true')
 a=p.parse_args();source_bytes=a.raw.read_bytes();source_sha=hashlib.sha256(source_bytes).hexdigest()
 rows=[json.loads(line) for line in source_bytes.decode().splitlines()]
 a.output.mkdir(parents=True,exist_ok=False)
 for kind in ['cpu','gpu']:
  with (a.output/f'{kind}-source.jsonl').open('x') as f:
   for row in rows:
    selected=(row['device']=='cpu') if kind=='cpu' else (row['device']!='cpu' and row['impl']=='zenmetrics')
    if selected:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
 def run(script,args,log):
  with (a.output/log).open('x') as f:subprocess.run([sys.executable,str(BASE/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)
 run('requalify.py',[a.output/'gpu-source.jsonl',a.inventory,'--reference-jsonl',a.output/'cpu-source.jsonl','--output',a.output/'gpu-compared.jsonl'],'requalify.log')
 run('collect.py',['--output',a.output/'raw.jsonl',a.raw,a.output/'gpu-compared.jsonl'],'collect.log')
 options=['--sizes',a.sizes,'--threads',a.threads]
 if a.gpu_runtime:options+=['--gpu-runtime',a.gpu_runtime]
 if a.nn_cuda:options+=['--nn-cuda']
 run('coverage.py',[a.output/'raw.jsonl',a.inventory,a.pairs,a.output/'coverage.json',*options],'coverage.log')
 if digest(a.raw)!=source_sha:raise ValueError('Input collection changed during refresh; preserve this attempt and retry with immutable input')
 coverage=json.loads((a.output/'coverage.json').read_text());comparisons=json.loads((a.output/'gpu-compared.meta.json').read_text())
 summary={k:v for k,v in coverage.items() if k not in ['missing_cells','pending_qualification']}
 summary.update(source=str(a.raw.resolve()),source_sha256=source_sha,full_receipt=str((a.output/'coverage.json').resolve()),
  missing_count=len(coverage['missing_cells']),pending_count=len(coverage['pending_qualification']),
  refreshed_gpu_rows=comparisons['rows'],reproductions_requested=len(comparisons['reproduction_required']),
  policy='Saved-row comparisons and coverage only; no metric calls or timing fields changed')
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
