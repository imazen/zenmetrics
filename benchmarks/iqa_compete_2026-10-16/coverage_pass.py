#!/usr/bin/env python3
"""Run missing Rust coverage lanes; inner controllers serialize bounded cells."""
import argparse, concurrent.futures, datetime, hashlib, json, pathlib, subprocess, sys
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__)
for key in ['binary','pairs','root','models','model_map']:p.add_argument(key,type=P)
p.add_argument('--sizes',default='aic-small,aic-medium,aic-large,4k,8k')
p.add_argument('--ort-library',type=P,required=True)
p.add_argument('--cuda-libraries',required=True)
a=p.parse_args()
if not a.binary.is_file():raise FileNotFoundError(a.binary)
a.root.mkdir(parents=True,exist_ok=True)
script=P(__file__).with_name('run.py').resolve()
base=[sys.executable,str(script),str(a.binary.resolve()),str(a.pairs.resolve())]
common=['--host','i265','--sizes',a.sizes,'--threads','1,all','--model-root',str(a.models.resolve()),'--models-json',str(a.model_map.resolve()),'--coverage-first']
lanes={
 'cpu':base+[str((a.root/'cpu').resolve())]+common,
 'cuda':base+[str((a.root/'cuda').resolve())]+common+['--gpu-runtime','cuda'],
 'nn-cuda':base+[str((a.root/'nn-cuda').resolve())]+common+['--engine','ort-cuda','--ort-library',str(a.ort_library.resolve()),'--cuda-libraries',a.cuda_libraries],
}
manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 binary=str(a.binary.resolve()),binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),
 pairs=str(a.pairs.resolve()),pairs_sha256=hashlib.sha256(a.pairs.read_bytes()).hexdigest(),
 controller_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),commands=lanes,
 protocol='Coverage first, one pair per size, 5/3/1 samples, bounded cells, at most one contamination rerun; inner flock/run-heavy serialize all actual work')
with (a.root/'launch.json').open('x') as f:f.write(json.dumps(manifest,indent=2)+'\n')
def run(item):
 name,command=item
 with (a.root/f'{name}.log').open('x') as log:
  result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 receipt=dict(lane=name,returncode=result.returncode,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log=str((a.root/f'{name}.log').resolve()))
 with (a.root/f'{name}-exit.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
 return result.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
 codes=list(executor.map(run,lanes.items()))
raise SystemExit(int(any(codes)))
