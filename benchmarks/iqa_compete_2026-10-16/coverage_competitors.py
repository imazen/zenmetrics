#!/usr/bin/env python3
"""Continue competitor coverage after the already-running libvmaf lane."""
import argparse,datetime,json,os,pathlib,subprocess,sys,time
from native_run import ADAPTERS
from run import PYTHON_METRICS
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__)
for name in ['rust','jxl','pairs','planes','python','flip_python','references','vmaf_ledger','root']:p.add_argument(name,type=P)
p.add_argument('--wait-pid',type=int,required=True);a=p.parse_args()
a.root.mkdir(parents=True,exist_ok=True)
sizes=['cid22','aic-small','aic-medium','aic-large','2k','4k','8k']
expected={(m[0],size,n,mode) for m in ADAPTERS['libvmaf'] for size in sizes for n in [1,20] for mode in ['score','score+map']}
wait_stat=P(f'/proc/{a.wait_pid}/stat')
def identity():
 try:return wait_stat.read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
original=identity()
print('WAIT_FOR_LIBVMAF',a.wait_pid,flush=True)
while original is not None and identity()==original:time.sleep(10)
rows=[json.loads(s) for s in a.vmaf_ledger.read_text().splitlines()]
actual={(r['metric_name'],r['size_name'],r['threads'],r['mode']) for r in rows}
if actual!=expected:raise RuntimeError(f'libvmaf coverage controller stopped with {len(expected-actual)} cells missing; inspect its retained log')
folder=P(__file__).resolve().parent
commands={'libjxl':[sys.executable,str(folder/'native_run.py'),'libjxl',str(a.jxl.resolve()),str(a.rust.resolve()),str(a.pairs.resolve()),str(a.planes.resolve()),str((a.root/'libjxl').resolve()),'--sizes',','.join(sizes),'--coverage-first','--batch-seconds','1100','--batch-reserve-seconds','1000','--reference-jsonl',str(a.references.resolve())]}
for package in PYTHON_METRICS:
 interpreter=a.flip_python if package=='flip-evaluator' else a.python
 commands[package]=[sys.executable,str(folder/'run.py'),str(a.rust.resolve()),str(a.pairs.resolve()),str((a.root/package).resolve()),'--host','i265','--threads','1,all','--sizes',','.join(sizes),'--python-package',package,'--python-executable',str(interpreter.absolute()),'--reference-jsonl',str(a.references.resolve()),'--coverage-first','--batch-seconds','1100','--batch-reserve-seconds','1000']
with (a.root/'launch.json').open('x') as f:f.write(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),commands=commands),indent=2)+'\n')
for name,command in commands.items():
 print('START',name,flush=True)
 with (a.root/f'{name}.log').open('x') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 with (a.root/f'{name}-exit.json').open('x') as f:f.write(json.dumps(dict(returncode=result.returncode,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
 print('FINISH',name,result.returncode,flush=True)
 if result.returncode:raise RuntimeError('Competitor blocked: '+name+'; inspect '+str(a.root/f'{name}.log'))
