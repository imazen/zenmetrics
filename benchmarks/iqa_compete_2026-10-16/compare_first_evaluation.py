#!/usr/bin/env python3
"""Compare explicitly resident first evaluations in fresh, alternating processes."""
import argparse,json,os,pathlib,statistics,subprocess,sys

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan',type=pathlib.Path);p.add_argument('output',type=pathlib.Path)
 p.add_argument('--case');p.add_argument('--block',type=int,default=0);a=p.parse_args()
 plan=json.loads(a.plan.read_text())
 sys.path.insert(0,plan['harness_directory'])
 from compare_canonical import sha,compare
 from haarpsi_primary import execute
 plan=json.loads(a.plan.read_text())
 def verify():
  for path,expected in plan['bindings'].items():
   if sha(path)!=expected:raise ValueError('binding changed: '+path)
 if a.case:
  verify();c=next(c for c in plan['cases'] if c['id']==a.case);a.output.mkdir()
  env=dict(os.environ,TMPDIR=str(a.output),CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',LC_ALL='C')
  for key in ['RAYON_NUM_THREADS','RTEN_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[key]=str(c['threads'])
  if 'TORCH_HOME' in plan:env['TORCH_HOME']=plan['TORCH_HOME']
  arms={};errors={}
  for role in (['canonical','native'] if a.block%2==0 else ['native','canonical']):
   command=[v.format_map(dict(c,samples=1,output=str(a.output))) for v in c[role]['command']]
   try:
    role_env=dict(env,**c[role].get('environment',{}))
    rc,_=execute(command,role_env,a.output,role,1100)
    if rc:raise ValueError('exit '+str(rc))
    lines=[line.split() for line in (a.output/(role+'-stdout.log')).read_text().splitlines() if line.startswith('COLD ')]
    if len(lines)!=1:raise ValueError('exactly one first-evaluation observation required')
    arms[role]={'ns':int(lines[0][1]),'score':float(lines[0][2])}
   except Exception as error:errors[role]=str(error)
  result=dict(case=c['id'],arms=arms,errors=errors,phase=plan['phase'],passed=False)
  if len(arms)==2:
   delta,passed=compare(arms['native']['score'],arms['canonical']['score'],c['absolute_bound'],c.get('relative_bound',0))
   result.update(delta=delta,passed=passed,ratio=arms['native']['ns']/arms['canonical']['ns'])
  verify();(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return
 a.output.mkdir();(a.output/'plan.json').write_bytes(a.plan.read_bytes());verify()
 for c in plan['cases']:
  root=a.output/c['id'];root.mkdir();blocks=[]
  for i in range(3):
   directory=root/f'block-{i:02d}'
   command=['flock',str(pathlib.Path.home()/'tmp/lilith-heavy.lock'),'flock',str(pathlib.Path.home()/'tmp/lilith-wsl-heavy.lock'),str(pathlib.Path.home()/'work/claudehints/scripts/run-heavy'),'--mem','12G','--jobs',str(c['threads']),'--',sys.executable,'-B',str(pathlib.Path(__file__).resolve()),str(a.plan.resolve()),str(directory.resolve()),'--case',c['id'],'--block',str(i)]
   with (root/f'block-{i:02d}.log').open('x') as log:
    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
   blocks.append(json.loads((directory/'result.json').read_text()))
  value=dict(metric=c['metric'],case=c['id'],width=c['width'],height=c['height'],threads=c['threads'],phase=plan['phase'],plan_sha256=sha(a.plan),
   passed=all(b['passed'] is True for b in blocks),ratio=statistics.median(b['ratio'] for b in blocks) if all('ratio' in b for b in blocks) else None,
   blocks=blocks,qualification='Three first evaluations after construction, descriptive only; not a warm timing result or full startup measurement')
  (root/'summary.json').write_text(json.dumps(value,indent=2)+'\n');print('FIRST_CASE_DONE '+json.dumps(value),flush=True)

if __name__=='__main__':main()
