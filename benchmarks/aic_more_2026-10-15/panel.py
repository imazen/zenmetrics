"""61-pair or ladder panel, retaining published- and fitted-JND-unit errors."""
import argparse,csv,json,pathlib,subprocess,statistics
import numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('mode',choices=['cambi','butter-existing','butter-oracle','ssim1-oracle']);p.add_argument('output',type=pathlib.Path);p.add_argument('--ladder',action='store_true');a=p.parse_args()
home=pathlib.Path.home();scratch=home/'tmp/aic-variants-scratch/round4';config=json.load(open(home/'tmp/aic-site/config.json'))['mappings'];pub={r['distorted']:r for r in csv.DictReader(open(home/'tmp/aic-site/site_fullres_subset61.csv'))}
name={'cambi':'CAMBI','butter-existing':'Butteraugli-original','butter-oracle':'Butteraugli-original','ssim1-oracle':'SSIMULACRA1'}[a.mode]
m=config[name]
def jnd(x):return m['a']*max(0,m['sign']*(x-m['b']))**m['c']
if a.ladder:
 pairs=[dict(distorted=r.stem,ref_path=str(r),dist_path=str(r).replace('_ref.png','_dist.png')) for r in sorted((home/'tmp/nn-phase0/ladder').glob('*_ref.png'))]
else:pairs=list(csv.DictReader(open(home/'tmp/aic-audit/pairs.tsv'),delimiter='\t'))
rows=[]
for pair in pairs:
 r=np.asarray(Image.open(pair['ref_path']).convert('RGB'));d=np.asarray(Image.open(pair['dist_path']).convert('RGB'));h,w=r.shape[:2]
 if a.mode in ['cambi','butter-existing']:
  r.tofile(scratch/'r.rgb');d.tofile(scratch/'d.rgb');kind='cambi' if a.mode=='cambi' else 'butteraugli'
  cmd=[str(scratch/'aic-more-cambi-probe'),kind,str(w),str(h),str(scratch/'r.rgb'),str(scratch/'d.rgb')]
 else:
  binary=scratch/'ssimulacra1-oracle' if a.mode=='ssim1-oracle' else scratch/'oracles/butteraugli-71b18b636b9c7d1ae0c1d3730b85b3c127eb4511/butteraugli/butteraugli'
  cmd=[str(binary),pair['ref_path'],pair['dist_path']]
 result=subprocess.run(cmd,text=True,capture_output=True)
 if result.returncode:
  row={'key':pair['distorted'],'error':result.stderr.strip(),'returncode':result.returncode};rows.append(row);print(json.dumps(row),flush=True);continue
 score=float(result.stdout.splitlines()[0].split('\t')[-1]);row=dict(key=pair['distorted'],score=score,jnd=jnd(score))
 if not a.ladder:
  target=float(pub[pair['distorted']][name]);row.update(published=target,abs_error=abs(score-target),jnd_abs_error=abs(jnd(score)-jnd(target)))
 rows.append(row);print(json.dumps(row),flush=True)
with a.output.open('x') as f:json.dump(rows,f,indent=2);f.write('\n')
if not a.ladder:
 print('SUMMARY',json.dumps({k:{'median':statistics.median(r[k] for r in rows if k in r),'max':max(r[k] for r in rows if k in r)} for k in ['abs_error','jnd_abs_error']}),flush=True)
print('errors',sum('error' in r for r in rows),flush=True)
