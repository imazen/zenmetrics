"""C++ original Butteraugli vs production Rust, on all ladder and AIC pairs."""
import csv,json,pathlib,subprocess,statistics,os
import numpy as np
from PIL import Image
h=pathlib.Path.home();s=h/'tmp/aic-variants-scratch/round4';cfg=json.load(open(h/'tmp/aic-site/config.json'))['mappings']['Butteraugli-original']
def jnd(x):return cfg['a']*max(0,x-cfg['b'])**cfg['c']
results=pathlib.Path(os.environ.get('AIC_RESULTS_DIR',str(s)));binary=pathlib.Path(os.environ.get('AIC_BUTTER_PROBE',str(s/'butter-original-probe')))
pub={r['distorted']:r for r in csv.DictReader(open(h/'tmp/aic-site/site_fullres_subset61.csv'))}
for dataset in ['ladder','aic']:
 pairs=[dict(distorted=p.stem,ref_path=str(p),dist_path=str(p).replace('_ref.png','_dist.png')) for p in sorted((h/'tmp/nn-phase0/ladder').glob('*_ref.png'))] if dataset=='ladder' else list(csv.DictReader(open(h/'tmp/aic-audit/pairs.tsv'),delimiter='\t'))
 rows=[]
 for p in pairs:
  r=np.asarray(Image.open(p['ref_path']).convert('RGB'));d=np.asarray(Image.open(p['dist_path']).convert('RGB'));r.tofile(s/'r.rgb');d.tofile(s/'d.rgb')
  own=float(subprocess.check_output([str(binary),str(r.shape[1]),str(r.shape[0]),str(s/'r.rgb'),str(s/'d.rgb')]))
  oracle=float(subprocess.check_output([str(s/'oracles/butteraugli-71b18b636b9c7d1ae0c1d3730b85b3c127eb4511/butteraugli/butteraugli'),p['ref_path'],p['dist_path']]))
  row=dict(key=p['distorted'],ours=own,oracle=oracle,oracle_error=abs(own-oracle),oracle_jnd_error=abs(jnd(own)-jnd(oracle)))
  if dataset=='aic':
   target=float(pub[p['distorted']]['Butteraugli-original']);row.update(published=target,published_error=abs(own-target),published_jnd_error=abs(jnd(own)-jnd(target)))
  rows.append(row);print(dataset,json.dumps(row),flush=True)
 with (results/f'butter-{dataset}-verified.json').open('x') as f:json.dump(rows,f,indent=2)
 keys=['oracle_error','oracle_jnd_error']+(['published_error','published_jnd_error'] if dataset=='aic' else [])
 print('SUMMARY',dataset,json.dumps({k:dict(median=statistics.median(r[k] for r in rows),max=max(r[k] for r in rows)) for k in keys}),flush=True)
