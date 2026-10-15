"""Compare CLI CAMBI against pinned libvmaf defaults and the AIC column."""
import csv,json,pathlib,subprocess,statistics,os
import numpy as np
from PIL import Image
h=pathlib.Path.home();s=h/'tmp/aic-variants-scratch/round4';m=json.load(open(h/'tmp/aic-site/config.json'))['mappings']['CAMBI'];pub={r['distorted']:r for r in csv.DictReader(open(h/'tmp/aic-site/site_fullres_subset61.csv'))}
results=pathlib.Path(os.environ.get('AIC_RESULTS_DIR',str(s)));binary=pathlib.Path(os.environ.get('AIC_CAMBI_PROBE',str(s/'aic-more-final-probe')))
def jnd(x):return m['a']*max(0,x-m['b'])**m['c']
for dataset in ['ladder','aic']:
 oracle={r['key']:r for r in json.load(open(s/f'cambi-precise-settings-{dataset}.json'))}
 pairs=[dict(distorted=p.stem,ref_path=str(p),dist_path=str(p).replace('_ref.png','_dist.png')) for p in sorted((h/'tmp/nn-phase0/ladder').glob('*_ref.png'))] if dataset=='ladder' else list(csv.DictReader(open(h/'tmp/aic-audit/pairs.tsv'),delimiter='\t'))
 rows=[]
 for p in pairs:
  r=np.asarray(Image.open(p['ref_path']).convert('RGB'));d=np.asarray(Image.open(p['dist_path']).convert('RGB'));r.tofile(s/'r.rgb');d.tofile(s/'d.rgb')
  result=subprocess.run([str(binary),'cambi',str(r.shape[1]),str(r.shape[0]),str(s/'r.rgb'),str(s/'d.rgb')],capture_output=True,text=True)
  row={'key':p['distorted']};ref=oracle[p['distorted']]
  if result.returncode:row.update(ours_error=result.stderr.strip(),oracle_error=ref.get('default_error'),oracle=ref.get('default'))
  else:
   own=float(result.stdout.split('\t')[-1]);row['ours']=own
   if 'default' in ref:
    val=ref['default'];row.update(oracle=val,oracle_error=abs(own-val),oracle_jnd_error=abs(jnd(own)-jnd(val)))
   else:row['oracle_failure']=ref['default_error']
   if dataset=='aic':
    val=float(pub[p['distorted']]['CAMBI']);row.update(published=val,published_error=abs(own-val),published_jnd_error=abs(jnd(own)-jnd(val)))
  rows.append(row);print(dataset,json.dumps(row),flush=True)
 with (results/f'cambi-precise-{dataset}-verified.json').open('x') as f:json.dump(rows,f,indent=2)
 print('SUMMARY',dataset,json.dumps({k:dict(median=statistics.median(r[k] for r in rows if isinstance(r.get(k),(int,float))),max=max(r[k] for r in rows if isinstance(r.get(k),(int,float)))) for k in ['oracle_error','oracle_jnd_error','published_error','published_jnd_error'] if any(isinstance(r.get(k),(int,float)) for r in rows)}),flush=True)
 print('REFUSALS',dataset,sum('ours_error' in r for r in rows),sum('oracle_failure' in r for r in rows),flush=True)
