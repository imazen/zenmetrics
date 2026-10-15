"""Compare independent Rust WS-HV with the pinned external author notebook.
No author code is redistributed: supply WS_HV.ipynb from IVP-Lab/WS_HV
revision e31cb50fb8c0e12f7dc3eb185092b23e60ab1bc6 (see README).
"""
import csv,json,pathlib,subprocess,statistics
import cv2,numpy as np,pywt
home=pathlib.Path.home();s=home/'tmp/aic-variants-scratch/round4';nb=json.load(open(s/'WS_HV.ipynb'))
function=next(''.join(c['source']) for c in nb['cells'] if ''.join(c.get('source',[])).startswith('def IQA_WS_HV'))
ns={'np':np,'cv2':cv2,'pywt':pywt};exec(function,ns);oracle=ns['IQA_WS_HV']
pub={r['distorted']:r for r in csv.DictReader(open(home/'tmp/aic-site/site_fullres_subset61.csv'))};cfg=json.load(open(home/'tmp/aic-site/config.json'))['mappings']['WS-HV']
def jnd(x):return cfg['a']*max(0,cfg['b']-x)**cfg['c']
for dataset in ['ladder','aic']:
 pairs=[dict(distorted=p.stem,ref_path=str(p),dist_path=str(p).replace('_ref.png','_dist.png')) for p in sorted((home/'tmp/nn-phase0/ladder').glob('*_ref.png'))] if dataset=='ladder' else list(csv.DictReader(open(home/'tmp/aic-audit/pairs.tsv'),delimiter='\t'))
 rows=[]
 for p in pairs:
  rb=cv2.imread(p['ref_path']);db=cv2.imread(p['dist_path']);r=rb[:,:,::-1].copy();d=db[:,:,::-1].copy();r.tofile(s/'r.rgb');d.tofile(s/'d.rgb')
  own=list(map(float,subprocess.check_output([str(s/'ws-probe'),str(r.shape[1]),str(r.shape[0]),str(s/'r.rgb'),str(s/'d.rgb')]).split()))
  ref=[float(oracle(r,d)),float(oracle(rb,db))]
  row={'key':p['distorted']}
  for i,name in enumerate(['rgb','swapped']):
   row.update({name:own[i],name+'_oracle':ref[i],name+'_oracle_error':abs(own[i]-ref[i]),name+'_oracle_jnd_error':abs(jnd(own[i])-jnd(ref[i]))})
   if dataset=='aic':
    target=float(pub[p['distorted']]['WS-HV']);row.update(published=target);row.update({name+'_published_error':abs(own[i]-target),name+'_published_jnd_error':abs(jnd(own[i])-jnd(target))})
  rows.append(row);print(dataset,json.dumps(row),flush=True)
 with (s/f'ws-{dataset}-verified.json').open('x') as f:json.dump(rows,f,indent=2)
 keys=[k for k in rows[0] if k.endswith('_error')]
 print('SUMMARY',dataset,json.dumps({k:dict(median=statistics.median(r[k] for r in rows),max=max(r[k] for r in rows)) for k in keys}),flush=True)
