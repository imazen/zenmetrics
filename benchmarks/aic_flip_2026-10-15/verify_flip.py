"""Pinned flip-evaluator 1.7; raw maps, means, and published AIC columns."""
import csv, json, pathlib, subprocess, sys
import numpy as np
from PIL import Image
import flip_evaluator as flip
scratch = pathlib.Path.home()/'tmp/aic-variants-scratch'
probe = scratch/'flip-probe'
rows=[]
def compare(ref, dist, key, published=None):
    r=np.asarray(Image.open(ref).convert('RGB')); d=np.asarray(Image.open(dist).convert('RGB'))
    assert r.shape==d.shape
    r.tofile(scratch/'r.rgb'); d.tofile(scratch/'d.rgb')
    oracle, mean, parameters=flip.evaluate(str(ref), str(dist), 'LDR', applyMagma=False)
    raw=subprocess.check_output([str(probe),str(r.shape[1]),str(r.shape[0]),str(scratch/'r.rgb'),str(scratch/'d.rgb')])
    # Accept the initial map-only driver as well as the mean-prefixed driver.
    offset=8 if len(raw)==r.shape[0]*r.shape[1]*4+8 else 0
    ours=np.frombuffer(raw[offset:],dtype='<f4').reshape(oracle.shape)
    delta=np.abs(ours-oracle)
    ownmean=float(np.frombuffer(raw[:8],dtype="<f8")[0]) if offset else float(np.float32(np.cumsum(ours.ravel(),dtype=np.float32)[-1]/np.float32(ours.size)))
    row=dict(key=key, ours=ownmean,oracle=mean,map_max=float(delta.max()),mean_delta=abs(ownmean-mean), published=published, published_delta=None if published is None else abs(ownmean-published), oracle_published_delta=None if published is None else abs(mean-published))
    rows.append(row); print(json.dumps(row),flush=True)
    return row
if sys.argv[1]=='aic':
    published={r['distorted']:r for r in csv.DictReader(open(pathlib.Path.home()/'tmp/aic-site/site_fullres_subset61.csv'))}
    for row in csv.DictReader(open(pathlib.Path.home()/'tmp/aic-audit/pairs.tsv'),delimiter='\t'):
        compare(row['ref_path'], row['dist_path'],row['distorted'],float(published[row['distorted']]['FLIP']))
else:
    ladder=pathlib.Path.home()/'tmp/nn-phase0/ladder'
    for ref in sorted(ladder.glob('*_ref.png')):
        compare(ref,str(ref).replace('_ref.png','_dist.png'),ref.stem)
out=pathlib.Path(sys.argv[2]); out.open('x').write(json.dumps(rows,indent=2)+'\n')
print('SUMMARY',json.dumps({k:dict(median=float(np.median([r[k] for r in rows])),max=max(r[k] for r in rows)) for k in ['map_max','mean_delta']+(['published_delta','oracle_published_delta'] if sys.argv[1]=='aic' else [])}),flush=True)
assert max(r['map_max'] for r in rows)<=1e-6
assert max(r['mean_delta'] for r in rows)<=1e-6
