"""Paper/pyrtools comparison with explicit BT.601 ingress operations.

The convention search uses NumPy matrix multiplication for RGB->Y. BLAS can
fuse/reassociate the three products and round exact half ties differently.
This check uses the specified scalar expression, preserving the Rust gate's
1e-9 tolerance. It also measures that ingress distinction separately.
"""
import csv,json,pathlib,subprocess,sys
import numpy as np
from PIL import Image
from cw_variants import coarse,bands,similarities
scratch=pathlib.Path.home()/'tmp/aic-variants-scratch'
pub={r['distorted']:r for r in csv.DictReader(open(pathlib.Path.home()/'tmp/aic-site/site_fullres_subset61.csv'))}
rows=[]
for pair in csv.DictReader(open(pathlib.Path.home()/'tmp/aic-audit/pairs.tsv'),delimiter='\t'):
    r=np.asarray(Image.open(pair['ref_path']).convert('RGB'));d=np.asarray(Image.open(pair['dist_path']).convert('RGB'))
    def ingress(a):
        a=a.astype(np.float64)
        explicit=np.round(.299*a[:,:,0]+.587*a[:,:,1]+.114*a[:,:,2])
        blas=np.round(a@np.array([.299,.587,.114]))
        return explicit,int(np.count_nonzero(explicit!=blas))
    a,nr=ingress(r);b,nd=ingress(d)
    grid=sys.argv[2] if len(sys.argv)>2 else 'pyrtools'
    ref=similarities(bands(coarse(a,grid)),bands(coarse(b,grid)))['full']
    r.tofile(scratch/'r.rgb');d.tofile(scratch/'d.rgb')
    ours=float(subprocess.check_output([str(scratch/'cw-probe'),str(r.shape[1]),str(r.shape[0]),str(scratch/'r.rgb'),str(scratch/'d.rgb')]))
    target=float(pub[pair['distorted']]['CW-SSIM'])
    row=dict(key=pair['distorted'],ours=ours,oracle=ref,published=target,oracle_delta=abs(ours-ref),published_delta=abs(ours-target),blas_luma_different_ref=nr,blas_luma_different_dist=nd)
    rows.append(row);print(json.dumps(row),flush=True)
pathlib.Path(sys.argv[1]).open('x').write(json.dumps(rows,indent=2)+'\n')
print('SUMMARY',json.dumps({k:dict(median=float(np.median([r[k] for r in rows])),max=max(r[k] for r in rows)) for k in ['oracle_delta','published_delta']}),flush=True)
assert max(r['oracle_delta'] for r in rows)<1e-9
