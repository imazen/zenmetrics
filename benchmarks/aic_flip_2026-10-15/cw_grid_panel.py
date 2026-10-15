"""61-pair DFT-grid panel, including luma arithmetic sensitivity."""
import csv,json,pathlib,sys
import numpy as np
from PIL import Image
from cw_variants import coarse,bands,similarities
home=pathlib.Path.home();pub={r['distorted']:r for r in csv.DictReader(open(home/'tmp/aic-site/site_fullres_subset61.csv'))};rows=[]
for pair in csv.DictReader(open(home/'tmp/aic-audit/pairs.tsv'),delimiter='\t'):
    r=np.asarray(Image.open(pair['ref_path']).convert('RGB'));d=np.asarray(Image.open(pair['dist_path']).convert('RGB'))
    row={'key':pair['distorted'],'published':float(pub[pair['distorted']]['CW-SSIM'])}
    for ingress in ['explicit64','explicit32','blas64']:
        def luma(a):
            a=a.astype(np.float32 if ingress=='explicit32' else np.float64)
            if ingress=='blas64':return np.round(a@np.array([.299,.587,.114]))
            return np.round(.299*a[:,:,0]+.587*a[:,:,1]+.114*a[:,:,2]).astype(np.float64)
        a=coarse(luma(r),'fftfreq');b=coarse(luma(d),'fftfreq')
        for analytic in [False,True]:
            # Toggle angular interpolation alone; the radial masks stay LUT-based.
            scores=similarities(bands(a,analytic),bands(b,analytic))
            for sigma,value in scores.items():row[f'{ingress}_angular_{"analytic" if analytic else "lut"}_{sigma}']=value
    rows.append(row);print(json.dumps(row),flush=True)
pathlib.Path(sys.argv[1]).open('x').write(json.dumps(rows,indent=2)+'\n')
for k in rows[0]:
    if k in ['key','published']:continue
    delta=[abs(r[k]-r['published']) for r in rows];print('SUMMARY',k,np.median(delta),max(delta),flush=True)
