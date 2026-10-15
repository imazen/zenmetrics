"""Independent MIT pyrtools 1.0.11 complex pyramid / paper CW-SSIM search.
No pyiqa source or package is used. 4 levels, order 7 (8 orientations).
"""
import csv,json,pathlib,sys
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from pyrtools.pyramids import SteerablePyramidFreq
home=pathlib.Path.home()
pub={r['distorted']:r for r in csv.DictReader(open(home/'tmp/aic-site/site_fullres_subset61.csv'))}
rows=[]
def luma(path):
    a=np.asarray(Image.open(path).convert('RGB'),dtype=np.float64)
    return np.round(a@np.array([.299,.587,.114]))
for n,pair in enumerate(csv.DictReader(open(home/'tmp/aic-audit/pairs.tsv'),delimiter='\t')):
    a=SteerablePyramidFreq(luma(pair['ref_path']),height=4,order=7,is_complex=True)
    b=SteerablePyramidFreq(luma(pair['dist_path']),height=4,order=7,is_complex=True)
    row={'key':pair['distorted'],'published':float(pub[pair['distorted']]['CW-SSIM'])}
    for level in range(4):
        for border in ['valid','same']:
            scores=[]; cropped=[]
            for ori in range(8):
                x=a.pyr_coeffs[(level,ori)]; y=b.pyr_coeffs[(level,ori)]
                c=x*np.conj(y)
                corr=uniform_filter(c.real,7,mode='constant')+1j*uniform_filter(c.imag,7,mode='constant')
                energy=uniform_filter(np.abs(x)**2+np.abs(y)**2,7,mode='constant')
                q=np.divide(2*np.abs(corr),energy,out=np.ones_like(energy),where=energy>0)
                if border=='valid': q=q[3:-3,3:-3]
                hh,ww=q.shape
                gy=np.arange(hh)-(hh-1)/2; gx=np.arange(ww)-(ww-1)/2
                weight=np.exp(-(gy[:,None]**2+gx[None,:]**2)/(2*(x.shape[0]/4)**2))
                scores.append((float((q*weight).sum()/weight.sum()),float(q.mean())))
                cropweight=np.exp(-(gy[:,None]**2+gx[None,:]**2)/(2*(hh/4)**2))
                cropped.append(float((q*cropweight).sum()/cropweight.sum()))
            row[f'l{level}_{border}_gauss']=float(np.mean([x[0] for x in scores]))
            row[f'l{level}_{border}_gauss_crop_sigma']=float(np.mean(cropped))
            row[f'l{level}_{border}_uniform']=float(np.mean([x[1] for x in scores]))
    rows.append(row);print(json.dumps(row),flush=True)
    del a,b
pathlib.Path(sys.argv[1]).open('x').write(json.dumps(rows,indent=2)+'\n')
for key in rows[0]:
    if key in ['key','published']: continue
    delta=[abs(r[key]-r['published']) for r in rows]
    print('SUMMARY',key,np.median(delta),max(delta),flush=True)
