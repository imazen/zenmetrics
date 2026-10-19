"""Residual structure on the full native HDR3 panel; per-pair output stays private.
Usage: stratify.py DATA_ROOT SCORES CONFIG AGGREGATE_TSV PRIVATE_JSON
Requires numpy and Pillow for evaluation-only image inspection.
"""
import csv
import json
from pathlib import Path
import statistics as st
import struct
import sys
import numpy as np
from PIL import Image

root,scores,config,out,private=map(Path,sys.argv[1:])
def norm(name):return name.removeprefix('PTC_')
full={norm(r['distorted']):r for r in csv.DictReader((root/'aic2026/metrics_fullres.csv').open())}
crop={norm(r['distorted']):r for r in csv.DictReader((root/'aic2026/metrics_cropped.csv').open())}
assert full.keys()==crop.keys()
large={name.split('_')[0] for name in crop if any(crop[name][c]!=full[name][c] for c in ['HDR_VDP_2','HDR_VDP_3'])}
rows=list(csv.DictReader(scores.open(),delimiter='\t'));assert len(rows)==61
mapping=json.loads(config.read_text())['mappings']['HDR_VDP_3']
def jnd(x):
    y=mapping['sign']*(x-mapping['b'])
    return np.copysign(mapping['a']*abs(y)**mapping['c'],y)
def has_srgb(p):
    b=p.read_bytes();pos=8
    while pos<len(b):
        n=struct.unpack('>I',b[pos:pos+4])[0]
        if b[pos+4:pos+8]==b'sRGB':return True
        pos+=12+n
    return False
private_rows=[]
for r in rows:
    name=r['distorted'];source=norm(name).split('_')[0]
    path=root/'crop/img'/f'PTC_{source}_Ref_00.png'
    im=Image.open(path);assert im.mode=='RGB' and im.size==(840,944)
    rgb=np.asarray(im,dtype=np.float64)/255
    lin=np.where(rgb<=0.04045,rgb/12.92,((rgb+0.055)/1.055)**2.4)
    y=200*(0.2126*lin[:,:,0]+0.7152*lin[:,:,1]+0.0722*lin[:,:,2])
    ours=float(r['native']);target=float(crop[norm(name)]['HDR_VDP_3'])
    private_rows.append(dict(name=name,codec=norm(name).split('_')[1],category3=source in large,srgb=has_srgb(path),signed_raw=ours-target,abs_raw=abs(ours-target),abs_jnd=abs(jnd(ours)-jnd(target)),mean_nits=float(y.mean()),min_nits=float(y.min()),max_nits=float(y.max()),range_nits=float(y.max()-y.min())))
private.write_text(json.dumps(private_rows,indent=2)+'\n')
with out.open('w') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['group','value','n','median_raw','max_raw','median_jnd','max_jnd'])
    for key in ['category3','srgb','codec']:
        for value in sorted({r[key] for r in private_rows}):
            a=[r for r in private_rows if r[key]==value]
            w.writerow([key,value,len(a),st.median(r['abs_raw'] for r in a),max(r['abs_raw'] for r in a),st.median(r['abs_jnd'] for r in a),max(r['abs_jnd'] for r in a)])
with out.with_name(out.stem+'-correlations.tsv').open('w') as f:
    w=csv.writer(f,delimiter='\t');w.writerow(['feature','residual','n','pearson'])
    for feature in ['mean_nits','min_nits','max_nits','range_nits']:
        for residual in ['signed_raw','abs_raw']:
            w.writerow([feature,residual,61,float(np.corrcoef([r[feature] for r in private_rows],[r[residual] for r in private_rows])[0,1])])
print('61 pairs; reference geometry fixed at 840x944; per-pair values retained only in private JSON')
