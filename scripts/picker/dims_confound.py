#!/usr/bin/env python3
"""Is the image-dims signal in the projection a real codec axis or a corpus confound — and what
does dropping it cost? Compare: (a) CURRENT 101 qualified feats incl dim-proxies; (b) CONTENT-ONLY
(drop every dim/size-proxy feature); (c) CONTENT + ONE explicit isolated log_pixels term (legible
size axis, not buried). All + target_zq. Held-out RD overhead vs the corrected oracle."""
import pandas as pd, collections, statistics as st, re
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
BASE='/mnt/v/output/canonical-picker-2026-06-27'
SIDE='/mnt/v/output/router-features-2026-06-30/zenanalyze_features.parquet'
FAMS=[('jpeg','zenjpeg_lossy'),('webp','zenwebp_lossy'),('jxl','zenjxl_lossy'),('avif','zenavif_lossy')]
NAMES=[f for f,_ in FAMS]
side=pd.read_parquet(SIDE).drop_duplicates('variant_name').set_index('variant_name')
QCOLS=[c for c in side.columns if '@' in c]; BARE=[c.split('@')[0] for c in QCOLS]
snp=side[QCOLS].to_numpy(float); vix={v:i for i,v in enumerate(side.index)}
DIM_RE=re.compile(r'pixel|_dim|dim_|^dim|aspect|width|height|\barea|extent|resolution|megapix|scale|num_px|n_px',re.I)
dim_idx=[i for i,n in enumerate(BARE) if DIM_RE.search(n)]
print("dim/size-proxy features dropped for CONTENT-ONLY:", [BARE[i] for i in dim_idx])
content_idx=[i for i in range(len(QCOLS)) if i not in dim_idx]
TARGETS=list(range(48,89,4))
def reach(p,t):
    c=[b for z,b in p if z>=t]; return min(c) if c else None
def build(split):
    C={}
    for f,d in FAMS:
        df=pd.read_parquet(f'{BASE}/{d}/{split}.parquet',columns=['variant_name','score_zensim','encoded_bytes']).dropna()
        c=collections.defaultdict(list)
        for v,z,b in zip(df.variant_name,df.score_zensim,df.encoded_bytes): c[v].append((float(z),float(b)))
        C[f]=c
    rows=[]
    for v in C['jxl']:
        if v not in vix: continue
        m=re.search(r'scale(\d+)x(\d+)',v)
        if not m: continue
        lp=np.log(int(m.group(1))*int(m.group(2))); base=snp[vix[v]]
        for t in TARGETS:
            bb={f:reach(C[f].get(v,[]),t) for f in NAMES}; bb={f:b for f,b in bb.items() if b is not None}
            if len(bb)>=2: rows.append((base,lp,t,bb))
    return rows
TR,TE=build('train'),build('test')
def vec(r,kind):
    base,lp,t,_=r
    if kind=='all': x=list(base)+[t]
    elif kind=='content': x=[base[i] for i in content_idx]+[t]
    else: x=[base[i] for i in content_idx]+[lp,t]   # content + explicit size
    return x
def evalk(kind):
    Xtr=np.array([vec(r,kind) for r in TR]); Xte=np.array([vec(r,kind) for r in TE])
    Btr=[r[3] for r in TR]; Bte=[r[3] for r in TE]
    sc=StandardScaler().fit(Xtr); Ztr,Zte=sc.transform(Xtr),sc.transform(Xte)
    pred={}
    for f in NAMES:
        idx=[i for i,bb in enumerate(Btr) if f in bb]
        pred[f]=Ridge(alpha=2.0).fit(Ztr[idx],np.log([Btr[i][f] for i in idx])).predict(Zte)
    o=[]
    for i,bb in enumerate(Bte):
        pk=min(bb,key=lambda f:pred[f][i]); o.append(bb[pk]/min(bb.values())-1)
    o.sort(); return st.mean(o)*100,o[int(len(o)*.9)]*100
for k,lab in [('all','(a) all 101 qual incl dim-proxies + target  [CURRENT]'),
              ('content','(b) CONTENT-ONLY (dim-proxies dropped) + target'),
              ('contentsize','(c) CONTENT + 1 explicit log_pixels term + target')]:
    m,p90=evalk(k); print(f"  {lab:52} mean={m:5.2f}% p90={p90:5.2f}%")
