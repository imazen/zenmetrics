#!/usr/bin/env python3
"""Does targeting ssim2 vs zensim change WHICH FORMAT wins? If the family oracle agrees across the
two metrics, the shipped zensim router serves both (no ssim2 router needed). Per (variant, target),
compare argmin-bytes-to-reach(zensim>=t) vs argmin-bytes-to-reach(ssim2>=t) over the 4 lossy codecs."""
import pandas as pd, collections
BASE='/mnt/v/output/canonical-picker-2026-06-27'
FAMS=[('jpeg','zenjpeg_lossy'),('webp','zenwebp_lossy'),('jxl','zenjxl_lossy'),('avif','zenavif_lossy')]
NAMES=[f for f,_ in FAMS]
C={}
for f,d in FAMS:
    df=pd.read_parquet(f'{BASE}/{d}/train.parquet',columns=['variant_name','score_zensim','score_ssim2','encoded_bytes']).dropna()
    c=collections.defaultdict(list)
    for v,z,s,b in zip(df.variant_name,df.score_zensim,df.score_ssim2,df.encoded_bytes):
        c[v].append((float(z),float(s),float(b)))
    C[f]=c
def reach(points, idx, t):
    cand=[p[2] for p in points if p[idx]>=t]; return min(cand) if cand else None
TARGETS=[60,70,75,80,85,90]
allv=set(C['jxl'])
print(f"{'target':>7} {'n(both)':>8} {'agree%':>7}  winner-by-zensim vs winner-by-ssim2 (when they differ)")
for t in TARGETS:
    n=agree=0; diffs=collections.Counter()
    for v in allv:
        zw={f:reach(C[f].get(v,[]),0,t) for f in NAMES}; zw={f:b for f,b in zw.items() if b}
        sw={f:reach(C[f].get(v,[]),1,t) for f in NAMES}; sw={f:b for f,b in sw.items() if b}
        if len(zw)>=2 and len(sw)>=2:
            n+=1
            zwin=min(zw,key=zw.get); swin=min(sw,key=sw.get)
            if zwin==swin: agree+=1
            else: diffs[f'{zwin}->{swin}']+=1
    top=', '.join(f'{k}:{c}' for k,c in diffs.most_common(4))
    print(f"  zq/ss{t:>2} {n:>8} {100*agree/max(n,1):>6.1f}%  {top}")
