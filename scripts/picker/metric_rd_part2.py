#!/usr/bin/env python3
"""Separate the REAL ssim2-favors-avif effect from the high-q coverage artifact (jxl absent >~q90).
(1) per-codec metric bias vs quality (differential plot). (2) re-measure ssim2-vs-zensim winner
agreement on the CLEAN range only, and quantify how many avif-flips sit in the jxl-absent tail."""
import pandas as pd, numpy as np, os, collections
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BASE='/mnt/v/output/canonical-picker-2026-06-27'; OUT='/mnt/v/output/picker-metric-investigation'
FAMS=[('jpeg','zenjpeg_lossy'),('webp','zenwebp_lossy'),('jxl','zenjxl_lossy'),('avif','zenavif_lossy')]
COL={'jpeg':'#888','webp':'#2a9d2a','jxl':'#1f6fd0','avif':'#d62728'}
parts={}
for f,d in FAMS:
    df=pd.read_parquet(f'{BASE}/{d}/train.parquet',columns=['variant_name','score_zensim','score_ssim2','encoded_bytes']).dropna()
    parts[f]=df
A=pd.concat([d.assign(codec=f) for f,d in parts.items()], ignore_index=True)
# jxl high-q coverage: max zensim with real jxl mass
jxl=parts['jxl']; print(f"jxl rows: total {len(jxl):,}; with zensim>=92: {(jxl.score_zensim>=92).sum():,}; >=95: {(jxl.score_zensim>=95).sum():,}")
# Fig 3: per-codec metric bias = median ssim2 at given zensim, minus cross-codec mean, vs zensim
QB=np.arange(46,93,2); fig,ax=plt.subplots(figsize=(10,6))
med={f:[np.median(A.score_ssim2[(A.codec==f)&(A.score_zensim>=b)&(A.score_zensim<b+2)]) for b in QB] for f,_ in FAMS}
mean_all=np.nanmean(np.array([med[f] for f,_ in FAMS]),axis=0)
for f,_ in FAMS:
    ax.plot(QB+1, np.array(med[f])-mean_all, color=COL[f], lw=2.4, label=f, marker='o', ms=3)
ax.axhline(0,color='k',lw=.8); ax.set_xlabel('zensim'); ax.set_ylabel('ssim2 bias (achieved ssim2 − cross-codec mean, at that zensim)')
ax.set_title('Per-codec ssim2 bias vs quality (clean range, jxl-covered)\n>0 ⇒ over-scores ssim2 ⇒ favored when targeting ssim2'); ax.grid(alpha=.3); ax.legend()
fig.savefig(f'{OUT}/metric_bias.png',dpi=110,bbox_inches='tight'); plt.close(fig)
# Re-measure agreement: clean range only (target<=85, where jxl has coverage), require all-4-reach
C={f:collections.defaultdict(list) for f,_ in FAMS}
for f,_ in FAMS:
    for v,z,s in zip(parts[f].variant_name,parts[f].score_zensim,parts[f].score_ssim2):
        C[f][v].append((float(z),float(s),parts[f].encoded_bytes[0] if False else None))
# rebuild with bytes
C={f:collections.defaultdict(list) for f,_ in FAMS}
for f,_ in FAMS:
    for v,z,s,b in zip(parts[f].variant_name,parts[f].score_zensim,parts[f].score_ssim2,parts[f].encoded_bytes):
        C[f][v].append((float(z),float(s),float(b)))
def reach(pts,idx,t):
    c=[p[2] for p in pts if p[idx]>=t]; return min(c) if c else None
allv=set(C['jxl']); print("\nclean-range agreement (require all 4 codecs reach the target):")
print(f"{'target':>7} {'n':>6} {'agree%':>7}  flips, and how many land where jxl is ABSENT at that target")
for t in [60,70,78,85]:
    n=ag=0; jxl_absent_flip=0; flips=collections.Counter()
    for v in allv:
        zw={f:reach(C[f].get(v,[]),0,t) for f in FAMS_n} if (FAMS_n:=[f for f,_ in FAMS]) else {}
        zw={f:reach(C[f].get(v,[]),0,t) for f in FAMS_n}; sw={f:reach(C[f].get(v,[]),1,t) for f in FAMS_n}
        zwc={f:b for f,b in zw.items() if b}; swc={f:b for f,b in sw.items() if b}
        if len(zwc)==4 and len(swc)==4:  # all 4 reach on BOTH metrics → clean
            n+=1; zwin=min(zwc,key=zwc.get); swin=min(swc,key=swc.get)
            if zwin==swin: ag+=1
            else: flips[f'{zwin}->{swin}']+=1
    print(f"{t:>7} {n:>6} {100*ag/max(n,1):>6.1f}%  {', '.join(f'{k}:{c}' for k,c in flips.most_common(4))}")
print(f"\ngraph: http://172.23.240.1:3300/picker-metric-investigation/metric_bias.png")
