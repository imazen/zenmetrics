#!/usr/bin/env python3
"""Plot the ssim2->bpp RD curve (Cloudinary's axis) on OUR data. Lower bpp at a target ssim2 = win."""
import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BASE='/mnt/v/output/canonical-picker-2026-06-27'; OUT='/mnt/v/output/picker-metric-investigation'
FAMS=[('jpeg','zenjpeg_lossy'),('webp','zenwebp_lossy'),('jxl','zenjxl_lossy'),('avif','zenavif_lossy')]
COL={'jpeg':'#888','webp':'#2a9d2a','jxl':'#1f6fd0','avif':'#d62728'}
parts=[]
for f,d in FAMS:
    dfs=[pd.read_parquet(f'{BASE}/{d}/{sp}.parquet',columns=['variant_name','score_ssim2','encoded_bytes'])
         for sp in ['train','validate','test'] if os.path.exists(f'{BASE}/{d}/{sp}.parquet')]
    df=pd.concat(dfs,ignore_index=True).dropna(subset=['score_ssim2','encoded_bytes'])
    m=df.variant_name.str.extract(r'scale(\d+)x(\d+)').astype(float); df['px']=m[0]*m[1]
    df=df[df.px>0].copy(); df['bpp']=df.encoded_bytes.values*8.0/df.px.values; df['codec']=f
    parts.append(df[['codec','score_ssim2','bpp','px']])
A=pd.concat(parts,ignore_index=True)
QB=np.arange(50,97,2)
def med(sub,f):
    c=sub[sub.codec==f]; return [np.median(c.bpp[(c.score_ssim2>=b)&(c.score_ssim2<b+2)])
        if ((c.score_ssim2>=b)&(c.score_ssim2<b+2)).sum()>=8 else np.nan for b in QB]
fig,axes=plt.subplots(1,2,figsize=(16,7),sharey=True)
for ax,(sub,ttl) in zip(axes,[(A,'ALL sizes (n=%d)'%len(A)),(A[(A.px>=0.5e6)&(A.px<1e6)],'MED 0.5-1MP')]):
    for f,_ in FAMS: ax.plot(QB+1,med(sub,f),color=COL[f],lw=2.6,marker='o',ms=3,label=f)
    ax.axvspan(83,92,color='gold',alpha=.12,label='HQ (Cloudinary win-zone)')
    ax.set_yscale('log'); ax.set_xlabel('SSIMULACRA2 target (quality →)'); ax.set_title(ttl)
    ax.grid(alpha=.3,which='both'); ax.legend(loc='upper left')
axes[0].set_ylabel('median bits/pixel (log) — LOWER = fewer bytes at same quality = WIN')
fig.suptitle('OUR data, ssim2→bpp Pareto: JXL wins decisively at HQ (matches Cloudinary/libjxl)\n'
             'coverage thins for jxl above ssim2~94 (swept only to q90) → avif "wins" there by artifact',fontsize=12)
fig.savefig(f'{OUT}/hq_ssim2_pareto.png',dpi=110,bbox_inches='tight')
print(f"saved {OUT}/hq_ssim2_pareto.png")
print("http://172.23.240.1:3300/picker-metric-investigation/hq_ssim2_pareto.png")
