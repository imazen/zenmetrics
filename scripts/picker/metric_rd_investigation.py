#!/usr/bin/env python3
"""WHY does ssim2 favor AVIF? Build per-codec RD curves (bpp vs quality) in BOTH metrics + the
metric-transfer curve (ssim2 achieved at a given zensim, per codec) from the full canonical data.
If AVIF's transfer sits above JXL/WebP, it over-scores ssim2 → wins ssim2-targeting. Tails: p10/p90
bands + per-quality over-score, low-q vs high-q."""
import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BASE='/mnt/v/output/canonical-picker-2026-06-27'
OUT='/mnt/v/output/picker-metric-investigation'; os.makedirs(OUT, exist_ok=True)
FAMS=[('jpeg','zenjpeg_lossy'),('webp','zenwebp_lossy'),('jxl','zenjxl_lossy'),('avif','zenavif_lossy')]
COL={'jpeg':'#888','webp':'#2a9d2a','jxl':'#1f6fd0','avif':'#d62728'}
parts=[]
for f,d in FAMS:
    df=pd.read_parquet(f'{BASE}/{d}/train.parquet',columns=['variant_name','score_zensim','score_ssim2','encoded_bytes']).dropna()
    m=df.variant_name.str.extract(r'scale(\d+)x(\d+)').astype(float)
    px=m[0]*m[1]; df=df[px>0].copy(); df['bpp']=df.encoded_bytes.values*8.0/px[px>0].values
    df['codec']=f; parts.append(df[['codec','score_zensim','score_ssim2','bpp']])
A=pd.concat(parts, ignore_index=True)
print(f"loaded {len(A):,} rows across {len(FAMS)} codecs")
QB=np.arange(40,99,2)
def med_bpp(sub, metric):
    return [np.median(sub.bpp[(sub[metric]>=b)&(sub[metric]<b+2)]) if ((sub[metric]>=b)&(sub[metric]<b+2)).any() else np.nan for b in QB]
# --- Fig 1: RD curves, both metrics ---
fig,(a1,a2)=plt.subplots(1,2,figsize=(15,6),sharey=True)
for f,_ in FAMS:
    sub=A[A.codec==f]
    a1.plot(QB+1, med_bpp(sub,'score_zensim'), color=COL[f], lw=2.2, label=f)
    a2.plot(QB+1, med_bpp(sub,'score_ssim2'),  color=COL[f], lw=2.2, label=f)
for ax,t in [(a1,'zensim'),(a2,'ssim2')]:
    ax.set_yscale('log'); ax.set_xlabel(f'{t} (quality →)'); ax.set_title(f'RD curve vs {t}'); ax.grid(alpha=.3,which='both'); ax.legend()
a1.set_ylabel('median bits/pixel (log; lower = cheaper)')
fig.suptitle('Per-codec RD curves — which codec is cheapest at a given quality, by metric',fontsize=13)
fig.savefig(f'{OUT}/rd_curves.png',dpi=110,bbox_inches='tight'); plt.close(fig)
# --- Fig 2: metric transfer (ssim2 achieved at a given zensim), p10/p90 bands ---
fig,ax=plt.subplots(figsize=(9.5,7.5))
for f,_ in FAMS:
    sub=A[A.codec==f]
    p50,p10,p90=[],[],[]
    for b in QB:
        s=sub.score_ssim2[(sub.score_zensim>=b)&(sub.score_zensim<b+2)]
        if len(s): p50.append(np.median(s)); p10.append(np.percentile(s,10)); p90.append(np.percentile(s,90))
        else: p50.append(np.nan); p10.append(np.nan); p90.append(np.nan)
    ax.plot(QB+1,p50,color=COL[f],lw=2.4,label=f); ax.fill_between(QB+1,p10,p90,color=COL[f],alpha=.10)
ax.set_xlabel('zensim (on the codec\'s own encodes)'); ax.set_ylabel('ssim2 achieved (p10–p50–p90)')
ax.set_title('Metric transfer: ssim2 a codec achieves at a given zensim\nhigher line ⇒ the codec over-scores on ssim2 ⇒ favored when TARGETING ssim2')
ax.grid(alpha=.3); ax.legend()
fig.savefig(f'{OUT}/metric_transfer.png',dpi=110,bbox_inches='tight'); plt.close(fig)
# --- numeric: AVIF over-score vs mean(jxl,webp), per zensim bin (tails) ---
print("\nssim2 achieved at a given zensim (median), per codec — AVIF over-score = avif - mean(jxl,webp):")
print(f"{'zensim':>7} {'jpeg':>6} {'webp':>6} {'jxl':>6} {'avif':>6} {'avif_over':>10}")
for b in [48,55,62,70,78,85,90,95]:
    v={f: np.median(A.score_ssim2[(A.codec==f)&(A.score_zensim>=b)&(A.score_zensim<b+2)]) for f,_ in FAMS}
    over=v['avif']-np.mean([v['jxl'],v['webp']])
    print(f"{b:>7} {v['jpeg']:>6.1f} {v['webp']:>6.1f} {v['jxl']:>6.1f} {v['avif']:>6.1f} {over:>+10.2f}")
print(f"\ngraphs: http://172.23.240.1:3300/picker-metric-investigation/rd_curves.png")
print(f"        http://172.23.240.1:3300/picker-metric-investigation/metric_transfer.png")
