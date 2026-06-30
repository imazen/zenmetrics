#!/usr/bin/env python3
"""Is the ssim2-vs-zensim pick disagreement SCALE (a 1-D calibration fixes it → one router serves
both) or RANKING (need a separate ssim2 router)? Fit a global ssim2->zensim map M; compare the
ssim2-oracle at target ts against the zensim-oracle at the CALIBRATED-equivalent tz=M(ts)."""
import pandas as pd, collections, numpy as np
BASE='/mnt/v/output/canonical-picker-2026-06-27'
FAMS=[('jpeg','zenjpeg_lossy'),('webp','zenwebp_lossy'),('jxl','zenjxl_lossy'),('avif','zenavif_lossy')]
NAMES=[f for f,_ in FAMS]
C={}; allz=[]; alls=[]
for f,d in FAMS:
    df=pd.read_parquet(f'{BASE}/{d}/train.parquet',columns=['variant_name','score_zensim','score_ssim2','encoded_bytes']).dropna()
    c=collections.defaultdict(list)
    for v,z,s,b in zip(df.variant_name,df.score_zensim,df.score_ssim2,df.encoded_bytes):
        c[v].append((float(z),float(s),float(b))); allz.append(float(z)); alls.append(float(s))
    C[f]=c
# global monotone ssim2->zensim map: median zensim per ssim2 integer bin, then isotonic-ish cumcummax
az=np.array(allz); as_=np.array(alls)
bins=np.arange(0,101); Mraw=np.array([np.median(az[(as_>=b)&(as_<b+1)]) if ((as_>=b)&(as_<b+1)).any() else np.nan for b in bins])
# fill + enforce monotone
valid=~np.isnan(Mraw); Mraw=np.interp(bins, bins[valid], Mraw[valid]); M=np.maximum.accumulate(Mraw)
def calib(ts): return float(np.interp(ts, bins, M))
def reach(points, idx, t):
    cand=[p[2] for p in points if p[idx]>=t]; return min(cand) if cand else None
allv=set(C['jxl'])
print("ssim2-target ts -> calibrated zensim tz ; agreement of ssim2-oracle(ts) vs zensim-oracle(tz):")
print(f"{'ts(ssim2)':>9} {'tz=M(ts)':>9} {'n':>6} {'raw-agree':>10} {'calib-agree':>12}")
for ts in [60,70,75,80,85,90]:
    tz=calib(ts); n=raw=cal=0
    for v in allv:
        sw={f:reach(C[f].get(v,[]),1,ts) for f in NAMES}; sw={f:b for f,b in sw.items() if b}
        zw_raw={f:reach(C[f].get(v,[]),0,ts) for f in NAMES}; zw_raw={f:b for f,b in zw_raw.items() if b}  # uncalibrated (same numeric)
        zw_cal={f:reach(C[f].get(v,[]),0,tz) for f in NAMES}; zw_cal={f:b for f,b in zw_cal.items() if b}    # calibrated
        if len(sw)>=2 and len(zw_cal)>=2:
            n+=1; swin=min(sw,key=sw.get)
            if len(zw_raw)>=2 and swin==min(zw_raw,key=zw_raw.get): raw+=1
            if swin==min(zw_cal,key=zw_cal.get): cal+=1
    print(f"{ts:>9} {tz:>9.1f} {n:>6} {100*raw/max(n,1):>9.1f}% {100*cal/max(n,1):>11.1f}%")
