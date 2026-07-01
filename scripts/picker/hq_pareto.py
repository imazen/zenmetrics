#!/usr/bin/env python3
"""Does OUR JXL win the ssim2->bpp Pareto at HIGH QUALITY, like Cloudinary/libjxl shows?
Cloudinary: at ssim2~85, jxl ~1.3 bpp vs avif ~1.8 / webp ~1.9 / jpegli ~1.5 (big wins on LARGE imgs).
We measure the SAME axis (target ssim2 -> median bpp) on our canonical lossy data, per codec, per size
tier, with COVERAGE counts (verify the 'jxl swept only to q90' trap + our web-size corpus)."""
import pandas as pd, numpy as np, glob, re, os
BASE='/mnt/v/output/canonical-picker-2026-06-27'
FAMS=[('jpeg','zenjpeg_lossy'),('webp','zenwebp_lossy'),('jxl','zenjxl_lossy'),('avif','zenavif_lossy')]
parts=[]
for f,d in FAMS:
    dfs=[]
    for sp in ['train','validate','test']:
        p=f'{BASE}/{d}/{sp}.parquet'
        if os.path.exists(p):
            c=pd.read_parquet(p,columns=['variant_name','score_zensim','score_ssim2','encoded_bytes'])
            dfs.append(c)
    df=pd.concat(dfs,ignore_index=True).dropna(subset=['score_ssim2','encoded_bytes'])
    m=df.variant_name.str.extract(r'scale(\d+)x(\d+)').astype(float)
    df['px']=m[0]*m[1]
    df=df[df.px>0].copy(); df['bpp']=df.encoded_bytes.values*8.0/df.px.values
    df['codec']=f; parts.append(df[['codec','score_zensim','score_ssim2','bpp','px']])
A=pd.concat(parts,ignore_index=True)
print(f"loaded {len(A):,} lossy rows\n")

print("=== 1) COVERAGE: rows per codec at ssim2 thresholds (is the HQ win-zone even sampled?) ===")
print(f"{'codec':>6} {'total':>9} {'>=70':>8} {'>=80':>8} {'>=85':>8} {'>=88':>8} {'>=90':>8} {'>=92':>8} {'>=95':>7}")
for f,_ in FAMS:
    s=A[A.codec==f].score_ssim2
    print(f"{f:>6} {len(s):>9,} {(s>=70).sum():>8,} {(s>=80).sum():>8,} {(s>=85).sum():>8,} "
          f"{(s>=88).sum():>8,} {(s>=90).sum():>8,} {(s>=92).sum():>8,} {(s>=95).sum():>7,}")

print("\n=== 2) SIZE distribution (Cloudinary's big JXL wins are on 7-39MP; our corpus?) ===")
for lo,hi,lbl in [(0,0.25e6,'tiny <0.25MP'),(0.25e6,0.5e6,'small .25-.5'),(0.5e6,1e6,'med .5-1MP'),(1e6,4e6,'large 1-4MP'),(4e6,1e12,'xl >4MP')]:
    n=((A.px>=lo)&(A.px<hi)).sum(); print(f"  {lbl:>14}: {n:>9,} rows ({100*n/len(A):>4.1f}%)")
print(f"  max image: {A.px.max()/1e6:.2f} MP")

def pareto(sub, bins):
    """median bpp per codec at each ssim2 bin [b,b+w); winner = min bpp."""
    out={}
    for f,_ in FAMS:
        c=sub[sub.codec==f]; row=[]
        for b,w in bins:
            m=(c.score_ssim2>=b)&(c.score_ssim2<b+w); n=m.sum()
            row.append((np.median(c.bpp[m]) if n>=8 else np.nan, n))
        out[f]=row
    return out

BINS=[(70,3),(75,3),(80,3),(83,3),(85,3),(88,2),(90,2),(92,2),(94,2),(96,2)]
def show(title, sub):
    print(f"\n=== {title} (n={len(sub):,}) — median bpp at ssim2 target; [n]; WINNER=lowest bpp ===")
    o=pareto(sub,BINS)
    hdr='ssim2  '+''.join(f'{f:>16}' for f,_ in FAMS)+'   winner'
    print(hdr)
    for i,(b,w) in enumerate(BINS):
        cells=[]; vals={}
        for f,_ in FAMS:
            v,n=o[f][i]; vals[f]=v
            cells.append(f"{('%.3f'%v) if not np.isnan(v) else '  -  '}[{n:>4}]")
        valid={f:v for f,v in vals.items() if not np.isnan(v)}
        win=min(valid,key=valid.get) if valid else '-'
        # jxl gap vs best-other
        others={f:v for f,v in valid.items() if f!='jxl'}
        jgap=''
        if 'jxl' in valid and others:
            bo=min(others.values()); d=100*(valid['jxl']-bo)/bo
            jgap=f"  jxl {'+' if d>=0 else ''}{d:.0f}% vs best-other"
        print(f"{b:>3}-{b+w:<3}"+''.join(f'{c:>16}' for c in cells)+f"   {win:>5}{jgap}")

show("ALL SIZES", A)
show("LARGE 1-4MP (closest to Cloudinary's regime we have)", A[A.px>=1e6])
show("MED 0.5-1MP", A[(A.px>=0.5e6)&(A.px<1e6)])
print("\nDONE")
