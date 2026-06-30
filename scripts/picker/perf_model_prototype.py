#!/usr/bin/env python3
"""Codec PERFORMANCE model prototype: predict (bytes, achieved quality, encode time) from
(zenanalyze SOURCE features + quality setting + parsed encoder knobs) — no encoding, no post-encode
leak. Demonstrates the data+infra already support a learned RD+time surface per codec; the family
pick / knob pick / budget gate all fall out of argmin over it. Measured on AVIF (775k held-out
cells): bytes R2=0.992 (8.4% medAPE), zensim R2=0.976 (±1.4), encode_ms R2=0.982 (13% medAPE);
MLP(64,32) bytes R2=0.983 — the ZNPR-bakeable form. Source-only inputs (101 qualified feats)."""
import pandas as pd, numpy as np, re
from sklearn.ensemble import HistGradientBoostingRegressor as GBM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
BASE='/mnt/v/output/canonical-picker-2026-06-27'
SIDE='/mnt/v/output/router-features-2026-06-30/zenanalyze_features.parquet'
side=pd.read_parquet(SIDE).drop_duplicates('variant_name').set_index('variant_name')
QCOLS=[c for c in side.columns if '@' in c]
def pk(cell):  # avif knobs from the cell string (s{speed}-{noqm?}-{420?}-{bd10?}-{rgb?})
    m=re.match(r's(\d+)',str(cell)); sp=int(m.group(1)) if m else 4
    c=str(cell); return [sp,int('noqm' in c),int('420' in c),int('bd10' in c),int('rgb' in c)]
def load(split):
    df=pd.read_parquet(f'{BASE}/zenavif_lossy/{split}.parquet',
        columns=['variant_name','q','cell','encoded_bytes','score_zensim','encode_ms']).dropna()
    df=df[df.variant_name.isin(side.index)].reset_index(drop=True)
    F=side.loc[df.variant_name,QCOLS].to_numpy(float)
    return np.hstack([F, df[['q']].to_numpy(float), np.array([pk(c) for c in df.cell])]), df
if __name__=='__main__':
    Xtr,dtr=load('train'); Xte,dte=load('test')
    print(f"avif: train {len(dtr)} / test {len(dte)}; inputs={Xtr.shape[1]} (101 src feats + q + 5 knobs)")
    for tgt,log,lab in [('encoded_bytes',True,'bytes'),('score_zensim',False,'zensim'),('encode_ms',True,'encode_ms')]:
        ytr=dtr[tgt].to_numpy(float); yte=dte[tgt].to_numpy(float)
        if log: ytr,yte=np.log(ytr+1e-9),np.log(yte+1e-9)
        p=GBM(max_iter=400).fit(Xtr,ytr).predict(Xte)
        msg=f"medAPE={np.median(np.abs(np.expm1(p-yte)))*100:4.1f}%" if log else f"medAE={np.median(np.abs(p-yte)):.2f}"
        print(f"  {lab:10} GBM R2={r2_score(yte,p):.3f} {msg}")
