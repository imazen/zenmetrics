#!/usr/bin/env python3
"""Codec-PAIRWISE linear discriminants. For each of the 6 lossy pairs (A,B), a linear projection of
zenanalyze features predicts which wins (fewer bytes at target). A pairwise difference cancels the
COMMON size-dependence, so dims enter a discriminant only where the two codecs scale DIFFERENTLY
with size — the real signal, legible per-pair. Combine via round-robin (each codec's score = sum of
P(beats other)) -> order -> RD overhead vs the corrected oracle. Reports per-pair accuracy + top
discriminating features (so size-use is visible where it's real)."""
import pandas as pd, collections, statistics as st, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
BASE='/mnt/v/output/canonical-picker-2026-06-27'
SIDE='/mnt/v/output/router-features-2026-06-30/zenanalyze_features.parquet'
FAMS=[('jpeg','zenjpeg_lossy'),('webp','zenwebp_lossy'),('jxl','zenjxl_lossy'),('avif','zenavif_lossy')]
NAMES=[f for f,_ in FAMS]
side=pd.read_parquet(SIDE).drop_duplicates('variant_name').set_index('variant_name')
QCOLS=[c for c in side.columns if '@' in c]; BARE=[c.split('@')[0] for c in QCOLS]+['target_zq']
snp=side[QCOLS].to_numpy(float); vix={v:i for i,v in enumerate(side.index)}
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
    X,B=[],[]
    for v in C['jxl']:
        if v not in vix: continue
        base=snp[vix[v]]
        for t in TARGETS:
            bb={f:reach(C[f].get(v,[]),t) for f in NAMES}; bb={f:b for f,b in bb.items() if b is not None}
            if len(bb)>=2: X.append(list(base)+[t]); B.append(bb)
    return np.array(X),B
Xtr,Btr=build('train'); Xte,Bte=build('test')
sc=StandardScaler().fit(Xtr); Ztr,Zte=sc.transform(Xtr),sc.transform(Xte)
PAIRS=[(NAMES[i],NAMES[j]) for i in range(4) for j in range(i+1,4)]
clf={}; print("per-pair linear discriminant (test acc | top features driving 'A beats B'):")
for (A,B) in PAIRS:
    idx=[i for i,bb in enumerate(Btr) if A in bb and B in bb]
    y=np.array([1 if Btr[i][A]<Btr[i][B] else 0 for i in idx])
    m=LogisticRegression(max_iter=3000,C=0.5).fit(Ztr[idx],y); clf[(A,B)]=m
    tidx=[i for i,bb in enumerate(Bte) if A in bb and B in bb]
    yt=np.array([1 if Bte[i][A]<Bte[i][B] else 0 for i in tidx])
    acc=(m.predict(Zte[tidx])==yt).mean()
    w=m.coef_[0]; top=np.argsort(-np.abs(w))[:4]
    drv=", ".join(f"{'+' if w[k]>0 else '-'}{BARE[k]}" for k in top)
    print(f"  {A:>4} vs {B:<4} acc={acc:5.1%}  baserate(A wins)={yt.mean():4.0%}  {drv}")
proba={k:clf[k].predict_proba(Zte)[:,1] for k in clf}
def pwin(A,B,i): return proba[(A,B)][i] if (A,B) in proba else 1-proba[(B,A)][i]
o=[]
for i,bb in enumerate(Bte):
    S=list(bb); score={c:sum(pwin(c,d,i) for d in S if d!=c) for c in S}
    pk=max(score,key=score.get); o.append(bb[pk]/min(bb.values())-1)
o.sort(); print(f"\nround-robin pairwise-discriminant order: RD overhead mean={st.mean(o)*100:.2f}% p90={o[int(len(o)*.9)]*100:.2f}%  (per-family regression baseline: 3.85%)")

# --- margin-sum collapse: does round-robin of pre-sigmoid MARGINS (a per-family LINEAR score) hold up? ---
# score[c] = sum_{c'} oriented_margin(c,c'); linear in x => bakeable as one per-family linear layer.
import numpy as _np
marg={k:clf[k].decision_function(Zte) for k in clf}  # margin (>0 => A beats B)
def omarg(A,B,i): return marg[(A,B)][i] if (A,B) in marg else -marg[(B,A)][i]
o2=[]
for i,bb in enumerate(Bte):
    S=list(bb); score={c:sum(omarg(c,d,i) for d in S if d!=c) for c in S}
    pk=max(score,key=score.get); o2.append(bb[pk]/min(bb.values())-1)
o2.sort(); print(f"margin-sum collapse (per-family LINEAR, bakeable as 1-layer): mean={st.mean(o2)*100:.2f}% p90={o2[int(len(o2)*.9)]*100:.2f}%")
# confirm the collapsed per-family weights exist (sum of oriented pair coefs, on standardized space)
W={c:_np.zeros(Ztr.shape[1]) for c in NAMES}
for (A,B),m in clf.items():
    W[A]+=m.coef_[0]; W[B]-=m.coef_[0]
print("collapsed per-family linear weights built for:", list(W.keys()), "dim", len(W['jxl']))
