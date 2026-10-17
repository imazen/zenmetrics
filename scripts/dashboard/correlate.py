#!/usr/bin/env python3
"""Recompute raw score/human-label correlations and reference-cluster bootstrap CIs.

Inputs stay external; the output contains aggregate statistics and at most 24
real, deterministically selected scatter observations per metric. No remap is fit.
"""
import argparse
import csv
import hashlib
import json
import io
from pathlib import Path
import subprocess
import numpy as np
import pyarrow.parquet as pq
from scipy import stats


def measures(x,y):
    return np.array([stats.spearmanr(x,y).statistic,stats.pearsonr(x,y).statistic,stats.kendalltau(x,y,variant='b').statistic])


def compute(source,output,draws,seed):
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    table=pq.read_table(source).to_pydict()
    dataset=source.stem+' / historical scored snapshot'
    target='human_mos' if 'human_mos' in table else 'human_jnd' if 'human_jnd' in table else 'human_dmos'
    y=np.array(table[target],dtype=float)
    refs=np.array([Path(p).name for p in table['ref_path']])
    result=[];points=[]
    for metric in sorted(c for c in table if c.startswith('score_')):
        x=np.array(table[metric],dtype=float);mask=np.isfinite(x)&np.isfinite(y)
        # Explicitly account for missing identities instead of silently replacing scores.
        xx=x[mask];yy=y[mask];rr=refs[mask]
        if len(xx)<3 or np.ptp(xx)==0 or np.ptp(yy)==0:raise ValueError(f'{metric}: cannot correlate degenerate source')
        signed=measures(xx,yy);groups=[np.flatnonzero(rr==r) for r in np.unique(rr)]
        rng=np.random.default_rng(seed);samples=[]
        for _ in range(draws):
            indices=np.concatenate([groups[i] for i in rng.integers(0,len(groups),size=len(groups))])
            value=measures(xx[indices],yy[indices])
            if not np.all(np.isfinite(value)):raise ValueError('degenerate bootstrap replicate; protocol must be reviewed')
            samples.append(np.abs(value))
        ci=np.percentile(samples,[2.5,97.5],axis=0)
        row=dict(metric=metric,dataset=dataset,target=target,scope='corpus',subset='all finite scored rows',n=len(xx),excluded=int((~mask).sum()),reference_groups=len(groups),status='measured',srocc=abs(signed[0]),plcc=abs(signed[1]),krocc=abs(signed[2]),signed_srocc=signed[0],signed_plcc=signed[1],signed_krocc=signed[2],srocc_lo=ci[0,0],srocc_hi=ci[1,0],plcc_lo=ci[0,1],plcc_hi=ci[1,1],krocc_lo=ci[0,2],krocc_hi=ci[1,2],ci='95% percentile; reference-cluster bootstrap',draws=draws,seed=seed)
        result.append(row)
        indices=np.flatnonzero(mask)
        chosen=sorted(indices,key=lambda i:hashlib.sha256((str(i)+sha).encode()).hexdigest())[:24]
        for i in chosen:
            points.append(dict(metric=metric,dataset=dataset,pair=hashlib.sha256((sha+str(i)).encode()).hexdigest()[:16],mos=y[i],score=x[i],target=target,scope='deterministic 24-observation preview'))
    output.mkdir(parents=True,exist_ok=True)
    outputs=[]
    def encode(rows):
        f=io.StringIO();w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows);return f.getvalue()
    for suffix,rows in [('correlations',result),('points',points)]:
        chunk=[];index=0
        for row in rows:
            if chunk and len(encode(chunk+[row]).encode())>29000:
                file=output/(source.stem+'-'+suffix+f'-{index:02}.tsv')
                file.write_text(encode(chunk));outputs.append(file);index+=1;chunk=[]
            chunk.append(row)
        if chunk:
            file=output/(source.stem+'-'+suffix+f'-{index:02}.tsv')
            file.write_text(encode(chunk));outputs.append(file)
    commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip()
    meta=dict(source_basename=source.name,source_sha256=sha,generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),host='i265',build_commit=commit,seed=seed,draws=draws,target=target,input_rows=len(y),metrics=len(result),method='Raw unremapped Spearman, Pearson, Kendall tau-b; magnitudes in leaderboard, signed statistics retained. 95% percentile intervals from resampling reference-image clusters with replacement. Fixed images; not observer uncertainty. Historical snapshot; original scoring commits are not recoverable from this parquet.',command=f'python scripts/dashboard/correlate.py --source <read-only-source> --output-dir benchmarks/dashboard/quality --draws {draws} --seed {seed}',scatter='24 finite actual rows per metric, selected by SHA-256(row index + source hash); identity hashes only, original paths are not published')
    for file in outputs:file.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(dataset=source.stem,metrics=len(result),input_rows=len(y),draws=draws)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--draws',type=int,default=1000);p.add_argument('--seed',type=int,default=20261017);a=p.parse_args()
    compute(a.source,a.output_dir,a.draws,a.seed)
