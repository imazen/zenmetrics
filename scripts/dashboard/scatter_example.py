#!/usr/bin/env python3
"""Freeze a density-only public-data example from a qualified scatter-v2 record.

Usage: scatter_example.py <fulleval.json> <output.json>
No per-image identities, scores or MOS are exported. No statistics are refitted.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def extract(path):
    data=json.loads(path.read_text())
    record=data['scatter_assessment']['cid22']['mos']
    if record['schema']!='scatter-v2' or record['status']!='MEASURED':
        raise ValueError('qualified scatter-v2 required')
    pairs=data['per_pair']['cid22']; targets=pairs['mos']; mapped=pairs['normalized_pred']
    if len(targets)!=record['n'] or len(mapped)!=record['n']:
        raise ValueError('full-population count differs')
    lo,hi=min(targets),max(targets); bins=64
    if hi<=lo:raise ValueError('nondegenerate example required')
    def bin_of(x):return min(bins-1,max(0,int((x-lo)/(hi-lo)*bins)))
    density=Counter((bin_of(x),bin_of(y)) for x,y in zip(targets,mapped))
    panel=data['rank']['cid22']
    return dict(schema=1,description='CID22-A, SSIMULACRA2 stored peer scores, 2,192 admitted pairs from 25 sources; frozen human gauntlet 2026-10-18. Higher score means higher quality. Corpus-specific cached implementation; historical scoring source closure remains incomplete.',
                source_file=path.name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                provenance_url='https://github.com/imazen/zensimx/blob/main/benchmarks/gauntlet_human_2026-10-18.pointer.md',
                scientific_source_commit='b9196f8b',n=record['n'],
                panel={k:panel[k] for k in ['srocc','plcc','krocc','or','pwrc','z_rmse']},
                srocc_ci=panel['srocc_ci'],srocc_ci_method=panel['srocc_ci_method'],
                geo=record['geo'],raw=record['raw'],
                density={'bins':bins,'range':[lo,hi],'cells':[[x,y,n] for (x,y),n in sorted(density.items())]})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    result=extract(args.source)
    text=json.dumps(result,separators=(',',':'),allow_nan=False)+'\n'
    if len(text.encode())>30000:raise ValueError('aggregate exceeds publication size budget')
    args.output.write_text(text)
    print('Qualified example:',result['n'],'pairs;',len(text.encode()),'bytes; input',result['source_sha256'])
