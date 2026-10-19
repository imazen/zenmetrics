"""Collect all retained upstream attempts, including failures, without fitting.
Usage: collect.py SCRATCH PUBLISHED CONFIG OUTPUT_TSV
A screening panel can inform search but only all 61 printed matches qualify.
"""
import csv
import json
import math
from pathlib import Path
import statistics as st
import sys

scratch, published, config, output = map(Path, sys.argv[1:])
pub={r['distorted']:r for r in csv.DictReader(published.open())}
cfg=json.loads(config.read_text())['mappings']
fields=['trial','version','peak','bits','ppd','encoding','matrix','options','n','rc','median_raw','max_raw','median_jnd','max_jnd','spearman','exact_printed','qualification','tree_sha256','driver_sha256','scores_sha256','command']
def corr(a,b):
    a=[x-st.mean(a) for x in a];b=[x-st.mean(b) for x in b]
    return sum(x*y for x,y in zip(a,b))/math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
def ranks(a):
    return [st.mean(i for i,x in enumerate(sorted(a)) if x==v) for v in a]
with output.open('w') as f:
    writer=csv.DictWriter(f,fields,delimiter='\t');writer.writeheader()
    for p in sorted(scratch.glob('*.meta.json')):
        m=json.loads(p.read_text());cmd=m['argv']
        args=m.get('parameters')
        if args is None:
            i=cmd.index('--quiet');args=cmd[i+2:]
        tree,pairs,scores,version,peak,bits,ppd,enc,matrix,*opts=args
        row=dict(trial=p.stem.removesuffix('.meta'),version=version,peak=peak,bits=bits,ppd=ppd,encoding=enc,matrix=matrix,options=','.join(opts),n=0,rc=m.get('returncode','running'),qualification='failed-or-incomplete',tree_sha256=m['source_tree_sha256'],driver_sha256=m['driver_sha256'],scores_sha256=m.get('output_sha256',''),command=' '.join(cmd))
        data=list(csv.DictReader(open(scores),delimiter='\t')) if Path(scores).exists() else []
        if data:
            assert len({r['distorted'] for r in data}) == len(data), 'duplicate scored names'
            if len(data) == 61:
                assert {r['distorted'] for r in data} == set(pub), 'wrong complete panel'
            a=[float(r['q']) for r in data];b=[float(pub[r['distorted']]['HDR_VDP_'+version]) for r in data]
            assert all(math.isfinite(x) for x in a+b)
            mp=cfg['HDR_VDP_'+version]
            def jnd(x):
                y=mp['sign']*(x-mp['b'])
                return math.copysign(mp['a']*abs(y)**mp['c'],y)
            d=[abs(x-y) for x,y in zip(a,b)];dj=[abs(jnd(x)-jnd(y)) for x,y in zip(a,b)]
            row.update(n=len(data),median_raw=st.median(d),max_raw=max(d),median_jnd=st.median(dj),max_jnd=max(dj),spearman=corr(ranks(a),ranks(b)) if len(a)>1 else '',exact_printed=sum(f'{x:.6f}'==pub[r['distorted']]['HDR_VDP_'+version] for x,r in zip(a,data)))
            if m.get('returncode')==0:
                row['qualification']='screening-only' if len(data)!=61 else 'residual'
                if len(data)==61 and row['exact_printed']==61:row['qualification']='exact-61'
        writer.writerow(row)
