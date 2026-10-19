"""Collect all retained upstream attempts, including failures, without fitting.
Usage: collect.py SCRATCH PUBLISHED CONFIG OUTPUT_TSV
A screening panel can inform search but only all 61 printed matches qualify.
"""
import csv
import json
import math
import re
from pathlib import Path
import statistics as st
import sys

scratch, published, config, output = map(Path, sys.argv[1:])
pub={r['distorted']:r for r in csv.DictReader(published.open())}
cfg=json.loads(config.read_text())['mappings']
fields=['trial','version','release','peak','black','reflection','eotf','precision','bits','ppd','encoding','matrix','task','age','age_effects','emission','surround','mtf','peak_sensitivity','options','n','rc','median_raw','max_raw','median_jnd','max_jnd','spearman','exact_printed','qualification','tree_sha256','driver_sha256','scores_sha256','command']
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
        option_map=dict(x.split('=',1) for x in opts)
        match=re.search(r'ver\s*=\s*([0-9.]+)', (Path(tree)/'hdrvdp_version.m').read_text())
        release={'2.22':'2.2.2','2.21':'2.2.1','3.06':'3.0.6','3.07':'3.0.7'}[match.group(1).rstrip('.')]
        default_emission=('ccfl-lcd' if enc!='luminance' else 'd65') if version=='2' else ('oled' if enc=='XYZ' else 'led-lcd-srgb')
        row.update(release=release,black=float(peak)/1000 if matrix=='gog' else (1 if matrix=='display' else 0),reflection=0,eotf='gamma2.2' if matrix=='gog' else 'sRGB',precision='f32-transfer' if matrix=='single' else 'f64',task='quality' if version=='3' else 'Q',age='24' if version=='3' else 'not-applicable',age_effects='aesl,aod,slum' if version=='3' else 'not-applicable',emission=option_map.get('rgb_display',default_emission),surround=option_map.get('surround_l','1e-5') if version=='2' else option_map.get('surround','none'),mtf=option_map.get('do_mtf','1') if version=='2' else option_map.get('mtf','hdrvdp'),peak_sensitivity=option_map.get('peak_sensitivity','2.4') if version=='2' else 'not-applicable')
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
