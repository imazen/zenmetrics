#!/usr/bin/env python3
"""Run clean-room AIC variant validation; no pyiqa source or package used.

Invoke under the lane's flock/run-heavy wrapper. All scoring output is retained.
"""
import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('binary', type=Path)
p.add_argument('out', type=Path)
p.add_argument('metrics', nargs='+')
p.add_argument('--expect-small-ms-errors', action='store_true')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
h = Path.home()
site = {r['distorted']: r for r in csv.DictReader((h/'tmp/aic-site/site_fullres_subset61.csv').open())}
maps = json.loads((h/'tmp/aic-site/config.json').read_text())['mappings']
summary = []
# The rounded RGB base belongs to a separate lane; include its published
# mapping baseline without introducing a rounded SSIMc CLI metric here.
if any(metric.startswith('ssimc-') for metric in a.metrics):
    cfg=maps['SSIMc']
    offsets=[cfg['a']*max(0,cfg['b']-float(row['SSIMc']))**cfg['c']-float(row['JND_CVVDP']) for row in site.values()]
    summary.append(dict(metric='SSIMc published baseline',column='SSIMc',site_mapping='SSIMc',n=len(site),max_published_abs_delta='',median_jnd_offset=statistics.median(offsets),max_abs_jnd_offset=max(map(abs,offsets)),mapping_available=True,rejected_geometry=0))
for metric in a.metrics:
    dest = a.out/f'{metric}.tsv'
    pairs=h/'tmp/aic-audit/pairs.tsv'
    pair_rows=list(csv.DictReader(pairs.open(),delimiter='\t'))
    rejected=[]
    if a.expect_small_ms_errors and metric.startswith('ms-ssim'):
        import struct
        factor=int(metric.rsplit('-k',1)[1]) if '-k' in metric else 1
        supported=[]
        for row in pair_rows:
            with open(row['ref_path'],'rb') as f:
                header=f.read(24)
            assert header[:8]==b'\x89PNG\r\n\x1a\n'
            w,hgt=struct.unpack('>II',header[16:24])
            w//=factor;hgt//=factor
            for _ in range(4):w=(w+1)//2;hgt=(hgt+1)//2
            if min(w,hgt)<11:rejected.append(row)
            else:supported.append(row)
        pairs=a.out/f'{metric}.supported.tsv'
        with pairs.open('w') as out:
            writer=csv.DictWriter(out,fieldnames=pair_rows[0].keys(),delimiter='\t');writer.writeheader();writer.writerows(supported)
        for row in rejected:
            cmd=[str(a.binary),'score','--metric',metric,'--reference',row['ref_path'],'--distorted',row['dist_path']]
            result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            assert result.returncode!=0 and '11x11' in result.stdout, (row,result.stdout)
            with (a.out/f'{metric}.refusals.log').open('a') as log:log.write(row['distorted']+'\n'+result.stdout+'\n')
    expected=len(pair_rows)-len(rejected)
    with (a.out/f'{metric}.log').open('w') as log:
        subprocess.run([str(a.binary), 'batch', '--use-legacy-scheduler', '--pairs', str(pairs), '--output', str(dest), '--jobs', '1', '--metric', metric], stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = list(csv.DictReader(dest.open(), delimiter='\t'))
    assert len(rows) == expected, (metric, len(rows),expected)
    keys = [k for k in rows[0] if k.startswith(('fsim', 'vsi', 'ssim', 'ms_ssim'))]
    for key in keys:
        family = 'FSIMc' if key.startswith('fsimc') else 'FSIM' if key.startswith('fsim') else 'VSI' if key.startswith('vsi') else 'MS-SSIM-pyiqa' if key.startswith('ms_ssim') else 'SSIMc-pyiqa' if key.startswith('ssimc') else 'SSIM-pyiqa'
        suffix = metric.split('-f32', 1)[1] if '-f32' in metric else ''
        name = family + ('-f32'+suffix if '-f32' in metric else metric[len(family.lower()):] if family in ('FSIM', 'FSIMc', 'VSI') else '')
        if family == 'FSIMc':
            name = 'FSIMc' + metric[len('fsim'):]
        base = 'SSIMc' if family == 'SSIMc-pyiqa' else family
        values = [float(r[key]) for r in rows]
        assert all(__import__('math').isfinite(v) for v in values)
        delta = [abs(v-float(site[r['distorted']][base])) for r,v in zip(rows,values)] if name == base else []
        if metric in ('ssim-pyiqa','ms-ssim-pyiqa'):
            assert delta and max(delta) <= 1e-5, (metric,max(delta))
        mapped = []
        if name in maps:
            cfg=maps[name]
            mapped=[cfg['a']*max(0,cfg['b']-v)**cfg['c']-float(site[r['distorted']]['JND_CVVDP']) for r,v in zip(rows,values)]
        summary.append(dict(metric=metric,column=key,site_mapping=name,n=len(rows),max_published_abs_delta=max(delta) if delta else '',median_jnd_offset=statistics.median(mapped) if mapped else '',max_abs_jnd_offset=max(map(abs,mapped)) if mapped else '',mapping_available=bool(mapped),rejected_geometry=len(rejected)))
    print(metric, 'complete', flush=True)
with (a.out/'summary.tsv').open('w') as out:
    w=csv.DictWriter(out,fieldnames=summary[0].keys(),delimiter='\t');w.writeheader();w.writerows(summary)
