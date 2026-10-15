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
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
h = Path.home()
site = {r['distorted']: r for r in csv.DictReader((h/'tmp/aic-site/site_fullres_subset61.csv').open())}
maps = json.loads((h/'tmp/aic-site/config.json').read_text())['mappings']
summary = []
for metric in a.metrics:
    dest = a.out/f'{metric}.tsv'
    with (a.out/f'{metric}.log').open('w') as log:
        subprocess.run([str(a.binary), 'batch', '--use-legacy-scheduler', '--pairs', str(h/'tmp/aic-audit/pairs.tsv'), '--output', str(dest), '--jobs', '1', '--metric', metric], stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = list(csv.DictReader(dest.open(), delimiter='\t'))
    assert len(rows) == 61, (metric, len(rows))
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
        mapped = []
        if name in maps:
            cfg=maps[name]
            mapped=[cfg['a']*max(0,cfg['b']-v)**cfg['c']-float(site[r['distorted']]['JND_CVVDP']) for r,v in zip(rows,values)]
        summary.append(dict(metric=metric,column=key,site_mapping=name,n=len(rows),max_published_abs_delta=max(delta) if delta else '',median_jnd_offset=statistics.median(mapped) if mapped else '',max_abs_jnd_offset=max(map(abs,mapped)) if mapped else '',mapping_available=bool(mapped)))
    print(metric, 'complete', flush=True)
with (a.out/'summary.tsv').open('w') as out:
    w=csv.DictWriter(out,fieldnames=summary[0].keys(),delimiter='\t');w.writeheader();w.writerows(summary)
