#!/usr/bin/env python3
"""Check retained scoring outputs against supplied published/probe tables.

The probe is the audit's independently written numpy/scipy implementation,
not the pyiqa package. No scoring or fitting occurs in this verifier.
"""
import argparse
import csv
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('panel_root',type=Path)
p.add_argument('site',type=Path)
p.add_argument('probe',type=Path)
a=p.parse_args()
def read(path):
    return {r['distorted']:r for r in csv.DictReader(path.open(),delimiter='\t' if path.suffix=='.tsv' else ',')}
site=read(a.site)
probe=read(a.probe)
rows=[]
for metric,column,published in [('ssim-pyiqa','ssim_pyiqa','SSIM-pyiqa'),('ms-ssim-pyiqa','ms_ssim_pyiqa','MS-SSIM-pyiqa')]:
    ours=read(a.panel_root/f'{metric}.tsv')
    assert len(ours)==len(site)==61
    delta=[(abs(float(r[column])-float(site[n][published])),n) for n,r in ours.items()]
    worst=max(delta)
    assert worst[0]<=1e-5,(metric,worst)
    rows.append((metric,len(ours),*worst,'published',1e-5))
    if metric=='ms-ssim-pyiqa':
        key='S53_JAIB_07.png'
        error=abs(float(ours[key][column])-float(site[key][published]))
        rows.append((metric+' odd S53',1,error,key,'published',1e-5))
for family,probe_prefix in [('ssim','ssim_y_f32_box'),('ssimc','ssimc_f32_box')]:
    for factor in range(2,7):
        metric=f'{family}-pyiqa-f32-k{factor}'
        ours=read(a.panel_root/f'{metric}.tsv')
        assert len(ours)==61
        delta=[(abs(float(r[metric.replace('-','_')])-float(probe[n][f'{probe_prefix}_k{factor}'])),n) for n,r in ours.items()]
        worst=max(delta)
        assert worst[0]<=1e-6,(metric,worst)
        rows.append((metric,len(ours),*worst,'independent audit probe',1e-6))
with (a.panel_root/'gates.tsv').open('w') as out:
    writer=csv.writer(out,delimiter='\t');writer.writerow(['metric','n','max_abs_delta','worst_pair','oracle','tolerance']);writer.writerows(rows)
for row in rows:print(*row,sep='\t')
