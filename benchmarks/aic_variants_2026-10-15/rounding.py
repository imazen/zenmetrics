#!/usr/bin/env python3
"""Compare both requested luma tie rules on the same published 61 pairs."""
import argparse
import csv
import json
import statistics
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('even',type=Path)
p.add_argument('away',type=Path)
p.add_argument('site',type=Path)
p.add_argument('out',type=Path)
a=p.parse_args()
site={r['distorted']:r for r in csv.DictReader(a.site.open())}
measurements=[]
identical=True
loss={'ties_even':0.0,'half_away':0.0}
for metric,column,published in [('ssim-pyiqa','ssim_pyiqa','SSIM-pyiqa'),('ms-ssim-pyiqa','ms_ssim_pyiqa','MS-SSIM-pyiqa')]:
    outputs=[]
    for rule,root in [('ties_even',a.even),('half_away',a.away)]:
        rows={r['distorted']:r for r in csv.DictReader((root/f'{metric}.tsv').open(),delimiter='\t')}
        assert rows.keys()==site.keys() and len(rows)==61
        values={n:r[column] for n,r in rows.items()}
        delta=[abs(float(values[n])-float(site[n][published])) for n in site]
        assert max(delta)<=1e-5,(metric,rule,max(delta))
        loss[rule]+=sum(delta)
        measurements.append(dict(metric=metric,rule=rule,n=len(rows),median_abs_delta=statistics.median(delta),max_abs_delta=max(delta),mean_abs_delta=statistics.mean(delta)))
        outputs.append(values)
    identical &= outputs[0]==outputs[1]
# The coordinator's explicit fallback is half-away when CLI precision ties.
selected='half_away' if identical or loss['half_away']<=loss['ties_even'] else 'ties_even'
a.out.mkdir(parents=True,exist_ok=True)
with (a.out/'rounding.tsv').open('w') as out:
    writer=csv.DictWriter(out,fieldnames=measurements[0].keys(),delimiter='\t');writer.writeheader();writer.writerows(measurements)
decision=dict(selected=selected,identical_at_cli_precision=identical,total_absolute_error=loss,criterion='CLI-precision equality uses half-away; otherwise compare total absolute error over both 61-pair columns')
(a.out/'rounding.json').write_text(json.dumps(decision,indent=2)+'\n')
for row in measurements:print(row)
print(decision)
