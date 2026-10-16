"""Require unchanged scores except the measured SSIMULACRA1 block-order fix."""
import json,pathlib,statistics,csv
root=pathlib.Path.home()/'tmp/aic-variants-scratch';old=root/'round4';new=root/'round5/clean-oracles';dest=pathlib.Path('benchmarks/aic_review_2026-10-15');checks=[];summary={}
for metric,prefix,fields in [('butter','butter',['ours']),('ssim1','ssim1-opencv-order',['ours']),('ws','ws',['rgb','swapped']),('cambi','cambi-precise',['ours'])]:
 for panel in ['aic','ladder']:
  reference=(root/'round5' if metric=='ssim1' else old)/f'{prefix}-{panel}-verified.json'
  expected={r['key']:r for r in json.load(reference.open())};rows=json.load(open(new/f'{prefix}-{panel}-verified.json'))
  assert set(expected)=={r['key'] for r in rows}
  numeric=0;refusals=0
  for row in rows:
   previous=expected[row['key']]
   for field in fields:
    assert (field in row)==(field in previous)
    if field in row:assert float(row[field]).hex()==float(previous[field]).hex(),(metric,panel,row['key'],field)
   numeric+=fields[0] in row;refusals+='ours_error' in row
  checks.append(dict(metric=metric,panel=panel,numeric=numeric,refusals=refusals,bit_identical=True,baseline=str(reference)))
  keys=list(dict.fromkeys(k for r in rows for k in r))
  summary[metric+'_'+panel]={k:dict(median=statistics.median(r[k] for r in rows if isinstance(r.get(k),(int,float))),max=max(r[k] for r in rows if isinstance(r.get(k),(int,float)))) for k in keys if k.endswith('_error') and any(isinstance(r.get(k),(int,float)) for r in rows)}
  with (dest/f'{metric}_{panel}.tsv').open('x') as f:
   writer=csv.DictWriter(f,keys,delimiter='\t');writer.writeheader();writer.writerows({k:v.replace('\n',' | ') if isinstance(v,str) else v for k,v in row.items()} for row in rows)
with (dest/'panel_invariance.json').open('x') as f:json.dump(checks,f,indent=2)
with (dest/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(checks,indent=2))
