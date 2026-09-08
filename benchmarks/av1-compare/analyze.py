#!/usr/bin/env python3
"""Offline RD analysis of saved rows; never launches or schedules encodes.

Outputs raw per-cell summaries and log-linear matched-quality estimates.
Every estimate includes the two measured quantizer endpoints. No extrapolation.
Nonmonotone rate/quality segments are excluded and counted, not repaired.
"""
import argparse, collections, csv, json, math, statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('rows');p.add_argument('out');a=p.parse_args()
rows=[json.loads(s) for s in Path(a.rows).read_text().splitlines() if s.strip()]
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
groups=collections.defaultdict(list)
for r in rows:
 c=r['config'];key=(r['source'],c['width'],c['height'],c['backend'],c.get('bit_depth',8),c.get('chroma','420'),c['speed'],c['quantizer'],c['threads'],c.get('tune'),c.get('scm'),c.get('sb128',False),r.get('svt_reference') or c.get('svt_reference') or ('svt-hybrid-3115c0c1b23e860dfd75c94f6740e0298182dd13' if c['backend'] in ('zenav1-svt','c-svt-av1') else ''),c.get('zen_intra_edge_filter',False),r['binary_sha256'],r['source_sha256'],r['timing_scope'])
 groups[key].append(r)
fields=['source','width','height','backend','depth','chroma','preset','q','threads','tune','scm','sb128','svt_reference','zen_intra_edge_filter','binary_sha256','source_sha256','timing_scope']
cells=[]
for key,rs in sorted(groups.items(),key=lambda x:str(x[0])):
 if len({r['output_sha256'] for r in rs})!=1:raise SystemExit(f'nondeterministic output: {key}')
 r=rs[0];ns=[v['api_elapsed_ns']/1e6 for v in rs]
 d=dict(zip(fields,key));d.update(bytes=r['bytes'],bpp=8*r['bytes']/(d['width']*d['height']),ssim2=r['ssimulacra2'],codec_only_ssim2=r['ssimulacra2_codec_only'],
  conversion_ceiling_ssim2=r.get('conversion_ceiling_ssimulacra2',r.get('i420_ceiling_ssimulacra2')),median_ms=statistics.median(ns),min_ms=min(ns),max_ms=max(ns),rounds=len(rs),
  output_sha256=r['output_sha256'],binary_sha256=r['binary_sha256'],source_sha256=r['source_sha256'],revision=r['revision'])
 cells.append(d)
def save(name,ds):
 if not ds:return
 with (out/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(ds[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(ds)
save('cells.tsv',cells)
curves=collections.defaultdict(list)
for c in cells:curves[tuple(c[k] for k in fields if k!='q')].append(c)
matched=[];reversals=[]
for key,cs in curves.items():
 cs.sort(key=lambda c:c['q'])
 for lo,hi in zip(cs,cs[1:]):
  if lo['ssim2']<hi['ssim2'] or lo['bytes']<hi['bytes']:
   reversals.append({k:lo[k] for k in fields}|{'q_next':hi['q'],'ssim2':lo['ssim2'],'ssim2_next':hi['ssim2'],'bytes':lo['bytes'],'bytes_next':hi['bytes']});continue
  for target in [50,60,70,80,90,95,98]:
   if not hi['ssim2']<=target<lo['ssim2']:continue
   weight=(target-hi['ssim2'])/(lo['ssim2']-hi['ssim2'])
   interp=lambda field:math.exp(weight*math.log(lo[field])+(1-weight)*math.log(hi[field]))
   d={k:lo[k] for k in fields if k!='q'};d.update(target_ssim2=target,estimated_bytes=interp('bytes'),estimated_ms=interp('median_ms'),q_low=lo['q'],q_high=hi['q'],ssim2_low_q=lo['ssim2'],ssim2_high_q=hi['ssim2'],rounds=min(lo['rounds'],hi['rounds']))
   matched.append(d)
save('matched.tsv',matched);save('reversals.tsv',reversals)
print(json.dumps({'rows':len(rows),'cells':len(cells),'matched_brackets':len(matched),'reversed_segments':len(reversals),'min_rounds':min(c['rounds'] for c in cells)}))
