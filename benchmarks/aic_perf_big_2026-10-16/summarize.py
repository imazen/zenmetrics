"""Audit complete bounded gates and base/new/base/new measurements."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('scratch',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def resource(path):
    text=path.read_text()
    matches=re.findall(r'run-heavy: done rc=(\d+) (\d+)s \| peak-RSS ([0-9.]+)GiB \| min-avail (\d+)MiB \| peak-load ([0-9.]+)',text)
    assert len(matches)==1,(path,matches)
    rc,seconds,rss,available,load=matches[0]
    assert int(rc)==0 and int(seconds)<1200,(path,matches)
    return dict(seconds=int(seconds),peak_rss_gib=float(rss),min_available_mib=int(available),peak_load=float(load),log_sha256=sha(path))
rows=[];cells=[]
for metric in ['cambi','butter']:
 for width,height in [(1024,1024),(3355,2516)]:
  for threads in [1,16]:
   key=f'{metric}-{width}x{height}-{threads}t';root=a.scratch/'round14-abab-final'/key
   assert root.with_suffix('.exit').read_text().strip()=='0',key
   cell_rows=[]
   for sequence,name in enumerate(['base1','new1','base2','new2']):
    path=root/f'{name}.log';text=path.read_text()
    match=re.search(r'(\d+)x(\d+) 5 reps: ([0-9.]+) ms/call score (\S+) bits ([0-9a-f]+)',text)
    assert match and (int(match[1]),int(match[2]))==(width,height),(path,match)
    loads=[float(x) for x in re.search(r'load=(.*)',text)[1].split()[:3]]
    rss=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',text)[1])
    instruction_match=re.search(r'^\s*([\d,]+)\s+cpu_core/instructions/',text,re.M)
    cycle_match=re.search(r'^\s*([\d,]+)\s+cpu_core/cycles/',text,re.M)
    assert (instruction_match is not None)==(threads==1),(path,'instructions')
    assert (cycle_match is not None)==(threads==1),(path,'cycles')
    binary=Path(re.search(r'^binary=(.*?) threads=',text,re.M)[1])
    record=dict(metric=metric,width=width,height=height,thread_budget=threads,sequence=sequence,variant=name,mean_five_calls_ms=float(match[3]),score_bits=match[5],instructions=(int(instruction_match[1].replace(',','')) if instruction_match else None),cycles=(int(cycle_match[1].replace(',','')) if cycle_match else None),maximum_rss_kib=rss,load_1m=loads[0],load_5m=loads[1],load_15m=loads[2],binary_sha256=sha(binary),log_sha256=sha(path))
    cell_rows.append(record);rows.append(record)
   assert len({r['score_bits'] for r in cell_rows})==1,key
   assert cell_rows[0]['binary_sha256']==cell_rows[2]['binary_sha256'],key
   assert cell_rows[1]['binary_sha256']==cell_rows[3]['binary_sha256'],key
   base=[cell_rows[i]['mean_five_calls_ms'] for i in [0,2]]
   new=[cell_rows[i]['mean_five_calls_ms'] for i in [1,3]]
   cells.append(dict(key=key,base_ms=base,new_ms=new,median_base_ms=statistics.median(base),median_new_ms=statistics.median(new),base_over_new=statistics.median(base)/statistics.median(new),base_peak_rss_kib=max(cell_rows[i]['maximum_rss_kib'] for i in [0,2]),new_peak_rss_kib=max(cell_rows[i]['maximum_rss_kib'] for i in [1,3]),resource=resource(root.with_suffix('.log'))))
gates={name:resource(a.scratch/f'round14-{name}.log') for name in ['ci-final','feature-matrix-final','oracles-final','parity-final','residuals-final']}
matrix=(a.scratch/'round14-feature-matrix-final.log').read_text();assert '12/12 combinations passed' in matrix
oracle=a.scratch/'round14-oracles-final'
ladder=json.loads((oracle/'butter-ladder-verified.json').read_text());aic=json.loads((oracle/'butter-aic-verified.json').read_text())
assert len(ladder)==27 and len(aic)==61
assert max(r['oracle_error'] for r in ladder)<=4.8e-7
cambi={}
for dataset,expected_count in [('ladder',27),('aic',61)]:
 values=json.loads((oracle/f'cambi-precise-{dataset}-verified.json').read_text());assert len(values)==expected_count
 scored=[r for r in values if 'ours' in r];refused=[r for r in values if 'ours_error' in r]
 assert len(scored)+len(refused)==expected_count
 assert all(isinstance(r.get('oracle'),(int,float)) and 'oracle_failure' not in r for r in scored)
 assert all(r.get('oracle') is None and r.get('oracle_error') for r in refused)
 if dataset=='aic':assert len(scored)==61
 cambi[dataset]=dict(scored=len(scored),matching_refusals=len(refused),maximum_oracle_error=max(r['oracle_error'] for r in scored))
residual=json.loads((a.scratch/'round14-residuals-final.json').read_text());assert residual['numeric_residual_rows']==122
summary=dict(schema=1,host='i265',method='Base,new,base,new; each new process averages five calls; median of two process means per role. No confidence interval or cross-host extrapolation.',threading='CAMBI uses ordered column regions with the existing Rayon pool above its per-scale size threshold; its one-worker and small-scale paths use the original serial kernel. Butteraugli remains serial within one image. 16 is the requested thread budget.',target_cpu_native=False,cells=cells,gates=gates,oracle=dict(butter_ladder_maximum_error=max(r['oracle_error'] for r in ladder),butter_ladder_unchanged_bound=4.8e-7,butter_aic_maximum_error=max(r['oracle_error'] for r in aic),cambi=cambi),residual_audit=residual,full_measurements_path=str(a.output.with_suffix('.tsv')))
with a.output.with_suffix('.tsv').open('x',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
summary['full_measurements_sha256']=sha(a.output.with_suffix('.tsv'))
with a.output.with_suffix('.json').open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
for cell in cells:print(cell['key'],cell['median_base_ms'],cell['median_new_ms'],cell['base_over_new'])
print(json.dumps(dict(gates=gates,oracle=summary['oracle']),indent=2))
