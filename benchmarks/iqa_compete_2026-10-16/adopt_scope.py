#!/usr/bin/env python3
"""Retain a completed orphaned scope after its controller-only cancellation."""
import argparse,json,pathlib
from run import append,coverage_samples,normalize_nomap,resource,sha
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__)
for key in ['config','log','contention','output']:p.add_argument(key,type=P)
a=p.parse_args();text=a.log.read_text();accounting=json.loads(a.contention.read_text())
if 'run-heavy: done ' not in text:raise RuntimeError('Scope has not completed')
config=json.loads(a.config.read_text());rows={}
for line in text.splitlines():
 for prefix in ['IQA_ROW ','IQA_PARTIAL ']:
  if line.startswith(prefix):
   value=json.loads(line[len(prefix):]);rows[value['mode']]=value
for mode in ['score','score+map']:
 row=rows.get(mode,dict(config,mode=mode,status='skipped:completed_scope_without_timing_row',n=0))
 target=coverage_samples(row['size_name'])
 row.update(coverage_first=True,required_samples=target,source_sampling_protocol='legacy ten-round scope; interpreted against owner coverage minimum without another metric call',
  log_path=str(a.log.resolve()),log_sha256=sha(a.log),resource=resource(text),contention=accounting,
  contention_receipt=str(a.contention.resolve()),contention_peak_cores=accounting['max_competing_interval_cores'],
  timing_quality='contaminated' if accounting['contaminated'] else 'clean',
  initialization_diagnostics=[json.loads(s[len('IQA_WARMUP '):]) for s in text.splitlines() if s.startswith('IQA_WARMUP ')])
 if row['status']=='partial':
  row['source_status']='partial'
  row['status']=('variant_mismatch' if row.get('gpu_call_bit_variation_count',0) else 'ok') if row['n']>=target else 'skipped:scope_sample_limit'
 row['notes']=row.get('notes','')+'; Controller-only cancellation prevented continuation; active bounded scope completed and actual rows retained'
 normalize_nomap(row);append(a.output,row)
 print(json.dumps({k:row.get(k) for k in ['metric_name','size_name','threads','mode','status','n','required_samples','timing_quality']}))
