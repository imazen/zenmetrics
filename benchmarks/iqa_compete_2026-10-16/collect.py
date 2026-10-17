#!/usr/bin/env python3
"""Choose explicitly ordered qualified attempts; never discard source ledgers."""
import argparse,hashlib,json,pathlib,collections
from run import family, LEARNED, normalize_nomap, prior_admission_reason, native_input_refusal
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('ledgers',nargs='+',type=P);p.add_argument('--output',type=P,required=True);p.add_argument('--only-binary-sha');a=p.parse_args()
assert not a.output.exists(),'Choose a new output or preserve the prior collection first'
selected={};sources=[];admission_refusals=[]
for ledger in a.ledgers:
 digest=hashlib.sha256(ledger.read_bytes()).hexdigest();sources.append(dict(path=str(ledger),sha256=digest))
 for index,line in enumerate(ledger.read_text().splitlines(),1):
  row=json.loads(line)
  if a.only_binary_sha and row.get('impl_binary_sha')!=a.only_binary_sha:continue
  admission=prior_admission_reason(row)
  if admission:
   admission_refusals.append(dict(ledger=str(ledger),line=index,reason=admission));continue
  if 'contamination_classification' in row:
   quality=row['contamination_classification']
   if quality not in ['clean','contaminated'] or row.get('contaminated')!=(quality=='contaminated'):raise ValueError('Inconsistent external host protocol quality')
   row['timing_quality']=quality
  if row.get('coverage_first') and row.get('parity_checked') is False and (row['impl']!='zenmetrics' or row['device']!='cpu'):
   row.update(ref_score=None,abs_delta=None,relative_delta=None,comparison_status='Not requalified in coverage-first pass; self_warmup_score is separate')
  if row['impl']=='libvmaf' and row['status'].startswith('skipped:process_exit_'):
   refusal=native_input_refusal(row['metric_name'],P(row['log']).read_text())
   if refusal:
    row.update(source_status=row['status'],status='skipped:'+refusal,notes=row.get('notes','')+'; Public libvmaf entry point refuses its derived CAMBI window geometry; original log retained')
  normalize_nomap(row);key=tuple(row.get(k) for k in ['host','os','metric_name','impl','device','threads','mode','size_name'])
  if row['impl']=='zenmetrics':
   row['original_engine']=row.get('engine')
   row['engine']='cubecl' if row.get('gpu_runtime') and row['metric_name'].endswith('-gpu') else ('rten' if row['metric_name'] in LEARNED and row['device']=='cpu' else ('ort-cuda' if row['metric_name'] in LEARNED else 'native'))
  row.update(metric_family=family(row['metric_name']),source_ledger=str(ledger),source_ledger_sha256=digest,source_line=index)
  if row['impl']=='zenmetrics' and row['metric_name'].startswith('vmaf') and row['status']=='ok' and not row.get('model_preload_adapter'):
   row['status']='skipped:diagnostic_model_parsing_inside_timer'
  if row['impl']=='zenmetrics' and row.get('engine')=='ort-cuda' and row['status'] in ['ok','variant_mismatch'] and (not row.get('ort_intra_threads_configured') or row.get('ort_disable_telemetry') is not True):
   row['diagnostic_source_status']=row['status']
   row['status']='skipped:diagnostic_unverified_ort_thread_budget'
  selected[key]=row
with a.output.open('x') as f:
 for key,row in sorted(selected.items()):f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
receipt=dict(only_binary_sha256=a.only_binary_sha,sources=sources,excluded_admission_refusals=admission_refusals,selection='Last supplied ledger wins per cell; last row wins within ledger. Source attempts remain retained.',rows=len(selected),statuses=dict(collections.Counter(r['status'].split(':')[0] for r in selected.values())),output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())
a.output.with_suffix('.meta.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
