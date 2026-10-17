#!/usr/bin/env python3
"""Bounded in-process competitor cells. C/C++ shims perform timing internally."""
import argparse, atexit, hashlib, json, os, pathlib, statistics, subprocess, sys, time, uuid
from batch import BatchLease
from references import load_references, same_pair_reference
from run import metadata, sha, family, append, resource, THREAD_ENV, coverage_samples, admission_reason, prior_admission_reason
P=pathlib.Path
# Bounds are the existing parity gates, never fitted to these observations.
# Sources: docs/METRIC_PROVENANCE.md parity table; fast-ssim2 requires tier
# identity but records C++ version drift separately, so use exact comparison.
ADAPTERS={
 'libvmaf':[
 ('vmaf','vmaf','709-420',.02),('vmaf-neg','vmaf-neg','709-420',.02),
 ('vmaf-studio601','vmaf','601-420',.02),('vmaf-neg-studio601','vmaf-neg','601-420',.02),
 ('ssim-libvmaf','float_ssim','601-y',2e-4),('msssim-libvmaf','float_ms_ssim','601-y',2e-4),
 ('psnr-y-libvmaf','psnr','709-y',1e-10),('psnrhvs-daala','psnr_hvs','601-444',2e-4),
 ('cambi','cambi','601-420',1e-4),('vif','vif','601-y',5e-13)],
 'libjxl':[('butteraugli','butteraugli','rgb',0.),('ssim2','ssim2','rgb',0.)]}
RELATIVE_BOUNDS={'butteraugli':.001} # pinned butteraugli/tests/reference_parity.rs SCORE_TOLERANCE
def percentile(s,p):
 s=sorted(s);i=(len(s)-1)*p;lo=int(i);hi=min(lo+1,len(s)-1);return s[lo]+(s[hi]-s[lo])*(i-lo)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('engine',choices=ADAPTERS);p.add_argument('binary',type=P);p.add_argument('rust',type=P);p.add_argument('pairs',type=P);p.add_argument('planes',type=P);p.add_argument('root',type=P);p.add_argument('--sizes',default='cid22,2k');p.add_argument('--threads',default='1,20');p.add_argument('--coverage-first',action='store_true');p.add_argument('--reference-jsonl',type=P);p.add_argument('--batch-seconds',type=int,default=0);p.add_argument('--batch-reserve-seconds',type=int,default=120);p.add_argument('--host',default='i265');a=p.parse_args()
 a.root.mkdir(parents=True,exist_ok=True);(a.root/'logs').mkdir(exist_ok=True);(a.root/'cells').mkdir(exist_ok=True)
 meta=metadata(a.binary);meta.update(host=a.host,impl=a.engine,impl_version={'libvmaf':'3.2.0 (vmaf-head-sys 0.2.0 vendored source)','libjxl':'0.12.0'}[a.engine])
 if a.coverage_first and not a.reference_jsonl:raise ValueError('Coverage pass requires retained CPU references; no new Rust reference calls')
 lease=BatchLease(P.home()/'tmp/i265-heavy.lock',a.batch_seconds,a.root/'batch-holds.jsonl',reserve=a.batch_reserve_seconds) if a.batch_seconds else None
 if lease:atexit.register(lease.close)
 references=load_references([a.reference_jsonl]) if a.reference_jsonl else {}
 output=a.root/f'{a.host}.jsonl';done=set()
 if output.exists():
  for s in output.read_text().splitlines():
   r=json.loads(s)
   if not prior_admission_reason(r):done.add((r['metric_name'],r['size_name'],r['threads'],r['mode']))
 planes=json.loads(a.planes.read_text());pairs=json.loads(a.pairs.read_text())
 for pair in pairs:
  if pair['size_name'] not in a.sizes.split(','):continue
  paths={k:str((a.pairs.parent/pair[k]).resolve()) for k in ['reference','distorted']};hashes={k:sha(v) for k,v in paths.items()};pair_hash=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
  for threads in map(int,a.threads.split(',')):
   for metric,adapter,convention,bound in ADAPTERS[a.engine]:
    for mode in ['score','score+map']:
     key=(metric,pair['size_name'],threads,mode)
     if key in done:continue
     row=dict(meta|pair,metric_name=metric,metric_family=family(metric),threads=threads,mode=mode,pair_sha256=pair_hash,input_sha256=hashes,parity_abs_bound=bound,parity_source='docs/METRIC_PROVENANCE.md; pinned butteraugli/tests/reference_parity.rs (relative 0.001); exact where no cross-version bound is specified',n=0,status='unsupported')
     if mode=='score+map' and (a.engine=='libvmaf' or metric=='ssim2'):
      row['notes']='Public in-process adapter does not expose native map buffers; no file-based map output is timed';append(output,row);continue
     target=coverage_samples(pair['size_name']) if a.coverage_first else 10
     compare=not a.coverage_first or (pair['size_name']=='cid22' and threads==int(a.threads.split(',')[0]) and mode=='score')
     cached=same_pair_reference(references,row) if a.coverage_first and compare else None
     row.update(coverage_first=a.coverage_first,required_samples=target,parity_checked=compare,reference_jsonl=str(a.reference_jsonl) if a.reference_jsonl else None,reference_jsonl_sha256=sha(a.reference_jsonl) if a.reference_jsonl else None)
     native=pair if convention=='rgb' else next(r for r in planes if r['size_name']==pair['size_name'] and r['convention']==convention)
     base=a.pairs.parent if convention=='rgb' else a.planes.parent
     inputs={k:str((base/native[k]).resolve()) for k in paths}
     stem=f'{metric}-{pair["size_name"]}-{threads}t-{mode.replace("+","_")}-{uuid.uuid4().hex[:12]}'
     config=a.root/'cells'/f'{stem}.json';config.write_text(json.dumps(dict(pair|paths,metric_name=metric,probe_only=True)))
     command=[str(a.binary.resolve()),adapter,str(native['width']),str(native['height']),inputs['reference'],inputs['distorted'],str(threads)]
     command+=([mode,str(target)] if a.engine=='libjxl' else [str(target),'400' if convention.endswith('-y') else ('444' if convention.endswith('-444') else '420')])
     # One bounded process warms/times the native implementation, then probes
     # Rust outside its timer. Neither process start nor reference scoring is timed.
     env=os.environ.copy();env['IQA_HEAVY_LOCK_PATH']=str(P.home()/'tmp/i265-heavy.lock');env.update({k:str(threads) for k in THREAD_ENV});env['TMPDIR']=str(a.root/'scratch');P(env['TMPDIR']).mkdir(exist_ok=True)
     commands_file=a.root/'cells'/f'{stem}-commands.json';commands_file.write_text(json.dumps([command,*([[str(a.rust.resolve()),str(config.resolve())]] if compare and not a.coverage_first else [])]))
     contention_tries=[]
     while True:
      actual_scope,_=lease.prepare(1100) if lease else (1100,True)
      row.update(scope_seconds=1100,actual_scope_seconds=actual_scope,batch_hold_seconds=a.batch_seconds)
      if lease:env['IQA_HEAVY_LOCK_FD']=str(lease.fd)
      before=os.getloadavg()
      child=[sys.executable,str(P(__file__).resolve()),'--child',str(commands_file.resolve())]
      cpus=sorted(os.sched_getaffinity(0));chosen=[2 if 2 in cpus else cpus[0]] if threads==1 else cpus
      cmd=['flock',str(P.home()/'tmp/i265-heavy.lock'),str(P.home()/'work/claudehints/scripts/run-heavy'),'--mem','16G','--jobs','16','--','timeout','--kill-after=20s',f'{actual_scope}s','taskset','-c',','.join(map(str,chosen)),'env',*[f'{k}={v}' for k,v in env.items() if k in THREAD_ENV],*child]
      attempt_id=uuid.uuid4().hex[:12]
      monitor=a.root/'logs'/f'{stem}-{attempt_id}.contention.json'
      attempt_log=a.root/'logs'/f'{stem}-{attempt_id}.log'
      cmd=cmd[:2]+[sys.executable,str(P(__file__).with_name('contention.py').resolve()),str(monitor.resolve()),str(attempt_log.resolve()),'--']+cmd[2:]
      if lease:cmd=cmd[2:]
      with attempt_log.open('x') as output_log:
       completed=subprocess.run(cmd,env=env,stdout=output_log,stderr=subprocess.STDOUT,pass_fds=(lease.fd,) if lease else ())
      result=subprocess.CompletedProcess(cmd,completed.returncode,attempt_log.read_text(),'')
      if admission_reason(result.stdout+result.stderr):
       if lease:lease.close()
       if a.coverage_first:
        accounting=json.loads(monitor.read_text());contention_tries.append((accounting,result,attempt_log));break
       attempt_log=a.root/'logs'/f'{stem}-busy-{uuid.uuid4().hex[:8]}.log';attempt_log.write_text(result.stdout+result.stderr);time.sleep(30);continue
      accounting=json.loads(monitor.read_text());accounting['actual_scope_seconds']=actual_scope
      contention_tries.append((accounting,result,attempt_log))
      if accounting['contaminated'] and len(contention_tries)<(2 if a.coverage_first else 3) and (not a.coverage_first or result.returncode==0):continue
      accounting,result,selected_log=min(contention_tries,key=lambda item:(int(item[1].returncode!=0 or sum(s.startswith('SAMPLE ') for s in item[1].stdout.splitlines())<target) if a.coverage_first else 0,item[0]['max_competing_interval_cores']))
      row['actual_scope_seconds']=accounting['actual_scope_seconds']
      row.update(contention=accounting,contention_peak_cores=accounting['max_competing_interval_cores'],contention_receipt=str(selected_log.with_suffix('.contention.json')),timing_quality='contaminated' if accounting['contaminated'] else 'clean',contention_attempts=[dict(log=str(item[2]),max_competing_interval_cores=item[0]['max_competing_interval_cores']) for item in contention_tries])
      break
     log=a.root/'logs'/f'{stem}.log';log.write_text(result.stdout+result.stderr)
     native_loads={s.split(' ',1)[0]:json.loads(s.split(' ',1)[1]) for s in result.stdout.splitlines() if s.startswith(('IQA_LOAD_BEFORE ','IQA_LOAD_AFTER '))}
     row.update(load_before=native_loads.get('IQA_LOAD_BEFORE',list(before)),load_after=native_loads.get('IQA_LOAD_AFTER',list(os.getloadavg())),resource=resource(result.stdout+result.stderr),log=str(log),log_sha256=sha(log),native_input_convention=convention,native_width=native['width'],native_height=native['height'],native_input_sha256={k:sha(v) for k,v in inputs.items()})
     samples=[(int(s.split()[1]),float(s.split()[2])) for s in result.stdout.splitlines() if s.startswith('SAMPLE ')]
     probes=[json.loads(s[len('IQA_PROBE '):]) for s in result.stdout.splitlines() if s.startswith('IQA_PROBE ')]
     version=next((s[8:] for s in result.stdout.splitlines() if s.startswith('VERSION ')),None);row['impl_version']=version
     if result.returncode or (compare and not probes and not a.coverage_first) or (len(samples)<target if a.coverage_first else (len(samples)<10 and sum(t for t,_ in samples)<3e9)):
      reason=admission_reason(result.stdout) or {124:f'timeout_{actual_scope}s',137:'memory_guard_or_limit',0:'scope_sample_limit'}.get(result.returncode,f'process_exit_{result.returncode}')
      row.update(status=f'skipped:{reason}',n=len(samples),samples_ns=[t for t,_ in samples],sample_scores=[v for _,v in samples]);append(output,row);print('REFUSED',stem,row['status'],flush=True)
      if result.returncode not in [0,124,137] and not admission_reason(result.stdout):raise RuntimeError('Native competitor adapter blocked; inspect retained log before proceeding: '+str(log))
      continue
     import struct
     available=compare and (cached is not None or bool(probes))
     reference=cached['score'] if cached is not None else (struct.unpack('>d',bytes.fromhex(probes[0]['scores'][0][1]))[0] if available else None)
     column=cached.get('headline_declared_column') if cached is not None else (probes[0]['scores'][0][0] if available else None)
     row.update(parity_checked=available,reference_kind='retained_cpu_timing' if cached is not None else ('fresh_probe' if available else 'not_requalified_in_coverage_pass'),reference_binary_sha=cached.get('impl_binary_sha') if cached is not None else None)
     values=[t for t,_ in samples];scores=[v for _,v in samples];score=scores[-1];delta=abs(score-reference) if available else None
     row.update(reference_column=column,native_feature=adapter,n=len(samples),median_ns=statistics.median(values),mean_ns=statistics.mean(values),p10_ns=percentile(values,.1),p90_ns=percentile(values,.9),timed_seconds=sum(values)/1e9,score=score,ref_score=reference,abs_delta=delta,relative_delta=delta/abs(reference) if reference else None,status='ok' if not available or delta<=max(bound,abs(reference)*RELATIVE_BOUNDS.get(metric,0.)) else 'variant_mismatch',parity_rel_bound=RELATIVE_BOUNDS.get(metric,0.),samples_ns=values,sample_scores=scores,notes='Warmup one call; the recorded required number of individual timed calls. Native picture/input setup, context/model initialization and Rust reference probe excluded. Per-call metric work and native map release included. Input planes/normalized RGB are the C implementation input type; Rust RGB-to-component conversion remains inside its call.')
     if a.engine=='libjxl' and metric=='ssim2':row['notes']+=' ComputeSSIMULACRA2 has no worker-pool argument; the thread count is the declared budget, not a claim that this entry point uses all workers.'
     if metric=='psnrhvs-daala':row['notes']+=' Both APIs process full-size studio-601 YUV444 and compute all three channels; Y headline is compared.'
     if metric=='vif':row['notes']+=' libvmaf float_vif and MATLAB vifp are distinct variants; 8-bit studio-luma ingress also differs. Retain as variant comparison.';row['status']='variant_mismatch'
     append(output,row);print('RESULT',stem,row['status'],delta,flush=True)
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--child':
  print('IQA_PROCESS '+json.dumps(dict(pid=os.getpid(),pgid=os.getpgrp(),unix_s=int(time.time()))),flush=True)
  commands=json.loads(P(sys.argv[2]).read_text())
  before=list(os.getloadavg())
  print('IQA_LOAD_BEFORE '+json.dumps(before),flush=True)
  result=subprocess.run(commands[0])
  if result.returncode:raise SystemExit(result.returncode if result.returncode>0 else 128-result.returncode)
  print('IQA_LOAD_AFTER '+json.dumps(list(os.getloadavg())),flush=True)
  if len(commands)>1:subprocess.run(commands[1],check=True)
 else:main()
