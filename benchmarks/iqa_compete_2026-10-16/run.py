#!/usr/bin/env python3
"""Separately admit bounded IQA cells; retain every attempt and explicit refusal."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import time
import uuid
from chunks import combine
from admission_recovery import recover
from references import load_references
from parity import qualify_rows
from batch import BatchLease

THREAD_ENV = ['RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'RTEN_NUM_THREADS', 'OPENBLAS_NUM_THREADS']
TIMING = ['n', 'median_ns', 'p10_ns', 'p90_ns', 'mean_ns', 'score', 'ref_score',
          'abs_delta', 'relative_delta', 'load_before', 'load_after']
MODES = ['score', 'score+map']
LEARNED = {'lpips-alex','lpips-vgg','lpips-squeeze','stlpips-alex','stlpips-vgg','dists','deepdc','ckdn','afine','dreamsim','arniqa','maniqa','liqe','musiq','hyperiqa','paq2piq','wadiqam-fr-live','wadiqam-fr-tid','wadiqam-nr-live','wadiqam-nr-tid','wd','pieapp'}


# These frozen CLI routes have no native-map API, independently of score startup.
NO_MAPS = LEARNED | {'brisque', 'niqe', 'ilniqe', 'cambi'}

PYTHON_METRICS = {'scikit-image':['psnr','ssimc'], 'piq':['ssimc-pyiqa-f32','ms-ssim-pyiqa-f32','gmsd','ms-gmsd','fsim','vsi','haarpsi','mdsi','vif','iwssim-piq','brisque','lpips-vgg','dists'], 'torchmetrics':['psnr','ssimc-pyiqa-f32','ms-ssim-pyiqa-f32'], 'pytorch-msssim':['ssimc-pyiqa-f32','ms-ssim-pyiqa-f32'], 'lpips':['lpips-alex','lpips-vgg','lpips-squeeze'], 'DISTS-pytorch':['dists'],'stlpips-pytorch':['stlpips-alex','stlpips-vgg'],'DeepDC-PyTorch':['deepdc'],'flip-evaluator':['flip-v17'],'cvvdp':['cvvdp']}
PYTHON_MAPS = {
    "scikit-image": {"ssimc"}, "torchmetrics": {"ssimc-pyiqa-f32"},
    "flip-evaluator": {"flip-v17"}, "cvvdp": {"cvvdp"},
}

def normalize_nomap(row):
    impl, metric = row.get('impl'), row.get('metric_name')
    unavailable = (metric in NO_MAPS if impl == 'zenmetrics' else
        metric in PYTHON_METRICS.get(impl, ()) and metric not in PYTHON_MAPS.get(impl, ()))
    if (not unavailable or row.get('mode') != 'score+map' or row.get('status') == 'unsupported'):
        return
    if row.get('n', 0):
        raise ValueError('A frozen no-map route unexpectedly returned timed maps')
    row['source_status'] = row['status']
    row.update(status='unsupported', notes=row.get('notes', '') +
               '; This selected adapter route has no qualified native-map API; score startup outcome is separate')


def admission_reason(text):
    if 'busy load' in text or 'IQA_BUSY ' in text:
        return 'host_load'
    if re.search(r'run-heavy: only [0-9]+ MiB [^\n]*refusing to launch', text):
        return 'host_memory'
    if 'benchmark admission failed' in text and 'orphaned running reservation' in text:
        return 'zenbench_orphaned_reservation'
    return None

def gpu_dispatch_failure(text):
    return (('DSD-' in text and 'panicked at' in text)
            or 'wgpu error: Validation Error' in text
            or ('cubecl-wgpu:' in text and ('storage buffer binding' in text or 'workgroup' in text)))

def prior_admission_reason(row):
    status=row.get('source_status',row.get('status',''))
    if not status.startswith('skipped:process_exit_'):
        return None
    log=row.get('log_path') or row.get('log')
    if not log:
        legacy=re.search(r'(?:full log|retained attempt log) (.+)$',row.get('notes',''))
        if not legacy:raise ValueError('Refusal row lacks its log pointer')
        log=legacy[1]
    return admission_reason(Path(log).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, allow_nan=False, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())

def metadata(binary):
    cpu = platform.processor()
    tier='neon-capable' if platform.machine().lower() in ['arm64','aarch64'] else 'platform capability not queried'
    if Path('/proc/cpuinfo').exists():
        info=Path('/proc/cpuinfo').read_text()
        flags=next((s.split(':',1)[1].split() for s in info.splitlines() if s.startswith('flags')),[])
        tier='v4-capable' if 'avx512f' in flags else ('v3-capable' if 'avx2' in flags else 'scalar-capable')
        cpu = next(s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                   if s.startswith('model name'))
    elif platform.system() == 'Darwin':
        cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    gpu = None
    if shutil.which('nvidia-smi'):
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                             capture_output=True, text=True, check=False).stdout.strip()
    return dict(host=socket.gethostname().split('.')[0], cpu_model=cpu,
                os=platform.platform(), arch=platform.machine(), gpu=gpu,
                impl='zenmetrics', impl_version='zenmetrics-cli 0.6.0',
                impl_binary_sha=sha(binary), device='cpu', simd_tier=tier)

def family(name):
    for prefix in ['ms-ssim', 'ms-gmsd', 'stlpips', 'lpips', 'cw-ssim', 'vmaf', 'psnrhvs',
                   'psnr', 'ssimc', 'ssim', 'fsim', 'vsi', 'haarpsi', 'wadiqam']:
        if name == prefix or name.startswith(prefix + '-'):
            return prefix
    return name.removesuffix('-gpu')

def resource(text):
    match = re.search(r'run-heavy: done rc=(\d+) (\d+)s \| peak-RSS ([0-9.]+)GiB '
                      r'\| min-avail (\d+)MiB \| peak-load ([0-9.]+)', text)
    if not match:
        return None
    return dict(exit_code=int(match[1]), seconds=int(match[2]), peak_rss_gib=float(match[3]),
                min_available_mib=int(match[4]), peak_load=float(match[5]))

def run_cell(args, row, models, *, restore_chunks=True):
    count, name = row['threads'], row['metric_name']
    env = os.environ.copy()
    env.update({key: str(count) for key in THREAD_ENV})
    env.update(ZENBENCH_CPU_POLICY='unrestricted', ZENBENCH_NO_SAVE='1',
               TMPDIR=str(args.root / 'scratch'), ZENMETRICS_NN_ENGINE=args.engine,
               ZENMETRICS_NN_ACCEPT_WEIGHTS='ckdn,afine,dreamsim,liqe,hyperiqa,paq2piq,pieapp')
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    if args.engine=='ort-cuda':env['ORT_DISABLE_TELEMETRY']='1'
    if args.model_root:
        env['ZENMETRICS_NN_MODEL_DIR'] = str(args.model_root)
    if name in models:
        env['ZENMETRICS_NN_MODEL_DIR'] = models[name]
    row['model_dir'] = env.get('ZENMETRICS_NN_MODEL_DIR')
    if platform.system() == 'Linux':
        cpus = sorted(os.sched_getaffinity(0))
        chosen = [2 if 2 in cpus else cpus[0]] if count == 1 else cpus
        row.update(cpu_affinity=chosen,
                   placement_policy='unrestricted descriptive hybrid/all-core study; explicit OS affinity')
    else:
        chosen = None
        row['placement_policy'] = 'unrestricted descriptive study; no uniform-class certification'
    stem = f"{name}-{row['size_name']}-{count}t"
    pieces={mode:[] for mode in MODES}
    completed={}
    chunk_ledger=args.root/'chunk-attempts.jsonl'
    identity=['host','os','metric_name','size_name','threads','device','pair_sha256','impl_binary_sha','engine','model_dir','scope_seconds','chunk_budget_ms','sample_target_ns','ort_shared_library_sha256','ort_disable_telemetry']
    def retain(value):
        mode=value['mode']
        if value['status'] not in ['ok','partial','variant_mismatch']:
            completed[mode]=value
        elif value['status']=='partial':
            pieces[mode].append(value)
            aggregate=combine(pieces[mode])
            if aggregate['status']!='partial':completed[mode]=aggregate
        elif pieces[mode] and value.get('round_means_ns'):
            pieces[mode].append(value);completed[mode]=combine(pieces[mode])
        else:completed[mode]=value
    if restore_chunks and chunk_ledger.exists():
        for line in chunk_ledger.read_text().splitlines():
            value=json.loads(line)
            if all(value.get(k)==row.get(k) for k in identity) and not prior_admission_reason(value):retain(value)
    if set(completed)==set(MODES):return list(completed.values())
    while True:
        lease=getattr(args,'batch_lease',None)
        if lease:
            actual_scope,fresh_hold=lease.prepare(args.scope_seconds)
        else:actual_scope,fresh_hold=args.scope_seconds,True
        row['actual_scope_seconds']=actual_scope
        attempt = uuid.uuid4().hex[:12]
        config = args.root / 'cells' / f'{stem}-{attempt}.json'
        log = args.root / 'logs' / f'{stem}-{attempt}.log'
        row['created_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        row['attempt'] = attempt
        config.write_text(json.dumps(row) + '\n')
        command = [str(args.binary.resolve()), str(config.resolve())]
        if args.python_executable:
            command = [str(args.python_executable.absolute()),str(args.python_adapter),str(config.resolve())]
        # Set metric budgets after the independent build/job cap.
        command = ['env',*[f'{key}={env[key]}' for key in THREAD_ENV],
                   'ZENBENCH_CPU_POLICY=unrestricted','ZENBENCH_NO_SAVE=1',*command]
        if chosen:
            command = ['taskset','-c',','.join(map(str,chosen)),*command]
        system=platform.system()
        if system not in ['Linux','Darwin']:
            raise RuntimeError('A qualified native resource wrapper must be configured before running on '+system)
        timeout='gtimeout' if system=='Darwin' else 'timeout'
        memory='12G' if system=='Darwin' else '16G'
        host=args.host or socket.gethostname().split('.')[0]
        command = [str(Path.home()/'work/claudehints/scripts/run-heavy'),
                   '--mem',memory,'--jobs',str(max(1,(os.cpu_count() or 1)-4)),
                   '--',timeout,'--kill-after=20s',f'{actual_scope}s',*command]
        if not lease:command=['flock',str(Path.home()/f'tmp/{host}-heavy.lock'),*command]
        print('SUBMIT', stem, attempt, flush=True)
        with log.open('x') as f:
            process = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT,
                                     check=False,pass_fds=(lease.fd,) if lease else ())
        text = log.read_text()
        recovery=[]
        if process.returncode and platform.system()=='Linux':
            proofs=[json.loads(line[len('IQA_PROCESS '):]) for line in text.splitlines() if line.startswith('IQA_PROCESS ')]
            if len(proofs)>1:raise ValueError('Multiple process proofs in one Rust cell')
            if proofs:recovery=recover(proofs[0],args.root/'recovered-admission')
            for record in recovery:append(args.root/'admission-recovery.jsonl',record)
        admission=admission_reason(text)
        if lease and process.returncode==124 and not fresh_hold:
            admission='batch_budget_exhausted'
        if lease and admission:lease.close()
        if admission:
            append(args.root / 'admission.jsonl', dict(metric=name, size=row['size_name'],
                   threads=count, attempt=attempt, log=str(log), log_sha256=sha(log),
                   reason=admission+'; no metric timing admitted'))
            print('ADMISSION_WAIT', stem, log, flush=True)
            time.sleep(30)
            continue
        ort_budgets=[int(line.split()[1]) for line in text.splitlines() if line.startswith('IQA_ORT_THREADS ')]
        if any(value!=count for value in ort_budgets):raise ValueError('ORT session budget differs from cell budget')
        row['ort_intra_threads_configured']=ort_budgets or None
        row['ort_inter_threads_configured']=1 if ort_budgets else None
        rows=[]
        for line in text.splitlines():
            if line.startswith('IQA_ROW '):rows.append(json.loads(line[8:]))
            elif line.startswith('IQA_PARTIAL '):rows.append(json.loads(line[len('IQA_PARTIAL '):]))
        dispatch_failed=row['device']!='cpu' and gpu_dispatch_failure(text)
        for value in rows:
            if dispatch_failed and value['status'] in ['ok','partial','variant_mismatch']:
                value['diagnostic_source_status']=value['status']
                value.update(status='skipped:gpu_dispatch_failure',notes=value.get('notes','')+'; Device-service panic or WGPU validation error in retained log; score is unqualified')
            value['ort_intra_threads_configured']=ort_budgets or None
            value['ort_inter_threads_configured']=1 if ort_budgets else None
        modes={value['mode'] for value in rows}
        if len(rows)!=len(modes) or modes-set(MODES):raise ValueError(f'invalid emitted modes: {log}')
        if process.returncode or modes!=set(MODES):
            reason={124:f'timeout_{actual_scope}s',137:'memory_guard_or_limit'}.get(process.returncode,f'process_exit_{process.returncode}')
            for mode in set(MODES)-modes:
                rows.append(dict(row,mode=mode,status=f'skipped:{reason}',n=0,
                                 notes=f'No completed timing; retained attempt log {log}'))
        for value in rows:normalize_nomap(value)
        chunked=any(value['status']=='partial' for value in rows) or any(pieces.values())
        for value in rows:
            value['initialization_diagnostics']=[json.loads(s[len('IQA_WARMUP '):]) for s in text.splitlines() if s.startswith('IQA_WARMUP ')]
            value['admission_recovery']=recovery
            value['resource']=resource(text)
            value['resource_memory_method']=('macOS monitored process-tree RSS; no hard cgroup cap; availability is free+inactive+speculative page proxy' if platform.system()=='Darwin' else 'Linux run-heavy cgroup MemoryMax=16G when available; wrapper log states actual scope support')
            value['log_sha256']=sha(log);value['log_path']=str(log)
            if chunked:append(chunk_ledger,value)
            retain(value)
        print('RESULT',stem,[(v['mode'],v['status'],v.get('n')) for v in rows],flush=True)
        if set(completed)==set(MODES):return list(completed.values())
        print('CHUNK_CONTINUE',stem,{mode:sum(v['n'] for v in values) for mode,values in pieces.items()},flush=True)
        # The scope and lock have both ended. A fresh admitted, warmed process
        # contributes further actual rounds; no duration is extrapolated.

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary', type=Path)
    p.add_argument('pairs', type=Path)
    p.add_argument('root', type=Path)
    p.add_argument('--host', help='Public CPU-derived host alias, independent of local hostname')
    p.add_argument('--sizes', default='cid22,2k')
    p.add_argument('--metrics')
    p.add_argument('--exclude-metrics',default='')
    p.add_argument('--gpu-runtime', choices=['cuda','wgpu'])
    p.add_argument('--engine', choices=['rten','ort-cuda'], default='rten')
    p.add_argument('--ort-library', type=Path)
    p.add_argument('--cuda-libraries', help='Colon-separated CUDA/cuDNN library directories')
    p.add_argument('--threads', default='1,all')
    p.add_argument('--model-root', type=Path)
    p.add_argument('--models-json', type=Path)
    p.add_argument('--batch-seconds',type=int,default=0,help='Optional 1..1100-second holds spanning multiple cells; per-cell resource wrappers remain enabled')
    p.add_argument('--scope-seconds', type=int, default=1100, help='1..1100; shorter scopes support timeout-recovery controls')
    p.add_argument('--chunk-budget-ms', type=int, default=850000)
    p.add_argument('--sample-target-ns', type=int, default=1000000)
    p.add_argument('--python-package')
    p.add_argument('--python-executable', type=Path)
    p.add_argument('--reference-jsonl', action='append', type=Path, default=[])
    args = p.parse_args()
    if not 0<=args.batch_seconds<=1100:raise ValueError('batch hold must be within 0..1100 seconds')
    if not 1 <= args.scope_seconds <= 1100:raise ValueError('scope must be within 1..1100 seconds')
    args.root.mkdir(exist_ok=True, parents=True)
    for directory in ['cells', 'logs']:
        (args.root / directory).mkdir(exist_ok=True)
    models = json.loads(args.models_json.read_text()) if args.models_json else {}
    scales = {}
    scale_path=Path(__file__).resolve().parents[2]/'crates/zenmetrics-nn/models/jnd_scale.tsv'
    for line in scale_path.read_text().splitlines():
        if not line or line.startswith('#') or line.startswith('column\t'):continue
        column,value,*_=line.split('\t');scales[column]=float(value)
    def qualify(results):
        # CPU controllers can append new references while this controller runs.
        return qualify_rows(results,load_references(args.reference_jsonl),
                            inventory,scales,args.python_package)

    if args.ort_library:
        os.environ['ORT_DYLIB_PATH'] = str(args.ort_library.resolve())
    if args.cuda_libraries:
        os.environ['LD_LIBRARY_PATH'] = args.cuda_libraries + ':' + os.environ.get('LD_LIBRARY_PATH','')
    meta = metadata(args.binary)
    meta.update(device=args.gpu_runtime or ('cuda' if args.engine == 'ort-cuda' else 'cpu'), engine=args.engine, gpu_runtime=args.gpu_runtime)
    if args.ort_library:
        meta['ort_disable_telemetry']=True
        meta.update(ort_shared_library=str(args.ort_library.resolve()),ort_shared_library_sha256=sha(args.ort_library))
    if args.cuda_libraries:meta['cuda_library_dirs']=args.cuda_libraries.split(':')
    if args.host:
        meta["host"] = args.host
    if args.python_package:
        if not args.python_executable:raise ValueError('--python-package requires --python-executable')
        adapter=Path(__file__).with_name('python_cell.py');digest=sha(adapter)
        snapshot=args.root/f'python-adapter-{digest}.py'
        if not snapshot.exists():snapshot.write_bytes(adapter.read_bytes())
        if sha(snapshot)!=digest:raise ValueError('adapter snapshot hash mismatch')
        args.python_adapter=snapshot.resolve()
        meta.update(engine='numpy' if args.python_package in ['scikit-image','flip-evaluator'] else 'torch',impl=args.python_package,package=args.python_package,impl_binary_sha=sha(args.python_executable),adapter_sha256=digest)
    output = args.root / f"{meta['host']}.jsonl"
    inventory = json.loads(subprocess.check_output([args.binary, 'inventory'], text=True))
    if not (args.root / 'inventory.json').exists():
        (args.root / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    done = set()
    if output.exists():
        for line in output.read_text().splitlines():
            r = json.loads(line)
            if not prior_admission_reason(r):done.add((r['metric_name'], r['size_name'], r['threads'], r['mode'], r['device'],r['os']))
    selected = set(args.metrics.split(',')) if args.metrics else {r['metric_name'] for r in inventory}
    if args.metrics=='learned':selected=set(LEARNED)
    excluded=set(filter(None,args.exclude_metrics.split(',')))
    if not excluded <= {r['metric_name'] for r in inventory}:raise ValueError('excluded metric absent from executable inventory')
    if args.python_package and not args.metrics:
        selected = set(PYTHON_METRICS[args.python_package])
    if args.gpu_runtime and not args.metrics:
        selected = {r['metric_name'] for r in inventory if r['gpu']}
    if args.engine == 'ort-cuda' and not args.metrics:
        selected &= LEARNED
    selected-=excluded
    if not selected <= {r['metric_name'] for r in inventory}:
        raise ValueError('selected metric absent from executable inventory')
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    threads = [available if value == 'all' else int(value) for value in args.threads.split(',')]
    args.batch_lease=None
    if args.batch_seconds:
        host=args.host or socket.gethostname().split('.')[0]
        args.batch_lease=BatchLease(Path.home()/f'tmp/{host}-heavy.lock',args.batch_seconds,args.root/'batch-holds.jsonl')
    try:
        for pair in json.loads(args.pairs.read_text()):
            if pair['size_name'] not in args.sizes.split(','):
                continue
            paths = {key: str((args.pairs.parent / pair[key]).resolve())
                     for key in ['reference', 'distorted']}
            hashes = {key: sha(path) for key, path in paths.items()}
            pair_sha = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
            for count in threads:
                for metric in inventory:
                    name = metric['metric_name']
                    if name not in selected:
                        continue
                    if all((name, pair['size_name'], count, mode, meta['device'],meta['os']) in done for mode in MODES):
                        continue
                    row = dict(meta | pair | paths, scope_seconds=args.scope_seconds, chunk_budget_ms=args.chunk_budget_ms, sample_target_ns=args.sample_target_ns, metric_name=name, metric_family=family(name), declared_columns=metric["columns"],
                               threads=count, mode='score', pair_sha256=pair_sha, input_sha256=hashes)
                    if not args.python_package:
                        row['engine']=args.engine if name in LEARNED else ('cubecl' if args.gpu_runtime else 'native')
                    row.update({key: None for key in TIMING})
                    results=run_cell(args, row, models)
                    mismatch=qualify(results) if (meta['device']!='cpu' or args.python_package) else False
                    if meta['device']!='cpu' and mismatch:
                        # i265 requires reproduction: preserve the entire first
                        # attempt and rerun the same bounded cell, including timing.
                        for value in results:append(args.root/'gpu-mismatch-attempts.jsonl',value)
                        first_attempts=[{k:v.get(k) for k in ['mode','score','abs_delta','status','log_path','log_sha256']} for v in results]
                        results=run_cell(args,row,models,restore_chunks=False)
                        qualify(results)
                        for value in results:value['mismatch_first_attempt']=first_attempts
                    for result in results:
                        append(output, result)
                        done.add((name, pair['size_name'], count, result['mode'], meta['device'],meta['os']))
                    if args.python_package and any(r['status'].startswith('skipped:process_exit_') for r in results):
                        raise RuntimeError('Competitor adapter failed; inspect retained logs before proceeding')
    finally:
        if args.batch_lease:args.batch_lease.close()

if __name__ == '__main__':
    main()
