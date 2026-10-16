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

THREAD_ENV = ['RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'RTEN_NUM_THREADS', 'OPENBLAS_NUM_THREADS']
TIMING = ['n', 'median_ns', 'p10_ns', 'p90_ns', 'mean_ns', 'score', 'ref_score',
          'abs_delta', 'relative_delta', 'load_before', 'load_after']
MODES = ['score', 'score+map']

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

def run_cell(args, row, models):
    count, name = row['threads'], row['metric_name']
    env = os.environ.copy()
    env.update({key: str(count) for key in THREAD_ENV})
    env.update(ZENBENCH_CPU_POLICY='unrestricted', ZENBENCH_NO_SAVE='1',
               TMPDIR=str(args.root / 'scratch'), ZENMETRICS_NN_ENGINE=args.engine,
               ZENMETRICS_NN_ACCEPT_WEIGHTS='ckdn,afine,dreamsim,liqe,hyperiqa,paq2piq,pieapp')
    Path(env['TMPDIR']).mkdir(exist_ok=True)
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
    while True:
        attempt = uuid.uuid4().hex[:12]
        config = args.root / 'cells' / f'{stem}-{attempt}.json'
        log = args.root / 'logs' / f'{stem}-{attempt}.log'
        row['created_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        row['attempt'] = attempt
        config.write_text(json.dumps(row) + '\n')
        command = [str(args.binary.resolve()), str(config.resolve())]
        if args.python_executable:
            command = [str(args.python_executable.absolute()),str(Path(__file__).with_name('python_cell.py').resolve()),str(config.resolve())]
        if chosen:
            # Set metric budgets after run-heavy sets its independent build/job cap.
            command = ['taskset', '-c', ','.join(map(str, chosen)), 'env',
                       *[f'{key}={env[key]}' for key in THREAD_ENV],
                       'ZENBENCH_CPU_POLICY=unrestricted', 'ZENBENCH_NO_SAVE=1', *command]
            command = ['flock', str(Path.home() / 'tmp/i265-heavy.lock'),
                       str(Path.home() / 'work/claudehints/scripts/run-heavy'),
                       '--mem', '16G', '--jobs', '16', '--', 'timeout', '--kill-after=20s',
                       '1100s', *command]
        print('RUN', stem, attempt, flush=True)
        with log.open('x') as f:
            process = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT,
                                     check=False)
        text = log.read_text()
        if 'busy load' in text or 'IQA_BUSY ' in text:
            append(args.root / 'admission.jsonl', dict(metric=name, size=row['size_name'],
                   threads=count, attempt=attempt, log=str(log), log_sha256=sha(log),
                   reason='host load refused; no metric timing admitted'))
            print('ADMISSION_WAIT', stem, log, flush=True)
            time.sleep(30)
            continue
        rows = [json.loads(s[8:]) for s in text.splitlines() if s.startswith('IQA_ROW ')]
        modes = {r['mode'] for r in rows}
        if len(rows) != len(modes) or modes - set(MODES):
            raise ValueError(f'invalid emitted modes: {log}')
        if process.returncode or modes != set(MODES):
            reason = {124: 'timeout_1100s', 137: 'memory_limit_16GiB'}.get(
                      process.returncode, f'process_exit_{process.returncode}')
            for mode in set(MODES) - modes:
                rows.append(dict(row, mode=mode, status=f'skipped:{reason}', n=0,
                            notes=f'No completed timing; retained attempt log {log}'))
        for result in rows:
            result['resource'] = resource(text)
            result['log_sha256'] = sha(log)
            result['log_path'] = str(log)
        print('RESULT', stem, [(r['mode'], r['status']) for r in rows], flush=True)
        return rows

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary', type=Path)
    p.add_argument('pairs', type=Path)
    p.add_argument('root', type=Path)
    p.add_argument('--host', help='Public CPU-derived host alias, independent of local hostname')
    p.add_argument('--sizes', default='cid22,2k')
    p.add_argument('--metrics')
    p.add_argument('--gpu-runtime', choices=['cuda','wgpu'])
    p.add_argument('--engine', choices=['rten','ort-cuda'], default='rten')
    p.add_argument('--ort-library', type=Path)
    p.add_argument('--cuda-libraries', help='Colon-separated CUDA/cuDNN library directories')
    p.add_argument('--threads', default='1,all')
    p.add_argument('--model-root', type=Path)
    p.add_argument('--models-json', type=Path)
    p.add_argument('--python-package')
    p.add_argument('--python-executable', type=Path)
    p.add_argument('--reference-jsonl', action='append', type=Path, default=[])
    args = p.parse_args()
    args.root.mkdir(exist_ok=True, parents=True)
    for directory in ['cells', 'logs']:
        (args.root / directory).mkdir(exist_ok=True)
    models = json.loads(args.models_json.read_text()) if args.models_json else {}
    references = {}
    for path in args.reference_jsonl:
        for line in path.read_text().splitlines():
            value = json.loads(line)
            if value['device']=='cpu' and value['mode']=='score' and value['status']=='ok':
                references[(value['metric_name'],value['size_name'],value['threads'])]=value
    scales = {}
    scale_path=Path(__file__).resolve().parents[2]/'crates/zenmetrics-nn/models/jnd_scale.tsv'
    for line in scale_path.read_text().splitlines():
        if not line or line.startswith('#') or line.startswith('column\t'):continue
        column,value,*_=line.split('\t');scales[column]=float(value)
    def qualify(results):
        mismatch=False
        for value in results:
            if value.get('score') is None or value['status'] not in ['ok','variant_mismatch']:continue
            name=value['metric_name'];base=name.removesuffix('-gpu')
            ref=references.get((base,value['size_name'],value['threads']))
            if ref is None:continue
            reference=ref['score'];delta=abs(value['score']-reference)
            if name.endswith('-gpu'):
                bounds={'ssim2-gpu':max(.1,abs(reference)*.005),'dssim-gpu':abs(reference)*.01,
                        'butteraugli-gpu':abs(reference)*.002,'iwssim-gpu':.001,'cvvdp-gpu':.001}
                bound=bounds.get(name,0.)
                source='GPU crate parity_lock/opaque parity gates; CVVDP/IW-SSIM 1e-3 absolute'
            elif args.python_package and name not in {'lpips-alex','lpips-vgg','lpips-squeeze','dists','stlpips-alex','stlpips-vgg','deepdc'}:
                bound={'psnr':1e-10,'ssimc':7.5e-7,'ssimc-pyiqa-f32':7.5e-7,'ms-ssim-pyiqa-f32':1.9e-6,'gmsd':1e-14,'fsim':5e-5,'fsimc':5e-5,'vsi':1e-4,'haarpsi':5e-5,'flip-v17':1e-6,'cvvdp':1e-3,'vif':5e-13,'iwssim-piq':1e-4,'mdsi':0.,'ms-gmsd':0.,'ms-gmsdc':0.,'brisque':15.138428e-3}.get(name,0.)
                source='docs/METRIC_PROVENANCE.md; exact comparison for variants without a documented cross-package bound'
            else:
                column=next(m['columns'][0] for m in inventory if m['metric_name']==name)
                bound=scales[column]*1e-3
                source='zenmetrics-nn/models/jnd_scale.tsv; 1e-3 JND unchanged parity gate'
            value.update(ref_score=reference,abs_delta=delta,relative_delta=delta/abs(reference) if reference else None,
                         parity_abs_bound=bound,parity_source=source,reference_binary_sha=ref['impl_binary_sha'])
            if delta>bound:value['status']='variant_mismatch';mismatch=True
        return mismatch

    if args.ort_library:
        os.environ['ORT_DYLIB_PATH'] = str(args.ort_library.resolve())
    if args.cuda_libraries:
        os.environ['LD_LIBRARY_PATH'] = args.cuda_libraries + ':' + os.environ.get('LD_LIBRARY_PATH','')
    meta = metadata(args.binary)
    meta.update(device=args.gpu_runtime or ('cuda' if args.engine == 'ort-cuda' else 'cpu'), engine=args.engine, gpu_runtime=args.gpu_runtime)
    if args.host:
        meta["host"] = args.host
    if args.python_package:
        if not args.python_executable:raise ValueError('--python-package requires --python-executable')
        meta.update(impl=args.python_package,package=args.python_package,impl_binary_sha=sha(args.python_executable),adapter_sha256=sha(Path(__file__).with_name('python_cell.py')))
    output = args.root / f"{meta['host']}.jsonl"
    inventory = json.loads(subprocess.check_output([args.binary, 'inventory'], text=True))
    if not (args.root / 'inventory.json').exists():
        (args.root / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    done = set()
    if output.exists():
        for line in output.read_text().splitlines():
            r = json.loads(line)
            done.add((r['metric_name'], r['size_name'], r['threads'], r['mode'], r['device']))
    selected = set(args.metrics.split(',')) if args.metrics else {r['metric_name'] for r in inventory}
    if args.python_package and not args.metrics:
        selected = set({'scikit-image':['psnr','ssimc'], 'piq':['ssimc-pyiqa-f32','ms-ssim-pyiqa-f32','gmsd','ms-gmsd','fsim','vsi','haarpsi','mdsi','vif','iwssim-piq','brisque'], 'torchmetrics':['psnr','ssimc-pyiqa-f32','ms-ssim-pyiqa-f32'], 'pytorch-msssim':['ssimc-pyiqa-f32','ms-ssim-pyiqa-f32'], 'lpips':['lpips-alex','lpips-vgg','lpips-squeeze'], 'DISTS-pytorch':['dists'],'stlpips-pytorch':['stlpips-alex','stlpips-vgg'],'DeepDC-PyTorch':['deepdc'],'flip-evaluator':['flip-v17'],'cvvdp':['cvvdp']}[args.python_package])
    if args.gpu_runtime and not args.metrics:
        selected = {r['metric_name'] for r in inventory if r['gpu']}
    if args.engine == 'ort-cuda' and not args.metrics:
        selected &= {'lpips-alex','lpips-vgg','lpips-squeeze','stlpips-alex','stlpips-vgg','dists','deepdc','ckdn','afine','dreamsim','arniqa','maniqa','liqe','musiq','hyperiqa','paq2piq','wadiqam-fr-live','wadiqam-fr-tid','wadiqam-nr-live','wadiqam-nr-tid','wd','pieapp'}
    if not selected <= {r['metric_name'] for r in inventory}:
        raise ValueError('selected metric absent from executable inventory')
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    threads = [available if value == 'all' else int(value) for value in args.threads.split(',')]
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
                if all((name, pair['size_name'], count, mode, meta['device']) in done for mode in MODES):
                    continue
                row = dict(meta | pair | paths, metric_name=name, metric_family=family(name),
                           threads=count, mode='score', pair_sha256=pair_sha, input_sha256=hashes)
                row.update({key: None for key in TIMING})
                results=run_cell(args, row, models)
                mismatch=qualify(results) if (meta['device']!='cpu' or args.python_package) else False
                if meta['device']!='cpu' and mismatch:
                    # i265 requires reproduction: preserve the entire first
                    # attempt and rerun the same bounded cell, including timing.
                    for value in results:append(args.root/'gpu-mismatch-attempts.jsonl',value)
                    first_attempts=[{k:v.get(k) for k in ['mode','score','abs_delta','status','log_path','log_sha256']} for v in results]
                    results=run_cell(args,row,models)
                    qualify(results)
                    for value in results:value['mismatch_first_attempt']=first_attempts
                for result in results:
                    append(output, result)
                    done.add((name, pair['size_name'], count, result['mode'], meta['device']))

if __name__ == '__main__':
    main()
