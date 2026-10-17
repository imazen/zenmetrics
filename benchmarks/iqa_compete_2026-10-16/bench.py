#!/usr/bin/env python3
"""Build, verify inputs, run and resume a content-addressed IQA competition."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import shlex
import subprocess
import sys
import tarfile
import tomllib
import urllib.request

from resources import locked, scoped, resource_executable
from run import LEARNED, PYTHON_METRICS, metadata, sha
from native_run import ADAPTERS

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
SIZES = 'cid22,aic-small,aic-medium,aic-large,2k,4k,8k'
SOURCES = {
    'libjxl': ('https://codeload.github.com/libjxl/libjxl/tar.gz/refs/tags/v0.12.0',
               '03e9be69a30be4011f559da75328b6d7cea8ad921fabfbd551ce10bf45cdc992', 'libjxl-0.12.0'),
    'libvmaf': ('https://static.crates.io/crates/vmaf-head-sys/vmaf-head-sys-0.2.0.crate',
                'efe6ac72bb84ea796280e198f1d81454f92974e3fb4d57b021fcd9b0ff640615', 'vmaf-head-sys-0.2.0'),
}


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Checksum differs: {path}')


def fetch(name, cache):
    url, checksum, directory = SOURCES[name]
    root = cache / name
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'source.tar.gz'
    if not archive.exists():
        partial = root / 'download.partial'
        with urllib.request.urlopen(url, timeout=60) as source, partial.open('wb') as target:
            shutil.copyfileobj(source, target, 1 << 20)
        checked(partial, checksum)
        partial.rename(archive)
    checked(archive, checksum)
    destination = root / directory
    # Re-extract the verified source so local source-cache edits cannot change builds.
    with tarfile.open(archive) as source:
        source.extractall(root, filter='data')
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sizes', nargs='?', default=SIZES)
    p.add_argument('metrics', nargs='?', default='all')
    p.add_argument('--host', required=True, help='Public CPU-derived host label')
    p.add_argument('--corpus', type=Path, default=os.environ.get('IQA_CORPUS'),
                   help='AIC2026 image directory containing S67_Ref_00.png and S67_CCM_15.png')
    p.add_argument('--cache', type=Path, default=Path.home() / 'tmp/bench-iqa')
    p.add_argument('--output', type=Path)
    p.add_argument('--build-cache', type=Path, help='Owned incremental Cargo snapshot; sources are re-exported and rebuilt')
    p.add_argument('--dashboard-dir', type=Path, default=BASE / 'repro-results')
    p.add_argument('--skip-python', action='store_true', help='Explicitly exclude Python competitors')
    p.add_argument('--gpu-runtime', choices=['none', 'cuda', 'wgpu'], default='none',
                   help='Explicit backend selection; CPU-only by default')
    p.add_argument('--model-root', type=Path, help='Caller-provided licensed model weights for learned metrics')
    a = p.parse_args()
    if platform.system() not in ['Linux', 'Darwin']:
        p.error('Windows needs a qualified Job-object memory/timeout supervisor; see PORTABILITY.md')
    if not a.corpus:
        p.error('--corpus or IQA_CORPUS must name the AIC2026 corpus image directory')
    if not set(a.sizes.split(',')) <= set(SIZES.split(',')):
        p.error('unknown resolution')
    import re
    if not re.fullmatch(r'[a-z][a-z0-9-]*', a.host):
        p.error('invalid public host label')
    dirty = subprocess.check_output(['jj', 'diff', '--summary'], cwd=REPO, text=True)
    if dirty.strip():
        raise RuntimeError('Commit working-copy changes before benchmarking: ' + dirty)
    commit = subprocess.check_output(['jj', 'log', '-r', '@', '--no-graph', '-T', 'commit_id'], cwd=REPO, text=True).strip()
    cache = a.cache.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    scratch = cache / 'scratch'
    scratch.mkdir(exist_ok=True)
    snapshot = (a.build_cache or cache / 'snapshot').expanduser().resolve()
    env = dict(os.environ, TMPDIR=str(scratch), LOCK_SNAP_DIR=str(snapshot))
    env.pop('RUSTFLAGS', None)  # No inherited target-cpu or arithmetic-changing flags.
    env['CARGO_PROFILE_RELEASE_LTO'] = 'thin'
    env['IQA_GPU_RUNTIME'] = '' if a.gpu_runtime == 'none' else a.gpu_runtime
    logs = cache / 'logs' / commit
    logs.mkdir(parents=True, exist_ok=True)
    jobs = max(1, (os.cpu_count() or 1) - 4)

    def run(command, label, heavy=False):
        log = logs / f'{label}-{datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")}.log'
        actual = locked(scoped(command, seconds=1100, memory='12G' if platform.system() == 'Darwin' else '16G', jobs=jobs),
                        Path.home() / f'tmp/{a.host}-heavy.lock') if heavy else command
        print('RUN', label, 'log:', log, flush=True)
        with log.open('x') as out:
            result = subprocess.run(actual, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT)
        if heavy:
            from batch import yield_after_release
            yield_after_release(Path.home() / f'tmp/{a.host}-heavy.lock')
        if result.returncode:
            raise RuntimeError(f'{label} failed ({result.returncode}); inspect {log}')
        return log

    manifest = json.loads((BASE / 'inputs.meta.json').read_text())
    source = manifest['source']
    for name, checksum in source['encoded_sha256'].items():
        checked(a.corpus / name, checksum)
    run(['bash', str(BASE / 'build_cpu.sh'), 'cached'], 'rust-build', True)
    rust = snapshot / 'target/release/iqa-compete'
    inventory = json.loads(subprocess.check_output([rust, 'inventory'], text=True))
    selected = {r['metric_name'] for r in inventory if a.gpu_runtime != 'none' or not r['gpu']} if a.metrics == 'all' else set(a.metrics.split(','))
    if not selected <= {r['metric_name'] for r in inventory}:
        raise ValueError('Unknown metric selection')
    if selected & LEARNED and not a.model_root:
        raise ValueError('Learned metrics require --model-root with caller-provided licensed weights')
    binaries = {'rust': rust}
    for name in SOURCES:
        source_dir = fetch(name, cache / 'sources')
        build = cache / 'native' / commit / name
        binary = build / ('iqa-jxl' if name == 'libjxl' else 'libvmaf-compute')
        receipt = build / 'build.json'
        identity = dict(commit=commit, source_sha256=SOURCES[name][1], shim_sha256=sha(BASE / f'{name}_compute.cpp'))
        if receipt.exists():
            recorded = json.loads(receipt.read_text())
            if recorded['identity'] != identity:
                raise ValueError('Native build cache identity differs')
            checked(binary, recorded['binary_sha256'])
        else:
            if build.exists():
                raise RuntimeError(f'Incomplete native build retained at {build}; rename it before retrying')
            build.parent.mkdir(parents=True, exist_ok=True)
            run(['bash', str(BASE / f'build_{name}.sh'), str(source_dir), str(build)], name + '-build', True)
            receipt.write_text(json.dumps(dict(identity=identity, binary_sha256=sha(binary)), indent=2) + '\n')
        binaries[name] = binary
    python = None
    if not a.skip_python:
        project = BASE / 'python-env'
        locktext = (project / 'uv.lock').read_text()
        if 'name = "pyiqa"' in locktext.lower():
            raise ValueError('Excluded dependency in Python lock')
        env['UV_PROJECT_ENVIRONMENT'] = str(cache / 'python' / sha(project / 'uv.lock'))
        run(['uv', 'sync', '--frozen', '--project', str(project), '--python', '3.12', '--no-install-project'], 'python-sync', True)
        python = Path(env['UV_PROJECT_ENVIRONMENT']) / 'bin/python'
        run([str(python), str(BASE / 'python_preflight.py')], 'python-preflight')
    inputs = cache / 'inputs' / key(dict(commit=commit, source=source))
    if not inputs.exists():
        inputs.parent.mkdir(parents=True, exist_ok=True)
        run([str(rust), 'prepare', str((a.corpus / source['reference']).resolve()),
             str((a.corpus / source['distorted']).resolve()), str(inputs)], 'prepare', True)
    for pair in manifest['derivatives']:
        for role in ['reference', 'distorted']:
            checked(inputs / pair[role], pair['sha256'][role])
    model_files = {}
    if a.model_root:
        model_files = {str(f.relative_to(a.model_root)): sha(f) for f in sorted(a.model_root.rglob('*')) if f.is_file()}
    hardware = metadata(rust)
    identity = dict(commit=commit, host=a.host, os=platform.platform(), hardware={k:hardware[k] for k in ['cpu_model','arch','gpu','simd_tier']}, input_manifest_sha256=sha(BASE / 'inputs.meta.json'),
                    binaries={k:sha(v) for k,v in binaries.items()}, gpu_runtime=a.gpu_runtime,
                    python_lock=None if a.skip_python else sha(BASE / 'python-env/uv.lock'), models=model_files,
                    sampling='5 below 2K; 3 AIC-large/2K/4K; 1 8K; 1/all threads; both modes')
    root = (a.output or cache / 'results' / key(identity)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    meta = root / 'run.meta'
    if meta.exists() and json.loads(meta.read_text())['identity'] != identity:
        raise ValueError('Output content key differs; choose another --output')
    command_line = ['just', 'bench-iqa', *sys.argv[1:]]
    record = dict(identity=identity, build_commit=commit, content_key=key(identity), command=command_line,
                  host=a.host, hardware=hardware, sizes=a.sizes, metrics=sorted(selected),
                  compiler_flags='--cfg iqa_compete_ingress --cfg iqa_compete_cached; release LTO thin; runtime dispatch',
                  competitor_versions={'libjxl':'v0.12.0 a7a9c787', 'libvmaf':'3.2.0 from vmaf-head-sys 0.2.0'},
                  resource_wrapper=str(resource_executable()) if resource_executable() else None,
                  resource_wrapper_present=resource_executable() is not None,
                  toolchains={cmd:subprocess.check_output([cmd, '--version'], text=True).strip() for cmd in ['rustc','cargo','cmake','c++']},
                  competitors={k:dict(url=v[0],sha256=v[1]) for k,v in SOURCES.items()},
                  python_explicitly_skipped=a.skip_python)
    meta.write_text(json.dumps(record, indent=2) + '\n')
    cpu = sorted(selected - {r['metric_name'] for r in inventory if r['gpu']})
    common = ['--host', a.host, '--sizes', a.sizes, '--threads', '1,all', '--coverage-first']
    if a.model_root:
        common += ['--model-root', str(a.model_root.resolve())]
    if cpu:
        run([sys.executable,str(BASE/'run.py'),str(rust),str(inputs/'pairs.json'),str(root/'rust'),*common,'--metrics',','.join(cpu)], 'rust-cells')
    if a.gpu_runtime != 'none':
        gpu = sorted(selected & {r['metric_name'] for r in inventory if r['gpu']})
        if gpu:
            run([sys.executable,str(BASE/'run.py'),str(rust),str(inputs/'pairs.json'),str(root/'gpu'),*common,'--metrics',','.join(gpu),'--gpu-runtime',a.gpu_runtime,'--isolate-modes'], 'gpu-cells')
    reference = root / 'rust' / f'{a.host}.jsonl'
    planes = root / 'planes'
    if not planes.exists():
        run([str(rust),'prepare-planes',str(inputs/'pairs.json'),str(planes)],'planes',True)
    for name, adapters in ADAPTERS.items():
        metrics = sorted(selected & {r[0] for r in adapters})
        if metrics:
            run([sys.executable,str(BASE/'native_run.py'),name,str(binaries[name]),str(rust),str(inputs/'pairs.json'),str(planes/'planes.json'),str(root/name),
                 '--host',a.host,'--sizes',a.sizes,'--threads','1,all','--coverage-first','--reference-jsonl',str(reference),'--metrics',','.join(metrics)],name+'-cells')
    if python:
        for package, names in PYTHON_METRICS.items():
            metrics = sorted(selected & set(names))
            if metrics:
                run([sys.executable,str(BASE/'run.py'),str(rust),str(inputs/'pairs.json'),str(root/package),*common,'--metrics',','.join(metrics),
                     '--python-package',package,'--python-executable',str(python),'--reference-jsonl',str(reference)],package+'-cells')
    export(root, a.dashboard_dir, key(identity), record)
    print('RESULTS', root, flush=True)


def export(root, destination, cohort, meta):
    # Project onto the dashboard's registered competition schema, keeping raw receipts separately.
    registry = tomllib.loads((REPO / 'benchmarks/dashboard/datasets.toml').read_text())
    spec = next(r for r in registry['dataset'] if r['glob'] == 'benchmarks/iqa_compete_2026-10-16/repro-results/*/*.json')
    columns = set(spec['columns'])
    rows = []
    for path in sorted(root.glob('*/*.jsonl')):
        if path.name != f'{meta["identity"]["host"]}.jsonl':
            continue
        rows += [{k:v for k,v in json.loads(line).items() if k in columns} for line in path.read_text().splitlines()]
    out = destination / cohort
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0,len(rows),8):
        (out / f'cells-{i//8:04}.json').write_text(json.dumps(rows[i:i+8],indent=2,allow_nan=False)+'\n')
    (out / 'run.meta').write_text('commit = '+meta['identity']['commit']+'\nhost = '+meta['identity']['host']+'\nos = '+platform.system()+'\ncommand = '+shlex.join(meta['command'])+'\n'+json.dumps(meta,indent=2)+'\n')
    sys.path.insert(0,str(REPO/'scripts/dashboard'))
    import build
    count = sum(len(list(build.adapt(path,spec))) for path in sorted(out.glob('cells-*.json')))
    if count != len(rows):
        raise ValueError('Dashboard did not register every exported row')
    (root/'dashboard-registration.meta').write_text(json.dumps(dict(rows=count,glob=spec['glob'],destination=str(out)))+'\n')
    print('DASHBOARD_REGISTERED', count, out, flush=True)


if __name__ == '__main__':
    main()
