#!/usr/bin/env python3
"""Freeze canonical scalar CPU reruns using existing workers and verified RGB assets."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from prepare_durable import sha
from mac_service import verify

HERE = Path(__file__).resolve().parent
SIZES = ('cid22', 'aic-small', 'aic-medium', '2k', 'aic-large', '4k')


def cells(metrics, pairs):
    result = []
    for size in SIZES:
        pair = pairs[size]
        if max(pair['width'], pair['height']) >= 7680:
            raise ValueError('8K geometry refused')
        for metric in metrics:
            for threads in (1, 8):
                for phase, repeats in [('cold', 3), ('warm', 1)]:
                    for repeat in range(repeats):
                        result.append(dict(metric_name=metric, size_name=size,
                            width=pair['width'], height=pair['height'], threads=threads,
                            reference=f'inputs/{size}-reference.rgb',
                            distorted=f'inputs/{size}-distorted.rgb', device='cpu', impl='zenmetrics',
                            only_mode='score', timing_phase=phase, cold_repeat=repeat,
                            required_samples=1 if phase == 'cold' else 10,
                            sample_target_ns=100_000_000, coverage_first=False,
                            chunk_budget_ms=800_000, locked_process_accounting=True,
                            input_method=pair['method']))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'portable', 'inputs', 'output', 'snapshot'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--metrics', default='changed', choices=('changed', 'all', 'dvifmish'))
    p.add_argument('--zenbench', type=Path)
    p.add_argument('--cpu-set')
    a = p.parse_args()
    if bool(a.zenbench) != bool(a.cpu_set):
        p.error('placement requires both zenbench and cpu-set')
    meta = json.loads((a.portable / 'run.meta').read_bytes())
    inventory = json.loads(subprocess.check_output([str(a.binary.resolve()), 'inventory']))
    from run import LEARNED
    eligible = [r['metric_name'] for r in inventory if r['compiled'] and not r['gpu']
                and r['metric_name'] not in LEARNED | {'ahiq', 'dmm'}]
    configs = [json.loads(f.read_bytes()) for f in a.portable.glob('*.json')
               if f.name not in ('inputs.meta.json',)]
    metrics = list(dict.fromkeys(c['metric_name'] for c in configs if 'metric_name' in c)) if a.metrics == 'changed' else ([m for m in eligible if m.startswith('dvifmish-')] if a.metrics == 'dvifmish' else eligible)
    if not metrics or set(metrics) - set(eligible):
        raise ValueError('requested routes absent from native non-NN inventory')
    # Put absolute-luminance refusals and expensive MDCT last without changing ingress.
    metrics = [m for m in metrics if m not in ('hdrvdp', 'hdrvdp3', 'mdctpsnr')] + [m for m in metrics if m in ('hdrvdp', 'hdrvdp3', 'mdctpsnr')]
    provenance = json.loads((HERE / 'inputs.meta.json').read_bytes())
    pairs = {r['size_name']: r for r in provenance['derivatives']}
    for size in SIZES:
        for role in ('reference', 'distorted'):
            f = a.inputs / f'{size}-{role}.rgb'
            if sha(f) != pairs[size]['sha256'][role] or f.stat().st_size != pairs[size]['width'] * pairs[size]['height'] * 3:
                raise ValueError('canonical RGB hash/geometry mismatch: ' + str(f))
    source = meta['build_commit']
    names = ['Cargo.toml', 'Cargo.lock', 'benchmarks/heaptrack/drivers/cpu_profile/Cargo.toml',
             'benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete.rs']
    source_hashes = {n: sha(a.snapshot / 'work/zenmetrics' / n) for n in names}
    a.output.mkdir(parents=True, exist_ok=False)
    for directory in ('bin', 'inputs', 'tools'):
        (a.output / directory).mkdir()
    shutil.copy2(a.binary, a.output / 'bin/iqa-compete')
    for size in SIZES:
        for role in ('reference', 'distorted'):
            name = f'{size}-{role}.rgb'
            shutil.copy2(a.inputs / name, a.output / 'inputs' / name)
    for name in ('durable.py', 'admission_recovery.py', 'mac_service.py', 'select_durable.py', 'inputs.meta.json'):
        shutil.copy2(HERE / name, a.output / name)
    wrapper = Path.home() / 'work/claudehints/scripts/run-heavy'
    shutil.copy2(wrapper, a.output / 'tools/run-heavy')
    companion = wrapper.with_name('run-heavy-macos.py')
    if companion.exists():
        shutil.copy2(companion, a.output / 'tools/run-heavy-macos.py')
    build = dict(source_commit=source, binary_sha256=sha(a.binary),
                 features='existing native worker; see original run.meta and snapshot manifests',
                 original_run_meta=meta, snapshot=str(a.snapshot), source_files=source_hashes,
                 inventory=inventory, provenance='Existing built worker reused; no new compilation claimed')
    patch = a.snapshot.parent / 'dvifmish-patch-v2.json'
    if patch.exists():
        build['private_dependency_patch'] = json.loads(patch.read_bytes())
    counters = a.snapshot / 'mac-cpu-counters.meta.json'
    if counters.exists():
        build['zenbench_private_correction'] = json.loads(counters.read_bytes())
    (a.output / 'build.json').write_text(json.dumps(build, indent=2) + '\n')
    hashes = {str(f.relative_to(a.output)): sha(f) for f in (a.output / 'inputs').glob('*')}
    manifest = dict(schema=1, binary='bin/iqa-compete', binary_sha256=sha(a.binary),
        input_sha256=hashes, cells=cells(metrics, pairs), metadata=dict(source_commit=source,
        runner_sha256={n: sha(a.output / n) for n in ('durable.py', 'admission_recovery.py', 'mac_service.py')},
        build_receipt_sha256=sha(a.output / 'build.json'), metrics=metrics, sizes=list(SIZES), threads=[1, 8],
        protocol_id='durable-cold3-warm10-100ms-scalar-v1',
        cold_definition='fresh process: scorer construction and first evaluation after RGB input loading; OS caches not flushed',
        warm_definition='10 zenbench rounds, 100 ms target per round, after initialization and warmup',
        portable_evidence_policy='supplemental only; no portable cells retained as canonical',
        deadline_is_stop=False))
    if a.zenbench:
        shutil.copy2(a.zenbench, a.output / 'tools/zenbench')
        hashes['tools/zenbench'] = sha(a.zenbench)
        manifest['metadata']['cpu_placement'] = dict(policy='performance', cpus=list(map(int, a.cpu_set.split(','))), zenbench='tools/zenbench')
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    sums = {str(f.relative_to(a.output)): sha(f) for f in a.output.rglob('*') if f.is_file()}
    (a.output / 'SHA256SUMS.json').write_text(json.dumps(sums, indent=2) + '\n')
    verify(a.output)
    print(json.dumps(dict(bundle=str(a.output), cells=len(manifest['cells']), metrics=len(metrics), binary_sha256=sha(a.binary))))


if __name__ == '__main__':
    main()
