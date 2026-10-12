#!/usr/bin/env python3
"""Real IQA observation through bundle verification and the fleet runner."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

binaries, runner, case_path, root = [Path(s).resolve() for s in sys.argv[1:]]
root.mkdir(parents=True, exist_ok=False)
(root / "queue").mkdir()
case = json.loads(case_path.read_bytes())
for key in ('reference', 'distorted'):
    shutil.copyfile(case_path.parent / case[key], root / key)
    case[key] = key
(root / 'case.json').write_text(json.dumps(case))
env = dict(os.environ, RAYON_NUM_THREADS=str(case['threads']), ZENBENCH_OPERATIONS='1',
           ZENBENCH_CPU_POLICY='unrestricted', ZENBENCH_COORDINATION_DIR=str(root / 'queue'))
env.pop('ZENBENCH_RESULT_FILE', None)
def call(label, args, expected=0, extra=None):
    result = subprocess.run([str(a) for a in args], cwd=root, env=dict(env, **(extra or {})),
                            capture_output=True, timeout=90)
    (root / f'{label}.stdout').write_bytes(result.stdout)
    (root / f'{label}.stderr').write_bytes(result.stderr)
    assert result.returncode == expected, (label, result.returncode, result.stderr.decode())
    return result

correctness = json.loads(call('native', [binaries / 'latency', 'fleet', root / 'case.json']).stdout)['driver']['correctness']
for label, value in [('absolute', str(root / 'reference')), ('parent', '../reference')]:
    invalid = dict(case, reference=value)
    (root / 'invalid.json').write_text(json.dumps(invalid))
    refused = call(label, [binaries / 'memory', 'fleet', root / 'invalid.json'], 2)
    assert b'input path must be relative' in refused.stderr
(root / 'bundle.json').write_text(json.dumps({'files': {'case.json': {}, 'reference': {}}}))
refused = call('unlisted', [binaries / 'memory', 'fleet', root / 'case.json'], 2,
               {'ZENBENCH_RESULT_FILE': str(root / 'should-not-exist.json')})
assert b'IQA input is not bound by the bundle: distorted' in refused.stderr
assert not (root / 'should-not-exist.json').exists()
refused = call('memory-timing-refusal', [binaries / 'memory', '64', root / 'should-not-exist.tsv'], 2)
assert b'memory-only build' in refused.stderr
assert not (root / 'should-not-exist.tsv').exists()
for variant, backend in [('latency', 'os_peak'), ('memory', 'driver')]:
    files = {name: {'source': str(root / name), 'executable': False}
             for name in ('case.json', 'reference', 'distorted')}
    files['driver'] = {'source': str(binaries / variant), 'executable': True}
    caps = json.loads(call(f'{variant}-capabilities', [runner, 'capabilities']).stdout)
    # Native platform names used by the runner's target contract.
    process = caps['runner']
    target_os = 'darwin' if process['os'] == 'macos' else process['os']
    target = f"{process['process_arch']}-test-{target_os}"
    spec = {'schema': 1, 'target': target, 'files': files, 'plan': {
        'schema': 1, 'program': 'driver', 'args': ['fleet', 'case.json'],
        'env': {'RAYON_NUM_THREADS': str(case['threads'])}, 'backend': backend,
        'operations': 1, 'timeout_seconds': 30, 'expected_correctness': correctness,
        'workload': case}}
    (root / 'spec.json').write_text(json.dumps(spec))
    bundle = root / f'{variant}-bundle'
    call(f'{variant}-bundle', [runner, 'bundle', root / 'spec.json', bundle])
    attempt = root / f'{variant}-attempt'
    call(f'{variant}-run', [runner, 'fleet-run', bundle, '--output', attempt,
         '--experiment', 'iqa-control', '--machine', 'control-cpu', '--domain', 'unrestricted',
         '--replicate', '0', '--attempt', variant, '--wait', '10'])
    packet = json.loads((attempt / 'packet.json').read_bytes())
    assert packet['status'] == 'complete', packet
    assert packet['memory'], packet
    call(f'{variant}-read', [runner, 'fleet-read', attempt])
print(json.dumps({'bundle_controls': 2, 'input_refusals': 3, 'memory_timing_refusal': 'passed'}))
