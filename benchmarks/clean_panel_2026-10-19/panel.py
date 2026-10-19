"""Explicit paired blocks; preserve rejected attempts and never pool build identities."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import uuid
import signal
import containment
import workload
import runtime_closure

METRICS = set('brisque butteraugli butteraugli-original cambi cambi-aic2026 mdctpsnr psnr-y-libvmaf ssim2 vmaf vmaf-4k vmaf-neg vmaf-neg-studio601 vmaf-studio601 vmaf-v1 vsi vsi-k2 vsi-k3 vsi-k4 vsi-nodownscale'.split())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify(manifest):
    if manifest['metric'] not in METRICS or manifest['shape'] != [512, 512, 1, 'score']:
        raise ValueError('outside authorized panel')
    if manifest['scope']['ours'] != manifest['scope']['canonical']:
        raise ValueError('scope mismatch')
    if set(manifest['arms']) != {'ours', 'canonical'}:
        raise ValueError('two arms required')
    if not manifest['artifacts'] or not manifest['inputs']:
        raise ValueError('missing artifacts')
    for item in manifest['artifacts'] + manifest['inputs']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('artifact changed: ' + item['path'])
    for arm in manifest['arms'].values():
        if not arm['build_identity'] or not arm['argv']:
            raise ValueError('missing build identity')
    workload.validate_contract(manifest)
    runtime_closure.verify(manifest['native_closure'])
    bound = manifest['parity_abs_bound']
    if not math.isfinite(bound) or bound < 0:
        raise ValueError('invalid parity gate')


def qualify_block(block, manifest):
    """One contaminated arm rejects its entire block; inner samples are not pairs."""
    if manifest.get('schema') != 2:
        raise ValueError('unsupported receipt schema')
    if block.get('error') or block.get('identity_error'):
        raise ValueError('terminal block failure: ' + str(block.get('error') or block.get('identity_error')))
    if block['manifest_sha256'] != identity(manifest):
        raise ValueError('manifest identity mismatch')
    if len(block['arms']) != 2 or {a['arm'] for a in block['arms']} != {'ours', 'canonical'}:
        raise ValueError('missing or duplicate arm')
    values = {}
    scores = {}
    for arm in block['arms']:
        name = arm['arm']
        if any(arm.get('observed_workload', {}).get(k) != manifest['workload'][k] for k in workload.FIELDS):
            raise ValueError('observed workload does not match manifest')
        if arm.get('observed_validated') is not True:
            raise ValueError('unvalidated emitted workload')
        if arm['block_id'] != block['block_id']:
            raise ValueError('wrong block pairing')
        if arm['build_identity'] != manifest['arms'][name]['build_identity']:
            raise ValueError('wrong build')
        if arm['input_identity'] != identity(manifest['inputs']):
            raise ValueError('wrong input')
        if arm['scope'] != manifest['scope'][name]:
            raise ValueError('wrong scope')
        if arm['exit_code'] != 0 or arm['contaminated'] is not False:
            raise ValueError('failed or contaminated arm')
        if len(arm['samples']) != manifest['arms'][name]['sample_count'] or any(s['gate_clean'] is not True for s in arm['samples']):
            raise ValueError('unclean or missing sample')
        times = [s['ns'] for s in arm['samples']]
        if any(not math.isfinite(v) or v <= 0 for v in times):
            raise ValueError('invalid time')
        if type(arm['score']) not in (int, float) or not math.isfinite(arm['score']):
            raise ValueError('invalid score')
        values[name] = statistics.median(times)
        scores[name] = arm['score']
    if abs(scores['ours'] - scores['canonical']) > manifest['parity_abs_bound']:
        raise ValueError('parity failure')
    return values['ours'] / values['canonical']


def precision(lo, estimate, hi):
    return 0 < lo <= estimate <= hi and max((estimate-lo)/estimate, (hi-estimate)/estimate) <= .03


def summarize(blocks, manifest):
    seen = set()
    ratios, rejected = [], []
    for block in blocks:
        if block['block_id'] in seen:
            raise ValueError('duplicate block')
        seen.add(block['block_id'])
        try:
            ratios.append(qualify_block(block, manifest))
        except (ValueError, KeyError) as error:
            rejected.append({'block_id': block['block_id'], 'reason': str(error)})
    result = dict(accepted_blocks=len(ratios), rejected=rejected, qualified=False)
    # Pilot is an acquisition control. No interval or speed claim at small n.
    if len(ratios) < 10:
        return result | {'status': 'insufficient-independent-blocks'}
    estimate = statistics.median(ratios)
    rng = random.Random(1)
    draws = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(20000))
    lo, hi = draws[499], draws[19499]
    narrow = precision(lo, estimate, hi)
    crossing = lo <= 1.08 <= hi
    # This descriptive fixed-look bootstrap is not authorization for optional stopping.
    return result | dict(estimate=estimate, lo=lo, hi=hi, precision_pass=narrow,
                         threshold_crossing=crossing, status='review-required',
                         method='median of explicit block ratios; percentile bootstrap, 20000, seed1; fixed-look only')


def run_arm(name, manifest, block_id, directory):
    spec = manifest['arms'][name]
    log = directory / (name + '.log')
    contention = directory / (name + '.contention.json')
    command = [sys.executable, manifest['contention_script'], str(contention), str(log), '--',
               manifest['run_heavy'], '--mem', '12G', '--jobs', '1', '--',
               manifest['taskset'], '-c', str(manifest['cpu']), *spec['argv']]
    env = runtime_closure.environment(manifest['native_closure']['loader'])
    env.update({k: '1' for k in ('RAYON_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','RTEN_NUM_THREADS')})
    env.pop('PYTHONPATH', None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(directory), ZENBENCH_NO_SAVE='1', ZENBENCH_CPU_POLICY='unrestricted')
    with log.open('x') as output:
        code = containment.execute(command, output, env, directory/(name+'.containment.json'))
    text = log.read_text()
    containment.write(directory/(name+'.execution.json'), dict(exit_code=code, log=str(log), log_sha256=digest(log), command=command))
    accounting = json.loads(contention.read_text()) if contention.exists() else {}
    observed = workload.parse(text, name, manifest, accounting.get('contaminated') is False)
    contained = json.loads((directory/(name+'.containment.json')).read_text())
    process = observed['observed_process']
    matched = [p for p in contained['owned'] if p['pid']==process['pid'] and p['pgid']==process['pgid'] and p['exe']==str(Path(spec['argv'][0]).resolve())]
    if contained['status'] != 'empty' or contained['remaining'] or len(matched) != 1:
        raise ValueError('emitted process is not a verified contained scorer')
    observed['observed_process'] = dict(process, start=matched[0]['start'], exe=matched[0]['exe'])
    return dict(arm=name, block_id=block_id, build_identity=spec['build_identity'],
                input_identity=identity(manifest['inputs']), scope=manifest['scope'][name],
                exit_code=code, contaminated=accounting.get('contaminated', True), **observed, log=str(log), log_sha256=digest(log),
                contention=str(contention), resource_lines=[l for l in text.splitlines() if l.startswith('run-heavy:')])


def acquire(manifest, output, count):
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(sig, frame):
        raise KeyboardInterrupt('signal ' + str(sig))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return _acquire(manifest, output, count)
    finally:
        signal.signal(signal.SIGTERM, previous)


def _acquire(manifest, output, count):
    output.mkdir(parents=True, exist_ok=False)
    containment.write(output/'manifest.json', manifest)
    blocks = []
    for index in range(count):
        if (output/'STOP').exists(): break
        with (Path.home()/'tmp/lilith-heavy.lock').open('a') as first, (Path.home()/'tmp/lilith-wsl-heavy.lock').open('a') as second:
            fcntl.flock(first, fcntl.LOCK_EX)
            fcntl.flock(second, fcntl.LOCK_EX)
            if (output/'STOP').exists(): break
            block_id = str(uuid.uuid4())
            directory = output/block_id
            directory.mkdir()
            order = ['ours','canonical'] if index % 2 == 0 else ['canonical','ours']
            block = dict(block_id=block_id, manifest_sha256=identity(manifest), order=order, arms=[],
                         arm_states={n:'not-started' for n in order}, status='started')
            containment.write(directory/'started.json', block)
            phase = 'pre-verification'
            interrupted = False
            try:
                verify(manifest)
                for name in order:
                    phase = 'arm:' + name
                    block['arm_states'][name] = 'running'
                    containment.write(directory/'progress.json', block)
                    arm = run_arm(name, manifest, block_id, directory)
                    block['arms'].append(arm)
                    block['arm_states'][name] = 'completed'
                    containment.write(directory/(name+'.arm.json'), arm)
                    containment.write(directory/'progress.json', block)
                phase = 'post-verification'
                verify(manifest)
                block['status'] = 'completed'
            except BaseException as error:
                interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
                block['status'] = 'rejected'
                block['error'] = dict(phase=phase, type=type(error).__name__, message=str(error))
                if phase.startswith('arm:'):
                    block['arm_states'][phase[4:]] = 'failed'
                if phase == 'post-verification':
                    block['identity_error'] = str(error)
                    for arm in block['arms']: arm['contaminated'] = True
            finally:
                # Final receipts cannot be bypassed by another ordinary signal.
                previous = {sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
                for sig in previous: signal.signal(sig, signal.SIG_IGN)
                block['attempt_files'] = [dict(path=p.name,sha256=digest(p)) for p in sorted(directory.iterdir()) if p.is_file() and p.name not in ('block.json','progress.json')]
                containment.write(directory/'block.json', block)
                blocks.append(block)
                result = summarize(blocks, manifest)
                containment.write(output/'summary.json', result)
                print(json.dumps(result), flush=True)
                for sig,handler in previous.items(): signal.signal(sig,handler)
            if block['status'] == 'rejected':
                containment.write(output/'halted.json', dict(block_id=block_id, interrupted=interrupted))
                break
    if (output/'STOP').exists():
        containment.write(output/'cancelled.json', {'completed_blocks':len(blocks)})
    return blocks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--pilot-blocks',type=int,default=4,choices=range(1,5))
    args=parser.parse_args()
    acquire(json.loads(args.manifest.read_text()),args.output,args.pilot_blocks)


if __name__=='__main__':
    main()
