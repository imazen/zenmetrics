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
    bound = manifest['parity_abs_bound']
    if not math.isfinite(bound) or bound < 0:
        raise ValueError('invalid parity gate')


def qualify_block(block, manifest):
    """One contaminated arm rejects its entire block; inner samples are not pairs."""
    if block['manifest_sha256'] != identity(manifest):
        raise ValueError('manifest identity mismatch')
    if len(block['arms']) != 2 or {a['arm'] for a in block['arms']} != {'ours', 'canonical'}:
        raise ValueError('missing or duplicate arm')
    values = {}
    scores = {}
    for arm in block['arms']:
        name = arm['arm']
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
        if not isinstance(arm['score'], (int, float)) or not math.isfinite(arm['score']):
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
               sys.executable, manifest['resources_script'], 'timeout', '120',
               'taskset', '-c', str(manifest['cpu']), *spec['argv']]
    env = os.environ.copy()
    env.update({k: '1' for k in ('RAYON_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','RTEN_NUM_THREADS')})
    env.pop('PYTHONPATH', None)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(directory), ZENBENCH_NO_SAVE='1', ZENBENCH_CPU_POLICY='unrestricted')
    with log.open('x') as output:
        code = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT).returncode
    text = log.read_text()
    accounting = json.loads(contention.read_text()) if contention.exists() else {}
    samples, score = [], None
    if spec['protocol'] == 'iqa-row':
        rows = [json.loads(l[8:]) for l in text.splitlines() if l.startswith('IQA_ROW ')]
        if len(rows) == 1 and rows[0].get('status') == 'ok':
            row = rows[0]
            score = row['score']
            samples = [{'ns': s['elapsed_ns'][0] / s['iterations'], 'gate_clean': s['gate_clean']}
                       for s in row.get('zenbench_samples', [])]
    elif spec['protocol'] == 'canonical-samples':
        for line in text.splitlines():
            if line.startswith('SAMPLE '):
                _, _, ns, value = line.split()
                samples.append({'ns': int(ns), 'gate_clean': accounting.get('contaminated') is False})
                score = float(value)
    else:
        raise ValueError('unknown protocol')
    return dict(arm=name, block_id=block_id, build_identity=spec['build_identity'],
                input_identity=identity(manifest['inputs']), scope=manifest['scope'][name],
                exit_code=code, contaminated=accounting.get('contaminated', True), samples=samples,
                score=score, log=str(log), log_sha256=digest(log),
                contention=str(contention), resource_lines=[l for l in text.splitlines() if l.startswith('run-heavy:')])


def acquire(manifest, output, count):
    output.mkdir(parents=True, exist_ok=False)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    blocks = []
    # One bounded block per lock hold. No sleep-for-quiet loop under the locks.
    for index in range(count):
        if (output/'STOP').exists():
            break
        with (Path.home()/'tmp/lilith-heavy.lock').open('a') as first, (Path.home()/'tmp/lilith-wsl-heavy.lock').open('a') as second:
            fcntl.flock(first, fcntl.LOCK_EX)
            fcntl.flock(second, fcntl.LOCK_EX)
            if (output/'STOP').exists():
                break
            verify(manifest)
            block_id = str(uuid.uuid4())
            directory = output/block_id
            directory.mkdir()
            order = ['ours','canonical'] if index % 2 == 0 else ['canonical','ours']
            block = dict(block_id=block_id, manifest_sha256=identity(manifest), order=order, arms=[])
            (directory/'started.json').write_text(json.dumps(block))
            for name in order:
                block['arms'].append(run_arm(name,manifest,block_id,directory))
            try:
                verify(manifest)
            except ValueError as error:
                block['identity_error'] = str(error)
                for arm in block['arms']:
                    arm['contaminated'] = True
            (directory/'block.json').write_text(json.dumps(block,indent=2))
            blocks.append(block)
        # Expensive statistical processing is outside the acquisition boundary.
        result = summarize(blocks,manifest)
        (output/'summary.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    if (output/'STOP').exists():
        (output/'cancelled.json').write_text(json.dumps({'completed_blocks':len(blocks)}))
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
