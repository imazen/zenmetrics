#!/usr/bin/env python3
"""Check gradient-policy scalar identity, then replace only unexecuted plans.

The queue reads each plan when it starts that job. Existing helpers, weights,
bounds and native workers stay intact. Author-default plans remain as backups.
Run under the same heavy admission as the measurements, using the Chainer venv.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

from compare_canonical import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queue', type=Path)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    old_helper = base / 'canonical_wadiqam_cell.py'
    new_helper = base / 'canonical_wadiqam_inference_cell.py'
    jobs = json.loads(args.queue.read_text())['jobs']
    selected = []
    for job in jobs:
        path = Path(job['plan'])
        plan = json.loads(path.read_text())
        names = {case['metric'] for case in plan['cases']}
        if len(names) == 1 and next(iter(names)).startswith('wadiqam-'):
            selected.append((job, path, plan))
    if len(selected) != 4 or {next(iter({c['metric'] for c in p['cases']})) for _, _, p in selected} != {
        'wadiqam-fr-live', 'wadiqam-fr-tid', 'wadiqam-nr-live', 'wadiqam-nr-tid'
    }:
        raise ValueError('expected exactly the four original WaDIQaM routes')
    targets = {str(path.resolve()) for _, path, _ in selected}
    for job in jobs:
        if targets & set(json.loads(Path(job['plan']).read_text())['bindings']):
            raise ValueError('another plan binds a future plan file')
    for job, path, plan in selected:
        if Path(job['output']).exists() or path.with_suffix(path.suffix + '.author-default.bak').exists():
            raise ValueError('plan is already acquired or has a retained backup')
        if any(case['canonical']['command'][1] != str(old_helper) for case in plan['cases']):
            raise ValueError('unexpected reference callable')
    config_paths = {plan['cases'][0]['canonical']['command'][2] for _, _, plan in selected}
    if len(config_paths) != 1:
        raise ValueError('control requires one explicit original-reference configuration')
    config_path = Path(next(iter(config_paths)))
    config = json.loads(config_path.read_text())
    os.environ['SETUPTOOLS_USE_DISTUTILS'] = 'stdlib'
    import chainer
    import numpy as np
    from canonical_wadiqam_inference_cell import load_model
    chainer.global_config.train = False
    chainer.global_config.cudnn_deterministic = True
    original_policy = bool(chainer.global_config.enable_backprop)
    controls = []
    inputs = {}
    weights = {}
    for _, path, plan in selected:
        case = plan['cases'][0]
        command = [part.format_map(dict(case, samples=3))
                   for part in case['canonical']['command']]
        command[1] = str(new_helper)
        environment = dict(os.environ)
        environment.pop('SETUPTOOLS_USE_DISTUTILS', None)
        setup = subprocess.run([*command, '--setup-only'], env=environment,
                               capture_output=True, text=True, timeout=120)
        if setup.returncode or 'REFERENCE_INFERENCE enable_backprop=false' not in setup.stdout:
            raise ValueError('inference entry point refused: ' + setup.stderr)
        mode, dataset = case['metric'].split('-')[1:]
        model, oracle = load_model(config, case['metric'])
        weight = Path(config['weights']) / f'{mode}_{dataset}_weighted.model'
        weights[str(weight)] = sha(weight)
        if plan['bindings'].get(str(weight)) != weights[str(weight)]:
            raise ValueError('control checkpoint differs from the frozen plan')
        images = []
        for role in ('reference', 'distorted'):
            image = Path(case[role])
            digest = sha(image)
            if plan['bindings'][str(image)] != digest:
                raise ValueError('control input differs from the frozen plan')
            inputs[str(image)] = digest
            images.append(np.fromfile(image, np.uint8).reshape(case['height'], case['width'], 3))
        reference, distorted = images
        values = {}
        for enabled in (True, False):
            chainer.global_config.enable_backprop = enabled
            scores = [oracle.score_image(model, distorted, reference)] if mode == 'fr' else [
                oracle.score_image(model, distorted), oracle.score_image(model, reference)]
            if not all(math.isfinite(score) for score in scores):
                raise ValueError('nonfinite scalar in gradient-policy control')
            values[str(enabled).lower()] = dict(scores=scores, bits=[struct.pack('>d', s).hex() for s in scores])
        if values['true']['bits'] != values['false']['bits']:
            raise ValueError('gradient policy changes scalar outputs: ' + case['metric'])
        controls.append(dict(metric=case['metric'], old_plan_sha256=sha(path),
                             inference_setup_stdout=setup.stdout, **values))
        print('BIT_IDENTICAL', case['metric'], values['false']['scores'], flush=True)
    receipt = dict(default_enable_backprop=original_policy, chainer=chainer.__version__,
        numpy=np.__version__, source=str(Path(__file__).resolve()), source_sha256=sha(__file__),
        original_helper_sha256=sha(old_helper), inference_helper_sha256=sha(new_helper),
        config_sha256=sha(config_path), oracle_sha256=sha(config['oracle']), inputs=inputs,
        weights=weights, controls=controls,
        cpu_overlay='NumPy aliases in author modules; Chainer CUDA backend unchanged',
        scope='same original models and RGB input; only gradient recording differs between control arms')
    with args.receipt.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    revisions = []
    for (job, path, plan), control in zip(selected, controls):
        if Path(job['output']).exists() or sha(path) != control['old_plan_sha256']:
            raise ValueError('future job or plan changed during the control')
        new = copy.deepcopy(plan)
        new['replaces_unexecuted_plan_sha256'] = control['old_plan_sha256']
        new['bindings'][str(new_helper)] = sha(new_helper)
        new['bindings'][str(Path(__file__).resolve())] = sha(__file__)
        new['bindings'][str(args.receipt.resolve())] = sha(args.receipt)
        for case in new['cases']:
            case['canonical']['command'][1] = str(new_helper)
            case['canonical']['label'] += ' / inference'
            case['scope'] += '; author-module CPU aliases, Chainer gradient recording disabled'
        pending = path.with_suffix(path.suffix + '.inference-next')
        with pending.open('x') as stream:
            json.dump(new, stream, indent=2)
            stream.write('\n')
        backup = path.with_suffix(path.suffix + '.author-default.bak')
        path.rename(backup)
        pending.rename(path)
        revisions.append(dict(plan=str(path), backup=str(backup), old_sha256=sha(backup), new_sha256=sha(path)))
    with args.receipt.with_name(args.receipt.stem + '-plans.json').open('x') as stream:
        json.dump(dict(control_sha256=sha(args.receipt), revisions=revisions), stream, indent=2)
        stream.write('\n')
    print('UPDATED_UNEXECUTED_PLANS', len(revisions), flush=True)


if __name__ == '__main__':
    main()
