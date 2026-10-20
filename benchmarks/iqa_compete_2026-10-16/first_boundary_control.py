#!/usr/bin/env python3
"""Independent fresh-process original/candidate bit controls; no timing ratios.

Caller owns shared heavy admission. Each case executes twice in alternating
order, with raw logs and failures retained. This never runs native comparisons.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(bindings):
    if not bindings:
        raise ValueError('nonempty source/runtime/input bindings required')
    for name, expected in bindings.items():
        path = Path(name)
        if not path.is_absolute() or len(expected) != 64 or sha(path) != expected:
            raise ValueError('binding missing or changed: ' + name)


def score_bits(text, auxiliary):
    observations = [line.split() for line in text.splitlines()
                    if line.startswith(('COLD ', 'SAMPLE '))]
    if not observations:
        raise ValueError('original-call scalar observations missing')
    values = []
    for row in observations:
        if len(row) != 3 or not math.isfinite(float(row[1])) or float(row[1]) <= 0:
            raise ValueError('invalid original-call observation')
        score = float(row[2])
        if not math.isfinite(score):
            raise ValueError('nonfinite original-call score')
        values.append(struct.pack('>d', score).hex())
    if len(set(values)) != 1:
        raise ValueError('original-call score bits vary within process')
    reference = [line.split() for line in text.splitlines() if line.startswith('REFERENCE_SCORE ')]
    if bool(reference) != auxiliary or (auxiliary and len(reference) != len(observations)):
        raise ValueError('reference-image NR score observations missing or unexpected')
    ref_bits = []
    for row in reference:
        if len(row) != 2 or not math.isfinite(float(row[1])):
            raise ValueError('nonfinite reference-image NR score')
        ref_bits.append(struct.pack('>d', float(row[1])).hex())
    if len(set(ref_bits)) > 1:
        raise ValueError('reference-image NR score bits vary within process')
    return dict(headline=values[0], reference=ref_bits[0] if auxiliary else None)


def validate_case(case, bindings):
    if not isinstance(case['threads'], int) or case['threads'] <= 0:
        raise ValueError('positive explicit thread budget required')
    for field in ('reference', 'distorted', 'candidate_receipt'):
        if str(Path(case[field]).resolve(strict=True)) not in bindings:
            raise ValueError('case artifact not bound: ' + field)
    receipt = json.loads(Path(case['candidate_receipt']).read_text())
    for field in ('source', 'output'):
        if bindings.get(receipt[field]) != receipt[field + '_sha256']:
            raise ValueError('candidate source/output does not match preparation receipt')
    for role in ('original', 'candidate'):
        command = case[role]['command']
        if not command or not Path(command[0]).is_absolute():
            raise ValueError('explicit absolute runtime required')
        for arg in command:
            path = Path(arg)
            if path.is_file() and str(path.resolve()) not in bindings:
                raise ValueError('command file not bound: ' + arg)
        if role == 'candidate' and receipt['output'] not in command:
            raise ValueError('candidate command does not use emitted adapter')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    plan_data = args.plan.read_bytes()
    plan = json.loads(plan_data)
    if plan['schema'] != 1 or not plan['cases']:
        raise ValueError('nonempty first-boundary control plan required')
    bindings = plan['bindings']
    verify(bindings)
    helper_dir = Path(plan['harness_directory']).resolve(strict=True)
    for name in ('haarpsi_primary.py', 'octave_primary.py', 'supervision.py'):
        if str(helper_dir / name) not in bindings:
            raise ValueError('supervisor closure not bound: ' + name)
    sys.path.insert(0, str(helper_dir))
    from haarpsi_primary import execute
    controller_hash = sha(__file__)
    if len({case['id'] for case in plan['cases']}) != len(plan['cases']):
        raise ValueError('duplicate control case')
    for case in plan['cases']:
        if Path(case['id']).name != case['id'] or case['id'] in ('.', '..'):
            raise ValueError('control id must be a single directory name')
        validate_case(case, bindings)
    args.output.mkdir()
    root = args.output.resolve()
    (root / 'plan.json').write_bytes(plan_data)
    rows = []
    for case in plan['cases']:
        directory = root / case['id']
        directory.mkdir()
        row = dict(case=case['id'], metric=case['metric'], observations=[], errors=[])
        for iteration in range(2):
            for role in (('original', 'candidate') if iteration == 0 else ('candidate', 'original')):
                verify(bindings)
                out = directory / f'{iteration}-{role}'
                out.mkdir()
                env = dict(os.environ, TMPDIR=str(out), LC_ALL='C', CUDA_VISIBLE_DEVICES='',
                           HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
                for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                            'RAYON_NUM_THREADS', 'RTEN_NUM_THREADS'):
                    env[key] = str(case['threads'])
                env.update(case[role].get('environment', {}))
                try:
                    command = case[role]['command']
                    rc, wrapped = execute(command, env, out, role, 1100)
                    (out / 'command.json').write_text(json.dumps(wrapped) + '\n')
                    if rc:
                        raise ValueError(f'original-call process exited {rc}; see {out}')
                    text = (out / f'{role}-stdout.log').read_text()
                    if role == 'candidate' and sum(line.startswith('COLD ') for line in text.splitlines()) != 1:
                        raise ValueError('candidate must emit exactly one first evaluation')
                    bits = score_bits(text, case['auxiliary_reference'])
                    row['observations'].append(dict(role=role, iteration=iteration, bits=bits,
                        stdout_sha256=sha(out / f'{role}-stdout.log'),
                        stderr_sha256=sha(out / f'{role}-stderr.log')))
                except Exception as error:
                    row['errors'].append(dict(role=role, iteration=iteration, error=str(error)))
                verify(bindings)
        observed = [item['bits'] for item in row['observations']]
        row['passed'] = not row['errors'] and len(observed) == 4 and all(v == observed[0] for v in observed)
        (directory / 'result.json').write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
        rows.append(row)
        print('FIRST_BOUNDARY_CONTROL ' + json.dumps(row), flush=True)
    verify(bindings)
    if sha(__file__) != controller_hash or args.plan.read_bytes() != plan_data:
        raise ValueError('control controller or plan changed')
    result = dict(schema=1, passed=all(row['passed'] for row in rows), cases=rows,
        plan_sha256=hashlib.sha256(plan_data).hexdigest(), controller_sha256=controller_hash,
        qualified_timing_cells=0, qualification='Original-call bit identity only; no native numerical or timing qualification')
    (root / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
