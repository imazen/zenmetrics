#!/usr/bin/env python3
"""Run explicitly bound Rust/canonical scalar comparisons, retaining both arms.

Ratios are descriptive medians of alternating acquisition blocks, not a formal
regression decision. A mismatched boundary never produces a public speed ratio.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from supervision import scoring_command


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse(text, protocol):
    if protocol == 'iqa-row':
        rows = [json.loads(s[8:]) for s in text.splitlines() if s.startswith('IQA_ROW ')]
        if len(rows) != 1 or rows[0]['status'] != 'ok':
            raise ValueError('missing successful scalar row')
        row = rows[0]
        return float(row['score']), list(map(float, row['round_means_ns'])), row
    if protocol == 'samples':
        rows = [s.split() for s in text.splitlines() if s.startswith('SAMPLE ')]
        if not rows:
            raise ValueError('canonical samples missing')
        values = [(float(r[-1]), float(r[-2])) for r in rows]
        if len({score for score, _ in values}) != 1:
            raise ValueError('canonical scores changed across samples')
        return values[0][0], [ns for _, ns in values], {}
    raise ValueError('unknown output protocol: ' + protocol)


def compare(ours, canonical, absolute, relative=0):
    if not all(math.isfinite(s) for s in (ours, canonical)):
        raise ValueError('nonfinite score')
    delta = abs(ours-canonical)
    if absolute is None:
        return delta, None
    return delta, delta <= max(absolute, relative*abs(canonical))


def summarize(blocks, matched):
    if not blocks or any(b.get('errors') or b['passed'] is False for b in blocks):
        return None
    if not matched:
        return None
    return statistics.median(b['rust_ns']/b['canonical_ns'] for b in blocks)


def run_arm(plan, case, role, root, samples):
    root.mkdir()
    env = dict(os.environ, TMPDIR=str(root), LC_ALL='C', CUDA_VISIBLE_DEVICES='',
               ZENBENCH_NO_SAVE='1', ZENBENCH_CPU_POLICY='unrestricted',
               HF_HUB_OFFLINE='1')
    for key in ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = str(case['threads'])
    if role == 'rust':
        config = dict(width=case['width'], height=case['height'], reference=case['reference'],
                      distorted=case['distorted'], metric_name=case['metric'], threads=case['threads'],
                      only_mode='score', timing_phase='warm', required_samples=samples,
                      sample_target_ns=10_000_000, chunk_budget_ms=240_000,
                      locked_process_accounting=True)
        (root/'config.json').write_text(json.dumps(config)+'\n')
        command = [plan['binary'], str(root/'config.json')]
        protocol = 'iqa-row'
    else:
        replacements = dict(case, samples=samples, output=str(root))
        command = [part.format_map(replacements) for part in case['canonical']['command']]
        protocol = case['canonical'].get('protocol', 'samples')
    wrapped = scoring_command(command, system='Linux', seconds=300, memory_log=root/'memory.txt')
    (root/'command.json').write_text(json.dumps(wrapped)+'\n')
    with (root/'stdout.log').open('x') as out, (root/'stderr.log').open('x') as err:
        completed = subprocess.run(wrapped, env=env, stdout=out, stderr=err, cwd=root)
    if completed.returncode:
        raise ValueError(f'{role} process exit {completed.returncode}; {root}/stderr.log')
    score, times, extra = parse((root/'stdout.log').read_text(), protocol)
    if len(times) < samples or not all(math.isfinite(t) and t > 0 for t in times):
        raise ValueError('insufficient or invalid timing observations')
    return dict(score=score, times_ns=times, median_ns=statistics.median(times), extra=extra)


def block(plan_path, case_id, number, output, samples):
    plan = json.loads(plan_path.read_text())
    for path, expected in plan['bindings'].items():
        if sha(path) != expected:
            raise ValueError('bound artifact changed: '+path)
    case = next(c for c in plan['cases'] if c['id'] == case_id)
    output.mkdir()
    arms = {}
    errors = {}
    for role in (('canonical', 'rust') if number % 2 == 0 else ('rust', 'canonical')):
        try:
            arms[role] = run_arm(plan, case, role, output/role, samples)
        except Exception as error:
            errors[role] = str(error)
    result = dict(order=['canonical', 'rust'] if number % 2 == 0 else ['rust', 'canonical'],
                  arms=arms, errors=errors, passed=False)
    if len(arms) == 2:
        result['delta'], result['passed'] = compare(arms['rust']['score'], arms['canonical']['score'],
                                                   case['absolute_bound'], case.get('relative_bound', 0))
        result.update(rust_ns=arms['rust']['median_ns'], canonical_ns=arms['canonical']['median_ns'])
    for path, expected in plan['bindings'].items():
        if sha(path) != expected:
            raise ValueError('bound artifact changed during comparison: '+path)
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--blocks', type=int, default=3)
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--metrics', help='Explicit comma-separated subset of the frozen plan')
    p.add_argument('--block-case')
    p.add_argument('--block-number', type=int, default=0)
    a = p.parse_args()
    a.plan = a.plan.resolve(strict=True)
    a.output = a.output.absolute()
    if a.block_case:
        block(a.plan, a.block_case, a.block_number, a.output, a.samples)
        return
    if a.blocks < 3 or a.samples < 3:
        p.error('at least three alternating blocks and three samples required')
    plan = json.loads(a.plan.read_text())
    names = {c['metric'] for c in plan['cases']}
    requested = set(a.metrics.split(',')) if a.metrics else names
    if not requested <= names:
        p.error('metrics absent from frozen plan: '+','.join(sorted(requested-names)))
    a.output.mkdir()
    (a.output/'plan.json').write_bytes(a.plan.read_bytes())
    rows = []
    for case in plan['cases']:
        if case['metric'] not in requested:
            continue
        case_dir = a.output/case['id']
        case_dir.mkdir()
        blocks = []
        for number in range(a.blocks):
            out = case_dir/f'block-{number:02d}'
            command = [sys.executable, str(Path(__file__).resolve()), str(a.plan), str(out),
                       '--block-case', case['id'], '--block-number', str(number), '--samples', str(a.samples)]
            wrapped = ['flock', str(Path.home()/'tmp/lilith-heavy.lock'), 'flock', str(Path.home()/'tmp/lilith-wsl-heavy.lock'),
                       str(Path.home()/'work/claudehints/scripts/run-heavy'), '--mem', '12G', '--jobs', str(case['threads']), '--', *command]
            with (case_dir/f'block-{number:02d}-wrapper.log').open('x') as log:
                r = subprocess.run(wrapped, stdout=log, stderr=subprocess.STDOUT)
            if r.returncode or not (out/'result.json').exists():
                blocks.append(dict(passed=False, errors={'block': 'failed; inspect wrapper log'}))
            else:
                blocks.append(json.loads((out/'result.json').read_text()))
        ratio = summarize(blocks, case['matched_scope'])
        row = dict(metric=case['metric'], case=case['id'], width=case['width'], height=case['height'],
                   threads=case['threads'], canonical=case['canonical']['label'],
                   passed=(False if any(b.get('errors') or b['passed'] is False for b in blocks)
                           else None if any(b['passed'] is None for b in blocks) else True), absolute_bound=case['absolute_bound'],
                   max_abs_delta=max((b['delta'] for b in blocks if 'delta' in b), default=None),
                   ratio=ratio, matched_scope=case['matched_scope'], scope=case['scope'],
                   qualification='descriptive; no confidence interval or regression acceptance',
                   rust_source=plan['rust_source'], plan_sha256=sha(a.plan))
        (case_dir/'summary.json').write_text(json.dumps(row, indent=2)+'\n')
        rows.append(row)
        with (a.output/'summary.tsv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(row), delimiter='\t', lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
        print('CASE_DONE', len(rows), json.dumps(row), flush=True)
    if any(row['passed'] is False for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
