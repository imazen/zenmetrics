#!/usr/bin/env python3
"""Admit one breadth-first pass over existing frozen canonical benchmark plans.

Existing controllers and their acquired samples remain intact. Both shared
heavy locks stay held for this pass, with a separate resource scope per block.
Only one one-thread case is selected per uncovered metric name. The time budget
controls admission of another case; it never kills an acquired measurement.
"""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'benchmarks/iqa_compete_2026-10-16'))
from compare_canonical import sha, summarize


def covered_names(root):
    dispositions_path = root / 'contract-dispositions.json'
    dispositions = json.loads(dispositions_path.read_text()) if dispositions_path.exists() else {}
    covered = set()
    for path in root.glob('*-run/*/summary.json'):
        row = json.loads(path.read_text())
        if dispositions.get(row['plan_sha256'] + ':' + row['metric']):
            continue
        if row['passed'] is not False and row.get('max_abs_delta') is not None:
            covered.add(row['metric'])
    return covered


def first_cases(plan, covered):
    selected = {}
    for case in plan['cases']:
        if case['threads'] != 1 or case['metric'] in covered:
            continue
        selected.setdefault(case['metric'], case)
    return list(selected.values())


def priority(case):
    name = case['metric']
    # Cheap analytical rows first; large reference pipelines follow the first
    # pass through the existing SSIM and small classical adapters.
    rank = (3 if name.startswith('psnrhvs') else
            0 if name.startswith('psnr') or name == 'ciede2000' else
            1 if 'pyiqa' in name and 'ssim' in name else
            3 if name.startswith('hdrvdp') or name == 'mad' else 2)
    return rank, name


def write_summary(root, rows):
    if not rows:
        return
    with (root / 'summary.tsv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence_root', type=Path)
    p.add_argument('output_prefix', type=Path)
    p.add_argument('--plan', type=Path, action='append', required=True)
    p.add_argument('--seconds', type=int, default=900)
    p.add_argument('--samples', type=int, default=3)
    a = p.parse_args()
    if a.seconds <= 0 or a.samples < 3:
        p.error('positive budget and at least three samples required')
    a.evidence_root = a.evidence_root.resolve(strict=True)
    a.output_prefix = a.output_prefix.absolute()
    state_path = a.output_prefix.with_suffix('.json')
    if state_path.exists():
        p.error('new output prefix required')
    wrapper = Path.home() / 'work/claudehints/scripts/run-heavy'
    runner = HERE / 'benchmarks/iqa_compete_2026-10-16/compare_canonical.py'
    state = {'status': 'waiting-for-admission', 'completed': [], 'pending': [],
             'seconds': a.seconds, 'policy': 'first uncovered one-thread case per name'}
    state_path.write_text(json.dumps(state, indent=2) + '\n')
    with (Path.home() / 'tmp/lilith-heavy.lock').open('a+') as first:
        fcntl.flock(first, fcntl.LOCK_EX)
        with (Path.home() / 'tmp/lilith-wsl-heavy.lock').open('a+') as second:
            fcntl.flock(second, fcntl.LOCK_EX)
            covered = covered_names(a.evidence_root)
            queue = []
            rows_by_root = {}
            for index, path in enumerate(a.plan):
                original = json.loads(path.read_text())
                cases = first_cases(original, covered)
                covered.update(c['metric'] for c in cases)
                if not cases:
                    continue
                root = a.output_prefix.with_name(a.output_prefix.name + f'-{index}-run')
                root.mkdir()
                plan = dict(original, cases=cases)
                plan_path = root / 'plan.json'
                plan_path.write_text(json.dumps(plan, indent=2) + '\n')
                rows_by_root[root] = []
                queue.extend((case, plan_path, root) for case in cases)
            queue.sort(key=lambda job: priority(job[0]))
            state.update(status='running', admitted_unix=time.time(),
                         pending=[c['metric'] for c, _, _ in queue])
            state_path.write_text(json.dumps(state, indent=2) + '\n')
            started = time.monotonic()
            for case, plan_path, root in queue:
                if time.monotonic() - started >= a.seconds:
                    break
                directory = root / case['id']
                directory.mkdir()
                blocks = []
                for number in range(3):
                    output = directory / f'block-{number:02d}'
                    command = [str(wrapper), '--mem', '12G', '--jobs', '1', '--',
                               sys.executable, str(runner), str(plan_path), str(output),
                               '--block-case', case['id'], '--block-number', str(number),
                               '--samples', str(a.samples)]
                    with (directory / f'block-{number:02d}-wrapper.log').open('x') as log:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                        state.update(metric=case['metric'], block=number, child_pid=child.pid)
                        state_path.write_text(json.dumps(state, indent=2) + '\n')
                        rc = child.wait()
                    if rc or not (output / 'result.json').exists():
                        blocks.append(dict(passed=False, errors={'block': 'failed; inspect wrapper log'}))
                    else:
                        blocks.append(json.loads((output / 'result.json').read_text()))
                row = {key: case[key] for key in ['metric', 'width', 'height', 'threads',
                       'absolute_bound', 'matched_scope', 'scope']}
                row.update(case=case['id'], canonical=case['canonical']['label'],
                    passed=(False if any(b.get('errors') or b['passed'] is False for b in blocks)
                            else None if any(b['passed'] is None for b in blocks) else True),
                    max_abs_delta=max((b['delta'] for b in blocks if 'delta' in b), default=None),
                    ratio=summarize(blocks, case['matched_scope']),
                    qualification='descriptive; no confidence interval or regression acceptance',
                    rust_source=json.loads(plan_path.read_text())['rust_source'], plan_sha256=sha(plan_path))
                (directory / 'summary.json').write_text(json.dumps(row, indent=2) + '\n')
                rows_by_root[root].append(row)
                write_summary(root, rows_by_root[root])
                state['completed'].append(case['metric'])
                state['pending'].remove(case['metric'])
                state_path.write_text(json.dumps(state, indent=2) + '\n')
                print('COVERAGE_DONE ' + json.dumps(row), flush=True)
            state.update(status='budget-reached' if state['pending'] else 'complete',
                         elapsed_seconds=time.monotonic() - started)
            state_path.write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
