#!/usr/bin/env python3
"""Run four headless HDR score controls in one caller-admitted resource scope.

This successor retains the frozen per-arm cohort unchanged. Each evaluator is
still a fresh process, with the original per-arm source/runtime/scalar checks
and process memory receipt. It never qualifies comparative timings.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys

from compare_canonical import sha


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run(args):
    original = args.cohort.resolve(strict=True)
    worker_path = args.worker.resolve(strict=True)
    if sha(original) != args.cohort_sha or sha(worker_path) != args.worker_sha:
        raise ValueError('explicit frozen cohort/worker hashes differ')
    cohort = json.loads(original.read_text())
    if cohort['kind'] != 'photo512-warm-t1-t8':
        raise ValueError('only the four-case headless warm control cohort is supported')
    args.output.mkdir()
    root = args.output.resolve()
    for name in ('admission', 'arms', 'cases'):
        (root / name).mkdir()
    controller = root / 'controller.py'
    shutil.copy2(__file__, controller)
    source = Path(__file__).resolve()
    cohort = copy.deepcopy(cohort)
    cohort['bindings'].update({str(original): args.cohort_sha,
        str(worker_path): args.worker_sha, str(controller): sha(controller), str(source): sha(source)})
    helpers = root / 'helpers'
    helpers.mkdir()
    for path in worker_path.parent.glob('*.py'):
        destination = helpers / path.name
        shutil.copy2(path, destination)
        cohort['bindings'][str(destination)] = sha(destination)
    worker_path = helpers / worker_path.name
    if sha(worker_path) != args.worker_sha:
        raise ValueError('frozen worker copy changed')
    for field in ('case_specs', 'catalog'):
        destination = root / Path(cohort[field]).name
        shutil.copy2(cohort[field], destination)
        cohort[field] = str(destination)
        cohort['bindings'][str(destination)] = sha(destination)
    cohort['admission_granularity'] = 'one caller-admitted batch, four warm score-control cases'
    cohort['parent_cohort'] = str(original)
    cohort['parent_cohort_sha256'] = args.cohort_sha
    write(root / 'cohort.json', cohort)
    cohort_sha = sha(root / 'cohort.json')
    sys.path.insert(0, str(worker_path.parent))
    spec = importlib.util.spec_from_file_location('bound_hdr_worker', worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    checked, plan = worker.context(root, cohort_sha)
    cases = worker.selected_cases(checked, plan)
    if (len(cases) != 4 or {(c['metric'], c['threads'], c['phase']) for c in cases}
            != {(metric, threads, 'warm') for metric in ('hdrvdp', 'hdrvdp3') for threads in (1, 8)}):
        raise ValueError('both routes at one/eight threads required, warm phase only')
    worker.seal(root, cohort_sha, args.preflight_sha)
    seal_sha = sha(root / 'preflight-seal.json')
    failed = False
    audits = []
    for case in cases:
        for ordinal in range(4):
            code = worker.arm(root, cohort_sha, seal_sha, case['id'], ordinal)
            if code not in (0, 1):
                raise ValueError('source/runtime verification failure; further scoring refused')
            failed |= code != 0
        directory = root / 'admission' / case['id']
        directory.mkdir()
        audit = directory / 'audit-result.json'
        code = worker.audit(root, cohort_sha, seal_sha, case['id'], 3, audit)
        failed |= code != 0
        audits.append(dict(case=case['id'], audit=str(audit), sha256=sha(audit), passed=code == 0))
    worker.context(root, cohort_sha)
    worker.checked_seal(root, checked, cohort_sha, seal_sha)
    record = dict(cases=audits, passed=not failed, matched_scope=False, qualified_timing_cells=0,
        cohort_sha256=cohort_sha, preflight_seal_sha256=seal_sha,
        controller_sha256=sha(controller), worker_sha256=args.worker_sha,
        admission='Caller holds both host locks and run-heavy 12G/8 jobs; each child uses its explicit case thread budget',
        qualification='Exact default/resident score controls only; native and cell-protocol comparisons remain separate')
    write(root / 'result.json', record)
    print(json.dumps(record), flush=True)
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'worker', 'output'):
        parser.add_argument(name, type=Path)
    for name in ('cohort-sha', 'worker-sha', 'preflight-sha'):
        parser.add_argument('--' + name, required=True)
    raise SystemExit(run(parser.parse_args()))


if __name__ == '__main__':
    main()
