#!/usr/bin/env python3
"""Prepare fresh HDR cell plans and retain four actual protocol-control envelopes.

Caller supplies shared heavy admission. The original scorer is invoked through
the frozen resident-cell adapter; no metric arithmetic is implemented here.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from compare_canonical import sha


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    request = json.loads(manifest_bytes)
    for name in ('cohort', 'seal', 'cell', 'batch_receipt'):
        record = request[name]
        if sha(record['path']) != record['sha256']:
            raise ValueError('bound control input changed: ' + name)
    cohort_path = Path(request['cohort']['path']).resolve(strict=True)
    batch_root = cohort_path.parent
    worker_path = batch_root / 'helpers/first_boundary_hdrvdp_arm_worker.py'
    cohort = json.loads(cohort_path.read_text())
    if sha(worker_path) != cohort['bindings'][str(worker_path)]:
        raise ValueError('frozen worker differs')
    sys.path.insert(0, str(worker_path.parent))
    worker = module('hdr_cell_control_worker', worker_path)
    checked, parent = worker.context(batch_root, request['cohort']['sha256'])
    worker.checked_seal(batch_root, checked, request['cohort']['sha256'], request['seal']['sha256'])
    cases = worker.selected_cases(checked, parent)
    if len(cases) != 4:
        raise ValueError('four completed headless warm cases required')
    references = {}
    for case in cases:
        value = json.loads((batch_root / 'cases' / (case['id'] + '.json')).read_text())
        if value['passed'] is not True:
            raise ValueError('failed original/resident score control')
        references[(case['metric'], case['threads'])] = value['observations'][0]['score_bits']
    args.output.mkdir()
    root = args.output.resolve()
    (root / 'envelopes').mkdir()
    (root / 'controls').mkdir()
    shutil.copy2(__file__, root / 'controller.py')
    write(root / 'input-manifest.json', request)
    cell = module('hdr_protocol_cell', Path(request['cell']['path']))
    cell.prepare(Path(cohort['parent_plan']), cohort['parent_plan_sha256'],
                 Path(request['seal']['path']), request['seal']['sha256'], root / 'candidate')
    index = json.loads((root / 'candidate/index.json').read_text())
    records = []
    failed = False
    for path in index['metric_plans'].values():
        plan = json.loads(Path(path).read_text())
        for row in plan['cases']:
            control_root = root / 'controls' / row['id']
            control_root.mkdir()
            scratch = control_root / 'cell'
            scratch.mkdir()
            command = [arg.format_map(dict(row, samples=1, output=str(control_root)))
                       for arg in row['canonical']['command']]
            write(control_root / 'command.json', command)
            env = dict(os.environ, TMPDIR=str(scratch), LC_ALL='C')
            for name in worker.THREAD_ENV:
                env[name] = str(row['threads'])
            stdout, stderr = control_root / 'stdout.log', control_root / 'stderr.log'
            with stdout.open('xb') as out, stderr.open('xb') as err:
                code = subprocess.run(command, env=env, stdout=out, stderr=err).returncode
            passed = code == 0
            error = None
            if passed:
                try:
                    passed = worker.scalar_bits(stdout.read_text(), 2, True) == references[(row['metric'], row['threads'])]
                except ValueError as exception:
                    passed = False
                    error = str(exception)
            envelope = dict(schema=1, kind='hdr-resident-cell-protocol', case_id=row['id'],
                metric=row['metric'], threads=row['threads'], samples=1, returncode=code,
                command=command, root=str(control_root), cell_root=str(scratch),
                stdout=str(stdout), stderr=str(stderr), passed=passed, error=error,
                artifacts=worker.files_under(control_root))
            destination = root / 'envelopes' / (row['id'] + '.json')
            write(destination, envelope)
            records.append(dict(path=str(destination), sha256=sha(destination)))
            failed |= not passed
            print('HDR_CELL_PROTOCOL ' + json.dumps(dict(case=row['id'], passed=passed, returncode=code)), flush=True)
    result = dict(request)
    result.pop('cell')
    result.update(schema=1, candidate_plans=[dict(path=p, sha256=sha(p)) for p in index['metric_plans'].values()],
                  cell_controls=records)
    for record in result['native_controls']:
        record['artifacts'] = worker.files_under(Path(record['root']))
    write(root / 'promotion-request.json', result)
    write(root / 'result.json', dict(passed=not failed, controls=len(records),
        matched_scope=False, qualified_timing_cells=0,
        request_sha256=sha(root / 'promotion-request.json'), controller_sha256=sha(root / 'controller.py')))
    if args.manifest.read_bytes() != manifest_bytes:
        raise ValueError('input control manifest changed')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
