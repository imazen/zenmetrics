#!/usr/bin/env python3
"""Export completed fresh-worker profiles separately from warm timing evidence."""
import argparse
import json
from pathlib import Path
import re

from compare_canonical import sha
from native_classical_first_cell import parse_native
from reference_publication import register_outputs


def publish(index, output):
    definition = json.loads(index.read_text())
    rows = []
    for job in definition['profiles']:
        root = Path(job['output']).resolve(strict=True)
        receipt_path = root / 'receipt.json'
        receipt = json.loads(receipt_path.read_text())
        request_path = root / 'request.json'
        request = json.loads(request_path.read_text())
        plan_path = Path(request['plan'])
        if sha(plan_path) != receipt['plan_sha256'] or request['plan_sha256'] != receipt['plan_sha256']:
            raise ValueError('first-call plan identity differs')
        if request['case'] != receipt['case'] or request['repeats'] != receipt['repetitions']:
            raise ValueError('capture request differs from completed receipt')
        plan = json.loads(plan_path.read_text())
        case = next(c for c in plan['cases'] if c['id'] == receipt['case'])
        if (plan['phase'] != 'scorer_construct_first_eval' or case['metric'].endswith('-gpu')
                or case['first_call_scope'].get('matched') is not True):
            raise ValueError('completed CPU first-call capture required')
        for path, expected in plan['bindings'].items():
            if sha(path) != expected:
                raise ValueError('first-call artifact changed: ' + path)
        if sha(Path(__file__).with_name('profile_classical_first.py')) != receipt['controller_sha256']:
            raise ValueError('capture helper differs; preserve its exact source before exporting')
        capture = Path(receipt['perf_data']).parent
        config = capture / 'config.json'
        if sha(receipt['perf_data']) != receipt['perf_data_sha256'] or sha(config) != receipt['config_sha256']:
            raise ValueError('recording/configuration changed')
        if json.loads(config.read_text()) != json.loads((root / 'config.json').read_text()):
            raise ValueError('adopted configuration differs')
        logs = sorted(capture.glob('worker-*.log'))
        if {p.name: sha(p) for p in logs} != receipt['worker_log_sha256']:
            raise ValueError('completed worker logs differ')
        observed = [parse_native(p.read_text(), case) for p in logs]
        if len(observed) != receipt['repetitions'] or sorted({r['score_bits'] for r in observed}) != receipt['score_bits']:
            raise ValueError('completed first-call score/repetition checks differ')
        symbols = []
        for line in (root / 'report.txt').read_text().splitlines():
            found = re.match(r'^\s+([\d.]+)%\s+(\S+)\s+(\S+)\s+\[([.k])\]\s+(.*)$', line)
            if found:
                symbols.append(dict(self_percent=float(found[1]), process=found[2], object=found[3], domain=found[4], symbol=found[5]))
        resources = []
        for role, path in job['resource_logs'].items():
            path = Path(path)
            lines = [s for s in path.read_text().splitlines() if s.startswith('run-heavy: done ')]
            if not lines:
                raise ValueError('resource receipt missing')
            resources.append(dict(role=role, path=str(path), sha256=sha(path), receipts=lines))
        if not any('rc=0 ' in line for r in resources for line in r['receipts']):
            raise ValueError('successful capture or adoption receipt required')
        rows.append(dict(case=case['id'], metric=case['metric'], width=case['width'], height=case['height'], threads=case['threads'], rust_source=case['rust_source'], native_invocation_sha256=plan['bindings'][case['binary']], raw_directory=str(root), receipt=receipt, receipt_sha256=sha(receipt_path), request_sha256=sha(request_path), report_sha256=sha(root/'report.txt'), script_sha256=sha(root/'script.txt'), top_self_symbols=symbols[:8], resource_receipts=resources, source_note=job.get('source_note')))
    if not rows or len({r['case'] for r in rows}) != len(rows):
        raise ValueError('distinct completed profiles required')
    payloads = [(f'profile-{n:02d}.json', json.dumps(row, indent=2, allow_nan=False)+'\n') for n, row in enumerate(rows)]
    if any(len(text.encode()) > 30000 for _, text in payloads):
        raise ValueError('first-call profile projection exceeds generated-data cap')
    output.mkdir()
    artifacts, shards = [], []
    for (name, text), row in zip(payloads, rows):
        path = output/name
        path.write_text(text)
        artifacts.append((path, 'provenance'))
        shards.append(dict(file=name, case=row['case'], sha256=sha(path)))
    path = output/'index.json'
    path.write_text(json.dumps(dict(source_index=str(index), source_index_sha256=sha(index), exporter_sha256=sha(__file__), profiles=shards, qualification='Whole fresh CPU worker processes, including startup, RGB file loading and teardown. Each worker performs only its first scorer evaluation; all completed scalar observations are revalidated. CPU cycle samples do not isolate the internal clock or worker-thread spans, measure GPU kernels, establish a causal share of a timing gap, or qualify a regression. Instrumented durations are not comparison evidence.'), indent=2)+'\n')
    artifacts.append((path, 'provenance'))
    register_outputs(artifacts)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(dict(profiles=publish(args.index, args.output), output=str(args.output))))


if __name__ == '__main__':
    main()
