#!/usr/bin/env python3
"""Retain completed observations from a size-capped recording; run no scorer."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from compare_canonical import sha
from native_classical_first_cell import parse_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('case')
    parser.add_argument('capture', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    case = next(c for c in plan['cases'] if c['id'] == args.case)
    request_path = args.capture/'request.json'
    request = json.loads(request_path.read_text())
    if (plan['phase'] != 'scorer_construct_first_eval'
            or case['metric'].endswith('-gpu')
            or case['first_call_scope'].get('matched') is not True
            or request['case'] != args.case
            or request['plan_sha256'] != sha(args.plan)
            or request['binary'] != case['binary']
            or request['threads'] != case['threads']):
        raise ValueError('matched CPU first-call capture identity required')
    expected_config = dict(width=case['width'], height=case['height'],
        reference=case['reference'], distorted=case['distorted'],
        metric_name=case['metric'], threads=case['threads'], only_mode='score',
        timing_phase='cold', probe_only=False, required_samples=1,
        locked_process_accounting=True)
    config_path = args.capture/'config.json'
    if json.loads(config_path.read_text()) != expected_config:
        raise ValueError('capture configuration differs from frozen case')
    record_log = args.capture/'record.log'
    if 'perf size limit reached' not in record_log.read_text():
        raise ValueError('explicit perf size-limit stop required')
    bindings = dict(plan['bindings'])
    for p in [args.plan, request_path, config_path, record_log,
              args.capture/'perf.data', Path(__file__),
              Path(__file__).with_name('native_classical_first_cell.py')]:
        absolute, observed = str(p.resolve()), sha(p)
        if absolute in bindings and bindings[absolute] != observed:
            raise ValueError('original artifact binding changed: '+absolute)
        bindings[absolute] = observed
    logs = sorted(args.capture.glob('worker-*.log'))
    if not logs or [p.name for p in logs] != [f'worker-{n:03d}.log' for n in range(len(logs))]:
        raise ValueError('contiguous retained worker logs required')
    complete, incomplete = [], []
    for number, path in enumerate(logs):
        text = path.read_text()
        bindings[str(path.resolve())] = sha(path)
        if 'IQA_ROW ' not in text:
            if number != len(logs)-1:
                raise ValueError('incomplete worker before final size-limit stop')
            incomplete.append(path.name)
        else:
            complete.append(dict(log=path.name, score_bits=parse_native(text, case)['score_bits']))
    if not complete or not len(complete) < request['repeats']:
        raise ValueError('nonempty incomplete recording prefix required')

    def verify():
        for path, expected in bindings.items():
            if sha(path) != expected:
                raise ValueError('retained artifact changed: '+path)

    verify()
    if shutil.disk_usage(args.output.parent).free < 1500*1024*1024:
        raise ValueError('1500 MiB free disk required before postprocessing')
    args.output.mkdir()
    perf = str(Path(shutil.which('perf')).resolve())
    commands = dict(script=[perf, 'script', '-F', 'comm,pid,tid,time,event,ip,sym,dso'],
                    report=[perf, 'report', '--stdio', '--no-children'],
                    header=[perf, 'report', '--header-only', '--stdio'])
    for name, prefix in commands.items():
        with (args.output/(name+'.txt')).open('x') as out, (args.output/(name+'-stderr.log')).open('x') as err:
            subprocess.run([*prefix, '-i', str(args.capture/'perf.data')],
                           stdout=out, stderr=err, check=True)
    verify()
    result = dict(case=args.case, partial_capture=True, requested_repetitions=request['repeats'],
        completed_repetitions=len(complete), completed=complete, incomplete_worker_logs=incomplete,
        native_metric_calls=0, capture=str(args.capture.resolve()),
        plan_sha256=sha(args.plan), request_sha256=sha(request_path),
        config_sha256=sha(config_path), perf_data_sha256=sha(args.capture/'perf.data'),
        worker_log_sha256={p.name: sha(p) for p in logs},
        postprocessor_sha256=sha(__file__), perf_tool_sha256=sha(perf),
        output_sha256={name+'.txt': sha(args.output/(name+'.txt')) for name in commands},
        qualification='Size-capped whole-process diagnostic, with every completed scalar revalidated and any interrupted final worker retained explicitly. No scorer rerun, complete requested capture, isolated cold clock, timing ratio, numerical parity acceptance or qualified regression is claimed.')
    with (args.output/'receipt.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(case=args.case, partial=True, completed=len(complete), requested=request['repeats'])))


if __name__ == '__main__':
    main()
