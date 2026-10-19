#!/usr/bin/env python3
"""Profile completed >1.08 exploratory cells using their exact Rust configs.

Each native profile and report is serialized in one contained scope. Completed
profiles are immutable and hash-checked on later catch-up invocations.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(plan_path):
    plan = json.loads(plan_path.read_text())
    root = plan_path.parent
    env = dict(os.environ, TMPDIR=str(root), ZENBENCH_CPU_POLICY='unrestricted', ZENBENCH_NO_SAVE='1')
    for key in ['RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        env[key] = str(plan['threads'])
    command = ['taskset', '-c', '0' if plan['threads'] == 1 else '0-' + str(plan['threads'] - 1),
               plan['perf'], 'record', '-e', 'cpu-clock', '-F', '999', '-g', '--call-graph', 'dwarf',
               '-o', str(root / 'perf.data'), '--', plan['binary'], str(root / 'config.json')]
    print('COMMAND', json.dumps(command), flush=True)
    subprocess.run(command, env=env, check=True)
    command = [plan['perf'], 'report', '--stdio', '--no-children', '--sort', 'symbol', '--percent-limit', '0.5',
               '-i', str(root / 'perf.data')]
    print('COMMAND', json.dumps(command), flush=True)
    with (root / 'report.txt').open('x') as f:
        subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path)
    p.add_argument('--panel', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--adopt', action='append', default=[])
    a = p.parse_args()
    if a.worker:
        return worker(a.worker.resolve(strict=True))
    assert a.panel and a.output
    panel = a.panel.resolve(strict=True)
    adopted = dict(item.split('=', 1) for item in a.adopt)
    perf = Path(shutil.which('perf')).resolve(strict=True)
    script = Path(__file__).resolve()
    identity = dict(panel=str(panel), worker_sha256=sha(script), perf=str(perf), perf_sha256=sha(perf),
                    event='cpu-clock', frequency=999, profile_scope='whole native Rust benchmark process, including warmup and harness',
                    perf_version=subprocess.check_output([str(perf), '--version'], text=True).strip())
    if a.output.exists():
        assert json.loads((a.output / 'identity.json').read_text()) == identity
    else:
        a.output.mkdir()
        (a.output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    root = a.output.resolve()
    records = []
    for path in sorted(panel.glob('*/outcome.json')):
        outcome = json.loads(path.read_text())
        if 'row' not in outcome or outcome['row']['ratio'] <= 1.08:
            continue
        case = outcome['case']
        directory = root / case
        record_path = directory / 'profile.json'
        if record_path.exists():
            record = json.loads(record_path.read_text())
            assert record['outcome_sha256'] == sha(path)
            for filename, digest in record['files'].items():
                assert sha(Path(filename)) == digest
            records.append(record)
            continue
        assert not directory.exists(), 'incomplete profile retained; inspect before a separate retry'
        packet_path = path.parent / 'attempt/packet.json.gz'
        packet = json.loads(gzip.decompress(packet_path.read_bytes()))
        invocation = next(row for row in packet['invocations'] if row['arm'] == 1 and row['status'] == 'completed')
        config = Path(invocation['output']) / 'config.json'
        binary = Path(packet['plan']['cases'][0]['inputs']['binary'])
        directory.mkdir()
        shutil.copyfile(config, directory / 'config.json')
        plan = dict(binary=str(binary), binary_sha256=sha(binary), config_sha256=sha(config),
                    threads=outcome['row']['threads'], perf=str(perf))
        (directory / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        if case in adopted:
            prior = Path(adopted[case]).resolve(strict=True)
            assert sha(prior / 'config.json') == sha(config)
            files = [prior / 'perf.data', prior / 'report.txt', prior / 'config.json', prior.with_suffix('.log')]
            returncode = 0
        else:
            command = ['flock', str(Path.home() / 'tmp/lilith-heavy.lock'), 'flock', str(Path.home() / 'tmp/lilith-wsl-heavy.lock'),
                       str(Path.home() / 'work/claudehints/scripts/run-heavy'), '--mem', '12G', '--jobs', str(plan['threads']),
                       '--', sys.executable, '-B', str(script), '--worker', str(directory / 'plan.json')]
            with (directory / 'wrapper.log').open('x') as f:
                result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
            returncode = result.returncode
            files = [directory / name for name in ['perf.data', 'report.txt', 'config.json', 'plan.json', 'wrapper.log']]
        record = dict(case=case, outcome_sha256=sha(path), packet_sha256=sha(packet_path),
                      plan=plan, returncode=returncode, adopted=case in adopted,
                      files={str(p): sha(p) for p in files if p.exists()})
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
        print('PROFILE_DONE', case, returncode, flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    (root / ('index-' + stamp + '.json')).write_text(json.dumps(dict(profiles=records), indent=2) + '\n')
    if any(r['returncode'] != 0 for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
