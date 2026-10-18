#!/usr/bin/env python3
"""Run every declared CLI name on the complexity fixture, retaining refusals.

Pass --lock to acquire the machine heavy lock separately for each attempt,
or hold it in the caller. Each attempt gets its own hard resource boundary,
log and timeout. This is an execution inventory, not a timing result.
Existing rows are never overwritten. PieAPP is excluded by owner direction.
"""
import argparse
import csv
import os
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bin', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--logs', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=120)
    p.add_argument('--lock', type=Path, help='shared heavy lock, acquired separately per attempt')
    p.add_argument('--metrics', default='')
    p.add_argument('--shapes', default='64x64,1024x1024,3840x2160')
    p.add_argument('--threads', default='1,8')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = (root / 'crates/zenmetrics-cli/src/metrics/mod.rs').read_text()
    enum = source.split('pub enum MetricKind {', 1)[1].split('\n}', 1)[0]
    names = a.metrics.split(',') if a.metrics else re.findall(r'#\[value\(name = "([^"]+)"\)\]', enum)
    a.logs.mkdir(parents=True, exist_ok=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.out.exists():
        with a.out.open() as f:
            done = {(r['metric'], r['shape'], r['threads']) for r in csv.DictReader(f, delimiter='\t')}
    exists = a.out.exists()
    with a.out.open('a') as f:
        w = csv.writer(f, delimiter='\t')
        if not exists:
            w.writerow(['metric', 'shape', 'threads', 'status', 'exit_code', 'detail', 'log'])
        for shape in a.shapes.split(','):
            for threads in map(int, a.threads.split(',')):
                cpus = ','.join(str(i) for i in range(2, 2+threads))
                for name in names:
                    if (name, shape, str(threads)) in done:
                        continue
                    log = a.logs / f'{name}-{shape}-{threads}.log'
                    attempt = 1
                    while log.exists():
                        attempt += 1
                        log = a.logs / f'{name}-{shape}-{threads}.attempt{attempt}.log'
                    status, code, detail = 'skipped by owner', '', 'PieAPP excluded from all timing, fix and tally work'
                    if name not in ('pieapp', 'pieapp-pyiqa'):
                        env = dict(os.environ, TMPDIR=str(Path.home() / 'tmp'), RAYON_NUM_THREADS=str(threads),
                                   OMP_NUM_THREADS=str(threads), RTEN_NUM_THREADS=str(threads))
                        cmd = [str(Path.home() / 'work/claudehints/scripts/run-heavy'), '--mem', '24G', '--jobs', str(threads),
                               '--', 'timeout', '--kill-after=5s', str(a.timeout), 'taskset', '-c', cpus,
                               a.bin, shape, name, '1']
                        if a.lock:
                            cmd = ['flock', str(a.lock)] + cmd
                        with log.open('x') as output:
                            code = subprocess.run(cmd, env=env, stdout=output, stderr=subprocess.STDOUT).returncode
                        text = log.read_text()
                        lines = text.splitlines()
                        detail = next((line for line in lines if line.startswith('REFUSED ')), '')
                        score = next((line for line in lines if line.startswith(name + ' ') and 'score=' in line), '')
                        status = 'complete' if code == 0 and score else 'refused'
                        if code == 124:
                            status = 'timeout'
                        elif code == 137:
                            status = 'resource termination'
                        elif code == 101:
                            status = 'panic'
                        if score and any(x in score.lower() for x in ('nan', 'inf')):
                            status = 'nonfinite'
                        detail = detail or score or next((line for line in lines if 'error' in line.lower()), f'exit {code}; see log')
                    w.writerow([name, shape, threads, status, code, detail[:160], log.name if code != '' else ''])
                    f.flush()
                    print(name, shape, threads, status, flush=True)


if __name__ == '__main__':
    main()
