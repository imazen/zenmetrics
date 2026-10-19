#!/usr/bin/env python3
"""One additional unchanged-config scope for flagged portable CPU rows.

Caller holds the host's shared benchmark lock. Originals are read-only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def retry(parent, output, wrapper):
    meta_raw = (parent / 'run.meta').read_bytes()
    meta = json.loads(meta_raw)
    rows_raw = (parent / 'rows.jsonl').read_bytes()
    rows = [json.loads(line) for line in rows_raw.splitlines()]
    selected = [r for r in rows if r['status'] == 'ok' and
                any(not s['gate_clean'] for s in r.get('zenbench_samples', []))]
    if not selected:
        raise ValueError('no flagged successful rows to remeasure')
    binary = Path(meta['command'][0])
    worker = binary.with_name('iqa-compete')
    if sha(worker.read_bytes()) != meta['binary_file_sha256']:
        raise ValueError('parent worker binary changed')
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for row in selected:
        matches = []
        for path in parent.glob(f"{row['metric_name']}-{row['size_name']}-{row['threads']}-*.json"):
            raw = path.read_bytes()
            config = json.loads(raw)
            log = path.with_suffix('.log')
            logged = [json.loads(line[8:]) for line in log.read_text().splitlines() if line.startswith('IQA_ROW ')]
            if row in logged:
                matches.append((raw, config, sha(log.read_bytes())))
        if len(matches) != 1:
            raise ValueError('parent row/config/log binding is ambiguous')
        raw, config, log_sha = matches[0]
        if config.get('only_mode') != 'score' or config['impl_binary_sha'] != meta['binary_file_sha256']:
            raise ValueError('not an unchanged scalar parent scope')
        token = sha(json.dumps([row['metric_name'], row['size_name'], row['threads']]).encode())[:16]
        directory = output / token
        directory.mkdir()
        config_path = directory / f"{row['metric_name']}-{row['size_name']}-{row['threads']}-0.json"
        with config_path.open('xb') as stream:
            stream.write(raw)
        cases.append(dict(directory=str(directory), config=str(config_path), config_sha256=sha(raw),
                          parent_log_sha256=log_sha, parent_row=row, new_attempt_cap=1))
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), parent=str(parent),
                parent_rows_sha256=sha(rows_raw), parent_meta_sha256=sha(meta_raw),
                runner_sha256=sha(Path(__file__).read_bytes()), coordinator_sha256=sha(binary.read_bytes()),
                worker_sha256=meta['binary_file_sha256'], quiet_settle_seconds=5, cases=cases)
    write(output / 'manifest.json', plan)
    frozen = sha((output / 'manifest.json').read_bytes())
    for case in cases:
        if sha((output / 'manifest.json').read_bytes()) != frozen or (parent / 'rows.jsonl').read_bytes() != rows_raw:
            raise ValueError('retry manifest or parent evidence changed')
        config_path = Path(case['config'])
        raw = config_path.read_bytes()
        if sha(raw) != case['config_sha256']:
            raise ValueError('retry config changed')
        config = json.loads(raw)
        for role in ('reference', 'distorted'):
            if sha(Path(config[role]).read_bytes()) != config['input_sha256'][role]:
                raise ValueError('parent input changed')
        time.sleep(5)  # Admission cadence only; outside all metric clocks.
        memory = '12G' if sys.platform == 'darwin' else '16G'
        jobs = max(meta['hardware']['threads'] - 4, 1)
        command = [str(wrapper), '--mem', memory, '--jobs', str(jobs), '--', str(binary),
                   '--child', str(config['threads']), str(config_path)]
        directory = Path(case['directory'])
        log_path = config_path.with_suffix('.log')
        with log_path.open('xb') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        logged = [json.loads(line[8:]) for line in log_path.read_text().splitlines() if line.startswith('IQA_ROW ')]
        if len(logged) > 1 or any(r['mode'] != 'score' for r in logged):
            raise ValueError('unexpected retry row modes/count')
        with (directory / 'rows.jsonl').open('x') as stream:
            for row in logged:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        write(directory / 'run.meta', dict(build_commit=meta['build_commit'], host=meta['host'],
            binary_file_sha256=meta['binary_file_sha256'], command=command,
            parent_rows_sha256=sha(rows_raw), parent_log_sha256=case['parent_log_sha256'],
            config_sha256=case['config_sha256'], child_returncode=result.returncode,
            maximum_new_attempts=1, parent_score_bits=case['parent_row'].get('score_bits'),
            observed_score_bits=[r.get('score_bits') for r in logged]))
        print(json.dumps(dict(case=directory.name, returncode=result.returncode, rows=len(logged),
                             clean=bool(logged) and all(s['gate_clean'] for r in logged for s in r.get('zenbench_samples', [])))), flush=True)
    print('RETRY_SCOPES_FINISHED', len(cases), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('wrapper', type=Path)
    args = parser.parse_args()
    retry(args.parent, args.output, args.wrapper)
