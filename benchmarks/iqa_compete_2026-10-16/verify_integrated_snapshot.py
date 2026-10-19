#!/usr/bin/env python3
"""Verify one immutable combined tree using CI-pinned siblings, under caller locks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot', type=Path)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(['jj', 'log', '--no-graph', '-r', '@', '-T', 'commit_id'], cwd=repo, text=True).strip()
    output = a.snapshot.resolve()
    assert not output.exists(), output
    output.mkdir(parents=True)
    env = dict(os.environ, LOCK_SNAP_DIR=str(output), TMPDIR=str(output),
               CARGO_TARGET_DIR=str(output / 'target'), CARGO_PROFILE_RELEASE_LTO='thin')
    for key in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_RUSTFLAGS']:
        env.pop(key, None)
    commands = []
    receipt = dict(source=revision, commands=commands, outcome='running', native_cpu_flags=False)
    def run(command, cwd):
        record = dict(command=command, cwd=str(cwd), started=time.time())
        commands.append(record)
        print('CHECK', json.dumps(command), flush=True)
        result = subprocess.run(command, cwd=cwd, env=env)
        record.update(returncode=result.returncode, finished=time.time())
        (output / 'verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
        result.check_returncode()
    try:
        run(['bash', str(repo / 'scripts/ci/lock.sh'), '--check', '--rev', revision], repo)
        tree = output / 'work/zenmetrics'
        run(['python3', '-B', '-m', 'unittest', 'discover', '-s', 'benchmarks/clean_panel_2026-10-19', '-p', 'test_*.py'], tree)
        packages = ['vif', 'msssim', 'fsim', 'brisque', 'niqe', 'piqe', 'zenmetrics-cli']
        run(['cargo', 'fmt', *[x for name in packages for x in ['-p', name]], '--', '--check'], tree)
        metric_packages = [x for name in packages[:-1] for x in ['-p', name]]
        run(['cargo', 'test', '--locked', *metric_packages, '--all-features'], tree)
        common = ['--locked', '-p', 'zenmetrics-cli', '--no-default-features', '--features', 'cpu-metrics,png,jpeg,bmp']
        run(['cargo', 'test', *common, '--all-targets'], tree)
        run(['cargo', 'test', *common, '--doc'], tree)
        run(['cargo', 'clippy', *common, '--all-targets', '--', '-D', 'warnings'], tree)
        run(['cargo', 'build', '--locked', '--release', '-p', 'cpu-profile', '--features', 'compete,metrics-parallel', '--bin', 'iqa-compete'], tree)
        binary = output / 'target/release/iqa-compete'
        receipt['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
        run([str(binary), 'inventory'], tree)
        run(['python3', '-B', 'benchmarks/iqa_compete_2026-10-16/checkpoint_math_observe.py'], tree)
        receipt['outcome'] = 'pass'
    finally:
        (output / 'verification.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
