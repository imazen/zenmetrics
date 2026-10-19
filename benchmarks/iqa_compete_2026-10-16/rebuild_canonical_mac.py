#!/usr/bin/env python3
"""Build a corrected immutable Mac worker, require strict controls, then install."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_coverage import freeze
from prepare_durable import sha

HERE = Path(__file__).resolve().parent


def checked_control(report):
    if len(report['cells']) != 4 or not report['complete']:
        raise ValueError('four complete native controls required')
    for cell in report['cells']:
        row = cell.get('row') or {}
        if cell['status'] != 'ok':
            raise ValueError('native control refused')
        if cell['config']['timing_phase'] == 'warm':
            samples = row.get('zenbench_samples', [])
            if row.get('n') != 10 or len(samples) != 10 or not all(s['gate_clean'] for s in samples):
                raise ValueError('native control has flagged/incomplete warm rounds')
    for threads in (1, 8):
        if len({c['row']['score_bits'] for c in report['cells'] if c['config']['threads'] == threads}) != 1:
            raise ValueError('cold/warm native score bits differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--stop-service', required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    home = Path.home()
    previous = home / 'tmp/iqa-changed-20261019T1620Z'
    snapshot = a.output / 'snapshot'
    wrapper = home / 'work/claudehints/scripts/run-heavy'
    env = dict(os.environ, PATH=str(home / '.cargo/bin') + ':/opt/homebrew/bin:' + os.environ['PATH'],
               TMPDIR=str(home / 'tmp'), CARGO_TARGET_DIR=str(snapshot / 'target'),
               RUSTFLAGS='--cfg iqa_compete_dvifmish')
    def run(argv, **kwargs):
        return subprocess.run(list(map(str, argv)), check=True, env=env, **kwargs)
    with (home / 'tmp/mac-heavy.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # No measurement holding the shared lock remains. Stop only this lane's
        # old ordinary-CPU coordinator, leaving every frozen artifact intact.
        run(['launchctl', 'disable', a.stop_service])
        run(['launchctl', 'bootout', a.stop_service])
        (a.output / 'boundary.json').write_text(json.dumps(dict(
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            shared_lock=str(home / 'tmp/mac-heavy.lock'), stopped_service=a.stop_service,
            retained_previous_bundle='iqa-mac-20261019-canonical-all'), indent=2) + '\n')
        (snapshot / 'work').mkdir(parents=True)
        for source in (previous / 'snapshot').iterdir():
            if source.name not in ('work', 'target') and source.is_dir():
                (snapshot / source.name).symlink_to(source, target_is_directory=True)
        for source in (previous / 'snapshot/work').iterdir():
            if source.name == 'zenmetrics':
                shutil.copytree(source, snapshot / 'work/zenmetrics')
            else:
                (snapshot / 'work' / source.name).symlink_to(source, target_is_directory=source.is_dir())
        tree = snapshot / 'work/zenmetrics'
        original = home / 'work/zen/zenmetrics'
        run([sys.executable, HERE / 'patch_dvifmish.py', tree, original, a.output / 'dvifmish-patch-v2.json'])
        run([sys.executable, HERE / 'patch_mac_cpu_counters.py', tree, original, snapshot / 'mac-cpu-counters.meta.json'])
        command = [wrapper, '--mem', '12G', '--jobs', '8', '--', 'cargo', 'build', '--locked', '--release',
                   '-p', 'cpu-profile', '--features', 'compete,metrics-parallel,zenmetrics-cli/nr,compete-dvifmish', '--bin', 'iqa-compete']
        (a.output / 'build-command.json').write_text(json.dumps(command, default=str, indent=2) + '\n')
        with (a.output / 'build.log').open('x') as log:
            run(command, cwd=tree, stdout=log, stderr=subprocess.STDOUT)
        portable = next((previous / 'results-canonical').glob('*/run.meta')).parent
        complete = a.output / 'all-compiled'
        run([sys.executable, HERE / 'prepare_canonical_cpu.py', '--binary', snapshot / 'target/release/iqa-compete',
             '--portable', portable, '--inputs', home / 'benchmark-bundles/iqa-mac-20261018-coverage-v2/512-nn-cpu/inputs',
             '--snapshot', snapshot, '--output', complete, '--metrics', 'all'])
        manifest = json.loads((complete / 'manifest.json').read_bytes())
        production = a.output / 'ordinary-cpu'
        ordinary = [c for c in manifest['cells'] if not c['metric_name'].startswith('dvifmish-')]
        freeze(complete, production, ordinary)
        control = a.output / 'control-bundle'
        selected = [c for c in ordinary if c['metric_name'] == 'ssim2' and c['size_name'] == 'cid22'
                    and (c['timing_phase'] == 'warm' or c['cold_repeat'] == 0)]
        freeze(production, control, selected)
        output = a.output / 'control-results'
        run([sys.executable, control / 'durable.py', 'run', control / 'manifest.json', output,
             '--run-heavy', control / 'tools/run-heavy', '--memory', '12G', '--jobs', '8',
             '--lock', a.output / 'private-control.lock', '--max-attempts', '2'])
        checked_control(json.loads((output / 'report.json').read_bytes()))
        (a.output / 'control-pass.json').write_text(json.dumps(dict(
            binary_sha256=sha(production / 'bin/iqa-compete'), report_sha256=sha(output / 'report.json'),
            strict_gate_clean=True), indent=2) + '\n')
    # The production worker reacquires the original shared lock for every scope.
    result = run([sys.executable, production / 'mac_service.py', 'install', production,
         a.output / 'results', '--run-heavy', production / 'tools/run-heavy',
         '--lock', home / 'tmp/mac-heavy.lock', '--jobs', '8'], capture_output=True, text=True)
    (a.output / 'installed.json').write_text(result.stdout)
    print(result.stdout, end='')


if __name__ == '__main__':
    main()
