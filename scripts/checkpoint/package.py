#!/usr/bin/env python3
"""Package a tested native CLI checkpoint; never publish or select a GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def artifact_hashes(package):
    return {p.relative_to(package).as_posix(): digest(p)
            for p in sorted(package.rglob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--source', required=True, help='Full immutable source commit')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--runner-source', required=True, type=Path)
    parser.add_argument('--cross', action='store_true', help='Use cross for i686 Linux validation')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / 'ci/checkpoint-profiles.json').read_text())
    if args.profile not in config['profiles']:
        parser.error('Unknown checkpoint profile')
    if len(args.source) != 40 or any(c not in '0123456789abcdef' for c in args.source):
        parser.error('--source must be a full lowercase Git commit SHA')
    if args.out.exists():
        parser.error('Output already exists; checkpoint evidence must not be overwritten')
    if os.environ.get('RUSTFLAGS') or os.environ.get('CARGO_ENCODED_RUSTFLAGS'):
        parser.error('Unrecorded Rust flags refused; checkpoint uses runtime CPU dispatch')
    compiler = subprocess.check_output(['rustc', '-Vv'], text=True)
    if f'release: {config["rust"]}\n' not in compiler:
        parser.error(f'Checkpoint requires Rust {config["rust"]}')
    runner_revision = subprocess.check_output(['git', '-C', str(args.runner_source), 'rev-parse', 'HEAD'], text=True).strip()
    if runner_revision != config['benchmark_runner']['commit']:
        parser.error('Benchmark runner source differs from the checkpoint pin')
    bake = root.parent / 'zensim/zensim/weights/d_sdr_add156_id100_negrich_dial_byid_2026-09-06.bin'
    if not bake.is_file():
        parser.error('Pinned zensim selector fixture is missing')
    gpu_expectation = os.environ.get('ZENMETRICS_TEST_GPU')
    if gpu_expectation not in ('present', 'absent'):
        parser.error('Caller must declare ZENMETRICS_TEST_GPU=present|absent')
    build_env = dict(os.environ, ZENMETRICS_TEST_ZENSIM_BAKE=str(bake))
    if args.cross:
        # Path dependencies may be members inheriting from a sibling workspace root.
        build_env['ZEN_CHECKPOINT_WORKSPACES'] = str(root.parent.parent)
        build_env['CROSS_CONFIG'] = str(root / 'ci/checkpoint-cross.toml')
    runner_metadata = ['cargo', 'metadata', '--locked', '--no-deps', '--format-version', '1',
                       '--manifest-path', str(args.runner_source / 'Cargo.toml')]
    subprocess.run(runner_metadata, cwd=root, env=build_env, stdout=subprocess.DEVNULL, check=True)
    features = ','.join(config['profiles'][args.profile])
    common = ['--locked', '-p', 'zenmetrics-cli', '--target', args.target, '--features', features]
    commands = [
        ['cargo', 'test', *common, '--all-targets'],
        ['cargo', 'test', *common, '--doc'],
        ['cargo', 'build', *common, '--release', '--bin', 'zenmetrics'],
    ]
    def execute(command, cwd=root):
        if args.cross and command[0] == 'cargo': command = ['cross', *command[1:]]
        subprocess.run(command, cwd=cwd, env=build_env, check=True)
    for command in commands:
        execute(command)
    commands.insert(0, runner_metadata)
    runner_command = ['cargo', 'build', '--locked', '--release', '--target', args.target,
                      '--manifest-path', str(args.runner_source / 'Cargo.toml'), '--bin', 'zenbench']
    execute(runner_command)
    commands.append(runner_command)
    worker_command = ['cargo', 'build', '--locked', '--release', '--target', args.target,
                      '-p', 'zenfleet-worker', '-p', 'zenfleet-ctl']
    execute(worker_command)
    commands.append(worker_command)
    suffix = '.exe' if 'windows' in args.target else ''
    target = Path(os.environ.get('CARGO_TARGET_DIR', root / 'target'))
    binary = target / args.target / 'release' / ('zenmetrics' + suffix)
    args.out.mkdir(parents=True)
    shipped = args.out / binary.name
    shutil.copy2(binary, shipped)
    for name in ['zenfleet-worker', 'zenfleet-ctl', 'zenbench']:
        shutil.copy2(target / args.target / 'release' / (name + suffix), args.out / (name + suffix))
    driver_features = []
    if args.profile == 'full-wgpu': driver_features = ['study-gpu-wgpu']
    elif args.profile == 'full-cuda': driver_features = ['study-gpu-cuda']
    driver_dir = args.out / 'drivers'
    driver_dir.mkdir()
    for variant in ['latency', 'memory']:
        selected = driver_features + (['heap-phases'] if variant == 'memory' else [])
        command = ['cargo', 'build', '--locked', '--release', '--target', args.target,
                   '-p', 'cpu-profile', '--bin', 'new-metrics-wall']
        if selected: command += ['--features', ','.join(selected)]
        execute(command)
        commands.append(command)
        output = driver_dir / variant
        shutil.copy2(target / args.target / 'release' / ('new-metrics-wall' + suffix), output)
        build = {'schema': 1, 'target': args.target, 'binary_sha256': digest(output),
                 'source_revision': args.source, 'dependency_lock_sha256': digest(root / 'Cargo.lock'),
                 'compiler': compiler, 'features': selected, 'rustflags': [], 'profile': 'release',
                 'instrumented': variant == 'memory'}
        (driver_dir / (variant + '-build.json')).write_text(json.dumps(build, indent=2) + '\n')
    control_reports = {}
    for checker in ['check_iqa_study.py', 'check_iqa_hdr.py']:
        command = ['python3', str(root / 'scripts/jobsys' / checker), str(driver_dir.resolve()),
                   str((args.out / checker.removesuffix('.py')).resolve())]
        output = subprocess.check_output(command, text=True)
        print(output, end='')
        control_reports[checker] = json.loads(output)
        commands.append(command)
    adapter_dir = args.out / 'adapter'
    adapter_dir.mkdir()
    shutil.copy2(root / 'scripts/jobsys/benchmark_exec.py', adapter_dir / 'benchmark_exec.py')
    shutil.copy2(root / 'scripts/jobsys/benchmark-exec.cmd', adapter_dir / 'benchmark-exec.cmd')
    catalog = subprocess.check_output([str(shipped.resolve()), 'list-metrics'], text=True)
    (args.out / 'compiled-metrics.txt').write_text(catalog)
    shutil.copy2(root / 'Cargo.lock', args.out / 'Cargo.lock')
    shutil.copy2(args.runner_source / 'Cargo.lock', args.out / 'runner-Cargo.lock')
    shutil.copy2(root / 'ci/sibling-pins.tsv', args.out / 'sibling-pins.tsv')
    shutil.copy2(root / 'ci/checkpoint-profiles.json', args.out / 'checkpoint-profiles.json')
    shutil.copy2(root / 'LICENSE-AGPL3', args.out / 'LICENSE-AGPL3')
    metadata = {
        'schema': 1, 'source_commit': args.source, 'profile': args.profile,
        'target': args.target, 'rustc': compiler, 'features': config['profiles'][args.profile],
        'default_features': True, 'rustflags': [], 'commands': [['cross', *c[1:]] if args.cross and c[0] == 'cargo' else c for c in commands],
        'qualification': {'native_cli_tests': 'passed', 'test_gpu_expectation': gpu_expectation, 'doc_tests': 'passed',
                          'iqa_functional_controls': control_reports,
                          'hardware_gpu': 'not qualified', 'benchmark_admission': 'not run'},
        'benchmark_runner_commit': runner_revision,
        'test_fixture': {'zensim_selector_bake': bake.name, 'sha256': digest(bake)},
        'files': artifact_hashes(args.out),
    }
    (args.out / 'checkpoint.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(args.out)


if __name__ == '__main__':
    main()
