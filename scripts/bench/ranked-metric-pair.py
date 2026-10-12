#!/usr/bin/env python3
"""Prepare an exploratory external A/B plan, or check exact wall score bits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def score_bits(directory):
    text = (Path(directory) / 'stdout.txt').read_text()
    matches = re.findall(r'bits ([0-9a-f]{16}(?:/[0-9a-f]{16})*)\)', text)
    if not matches:
        digests = re.findall(r'digest=0x([0-9a-f]{16})', text)
        if not digests:
            raise ValueError(f'expected score bits or map digests in {directory}')
        return digests
    if len(matches) != 1:
        raise ValueError(f'expected exactly one score bit pattern in {directory}')
    return matches[0]


def fleet_plan(args):
    """Bind a controlled all-product fleet case to two ordinary binaries."""
    case_path = Path(args.case).resolve(strict=True)
    case = json.loads(case_path.read_bytes())
    control_path = Path(args.control).resolve(strict=True)
    control = json.loads(control_path.read_bytes())
    driver = control['driver']
    if (case['schema'] != 2 or case['products'] != 'all' or case['reuse'] != 'cold'
            or case['warmup_operations'] != 0 or control['timing_usable'] is not True
            or driver['schema'] != 1 or driver['memory'] != []
            or driver['operations'] != 2 or set(driver['correctness']) != {'products'}):
        raise ValueError('fleet pair needs a controlled ordinary two-operation cold all-product case')
    workload = control['workload']
    for observed, declared in [('metric', 'metric'), ('width', 'width'), ('height', 'height'),
                               ('source_stride_bytes', 'stride_bytes'), ('rayon_threads', 'threads'),
                               ('pixel_format', 'pixel_format'), ('reuse', 'reuse'),
                               ('warmup_operations', 'warmup_operations')]:
        if workload[observed] != case[declared]:
            raise ValueError('control workload differs from declared fleet case')
    products = driver['correctness']['products']
    if len(products) != 2 or not products[0] or products[0] != products[1]:
        raise ValueError('control needs repeatable scalar products')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    runner = Path(args.runner).resolve(strict=True)
    checker = root / 'checker.py'
    shutil.copyfile(Path(__file__).resolve().parents[1] / 'jobsys/check_benchmark_pair_output.py', checker)
    expected = root / 'expected.json'
    expected.write_text(json.dumps(dict(operations=2, correctness=driver['correctness'])) + '\n')
    inputs = dict(checker=str(checker), expected=str(expected), control=str(control_path))
    arms = []
    for label, binary, revision in [('baseline', args.baseline, args.baseline_revision),
                                    ('candidate', args.candidate, args.candidate_revision)]:
        binary = Path(binary).resolve(strict=True)
        files = {'driver': dict(source=str(binary), executable=True),
                 case_path.name: dict(source=str(case_path), executable=False)}
        for side in ['reference', 'distorted']:
            relative = Path(case[side])
            source = (case_path.parent / relative).resolve(strict=True)
            if relative.is_absolute() or not source.is_relative_to(case_path.parent):
                raise ValueError('case input escapes its directory')
            files[str(relative)] = dict(source=str(source), executable=False)
        spec = dict(schema=1, target='x86_64-unknown-linux-gnu', files=files, plan=dict(
            schema=1, program='driver', args=['fleet', case_path.name],
            env={'RAYON_NUM_THREADS': str(case['threads'])}, backend='wall', operations=2,
            timeout_seconds=60, expected_correctness=driver['correctness'], workload=case))
        spec_path = root / f'{label}-spec.json'
        spec_path.write_text(json.dumps(spec) + '\n')
        bundle = root / label
        result = subprocess.run([str(runner), 'bundle', str(spec_path), str(bundle)],
                                capture_output=True, timeout=60)
        (root / f'{label}-bundle.stdout').write_bytes(result.stdout)
        (root / f'{label}-bundle.stderr').write_bytes(result.stderr)
        if result.returncode:
            raise ValueError(f'{label} bundle refused; see retained stderr')
        for path in bundle.rglob('*'):
            if path.is_file():
                inputs[f'{label}_{path.relative_to(bundle)}'] = str(path)
        arms.append(dict(id=label, program=str(bundle / 'driver'),
                         args=['fleet', str(bundle / case_path.name)], cwd=str(bundle),
                         env={'RAYON_NUM_THREADS': str(case['threads']), 'ZENBENCH_OPERATIONS': '2',
                              'ZENBENCH_RESULT_FILE': '${output}/driver.json', 'ZENMETRICS_PAIR_TIMING': '1'},
                         revision=revision + ';sha256=' + hashlib.sha256(binary.read_bytes()).hexdigest(),
                         timeout_seconds=60))
    plan = dict(schema=1, baseline=arms[0], candidates=[arms[1]],
                cases=[dict(id=f"{case['metric']}-{case['width']}x{case['height']}", inputs=inputs)],
                metric=dict(extract=dict(kind='prefix', prefix='METRIC_NS='), unit='ns', expected=2, divide_by=1),
                check=dict(kind='command', program=sys.executable,
                           args=[str(checker), '${baseline}', '${candidate}', str(expected)], timeout_seconds=10),
                warmup='Zero declared warmup; preparation excluded from two complete cold scoring calls',
                blocks=4, aa=True, deadline_seconds=180)
    (root / 'pair.json').write_text(json.dumps(plan, indent=2) + '\n')
    (root / 'identity.json').write_text(json.dumps(dict(
        runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
        control_sha256=hashlib.sha256(control_path.read_bytes()).hexdigest(),
        qualification='functional exact-product control; exploratory pair, no confirmation acceptance'), indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check')
    check.add_argument('baseline')
    check.add_argument('candidate')
    plan = sub.add_parser('plan')
    for name in ['baseline', 'candidate', 'baseline_revision', 'candidate_revision', 'output']:
        plan.add_argument(name)
    plan.add_argument('--threads', type=int, default=1)
    plan.add_argument('--cpus', default='2')
    plan.add_argument('--hdr-version', choices=['v2', 'v3'])
    plan.add_argument('--side', type=int, default=250)
    fleet = sub.add_parser('fleet-plan')
    for name in ['baseline', 'candidate', 'baseline_revision', 'candidate_revision',
                 'case', 'control', 'runner', 'output']:
        fleet.add_argument(name)
    args = parser.parse_args()
    if args.command == 'fleet-plan':
        fleet_plan(args)
        return
    if args.command == 'check':
        before, after = score_bits(args.baseline), score_bits(args.candidate)
        if before != after:
            raise ValueError(f'score bits changed: {before} != {after}')
        print(f'exact score bits: {before}')
        return
    if args.threads < 1:
        raise ValueError('threads must be positive')
    def arm(label, path, revision):
        binary = Path(path).resolve(strict=True)
        return dict(id=label, program=str(binary),
                    args=(['${width}', '${repetitions}', args.hdr_version] if args.hdr_version
                          else ['${width}', '${height}', '${repetitions}']),
                    revision=revision + ';sha256=' + hashlib.sha256(binary.read_bytes()).hexdigest(),
                    timeout_seconds=60)
    value = dict(
        schema=1, baseline=arm('before', args.baseline, args.baseline_revision),
        candidates=[arm('after', args.candidate, args.candidate_revision)],
        env=dict(RAYON_NUM_THREADS=str(args.threads), ZENBENCH_CPU_SET=args.cpus),
        record_env=['TMPDIR'],
        cases=[dict(id=(f'{args.hdr_version}-{args.side}x{args.side}' if args.hdr_version else '3355x2516'),
                    fields=dict(width=str(args.side) if args.hdr_version else '3355',
                                height=str(args.side) if args.hdr_version else '2516',
                                repetitions='3' if args.hdr_version else '5'))],
        metric=dict(extract=dict(kind='regex',
                                pattern=r'in ([0-9.]+) s' if args.hdr_version else r': ([0-9.]+) ms/call',
                                capture=1),
                    unit='s' if args.hdr_version else 'ms', expected=3 if args.hdr_version else 1,
                    divide_by=1),
        check=dict(kind='command', program=sys.executable,
                   args=[str(Path(__file__).resolve()), 'check', '${baseline}', '${candidate}'],
                   timeout_seconds=10),
        warmup=('No excluded warmup; three native HDR metric calls with output-map digests' if args.hdr_version
                else 'No excluded warmup; five complete metric calls, input generation outside internal timing'),
        blocks=4, aa=True, deadline_seconds=180)
    with Path(args.output).open('x') as output:
        json.dump(value, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
