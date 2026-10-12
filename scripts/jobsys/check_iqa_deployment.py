#!/usr/bin/env python3
"""Run explicit CPU controls before collecting a native deployment panel."""
import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
for name in ['binaries', 'runner', 'checker_tree', 'output', 'machine', 'domain']:
    parser.add_argument(name)
parser.add_argument('--sdr-size', type=int, nargs=2, default=[256, 256], metavar=('WIDTH', 'HEIGHT'))
parser.add_argument('--hdr-size', type=int, nargs=2, default=[128, 96], metavar=('WIDTH', 'HEIGHT'))
args = parser.parse_args()
if any(value < 1 for value in [*args.sdr_size, *args.hdr_size]):
    parser.error('dimensions must be positive')
binaries, runner, checker_tree, root = [Path(getattr(args, name)).resolve()
                                      for name in ['binaries', 'runner', 'checker_tree', 'output']]
checkers = checker_tree / 'scripts/jobsys'
deployment = Path(__file__).with_name('benchmark_iqa_deployment.py').resolve()
root.mkdir(parents=True, exist_ok=False)
commands = [
    [checkers/'check_iqa_study.py', binaries, root/'sdr-controls', '--size', *args.sdr_size],
    [checkers/'check_iqa_hdr.py', binaries, root/'hdr-controls', '--size', *args.hdr_size],
    [deployment, binaries, runner, root/'sdr-controls', root/'hdr-controls',
     root/'benchmark-observations', args.machine, args.domain],
]
for command in commands:
    subprocess.run([sys.executable, '-B', *map(str, command)], check=True)
