#!/usr/bin/env python3
"""Freeze a randomized zenbench external-process HaarPSI experiment plan."""
import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import sys

from haarpsi_primary import SOURCE_SHA
from octave_primary import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['inputs', 'source', 'rust-build', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--blocks', type=int, required=True)
    p.add_argument('--pair')
    p.add_argument('--size')
    p.add_argument('--threads', type=int)
    p.add_argument('--metric', choices=['haarpsi', 'haarpsi-y'])
    a = p.parse_args()
    assert a.blocks >= 4
    a.inputs = a.inputs.resolve(strict=True)
    a.source = a.source.resolve(strict=True)
    a.rust_build = a.rust_build.resolve(strict=True)
    assert sha(a.source) == SOURCE_SHA
    base = Path(__file__).resolve().parent
    binary = a.rust_build / 'target/release/iqa-compete'
    receipt = a.rust_build / 'build-receipt.json'
    assert sha(binary) == json.loads(receipt.read_text())['binary_sha256']
    budgets = [1, len(os.sched_getaffinity(0)) - 4]
    assert a.threads is None or a.threads in budgets
    cases = []
    for row, metric, threads, phase in itertools.product(json.loads(a.inputs.read_text())['pairs'],
            [a.metric] if a.metric else ['haarpsi', 'haarpsi-y'], [a.threads] if a.threads else budgets, ['cold', 'warm']):
        if a.pair and row['pair'] != a.pair or a.size and row['size_name'] != a.size:
            continue
        inputs = {name: str((a.inputs.parent / row[name]).resolve(strict=True)) for name in ['reference', 'distorted']}
        assert all(sha(Path(path)) == row[name + '_sha256'] for name, path in inputs.items())
        inputs.update(source=str(a.source), binary=str(binary), receipt=str(receipt),
                      helper=str(base / 'paired_primary_arm.py'), haar_helper=str(base / 'haarpsi_primary.py'),
                      octave_helper=str(base / 'octave_primary.py'), supervisor=str(base / 'supervision.py'),
                      octave=str(Path(shutil.which('octave-cli')).resolve(strict=True)))
        cases.append(dict(id=f'{metric}-{row["pair"]}-{row["size_name"]}-t{threads}-{phase}', inputs=inputs,
                          fields=dict(metric=metric, phase=phase, width=str(row['width']), height=str(row['height']),
                                      threads=str(threads), cpus='0' if threads == 1 else '0-' + str(threads - 1))))
    assert cases
    common = [str(Path(sys.executable).resolve()), '${helper}', '--metric', '${metric}', '--phase', '${phase}',
              '--width', '${width}', '--height', '${height}', '--threads', '${threads}']
    for name in ['reference', 'distorted', 'source', 'binary', 'receipt', 'output']:
        common.extend(['--' + name, '${' + name + '}'])
    arms = [dict(id=arm, program=str(Path(shutil.which('taskset')).resolve(strict=True)),
                 args=['-c', '${cpus}', *common, '--arm', arm], revision=revision, timeout_seconds=320)
            for arm, revision in [('canonical', 'HaarPSI.m 3b9599b0c1911ee08f1bb991bb19e19d86727819'),
                                  ('rust', 'e2d414c33a6d3c7da50a818fa36e56654d3ebeef')]]
    plan = dict(schema=1, baseline=arms[0], candidates=arms[1:], cases=cases,
                metric=dict(extract=dict(kind='prefix', prefix='PAIR_TIME_NS '), unit='ns', expected=1, divide_by=1),
                check=dict(kind='command', program=str(Path(sys.executable).resolve()),
                           args=[str(base / 'paired_primary_arm.py'), '--check', '${baseline}', '${candidate}'], timeout_seconds=30),
                warmup='Cold: first scorer construction/evaluation after resident RGB preparation. Warm: arithmetic mean of ten in-process score timing records after warmup; Rust records are zenbench round batch means, canonical records are individual calls. Raw reads and process startup excluded. Fresh randomized A/B/A blocks; no pooled batch pairing.',
                blocks=a.blocks, aa=True, deadline_seconds=43200, seed=20261019,
                env=dict(LC_ALL='C', ZENBENCH_NO_SAVE='1'),
                record_env=['PATH', 'TMPDIR', 'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'ZENBENCH_CPU_POLICY'])
    a.output.mkdir()
    (a.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (a.output / 'preparation.json').write_text(json.dumps(dict(producer_sha256=sha(Path(__file__)), inputs_sha256=sha(a.inputs),
        cases=len(cases), blocks=a.blocks, invocations=len(cases)*a.blocks*3, mode='iterate; no confirmation verdict inferred'), indent=2) + '\n')


if __name__ == '__main__':
    main()
