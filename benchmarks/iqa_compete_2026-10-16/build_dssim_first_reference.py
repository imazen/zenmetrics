#!/usr/bin/env python3
"""Link the first-call adapter against a verified, retained author archive.

Caller supplies heavy admission. This builder executes no metric or native port.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from compare_canonical import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parent_receipt', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--parent-sha256', required=True)
    args = parser.parse_args()
    args.parent_receipt = args.parent_receipt.resolve(strict=True)
    if sha(args.parent_receipt) != args.parent_sha256:
        raise ValueError('parent build receipt changed')
    parent = json.loads(args.parent_receipt.read_text())
    if parent.get('schema') != 'dssim-fullcall-build/1' or parent.get('passed') is not True:
        raise ValueError('successful full-call author-archive build required')
    source = Path(__file__).with_name('dssim_first_reference.c').resolve(strict=True)
    original = args.parent_receipt.parent/'inputs/dssim_fullcall_reference.c'
    bindings = {str(args.parent_receipt.resolve()): args.parent_sha256,
                str(source): sha(source), str(Path(__file__).resolve()): sha(__file__)}
    for identity in [*parent['assets'].values(), *parent['compiler_tools'].values(), *parent['runtime_files'].values()]:
        name, digest = identity['path'], identity['sha256']
        if name in bindings and bindings[name] != digest:
            raise ValueError('conflicting parent artifact identity')
        bindings[name] = digest
    for path, digest in [(original, parent['new_source_sha256']),
            (args.parent_receipt.parent/'inputs/dssim.h', parent['header_sha256']),
            (args.parent_receipt.parent/'inputs/libdssim_core.a', parent['library_sha256'])]:
        absolute = str(path.resolve())
        if absolute in bindings and bindings[absolute] != digest:
            raise ValueError('conflicting original source/header/archive identity')
        bindings[absolute] = digest
    def verify():
        for path, expected in bindings.items():
            if sha(path) != expected:
                raise ValueError('bound parent/source/tool artifact changed: '+path)
    verify()
    command = list(parent['compile']['argv'])
    if (command.count(str(original)) != 1 or command[-2] != '-o'
            or command[-1] != parent['binary']['path']
            or any('native' in value for value in command if value.startswith(('-march=', '-mcpu=', 'target-cpu=')))):
        raise ValueError('unexpected parent link command or native CPU target')
    args.output.mkdir()
    frozen = args.output.resolve()/'dssim_first_reference.c'
    frozen.write_bytes(source.read_bytes())
    binary = args.output.resolve()/'dssim-first-reference'
    command[command.index(str(original))] = str(frozen)
    command[-1] = str(binary)
    with (args.output/'compile.command.json').open('x') as stream:
        json.dump(command, stream, indent=2)
    with (args.output/'compile.stdout').open('x') as out, (args.output/'compile.stderr').open('x') as err:
        subprocess.run(command, stdout=out, stderr=err, check=True, timeout=120,
                       env=dict(os.environ, TMPDIR=str(args.output.resolve())))
    verify()
    if sha(frozen) != sha(source):
        raise ValueError('first-call source copy changed')
    result = dict(source=str(source), source_sha256=sha(source), output=str(binary),
        output_sha256=sha(binary), parent_receipt_sha256=args.parent_sha256,
        compiler_command=command, bindings=bindings, native_metric_calls=0,
        qualification='Link/source verification only. COLD retains original context/images beyond elapsed capture; subsequent SAMPLE retains the full-call free boundary. Original author archive/header/compiler and scalar arithmetic are unchanged. Independent scalar controls are still required; no accuracy or timing qualification is supplied by this build.')
    with (args.output/'build-receipt.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(binary=str(binary), sha256=sha(binary), native_metric_calls=0)))


if __name__ == '__main__':
    main()
