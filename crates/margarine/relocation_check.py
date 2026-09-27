#!/usr/bin/env python3
"""Require two builds to produce identical scalar output and native diffmaps.

The input TSV supplies reference/distorted paths. Both binaries see the same
verified files; each result and map is retained, including failures.
"""
import argparse
import csv
import json
import hashlib
import io
from pathlib import Path
import subprocess

from score_manifest import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pairs', type=Path)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--native-strip', type=int,
                        help='compare research commands with this row count instead of the named CLI')
    args = parser.parse_args()
    if args.native_strip is not None and args.native_strip <= 0:
        parser.error('native strip rows must be positive')
    with args.pairs.open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    if not rows or any(not row.get(k) for row in rows for k in ('reference', 'distorted')):
        raise ValueError('expected nonempty reference/distorted manifest')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(pairs=len(rows), verified=0, status='running',
                    pairs_sha256=digest(args.pairs),
                    binaries={name:dict(path=str(path.resolve()), sha256=digest(path))
                              for name, path in [('before', args.before), ('after', args.after)]},
                    native_strip=args.native_strip, inputs={}, records=[])
    target = args.output/'_MANIFEST.json'
    def save():
        target.write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    with (args.output/'progress.log').open('x', buffering=1) as log:
        for index, row in enumerate(rows):
            pair = [str(Path(row[k]).resolve()) for k in ('reference', 'distorted')]
            expected_inputs = {path:digest(Path(path)) for path in pair}
            manifest['inputs'].update(expected_inputs)
            outputs = {}
            for name, binary in [('before', args.before), ('after', args.after)]:
                stem = args.output/f'{index:04}-{name}'
                map_path = stem.with_suffix('.f32le')
                command = ([str(binary.resolve()), '--native-strip', str(args.native_strip), *pair, str(map_path)]
                           if args.native_strip else
                           [str(binary.resolve()), '--all-scores', '--diffmap', str(map_path), *pair])
                with stem.with_suffix('.stdout').open('xb') as stdout, stem.with_suffix('.stderr').open('xb') as stderr:
                    subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
                text = stem.with_suffix('.stdout').read_text()
                if args.native_strip:
                    scalar_rows = list(csv.DictReader(io.StringIO(text), delimiter='\t'))
                    if len(scalar_rows) != 1 or scalar_rows[0].pop('diffmap') != str(map_path):
                        raise ValueError('unexpected research output or diffmap path')
                    scalars = scalar_rows[0]
                else:
                    scalars = json.loads(text)
                scalar_hash = hashlib.sha256(json.dumps(scalars, sort_keys=True).encode()).hexdigest()
                outputs[name] = dict(scalars_sha256=scalar_hash,
                                     map_sha256=digest(map_path), map_bytes=map_path.stat().st_size)
            manifest['records'].append(dict(index=index, **outputs))
            save()
            if outputs['before'] != outputs['after']:
                raise ValueError(f'relocation changed pair {index}: {pair}')
            if expected_inputs != {path:digest(Path(path)) for path in pair}:
                raise ValueError(f'input changed during replay: {pair}')
            manifest['verified'] += 1
            save()
            message = f'Exact relocation replay {index+1}/{len(rows)}'
            print(message, flush=True)
            print(message, file=log, flush=True)
    manifest['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
