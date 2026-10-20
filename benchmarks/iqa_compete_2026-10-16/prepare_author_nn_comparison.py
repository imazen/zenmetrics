#!/usr/bin/env python3
"""Freeze explicit original-network callers and verified copied native models.

The supplied manifest names each original source/weight asset and its runtime.
No model, reference implementation, numerical bound or inference engine is chosen
implicitly. Hashing belongs under shared resource admission.
"""
import argparse
import json
from pathlib import Path
import shlex

from compare_canonical import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    here = Path(__file__).resolve().parent
    scales_path = here.parents[1]/'crates/zenmetrics-nn/models/jnd_scale.tsv'
    scales = {}
    for line in scales_path.read_text().splitlines():
        if line and not line.startswith(('#', 'column')):
            fields = line.split('\t')
            scales[fields[0]] = float(fields[1])*.001
    pairs = json.loads(Path(manifest['pairs']).read_text())
    if isinstance(pairs, dict):
        pairs = pairs['pairs']
    pair = next(row for row in pairs if row['size_name'] == manifest.get('size', 'cid22'))
    paths = {role: str((Path(manifest['pairs']).parent/pair[role]).resolve(strict=True))
             for role in ('reference', 'distorted')}
    for path in paths.values():
        if Path(path).stat().st_size != pair['width']*pair['height']*3:
            raise ValueError('RGB byte count differs')
    receipt_path = Path(manifest['model_receipt'])
    verified = {row['path']: row['sha256'] for row in json.loads(receipt_path.read_text())['artifacts']}
    a.output.mkdir()
    for metric, spec in manifest['metrics'].items():
        selected = list(Path(spec['models']).glob(spec.get('model_prefix', metric.replace('-', '_'))+'*'))
        if not selected:
            raise ValueError('missing explicit native model: '+metric)
        for path in selected:
            if verified.get(str(path.resolve())) != sha(path):
                raise ValueError('native model differs from copied receipt: '+str(path))
        bound = [a.manifest.resolve(), scales_path, receipt_path, Path(manifest['binary']),
                 Path(manifest['pairs']), *map(Path, paths.values()), *selected,
                 Path(__file__).resolve(), here/'canonical_worker_env.py',
                 here/'compare_canonical.py', here/'compare_canonical_samples.py',
                 here/'compare_canonical_budgeted.py', Path(spec['helper']), Path(spec['config'])]
        for original in spec['assets']:
            path = Path(original).resolve(strict=True)
            if path.is_dir():
                bound.extend(item for item in path.rglob('*') if item.is_file()
                             and '.git' not in item.parts and '__pycache__' not in item.parts
                             and item.suffix != '.pyc')
            else:
                bound.append(path)
        settings = a.output/(metric+'.settings.json')
        with settings.open('x') as stream:
            json.dump(dict(binary=manifest['binary'], environment=dict(
                ZENMETRICS_NN_ENGINE=spec.get('engine', 'rten'),
                ZENMETRICS_NN_MODEL_DIR=spec['models'],
                ZENMETRICS_NN_ACCEPT_WEIGHTS=spec.get('consent', ''))), stream, indent=2)
        worker = a.output/(metric+'.worker')
        with worker.open('x') as stream:
            stream.write('#!/bin/sh\nexec '+shlex.join([spec['python'], str(here/'canonical_worker_env.py'),
                                                       str(settings.resolve())])+' "$@"\n')
        worker.chmod(0o755)
        bound.extend([settings, worker, Path(spec['python'])])
        cases = []
        for threads in manifest.get('threads', [1, 8]):
            cases.append(dict(id=f'{metric}-original-{pair["size_name"]}-t{threads}', metric=metric,
                width=pair['width'], height=pair['height'], threads=threads, **paths,
                canonical=dict(label=spec['label'], command=[spec['python'], spec['helper'], spec['config'],
                    '{metric}', '{width}', '{height}', '{reference}', '{distorted}', '{threads}', '{samples}']),
                absolute_bound=scales[spec.get('column', metric.replace('-', '_'))], matched_scope=True,
                scope='Resident RGB8 conversion and original CPU callable, both NR images, temporary output release; immutable weights resident; native '+spec.get('engine', 'rten')))
        path = a.output/(metric+'.plan.json')
        with path.open('x') as stream:
            json.dump(dict(schema=1, binary=str(worker.resolve()), rust_source=manifest['rust_source'],
                arm_timeout_seconds=1200, rust_chunk_budget_ms=850_000,
                bindings={str(item.absolute()): sha(item) for item in bound}, cases=cases), stream, indent=2)
            stream.write('\n')
        print(json.dumps(dict(metric=metric, cases=len(cases), plan=str(path))), flush=True)


if __name__ == '__main__':
    main()
