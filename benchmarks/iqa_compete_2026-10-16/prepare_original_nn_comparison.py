#!/usr/bin/env python3
"""Freeze resident RGB comparisons with original LPIPS/ST-LPIPS/DISTS/DeepDC."""
import argparse
import json
from pathlib import Path

from compare_canonical import sha
from supervision import timeout_command


METRICS = ['lpips-alex','lpips-vgg','lpips-squeeze','stlpips-alex','stlpips-vgg','dists','deepdc']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['binary','python','models','pairs','output','model-receipt']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--threads', default='1,8')
    p.add_argument('--sizes', default='cid22')
    p.add_argument('--metrics', default=','.join(METRICS))
    p.add_argument('--weight', action='append', type=Path, default=[])
    a = p.parse_args()
    timeout_command("Linux", 1100)
    selected = a.metrics.split(',')
    if not set(selected) <= set(METRICS):
        p.error('unsupported original model')
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    helper = here/'canonical_original_model_cell.py'
    environment = here/'canonical_worker_env.py'
    convention = repo/'crates/zenmetrics-nn/models/deepdc/export.py'
    package_root = a.python.absolute().parent.parent/'lib/python3.12/site-packages'
    bound = [a.binary.resolve(strict=True), a.python.absolute(), a.pairs.resolve(strict=True),
             a.model_receipt.resolve(strict=True), here/'compare_canonical.py',
             here/'compare_canonical_samples.py', here/'compare_canonical_budgeted.py', Path(__file__).resolve(),
             helper, environment, convention, *a.weight]
    scales = {}
    for line in (repo/'crates/zenmetrics-nn/models/jnd_scale.tsv').read_text().splitlines():
        if line and not line.startswith('#') and not line.startswith('column'):
            fields = line.split('\t');scales[fields[0]] = float(fields[1])*.001
    pairs = json.loads(a.pairs.read_text())
    if isinstance(pairs, dict):pairs = pairs['pairs']
    prefixes = {name:name.replace('-','_') for name in selected}
    receipt = json.loads(a.model_receipt.read_text())
    verified = {row['path']:row['sha256'] for row in receipt['artifacts']}
    for prefix in prefixes.values():
        files = list(a.models.glob(prefix+'*'))
        if not files:raise ValueError('missing frozen model: '+prefix)
        for file in files:
            if verified.get(str(file.resolve())) != sha(file):
                raise ValueError('model is absent from verified copy receipt: '+str(file))
            bound.append(file.resolve())
    for package in ['lpips','stlpips_pytorch','DISTS_pytorch','DeepDC_PyTorch']:
        directory = package_root/package
        if not directory.exists():raise ValueError('missing original package: '+str(directory))
        bound.extend(path for path in directory.rglob('*') if path.is_file() and path.suffix in ('.py','.pt','.pth'))
    settings = a.output.with_suffix('.settings.json')
    with settings.open('x') as stream:
        json.dump(dict(binary=str(a.binary.resolve()), environment={
            'ZENMETRICS_NN_ENGINE':'rten','ZENMETRICS_NN_MODEL_DIR':str(a.models.resolve())}), stream, indent=2)
    worker = a.output.with_suffix('.worker')
    with worker.open('x') as stream:
        import shlex
        stream.write('#!/bin/sh\nexec '+shlex.join([str(a.python.absolute()),str(environment),str(settings.resolve())])+' "$@"\n')
    worker.chmod(0o755)
    bound.extend([worker.resolve(),settings.resolve()])
    labels = {'lpips':'lpips 0.1.4','stlpips':'stlpips-pytorch 0.0.2','dists':'DISTS-pytorch 0.1','deepdc':'DeepDC-PyTorch 0.2 / input normalization'}
    cases = []
    for pair in pairs:
        if pair['size_name'] not in a.sizes.split(','):continue
        paths = {role:str((a.pairs.parent/pair[role]).resolve(strict=True)) for role in ['reference','distorted']}
        for path in paths.values():
            if Path(path).stat().st_size != pair['width']*pair['height']*3:raise ValueError('RGB size mismatch')
            bound.append(Path(path))
        for name in selected:
            for threads in map(int,a.threads.split(',')):
                cases.append(dict(id=f'{name}-{pair.get("pair","pair")}-{pair["size_name"]}-t{threads}',metric=name,
                    width=pair['width'],height=pair['height'],threads=threads,**paths,
                    canonical=dict(label=labels[name.split('-')[0]],command=[str(a.python.absolute()),str(helper),
                        '{metric}','{width}','{height}','{reference}','{distorted}','{threads}','{samples}',
                        '--deepdc-convention',str(convention)]),absolute_bound=scales[prefixes[name]],
                    matched_scope=True,scope='Resident RGB8 conversion, original CPU model callable and temporary output release; immutable weights resident; native RTEN engine'))
    if not cases:p.error('no selected inputs')
    with a.output.open('x') as stream:
        json.dump(dict(schema=1,binary=str(worker.resolve()),rust_source=a.source,
                       arm_timeout_seconds=1100,rust_chunk_budget_ms=850_000,
                       bindings={str(path):sha(path) for path in bound},cases=cases),stream,indent=2)
        stream.write('\n')
    print(json.dumps(dict(cases=len(cases),metrics=selected)))


if __name__=='__main__':main()
