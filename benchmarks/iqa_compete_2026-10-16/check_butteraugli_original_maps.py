#!/usr/bin/env python3
"""Compare frozen original-Google Butteraugli maps, without a new map bound."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from compare_canonical import sha
from haarpsi_primary import execute


MODULE = 'crates/zenmetrics-cli/src/metrics/butteraugli_original.rs'
AUTHOR = '71b18b636b9c7d1ae0c1d3730b85b3c127eb4511'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('repo', 'native-source', 'native-revision', 'reference-source',
                'reference-plan', 'pairs', 'output'):
        p.add_argument('--' + key, required=True)
    a = p.parse_args()
    root = Path(a.output).resolve()
    root.mkdir()
    here = Path(__file__).resolve().parent
    reference = Path(a.reference_source).resolve()
    revision = subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != AUTHOR:
        raise ValueError('expected the pinned original Google source')
    source = Path(a.native_source)
    committed = subprocess.check_output(['git', '-C', a.repo, 'show', a.native_revision + ':' + MODULE])
    if source.read_bytes() != committed:
        raise ValueError('native module differs from declared published source')
    plan_path = Path(a.reference_plan)
    plan = json.loads(plan_path.read_text())
    selected = {path: digest for path, digest in plan['bindings'].items()
                if Path(path).is_relative_to(reference)}
    if len(selected) < 3:
        raise ValueError('original reference source bindings required')
    for path, digest in selected.items():
        if sha(path) != digest:
            raise ValueError('reference source changed: ' + path)
    frozen = root / 'butteraugli-original-source.rs'
    frozen.write_bytes(committed)
    driver = root / 'native.rs'
    driver.write_bytes((here / 'butteraugli_original_map.rs').read_bytes())
    rust, cpp = root / 'native-map', root / 'reference-map'
    commands = [
        ['rustc', '--edition=2024', '-C', 'opt-level=3', str(driver), '-o', str(rust)],
        ['g++', '-O3', '-std=c++20', '-I' + str(reference),
         str(here / 'butteraugli_original_map.cpp'),
         str(reference / 'butteraugli/butteraugli.cc'), '-lpng', '-ljpeg', '-o', str(cpp)],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    pairs = Path(a.pairs).resolve()
    receipt = dict(native_revision=a.native_revision, native_module_sha256=sha(frozen),
                   reference_origin='https://github.com/google/butteraugli', reference_revision=revision,
                   reference_source_sha256=selected, reference_plan_sha256=sha(plan_path),
                   commands=commands, binaries={str(binary): sha(binary) for binary in (rust, cpp)},
                   helper_sources={name: sha(here / name) for name in
                                   ('butteraugli_original_map.rs', 'butteraugli_original_map.cpp',
                                    'check_butteraugli_original_maps.py', 'compare_canonical.py',
                                    'haarpsi_primary.py', 'octave_primary.py', 'supervision.py',
                                    'check_fixed_factor_oracle.py', 'fixed_factor_oracle.py',
                                    'image_resize_source.py')},
                   input_manifest_sha256=sha(pairs),
                   scope='Accuracy only. Both original callables use one effective worker under budgets 1/28.')
    (root / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    rows = []
    for pair in json.loads(pairs.read_text())['pairs']:
        w, h = pair['width'], pair['height']
        inputs = [(pairs.parent / pair[role]).resolve(strict=True) for role in ('reference', 'distorted')]
        for role, path in zip(('reference', 'distorted'), inputs):
            if path.stat().st_size != w * h * 3 or sha(path) != pair[role + '_sha256']:
                raise ValueError('input identity differs')
        for budget in (1, 28):
            folder = root / f'{pair["pair"]}-{pair["size_name"]}-t{budget}'
            folder.mkdir()
            env = dict(os.environ, TMPDIR=str(folder))
            for key in ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                env[key] = str(budget)
            scores, maps = {}, {}
            for role, binary in (('native', rust), ('canonical', cpp)):
                destination = folder / (role + '.f32')
                command = [str(binary), str(w), str(h), *map(str, inputs), str(budget), str(destination)]
                rc, wrapped = execute(command, env, folder, role, 1100)
                (folder / (role + '-command.json')).write_text(json.dumps(wrapped) + '\n')
                if rc:
                    raise ValueError('map probe failed; retained ' + str(folder))
                lines = [s.split() for s in (folder / (role + '-stdout.log')).read_text().splitlines()
                         if s.startswith('BUTTER_ORIGINAL_MAP ')]
                if len(lines) != 1 or list(map(int, lines[0][2:])) != [w, h, budget, 1]:
                    raise ValueError('map geometry or effective worker mismatch')
                scores[role] = float(lines[0][1])
                maps[role] = np.fromfile(destination, dtype='<f4').reshape(h, w)
            finite = np.isfinite(maps['native']) & np.isfinite(maps['canonical'])
            matching = ((np.isnan(maps['native']) & np.isnan(maps['canonical']))
                        | (np.isinf(maps['native']) & (maps['native'] == maps['canonical'])))
            delta = np.abs(maps['native'][finite].astype(np.float64) - maps['canonical'][finite])
            row = dict(pair=pair['pair'], size=pair['size_name'], width=w, height=h,
                       requested_worker_budget=budget, native_effective_threads=1, canonical_effective_threads=1,
                       scores=scores, scalar_abs_delta=abs(scores['native'] - scores['canonical']),
                       scalar_bound=None, map_bound=None, finite_pixels=int(finite.sum()),
                       nonfinite_mismatches=int((~finite & ~matching).sum()),
                       max_map_abs_delta=float(delta.max()) if delta.size else None,
                       map_bit_identical=sha(folder / 'native.f32') == sha(folder / 'canonical.f32'),
                       files={path.name: sha(path) for path in folder.iterdir() if path.is_file()})
            (folder / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
            rows.append(row)
            print('ORIGINAL_MAP_CASE', folder.name, row['scalar_abs_delta'],
                  row['max_map_abs_delta'], row['nonfinite_mismatches'], flush=True)
    for path, digest in selected.items():
        if sha(path) != digest:
            raise ValueError('reference changed during map acquisition')
    (root / 'completion.json').write_text(json.dumps(dict(cases=rows,
        build_receipt_sha256=sha(root / 'build-receipt.json'),
        qualification='Raw accuracy observations; no assigned scalar/map bound or speed ratio'), indent=2) + '\n')


if __name__ == '__main__':
    main()
