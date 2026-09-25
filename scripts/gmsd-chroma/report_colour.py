#!/usr/bin/env python3
"""Report a previously frozen TRAIN-only table via the existing stats owner.

Required columns: pair_key, stimulus_id, ref_sha, set, distortion, role, quality,
gmsd, mdsi, ms_gmsdc, fast_ssim2. All predictions use the same admitted rows.
The join/freeze step must produce a manifest with table_sha256 before this
script may read human values. A measured MDSI parity pass is also required.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

OWNER = Path('/home/lilith/work/zen/zensim')
EXPECTED = {
    'scripts/lib/zen_stats.py': '6e2bed69441195674e3a397f11a8eeb4b40e51dd271ebbd124e6702b68eb3d35',
    'scripts/band_reliability.py': 'ce98f801c3bb420e5363b3be52192df98fbb538e106d8c5060e2f81f8574b244',
}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--table', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--parity', type=Path, required=True)
    ap.add_argument('--ms-parity', type=Path, required=True)
    ap.add_argument('--panel', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--owner', type=Path, default=OWNER,
                    help='Frozen copy of the existing statistics owner')
    args = ap.parse_args()
    assert str(args.out.resolve()).startswith('/var/tmp/gmsd-chroma/')
    manifest = json.loads(args.manifest.read_text())
    assert manifest['table_sha256'] == sha(args.table)
    assert manifest['panel_sha256'] == sha(args.panel)
    assert manifest['implementation_frozen_before_labels'] is True
    assert manifest['population_committed_before_labels'] is True
    assert json.loads(args.parity.read_text())['passed'] is True
    assert json.loads(args.ms_parity.read_text())['passed'] is True
    for rel, digest in EXPECTED.items():
        assert sha(args.owner/rel) == digest, ('stats owner changed', rel)
    os.environ['ZEN_PANEL_BIN'] = str(args.panel.resolve())
    os.environ['TMPDIR'] = '/var/tmp/gmsd-chroma/tmp'
    sys.path.insert(0, str(args.owner))
    from scripts.lib import zen_stats
    from scripts.band_reliability import marginal_ci
    import numpy as np

    with args.table.open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    assert rows
    assert all(r['role'] == 'TRAIN' for r in rows)
    assert len({(r['set'], r['stimulus_id']) for r in rows}) == len(rows)
    arms = ['gmsd', 'mdsi', 'ms_gmsdc', 'fast_ssim2']
    assert all(math.isfinite(float(r[c])) for r in rows for c in ['quality', *arms])
    assert {r['set'] for r in rows} == {'kadid_train', 'tid_train', 'konfig_train'}
    groups = sorted({(r['set'], r['distortion']) for r in rows})
    groups = [(s, 'ALL_COLOUR') for s in sorted({r['set'] for r in rows})] + groups
    result = []
    for dataset, kind in groups:
        selected = [r for r in rows if r['set'] == dataset and
                    (kind == 'ALL_COLOUR' or r['distortion'] == kind)]
        refs = [r['ref_sha'] for r in selected]
        assert len(set(refs)) >= 3
        y = [float(r['quality']) for r in selected]
        for arm in arms:
            # Fixed distance orientation; stored SSIMULACRA2 is quality.
            sign = 1 if arm == 'fast_ssim2' else -1
            x = [sign*float(r[arm]) for r in selected]
            full = zen_stats.panel(x, y)
            signed = zen_stats.panel_batch([('point', x, y)], stats='srocc')[0]
            assert full['n_dropped'] == signed['n_dropped'] == 0
            _, lo, hi = marginal_ci(x, y, np.arange(len(y)), B=2000,
                                    seed=20260924, cluster=refs)
            assert math.isfinite(lo) and math.isfinite(hi)
            result.append(dict(set=dataset, distortion=kind, arm=arm,
                               n=len(y), references=len(set(refs)),
                               srocc_signed=signed['srocc_signed'],
                               srocc_abs=full['srocc'], ci95=[lo, hi],
                               in_sample_teacher_caveat=(arm == 'fast_ssim2'),
                               few_clusters=len(set(refs)) < 10))
    output = dict(schema='gmsd-chroma-colour-v1', table_sha256=sha(args.table),
                  manifest_sha256=sha(args.manifest), parity_sha256=sha(args.parity),
                  ms_parity_sha256=sha(args.ms_parity),
                  panel_sha256=sha(args.panel), owner_sha256=EXPECTED,
                  seed=20260924, bootstrap_replicates=2000, rows=result)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    for r in result:
        print(json.dumps(r, sort_keys=True))
    print('output_sha256', sha(args.out))


if __name__ == '__main__':
    main()
