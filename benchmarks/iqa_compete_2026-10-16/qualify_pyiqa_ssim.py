#!/usr/bin/env python3
"""Apply the authorized scalar policy to an immutable, previously joined grid."""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from comparison_tolerances import PYIQA_SSIM_BOUNDS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verdict(delta, bound):
    return 'pass' if math.isfinite(delta) and delta <= bound else 'fail'


def controls():
    for bound in set(PYIQA_SSIM_BOUNDS.values()):
        assert verdict(math.nextafter(bound, 0), bound) == 'pass'
        assert verdict(bound, bound) == 'pass'
        assert verdict(math.nextafter(bound, math.inf), bound) == 'fail'
        assert verdict(float('nan'), bound) == 'fail'
        assert verdict(float('inf'), bound) == 'fail'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('joined', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    controls()
    index_path = a.joined / 'primary-pyiqa-ssim.json'
    old = json.loads(index_path.read_text())
    assert old['rows'] == 480 and old['gate'].startswith('UNBOUND')
    primary_errors, rust_errors = {}, {}
    checked = []
    # Qualify only explicit matching window-size errors, never generic failures.
    for artifact in old['artifacts']:
        root = Path(artifact['root'])
        manifest = root / 'joined-files.sha256.jsonl'
        assert sha(manifest) == artifact['manifest_sha256']
        expected = {r['path']: r['sha256'] for r in map(json.loads, manifest.read_text().splitlines())}
        pattern = 'results.json' if artifact['cohort'] in ['multiscale', 'single_scale'] else 'events.jsonl'
        for path in sorted(root.rglob(pattern)):
            assert sha(path) == expected[str(path.relative_to(root))]
            checked.append(dict(path=str(path), sha256=sha(path)))
            if pattern == 'results.json':
                group = path.parent.name
                for row in json.loads(path.read_text())['rows']:
                    for arm in ['baseline', 'candidate']:
                        if row[arm]['returncode']:
                            rust_errors[(group, row['metric'], arm)] = row[arm]['stderr']
            else:
                for row in map(json.loads, path.read_text().splitlines()):
                    if row.get('status') == 'canonical-error':
                        key = (row['pair'], row['size'], row['threads'], row['factor'], row['convention'])
                        primary_errors[key] = row['error']
    a.output.mkdir()
    outputs, counts = [], collections.Counter()
    seen = collections.Counter()
    for shard in old['outputs']:
        path = a.joined / shard['path']
        assert sha(path) == shard['sha256']
        rows = []
        for r in csv.DictReader(path.open(), delimiter='\t'):
            metric = r['metric']; bound = PYIQA_SSIM_BOUNDS[metric]
            seen[metric] += 1
            item = {k: r[k] for k in ['metric', 'pair', 'size', 'threads', 'canonical', 'baseline', 'candidate', 'baseline_abs_delta', 'candidate_abs_delta']}
            item['bound'] = bound
            for arm in ['baseline', 'candidate']:
                if r['canonical_status'] == 'scored':
                    result = verdict(float(r[arm + '_abs_delta']), bound) if r[arm] else 'refusal-mismatch'
                else:
                    assert r['canonical_status'] == 'canonical-error:RuntimeError'
                    key = (r['pair'], r['size'], int(r['threads']), int(r['factor']), r['convention'])
                    assert "Kernel size can't be greater than actual input size" in primary_errors[key]
                    if r[arm]:
                        result = 'refusal-mismatch'
                    else:
                        group = f"{r['pair']}-{r['size']}-t{r['threads']}"
                        assert r[arm + '_returncode'] == '1'
                        assert 'valid 11x11 window requires at least 11x11' in rust_errors[(group, metric, arm)]
                        result = 'matching-size-refusal'
                item[arm + '_status'] = result
                counts[(arm, result)] += 1
            item['qualified_ratio'] = False
            rows.append(item)
        stream = io.StringIO(newline='')
        w = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        w.writeheader(); w.writerows(rows)
        data = stream.getvalue().encode(); assert len(data) <= 30_000
        name = shard['path'].replace('primary-pyiqa-', 'approved-pyiqa-')
        (a.output / name).write_bytes(data)
        outputs.append(dict(path=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    assert set(seen) == set(PYIQA_SSIM_BOUNDS) and set(seen.values()) == {24}
    result = dict(rows=sum(seen.values()), source_index=dict(path=str(index_path.resolve()), sha256=sha(index_path)),
                  policy='User-authorized 2026-10-19: 1e-6 unrounded/direct/fixed; 5e-5 original rounded classes; absolute inclusive',
                  policy_source_sha256=sha(Path(__file__).with_name('comparison_tolerances.py')),
                  classifier_source_sha256=sha(Path(__file__)), outputs=outputs, checked_error_artifacts=checked,
                  counts=[dict(arm=k[0], status=k[1], rows=v) for k,v in sorted(counts.items())],
                  qualified_timing_cells=0, same_implementation_golden_tests='unchanged')
    # Keep the long raw-path verification list outside the compact public index.
    (a.output / 'checked-error-artifacts.json').write_text(json.dumps(result.pop('checked_error_artifacts'), indent=2)+'\n')
    result['checked_error_artifacts'] = dict(path=str((a.output / 'checked-error-artifacts.json').resolve()), sha256=sha(a.output / 'checked-error-artifacts.json'))
    (a.output / 'approved-pyiqa-ssim.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
