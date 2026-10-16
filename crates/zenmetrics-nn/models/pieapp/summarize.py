#!/usr/bin/env python3
"""Validate complete KonJND coverage and write compact calibration provenance."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('pairs', type=Path)
    p.add_argument('scores', type=Path)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    with a.pairs.open() as f:
        pairs = list(csv.DictReader(f, delimiter='\t'))
    with a.scores.open() as f:
        scores = list(csv.DictReader(f, delimiter='\t'))
    expected = {(r['ref_path'], r['dist_path']): r for r in pairs}
    actual = {(r['ref_path'], r['dist_path']): float(r['pieapp']) for r in scores}
    assert len(expected) == len(actual) == len(pairs) == len(scores) == 1008
    assert expected.keys() == actual.keys()
    assert len({r['ref_path'] for r in pairs}) == 1008
    groups = {}
    for key, value in actual.items():
        assert math.isfinite(value)
        groups.setdefault(expected[key]['codec'].lower(), []).append(value)
    assert sorted(map(len, groups.values())) == [504, 504]
    pooled = statistics.median(actual.values())
    assert pooled > 0
    image_manifest = hashlib.sha256()
    image_paths = sorted({path for pair in pairs for path in (pair['ref_path'], pair['dist_path'])})
    for path in image_paths:
        image_manifest.update((path + '\t' + digest(Path(path)) + '\n').encode())
    result = dict(column='pieapp', jnd=pooled, count=1008,
                  codecs={k: dict(count=len(v), median=statistics.median(v)) for k,v in groups.items()},
                  pairs_sha256=digest(a.pairs), scores_sha256=digest(a.scores),
                  images=len(image_paths), image_manifest_sha256=image_manifest.hexdigest(),
                  scores_path=str(a.scores.resolve()), method='Pooled median at round-half-up mean PJND, all 1008 KonJND pairs')
    with a.out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
