#!/usr/bin/env python3
"""Create a new immutable subset bundle without rebuilding or changing the source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from mac_service import verify


def select(cells, metrics=None, sizes=None, threads=None):
    chosen = cells
    for key, requested in [('metric_name', metrics), ('size_name', sizes), ('threads', threads)]:
        if requested is not None:
            missing = set(requested) - {c[key] for c in cells}
            if missing:
                raise ValueError(f'unknown {key}: {sorted(missing)}')
            chosen = [c for c in chosen if c[key] in requested]
    if not chosen:
        raise ValueError('selection has no cells')
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    p.add_argument('--metrics')
    p.add_argument('--sizes')
    p.add_argument('--threads')
    a = p.parse_args()
    source, destination = a.source.resolve(), a.destination.resolve()
    if destination.is_relative_to(source):
        p.error('destination must be outside source bundle')
    verify(source)
    raw = (source / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    manifest['cells'] = select(manifest['cells'], a.metrics.split(',') if a.metrics else None,
                               a.sizes.split(',') if a.sizes else None,
                               list(map(int, a.threads.split(','))) if a.threads else None)
    manifest['metadata']['subset_of_manifest_sha256'] = hashlib.sha256(raw).hexdigest()
    for key, field in [('metrics', 'metric_name'), ('sizes', 'size_name'), ('threads', 'threads')]:
        manifest['metadata'][key] = list(dict.fromkeys(c[field] for c in manifest['cells']))
    shutil.copytree(source, destination)
    verify(destination)
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    hashes = json.loads((destination / 'SHA256SUMS.json').read_text())
    hashes['manifest.json'] = hashlib.sha256((destination / 'manifest.json').read_bytes()).hexdigest()
    (destination / 'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2) + '\n')
    verify(destination)
    print(json.dumps(dict(bundle=str(destination), cells=len(manifest['cells']), manifest_sha256=hashes['manifest.json'])))


if __name__ == '__main__':
    main()
