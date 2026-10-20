#!/usr/bin/env python3
"""Verify explicitly selected files copied from an existing frozen model tree."""
import argparse
import json
from pathlib import Path

from compare_canonical import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--omit-full-graphs', action='store_true')
    p.add_argument('--omit-directory', action='append', default=[])
    a = p.parse_args()
    source = a.root/'locations.json'
    old = json.loads(source.read_text())
    artifacts, omitted = [], []
    for record in old['artifacts']:
        parent = Path(record['destination']).parent.name
        name = Path(record['destination']).name
        if parent in a.omit_directory or (a.omit_full_graphs and name.endswith('_full.onnx') and name != 'maniqa_full.onnx'):
            omitted.append(dict(directory=parent, file=name))
            continue
        path = a.root/parent/name
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('copied model differs from frozen manifest: '+str(path))
        artifacts.append(dict(path=str(path.resolve()), sha256=record['sha256'], bytes=record['bytes']))
    if not artifacts:
        p.error('no verified artifacts')
    with a.output.open('x') as stream:
        json.dump(dict(source_locations_sha256=sha(source), files=len(artifacts),
                       bytes=sum(r['bytes'] for r in artifacts), artifacts=artifacts,
                       explicitly_omitted=omitted), stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(verified=len(artifacts), omitted=len(omitted))))


if __name__ == '__main__':
    main()
