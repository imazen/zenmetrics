#!/usr/bin/env python3
"""Hydrate pinned canonical training PNGs; emit inputs for canonical `declare`.

This ingests sources only. Scheduling, retries of encode jobs, claims and
ledgers remain owned by zenfleet. No validation/test sources are downloaded.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/picker'))
from origin_split import split_of

PIN = '187fbf338ce08e8e6654db7f04ddae58d5263da2'
REFERENCE = 'svt-mainline-4.2.0-9292ec8e32bce26f781f277ec8739b53426c4300'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sources(canonical):
    provenance = json.loads((canonical / 'provenance.json').read_text())
    if provenance['commit'] != PIN:
        raise ValueError('unexpected canonical source revision')
    hashes = {row['path']: row['sha256'] for row in provenance['files']}
    name = 'manifests/train.tsv'
    if sha(canonical / name) != hashes[name]:
        raise ValueError('training manifest hash mismatch')
    with (canonical / name).open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    ids = set()
    for row in rows:
        if row['split'] != 'train' or split_of(row['id']) != 'train':
            raise ValueError(f"held-out origin in training manifest: {row['id']}")
        if row['id'] in ids:
            raise ValueError('duplicate origin')
        ids.add(row['id'])
    return rows


def arms():
    result = []
    # Broad quality anchors; refine uncertain curves after the population scout.
    for qp in (5, 16, 27, 38, 49, 60):
        for speed in (-1, 0, 4, 8, 13):
            result.append(dict(backend='zenav1-svt', speed=speed,
                               quantizer=qp, svt_reference=REFERENCE))
        for tool in ('zen_intra_edge_filter', 'zen_restoration_unit_search'):
            result.append(dict(backend='zenav1-svt', speed=-1,
                               quantizer=qp, svt_reference=REFERENCE, **{tool: True}))
        for backend in ('libaom', 'zenav1-aom'):
            for speed in (0, 4, 8):
                result.append(dict(backend=backend, speed=speed, quantizer=qp))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('canonical', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--hydrate', action='store_true')
    args = parser.parse_args()
    rows = sources(args.canonical)
    args.output.mkdir(parents=True, exist_ok=True)
    corpus = args.output / 'corpus'
    corpus.mkdir(exist_ok=True)
    missing = [r['id'] for r in rows if not r['png_v3_sdr_url']]
    plan = dict(protocol='imazen26-av1-training-scout-v1', canonical_commit=PIN,
                origins=len(rows), sdr_sources=len(rows)-len(missing),
                missing_sdr_origins=missing, max_edges=[512], repeats=3,
                arms=arms(), held_out_used=False,
                limitations=['512px SDR 8-bit 420 scout only; not full size/format coverage',
                             'native HDR, alpha, lossless and rav1e need separate strata',
                             'not a minimum representative set or calibrated routing policy'])
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(f"training origins={len(rows)} SDR={plan['sdr_sources']} missing={missing}; "
          f"{len(plan['arms'])} cells/origin, 3 rounds; hydration={args.hydrate}", flush=True)
    if not args.hydrate:
        return
    # Checkpoints survive interruption. Their hashes detect changed local caches.
    with (args.output / 'sources.jsonl').open('w') as records, \
            (args.output / 'declare-input.jsonl').open('w') as declarations:
        for index, row in enumerate(rows, 1):
            url = row['png_v3_sdr_url']
            if not url:
                continue
            if not url.startswith('https://codec-corpus.r2.imazen.org/imazen-26-png-v3/'):
                raise ValueError('unregistered source URL')
            dest = corpus / f"{row['id']}.sdr.png"
            checkpoint = corpus / f"{row['id']}.json"
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                if saved['url'] != url or saved['sha256'] != sha(dest):
                    raise ValueError(f'cached source changed: {dest}')
            else:
                partial = dest.with_suffix('.partial')
                subprocess.run(['curl', '--fail', '--location', '--silent',
                                '--show-error', '--retry', '3', '--max-time', '120',
                                '--output', str(partial), url], check=True)
                with partial.open('rb') as stream:
                    if stream.read(8) != b'\x89PNG\r\n\x1a\n':
                        raise ValueError(f'not PNG: {url}')
                partial.replace(dest)
                saved = dict(origin=row['id'], split='train', url=url,
                             sha256=sha(dest), bytes=dest.stat().st_size,
                             canonical_commit=PIN, raw_source_sha256=row['sha256'])
                checkpoint.write_text(json.dumps(saved)+'\n')
            records.write(json.dumps(saved)+'\n'); records.flush()
            knobs = dict(max_edges=plan['max_edges'], repeats=3, arms=plan['arms'])
            item = dict(image_path=dest.name, source_sha=saved['sha256'],
                        codec='av1-compare', q=0, knob_tuple_json=json.dumps(knobs))
            declarations.write(json.dumps(item)+'\n'); declarations.flush()
            print(f"{index}/{len(rows)} hydrated {row['id']} {saved['bytes']} bytes", flush=True)
    (args.output / 'hydration-complete.json').write_text(json.dumps(plan)+'\n')


if __name__ == '__main__':
    main()
