#!/usr/bin/env python3
"""Index the decoded pixels of the E36 native ladders (zensimx E36 registration, 2026-10-17) for `evif_features.py`.

usage: e36_pack_index.py STORE_ROOT REFERENCES_JSON REFERENCE_ROOT OUT_DIR [--list FILES] [--max-area N] [--loose-dir DIR]

STORE_ROOT is the directory holding the three E36 archive trees (`e36-bulk-2026-10-17{,-disk1,-cache}`). A bundle is a
plain tar of one encode: `metadata.json`, `encoded/`, `pixels/<sha>.png` (the decoded distorted RGB8 view), features,
maps and diagnostics. Bundles sit loose (`blobs/aa/<sha>.tar`), zstd-compressed (`blobs/aa/<sha>.tar.zst`) or inside
outer tar packs (`packs/<sha>.tar`). REFERENCES_JSON is the generation's `references.json`; REFERENCE_ROOT the
directory its `destination` paths are rebased onto (`references/<size>/<origin>.png`).

Writes to OUT_DIR:
- `index/<aa>.tsv`: `png_sha256 - data_offset size relpath` per decoded view, relpath under STORE_ROOT (the
  executor's ZEN_EVIF_PACKS_DIR). A zstd bundle's PNG is copied loose to `--loose-dir` (under STORE_ROOT) first.
- `cells.parquet`: one row per bundle with its metadata (codec, content, size, q, split, teacher scalars, hashes).
- `selection.jsonl`: per reference view (area below --max-area), its PNG hashes: the `evif_features_spec.py` input.
- `MANIFEST.json`: counts, refusals and shard hashes.
Every tar header is read (no layout is assumed); one PNG per 64 bundles is hash-checked here, and the feature executor
hash-checks every PNG it reads.
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
from compression import zstd
import hashlib
import io
import json
from pathlib import Path
import tarfile

from PIL import Image

META = ('codec', 'content', 'size', 'q', 'split', 'mode_key', 'panel', 'physical_alias', 'family', 'origin', 'width',
        'height', 'butteraugli_pnorm3', 'cvvdp_standard_fhd', 'encoded_sha256', 'encoded_bytes', 'distorted_sha256',
        'distorted_pixels_sha256', 'reference_sha256', 'reference_pixels_sha256', 'generation_view_sha256',
        'native_codec', 'build_commit')


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Window(io.RawIOBase):
    """A seekable read-only view of [start, start+size) of an open file (a bundle inside a pack)."""

    def __init__(self, f, start, size):
        self.f, self.start, self.size, self.pos = f, start, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        n = max(0, min(len(b), self.size - self.pos))
        self.f.seek(self.start + self.pos)
        data = self.f.read(n)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def read_bundle(fileobj, base, check):
    """(metadata, png member offset relative to the file, size, png bytes if checked)."""
    with tarfile.open(fileobj=fileobj, mode='r:') as tar:
        meta = png = None
        for info in tar:
            if info.name == 'metadata.json':
                meta = json.loads(tar.extractfile(info).read())
            elif info.name.startswith('pixels/') and info.name.endswith('.png'):
                png = info
        if meta is None or png is None:
            raise ValueError('bundle without metadata or pixels')
        if png.name != meta['distorted_member']:
            raise ValueError('pixels member differs from metadata')
        data = tar.extractfile(png).read() if check else None
        return meta, base + png.offset_data, png.size, data


def scan(store, rel, loose_dir, counter):
    path = store / rel
    out = []
    check = lambda: next(counter) % 64 == 0
    if rel.endswith('.tar.zst'):
        raw = zstd.decompress(path.read_bytes())
        meta, offset, size, _ = read_bundle(io.BytesIO(raw), 0, False)
        png = raw[offset:offset + size]
        digest = meta['distorted_sha256']
        if sha(png) != digest:
            raise ValueError(f'{rel}: pixels hash differs')
        dest = loose_dir / digest[:2] / digest
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix('.part')
            tmp.write_bytes(png)
            tmp.rename(dest)
        out.append((meta, str(dest.relative_to(store)), 0, size, rel))
        return out
    with path.open('rb') as f:
        if '/packs/' in rel:
            with tarfile.open(fileobj=f, mode='r:') as outer:
                members = [(m.name, m.offset_data, m.size) for m in outer if m.isfile()]
            for name, start, size in members:
                meta, offset, psize, data = read_bundle(Window(f, start, size), start, check())
                if data is not None and sha(data) != meta['distorted_sha256']:
                    raise ValueError(f'{rel}:{name}: pixels hash differs')
                out.append((meta, rel, offset, psize, f'{rel}:{name}'))
        else:
            meta, offset, psize, data = read_bundle(f, 0, check())
            if data is not None and sha(data) != meta['distorted_sha256']:
                raise ValueError(f'{rel}: pixels hash differs')
            out.append((meta, rel, offset, psize, rel))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('store', type=Path)
    ap.add_argument('references_json', type=Path)
    ap.add_argument('reference_root', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--max-area', type=int, default=1_000_000, help='reference views at or above this area are left out')
    ap.add_argument('--loose-dir', default='e36-rev7-features-2026-11-01/loose-png')
    ap.add_argument('--threads', type=int, default=32)
    ap.add_argument('--limit', type=int, default=0, help='index only the first N archive files (a timing probe)')
    ap.add_argument('--list', type=Path, help='archive files, one relpath per line (listing the share over NFS is slow;\n'
                    'list them on the NAS: find <trees>/{blobs,packs} -type f)')
    a = ap.parse_args()
    import itertools
    import threading
    lock = threading.Lock()
    raw_counter = itertools.count()

    class Counter:
        def __next__(self):
            with lock:
                return next(raw_counter)

    counter = Counter()
    (a.out / 'index').mkdir(parents=True, exist_ok=False)
    loose = a.store / a.loose_dir
    rels = sorted(a.list.read_text().split()) if a.list else []
    for tree in [] if a.list else ('e36-bulk-2026-10-17', 'e36-bulk-2026-10-17-disk1', 'e36-bulk-2026-10-17-cache'):
        for sub in ('blobs', 'packs'):
            root = a.store / tree / sub
            if root.is_dir():
                rels += sorted(str(p.relative_to(a.store)) for p in root.rglob('*') if p.is_file())
    if a.limit:
        rels = rels[:a.limit]
    rows, shards = [], collections.defaultdict(set)
    seen_png = {}
    with ThreadPoolExecutor(max_workers=a.threads) as pool:
        for result in pool.map(lambda r: scan(a.store, r, loose, counter), rels):
            for meta, relpath, offset, size, bundle in result:
                digest = meta['distorted_sha256']
                rows.append({k: meta.get(k) for k in META} | dict(bundle=bundle))
                if digest not in seen_png:
                    seen_png[digest] = (relpath, offset, size)
                    shards[digest[:2]].add(f'{digest}\t-\t{offset}\t{size}\t{relpath}')
    import pyarrow as pa
    import pyarrow.parquet as pq
    for r in rows:
        r['mode_key'] = r['mode_key'] if isinstance(r['mode_key'], str) else json.dumps(r['mode_key'])
    pq.write_table(pa.Table.from_pylist(rows), a.out / 'cells.parquet', compression='zstd')
    shard_shas = {}
    for prefix, lines in sorted(shards.items()):
        data = ('\n'.join(sorted(lines)) + '\n').encode()
        (a.out / 'index' / f'{prefix}.tsv').write_bytes(data)
        shard_shas[prefix] = sha(data)
    # References: the generation's records, rebased onto REFERENCE_ROOT; RGB8 pixel hash as the extractor computes it.
    refs = {}
    for r in json.loads(a.references_json.read_text())['rows']:
        refs[r['sha256']] = r
    per_ref = collections.defaultdict(dict)
    for r in rows:
        per_ref[r['reference_sha256']][r['distorted_sha256']] = None
    refused = collections.Counter()
    sources = {}
    with (a.out / 'selection.jsonl').open('w') as sel:
        for ref_sha in sorted(per_ref):
            r = refs.get(ref_sha)
            if r is None:
                refused['reference not in references.json'] += len(per_ref[ref_sha])
                continue
            if r['width'] * r['height'] >= a.max_area:
                refused['reference area at or above max'] += len(per_ref[ref_sha])
                continue
            path = a.reference_root / str(r['size']) / Path(r['destination']).name
            raw = path.read_bytes()
            if sha(raw) != ref_sha:
                raise ValueError(f'{path}: reference bytes differ')
            image = Image.open(path)
            src = dict(image_path=str(path), source_sha256=ref_sha, pixels_sha256=sha(image.convert('RGB').tobytes()),
                       identity=json.dumps(dict(content=r['content'], family=r['family'], origin=r['origin'],
                                                reference_key=r['reference_key'], size=r['size'], split=r['split']),
                                           sort_keys=True),
                       width=image.width, height=image.height)
            sources[ref_sha] = src
            sel.write(json.dumps(dict(job=r['reference_key'], view=r['reference_key'], source=src,
                                      evif_shas=sorted(per_ref[ref_sha]))) + '\n')
    manifest = dict(store=str(a.store), archive_files=len(rels), bundles=len(rows), unique_pngs=len(seen_png),
                    references=len(sources), selected_pngs=sum(len(per_ref[s]) for s in sources), refused=dict(refused),
                    max_area=a.max_area, selection_sha256=sha((a.out / 'selection.jsonl').read_bytes()),
                    cells_sha256=sha((a.out / 'cells.parquet').read_bytes()), shards=shard_shas)
    (a.out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'shards'}))


if __name__ == '__main__':
    main()
