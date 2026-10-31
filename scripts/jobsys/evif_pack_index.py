#!/usr/bin/env python3
"""Index the EVIF members of a chroma-trade run's tar packs for `evif_features.py`.

usage: evif_pack_index.py ASSEMBLED_DIR PACK_CAS_DIR OUT_DIR

ASSEMBLED_DIR holds one directory per trade job with its JOB.json and SPEC.json (the
run's `*-assembled` mirror); PACK_CAS_DIR is the run's CAS (`aa/bb/<sha>`), holding each
job's result object and its tar packs. Writes to OUT_DIR:

- `index/<aa>.tsv`: `member_sha256 pack_sha256 data_offset size` for every `.evif`
  member, sharded by the member hash prefix (`ZEN_EVIF_PACK_INDEX`).
- `selection.jsonl`: one line per trade job with its view reference (path, file and RGB8
  pixel hashes, identity) and its EVIF member hashes, the input of a features spec.
- `MANIFEST.json`: counts, the number of packs probed, and the shard hashes.

No tar header is read: offsets follow from the member list (see index_job); one member
per pack is hash-checked here and the feature executor hash-checks every member it
reads. A job's members are listed in pack order, so a feature job reads a contiguous range.
"""
import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image


def sha(data):
    return hashlib.sha256(data).hexdigest()


def cas(root, digest):
    return root / digest[:2] / digest[2:4] / digest


def index_job(job_dir, packs):
    job = json.loads((job_dir / 'JOB.json').read_text())
    result = json.loads(cas(packs, job['output_sha']).read_bytes())
    objects = {o['path']: o['sha256'] for o in result['objects']}
    wanted = {m['name']: m for m in result['members'] if m['kind'] == 'evif'}
    found = {}
    checked = []
    # Packs are plain ustar written in member-list order: a 512-byte header, then the data
    # padded to 512. Offsets follow from the sizes, so no header is read over NFS. The pack's
    # size (end + two zero blocks, rounded to a 10240-byte record) and a hash of its LAST
    # member check the layout: any extra header block would shift that member.
    for pack in sorted({m['pack'] for m in wanted.values()}):
        pack_sha = objects[pack]
        offset = 0
        last = None
        for m in result['members']:
            if m['pack'] != pack:
                continue
            offset += 512
            last = (m['sha256'], pack_sha, offset, m['bytes'])
            if m['name'] in wanted:
                if m['name'] in found:
                    raise ValueError(f'{job_dir.name}: duplicate member {m["name"]}')
                found[m['name']] = last
            offset += (m['bytes'] + 511) // 512 * 512
        path = cas(packs, pack_sha)
        if path.stat().st_size != -(-(offset + 1024) // 10240) * 10240:
            raise ValueError(f'{job_dir.name}: {pack} size does not match a plain ustar layout')
        with path.open('rb') as f:
            f.seek(last[2])
            if sha(f.read(last[3])) != last[0]:
                raise ValueError(f'{job_dir.name}: {pack} last member hash differs')
        checked.append(last[0])
    if found.keys() != wanted.keys():
        raise ValueError(f'{job_dir.name}: {len(wanted) - len(found)} members not in packs')
    spec = json.loads((job_dir / 'SPEC.json').read_text())
    (view,) = [v for v in spec['views'] if v['id'] == result['view']]
    return job, view, found, checked


def reference(view):
    raw = Path(view['path']).read_bytes()
    if sha(raw) != view['sha256']:
        raise ValueError(f'{view["id"]}: reference bytes differ from the spec')
    image = Image.open(view['path'])
    return dict(image_path=view['path'], source_sha256=view['sha256'],
                pixels_sha256=sha(image.convert('RGB').tobytes()),
                identity=json.dumps(dict(content_class=view['content_class'], id=view['id'],
                                         mlic_role=view['role'], reference_group=view['source'],
                                         type=view['class']), sort_keys=True),
                width=image.width, height=image.height)


def main():
    assembled, packs, out = map(Path, sys.argv[1:4])
    (out / 'index').mkdir(parents=True, exist_ok=False)
    shards = collections.defaultdict(list)
    views = {}
    jobs = members = 0
    checked = []
    job_dirs = sorted(p for p in assembled.iterdir() if p.is_dir())
    # Reading tar headers over NFS is latency-bound; jobs are independent.
    with ThreadPoolExecutor(max_workers=16) as pool, (out / 'selection.jsonl').open('w') as selection:
        for job_dir, (job, view, found, probes) in zip(job_dirs, pool.map(lambda d: index_job(d, packs), job_dirs)):
            checked += probes
            for digest, pack_sha, offset, size in found.values():
                shards[digest[:2]].append(f'{digest}\t{pack_sha}\t{offset}\t{size}')
            if view['id'] not in views:
                views[view['id']] = reference(view)
            selection.write(json.dumps(dict(job=job_dir.name, job_id=job['job_id'], view=view['id'],
                                            source=views[view['id']],
                                            evif_shas=list(dict.fromkeys(
                                                v[0] for v in sorted(found.values(), key=lambda v: (v[1], v[2])))))) + '\n')
            jobs += 1
            members += len(found)
    shard_shas = {}
    for prefix, lines in sorted(shards.items()):
        data = ('\n'.join(sorted(set(lines))) + '\n').encode()
        (out / 'index' / f'{prefix}.tsv').write_bytes(data)
        shard_shas[prefix] = sha(data)
    unique = len({line.split('\t')[0] for v in shards.values() for line in v})
    manifest = dict(assembled=str(assembled), packs=str(packs), jobs=jobs, views=len(views),
                    members=members, unique_members=unique, packs_checked=len(checked),
                    selection_sha256=sha((out / 'selection.jsonl').read_bytes()), shards=shard_shas)
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'shards'}))


if __name__ == '__main__':
    main()
