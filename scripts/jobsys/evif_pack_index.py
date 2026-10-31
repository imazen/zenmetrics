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
- `MANIFEST.json`: counts, one hash-checked member per pack, and the shard hashes.

Only tar headers are read, plus one member per pack; the feature executor hash-checks
every member it reads.
"""
import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import tarfile

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
    for pack in sorted({m['pack'] for m in wanted.values()}):
        pack_sha = objects[pack]
        with tarfile.open(cas(packs, pack_sha)) as tar:
            for info in tar:
                m = wanted.get(info.name)
                if m is None:
                    continue
                if info.size != m['bytes'] or info.name in found:
                    raise ValueError(f'{job_dir.name}: {info.name} size or duplicate')
                found[info.name] = (m['sha256'], pack_sha, info.offset_data, info.size)
        probe = next(v for v in found.values() if v[1] == pack_sha)
        with cas(packs, pack_sha).open('rb') as f:
            f.seek(probe[2])
            if sha(f.read(probe[3])) != probe[0]:
                raise ValueError(f'{job_dir.name}: {pack} member hash differs')
        checked.append(probe[0])
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
                                            evif_shas=sorted({v[0] for v in found.values()}))) + '\n')
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
