#!/usr/bin/env python3
"""Materialize a closed, hash-bound candidate archive without changing sources.

The explicit plan lists every source, archive-relative path and expected SHA.
Use the canonical dataset mirror separately for NAS copying and readback.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def snapshot(plan, output):
    value = json.loads(plan.read_text())
    if value.get('schema') != 'byc-candidate-archive-plan-v1' or not value.get('files'):
        raise ValueError('an explicit nonempty candidate file plan is required')
    if output.exists():
        raise FileExistsError('preserve the previous archive attempt')
    seen = set()
    for row in value['files']:
        rel = Path(row['relative_path'])
        if (str(rel) == '.' or rel.is_absolute() or '..' in rel.parts or str(rel) in seen
                or str(rel) in ('SOURCE_INDEX.json','ARCHIVE_PLAN.json')):
            raise ValueError('archive paths must be unique, relative and contained')
        seen.add(str(rel))
        source = Path(row['source']).resolve(strict=True)
        if not source.is_file() or digest(source) != row['sha256']:
            raise ValueError(f'preserved source differs: {row["source"]}')
        if source.is_relative_to(output.resolve()):
            raise ValueError('archive must not contain its own output')
    output.mkdir()
    total = 0
    for row in value['files']:
        source = Path(row['source']).resolve(strict=True)
        target = output/row['relative_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if target.is_symlink() or digest(target) != row['sha256']:
            raise ValueError('archive bytes changed during materialization')
        total += target.stat().st_size
    shutil.copyfile(plan, output/'ARCHIVE_PLAN.json')
    result = dict(schema='byc-candidate-snapshot-v1',
                  observed_at=datetime.now(timezone.utc).isoformat(),
                  plan_sha256=digest(plan), files=len(value['files']), bytes=total,
                  sources_preserved=True, source_inventory=value['files'])
    with (output/'SOURCE_INDEX.json').open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    result=snapshot(a.plan,a.out)
    print(json.dumps({k:v for k,v in result.items() if k!='source_inventory'}))


if __name__ == '__main__':
    main()
