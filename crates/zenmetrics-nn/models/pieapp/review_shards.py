"""Partition unchanged PieAPP goldens by source for bounded lock holds."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('models',type=Path)
p.add_argument('out',type=Path)
a=p.parse_args()
original=(a.models/'pieapp_goldens.tsv').read_bytes()
assert original == Path('benchmarks/pieapp_2026-10-16_goldens.tsv').read_bytes()
lines=original.decode().splitlines(keepends=True)
groups=defaultdict(list)
for row in lines[1:]:groups[row.split('\t')[0]].append(row)
assert len(lines)==25 and len(groups)==3
assert sorted(row for rows in groups.values() for row in rows)==sorted(lines[1:])
a.out.mkdir(parents=True,exist_ok=False)
manifest={'original_sha256':hashlib.sha256(original).hexdigest(),'total_pairs':24,'shards':{}}
for source,rows in sorted(groups.items()):
    directory=a.out/source;directory.mkdir()
    for path in a.models.iterdir():
        if path.is_file() and path.name!='pieapp_goldens.tsv':(directory/path.name).symlink_to(path.resolve())
    payload=lines[0]+''.join(rows)
    (directory/'pieapp_goldens.tsv').write_text(payload)
    manifest['shards'][source]={'pairs':len(rows),'sha256':hashlib.sha256(payload.encode()).hexdigest(),'keys':[row.split('\t')[:2] for row in rows]}
(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
