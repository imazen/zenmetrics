"""Partition complete measured KonJND pairs without changing any row."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__);p.add_argument('pairs',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
text=a.pairs.read_text();lines=text.splitlines(keepends=True)
assert len(lines)==1009
rows=list(csv.DictReader(lines,delimiter='\t'))
assert len({(r['ref_path'],r['dist_path']) for r in rows})==1008
a.out.mkdir(parents=True,exist_ok=False)
manifest={'pairs':1008,'original_sha256':hashlib.sha256(a.pairs.read_bytes()).hexdigest(),'shards':[]}
covered=[]
for index in range(4):
    subset=lines[1+252*index:1+252*(index+1)];assert len(subset)==252;covered+=subset
    path=a.out/f'pairs-{index}.tsv';path.write_text(lines[0]+''.join(subset))
    manifest['shards'].append(dict(index=index,pairs=252,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
assert covered==lines[1:]
(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
