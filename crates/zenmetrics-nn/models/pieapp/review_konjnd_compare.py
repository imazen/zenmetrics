"""Compare all corrected-graph scores to the preserved original calibration."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('baseline',type=Path);p.add_argument('directory',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
reference=json.loads(Path('benchmarks/pieapp_2026-10-16_konjnd.json').read_text())
assert hashlib.sha256(a.baseline.read_bytes()).hexdigest()==reference['scores_sha256']
def load(path):
    with path.open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
    result={(r['ref_path'],r['dist_path']):float(r['pieapp']) for r in rows}
    assert len(rows)==len(result) and all(math.isfinite(v) for v in result.values())
    return result
old=load(a.baseline);new={};hashes={}
for index in range(4):
    path=a.directory/f'scores-{index}.tsv';part=load(path)
    assert len(part)==252 and not (new.keys()&part.keys())
    new.update(part);hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
assert len(old)==len(new)==1008 and old.keys()==new.keys()
image_root=hashlib.sha256()
for name in sorted({p for pair in new for p in pair}):
    digest=hashlib.sha256()
    with Path(name).open('rb') as image:
        for chunk in iter(lambda:image.read(1024*1024),b''):digest.update(chunk)
    image_root.update((name+'\t'+digest.hexdigest()+'\n').encode())
assert image_root.hexdigest()==reference['image_manifest_sha256']
bits=lambda v:struct.pack('<d',v)
deltas=[abs(new[k]-old[k]) for k in old]
receipt=dict(image_manifest_sha256=image_root.hexdigest(),pairs=1008,bit_identical_pairs=sum(bits(old[k])==bits(new[k]) for k in old),baseline_median=statistics.median(old.values()),corrected_graph_median=statistics.median(new.values()),maximum_absolute_change=max(deltas),median_absolute_change=statistics.median(deltas),maximum_change_in_fixed_jnd=max(deltas)/reference['jnd'],fixed_jnd=reference['jnd'],jnd_gate_changed=False,baseline_sha256=reference['scores_sha256'],corrected_score_files=hashes)
with a.out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2))
