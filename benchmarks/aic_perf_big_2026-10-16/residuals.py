"""Record common residual bits after the complete baseline/candidate bit gate.

The input table is written only after all independent binary comparisons pass.
Equal score bits passed to unchanged CSV/mapping arithmetic yield equal residual
bits; this tool records those common values rather than simulating two scores.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct

p=argparse.ArgumentParser(description=__doc__)
for name in ['parity','pairs','published','config','output']:p.add_argument(name,type=Path)
a=p.parse_args()
rows=list(csv.DictReader(a.parity.open(),delimiter='\t'))
pairs=list(csv.DictReader(a.pairs.open(),delimiter='\t'))
published={r['distorted']:r for r in csv.DictReader(a.published.open())}
config=json.loads(a.config.read_text())['mappings']
assert len(rows)==122 and len(pairs)==61
keys={r['distorted'] for r in pairs}
assert len(keys)==61 and {r['key'] for r in rows}==keys
result=[]
for metric,column,kind in [('butter','Butteraugli-original','f'),('cambi','CAMBI','d')]:
    subset=[r for r in rows if r['metric']==metric]
    assert len(subset)==61 and {r['key'] for r in subset}==keys
    mapping=config[column]
    def jnd(x):return mapping['a']*max(0,x-mapping['b'])**mapping['c']
    for row in subset:
        score=struct.unpack('>'+kind,bytes.fromhex(row['score_bits']))[0]
        target=float(published[row['key']][column])
        bits=lambda x:struct.pack('>d',x).hex()
        result.append(dict(key=row['key'],metric=metric,common_score_bits=row['score_bits'],common_native_residual_bits=bits(score-target),common_jnd_residual_bits=bits(jnd(score)-jnd(target))))
with a.output.with_suffix('.tsv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(result[0]),delimiter='\t');writer.writeheader();writer.writerows(result)
manifest=hashlib.sha256()
images=sorted({r[k] for r in pairs for k in ['ref_path','dist_path']})
for name in images:
    digest=hashlib.sha256()
    with Path(name).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    manifest.update((name+'\t'+digest.hexdigest()+'\n').encode())
receipt=dict(pairs_per_metric=61,metrics=['butter','cambi'],numeric_residual_rows=len(result),residuals_unchanged_by_exact_score_identity=True,common_residual_table_sha256=hashlib.sha256(a.output.with_suffix('.tsv').read_bytes()).hexdigest(),input_images=len(images),input_image_manifest_sha256=manifest.hexdigest(),input_files={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.parity,a.pairs,a.published,a.config]})
with a.output.with_suffix('.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2))
