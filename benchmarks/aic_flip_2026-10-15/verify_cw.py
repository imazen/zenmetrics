"""Compare Rust with the independent pyrtools panel and published CW-SSIM."""
import csv,json,pathlib,subprocess,sys
import numpy as np
from PIL import Image
scratch=pathlib.Path.home()/'tmp/aic-variants-scratch'
expected={r['key']:r for r in json.load(open(sys.argv[1]))}
rows=[]
for pair in csv.DictReader(open(pathlib.Path.home()/'tmp/aic-audit/pairs.tsv'),delimiter='\t'):
    r=np.asarray(Image.open(pair['ref_path']).convert('RGB'));d=np.asarray(Image.open(pair['dist_path']).convert('RGB'))
    r.tofile(scratch/'r.rgb');d.tofile(scratch/'d.rgb')
    ours=float(subprocess.check_output([str(scratch/'cw-probe'),str(r.shape[1]),str(r.shape[0]),str(scratch/'r.rgb'),str(scratch/'d.rgb')]))
    oracle=expected[pair['distorted']][sys.argv[3]];pub=expected[pair['distorted']]['published']
    row=dict(key=pair['distorted'],ours=ours,oracle=oracle,published=pub,oracle_delta=abs(ours-oracle),published_delta=abs(ours-pub))
    rows.append(row);print(json.dumps(row),flush=True)
pathlib.Path(sys.argv[2]).open('x').write(json.dumps(rows,indent=2)+'\n')
print('SUMMARY',json.dumps({k:dict(median=float(np.median([r[k] for r in rows])),max=max(r[k] for r in rows)) for k in ['oracle_delta','published_delta']}),flush=True)
assert max(r['oracle_delta'] for r in rows)<1e-9
