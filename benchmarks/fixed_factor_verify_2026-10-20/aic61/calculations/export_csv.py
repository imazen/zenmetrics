#!/usr/bin/env python3
"""Export retained AIC61 oracle/Rust observations without rescoring images."""
import csv,json,gzip,hashlib,io,sys
from pathlib import Path
if len(sys.argv) != 3:
 raise SystemExit('usage: export_csv.py RAW_DIRECTORY OUTPUT_DIRECTORY')
root=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
rust=[json.loads(l) for l in (root/'rust.jsonl').read_text().splitlines()]
oracle={}
for file in ['pyiqa.tsv','octave.tsv']:
 for r in csv.DictReader((root/file).open(),delimiter='\t'): oracle[r['pair'],r['variant']]=r['value']
assert len(rust)==len(oracle)==2074
records={}
for r in rust:
 pair,v=r['pair'],r['variant']; o=oracle[pair,v]
 if r['returncode']:
  assert o.startswith('error:') and '11x11' in r['stderr'] and v in ['ms-ssim-pyiqa-f32-k5','ms-ssim-pyiqa-f32-k6']
  record=dict(pair=pair,variant=v,ours='',oracle='',abs_delta='',status='matching_size_refusal')
 else:
  a,b=r['value'],float(o)
  record=dict(pair=pair,variant=v,ours=format(a,'.17g'),oracle=format(b,'.17g'),abs_delta=format(abs(a-b),'.17g'),status='scored')
 records[pair,v]=record
families={'ssim-control':['ssim-pyiqa'],'ssim-y':[f'ssim-pyiqa-f32'+('' if k==1 else f'-k{k}') for k in range(1,7)],'ssim-rgb':[f'ssimc-pyiqa-f32'+('' if k==1 else f'-k{k}') for k in range(1,7)],'ms-ssim':[f'ms-ssim-pyiqa-f32'+('' if k==1 else f'-k{k}') for k in range(1,7)]}
for b in ['fsim','fsimc','vsi']:families[b]=[b+s for s in ['', '-k2','-k3','-k4','-nodownscale']]
pairs=sorted({r['pair'] for r in rust}); fields=['ours','oracle','abs_delta','status'];manifest=[]; exported=0
for family,variants in families.items():
 columns=['pair']+[v+'__'+f for v in variants for f in fields];s=io.StringIO(newline='');w=csv.DictWriter(s,columns,lineterminator='\n');w.writeheader()
 for pair in pairs:
  row={'pair':pair}
  for v in variants:
   rec=records[pair,v];exported+=1
   for f in fields:row[v+'__'+f]=rec[f]
  w.writerow(row)
 data=s.getvalue().encode(); assert len(data)<=30000,(family,len(data))
 path=out/(family+'.csv');path.write_bytes(data)
 zipped=gzip.compress(data,compresslevel=9,mtime=0)
 manifest.append(dict(file=path.name,bytes=len(data),gzip_bytes=len(zipped),sha256=hashlib.sha256(data).hexdigest(),variants=len(variants)))
assert exported==2074
s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=['pair','variant','ours','oracle','abs_delta','status'],lineterminator='\n');w.writeheader();w.writerows(records[k] for k in sorted(records))
full=s.getvalue().encode()
manifest=dict(files=manifest,total_csv_bytes=sum(r['bytes'] for r in manifest),total_gzip_bytes=sum(r['gzip_bytes'] for r in manifest),combined_long_csv_bytes=len(full),combined_long_gzip_bytes=len(gzip.compress(full,compresslevel=9,mtime=0)))
(out/'sizes.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
