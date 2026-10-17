#!/usr/bin/env python3
"""Attach fresh exact-pair CPU comparisons without changing saved measurements."""
import argparse,copy,hashlib,json,pathlib
from references import load_references
from parity import qualify_rows, reproduced_residual
P=pathlib.Path

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw',type=P);p.add_argument('inventory',type=P)
    p.add_argument('--reference-jsonl',action='append',type=P,required=True)
    p.add_argument('--output',type=P,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.meta.json').exists():
        raise ValueError('Preserve earlier results and choose a new output')
    reference_receipts=[dict(path=str(path),sha256=digest(path)) for path in a.reference_jsonl]
    references=load_references(a.reference_jsonl)
    if reference_receipts!=[dict(path=str(path),sha256=digest(path)) for path in a.reference_jsonl]:
        raise ValueError('Reference ledger changed during reading; retry with a stable snapshot')
    raw_bytes=a.raw.read_bytes();raw_digest=hashlib.sha256(raw_bytes).hexdigest()
    inventory=json.loads(a.inventory.read_text());scales={}
    scale_path=P(__file__).resolve().parents[2]/'crates/zenmetrics-nn/models/jnd_scale.tsv'
    for line in scale_path.read_text().splitlines():
        if not line or line.startswith('#') or line.startswith('column\t'):continue
        column,value,*_=line.split('\t');scales[column]=float(value)
    rows=[];reproductions=[]
    for line in raw_bytes.decode().splitlines():
        original=json.loads(line);row=copy.deepcopy(original)
        if row['impl']!='zenmetrics' or row['device']=='cpu':
            raise ValueError('This receipt handles zenmetrics GPU observations only')
        mismatch=qualify_rows([row],references,inventory,scales)
        if mismatch:
            row['reproduction_required']=not reproduced_residual(row)
            if row['reproduction_required']:
                reproductions.append({k:row[k] for k in ['metric_name','size_name','threads','mode','pair_sha256']})
        for key in ['n','median_ns','mean_ns','p10_ns','p90_ns','score','load_before','load_after','resource']:
            if row.get(key)!=original.get(key):raise AssertionError('Comparison changed measurement: '+key)
        row['comparison_input_sha256']=raw_digest
        rows.append(row)
    with a.output.open('x') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    receipt=dict(input=str(a.raw),input_sha256=raw_digest,references=reference_receipts,
                 inventory_sha256=digest(a.inventory),jnd_scale_sha256=digest(scale_path),
                 output_sha256=digest(a.output),rows=len(rows),reproduction_required=reproductions,
                 policy='Only comparison fields updated; no timed calls executed; new mismatches require independent reproduction')
    a.output.with_suffix('.meta.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
