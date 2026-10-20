#!/usr/bin/env python3
"""Combine explicit reference plans for the 27 changed routes, in retained order."""
import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess

from compare_canonical import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--threads',default='1,8')
    a=p.parse_args();manifest=json.loads(a.manifest.read_text())
    here=Path(__file__).resolve().parent
    order=here.parent/'rebenchmark_since_f7668c7/order.tsv'
    with order.open() as stream:
        names=[row['metric'] for row in csv.DictReader((line for line in stream if not line.startswith('#')),delimiter='\t')]
    if set(names)!=set(manifest['metric_plans']):
        raise ValueError('reference manifest must name exactly the changed routes')
    binary=Path(manifest['binary']).resolve(strict=True)
    inventory=json.loads(subprocess.check_output([str(binary),'inventory']))
    if not set(names)<={row['metric_name'] for row in inventory if row['compiled']}:
        raise ValueError('changed routes are missing from the supplied worker')
    bindings={str(path.resolve()):sha(path) for path in [a.manifest,binary,order,Path(__file__),
        here/'compare_canonical.py',here/'compare_canonical_samples.py',here/'compare_canonical_budgeted.py']}
    plans={};cases=[]
    for name in names:
        path=Path(manifest['metric_plans'][name]).resolve(strict=True)
        if path not in plans:
            plan=json.loads(path.read_text())
            for asset,expected in plan['bindings'].items():
                if sha(asset)!=expected:raise ValueError('reference binding changed: '+asset)
                if asset in bindings and bindings[asset]!=expected:raise ValueError('incompatible source binding')
                bindings[asset]=expected
            bindings[str(path)]=sha(path);plans[path]=plan
        candidates=[row for row in plans[path]['cases'] if row['metric']==name and row['matched_scope']]
        if not candidates:raise ValueError('no matched reference case: '+name)
        original=min(candidates,key=lambda row:row['threads'])
        for threads in map(int,a.threads.split(',')):
            row=copy.deepcopy(original);row['threads']=threads
            row['id']=f'{name}-changed-{row["width"]}x{row["height"]}-t{threads}'
            cases.append(row)
    with a.output.open('x') as stream:
        json.dump(dict(schema=1,binary=str(binary),rust_source=manifest['rust_source'],bindings=bindings,cases=cases,
            scheduling='Ascending retained historical duration; sizes and exact source contracts remain per case',
            ratio_definition='Rust/canonical; descriptive alternating-block medians'),stream,indent=2)
        stream.write('\n')
    print(json.dumps(dict(metrics=len(names),cases=len(cases),plan=str(a.output))))


if __name__=='__main__':main()
