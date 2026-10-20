#!/usr/bin/env python3
"""Freeze explicit reference contracts in a declared retained-duration order.

The historical 27-route preparer stays unchanged for its frozen plans. This
successor accepts the scheduling table in the manifest, including later changes.
"""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import subprocess

from compare_canonical import sha


def prepare(manifest_path, output, threads):
    manifest_path=manifest_path.resolve(strict=True)
    manifest=json.loads(manifest_path.read_text())
    here=Path(__file__).resolve().parent
    order=Path(manifest.get('order',here.parent/'rebenchmark_since_f7668c7/order.tsv')).resolve(strict=True)
    with order.open() as stream:
        rows=list(csv.DictReader((line for line in stream if not line.startswith('#')),delimiter='\t'))
    names=[row['metric'] for row in rows]
    times=[float(row['historical_ms']) for row in rows]
    if not names or len(set(names))!=len(names) or any(not math.isfinite(t) or t<=0 for t in times) or times!=sorted(times):
        raise ValueError('unique routes in ascending positive retained-duration order required')
    if set(names)!=set(manifest['metric_plans']):
        raise ValueError('reference manifest must match the declared scheduling scope')
    if not threads or len(set(threads))!=len(threads) or any(t<1 for t in threads):
        raise ValueError('distinct positive thread counts required')
    binary=Path(manifest['binary']).resolve(strict=True)
    inventory=json.loads(subprocess.check_output([str(binary),'inventory']))
    if not set(names)<={row['metric_name'] for row in inventory if row['compiled']}:
        raise ValueError('requested routes are missing from the supplied native worker')
    assets=[manifest_path,binary,order,Path(__file__),here/'compare_canonical.py',
            here/'compare_canonical_samples.py',here/'compare_canonical_budgeted.py']
    if 'build_receipt' in manifest:assets.append(Path(manifest['build_receipt']).resolve(strict=True))
    bindings={str(path.resolve()):sha(path) for path in assets}
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
        for count in threads:
            row=copy.deepcopy(original);row['threads']=count
            row['id']=f'{name}-changed-{row["width"]}x{row["height"]}-t{count}'
            cases.append(row)
    with output.open('x') as stream:
        json.dump(dict(schema=1,binary=str(binary),rust_source=manifest['rust_source'],bindings=bindings,cases=cases,
            scheduling='Ascending retained historical duration; sizes and source contracts remain per case',
            ratio_definition='Rust/canonical; descriptive alternating-block medians'),stream,indent=2)
        stream.write('\n')
    return len(names),len(cases)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--threads',default='1,8')
    a=p.parse_args();metrics,cases=prepare(a.manifest,a.output,list(map(int,a.threads.split(','))))
    print(json.dumps(dict(metrics=metrics,cases=cases,plan=str(a.output))))


if __name__=='__main__':main()
