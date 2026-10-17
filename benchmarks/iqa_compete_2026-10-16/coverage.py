#!/usr/bin/env python3
"""Audit requested Rust cells against the frozen inventory; never infer completion."""
import argparse,collections,hashlib,itertools,json,pathlib
from run import LEARNED
P=pathlib.Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw',type=P);p.add_argument('inventory',type=P);p.add_argument('pairs',type=P);p.add_argument('output',type=P)
    p.add_argument('--sizes',default='cid22,2k');p.add_argument('--threads',default='1,20')
    p.add_argument('--gpu-runtime',choices=['cuda','wgpu']);p.add_argument('--nn-cuda',action='store_true')
    p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    assert not a.output.exists(),'Preserve earlier coverage receipts'
    raw=a.raw.read_bytes();inventory=json.loads(a.inventory.read_text());pairs=json.loads(a.pairs.read_text())
    sizes=a.sizes.split(',');threads=list(map(int,a.threads.split(',')))
    if not set(sizes)<={r['size_name'] for r in pairs}:raise ValueError('Unknown requested size')
    if not threads or any(n<1 for n in threads):raise ValueError('Positive thread budgets required')
    geometries={r['size_name']:(r['width'],r['height']) for r in pairs}
    expected=set()
    for metric in inventory:
        name=metric['metric_name'];devices=[]
        if not metric['gpu']:devices.append('cpu')
        elif a.gpu_runtime:devices.append(a.gpu_runtime)
        if name in LEARNED and a.nn_cuda:devices.append('cuda')
        for device,size,count,mode in itertools.product(devices,sizes,threads,['score','score+map']):
            expected.add((name,device,size,count,mode))
    actual={};testbeds=set();pair_hashes=collections.defaultdict(set)
    for line in raw.decode().splitlines():
        row=json.loads(line)
        if row['impl']!='zenmetrics':continue
        testbeds.add((row['host'],row['os']))
        if len(testbeds)>1:raise ValueError('Audit one host/OS testbed per invocation')
        key=tuple(row[k] for k in ['metric_name','device','size_name','threads','mode'])
        if key in actual:raise ValueError('Collect duplicate attempts before coverage checking: '+str(key))
        if row['size_name'] in sizes:
            if (row['width'],row['height'])!=geometries[row['size_name']]:raise ValueError('Cell geometry differs from requested input')
            pair_hashes[row['size_name']].add(row['pair_sha256'])
        actual[key]=row
    if any(len(values)!=1 for values in pair_hashes.values()):raise ValueError('A size label contains multiple input pairs')
    missing=sorted(expected-actual.keys());pending=[];counts=collections.Counter()
    for key in sorted(expected&actual.keys()):
        row=actual[key];status=row['status'];counts[status.split(':',1)[0]]+=1
        reason=None
        if status in ['ok','variant_mismatch']:
            if row.get('n',0)<10:reason='fewer than ten actual Rust rounds'
            elif row.get('reproduction_required'):reason='GPU mismatch reproduction pending'
            elif key[1]!='cpu' and row.get('ref_score') is None:reason='same-pair CPU comparison missing'
        elif status=='partial':reason='sampling incomplete'
        elif status.startswith('skipped:process_exit_'):reason='unexpected process failure requires investigation'
        elif 'diagnostic_' in status:reason='diagnostic adapter does not qualify'
        elif status=='unsupported' and not row.get('notes'):reason='unsupported route lacks reason'
        elif not status.startswith('skipped:') and status!='unsupported':reason='unknown status'
        if reason:pending.append(dict(cell=list(key),reason=reason))
    receipt=dict(raw=str(a.raw),raw_sha256=hashlib.sha256(raw).hexdigest(),inventory_sha256=hashlib.sha256(a.inventory.read_bytes()).hexdigest(),
        pairs_manifest_sha256=hashlib.sha256(a.pairs.read_bytes()).hexdigest(),expected_cells=len(expected),present_requested_cells=len(expected&actual.keys()),
        missing_cells=missing,pending_qualification=pending,status_counts=dict(counts),extra_cells=len(actual.keys()-expected),
        complete=not missing and not pending,
        contract='CPU routes for non-GPU names, requested GPU routes and optional learned ORT-CUDA; explicit refusals are attempts, not timings')
    a.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['missing_cells','pending_qualification']}|dict(missing_count=len(missing),pending_count=len(pending))))
    if a.require_complete and not receipt['complete']:raise SystemExit(1)
if __name__=='__main__':main()
