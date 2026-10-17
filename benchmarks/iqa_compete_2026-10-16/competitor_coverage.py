#!/usr/bin/env python3
"""Audit native/Python competitor attempts against the selected public adapters."""
import argparse,collections,hashlib,itertools,json,pathlib
from native_run import ADAPTERS
from run import PYTHON_METRICS
P=pathlib.Path
FIELDS=['impl','metric_name','device','size_name','threads','mode']

def audit(rows,pairs,sizes,threads,implementations):
    available={engine:[item[0] for item in adapters] for engine,adapters in ADAPTERS.items()}|PYTHON_METRICS
    if not set(implementations)<=available.keys():raise ValueError('Unknown requested implementation')
    geometry={pair['size_name']:(pair['width'],pair['height']) for pair in pairs}
    if not set(sizes)<=geometry.keys():raise ValueError('Unknown requested size')
    if not threads or any(n<1 for n in threads):raise ValueError('Positive thread budgets required')
    expected={(impl,metric,'cpu',size,count,mode) for impl in implementations
        for metric,size,count,mode in itertools.product(available[impl],sizes,threads,['score','score+map'])}
    actual={};testbeds=set();hashes=collections.defaultdict(set)
    for row in rows:
        if row['impl']=='zenmetrics':continue
        testbeds.add((row['host'],row['os']))
        if len(testbeds)>1:raise ValueError('Audit one host/OS testbed at a time')
        key=tuple(row[k] for k in FIELDS)
        if key in actual:raise ValueError('Collect duplicate attempts before auditing')
        actual[key]=row
        if row['size_name'] in sizes:
            if (row['width'],row['height'])!=geometry[row['size_name']]:raise ValueError('Wrong requested geometry')
            hashes[row['size_name']].add(row['pair_sha256'])
    if any(len(values)!=1 for values in hashes.values()):raise ValueError('A size label has multiple raw-pixel identities')
    counts=collections.Counter();pending=[]
    for key in sorted(expected&actual.keys()):
        row=actual[key];status=row['status'];counts[status.split(':',1)[0]]+=1;reason=None
        if status in ['ok','variant_mismatch']:
            if not (row.get('n',0)>=10 or (row.get('n',0)>=3 and row.get('timed_seconds',0)>=3)):reason='insufficient actual samples'
            elif row.get('ref_score') is None:reason='same-pair Rust reference missing'
            elif row.get('reproduction_required'):reason='mismatch reproduction pending'
        elif status.startswith('skipped:missing_same-pair_'):reason='same-pair Rust reference missing'
        elif status.startswith('skipped:process_exit_'):reason='unexpected adapter failure requires investigation'
        elif status=='unsupported' and not row.get('notes'):reason='unsupported route lacks reason'
        elif 'diagnostic_' in status:reason='diagnostic adapter does not qualify'
        elif status!='unsupported' and not status.startswith('skipped:'):reason='unknown status'
        if reason:pending.append(dict(cell=list(key),reason=reason))
    missing=sorted(expected-actual.keys())
    return dict(expected_cells=len(expected),present_requested_cells=len(expected&actual.keys()),
        requested_sizes=sizes,requested_threads=threads,
        adapter_inventory_sha256=hashlib.sha256(json.dumps({k:available[k] for k in implementations},sort_keys=True).encode()).hexdigest(),
        missing_cells=missing,pending_qualification=pending,status_counts=dict(counts),
        timing_quality_counts=dict(collections.Counter(r.get("timing_quality","legacy_unmonitored") for k,r in actual.items() if k in expected)),
        extra_cells=len(actual.keys()-expected),complete=not missing and not pending,
        contract='Selected public CPU competitor adapters; refusals are attempts, not latency measurements')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['raw','pairs','output']:p.add_argument(name,type=P)
    p.add_argument('--sizes',default='cid22,2k');p.add_argument('--threads',default='1,20')
    p.add_argument('--implementations',default=','.join([*ADAPTERS,*PYTHON_METRICS]))
    p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    raw=a.raw.read_bytes();pair_bytes=a.pairs.read_bytes()
    receipt=audit([json.loads(s) for s in raw.decode().splitlines()],json.loads(pair_bytes),
        a.sizes.split(','),list(map(int,a.threads.split(','))),a.implementations.split(','))
    receipt.update(raw=str(a.raw.resolve()),raw_sha256=hashlib.sha256(raw).hexdigest(),
        pairs=str(a.pairs.resolve()),pairs_sha256=hashlib.sha256(pair_bytes).hexdigest())
    with a.output.open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['missing_cells','pending_qualification']}|
        dict(missing_count=len(receipt['missing_cells']),pending_count=len(receipt['pending_qualification']))))
    if a.require_complete and not receipt['complete']:raise SystemExit(1)
if __name__=='__main__':main()
