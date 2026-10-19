"""Retain gate text; require exact snapshot attribution before diagnosis readiness."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def parse(text):
    foreign=[];gates=[]
    for number,line in enumerate(text.splitlines(),1):
        if line.startswith('[zenbench gate] foreign '):
            match=re.fullmatch(r'\[zenbench gate\] foreign pid=(\d+) name=(.+) cpu_pct=([0-9.eE+-]+)',line)
            if not match:raise ValueError('malformed foreign trace')
            pid, cpu = int(match[1]), float(match[3])
            if pid <= 0 or not math.isfinite(cpu) or cpu < 0:raise ValueError('invalid foreign trace')
            foreign.append(dict(line=number,pid=pid,name=match[2],cpu_pct=cpu))
        elif line.startswith('[zenbench gate] clean='):
            match=re.search(r'clean=(true|false) reason=(.*?) state=(.*)$',line)
            if not match:raise ValueError('malformed gate trace')
            gates.append(dict(line=number,clean=match[1]=='true',reason=match[2],state=match[3]))
    return dict(foreign=foreign,gates=gates,attribution_ready=False,
        missing=['snapshot ID/time and age correlation','PID start/group/session/cgroup identity','owned classification and cumulative CPU deltas'])


def snapshot_hash(snapshot):
    return hashlib.sha256(json.dumps(snapshot,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_attribution(value):
    # Structural validity never establishes live artifact identity or causality.
    if value.get('schema')!=2 or value.get('synthetic') is not False:
        raise ValueError('not a real attribution receipt declaration')
    for name in ('binary_sha256','config_sha256','source_sha256','trace_sha256','outer_receipt_sha256'):
        if not re.fullmatch('[0-9a-f]{64}',value.get(name,'')):raise ValueError('missing binding '+name)
    snapshots={}
    if not value.get('snapshots') or not value.get('checks'):raise ValueError('missing snapshots/checks')
    for o in value['snapshots']:
        sid=o.get('snapshot_id')
        if not isinstance(sid,str) or not sid or sid in snapshots:raise ValueError('duplicate/redefined snapshot')
        if type(o.get('sample_monotonic_ns')) is not int or o['sample_monotonic_ns']<0:raise ValueError('missing snapshot time')
        ids=set();count=0
        for p in o['processes']:
            for name in ('pid','start_ticks','pgid','sid'):
                if type(p.get(name)) is not int or p[name]<=0:raise ValueError('missing process identity')
            if p['pid'] in ids:raise ValueError('duplicate/conflicting PID identity')
            ids.add(p['pid'])
            if not p.get('cgroup') or not p.get('ownership_basis'):raise ValueError('missing ownership')
            for name in ('counted','owned'):
                if type(p.get(name)) is not bool:raise ValueError('missing classification')
            for name in ('cpu_ticks_before','cpu_ticks_after','interval_ns'):
                if type(p.get(name)) is not int or p[name]<0:raise ValueError('missing CPU counters')
            if p['interval_ns']==0 or p['cpu_ticks_after']<p['cpu_ticks_before']:raise ValueError('invalid CPU delta')
            if p['counted'] and p['owned']:raise ValueError('owned process counted foreign')
            count+=p['counted']
        if type(o.get('heavy_process_count')) is not int or count!=o['heavy_process_count']:raise ValueError('count mismatch')
        snapshots[sid]=o
    checks=set();used=set()
    for c in value['checks']:
        cid=c.get('check_id');sid=c.get('snapshot_id')
        if not isinstance(cid,str) or not cid or cid in checks:raise ValueError('duplicate/missing check')
        checks.add(cid)
        if sid not in snapshots:raise ValueError('missing snapshot join')
        used.add(sid);o=snapshots[sid]
        for name in ('checked_monotonic_ns','observation_age_ns'):
            if type(c.get(name)) is not int or c[name]<0:raise ValueError('missing check time')
        if c['checked_monotonic_ns']-o['sample_monotonic_ns']!=c['observation_age_ns']:raise ValueError('age mismatch')
        if c.get('snapshot_sha256')!=snapshot_hash(o):raise ValueError('snapshot identity mismatch')
        if type(c.get('heavy_process_count')) is not int or c['heavy_process_count']!=o['heavy_process_count']:raise ValueError('check count mismatch')
        expected=sorted([p['pid'],p['start_ticks']] for p in o['processes'] if p['counted'])
        if c.get('counted_identities')!=expected:raise ValueError('check process identity mismatch')
    return dict(schema_valid=True,attribution_ready=False,snapshot_count=len(snapshots),check_count=len(checks),
                unconsumed_snapshot_ids=sorted(set(snapshots)-used))


def verify_artifact_join(value, paths):
    """Verify actual bytes and exact emitted events, still not live causality."""
    result=validate_attribution(value)
    for name in ('binary','config','source','trace','outer_receipt'):
        raw=Path(paths[name]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=value[name+'_sha256']:raise ValueError('artifact content mismatch: '+name)
    events=[json.loads(line) for line in Path(paths['trace']).read_text().splitlines()]
    expected=[dict(kind='snapshot',value=o) for o in value['snapshots']]+[dict(kind='check',value=c) for c in value['checks']]
    normalize=lambda e:json.dumps(e,sort_keys=True,separators=(',',':'))
    if sorted(map(normalize,events))!=sorted(map(normalize,expected)):raise ValueError('emitted snapshot/check join mismatch')
    return result | dict(artifact_bytes_verified=True,emitted_events_verified=True,
                         missing=['live instrumentation qualification','outer clock/ownership correlation'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    raw=a.log.read_bytes();result=parse(raw.decode())
    result.update(raw_sha256=hashlib.sha256(raw).hexdigest(),raw_file='gate.log')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'gate.log').write_bytes(raw)
    (a.output/'parsed.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
