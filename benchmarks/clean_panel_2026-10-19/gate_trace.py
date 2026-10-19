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


def validate_attribution(value):
    # Proposed future snapshot sidecar. Existing text alone cannot satisfy this.
    if value.get('schema')!=1 or value.get('synthetic') is not False:
        raise ValueError('not a real attribution receipt')
    for name in ('binary_sha256','config_sha256','source_sha256','trace_sha256','outer_receipt_sha256'):
        if not re.fullmatch('[0-9a-f]{64}',value.get(name,'')):raise ValueError('missing binding '+name)
    seen=set()
    if not value.get('observations'):raise ValueError('missing observations')
    for o in value['observations']:
        if not o.get('snapshot_id') or o['snapshot_id'] in seen:raise ValueError('duplicate/missing snapshot')
        seen.add(o['snapshot_id'])
        for name in ('sample_monotonic_ns','checked_monotonic_ns','observation_age_ns'):
            if type(o.get(name)) is not int or o[name]<0:raise ValueError('missing observation time')
        if o['checked_monotonic_ns']-o['sample_monotonic_ns']!=o['observation_age_ns']:raise ValueError('age mismatch')
        ids=set();count=0
        for p in o['processes']:
            for name in ('pid','start_ticks','pgid','sid'):
                if type(p.get(name)) is not int or p[name]<=0:raise ValueError('missing process identity')
            key=(p['pid'],p['start_ticks'])
            if key in ids:raise ValueError('duplicate process')
            ids.add(key)
            if not p.get('cgroup') or not p.get('ownership_basis'):raise ValueError('missing ownership')
            for name in ('counted','owned'):
                if type(p.get(name)) is not bool:raise ValueError('missing classification')
            for name in ('cpu_ticks_before','cpu_ticks_after','interval_ns'):
                if type(p.get(name)) is not int or p[name]<0:raise ValueError('missing CPU counters')
            if p['interval_ns']==0 or p['cpu_ticks_after']<p['cpu_ticks_before']:raise ValueError('invalid CPU delta')
            if p['counted'] and p['owned']:raise ValueError('owned process counted foreign')
            count+=p['counted']
        if type(o.get('heavy_process_count')) is not int or count!=o['heavy_process_count']:raise ValueError('count mismatch')
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    raw=a.log.read_bytes();result=parse(raw.decode())
    result.update(raw_sha256=hashlib.sha256(raw).hexdigest(),raw_file='gate.log')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'gate.log').write_bytes(raw)
    (a.output/'parsed.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
