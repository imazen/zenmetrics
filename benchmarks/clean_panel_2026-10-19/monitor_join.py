"""Bounded Linux owned-nonmetric joins; never confers attribution readiness."""
import hashlib,json
from pathlib import Path
import gate_trace

def integer(v):
    if type(v) is not int or v<0:raise ValueError('invalid clock value')
    return v

def common_offset(value,outer):
    names=('launch_before_ns','ready_received_ns','go_ns','exit_observed_ns','finally_ns')
    clocks=[integer(outer[k]) for k in names]
    if clocks!=sorted(clocks):raise ValueError('outer chronology')
    held=integer(outer['held']['observed_monotonic_ns']);launched=integer(outer['launch']['observed_monotonic_ns'])
    if not held<=clocks[0] or not clocks[1]<=launched<=clocks[2]:raise ValueError('held/launch receipt chronology')
    lo,hi=clocks[:2];go,exit=clocks[2:4]
    stamps=[]
    sample_times=[integer(s['sample_monotonic_ns']) for s in value['snapshots']]
    if sample_times!=sorted(sample_times):raise ValueError('snapshot chronology')
    for snapshot in value['snapshots']:
        sample=integer(snapshot['sample_monotonic_ns'])
        start=integer(snapshot.get('counter_interval_start_ns',sample))
        if start>sample:raise ValueError('snapshot chronology')
        # Includes final unused refresh: it is after GO and before joined exit.
        stamps.extend((start,sample))
    last=-1
    for check in value['checks']:
        stamp=integer(check['checked_monotonic_ns'])
        if stamp<last:raise ValueError('check chronology')
        last=stamp;stamps.append(stamp)
    for event in value.get('monitor_events',[]):
        v=event['value']
        if event['kind']=='publication':stamps.append(integer(v['publication_monotonic_ns']))
        elif event['kind']=='gate':
            stamps.extend(integer(v[k]) for k in ('entry_monotonic_ns','hook_monotonic_ns'))
            if v['entry_monotonic_ns']>v['hook_monotonic_ns']:raise ValueError('gate chronology')
            read=v['read']
            if read is not None:stamps.append(integer(read['observed_monotonic_ns']))
    for stamp in stamps:lo=max(lo,go-stamp);hi=min(hi,exit-stamp)
    if lo>hi:raise ValueError('no common clock origin offset')
    return dict(min_ns=lo,max_ns=hi,exact=False)

def verify_outer(value,outer):
    rival=outer['rival'];leader=outer['launch']['child'];owner=outer['launch']['owner'];held=outer['held']
    if owner!=held['owner'] or owner!=outer['controller']:raise ValueError('controller receipt identity mismatch')
    if outer['launch']['token']!=held['token'] or outer['launch']['locks']!=held['locks'] or outer['locks_still_held']!=held['locks']:raise ValueError('held/launch/cleanup lock mismatch')
    for child in (rival,leader):
        if child['ppid']!=owner['pid']:raise ValueError('owned child parent mismatch')
    for role,identity,owned,counted in [('rival',rival,False,True),('monitor',leader,True,False),('controller',owner,True,False)]:
        matches=[p for s in value['snapshots'] for p in s['processes'] if p['pid']==identity['pid']]
        if not matches or any(p['start_ticks']!=identity['start_ticks'] for p in matches):raise ValueError(role+' lifetime not joined')
        if not any(p['owned'] is owned and p['counted'] is counted for p in matches):raise ValueError(role+' classification not joined')
    if outer.get('cleanup_complete') is not True or outer.get('rival_reaped') is not True or outer.get('child_returncode')!=0:raise ValueError('missing owned cleanup')
    return common_offset(value,outer)

def verify_events(value):
    snapshots={s['snapshot_id']:s for s in value['snapshots']};checks={c['check_id']:c for c in value['checks']}
    publications={};gates=[];used=set();ids=set()
    for event in value.get('monitor_events',[]):
        v=event['value'];kind=event['kind']
        if kind=='publication':
            sid=v['snapshot_id']
            if sid in publications or sid not in snapshots:raise ValueError('publication join')
            if integer(v['publication_monotonic_ns'])<snapshots[sid]['sample_monotonic_ns']:raise ValueError('publication before sample')
            publications[sid]=v
        elif kind=='gate':gates.append(v)
        else:raise ValueError('unknown monitor event')
    if set(publications)!=set(snapshots):raise ValueError('missing publication including unused refresh')
    previous_hook=-1
    for g in gates:
        if integer(g['entry_monotonic_ns'])<previous_hook:raise ValueError('overlapping/reordered gate calls')
        previous_hook=integer(g['hook_monotonic_ns'])
        if g['gate_id'] in ids:raise ValueError('duplicate gate')
        ids.add(g['gate_id']);read=g['read'];status=g['status'];cid=g['check_id']
        if read is None:raise ValueError('missing actual read evidence')
        observed=integer(read['observed_monotonic_ns'])
        if not g['entry_monotonic_ns']<=observed<=g['hook_monotonic_ns']:raise ValueError('read outside gate')
        if read['status']=='missing':
            if status!='unknown' or any(g[k] is not None for k in ('clean','reason','state','check_id','gate_observation_age_ns')):raise ValueError('missing gate not unknown')
            if read['snapshot_id'] is not None or read['publication_monotonic_ns'] is not None or read['age_ns'] is not None:raise ValueError('missing read has snapshot')
            continue
        sid=read['snapshot_id']
        if sid not in publications or read['publication_monotonic_ns']!=publications[sid]['publication_monotonic_ns']:raise ValueError('chosen publication mismatch')
        age=observed-read['publication_monotonic_ns']
        if age<0 or read['age_ns']!=age:raise ValueError('actual cached age mismatch')
        if read['status']=='stale':
            if age<=1_000_000_000 or status!='unknown' or any(g[k] is not None for k in ('clean','reason','state','check_id','gate_observation_age_ns')):raise ValueError('stale gate not unknown')
            continue
        if read['status']!='fresh' or age>1_000_000_000:raise ValueError('freshness mismatch')
        if cid not in checks or cid in used or checks[cid]['snapshot_id']!=sid:raise ValueError('decision check join')
        used.add(cid)
        if g['gate_observation_age_ns']!=age or status!='decision' or type(g['clean']) is not bool:raise ValueError('gate actual age/status mismatch')
        if not observed<=checks[cid]['checked_monotonic_ns']<=g['hook_monotonic_ns']:raise ValueError('check outside observed gate interval')
        if g['clean']!=(g['reason']=='None'):raise ValueError('clean/reason disagreement')
        if g['state']['heavy_process_count']!=snapshots[sid]['heavy_process_count']:raise ValueError('decision count mismatch')
        if g['reason'].startswith('Some(HeavyProcesses(') and g['reason']!=f"Some(HeavyProcesses({snapshots[sid]['heavy_process_count']}))":raise ValueError('heavy reason mismatch')
    if used!=set(checks):raise ValueError('checks missing decisions')
    return dict(gate_events=len(gates),publications=len(publications),statuses={kind:sum(g['read']['status']==kind for g in gates) for kind in ('fresh','stale','missing')})

def verify_artifacts(value,paths):
    result=gate_trace.validate_attribution(value)
    for key,path in paths.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=value[key+'_sha256']:raise ValueError('artifact mismatch '+key)
    events=[json.loads(line) for line in Path(paths['trace']).read_text().splitlines()]
    expected=[dict(kind='snapshot',value=s) for s in value['snapshots']]+[dict(kind='check',value=c) for c in value['checks']]+value['monitor_events']
    canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'))
    if sorted(map(canonical,events))!=sorted(map(canonical,expected)):raise ValueError('exact event membership mismatch')
    semantic=verify_events(value)
    outer=json.loads(Path(paths['outer_receipt']).read_text())
    if json.loads(Path(paths['held_receipt']).read_text())!=outer['held'] or json.loads(Path(paths['launch_receipt']).read_text())!=outer['launch']:raise ValueError('ownership record/outer mismatch')
    return result|semantic|dict(artifact_bytes_verified=True,emitted_events_verified=True,attribution_ready=False)
