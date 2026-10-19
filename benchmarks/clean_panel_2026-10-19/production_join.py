"""Join retained production wrapper events to actual NONMETRIC snapshots."""
import argparse,copy,hashlib,json,os
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root):
    run=root/'runtime-locks';outer=json.loads((run/'wrapper.json').read_text())
    held=json.loads((run/'production-held.json').read_text())
    inventory=json.loads((root/'inventory.json').read_text())
    control=json.loads((root/'runtime/outer.json').read_text())
    value=json.loads((root/'runtime/attribution.json').read_text())
    events=[json.loads(l) for l in (run/'events.jsonl').read_text().splitlines()]
    if outer['returncode']!=0:raise ValueError('command failed')
    if outer['inventory_sha256']!=sha(root/'inventory.json') or outer['production_receipt_sha256']!=sha(run/'production-held.json') or outer['control_outer_sha256']!=sha(root/'runtime/outer.json'):raise ValueError('artifact binding')
    expected=inventory['locks']
    if [{k:r[k] for k in ('path','device','inode')} for r in held['locks']]!=expected:raise ValueError('inventory join')
    if [e['monotonic_ns'] for e in events]!=sorted(e['monotonic_ns'] for e in events):raise ValueError('event chronology')
    kinds=[e['kind'] for e in events if e['kind'] not in ('cleanup-wait','quarantined')]
    if kinds!=['waiting','acquired','waiting','acquired','command-launch','cleanup-confirmed','owner-unlocked','owner-unlocked','released']:raise ValueError('lifecycle order')
    acquired=[e for e in events if e['kind']=='acquired'];unlocked=[e for e in events if e['kind']=='owner-unlocked']
    if [e['order'] for e in acquired]!=[0,1] or [e['order'] for e in unlocked]!=[1,0]:raise ValueError('lock order')
    if [e['proof'] for e in acquired]!=held['locks']:raise ValueError('kernel receipt join')
    for e in unlocked:
        row=held['locks'][e['order']]
        if e['fd']!=row['fd'] or e['path_identity']!={k:row[k] for k in ('path','device','inode')}:raise ValueError('release identity')
        if row['kernel_lock'] not in e['before_fdinfo'].splitlines() or 'FLOCK' in e['after_fdinfo']:raise ValueError('owner release proof')
    owner=held['owner']
    if [r['order'] for r in held['locks']]!=[0,1] or len({r['fd'] for r in held['locks']})!=2:raise ValueError('held descriptor order')
    for row in held['locks']:
        fields=row['kernel_lock'].split()
        if len(fields)!=9 or fields[0]!='lock:' or fields[2:5]!=['FLOCK','ADVISORY','WRITE'] or fields[5]!=str(owner['pid']) or fields[7:]!=['0','EOF']:raise ValueError('kernel owner record')
        device=fields[6].split(':')
        if len(device)!=3 or (int(device[0],16),int(device[1],16),int(device[2]))!=(os.major(row['device']),os.minor(row['device']),row['inode']):raise ValueError('kernel inode record')
    if events[-1]['owner']!=outer['wrapper']:raise ValueError('released owner')
    launches=[e for e in events if e['kind']=='command-launch']
    if launches[0]['owner']!=outer['wrapper'] or launches[0]['command']!=outer['command']:raise ValueError('command launch binding')
    for e in acquired:
        if e['owner']!=outer['wrapper']:raise ValueError('acquire owner')
    for key,v in owner.items():
        if outer['wrapper'][key]!=v:raise ValueError('wrapper identity')
    for witness in (control['production_before'],control['production_after'],outer['before_release']):
        if len(witness['fdinfo'])!=2 or not witness['chain']:raise ValueError('empty production witness')
        for row,raw in zip(held['locks'],witness['fdinfo']):
            if row['kernel_lock'] not in raw.splitlines():raise ValueError('kernel witness missing')
        chain=witness['chain']
        if chain[-1]!=outer['wrapper'] or len({p['pid'] for p in chain})!=len(chain):raise ValueError('ancestry endpoint')
        for a,b in zip(chain,chain[1:]):
            if a['ppid']!=b['pid']:raise ValueError('ancestry edge')
        if witness['held_sha256']!=outer['production_receipt_sha256']:raise ValueError('witness bytes')
        if witness['owner']!=outer['wrapper']:raise ValueError('witness identity')
        if not acquired[-1]['monotonic_ns']<=witness['observed_monotonic_ns']<unlocked[0]['monotonic_ns']:raise ValueError('witness outside hold')
    if control['production_before']['chain']!=control['production_after']['chain']:raise ValueError('ancestry lifetime changed')
    for witness in (control['production_before'],control['production_after']):
        for k,v in control['controller'].items():
            if witness['chain'][0][k]!=v:raise ValueError('controller ancestry root')
    if outer['before_release']['chain']!=[outer['wrapper']]:raise ValueError('release ancestry root')
    cleanup=[e for e in events if e['kind']=='cleanup-confirmed'][0]
    c=cleanup['cleanup']
    if c.get('command_reaped') is not True or c.get('monitor_control_complete') is not True or c.get('returncode')!=0:raise ValueError('cleanup not proven')
    if not outer['command_reaped_ns']<=cleanup['monotonic_ns']<unlocked[0]['monotonic_ns']:raise ValueError('cleanup chronology')
    if not outer.get('descendants_confirmed_exited') or c.get('retained_descendants_exited')!=outer['descendants_confirmed_exited']:raise ValueError('descendant exit receipt')
    required=control['production_after']['chain'][:-1]+[control['rival'],control['launch']['child']]
    if outer['descendants_confirmed_exited']!=required:raise ValueError('descendant membership')
    child=outer['command_child']
    if child['ppid']!=owner['pid'] or not any(p['pid']==child['pid'] and p['start_ticks']==child['start_ticks'] for p in outer['observed_session_members_exited']):raise ValueError('command child lifetime')
    for witness in (control['production_before'],control['production_after']):
        if not any(p['pid']==child['pid'] and p['start_ticks']==child['start_ticks'] for p in witness['chain']):raise ValueError('command child ancestry')
    processes=[p for s in value['snapshots'] for p in s['processes'] if p['pid']==owner['pid']]
    if not processes or any(p['start_ticks']!=owner['start_ticks'] or not p['owned'] or p['counted'] for p in processes):raise ValueError('wrapper snapshot lifetime/classification')
    if not outer['command_reaped_ns']<unlocked[0]['monotonic_ns']:raise ValueError('release before reap')
    return dict(production_lock_join=True,wrapper_pid=owner['pid'],wrapper_start_ticks=owner['start_ticks'],locks=expected,attribution_ready=False,scope='retained owned NONMETRIC execution only')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(verify(a.root),indent=2))
