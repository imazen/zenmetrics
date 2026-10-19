"""Join retained production wrapper events to actual NONMETRIC snapshots."""
import argparse,copy,hashlib,json
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
    kinds=[e['kind'] for e in events if e['kind']!='cleanup-wait']
    if kinds!=['waiting','acquired','waiting','acquired','command-launch','cleanup-confirmed','owner-unlocked','owner-unlocked','released']:raise ValueError('lifecycle order')
    acquired=[e for e in events if e['kind']=='acquired'];unlocked=[e for e in events if e['kind']=='owner-unlocked']
    if [e['order'] for e in acquired]!=[0,1] or [e['order'] for e in unlocked]!=[1,0]:raise ValueError('lock order')
    if [e['proof'] for e in acquired]!=held['locks']:raise ValueError('kernel receipt join')
    for e in unlocked:
        if 'FLOCK' not in e['before_fdinfo'] or 'FLOCK' in e['after_fdinfo']:raise ValueError('owner release proof')
    owner=held['owner']
    for key,v in owner.items():
        if outer['wrapper'][key]!=v:raise ValueError('wrapper identity')
    for witness in (control['production_before'],control['production_after'],outer['before_release']):
        if witness['held_sha256']!=outer['production_receipt_sha256']:raise ValueError('witness bytes')
        if witness['owner']!=outer['wrapper']:raise ValueError('witness identity')
        if not acquired[-1]['monotonic_ns']<=witness['observed_monotonic_ns']<unlocked[0]['monotonic_ns']:raise ValueError('witness outside hold')
    processes=[p for s in value['snapshots'] for p in s['processes'] if p['pid']==owner['pid']]
    if not processes or any(p['start_ticks']!=owner['start_ticks'] or not p['owned'] or p['counted'] for p in processes):raise ValueError('wrapper snapshot lifetime/classification')
    if not outer['command_reaped_ns']<unlocked[0]['monotonic_ns']:raise ValueError('release before reap')
    return dict(production_lock_join=True,wrapper_pid=owner['pid'],wrapper_start_ticks=owner['start_ticks'],locks=expected,attribution_ready=False,scope='retained owned NONMETRIC execution only')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(json.dumps(verify(a.root),indent=2))
