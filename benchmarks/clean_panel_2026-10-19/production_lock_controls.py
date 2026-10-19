"""Owned fixture controls: never inspect or modify production lock paths."""
import argparse,json
from pathlib import Path
import production_locks as locks

def run(root):
    root.mkdir(exist_ok=False);paths=[root/'a.lock',root/'b.lock']
    for p in paths:p.write_text('owned fixture\n')
    expected=[locks.inspect(p) for p in paths];refused=[]
    for kind in ('missing','directory','symlink','inode'):
        rows=[dict(r) for r in expected]
        if kind=='missing':rows[0]['path']=str(root/'absent')
        if kind=='directory':rows[0]['path']=str(root)
        if kind=='symlink':
            link=root/'link';link.symlink_to(paths[0]);rows[0]['path']=str(link)
        if kind=='inode':rows[0]['inode']+=1
        try:locks.ExistingLocks(rows,root/kind)
        except (ValueError,FileNotFoundError):refused.append(kind)
        else:raise AssertionError('invalid inventory accepted: '+kind)
    m=locks.ExistingLocks(expected,root/'positive');held=m.acquire()
    receipt=root/'held.json';receipt.write_text(json.dumps(held));proof=locks.witness(receipt)
    assert proof['owner']['pid']==held['owner']['pid']
    m.release(dict(no_child_launched=True))
    events=[json.loads(line) for line in (root/'positive/events.jsonl').read_text().splitlines()]
    assert [e['kind'] for e in events]==['waiting','acquired','waiting','acquired','cleanup-confirmed','owner-unlocked','owner-unlocked','released']
    assert [locks.inspect(p) for p in paths]==expected
    assert all(p.read_text()=='owned fixture\n' for p in paths)
    (root/'result.json').write_text(json.dumps(dict(refused=refused,held_witness=proof,unchanged_inodes=True,attribution_ready=False),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);run(p.parse_args().output)
