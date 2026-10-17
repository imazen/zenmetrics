"""Verify an intentionally interrupted cell stayed unqualified and recovered its ticket."""
import argparse,hashlib,json,pathlib,socket
from admission_recovery import exited
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=P);a=p.parse_args()
rows=[json.loads(s) for s in (a.root/(socket.gethostname().split('.')[0]+'.jsonl')).read_text().splitlines()]
assert len(rows)==2,'control requires both explicit modes'
score=next(r for r in rows if r['mode']=='score')
assert score['status']=='skipped:timeout_1s' and score['n']==0,'interrupted samples must not qualify'
assert score['resource']['exit_code']==124,'expected scope timeout'
assert len(score['admission_recovery'])==1,'expected exactly one recovered reservation'
record=score['admission_recovery'][0];process=record['process']
assert exited(process['pid'],process['pgid']),'interrupted work must have exited'
archive,preserved=P(record['archive']),P(record['preserved'])
assert archive.read_bytes()==preserved.read_bytes(),'preserve the original record verbatim'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==record['original_sha256']
original=preserved.with_name(preserved.name.split('.bak-iqa-')[0])
assert not original.exists(),'recovered ticket must leave the active admission set'
receipt=dict(scope='Real Linux timeout/recovery control; not a metric timing result',status='passed',impl_binary_sha=score['impl_binary_sha'],process=process,reservation_sequence=record['sequence'],record_sha256=record['original_sha256'],resource=score['resource'],log_path=score['log_path'],log_sha256=score['log_sha256'])
with (a.root/'timeout-control.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
