"""Require actual timed rounds from multiple independent bounded scopes."""
import argparse,json,pathlib,socket
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=P);a=p.parse_args()
rows=[json.loads(s) for s in (a.root/(socket.gethostname().split('.')[0]+'.jsonl')).read_text().splitlines()]
# Append-only logs retain failed admission history; inspect the latest result.
score={r['mode']:r for r in rows}['score']
assert score['status']=='ok','bounded rounds did not qualify'
assert score['n']>=10 and len(score['round_means_ns'])==score['n'],'ten actual rounds required'
assert len(score['chunks'])>=2,'control must exercise independent scopes'
assert len({v['attempt'] for v in score['chunks']})==len(score['chunks'])
assert sum(v['n'] for v in score['chunks'])==score['n']
assert all(0<v['n']<10 for v in score['chunks']),'each source must be a partial chunk'
receipt=dict(scope='Actual PSNR multi-chunk sampling control',status='passed',n=score['n'],chunks=len(score['chunks']),chunk_round_counts=[v['n'] for v in score['chunks']],score_bits=score['score_bits'],impl_binary_sha=score['impl_binary_sha'],pair_sha256=score['pair_sha256'],chunk_budget_ms=score['chunk_budget_ms'],sample_target_ns=score['sample_target_ns'],resource=score['resource'])
with (a.root/'chunk-control.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
