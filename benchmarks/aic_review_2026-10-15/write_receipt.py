"""Retain complete Round 5 captures and write small hashed gate receipts."""
import pathlib,hashlib,json,shutil,subprocess,datetime
root=pathlib.Path.home()/'tmp/aic-variants-scratch';src=root/'round5';ret=root/'retained-round5';dest=pathlib.Path('benchmarks/aic_review_2026-10-15');ret.mkdir(exist_ok=True)
files=[]
for p in sorted(src.rglob('*')):
 if p.is_file() and p.suffix in ['.log','.json','.rs','.cpp']:
  q=ret/p.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True)
  digest=hashlib.sha256(p.read_bytes()).hexdigest()
  if q.exists():assert hashlib.sha256(q.read_bytes()).hexdigest()==digest,str(q)+' changed after retention'
  else:shutil.copy2(p,q)
  files.append(dict(path=str(q),bytes=q.stat().st_size,sha256=digest))
receipt=dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=subprocess.check_output(['hostname'],text=True).strip(),rustc=subprocess.check_output(['rustc','--version'],text=True).strip(),base='c07da3683d0beb91602157575b78351bc91047d0',retained=str(ret),files=files)
receipt['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pathlib.Path('crates/vmaf/src/cambi.rs'),pathlib.Path('crates/zenmetrics-cli/src/main.rs'),*[pathlib.Path('crates/zenmetrics-cli/src/metrics')/(n+'.rs') for n in ['butteraugli_original','ssimulacra1','ws_hv']]]}
receipt['verified_logs']={}
for name in ['ci.log','pin-sensitivity.log','clean-rerun.log','cambi-ref-refresh.log']:
 text=(src/name).read_text();lines=text.splitlines();done=[l for l in lines if l.startswith('run-heavy: done')]
 assert done and done[-1].startswith('run-heavy: done rc=0'),name+' did not finish successfully'
 receipt['verified_logs'][name]=[l for l in lines if l.startswith('test result:') or l.startswith('run-heavy: done')]
for name in ['oracle_pins.json','pin_residuals.json','pin_sensitivity.json','cambi_reference_invariance.json','panel_invariance.json','summary.json','ssim_order_summary.json']:
 receipt[name]=hashlib.sha256((dest/name).read_bytes()).hexdigest()
assert json.load(open(src/'clean-pins/oracle-pins-banded.json'))==json.load(open(dest/'oracle_pins.json'))
with (dest/'validation.meta').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt['verified_logs'],indent=2))
