#!/usr/bin/env python3
"""Run landing gates in separate bounded shared-lock holds, retaining full logs."""
import argparse,datetime,json,os,pathlib,subprocess,sys
from batch import yield_after_release
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=P);p.add_argument('snapshot',type=P)
p.add_argument('--stages',default='fmt,clippy,matrix,harness');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,LOCK_SNAP_DIR=str(a.snapshot.resolve()),TMPDIR=str(P.home()/'tmp/iqa-compete/build-scratch'))
P(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
commands={
 'fmt':['just','iqa-compete-fmt-check'],
 'clippy':['just','snap-cargo','@','clippy','-p','zenmetrics-cli','--no-default-features','--features','png,sweep,cpu-metrics,nn-ort,nr','--bin','zenmetrics','--','-D','warnings','-A','deprecated','-A','clippy::excessive_precision','-A','clippy::approx_constant','-A','clippy::erasing_op','-A','clippy::identity_op','-A','clippy::too_many_arguments','-A','clippy::doc_lazy_continuation','-A','clippy::if_same_then_else','-A','clippy::doc_overindented_list_items'],
 'matrix':['just','cli-feature-matrix'],
 'harness':[sys.executable,'-m','unittest','discover','-s','benchmarks/iqa_compete_2026-10-16','-p','test_*.py'],
 'driver':['env','IQA_GPU_RUNTIME=cuda','just','iqa-compete-clippy'],
}
for stage in a.stages.split(','):
 command=['flock',str(P.home()/'tmp/i265-heavy.lock'),str(P.home()/'work/claudehints/scripts/run-heavy'),'--mem','16G','--jobs','16','--','timeout','--kill-after=20s','1100s',*commands[stage]]
 print('START',stage,flush=True)
 with (a.output/f'{stage}.log').open('x') as log:result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 record=dict(stage=stage,command=command,returncode=result.returncode,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (a.output/f'{stage}.json').write_text(json.dumps(record,indent=2)+'\n')
 print('FINISH',stage,result.returncode,flush=True)
 yield_after_release(P.home()/'tmp/i265-heavy.lock')
 if result.returncode:raise SystemExit(result.returncode)
