#!/usr/bin/env python3
"""Stop proven drained FitCell containers; preserve logs and allocation receipts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--inventory', type=Path, required=True)
p.add_argument('--coverage', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
inventory = json.loads(a.inventory.read_text())
coverage = json.loads(a.coverage.read_text())
active = {v['worker'] for cell in coverage['missing'] for v in cell['claims'].values()
          if v.get('live') and not v.get('finished')}
a.out.mkdir()
remote = r'''
import json, subprocess, sys
name, run = sys.argv[1:]
def docker(*args):
    return subprocess.check_output(['docker',*args], text=True, stderr=subprocess.STDOUT)
info=json.loads(docker('inspect',name))[0]
env=dict(x.split('=',1) for x in info['Config']['Env'] if '=' in x)
if not info['State']['Running'] or env.get('ZEN_RUN') not in (run,'jobs/'+run):
    raise ValueError('allocation identity or state changed')
cid=info['Id']
recent=docker('logs','--tail','20',cid)
if 'run control active (paused/draining)' not in recent:
    raise ValueError('worker has not entered drained idle passes')
top=docker('top',cid,'-eo','pid,comm,args')
for line in top.splitlines()[1:]:
    fields=line.split(None,2)
    if len(fields)<3:continue
    _,comm,argv=fields
    if comm.startswith(('zensim_mlp','bake_dial','byc_predict','panel')) or (comm.startswith('python') and any(x in argv for x in ('fit_cell_exec.py','byc_cells.py','v2_lodo_mlp.py','v2_confirm_fit.py'))):
        raise ValueError('fit or prediction process still running')
logs=docker('logs','--timestamps',cid)
docker('stop','--time','30',cid)
after=json.loads(docker('inspect',cid))[0]
assert not after['State']['Running']
print(json.dumps(dict(allocation_id=cid,image=info['Config']['Image'],run=run,
                     process_inventory=top,state=after['State'],logs=logs)))
'''
stopped = []
for worker in inventory['workers']:
    if worker['kind'] != 'container' or worker['worker'] in active:
        continue
    result = subprocess.run(['ssh', worker['host'], 'python3', '-', worker['container'], worker['run']],
                            input=remote, text=True, capture_output=True, check=True)
    value = json.loads(result.stdout)
    log = a.out / (worker['worker']+'.log')
    log.write_text(value.pop('logs'))
    stopped.append(dict(host=worker['host'],worker=worker['worker'],log=str(log),**value))
    (a.out/'STOPPED.json').write_text(json.dumps(dict(observed_at=datetime.now(timezone.utc).isoformat(),
                                                   stopped=stopped),indent=2)+'\n')
    print(worker['worker']+' stopped after drained-idle proof',flush=True)
