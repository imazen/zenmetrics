#!/usr/bin/env python3
"""Stop one explicitly named idle controller, preserving any active evaluation."""
import argparse,json,os,pathlib,signal,time
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('pid',type=int);p.add_argument('script',type=P);p.add_argument('root',type=P);p.add_argument('proof',type=P);p.add_argument('--waiting-supervisor',action='store_true');a=p.parse_args()
proc=P(f'/proc/{a.pid}');argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
script=P(argv[1]);script=script if script.is_absolute() else (proc/'cwd').resolve()/script
if script.resolve()!=a.script.resolve() or str(a.root.resolve()) not in argv:raise RuntimeError('PID does not match the explicit controller identity')
if a.proof.exists():raise FileExistsError(a.proof)
stopped=[]
def freeze(pid):
 os.kill(pid,signal.SIGSTOP);stopped.append(pid)
 for _ in range(100):
  if '\nState:\tT' in P(f'/proc/{pid}/status').read_text():return
  time.sleep(.01)
 raise RuntimeError('Process did not stop')
try:
 freeze(a.pid);children=(proc/'task'/str(a.pid)/'children').read_text().split()
 if not children:
  if not a.waiting_supervisor or a.script.name!='coverage_competitors.py':raise RuntimeError('Expected one queued flock child')
 elif len(children)!=1:raise RuntimeError('Controller has multiple children; unchanged')
 proof=dict(pid=a.pid,argv=argv,children=[])
 for child in children:
  child=int(child);q=P(f'/proc/{child}');args=[x.decode() for x in (q/'cmdline').read_bytes().split(b'\0') if x]
  if P(args[0]).name!='flock' or not any(v.startswith(str(a.root.resolve())+'/') for v in args):raise RuntimeError('Child is not this controller queued flock')
  freeze(child);grand=(q/'task'/str(child)/'children').read_text().strip()
  if grand:raise RuntimeError('Active evaluation preserved; no process terminated')
  proof['children'].append(dict(pid=child,argv=args,verified_stopped_children=grand))
 for pid in reversed(stopped):os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
 stopped.clear()
 with a.proof.open('x') as f:f.write(json.dumps(proof,indent=2)+'\n')
 print(json.dumps(dict(stopped_idle_controller=a.pid,queued_children=[r['pid'] for r in proof['children']])))
finally:
 for pid in reversed(stopped):
  try:os.kill(pid,signal.SIGCONT)
  except ProcessLookupError:pass
