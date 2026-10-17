#!/usr/bin/env python3
"""Resume explicitly recorded coverage commands in their existing data roots."""
import argparse,concurrent.futures,datetime,json,pathlib,subprocess
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=P);p.add_argument('output',type=P);p.add_argument('--label',required=True);a=p.parse_args()
record=json.loads(a.manifest.read_text());jobs={r['lane']:r['command'] for r in record['rust']};jobs['native']=record['native']['command']
a.output.mkdir(parents=True,exist_ok=True)
def run(item):
 name,command=item;log=a.output/f'{name}-{a.label}.log'
 with log.open('x') as stream:
  process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
  launch=dict(lane=name,pid=process.pid,command=command,log=str(log.resolve()),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
  with (a.output/f'{name}-{a.label}-launch.json').open('x') as f:f.write(json.dumps(launch,indent=2)+'\n')
  print('START',name,process.pid,flush=True);code=process.wait()
 with (a.output/f'{name}-{a.label}-exit.json').open('x') as f:f.write(json.dumps(dict(returncode=code,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
 print('FINISH',name,code,flush=True);return code
with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:codes=list(pool.map(run,jobs.items()))
raise SystemExit(int(any(codes)))
