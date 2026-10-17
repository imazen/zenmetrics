#!/usr/bin/env python3
"""Retain immutable Task 4 observation/config receipts using read-only SSH."""
import argparse,base64,hashlib,json,pathlib,subprocess
P=pathlib.Path
HOSTS={'mac','r7900x'}


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('raw',type=P);parser.add_argument('output',type=P);args=parser.parse_args()
 rows=[json.loads(line) for line in args.raw.read_text().splitlines()]
 args.output.mkdir(exist_ok=False);retained={};manifest=[]
 for host in sorted({r['host'] for r in rows}):
  assert host in HOSTS
  selected=[r for r in rows if r['host']==host]
  paths=sorted({p for row in selected for p in row['host_observation_receipts']})
  assert all('/tmp/iqa-compete/task4/protocol-' in p and '/protocol-scopes/' in p and '..' not in P(p).parts for p in paths)
  # Only listed immutable receipts and their bound configurations are read.
  program='PATHS='+repr(paths)+'\n'+'''import pathlib,json,base64,hashlib
prefix=str(pathlib.Path.home()/'tmp/iqa-compete/task4')+'/'
for path in PATHS:
 assert path.startswith(prefix+'protocol-')
 scope=pathlib.Path(path).read_bytes();record=json.loads(scope);config=record['cell_config']
 assert config.startswith(prefix+'protocol-') and '/cells/' in config and '..' not in pathlib.Path(config).parts
 body=pathlib.Path(config).read_bytes()
 assert hashlib.sha256(body).hexdigest()==record['cell_config_sha256']
 print(json.dumps(dict(path=path,scope=base64.b64encode(scope).decode(),config_path=config,config=base64.b64encode(body).decode())),flush=True)
'''
  process=subprocess.run(['ssh',host,'python3','-'],input=program,capture_output=True,text=True,check=True)
  records=[json.loads(line) for line in process.stdout.splitlines()]
  assert {r['path'] for r in records}==set(paths)
  folder=args.output/host;folder.mkdir()
  for record in records:
   body=base64.b64decode(record['scope']);config=base64.b64decode(record['config']);scope=json.loads(body)
   name=hashlib.sha256(record['path'].encode()).hexdigest()
   scope_file=folder/(name+'.scope.json');config_file=folder/(name+'.config.json')
   scope_file.write_bytes(body);config_file.write_bytes(config)
   retained[record['path']]=(scope,json.loads(config))
   manifest.append(dict(host=host,source=record['path'],retained=str(scope_file.resolve()),sha256=hashlib.sha256(body).hexdigest(),source_config=record['config_path'],retained_config=str(config_file.resolve()),config_sha256=hashlib.sha256(config).hexdigest()))
 for row in rows:
  sources=[retained[p] for p in row['host_observation_receipts']]
  assert any(scope['contaminated'] for scope,_ in sources)==row['contaminated']
  for scope,config in sources:
   assert scope['protocol_sha256']==row['host_protocol']['protocol_sha256']
   assert scope['contaminated']==bool(scope['external_heavy_processes'])
   for key in ['metric_name','size_name','threads','pair_sha256','impl_binary_sha']:
    assert config[key]==row[key],(key,row['host'],row['metric_name'])
 result=dict(raw=str(args.raw.resolve()),raw_sha256=hashlib.sha256(args.raw.read_bytes()).hexdigest(),rows_verified=len(rows),scopes_retained=len(manifest),files=manifest,scope='Read-only transfer of selected immutable scope/config receipts; classification and configuration bindings verified, no independent remeasurement')
 (args.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='files'}))

if __name__=='__main__':main()
