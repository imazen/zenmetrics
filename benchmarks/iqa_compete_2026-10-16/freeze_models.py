#!/usr/bin/env python3
"""Copy the hashed model inventory into an owned, independently verified tree."""
import argparse,hashlib,json,pathlib,stat
P=pathlib.Path
EXTENSIONS={'.onnx','.rten','.meta','.f32','.json'}

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def freeze(default,overrides,manifest,output):
 mappings=json.loads(overrides.read_text());records=json.loads(manifest.read_text())['artifacts']
 directories={str(default),*mappings.values()}
 if directories!={r['directory'] for r in records}:raise ValueError('Model directory set differs from hashed inventory')
 destinations={directory:'models-'+hashlib.sha256(directory.encode()).hexdigest()[:16] for directory in directories}
 if len(set(destinations.values()))!=len(directories):raise ValueError('Destination identifier collision')
 for directory in directories:
  actual={p.name for p in P(directory).iterdir() if p.is_file() and p.suffix in EXTENSIONS}
  expected={r['file'] for r in records if r['directory']==directory}
  if actual!=expected:raise ValueError('Model artifact set changed: '+directory)
 output.mkdir(exist_ok=False,parents=True)
 (output/'PREPARATION_STARTED').write_text('Only COMPLETE and its matching locations.json qualify this tree.\n')
 copied=[]
 for record in records:
  name=record['file']
  if P(name).name!=name or name in ['.','..']:raise ValueError('Artifact name must be a basename')
  source=P(record['source'])
  if not source.is_absolute() or (P(record['directory'])/name).resolve(strict=True)!=source:raise ValueError('Source link target changed: '+str(source))
  before=source.stat()
  if (before.st_size,before.st_mtime_ns)!=(record['bytes'],record['mtime_ns']):raise ValueError('Source metadata changed: '+str(source))
  destination=output/destinations[record['directory']]/name;destination.parent.mkdir(exist_ok=True)
  h=hashlib.sha256()
  with source.open('rb') as incoming,destination.open('xb') as outgoing:
   for block in iter(lambda:incoming.read(1<<20),b''):outgoing.write(block);h.update(block)
  after=source.stat()
  if (after.st_size,after.st_mtime_ns)!=(before.st_size,before.st_mtime_ns):raise ValueError('Source changed during copy: '+str(source))
  if h.hexdigest()!=record['sha256'] or digest(destination)!=record['sha256']:raise ValueError('Artifact content differs from hashed inventory: '+str(source))
  destination.chmod(stat.S_IRUSR|stat.S_IRGRP|stat.S_IROTH)
  copied.append(dict(source=str(source),destination=str(destination.resolve()),bytes=record['bytes'],sha256=record['sha256']))
 locations={directory:str((output/name).resolve()) for directory,name in destinations.items()}
 model_map={metric:locations[directory] for metric,directory in mappings.items()}
 (output/'models.json').write_text(json.dumps(model_map,indent=2)+'\n')
 receipt=dict(default=locations[str(default)],overrides=str((output/'models.json').resolve()),source_manifest=str(manifest.resolve()),
  source_manifest_sha256=digest(manifest),artifacts=copied,files=len(copied),bytes=sum(r['bytes'] for r in copied),
  policy='Independent copies, initial SHA-256 checked during copy and on destination readback; data files read-only. Original files untouched.')
 (output/'locations.json').write_text(json.dumps(receipt,indent=2)+'\n')
 # Keep the preparation marker as historical evidence; completion is explicit.
 (output/'COMPLETE').write_text(digest(output/'locations.json')+'\n')
 return receipt

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['default','overrides','manifest','output']:p.add_argument(name,type=P)
 a=p.parse_args();r=freeze(a.default,a.overrides,a.manifest,a.output)
 print(json.dumps({k:v for k,v in r.items() if k!='artifacts'}))
if __name__=='__main__':main()
