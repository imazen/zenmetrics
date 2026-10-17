#!/usr/bin/env python3
"""Fingerprint the exact exported inference artifacts without loading models."""
import argparse,hashlib,json,pathlib
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('default',type=P);p.add_argument('overrides',type=P);p.add_argument('output',type=P);a=p.parse_args();assert not a.output.exists()
directories={str(a.default),*json.loads(a.overrides.read_text()).values()};cache={};rows=[]
for directory in sorted(directories):
 root=P(directory)
 for path in sorted(root.iterdir()):
  if not path.is_file() or path.suffix not in ['.onnx','.rten','.meta','.f32','.json']:continue
  source=path.resolve();info=source.stat()
  if source not in cache:
   h=hashlib.sha256()
   with source.open('rb') as f:
    for block in iter(lambda:f.read(1<<20),b''):h.update(block)
   cache[source]=h.hexdigest()
  rows.append(dict(directory=directory,file=path.name,source=str(source),bytes=info.st_size,mtime_ns=info.st_mtime_ns,sha256=cache[source]))
a.output.write_text(json.dumps(dict(artifacts=rows),indent=2)+'\n');print(json.dumps(dict(files=len(rows),unique_sources=len(cache),manifest_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())))
