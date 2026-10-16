#!/usr/bin/env python3
"""Fetch the requested official libjxl tag; keep archive, SHA and URL provenance."""
import argparse,hashlib,json,pathlib,tarfile,urllib.request
p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=pathlib.Path);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
url='https://codeload.github.com/libjxl/libjxl/tar.gz/refs/tags/v0.12.0'
archive=a.output/'libjxl-v0.12.0.tar.gz';digest=hashlib.sha256()
with urllib.request.urlopen(url,timeout=60) as source,archive.open('xb') as target:
    while chunk:=source.read(1<<20):target.write(chunk);digest.update(chunk)
with tarfile.open(archive) as source:source.extractall(a.output,filter='data')
(a.output/'source.meta.json').write_text(json.dumps(dict(url=url,tag='v0.12.0',sha256=digest.hexdigest(),upstream='https://github.com/libjxl/libjxl/releases/tag/v0.12.0'),indent=2)+'\n')
print(digest.hexdigest(),archive)
