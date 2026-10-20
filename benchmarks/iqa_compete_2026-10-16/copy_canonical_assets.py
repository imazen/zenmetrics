#!/usr/bin/env python3
"""Copy explicit reference assets into a new owned directory, preserving sources.

Run under shared admission and the resource wrapper. A manifest names the host
and each source/name pair; `python_only` copies source modules without weights.
"""
import argparse
import json
from pathlib import Path
import subprocess

from compare_canonical import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    a.output.mkdir()
    records = []
    for item in manifest['assets']:
        name, source = item['name'], item['source']
        if name in ('.','..') or Path(name).name != name or not Path(source).is_absolute():
            raise ValueError('basename destination and absolute source required')
        destination = a.output/name
        command = ['rsync','-a','--protect-args']
        if item.get('python_only'):
            command += ['--prune-empty-dirs','--exclude=.git','--include=*/',
                        '--include=*.py','--include=*.txt','--exclude=*']
        if item.get('directory'):
            source += '/'
            destination.mkdir()
        command += [manifest['host']+':'+source,str(destination)]
        subprocess.run(command,check=True)
        files = sorted(path for path in destination.rglob('*') if path.is_file()) if destination.is_dir() else [destination]
        for path in files:
            records.append(dict(file=str(path.relative_to(a.output)),bytes=path.stat().st_size,sha256=sha(path)))
    (a.output/'copy-receipt.json').write_text(json.dumps(dict(manifest_sha256=sha(a.manifest),artifacts=records),indent=2)+'\n')
    print(json.dumps(dict(files=len(records),bytes=sum(row['bytes'] for row in records))))


if __name__=='__main__':main()
