#!/usr/bin/env python3
"""Append owned workspace/scratch paths to the required footprint manifest.

Metadata only; does not follow symlinks, read corpus files or remove files.
Run under heavy once dependency/container caches exist (many directory
entries). Source hashes are separately recorded by command checkpoints.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess

subprocess.run(['python3','scripts/gmsd-chroma/preflight.py'],check=True)
manifest = Path('/home/lilith/tmp/devin/rev4_gmsd-chroma_manifest.tsv')
seen = set()
if manifest.exists():
    for line in manifest.read_text().splitlines():
        columns = line.split('\t')
        if len(columns) >= 2:
            seen.add(columns[1])
workspace = Path('/home/lilith/work/zen/zenmetrics--gmsd-chroma')
relative = subprocess.check_output(
    ['jj', 'diff', '--from', 'be6e8a96', '--name-only'], cwd=workspace, text=True).splitlines()
paths = {str(workspace/p) for p in relative}
paths.update(['/home/lilith/work/zen/zensim/.workongoing', str(workspace/'.workongoing')])
errors = []
for directory, dirs, files in os.walk('/var/tmp/gmsd-chroma', followlinks=False,
                                     onerror=lambda e: errors.append(str(e))):
    paths.update(str(Path(directory)/p) for p in dirs+files)
# The throwaway Docker daemon creates root-owned cache directories. Enumerate
# names only, under this lane's root, to include those in the same manifest.
if errors:
    code = """
import json, os
root='/var/tmp/gmsd-chroma'
paths=[]
def fail(error):
    raise error
for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
    paths.extend(os.path.join(directory,p) for p in dirs+files)
print(json.dumps(paths))
"""
    names = json.loads(subprocess.check_output(['sudo','-n','python3','-c',code],text=True))
    assert all(p.startswith('/var/tmp/gmsd-chroma/') for p in names)
    paths.update(names)
    errors.clear()
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
with manifest.open('a') as f:
    for path in sorted(paths-seen):
        f.write(f'{now}\t{path}\tcreated-or-modified\n')
print('new_manifest_paths', len(paths-seen))
print('inaccessible_paths', len(errors))
for error in errors:
    print(error)
if errors:
    raise SystemExit(1)
