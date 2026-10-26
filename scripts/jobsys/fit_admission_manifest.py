#!/usr/bin/env python3
"""Change only fit scheduling hints and prove native job IDs remain identical."""
import argparse
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--ctl', required=True)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--peak-mem-bytes', type=int, required=True)
p.add_argument('--threads', type=int, required=True)
a = p.parse_args()
if a.peak_mem_bytes <= 0 or a.threads <= 0:
    p.error('positive scheduling envelopes required')
jobs = json.loads(a.manifest.read_text())
if not jobs or any(j['kind']['kind'] != 'fit_cell' for j in jobs):
    raise ValueError('only declared FitCells may receive these scheduling hints')
before = subprocess.check_output([a.ctl, 'ids', '--manifest', str(a.manifest)])
for job in jobs:
    job['hint'].update(peak_mem_bytes=a.peak_mem_bytes, threads=a.threads)
with a.out.open('x') as f:
    json.dump(jobs, f, indent=2)
    f.write('\n')
after = subprocess.check_output([a.ctl, 'ids', '--manifest', str(a.out)])
if before != after:
    raise ValueError('scheduling envelope unexpectedly changed native identities; refuse publication')
print(json.dumps(dict(cells=len(jobs),native_ids_unchanged=True,
                      peak_mem_bytes=a.peak_mem_bytes,threads=a.threads,out=str(a.out))))
