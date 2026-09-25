#!/usr/bin/env python3
"""Run on the authorized worker through its serialized heavy wrapper."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/var/tmp/gmsd-chroma')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


assert shutil.disk_usage('/home').free >= 20*1024**3
inputs = ROOT/'colour_v2/raw_pairs.tsv'
inventory = json.loads((ROOT/'peer_transfer_manifest.json').read_text())
for relative,digest in inventory['file_sha256'].items():
    assert sha(ROOT/relative) == digest, relative
build = json.loads((ROOT/'fast_peer_build.json').read_text())
assert build['commit'] == 'f011259a0cd2fb7f538e08398ba41c0ccd0dfcb4'
binary = ROOT/'peer-target/release/gmsd-chroma-fast-ssim2-peer'
assert sha(binary) == build['binary_sha256']
output = ROOT/'colour_v2/fast_ssim2.tsv'
assert not output.exists()
worker = ROOT/'target/release/zenfleet-worker'
assert sha(worker) == json.loads((ROOT/'peer_worker_build.json').read_text())['binary_sha256']
run = ROOT/'peer_worker_run'
run.mkdir(exist_ok=False)
command = [str(worker),'--manifest',str(ROOT/'peer_jobs.json'),
           '--ledger-out',str(run/'ledger.parquet'),'--blobs',str(run/'blobs'),
           '--exec',str(ROOT/'peer_jobexec.py'),'--worker','gmsd-chroma-worker-2',
           '--provider','authorized-worker']
print('starting_existing_zenfleet_worker',flush=True)
with (run/'worker.log').open('wb') as log:
    process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                               env={**os.environ,'ZEN_CHUNK_WALL_SEC':'0',
                                    'TMPDIR':str(ROOT/'tmp'),'PYTHONDONTWRITEBYTECODE':'1'})
    while process.poll() is None:
        progress = ROOT/'peer_progress.jsonl'
        done = len(progress.read_text().splitlines()) if progress.exists() else 0
        print('peer_pairs_completed',done,flush=True)
        time.sleep(15)
assert process.returncode == 0, 'zenfleet worker failed; retained worker.log'
# The owner creates one content-addressed result blob per successful cell.
values = []
for path in sorted((run/'blobs').rglob('*')):
    if path.is_file():
        assert path.name == sha(path), 'worker content address mismatch'
        values.append(json.loads(path.read_text()))
assert len(values) == 1539, 'worker did not complete the full population'
with output.open('x') as f:
    writer = csv.DictWriter(f,fieldnames=['pair_key','fast_ssim2'],delimiter='\t',lineterminator='\n')
    writer.writeheader()
    writer.writerows(sorted(values,key=lambda r:r['pair_key']))
with inputs.open() as f:
    expected = {r['pair_key'] for r in csv.DictReader(f,delimiter='\t')}
with output.open() as f:
    rows = list(csv.DictReader(f,delimiter='\t'))
assert len(rows) == len(expected) == 1539
assert {r['pair_key'] for r in rows} == expected
assert all(math.isfinite(float(r['fast_ssim2'])) for r in rows)
manifest = dict(producer='fast-ssim2-cpu',table_sha256=sha(output),
                source_sha256=inventory['file_sha256'],
                source_commit=build['commit'],binary_sha256=sha(binary),
                input_sha256=sha(inputs),build_manifest_sha256=sha(ROOT/'fast_peer_build.json'),
                join_description='Exact pair_key join to all frozen TRAIN rows; identical content-hashed RGB8 inputs.',
                rows=len(rows),roles=['TRAIN'],human_labels_read=False,
                worker_binary_sha256=sha(worker),worker_argv=command,
                ledger_sha256=sha(run/'ledger.parquet'),worker_log_sha256=sha(run/'worker.log'),
                job_manifest_sha256=sha(ROOT/'peer_jobs.json'))
output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
# Enumerate only this lane's remote output root, without following symlinks.
paths = []
for directory, dirs, files in os.walk(ROOT,followlinks=False):
    paths.extend(str(Path(directory)/name) for name in dirs+files)
(ROOT/'peer_remote_footprint.json').write_text(json.dumps(sorted(paths),indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='source_sha256'},sort_keys=True))
