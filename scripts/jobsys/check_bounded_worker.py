#!/usr/bin/env python3
"""Exercise bounded worker passes against local synthetic jobs and durable ledgers."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=Path, required=True)
    p.add_argument('--scratch', type=Path, required=True)
    a = p.parse_args()
    a.scratch.mkdir(parents=True, exist_ok=False)
    worker = str(a.worker.resolve(strict=True))
    capability = json.loads(subprocess.check_output([worker, 'capabilities'], text=True))
    assert capability.get('max_chunks_per_pass') is True
    manifest = a.scratch/'manifest.json'
    manifest.write_text(json.dumps([
        dict(kind=dict(kind='encode',codec='zenjpeg',q=80,knobs='{}',hdr=False),
             inputs=[f'{i:064x}'], cell=dict(image_path=f'synthetic-{i}.png',
             codec='zenjpeg',q=80,knob_tuple_json='{}'), requires=[])
        for i in range(2)]))
    env = dict(os.environ, ZEN_MAX_CHUNKS_PER_PASS='1', ZEN_CHUNK_WALL_SEC='1',
               TMPDIR=str(a.scratch), ZEN_CLAIM_ORDER='manifest')
    for n, expected in enumerate((1, 1, 0), 1):
        command = [worker,'--manifest',str(manifest),'--ledger-out',
                   str(a.scratch/f'pass{n}.parquet'),'--blobs',str(a.scratch/'blobs'),
                   '--exec','/usr/bin/cat' if n < 3 else '/usr/bin/false']
        run = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=True)
        (a.scratch/f'pass{n}.log').write_text(run.stdout)
        assert f'done={expected} failed=0' in run.stdout, run.stdout
    report = dict(capability=capability, done_per_pass=[1,1,0],
                  durable_sidecar_reload=True, synthetic_only=True, scientific_fits=0)
    (a.scratch/'VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
