#!/usr/bin/env python3
"""Independent heaptrack global-peak control of a real instrumented IQA case."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

binary, case, root = [Path(s).resolve() for s in sys.argv[1:]]
root.mkdir(parents=True,exist_ok=False)
config=json.loads(case.read_bytes())
env=dict(os.environ,RAYON_NUM_THREADS=str(config['threads']),ZENBENCH_OPERATIONS='1')
env.pop('ZENBENCH_RESULT_FILE',None)
with (root/'driver.stdout').open('wb') as out, (root/'driver.stderr').open('wb') as err:
    subprocess.run(['heaptrack','--record-only','-o',str(root/'trace'),str(binary),'fleet',str(case)],
                   stdout=out,stderr=err,env=env,check=True,timeout=120)
traces=list(root.glob('trace*'))
assert len(traces)==1,traces
with (root/'analysis.txt').open('wb') as out:
    subprocess.run(['heaptrack_print','-f',str(traces[0]),'--print-flamegraph',str(root/'peak.stacks'),
                    '--flamegraph-cost-type=peak'],stdout=out,check=True,timeout=120)
rows=[json.loads(line) for line in (root/'driver.stdout').read_text().splitlines() if line.startswith('{')]
assert len(rows)==1, rows
reported=rows[0]['lifetime']['lifetime_peak_bytes']
with (root/'peak.stacks').open() as stacks:
    observed=sum(int(line.rsplit(' ',1)[1]) for line in stacks if 'HeapProfiler' in line)
assert observed==reported,(observed,reported)
record={'case':config,'ledger_bytes':reported,'heaptrack_rust_global_peak_bytes':observed,
        'binary_sha256':hashlib.file_digest(binary.open('rb'),'sha256').hexdigest(),
        'scope':'whole-process control; phase peaks are not inferred from heaptrack global peak'}
(root/'control.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
