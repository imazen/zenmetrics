#!/usr/bin/env python3
"""Actual driver timing extraction and qualified-output checker plumbing."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from check_benchmark_pair_output import check

binaries, source_case, root=(Path(p).resolve() for p in sys.argv[1:])
root.mkdir(parents=True,exist_ok=False)
case=json.loads(source_case.read_bytes())
for side in ['reference','distorted']:
    shutil.copyfile(source_case.parent/case[side],root/side);case[side]=side
(root/'case.json').write_text(json.dumps(case))
# This is a standalone protocol fixture, not a performance qualification.
(root/'bundle.json').write_text(json.dumps(dict(files={name:{} for name in ['case.json','reference','distorted']})))
env=dict(os.environ,RAYON_NUM_THREADS=str(case['threads']),ZENBENCH_OPERATIONS='2',ZENMETRICS_PAIR_TIMING='1')
outputs=[]
for variant in ['latency','memory']:
    directory=root/variant;directory.mkdir()
    result=subprocess.run([str(binaries/variant),'fleet',str(root/'case.json')],cwd=root,
        env=dict(env,ZENBENCH_RESULT_FILE=str(directory/'driver.json')),capture_output=True,timeout=120)
    (directory/'stdout.txt').write_bytes(result.stdout);(directory/'stderr.txt').write_bytes(result.stderr)
    assert result.returncode==0,result.stderr.decode()
    lines=result.stdout.decode().splitlines()
    value=json.loads(lines[0]);durations=[int(line.removeprefix('METRIC_NS=')) for line in lines[1:]]
    assert len(durations)==2 and durations==value['operation_ns'] and all(n>0 for n in durations)
    assert json.loads((directory/'driver.json').read_bytes())==value['driver']
    outputs.append(value['driver']['correctness'])
assert outputs[0]==outputs[1]
expected=root/'expected.json';expected.write_text(json.dumps(dict(operations=2,correctness=outputs[0])))
check(root/'latency',root/'latency',expected)
try: check(root/'latency',root/'memory',expected)
except ValueError: pass
else: raise AssertionError('instrumented output was accepted for paired latency')
print(json.dumps(dict(timing_line_extraction='exact',instrumentation_output_parity='exact',
                      instrumented_pair_output='refused',evidence='functional protocol only')))
