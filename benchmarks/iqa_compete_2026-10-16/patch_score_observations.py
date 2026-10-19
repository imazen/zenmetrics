#!/usr/bin/env python3
"""Overlay only deferred numerical telemetry onto an immutable private snapshot clone."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot', type=Path)
    p.add_argument('--base', required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    receipt = a.snapshot/'score-observations.meta.json'
    if receipt.exists():
        p.error('preserve existing telemetry overlay receipt')
    base, source = [subprocess.check_output(['git','rev-parse',r],text=True).strip() for r in (a.base,a.source)]
    worker = 'benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete.rs'
    module = 'benchmarks/heaptrack/drivers/cpu_profile/src/iqa_compete_score_observations.rs'
    tree = a.snapshot/'work/zenmetrics'
    old = subprocess.check_output(['git','show',base+':'+worker])
    if (tree/worker).read_bytes() != old or (tree/module).exists():
        raise ValueError('snapshot is not the untouched base worker')
    files = {}
    for name in (worker,module):
        data = subprocess.check_output(['git','show',source+':'+name])
        (tree/name).write_bytes(data)
        files[name] = dict(original_sha256=sha(old) if name==worker else None, adapted_sha256=sha(data))
    receipt.write_text(json.dumps(dict(base_commit=base,source_commit=source,files=files,
        scope='Deferred per-call telemetry only; metric implementations, gates and clocks unchanged'),indent=2)+'\n')


if __name__ == '__main__': main()
