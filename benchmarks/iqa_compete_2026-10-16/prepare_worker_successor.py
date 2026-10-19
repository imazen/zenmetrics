#!/usr/bin/env python3
"""Freeze a replacement worker/controller while preserving hashed input and asset bytes."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from mac_service import verify
from prepare_coverage import clone
from prepare_durable import HERE, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'binary', 'build', 'output'):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    verify(a.source)
    build = json.loads(a.build.read_bytes())
    if sha(a.binary) != build['binary_sha256']:
        raise ValueError('replacement worker differs from receipt')
    inventory = subprocess.check_output([str(a.binary.resolve()), 'inventory'])
    import hashlib
    if hashlib.sha256(inventory).hexdigest() != build['inventory_sha256']:
        raise ValueError('replacement inventory differs from receipt')
    manifest = json.loads((a.source/'manifest.json').read_bytes())
    names = {r['metric_name'] for r in json.loads(inventory) if r['compiled']}
    if any(c['metric_name'] not in names for c in manifest['cells']):
        raise ValueError('replacement lacks requested routes')
    clone(a.source, a.output)
    shutil.copy2(a.binary, a.output/'bin/iqa-compete')
    shutil.copy2(a.build, a.output/'build.json')
    scripts = ('durable.py','admission_recovery.py','mac_service.py','select_durable.py')
    for name in scripts:
        shutil.copy2(HERE/name, a.output/name)
    receipt = dict(source_manifest_sha256=sha(a.source/'manifest.json'),
        source_binary_sha256=manifest['binary_sha256'], replacement_binary_sha256=build['binary_sha256'],
        replacement_build_sha256=sha(a.build), preparation_sha256=sha(Path(__file__)),
        input_contract='Original input and model assets unchanged; execution cells selected by immutable successor scheduler')
    (a.output/'worker-successor.json').write_text(json.dumps(receipt,indent=2)+'\n')
    manifest['binary_sha256'] = build['binary_sha256']
    manifest['metadata'].update(source_commit=build['source_commit'], build_receipt_sha256=sha(a.build),
        features=build['features'], runner_sha256={n:sha(a.output/n) for n in scripts if n!='select_durable.py'},
        worker_successor_sha256=sha(a.output/'worker-successor.json'))
    for name in ('bin/iqa-compete','build.json',*scripts,'worker-successor.json'):
        if name in manifest['input_sha256'] or name=='worker-successor.json':
            manifest['input_sha256'][name] = sha(a.output/name)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    hashes = {f.relative_to(a.output).as_posix():sha(f) for f in a.output.rglob('*')
              if f.is_file() and f.name!='SHA256SUMS.json'}
    (a.output/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2)+'\n')
    verify(a.output)
    print(json.dumps(dict(bundle=str(a.output),manifest_sha256=hashes['manifest.json'],binary_sha256=build['binary_sha256'])))


if __name__ == '__main__': main()
