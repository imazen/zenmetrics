"""Source/build receipt wiring for a NEW observer cohort; never run a binary."""
import argparse
import json
from pathlib import Path
import shutil
from panel import digest
import patch_nr_result

DRIVER='benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete.rs'
BOUNDARY='paired-nr-timed-validation-v1'
PROTOCOL='iqa-row-nr-v2'


def stage(build):
    build=Path(build);source=build/'work/zenmetrics'/DRIVER
    # Source-only callers must supply their own isolated build snapshot.
    if source.resolve()==Path(__file__).resolve().parents[2]/DRIVER:
        raise ValueError('cannot stage into source checkout')
    receipt_dir=build/'nr-overlay';receipt_dir.mkdir(exist_ok=False)
    observer=source.with_name('nr_observer.rs')
    if observer.exists():raise ValueError('observer already present')
    before=source.read_bytes();(receipt_dir/'original.rs').write_bytes(before)
    source.write_text(patch_nr_result.patched(before.decode()))
    shutil.copy2(Path(__file__).with_name('nr_observer.rs'),observer)
    receipt=dict(schema=1,compiled=False,protocol=PROTOCOL,timing_boundary=BOUNDARY,
                 original_sha256=digest(receipt_dir/'original.rs'),
                 driver_sha256=digest(source),observer_sha256=digest(observer),
                 emitter_sha256=digest(Path(__file__).with_name('patch_nr_result.py')))
    (receipt_dir/'stage.json').write_text(json.dumps(receipt,indent=2))
    return receipt


def finish(build):
    build=Path(build);stage_receipt=json.loads((build/'nr-overlay/stage.json').read_text())
    source=build/'work/zenmetrics'/DRIVER
    if digest(source)!=stage_receipt['driver_sha256'] or digest(source.with_name('nr_observer.rs'))!=stage_receipt['observer_sha256']:
        raise ValueError('staged observer source changed')
    binary=build/'target/release/iqa-compete'
    raw=binary.read_bytes()
    if BOUNDARY.encode() not in raw or b'paired-nr-v2' not in raw:
        raise ValueError('binary lacks observer protocol markers; never relabel baseline')
    names=['build-command.txt','features.txt','metadata.json','rustc.txt','cargo.txt','nr-overlay/stage.json',
           'work/zenmetrics/'+DRIVER,'work/zenmetrics/'+str(Path(DRIVER).with_name('nr_observer.rs')),'target/release/iqa-compete']
    receipt=dict(schema=1,protocol=PROTOCOL,timing_boundary=BOUNDARY,build_success=True,
        provenance='builder calls finish only after successful cargo build; not reproducibility proof',
        files=[dict(path=n,sha256=digest(build/n)) for n in names])
    path=build/'nr-cohort.json'
    with path.open('x') as f:json.dump(receipt,f,indent=2)
    return receipt


def validate(build):
    build=Path(build);r=json.loads((build/'nr-cohort.json').read_text())
    if r.get('schema')!=1 or r.get('protocol')!=PROTOCOL or r.get('timing_boundary')!=BOUNDARY or r.get('build_success') is not True:
        raise ValueError('observer build receipt mismatch')
    expected={'build-command.txt','features.txt','metadata.json','rustc.txt','cargo.txt','nr-overlay/stage.json','work/zenmetrics/'+DRIVER,'work/zenmetrics/'+str(Path(DRIVER).with_name('nr_observer.rs')),'target/release/iqa-compete'}
    if len(r['files'])!=len(expected) or {f['path'] for f in r['files']}!=expected:raise ValueError('incomplete observer build binding')
    for f in r['files']:
        if digest(build/f['path'])!=f['sha256']:raise ValueError('observer build artifact changed')
    raw=(build/'target/release/iqa-compete').read_bytes()
    if BOUNDARY.encode() not in raw or b'paired-nr-v2' not in raw:raise ValueError('observer markers missing')
    return r

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['stage','finish','validate']);p.add_argument('build',type=Path);a=p.parse_args()
    print(json.dumps(globals()[a.operation](a.build)),flush=True)
