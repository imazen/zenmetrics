#!/usr/bin/env python3
"""Freeze a completed benchmark build and its actual Cargo/source provenance."""
import argparse,datetime,json,os,pathlib,re,shutil
from run import sha
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__)
for key in ['snapshot','build_log','output','metadata']:p.add_argument(key,type=P)
a=p.parse_args()
text=a.build_log.read_text()
if not re.search(r'run-heavy: done rc=0 ',text):raise RuntimeError('Successful bounded build receipt required')
if a.output.exists() or a.metadata.exists():raise FileExistsError('Preserve previously frozen binaries and metadata')
source=a.snapshot/'target/release/iqa-compete'
fingerprints=list((a.snapshot/'target/release/.fingerprint').glob('cpu-profile-*/bin-iqa-compete.json'))
fingerprint=max(fingerprints,key=lambda path:path.stat().st_mtime_ns)
config=json.loads(fingerprint.read_text())
flags=config['rustflags']
if any('target-cpu=native' in flag for flag in flags):raise RuntimeError('Native CPU flags are excluded')
if 'study-gpu-cuda' not in config['features']:raise RuntimeError('Expected full CUDA coverage build')
pending=a.output.with_name(a.output.name+'.publishing')
with source.open('rb') as src,pending.open('xb') as dst:shutil.copyfileobj(src,dst,1<<20)
digest=sha(source)
if sha(pending)!=digest:raise RuntimeError('Frozen binary differs from completed build')
record=dict(binary=str(a.output.resolve()),binary_sha256=digest,bytes=pending.stat().st_size,
 build_log=str(a.build_log.resolve()),build_log_sha256=sha(a.build_log),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 cargo_fingerprint=config,cargo_fingerprint_source=str(fingerprint),cargo_fingerprint_sha256=sha(fingerprint),
 driver_source_sha256=sha(a.snapshot/'work/zenmetrics/benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete.rs'),
 admission='Owner coverage-first: locked process accounting; load recorded without load-only refusal',
 sampling='5 at CID22/AIC-small/AIC-medium, 3 at AIC-large/2K/4K, 1 at 8K; one bounded scope; at most one contamination rerun')
for key,name in [('model_adapter','model-preload.meta.json'),('ort_threads','ort-threads.meta.json'),('luma_maps','luma-maps.meta.json')]:record[key]=json.loads((a.snapshot/name).read_text())
os.chmod(pending,0o555)
with a.metadata.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
os.rename(pending,a.output)
print(json.dumps({k:record[k] for k in ['binary','binary_sha256','bytes','created_utc']}))
