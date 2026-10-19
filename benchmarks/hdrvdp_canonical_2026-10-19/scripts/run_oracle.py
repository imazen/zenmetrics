"""Run one serialized, resource-capped oracle trial and retain source/argv receipts."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

args = sys.argv[1:]
assert len(args) >= 9
root = Path(__file__).resolve().parents[3]
tree, pairs, output = map(Path, args[:3])
assert not any(p.exists() for p in [output, output.with_suffix('.meta.json'),
               output.with_suffix('.log'), output.with_suffix('.driver.m')]), (
    'retain existing attempts; choose a new output path')
output.parent.mkdir(parents=True, exist_ok=True)
files = {str(f.relative_to(tree)): hashlib.sha256(f.read_bytes()).hexdigest()
         for f in sorted(tree.rglob('*')) if f.is_file()}
manifest = json.dumps(files, sort_keys=True).encode()
source_script = Path(__file__).with_name('oracle.m')
script = output.with_suffix('.driver.m')
assert not script.exists()
script.write_bytes(source_script.read_bytes())
home = Path.home()
cmd = ['flock', str(home/'tmp/dev-heavy.lock'),
       str(home/'work/claudehints/scripts/run-heavy'), '--mem', '24G', '--',
       'octave', '--quiet', str(script), *args]
env = dict(os.environ, TMPDIR=str(home/'tmp/hdrvdpcx'), OPENBLAS_NUM_THREADS='1')
meta = dict(started=datetime.now(timezone.utc).isoformat(), argv=cmd,
            host=os.uname().nodename, source_files=files,
            source_tree_sha256=hashlib.sha256(manifest).hexdigest(),
            driver_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
            pairs_sha256=hashlib.sha256(pairs.read_bytes()).hexdigest(),
            statistics_path=env.get('HDRVDP_STATISTICS_PATH'),
            fft_axis=env.get('HDRVDP_FFT_AXIS'),
            baseline_commit=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],cwd=root,text=True).strip())
meta_path = output.with_suffix('.meta.json')
meta_path.write_text(json.dumps(meta, indent=2)+'\n')
lock = root/'.workongoing'
assert 'hdrvdpcx' in lock.read_text()
lock.write_text(datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')+' hdrvdpcx oracle '+output.stem+'\n')
with output.with_suffix('.log').open('w') as log:
    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
meta.update(returncode=result.returncode, finished=datetime.now(timezone.utc).isoformat())
if output.exists():
    meta['output_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
meta_path.write_text(json.dumps(meta, indent=2)+'\n')
print(json.dumps({k:v for k,v in meta.items() if k != 'source_files'}, indent=2),flush=True)
sys.exit(result.returncode)
