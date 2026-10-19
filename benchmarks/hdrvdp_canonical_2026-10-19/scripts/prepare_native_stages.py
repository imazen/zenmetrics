"""Freeze unchanged Rust interpolation/spectral modules and data, build a tap."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

repo=Path(__file__).resolve().parents[3]; dest=Path(sys.argv[1]); assert not dest.exists()
dest.mkdir(parents=True)
source=repo/'crates/hdrvdp'
for name in ['src/interp.rs','src/spectral.rs','src/v3/spectral.rs']:
    p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,p)
shutil.copytree(source/'data',dest/'data')
files={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}
assert all(hashlib.sha256((source/n).read_bytes()).hexdigest()==v for n,v in files.items())
(dest/'source-sha256.json').write_text(json.dumps(files,indent=2)+'\n')
text=Path(__file__).with_name('native_stages.rs.in').read_text()
for key,name in [('INTERP','interp.rs'),('SPECTRAL2','spectral.rs'),('SPECTRAL3','v3/spectral.rs')]:
    text=text.replace('@'+key+'@',str(dest/'src'/name))
(dest/'probe.rs').write_text(text)
subprocess.run(['rustc','--edition=2024','-O',str(dest/'probe.rs'),'-o',str(dest/'native-stages')],check=True)
print('PASS: source hashes identical; native spectral probe built')
