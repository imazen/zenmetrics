"""Copy the existing probe/runtime source into an isolated build-only workspace.
Usage: prepare_rust_probe.py NEW_DIRECTORY
The shipped workspace/lock remain untouched. Unused test-only dependencies
are omitted from the scratch manifest; no runtime source file is edited.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tomllib

repo=Path(__file__).resolve().parents[3];dest=Path(sys.argv[1]);assert not dest.exists()
dest.mkdir(parents=True)
for name,source in [('hdrvdp',repo/'crates/hdrvdp'),('zenmetrics-abstractions',repo/'crates/zenmetrics-abstractions'),('vdpprobe',repo/'benchmarks/aic_confsearch_2026-10-17/vdpprobe')]:
    shutil.copytree(source,dest/name)
    original={str(p.relative_to(source/'src')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'src').rglob('*') if p.is_file()}
    copied={str(p.relative_to(dest/name/'src')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (dest/name/'src').rglob('*') if p.is_file()}
    assert original==copied
    (dest/(name+'-src-sha256.json')).write_text(json.dumps(original,indent=2)+'\n')
p=dest/'hdrvdp/Cargo.toml';s=p.read_text();p.write_text(s[:s.index('[dev-dependencies]')]+s[s.index('[lints]'):])
c=tomllib.loads((repo/'Cargo.toml').read_text());w=c['workspace']
lines=['[workspace]','members=["hdrvdp","zenmetrics-abstractions","vdpprobe"]','resolver="3"','[workspace.package]']
for k in ['edition','authors','homepage','repository','rust-version']:
    lines.append(k+' = '+json.dumps(w['package'][k]))
lines.append('[workspace.dependencies]')
for k in ['archmage','magetypes','image','rgb']:
    d=w['dependencies'][k]
    value='{'+', '.join(key+' = '+json.dumps(v) for key,v in d.items())+'}' if isinstance(d,dict) else json.dumps(d)
    lines.append(k+' = '+value)
lines.extend(['[workspace.lints.rust]','unsafe_code="forbid"','[patch.crates-io]'])
d=c['patch']['crates-io']['zenpixels']
lines.append('zenpixels = {'+', '.join(key+' = '+json.dumps(v) for key,v in d.items())+'}')
(dest/'Cargo.toml').write_text('\n'.join(lines)+'\n')
print('Prepared isolated probe; all runtime source hashes agree.')
