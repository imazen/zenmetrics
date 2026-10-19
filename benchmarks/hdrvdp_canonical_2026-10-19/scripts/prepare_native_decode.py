"""Copy the retained locked native probe workspace; add only a decode binary."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

source,dest=map(Path,sys.argv[1:]);assert not dest.exists()
assert not (source/'target').exists(), 'do not copy Cargo outputs'
shutil.copytree(source,dest)
original={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
assert all(hashlib.sha256((dest/n).read_bytes()).hexdigest()==h for n,h in original.items())
target=dest/'vdpprobe/src/bin/native-decode.rs';target.parent.mkdir(exist_ok=True)
assert not target.exists();shutil.copy2(Path(__file__).with_name('native_decode.rs'),target)
(dest/'decode-source-sha256.json').write_text(json.dumps(original,indent=2)+'\n')
