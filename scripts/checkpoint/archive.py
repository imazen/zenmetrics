#!/usr/bin/env python3
"""Archive a completed checkpoint without losing native executable permissions."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile

package, output = map(Path, sys.argv[1:])
manifest = json.loads((package / 'checkpoint.json').read_text())
actual = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
if actual != set(manifest['files']) | {'checkpoint.json'}:
    raise ValueError('Unlisted or missing checkpoint files')
for relative, expected in manifest['files'].items():
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Manifest path escapes package')
    path = package / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'Missing or nonregular artifact: {relative}')
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError(f'Artifact hash mismatch: {relative}')
with output.open('xb') as raw:
    with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as archive:
            for path in sorted(package.rglob('*')):
                if path.is_symlink(): raise ValueError('Symlink refused')
                info = archive.gettarinfo(str(path), arcname='checkpoint/' + str(path.relative_to(package)))
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mtime = 0
                if path.is_file():
                    with path.open('rb') as stream: archive.addfile(info, stream)
                else: archive.addfile(info)
with output.open('rb') as stream:
    checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
output.with_name(output.name + '.sha256').write_text(checksum + '  ' + output.name + '\n')
