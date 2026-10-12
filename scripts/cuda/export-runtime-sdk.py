#!/usr/bin/env python3
"""Export distro-owned CUDA headers and NVRTC into a scoped runtime SDK.

The host driver stays on the destination. This does not install system packages
or select a device; CUDA_PATH and LD_LIBRARY_PATH select the exported runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--headers-package', default='nvidia-cuda-dev')
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        parser.error('output already exists; preserve the previous SDK')
    package = subprocess.check_output(
        ['dpkg-query', '-W', '-f=${Package} ${Version}\n', args.headers_package], text=True).strip()
    listing = subprocess.check_output(['dpkg-query', '-L', args.headers_package], text=True)
    headers = [Path(line) for line in listing.splitlines()
               if line.startswith('/usr/include/') and Path(line).is_file()]
    if Path('/usr/include/cuda_runtime.h') not in headers:
        raise ValueError('selected package does not own cuda_runtime.h')
    cache = subprocess.check_output(['ldconfig', '-p'], text=True)
    libraries = [Path(line.split('=>', 1)[1].strip()).resolve(strict=True)
                 for line in cache.splitlines() if line.strip().startswith('libnvrtc.so ') and '=>' in line]
    if len(set(libraries)) != 1:
        raise ValueError('need one unambiguous cached unversioned libnvrtc.so')
    nvrtc = libraries[0]
    builtins = sorted({p.resolve(strict=True) for p in nvrtc.parent.glob('libnvrtc-builtins.so*') if p.is_file()})
    if len(builtins) != 1:
        raise ValueError('need one matching NVRTC builtins library; mixed versions require explicit separation')
    if builtins[0].name.split('.so.', 1)[1] != nvrtc.name.split('.so.', 1)[1]:
        raise ValueError('NVRTC and builtins versions differ')
    root.mkdir(parents=True)
    files = {}

    def copy(source, relative):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        after = hashlib.sha256(destination.read_bytes()).hexdigest()
        if before != after:
            raise ValueError(f'copy changed bytes: {relative}')
        files[str(relative)] = dict(sha256=after, bytes=destination.stat().st_size)

    for header in headers:
        copy(header, Path('include') / header.relative_to('/usr/include'))
    for library in [nvrtc, builtins[0]]:
        copy(library, Path('lib64') / library.name)
    aliases = {}
    for library in [nvrtc, builtins[0]]:
        base = library.name.split('.so', 1)[0] + '.so'
        major = library.name.split('.so.', 1)[1].split('.', 1)[0]
        names = {base, base + '.' + major}
        names.update(p.name for p in library.parent.glob(base + '*')
                     if p.is_file() and p.resolve(strict=True) == library and p.name != library.name)
        names.discard(library.name)
        for name in sorted(names):
            (root / 'lib64' / name).symlink_to(library.name)
            aliases['lib64/' + name] = library.name
    manifest = dict(schema=1, headers_package=package, files=files, aliases=aliases,
                    resolution=['dpkg-query -W/-L HEADERS_PACKAGE', 'ldconfig -p'],
                    boundary='NVRTC and headers only; destination uses its own NVIDIA driver')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(headers=len(headers), libraries=len(files)-len(headers), output=str(root))))


if __name__ == '__main__':
    main()
