#!/usr/bin/env python3
"""Fetch the pinned onnxruntime-web package and copy its JSPI runtime files.

The tarball is checked against the npm registry's sha512 integrity value
recorded here before anything is extracted.
"""
import base64
import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

VERSION = '1.30.0'
URL = f'https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{VERSION}.tgz'
INTEGRITY = 'sha512-q0y+JrrtukXSzsBWEMccVfqX25LRmosXHF+CaRJmg8pZClzcV7svNc4rKY3jL02Vb7QmRMDs1SigqR4CXAfKYQ=='
FILES = ('dist/ort.jspi.min.mjs', 'dist/ort-wasm-simd-threaded.jspi.mjs',
         'dist/ort-wasm-simd-threaded.jspi.wasm')
# The package declares MIT but ships no licence file; this is the
# onnxruntime repository's LICENSE at tag v1.30.0.
LICENSE = Path(__file__).resolve().parents[2] / 'web/metrics/onnxruntime-LICENSE.txt'
PAGES_FILE_LIMIT = 25 * 1024 * 1024


def main():
    out = Path(sys.argv[1])
    if out.exists():
        sys.exit(f'{out} exists; refusing to overwrite')
    with urllib.request.urlopen(URL, timeout=120) as response:
        data = response.read()
    got = 'sha512-' + base64.b64encode(hashlib.sha512(data).digest()).decode()
    if got != INTEGRITY:
        sys.exit(f'onnxruntime-web {VERSION} integrity mismatch: {got}')
    out.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        for name in FILES:
            member = archive.getmember('package/' + name)
            payload = archive.extractfile(member).read()
            if len(payload) > PAGES_FILE_LIMIT:
                sys.exit(f'{name} is {len(payload)} bytes, over the Pages file limit')
            (out / Path(name).name).write_bytes(payload)
    (out / 'LICENSE').write_bytes(LICENSE.read_bytes())
    (out / 'VERSION').write_text(f'onnxruntime-web {VERSION}\n{URL}\n{INTEGRITY}\n')


if __name__ == '__main__':
    main()
