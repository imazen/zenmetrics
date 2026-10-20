#!/usr/bin/env python3
"""Build the original mDCTpsnr sources with an explicit resident RGB caller."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from compare_canonical import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    source = a.source.resolve(strict=True)
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain']).strip():
        raise ValueError('original source must be clean')
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    a.output.mkdir()
    wrapper = Path(__file__).resolve().with_name('mdctpsnr_rgb_compare.cpp')
    compiler = Path(shutil.which('c++')).resolve(strict=True)
    sources = [path for directory in ['global', 'io', 'img', 'ctrafo', 'dct', 'measure']
               for path in sorted((source/directory).glob('*.cpp'))]
    flags = subprocess.check_output(['pkg-config', '--cflags', '--libs', 'libhwy'], text=True).split()
    binary = a.output.resolve()/'mdctpsnr-rgb'
    command = [str(compiler), '-O3', '-DNDEBUG', '-ffast-math', '-std=c++17',
               '-DHWY_SHARED_DEFINE', '-I'+str(source), str(wrapper), *map(str, sources),
               *flags, '-pthread', '-lm', '-o', str(binary)]
    subprocess.run(command, check=True)
    (a.output/'build.json').write_text(json.dumps(dict(revision=revision, command=command,
        compiler_sha256=sha(compiler), wrapper_sha256=sha(wrapper), binary_sha256=sha(binary),
        source_sha256={str(path.relative_to(source)): sha(path) for path in sources},
        scope='Resident P6 backing from unchanged RGB8 bytes; original OpenPNM conversion and Pooling.Measure, including object lifetimes; no file I/O or process launch in clock; Highway runtime dispatch'), indent=2)+'\n')
    print(str(binary))


if __name__ == '__main__':
    main()
