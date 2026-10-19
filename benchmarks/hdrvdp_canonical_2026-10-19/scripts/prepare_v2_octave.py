"""Prepare a separately identified Octave oracle tree; never edit the release.
Usage: prepare_v2_octave.py RELEASE_TREE NEW_TREE OCTAVE_HEADERS
Only the two unsupported ifft2 symmetric arguments change. Original upstream
pyramid C sources are compiled as Octave MEX, preserving the in-place route.
"""
from pathlib import Path
import shutil
import subprocess
import sys

source, dest, headers = map(Path, sys.argv[1:])
assert not dest.exists()
shutil.copytree(source, dest)
for name in ['fast_conv_fft.m', 'fast_gauss.m']:
    p = dest/name
    s = p.read_text()
    assert s.count(", 'symmetric'") == 1
    p.write_text(s.replace(", 'symmetric'", ''))
pyr = dest/'matlabPyrTools_1.4_fixed'
# Octave declares the MATLAB matrix API in mex.h; old sources include both.
(pyr/'MEX/matrix.h').write_text('#include <mex.h>\n')
for name in ['corrDn', 'upConv', 'pointOp']:
    sources = [pyr/'MEX'/(name+'.c')]
    if name in ['corrDn', 'upConv']:
        sources += [pyr/'MEX'/n for n in ['convolve.c', 'wrap.c', 'edges.c']]
    cmd = [str(headers.parents[2]/'bin/mkoctfile'), '--mex', '-std=gnu89',
           '-Wno-error=implicit-int', '-Wno-error=implicit-function-declaration',
           '-I'+str(pyr/'MEX'), '-I'+str(headers),
           '-o', str(pyr/(name+'.mex')), *map(str, sources)]
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)
