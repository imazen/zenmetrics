"""Prepare a separate 3.0.6 Octave compatibility tree; original archive stays pristine.
Usage: prepare_v306.py RELEASE_TREE NEW_TREE
"""
from pathlib import Path
import shutil
import sys

source,dest=map(Path,sys.argv[1:]);assert not dest.exists()
shutil.copytree(source,dest)
for name in ['fast_conv_fft.m','fast_gauss.m']:
    p=dest/'utils'/name;s=p.read_text();assert s.count(", 'symmetric'")==1
    p.write_text(s.replace(", 'symmetric'",''))
for name in ['reconSpyr.m','reconSpyrLevs.m']:
    p=dest/'matlabPyrTools_1.4_fixed'/name;s=p.read_text()
    assert s.count('is_mex = true;')==1
    p.write_text(s.replace('is_mex = true;', 'is_mex = false;'))

# Octave requires this abstract-method declaration in an @-folder.
cls=dest/'@hdrvdp_multscale';cls.mkdir()
shutil.move(dest/'utils/hdrvdp_multscale.m', cls/'hdrvdp_multscale.m')
