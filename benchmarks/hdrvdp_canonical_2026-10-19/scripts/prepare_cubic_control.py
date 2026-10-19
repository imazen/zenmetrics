"""Freeze a compiled Octave tree; change only its spectral interpolation model."""
from pathlib import Path
import shutil
import sys

source,target=map(Path,sys.argv[1:]);assert not target.exists()
shutil.copytree(source,target)
p=target/'load_spectral_resp.m';text=p.read_text()
old="interp1( D(:,1), D(:,k), lambda, 'pchip', 0 )"
assert text.count(old)==1
p.write_text(text.replace(old,'cubic_convolution_control(D(:,1), D(:,k), lambda, 0)'))
shutil.copy2(Path(__file__).with_name('cubic_convolution_control.m'),target)
