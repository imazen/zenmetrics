"""Clone the retained compiled Octave tree; replace only its two FFT adapters."""
from pathlib import Path
import shutil
import sys

source, target = map(Path, sys.argv[1:])
assert not target.exists(), 'retain existing source trees'
shutil.copytree(source, target)
old = 'real(ifft2( fX.*fH, size(fX,1), size(fX,2) ))'
for name in ['fast_conv_fft.m', 'fast_gauss.m']:
    path = target / name
    text = path.read_text()
    assert text.count(old) == 1
    path.write_text(text.replace(old, 'symmetric_ifft_control(fX.*fH)'))
shutil.copy2(Path(__file__).with_name('symmetric_ifft_control.m'), target)
