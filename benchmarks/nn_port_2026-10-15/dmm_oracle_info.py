#!/usr/bin/env python3
"""Resolve the numerical oracle's LAPACK constants and live BLAS dispatch."""
import ctypes
from pathlib import Path

import scipy
from scipy.linalg.lapack import ilaver, slamch

print('scipy', scipy.__version__)
print('LAPACK', ilaver())
epsilon = float(slamch('E'))
print("slamch('E')", epsilon, epsilon.hex())
assert epsilon == 2**-24
libraries = list((Path(scipy.__file__).parent.parent / 'scipy.libs').glob('*openblas*'))
assert len(libraries) == 1, f'Expected one SciPy OpenBLAS library, found {len(libraries)}'
library = ctypes.CDLL(str(libraries[0]))
for symbol in ['scipy_openblas_get_corename', 'scipy_openblas_get_config']:
    query = getattr(library, symbol)
    query.restype = ctypes.c_char_p
    print(symbol, query().decode())
