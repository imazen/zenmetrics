# Retained DMM arithmetic identity, 2026-10-17

Reference: 9c69ab54, before this fix round.
Current source: fa3e80ca1fb02584a467529c7237f2094303a6a1

Numerical section: byte-identical.
SHA256: eb2f91a953003e2323a15c2a782cade157f21b1ee7779dd5c213ff41b3650da1

Original SVD assertions: byte-identical.
SHA256: b22114fb808d811051eb7e98a5e96849b72533590220795926c27828f923361f

The numerical section spans `const D_EPS:` through the test-module boundary.
The original SVD assertions start at their np.linalg.svd reference comment.
New resource/cancellation controls precede those unchanged assertions.
Both frozen netlib TSVs are also byte-identical to a7e00774.
dmm_goldens_netlib_rten.tsv SHA256: 51c6d581f2f308e3a356f63bf5971a84c34da289ee2a5b0430c32d761a6394de
dmm_goldens_netlib_ort_cuda.tsv SHA256: dd54079bd0fe82c96b26d03173dcd85519f11a072242beb0045b313725c5632b
