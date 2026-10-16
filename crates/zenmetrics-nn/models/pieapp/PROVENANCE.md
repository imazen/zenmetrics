# PieAPP reconstruction sources

This implementation is reconstructed from Prashnani, Cai, Mostofi and Sen,
[CVPR 2018 paper](https://arxiv.org/abs/1806.02067), section 3 / figure 4,
and the official checkpoint's tensor names and shapes. The official Python
implementation is executed as a black-box oracle and its source is not read.
No pyiqa code, package or weights are used.

The paper specifies eleven 3×3 convolution/ReLU stages, 2×2 max-pooling after
every even stage, multiscale feature differences, two 512-hidden-unit heads
for patch error/weight, and weighted patch pooling. Its test procedure uses
1024 random 64×64 patches. The later
[official release README](https://github.com/prashnani/PerceptualImageError/blob/master/README.md)
documents deterministic dense 64×64 patches at stride 6 (default) and sparse
sampling at stride 27. We target the documented dense release convention;
endpoint inclusion, input units, flatten order and head details must be
qualified against black-box outputs before shipping.

Weights: [official UCSB checkpoint](https://web.ece.ucsb.edu/~ekta/projects/PieAPPv0.1/weights/PieAPPv0.1.pth),
modified BSD-2 terms for noncommercial use only. Weights are not bundled;
explicit `WeightsConsent` is required. The code written here is independent
of the noncommercial official source.

## Qualified patch network (2026-10-16)

The official model and CLI archive are pinned to commit
`0513613d1f575136ea52418e35f2cebc5a49f783` (blob hashes checked without
inspecting source). Official checkpoint SHA-256:
`0937b01480c7a637ae3018af755faa8ecde4788b52bb246b7ae62cf96fb6baf0`.

The same `compute_score` entry point used by the documented CLI returns
an aggregate, patch errors, and patch weights. Calling it as an opaque
oracle establishes BGR code-value inputs, NCHW feature flattening, and
the reconstruction's error/weight head. On one 64-square patch and two
128-square patch sets, aggregate differences are below 3e-7 native units.
This is a network control, not the full ladder/JND qualification.
The CLI's 128-square sampling still differs from a regular stride-six
patch set; no sampling equivalence is claimed yet.

The original Python 2.7 / torch 0.3.1 CPU oracle uses device redirection
only, plus the missing demo configuration `num_patches=1024` (paper's test
count). Public method input/output signatures are observed to call it;
no official implementation or feature internals are read. Modern-runtime
compatibility additionally adapts scalar parameter shape, integral reshape
sizes, scalar indexing, and contiguous device-copy layout. Small CLI
outputs agree with the original runtime at printed precision.
