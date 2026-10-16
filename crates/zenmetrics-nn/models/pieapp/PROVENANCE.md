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
