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
sampling at stride 27. Those modes belong to the demo CLI over sub-images.
We target an explicit stride-six public-entry-point convention;
the canonical valid-origin grid, BGR code values, NCHW flattening and head
are qualified against public black-box outputs.

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
patch set; wrapper sampling equivalence is not claimed.

The original Python 2.7 / torch 0.3.1 CPU oracle uses device redirection
only, plus demo configuration `num_patches=1024` (the paper's test
count). Oracle exception tracebacks printed two lines of the official demo
script; no implementation files were opened or read. Public method input/output signatures are observed to call it;
no official implementation or feature internals are read. Modern-runtime
compatibility additionally adapts scalar parameter shape, integral reshape
sizes, scalar indexing, and contiguous device-copy layout. Small CLI
outputs agree with the original runtime at printed precision.

## Head-convention evidence

The paper does not specify the division by 100 or the `abs(weight) + 1e-6`
transform. Those expressions were initially hypotheses in `probe.py`, rather
than conclusions drawn from the paper. The tensor metadata supplies the
learned scalar affine parameters, but does not establish either expression.

`review_head_probe.py` tests these hypotheses against the official public
`compute_score` patch-error and patch-weight outputs. Only the independent
reconstruction exposes raw head values to this probe. The official model's
intermediate features and implementation remain opaque. Candidate error
divisors are 1, 10, 100 and 255; weight candidates distinguish signed versus
absolute values and the added epsilon. Identity, black/white, photographic
and deterministic random patches provide controls. The committed review
receipt reports the measured discrimination and any unresolved ambiguity.

The [22-case public-output receipt](../../../../benchmarks/pieapp_review_2026-10-16/head-positive.json)
records exact equality for division by 100. The maximum absolute patch-error
differences for divisors 1, 10 and 255 are 461.6010956764221,
41.963736057281494 and 2.834151864051819 native units, respectively.
Adding `1e-6` likewise matches every public weight exactly; omitting it gives
maximum error 1.9073486328125e-6. For the identity patch, raw weight
7.4552412033081055 becomes the public 7.455242156982422 after the addition.
All these raw weights are positive, so this panel alone cannot distinguish
signed weights from absolute weights. A separate sign probe addresses that
ambiguity using only the independent network and opaque public outputs.

## Selected dense sampling contract

Canonical parity uses regular stride-six sampling against `compute_score`.
The demo wrapper has a separately recorded result. Valid origins are
`range(0, side - 63, 6)`, without an extra far-edge origin. On `1415_128`,
the CLI scores `1.6797861082006782`, while the 121-patch entry-point call
scores `1.6567797660827637` (absolute difference `0.0230063421179145`).
The documented sparse CLI option gives `1.7122649515845207`; no documented
option has explained the dense difference. Further wrapper probes are
restricted to documented options. See [DIVERGENCES](../../../../DIVERGENCES.md#pieapp-sampling-and-official-entry-point-2026-10-16).


The [full 24-pair demo panel](../../../../benchmarks/pieapp_review_2026-10-16/cli.json)
runs the documented dense CUDA CLI twice per pair; all printed scores repeat
exactly. The worst difference is `2207_2048`: demo `3.4582222` versus explicit
entry-point grid `2.4828926609204816`, or `0.9753295390795182` native units
(`1.879433106938455` operational JND). This modern compatible CUDA demo panel
retains values at their printed precision; the earlier original-runtime
128-square control above remains a separate observation. Neither panel
establishes equivalence between the demo's sub-image wrapper and the selected
full-image stride-six entry-point grid.

`oracle.py` calls the official method, consuming only its public patch
errors/weights and pooling those outputs in f64. `oracle_runtime_cpu.py`
and `oracle_runtime_cuda.py` preserve the independently written compatibility
adapters used as `sitecustomize.py` in the isolated oracle environments.
They adapt old Python/Torch runtime conventions; they contain no official
model source. CUDA disables TF32 and uses deterministic cuDNN. Oracle GPU
batches are repeated; any mismatch requires a third matching result.


## Native qualification

Both engines pass all 24 supported ladder pairs at 1e-3 operational JND.
RTen's reported worst error is 1.84e-6 JND; ort-cuda's is 2.69e-7 JND.
One operational JND is 0.5189487912492419 native units, the pooled median
across all 1008 KonJND threshold pairs. Both engines pass warm/cold bit-identity
checks. The [qualification receipt](../../../../benchmarks/pieapp_2026-10-16_qualification.json)
and [full record](../../../../benchmarks/pieapp_2026-10-16.md) retain hashes,
input bounds and the separately verified wrapper bookkeeping.
