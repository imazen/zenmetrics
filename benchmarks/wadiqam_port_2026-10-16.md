# WaDIQaM port — 2026-10-16

Branch `wadiqam-port`, base `nn-lpips-vgg@origin` 51d804f3 (post-WD merge).
Four columns: `wadiqam_{fr,nr}_{live,tid}` from upstream
`dmaniry/deepIQA` @51dece99 (`models/{fr,nr}_{live,tid}_weighted.model`,
MIT code and MIT weights — sha256s pinned in
`crates/zenmetrics-nn/models/wadiqam/export.py`).

## Protocol

Upstream `evaluate.py` verbatim: non-overlapping 32x32 stride-32 patches
at native resolution (edge partials dropped), raw u8->f32, no
normalisation, `top="weighted"` (the documented default; the paper's
randomly-sampled overlapping patch scheme is the training protocol).
Image score = `sum(y*a)/sum(a)` pooled host-side.

FR ships as two ONNX graphs — trunk `x[P,3,32,32] -> h[P,512]` and head
`f[P,1536] -> ya[2,P]` — so the reference's patch features cache under
`warm_reference` on both engines (host owns the `(h-h_ref, h, h_ref)`
concat). NR ships whole: `x -> ya[2,P]`.

## Parity vs the upstream chainer oracle (CPU fp32)

Oracle: `models/wadiqam/oracle.py` — upstream `fr_model.FRModel` /
`nr_model.Model` classes verbatim in a py3.10 chainer venv
(`~/tmp/wadiqam-oracle/`), `cuda.cupy` shimmed to numpy *after* weight
loading. Gate: 1e-3 of the KonJND-measured JND. 29 ladder pairs
including non-square and 4096x4096:

| column | engine | worst \|d\| | in JND |
|---|---|---|---|
| wadiqam_fr_live | rten | 2.289e-5 | 7.65e-7 |
| wadiqam_fr_live | ort-cuda | 3.147e-5 | 1.05e-6 |
| wadiqam_fr_tid | rten | 4.079e-5 | 7.83e-6 |
| wadiqam_fr_tid | ort-cuda | 1.908e-5 | 3.66e-6 |
| wadiqam_nr_live | rten | 6.198e-6 | 2.63e-7 |
| wadiqam_nr_live | ort-cuda | 2.427e-5 | 1.03e-6 |
| wadiqam_nr_tid | rten | 1.526e-5 | 5.54e-6 |
| wadiqam_nr_tid | ort-cuda | 1.526e-5 | 5.54e-6 |

Export self-check (`export.py`, torch mirror vs ONNX, 64 random patch
inputs): ya elementwise |d| <= 7.4e-5 (f32 reduction-order noise at
y~100 magnitudes), pooled score |d| <= 7.6e-6. Goldens check inside
export.py (`--goldens`): ONNX-vs-chainer worst 4.2e-5 over all four
columns' 29-pair ladders (fr_live 3.0e-5).

`warm_reference_equals_cold` passes on both engines — the FR warm path
reuses cached reference features (`reuses_reference_work() == true`
under rten *and* ort-cuda, since trunk and head are separate graphs) and
is bit-identical to the cold path.

## KonJND JND (1,008 pairs, this port, ort-cuda, 2026-10-16)

FR columns have an image-dependent identical-pair floor (the (0,h,h)
concat is not the head's zero: KonJND identical-pair medians 0.019
live / 12.5 tid, ranges -0.29..0.70 / 9.7..20.6, n=1008), so their JND
is the pooled median `|score(pair) - score(ref,ref)|` per reference —
same convention as ckdn. NR: pooled median `|score(dist) - score(ref)|`
from the paired columns. Script: `models/wadiqam/konjnd.py`.

| column | pooled | JPEG | BPG |
|---|---|---|---|
| wadiqam_fr_live | 29.932910 | 30.459814 | 29.453248 |
| wadiqam_fr_tid | 5.209576 | 6.383762 | 3.720833 |
| wadiqam_nr_live | 23.551063 | 24.827529 | 21.711020 |
| wadiqam_nr_tid | 2.757295 | 3.007036 | 2.525072 |

`wadiqam_nr_tid`'s 2.76 is a **noise floor, not a just-noticeable
step**: across the same pairs the distorted image outscores its
reference in 54.7% of cases, so the "median |delta|" is dominated by
sign-random disagreement rather than a consistent perceptual step. It
stays in `jnd_scale.tsv` as the parity tolerance.

## Score direction (measured, undocumented upstream)

Direction check: `models/wadiqam/direction.py` scores a monotone
distortion ladder — 12 KonJND references x JPEG q95, q80, q60, q40,
q25, q15, q8, q3 — and reports the Spearman rho of score vs
degradation per source (receipt:
`benchmarks/wadiqam_direction_2026-10-16.txt`; an independent run of
the same ladder with a different encoder gave fr_live +1.00,
fr_tid +1.00, nr_live +0.98 on 11/12, nr_tid +0.79 on 12/12).
Measured medians:

- `wadiqam_fr_live`: rho +1.00, rho>0 on 12/12 sources — distance-like.
- `wadiqam_fr_tid`: rho +1.00, rho>0 on 12/12 sources — distance-like.
- `wadiqam_nr_live`: rho +0.99, rho>0 on 11/12 — DMOS-like.
- `wadiqam_nr_tid`: rho +0.88, rho>0 on 12/12 — **rises with
  distortion** despite TID2013 being MOS-labelled. The MOS labels do
  not transfer to this head.
- `ScoreKind`: FR `Difference`; nr_live and nr_tid both `Distortion`.

## Memory

Upstream `evaluate.py` batches inference at 2,000 patches per graph
call; the port does the same (`PATCH_CHUNK`), so conv intermediates
stay bounded whatever the input size, and images that would yield
more than 32,768 patches (~5824^2) are refused. Measured peak RSS of
the whole `score` process (`/usr/bin/time -v`, 2026-10-16):

| input | patches | column | rten | ort-cuda |
|---|---|---|---|---|
| 4096x4096 | 16,384 | wadiqam_fr_live | 943 MiB | 1.42 GiB |
| 4096x4096 | 16,384 | wadiqam_nr_live | 875 MiB | 1.42 GiB |
| 4096x8192 | 32,768 (cap) | wadiqam_fr_live | 1.26 GiB | 1.71 GiB |
| 4096x8192 | 32,768 (cap) | wadiqam_nr_live | 1.14 GiB | 1.71 GiB |

ort-cuda's ~1.4 GiB floor is ORT + CUDA-provider load overhead, not
patch-count growth (the cap adds only ~300 MB over 4096^2).

## ort-cuda environment caveat: cuDNN 9.24 (measured 2026-10-16)

The parity numbers above were produced with cuDNN **9.19.0**
(`~/tmp/nnrt-research/venv-cu`). With cuDNN **9.24.0**
(`~/tmp/nr-learned-scratch/venv`) the same build fails on batch
>= ~1,500 patches: the cudnn-frontend plan for the 512->512 3x3 convs
at 2x2 spatial (`conv9`/`conv10`) reports a ~9-10 GiB workspace
(`Failed to allocate memory for requested buffer of size
10053746688`, BFCArena) that cudaMalloc cannot serve on the 16 GiB
RTX 4080 — independent of `cudnn_conv_algo_search` and of
`cudnn_conv_use_max_workspace` (a dead option on ORT 1.31's cudnn-FE
path). Reproduced in plain python `onnxruntime-gpu`; not a zenmetrics
bug. Keep the parity/CLI environment on cuDNN 9.19 (or another
version verified not to pick a giant-workspace engine) until this is
resolved upstream.
