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
export.py (`--goldens`): ONNX-vs-chainer worst 3.0e-5 on fr_live's 29
pairs.

`warm_reference_equals_cold` passes on both engines — the FR warm path
reuses cached reference features (`reuses_reference_work() == true`
under rten *and* ort-cuda, since trunk and head are separate graphs) and
is bit-identical to the cold path.

## KonJND JND (1,008 pairs, this port, ort-cuda, 2026-10-16)

FR columns have an image-dependent identical-pair floor (the (0,h,h)
concat is not the head's zero: ~0.2 live, ~13.7 tid), so their JND is
the pooled median `|score(pair) - score(ref,ref)|` per reference —
same convention as ckdn. NR: pooled median `|score(dist) - score(ref)|`
from the paired columns. Script: `models/wadiqam/konjnd.py`.

| column | pooled | JPEG | BPG |
|---|---|---|---|
| wadiqam_fr_live | 29.932910 | 30.459814 | 29.453248 |
| wadiqam_fr_tid | 5.209576 | 6.383762 | 3.720833 |
| wadiqam_nr_live | 23.551063 | 24.827529 | 21.711020 |
| wadiqam_nr_tid | 2.757295 | 3.007036 | 2.525072 |

## Score direction (measured, undocumented upstream)

- `wadiqam_fr_*`: distance-like — higher = more degraded vs reference
  (identical 0.2/13.7 vs distorted ~31-49 on ladder).
- `wadiqam_nr_live`: DMOS-like — higher = worse (ref 7.2 vs dist 19.9).
- `wadiqam_nr_tid`: MOS-like — higher = better (ref 53.1 vs dist 50.0).
- `ScoreKind`: FR `Difference`, nr_live `Distortion`, nr_tid `Quality`.
