# DMM probes: AIC-4 column convention + ladder parity distribution (2026-10-16)

Evidence for `crates/zenmetrics-nn/models/dmm/card.toml` and the nn-port report.
DMM is the Debiased Mapping FR metric (Baoliang93/DMM, IEEE TIP 2025); its head
is a per-patch SVD whose `wt` term depends on the orthonormal basis the solver
*draws* inside degenerate singular clusters — see the report's draw-sensitivity
section. All findings below are instances of that one property.

## AIC-4 `DMM` column: reproduced bit-exact, under the pyiqa convention

`aic_dmm_pyiqa_cuda.tsv` — **all 61 published values reproduced to 0.000e+00**
(median and max |Δ|) when the score is computed as AIC-4's harness did:

- resize: pyiqa `dmm_arch.py`'s `prepare_image_adt` — **square anamorphic**
  `F.interpolate(size=tar_size, mode='bilinear', align_corners=False)` on the
  [0,1] tensor (840×944 → 544×544), NOT upstream `DMM_PyTorch`'s
  aspect-preserving `F.resize` on a PIL image.
- solver: `torch.svd` on **CUDA** (cusolver draws) — upstream's own
  torch-CPU (MKL) run of the same pyiqa-convention pipeline differs by median
  9.05e-3 / max 0.114 (`aic_dmm_pyiqa.tsv`), i.e. the published column itself
  carries draw noise of that order.

`aic_dmm.tsv` — our native port on the same 61 pairs (upstream convention:
aspect-preserving Pillow-fixed-point bilinear + f32 LAPACK `sgesdd` head):
median |Δ| vs published 7.47e-2, max 0.417 — the square-vs-aspect convention
difference, not port error.

## Ladder parity distribution (24 goldens, upstream torch CPU oracle)

`dmm_ladder_{rten,ort}.tsv` — our port's score on every golden pair vs the
stored golden (`zenmetrics batch --metric dmm`, `ZENMETRICS_NN_ENGINE={rten,
ort-cuda}`; the parity test asserts each row ≤1e-3·JND = 1.9142e-4).

- rten: median |Δ| 1.77e-3, max 5.7e-2 (1/24 rows inside gate)
- ort-cuda: median |Δ| 8.1e-4, max 0.176 (2/24 inside gate)

Decomposition on the worst row (`2207_128`, golden 31.5127): MKL
`torch.svd` head on *our own* deterministic-ort taps = 31.498 (trunk noise
−0.014), our netlib-`sgesdd`-faithful f32 head on the same taps = 31.689.
The residual is solver-draw variance in deep degenerate clusters, not trunk
error: on the worst single patch (rank-1, 15-dim null cluster) MKL draws
`wt=3.090`, netlib `sgesdd` (scipy) 3.818, our transliteration 3.818 — ours
matches reference LAPACK bit-closely; MKL's proprietary draw is unreachable.
Even torch CPU vs torch CUDA differ ~9e-3 median on this metric.

Run: `dev`, 2026-10-16; `dmm_aic_probe.py` reproduces the pyiqa-convention
column (needs pyiqa's `dmm_arch.py` + the crop pairs/CSV paths in the script).
