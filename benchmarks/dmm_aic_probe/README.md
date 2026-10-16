# DMM probes: AIC-4 published-column comparison + ladder parity distribution (2026-10-16)

Evidence for `crates/zenmetrics-nn/models/dmm/card.toml` and the nn-port report.
DMM is the Debiased Mapping FR metric (Baoliang93/DMM, IEEE TIP 2025); its head
is a per-patch SVD whose `wt` term depends on the orthonormal basis the solver
*draws* inside degenerate singular clusters — see the report's draw-sensitivity
section. All findings below are instances of that one property.

## AIC-4 `DMM` column: not reproduced by the upstream convention

`aic_dmm.tsv` — our native port on the 61-pair crop set (upstream convention:
aspect-preserving Pillow-fixed-point bilinear + f32 LAPACK `sgesdd` head) vs the
published AIC-4 2026 `DMM` column: **median |Δ| 7.47e-2, max 0.417**. The
published column's eval pipeline is not upstream's, so we record it as not
reproduced by the upstream convention; `dmm_aic_probe.py` joins our batch
output against the published CSV.

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

Run: `dev`, 2026-10-16; `dmm_aic_probe.py` re-derives the native-vs-published
comparison above (paths are arguments, see script header).
