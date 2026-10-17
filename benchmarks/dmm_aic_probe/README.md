# DMM probes: AIC-4 published-column comparison + ladder parity distribution (2026-10-16)

Evidence for `crates/zenmetrics-nn/models/dmm/card.toml` and the nn-port report.
DMM is the Debiased Mapping FR metric (Baoliang93/DMM, IEEE TIP 2025); its head
is a per-patch SVD whose `wt` term depends on the orthonormal basis the solver
*draws* inside degenerate singular clusters — see the report's draw-sensitivity
section. All findings below are instances of that one property.

## AIC-4 `DMM` column: not reproduced by the upstream convention

`aic_dmm.tsv` — historical score revision 1 of our native port on the 61-pair crop set (upstream convention:
aspect-preserving Pillow-fixed-point bilinear + f32 LAPACK `sgesdd` head) vs the
published AIC-4 2026 `DMM` column: **median |Δ| 7.47e-2, max 0.417**. The
published evaluation pipeline remains undetermined; the column was not
reproduced by the upstream convention; `dmm_aic_probe.py` joins our batch
output against the published CSV.

## Historical revision-1 ladder distribution (24 upstream torch CPU goldens)

`dmm_ladder_{rten,ort}.tsv` — our port's score on every golden pair vs the
stored golden (`zenmetrics batch --metric dmm`, `ZENMETRICS_NN_ENGINE={rten,
ort-cuda}`). These MKL comparisons are recorded measurements. The current
`nn-parity-dmm` gate uses same-engine SciPy goldens: `kind=jnd` rows stay
at ≤1e-3·JND = 1.9142e-4, and `kind=pinned` rows retain their approved
measured-residual +5% limits.

- rten: median |Δ| 1.67336e-3, max 5.7e-2 (1/24 rows inside gate)
- ort-cuda: median |Δ| 7.33735e-4, max 0.176 (2/24 inside gate)

Decomposition on the worst row (`2207_128`, golden 31.5127): MKL
`torch.svd` head on *our own* deterministic-ort taps = 31.498 (trunk noise
−0.014), the revision-1 f32 head on the same taps = 31.689.
The residual is solver-draw variance in deep degenerate clusters, not trunk
error: on the worst single patch (rank-1, 15-dim null cluster) MKL draws
`wt=3.090`, netlib `sgesdd` (scipy) 3.818, the revision-1 port 3.818. This single-patch agreement does not establish
whole-panel parity or show that an MKL basis cannot be reproduced.

Run: `dev`, 2026-10-16; `dmm_aic_probe.py` re-derives the native-vs-published
comparison above (paths are arguments, see script header).
