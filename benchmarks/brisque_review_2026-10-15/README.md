# BRISQUE numerical review — 2026-10-15

The corrected port uses multiplication for integer powers and `libm::expf` for
RBF exponentiation. The fixed acceptance gate is 0.001 JND, or 0.015138428 score
units (`JND = 15.138428`). No tolerance or golden score was changed.

The oracle is [piq 0.8.0](https://github.com/photosynthesis-team/piq/tree/v0.8.0)
(Apache-2.0), running on torch 2.14.1+cu130 CPU, AVX2, one thread, glibc 2.43,
Intel oneAPI MKL 2024.2 build 20240605. MKL_CBWR, MKL_ENABLE_INSTRUCTIONS and
ATEN_CPU_CAPABILITY were unset. Runtime metadata is retained with the results.
The source base is nr-classic commit 5ae173ee.

`prepare.py` captures the actual oracle's features and scores on the unchanged
170 goldens plus 200 additional disjoint KonJND image paths. It produces 400,000
synthetic GGD/AGGD ratio inputs, half broad positive values and half enriched
near gamma-grid decision boundaries, seed 20261015. Torch integer pow versus
multiplication has zero ratio-input square/cube bit differences on this set.
`examples/review.rs` compiles the actual private Rust implementation; it does
not duplicate the ratio or scoring formulas. `analyze.py` compares ratios,
argmins, image alpha features and final scores.

| Stage | GGD / AGGD synthetic argmin flips | Golden maximum score / JND | Extra maximum score / JND |
|---|---:|---:|---:|
| Original math baseline | 67 / 8,457 | 0 / 0 | 0.001953125 / 0.00012901768928715718 |
| Multiplication, original exponential | 0 / 0 | 0 / 0 | 0 / 0 |
| Multiplication + libm exponential | 0 / 0 | 0.00634765625 / 0.0004193074901832608 | 0.0078125 / 0.0005160707571486287 |
| Independent pairwise summation + libm exponential (final) | 0 / 0 | 0.0078125 / 0.0005160707571486287 | 0.01025390625 / 0.0006773428687575752 |
| Computed f64 gamma tables + libm exponential, preceding reducer (rejected) | 47,428 / 31,382 | 0.0615234375 / 0.004064057212545451 | 0.005859375 / 0.0003870530678614715 |
| Computed f64 gamma tables + independent reducer (rejected) | 47,428 / 31,382 | 0.0615234375 / 0.004064057212545451 | 0.01171875 / 0.000774106135722943 |

All stages use the same 400,000 synthetic cases and the same 170 + 200 images.
The baseline has 257 GGD and 38,765 AGGD ratio-bit differences. Every corrected
stage has zero ratio-bit differences. Baseline, multiplication and libm have
zero image alpha-feature flips; computed tables have one flip in one golden
image and none in the extra panel. The final independent reducer has zero golden
alpha-feature flips and one flip in one extra image. This image panel does not reproduce the
reviewer's different 12-image maximum. Every image in these measured panels
produced a finite oracle score.

The rejected table experiment uses the existing f32 alpha grid, converts alpha
to f64, calls `libm::lgamma` and `libm::exp`, and rounds once to f32. Its source
is retained as `computed_experiment.rs`. `build_computed.py` patches only the
isolated pinned snapshot, builds a separate retained review executable, then
restores both the source and ordinary compiled artifacts. The latter rebuild
prevents stale experiment artifacts after a later export restores source mtimes.
The tracked crate is never modified by this experiment. This prototype uses
std OnceLock only and is not part of the shipped crate or its no_std path.

Computed R_GGD differs in 5,966/9,801 entries (max 41 ULP), R_AGGD in 5,384
(max 41 ULP), ETA_FACTOR in 4,514 (max 15 ULP). The final-image gate fails, so
all three oracle tables remain embedded. Repeating the experiment with the final
independent reducer also fails the golden gate: one alpha flip in each image
panel; the final computed-table row above records its residuals. `params.bin` contains every original
model/table f32 bit, in explicit little-endian order, with an aligned static
and safe const `from_le_bytes` decoding. Its 232,364 bytes replace 1,164,131
bytes of literal Rust. `tests/gen_params.py` verifies the piq model against
LIVE allmodel, regenerates the binary and small decoder, and records hashes.
The user expressly approved this fixed-model binary.

Commands (serialize heavy work through the shared flock/run-heavy wrapper):

```sh
just brisque-review-prepare "$HOME/tmp/nr-classic-scratch/venv/bin/python"
just brisque-review-build
just brisque-review-run "$LOCK_SNAP_DIR/target/release/examples/review" LABEL
just brisque-review-analyze "$HOME/tmp/nr-classic-scratch/venv/bin/python" LABEL --gate
just brisque-review-ci
just brisque-review-params "$HOME/tmp/nr-classic-scratch/venv/bin/python"
just brisque-review-computed-build
just brisque-review-run "$HOME/tmp/aic-variants-scratch/round6/computed-independent-review" computed-independent
just brisque-review-analyze "$HOME/tmp/nr-classic-scratch/venv/bin/python" computed-independent
```

Large input arrays, complete per-image captures, executables and full logs are
retained outside git at `~/tmp/aic-variants-scratch/round6` on the measured host.
`validation.meta` records hashes and resource receipts; small stage summaries
are committed here. No closed-library implementation or disassembly is used
by the corrected Rust crate. At coordinator direction, the PyTorch-derived
reducer and its notice were subsequently deleted and replaced with independently
written balanced pairwise summation. The final row above passes the unchanged
gate on both image panels. Output revision is now 4.
