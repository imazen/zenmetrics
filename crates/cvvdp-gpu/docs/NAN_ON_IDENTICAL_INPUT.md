# cvvdp NaN on identical/near-identical input pairs (2026-07-02)

## Report

`s3://zentrain/fill4-6codec-2026-07-01/canonical/fill4metrics_sidecar_2026-07-02.parquet`
(GPU CUDA `cvvdp-gpu` scoring, 4,180,943 rows) had 14,877 rows (0.356%) with
`score_cvvdp = NaN`. A prior forensic pass (concurrent session, 2026-07-02
~14:12–14:14, writeup at
`/mnt/v/datasets/fill4-6codec-2026-07-01/NAN_INVESTIGATION_2026-07-02.md`)
found two clusters:

- **Mode A (5,194 rows, 35%)** — correlated with lossless/near-lossless
  formats (zenpng lossless 5.85% NaN rate vs 0.18–0.50% for lossy codecs;
  34.9% of the NaN rows have `butteraugli == 0`, i.e. pixel-identical pairs;
  the rest have tiny-but-nonzero butteraugli). Verdict: "cvvdp returns NaN
  on zero-difference input pairs instead of the definitional max 10.0."
- **Mode B (9,683 rows, 65%)** — ordinary distorted pairs (median
  butteraugli 3.39), no size/origin/codec/q clustering. Cause undetermined;
  left NaN on purpose in the patched sidecar (masked per-row at train time,
  never silently defaulted to a value).

This doc covers the mode-A investigation and fix done in this crate
(`crates/cvvdp-gpu` + `crates/cvvdp`) on 2026-07-02. Mode B is NOT
addressed here — no evidence tied it to this crate specifically (see
"What wasn't found" below).

## What was verified before fixing anything

Every existing "identical input" test in both `cvvdp` and `cvvdp-gpu`
(`upstream_parity_extended.rs`, `diffmap_invariants.rs`,
`cvvdp_score_flat_vs_flat_yields_max_jod`) used either a byte ramp or a
PRNG-noise grid — **never a solid/flat color**, and never combined with a
byte-for-byte-identical distorted buffer at the exact sizes the flagged
rows show (e.g. `64x48`, `96x128`, `147x192`). That's a real, confirmed
coverage gap, closed by the two new regression tests added alongside this
doc (`cvvdp_score_identical_solid_colors_yield_exact_max_jod` in
`tests/it/pipeline_score.rs`, `identical_solid_colors_yield_exact_max_jod_and_zero_diffmap`
in the `cvvdp` crate's `tests/it/diffmap_invariants.rs`).

## What was NOT reproduced (extensive live testing, all clean)

Before landing the fix below, the following were run against current
`master` (no relevant commits to `crates/cvvdp{,-gpu}/src` since
2026-06-25, so this is the same code the fleet ran) with **zero** non-finite
results:

1. **CPU (`cvvdp`) `Cvvdp::score`/`score_with_diffmap`**: solid black/white/
   mid-gray/`1`, a byte ramp, a real 256×256 photo (`zenmetrics_corpus::source_png()`),
   all scored identical-vs-self. All exactly `10.0`, all-zero diffmap.
2. **GPU (`cvvdp-gpu`), cold `Cvvdp::new_with_geometry` + `.score()`**, matching
   `zenmetrics-cli`'s `CvvdpBatchScorer` construction exactly
   (`DisplayModel::STANDARD_4K` + `DisplayGeometry::STANDARD_4K`, not
   `CvvdpParams::PLACEHOLDER`'s bare display): same corpus as (1) plus the
   exact odd dimensions from the flagged rows (`64x48`, `96x128`, `147x192`,
   `768x576`) and flat-gray content at those sizes. All exactly `10.0`.
3. **GPU, reused instance ("hammer" test)**: one `Cvvdp` instance scoring
   300+ pairs in sequence (identical + extreme + ordinary, interleaved) —
   probing for state leakage across `CvvdpBatchScorer`'s per-`(w,h)` cache
   reuse. Zero non-finite results.
4. **GPU, warm-reference path** (`warm_reference` + `score_with_warm_ref_diffmap`,
   matching `run_score_file`'s `Orchestrator::run_all` warm-ref batch
   scoring per `docs/SCOREMANY_OPT.md` Part 2 — the actual production
   "score-many" path, distinct from `CvvdpBatchScorer`): 4 references ×
   200 variants each (identical / 1-LSB-perturbed / different, interleaved),
   800 total warm-ref calls. Zero non-finite results.
5. **Real production binary** (`zenmetrics score-pairs --metric cvvdp-gpu`,
   the exact code this doc's fix touches): a single identical pair, then
   3,000 REAL images (mixed photos/screenshots/wiki captures from
   `codec-corpus`, size-filtered to ≤900×900 to avoid an unrelated CUDA OOM)
   swept through `zenpng` (lossless) round-trip and scored — **0/3000
   failures**. (A first attempt at this without the size filter hit real
   `CUDA_ERROR_OUT_OF_MEMORY` panics from un-sorted wildly-varying image
   sizes thrashing the GPU allocator — that's a separate, already-understood
   issue, not this bug; it produced spurious "NaN-failures" in
   `score-pairs`' own accounting that are NOT the mode-A bug.)

Code review of every division/pow/log in the masking (`kernels/masking.rs`),
pooling (`kernels/pool.rs`), and Weber-contrast pyramid (`kernels/pyramid.rs`,
both the CPU `pyramid.rs` and GPU kernel versions) found the L_bkg
denominators already `max(0.01)`-clamped (both the per-pixel non-baseband
path and the baseband mean-of-clamped-terms path), and the masking `safe_pow`
denominators bounded away from zero (`1 + M_pool` with `M_pool >= 0`). GPU's
`pool_band_finalize` sanitizes a NaN/negative atomic partial via `.max(0.0)`
(Rust's NaN-avoiding `f32::max`); CPU's `lp_norm_mean`/`lp_norm_sum` do NOT
have an equivalent guard, but every live test that could have exercised that
gap came back clean, so no change was made there (see "Not fixed" below —
speculative sanitization without a reproduced trigger risks masking a REAL
mode-B-style bug the same way the investigation deliberately avoided masking
mode B).

## The fix

`Cvvdp::score` / `Cvvdp::score_with_diffmap` (both crates) and
`Cvvdp::compute_dkl_jod` (GPU, which `score`/`score_with_reference` both
funnel through) now short-circuit to `10.0` (zero diffmap) when
`reference_srgb == distorted_srgb` byte-for-byte, **before** any pipeline
work. This is not a numeric workaround — it's the metric's own definition
applied without floating-point risk: byte-identical inputs have zero
difference by construction, so no chain of `pow`/`log`/`div`, however deep,
gets a chance to turn "no difference" into a non-finite score. It's also a
free performance win for real sweep traffic (q=100 / lossless cells are
common).

This provably kills the 34.9%-of-mode-A exactly-`butteraugli==0` subset. It
does NOT, by itself, explain or fix the remaining ~65% of mode-A rows
(near-lossless, tiny nonzero residual) — those were part of the "no
non-finite result" live testing above (case 4's 1-LSB-perturbed variants,
case 5's real near-lossless-adjacent PNG round-trips), so no live trigger
was found for that subset either. See "Open questions" below.

## Not fixed / follow-ups

1. **Near-identical (not bit-exact) mode-A rows.** Could not reproduce.
   Leading alternative hypothesis, NOT verified: the fill4-6codec backfill's
   own dedup step ("dedup via per-`(encode_sha,metric)` MEDIAN", per
   `~/work/zen/DATA_PROVENANCE.md`) fans a single `encode_sha` out to many
   `(source, q)` cells whenever lossless/near-lossless encodes collapse to
   identical bytes — exactly the mode-A-correlated formats. If a naive
   (non-NaN-aware) median were used, ONE transient failure (worker
   crash/OOM/preemption, `re-claims` are explicitly mentioned) anywhere in a
   popular `encode_sha`'s many duplicate-claim scoring attempts would poison
   the whole group's median to NaN — independent of any cvvdp math bug, and
   independent of whether the OTHER duplicate attempts scored the identical
   content correctly. This is testable against the RAW (pre-dedup) fleet
   output (not available on this workstation) — worth a follow-up with that
   data.
2. **Mode B (9,683 rows, cause undetermined).** Out of scope here; no
   evidence found tying it to `cvvdp`/`cvvdp-gpu` specifically. Left NaN in
   the patched sidecar on purpose.
3. **Orchestrator OOM-fallback strip/capped-pyramid path.** All live tests
   above used the "Full" pipeline via direct `Cvvdp` construction. The
   `zenmetrics-orchestrator`'s advertised "OOM-safe fallback (GPU full →
   strip → CPU)" ladder was not independently exercised — if a production
   run fell back to strip mode for memory pressure, that's a materially
   different kernel path (`subtract_weber_3ch_strip_kernel`,
   `pool_band_3ch_offset_kernel`, etc.) this investigation did not
   specifically stress-test with identical/near-identical content, though
   spot-reading found the same `max(0.01)` / abs-before-pow guards there
   too.
4. **HDR EOTF branches (PQ, HLG).** `apply_eotf_branch`'s PQ branch
   (`kernels/color.rs`) takes `f32::powf(v, 1/m2)` on the raw (not
   `[0,1]`-clamped) input — a genuinely negative `v` (plausible for
   HDR linear-planes input with matrix-rounding noise) gives NaN
   (fractional power of a negative base). Not applicable to the SDR-only
   corpus behind this investigation, so not changed, but worth a defensive
   clamp if the HDR linear-planes path is audited next.
