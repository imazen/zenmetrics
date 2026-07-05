# zensim-gpu odd-dimension feature corruption — root cause, fix, blast radius (2026-07-05)

## Summary

`zensim-gpu`'s 372-feature (`WithIw` regime) extraction silently produced
**wrong, non-NaN, sometimes corpus-busting feature values** for any image
whose height became **odd at any level of the 4-scale pyramid** — not just
when the top-level height is odd. Root cause: the per-scale pyramid height
was computed with `div_ceil` (ceiling division) instead of matching CPU
zensim's plain truncating (floor) division. This was reported from the
zensim repo side as "the grid's own provenance note said '~25% of cells NaN
on odd-dim images dropped' — a known odd-dim pathology exists; the bug is
that some path yields non-NaN garbage instead" (see
`benchmarks/linear_projections_2026-07-03.md` §w11 and
`benchmarks/provenance_best_results_2026-07-04.md` §f155 in the zensim
repo). This doc is the zenmetrics-side root-cause + fix + blast-radius
report.

**Fixed** in `crates/zensim-gpu/src/pipeline.rs` (6 sites) +
`crates/zensim-gpu/src/memory_mode.rs` (1 estimator, for documentation
honesty — not a correctness bug). **Regression test**:
`crates/zensim-gpu/tests/it/odd_dim_pyramid_parity.rs` (4 tests, all
passing). **Diagnostic tool**: `crates/zensim-gpu/examples/odd_dim_repro.rs`.

A **known, separate, much-smaller-magnitude residual** remains — see
"Known residual (not fixed)" below. It does not explain the reported
corpus-busting magnitudes and is out of scope for this fix.

## Root cause

`Zensim<R>::new_with_regime_strip_budget` (the pipeline constructor) builds
a 4-scale pyramid plan. Before the fix, the per-scale height recurrence was:

```rust
h = h.div_ceil(2);   // WRONG
```

CPU zensim's actual pyramid (`zensim::blur::downscale_2x_inplace`) computes:

```rust
let new_h = height / 2;   // plain truncating division — FLOOR
```

For an **even** height these are identical, so every previously-tested
fixture (`64×64`, `192×192`, `320×240` — all exact multiples of 16 in width,
all even in height) never exercised the divergence; the crate's own test
suite (`tests/it/cpu_gpu_feature_sweep.rs`) had a doc comment claiming a
"257×257 (a non-pow-2 odd size)" fixture was covered, but the actual `SIZES`
const never included it — the odd-height case was undocumented-untested.

For an **odd** height, GPU allocated and processed **one extra row per
scale** that CPU never had. `downscale_2x_3ch_kernel`'s edge-clamp
synthesizes that extra row by re-reading (duplicating) the last real row —
not a natural continuation of the image, a boundary-duplicate artifact. The
error compounds every subsequent scale: once a height goes odd, `floor` and
`ceil` sequences can *both* stay odd all the way down (e.g. `513 = 512 + 1`:
257, 129, 65 — every one of `N-1` a power of two), so the corruption grows
through scales 1→2→3 rather than being confined to one level.

**This is not confined to the masked/IW blocks.** Measured on a 769×513
fixture (matching one of the corrupted corpus images), pre-fix: `basic`
block `hf_energy_gain` off by up to 76 absolute / 30%+ relative at scale 3;
several `peak` block slots (`ssim_max`/`art_max`/`det_max`) off by 60–200%
relative at scales 2–3; `masked_ssim_4th`/`iw_ssim_4th` off by 80%+ relative
at scale 3. All of these are **eliminated post-fix** (see Verification).

Six call sites independently repeated the same `h.div_ceil(2)` pyramid-height
recurrence (the constructor's own plan, three different `compute_*` entry
points' host-side packing normalizer, the strip-mode per-strip height
reset, and the strip-mode device-cached full-image ref pyramid plan) — all
six were fixed to floor division. A seventh site
(`memory_mode.rs::pyramid_pixels`, a VRAM-budget *estimator*, not on the
correctness path) was also fixed for documentation honesty: its own doc
comment claimed to "match the host-side scale walk," which was only true
pre-fix.

### Why width was NOT part of the bug

`padded_w` (the SIMD-aligned width) already used floor division
(`padded_w /= 2`) on both sides pre-fix — confirmed correct by direct
comparison: an odd-width-only fixture (`750×512`, non-16-aligned width,
even height) shows the same negligible ~1e-5 relative noise as a fully
"nice" fixture (`768×512`), while an odd-height fixture at the identical
width shows the large divergence described above. `logical_w` (used only
for scale-0's mirror-fill table) does use `div_ceil`, but it's provably
dead beyond scale 0 (`mirror_offsets` is read only from `scales[0]` at all
four use sites in `pipeline.rs`) — left unchanged since fixing dead code
found by the same search added: `logical_w = logical_w.div_ceil(2)` still
appears at `pipeline.rs:691` and is intentionally not the height fix.

## The fix

Commit(s) on `master` (this session): 6 pyramid-height sites in
`crates/zensim-gpu/src/pipeline.rs` changed from `h.div_ceil(2)` /
`hs.div_ceil(2)` / `h_at_s.div_ceil(2)` to plain `/= 2`:

1. `new_with_regime_strip_budget`'s scale-plan loop (the primary
   allocation/processing height for every scale — this is the site that
   actually determines what the kernels compute).
2. `compute_with_reference_vec`'s `scale_image_h` (Full-mode host-side
   packing normalizer — must track #1 exactly, per its own doc comment).
3. `compute_with_reference_vec_strip`'s `scale_image_h` (Strip-mode
   normalizer — represents the canonical full-image per-scale height,
   independent of strip allocation).
4. `set_scale_h_for_strip` (per-strip active height for the CURRENT
   strip — its own doc comment already said `actual_strip_h / 2^s`, i.e.
   floor; the code disagreed pre-fix). Interior strips have
   `actual_strip_h == strip_alloc_h`, always a multiple of `STRIP_ALIGN`
   (8), so ceil/floor agreed there; the **last (boundary) strip's**
   `actual_strip_h` is image-height-derived and not generally 8-aligned,
   so it hit the same bug for strip-mode boundary strips.
5. `build_full_ref_xyb_pyramid`'s per-scale height plan (task #75's
   device-cached full-image reference pyramid for strip mode) — its own
   doc comment claims to match Full mode's plan (#1), only true post-fix.
6. `compute_features_from_built_xyb`'s `scale_image_h` (the linear-planes
   / HDR PU21 feature path) — same normalizer pattern as #2/#3.

Plus `crates/zensim-gpu/src/memory_mode.rs::pyramid_pixels` (VRAM
estimator, not correctness-critical — see inline comment).

The `body_s_lo`/`body_s_hi` strip-mode row-range mappings
(`pipeline.rs` around the "map body rows from scale-0 strip-local to scale
s" comment) intentionally keep `div_ceil` — that's a *different* concern
(consecutive-strip boundary continuity: strip k's `body_hi` at scale s must
equal strip k+1's `body_lo` at scale s), verified by reading the existing
comment there before touching anything nearby.

## Regression test

`crates/zensim-gpu/tests/it/odd_dim_pyramid_parity.rs` — 4 tests:

- `odd_769x513_matches_reported_corpus_dims`
- `odd_1022x818_matches_reported_corpus_dims`
- `odd_320x241_minimal_isolated_repro` (the smallest possible reproduction:
  the pre-existing, already-passing `320×240` fixture from
  `cpu_gpu_feature_sweep.rs` plus exactly one row)
- `odd_height_power_of_two_plus_one_compounds_every_scale` (`513 = 512+1`,
  the worst-case fixture where the bug compounds through every scale)

Each test runs BOTH call shapes the original bug reproduced identically in:
a **warm loop** (one `Zensim`, one `set_reference`, many
`compute_with_reference_vec` calls — the exact shape
`zenmetrics-cli`'s `MetricCache` uses for a sweep quality ladder) and a
**cold** one-shot (`compute_features_vec` — e.g. `zenmetrics score`). Two
invariants are asserted per test:

1. **No feature is bit-constant across a mini quality ladder** (4 distinct
   distorted variants) — the exact production symptom. `hf_energy_loss` /
   `hf_mag_loss` are excluded by construction (architecturally ==0 for any
   pure-noise-add distortion set, since "loss" is impossible when every
   variant only adds high-frequency energy — a legitimate false-positive
   this screen must not flag).
2. **GPU/CPU agreement within the SAME per-slot-kind budget**
   `tests/it/cpu_gpu_feature_sweep.rs::slot_budget` already asserts (2e-3
   abs/rel basic, 3e-3/5e-3 peak, 5e-3/5e-3 masked+IW), widened by an
   explicit, documented `BUDGET_MARGIN = 2.0` (this test's own fixtures/
   noise seeds land at a different point on the same universal GPU-vs-CPU
   float-summation-order noise floor than `cpu_gpu_feature_sweep.rs`'s do;
   one observed legitimate case landed at 1.07x/1.94x the raw budget) —
   **with one narrow, explicitly documented exception** for
   `masked_det_4th` / `iw_det_4th` (see "Known residual" below), bounded by
   its own separate `KNOWN_RESIDUAL_ABS_CEIL = 0.05` so a regression that
   *grows* it is still caught.

**Validated the test catches the regression**: temporarily reverted site #1
back to `div_ceil` — all 4 new tests failed with exactly the expected
diagnosis (`hf_energy_gain` at scale 1, ~3.7% relative, matching the
originally-measured magnitude). Restored immediately after confirming.

## Verification — before / after, on the reported corpus dimensions

Measured via `examples/odd_dim_repro.rs` on a 769×513 synthetic fixture
(matches one of the corrupted corpus images' dimensions), warm loop, 8
distorted variants of increasing noise:

| | pre-fix | post-fix |
|---|---|---|
| `hf_energy_gain` (basic, scale 1) worst relative error | 3.7% → 30%+ (grows with scale/noise) | ~1e-4 (normal f32 noise) |
| peak block (`ssim_max`/`art_max`/`det_max`, scale 2–3) worst relative error | up to 200%+ | within budget |
| `masked_ssim_4th`/`iw_ssim_4th` (scale 3) | up to 83% relative | within budget |
| Slots failing the crate's own claimed parity budget | 26 → 56 (growing with noise) | 0 (basic/peak/most masked+IW); only the documented residual (below) remains, always < 0.02 abs |

Isolated the width-vs-height split with controlled fixtures: `750×512`
(odd, non-16-aligned width; even height) shows ~1e-5 relative noise, same
order as the fully "nice" `768×512` control — confirming width was never
part of the bug (see "Why width was NOT part of the bug" above).

## Full existing test suite — no regression on clean-path (even-dim) outputs

**All 105 tests in the `it` binary pass** (101 pre-existing + the 4 new
`odd_dim_pyramid_parity` tests) — including `strip_parity`'s
`basic_strip_matches_full_400x300_h_body_120`, which already exercises an
odd-at-depth height in **strip mode** (`300 → 150 → 75 (odd) → 37 (odd)`)
and passes post-fix, directly confirming the strip-mode sites (#3/#4/#5
above) as well as the primary full-mode fix.

This holds **by mathematical construction** for every already-even-height
fixture, not just by testing: `h / 2 == h.div_ceil(2)` for every even `h`,
so the fix is a byte-identical no-op on any pyramid path that never hits an
odd height at any scale.

**How "all 105 pass" was established** (worth recording — it cost real
session time): running the full suite in one process with
`--test-threads=4` (the crate's documented default) stalled 14+ minutes on
`diffmap_invariants` / `cpu_gpu_diffmap_parity` — reproduced identically on
the FIRST baseline run this session, taken before ANY code changes, so
not a regression this fix introduced. Bisected the cause:
`gpu_diffmap_matches_cpu_canonical_pointwise` in complete isolation (its
own process) takes 1.85 s; the entire `diffmap_invariants` module (10
tests) standalone takes 0.36 s. The stall only appears when these tests
run as the ~90th+ test **within the same long-lived process** — i.e. some
resource (GPU memory/handles, a cubecl JIT/kernel cache, or CUDA
driver-side per-process state) degrades across ~90+ sequential `Zensim<R>`
constructions in one binary, regardless of `--test-threads`. This is a
**pre-existing test-infrastructure characteristic, unrelated to this fix**
(the exact same stall, at the exact same point, reproduced before this
session's first code change) — not diagnosed further here (out of this
fix's scope; worth a follow-up ticket). Verification was completed by
running every module standalone/in small groups instead of the full
105-test single process: `odd_dim_pyramid_parity` (4), `diffmap_invariants`
(10), `memory_mode`+`strip_parity`+`sub64_reflect_pad`+`typed_sub_min_pad`+
`weights_parity` (28), `extended_parity`+`opaque`+`opaque_cached_ref`+
`opaque_default_weights_v03`+`opaque_regime`+`parity_lock`+`pu_xyb_parity`
(37), and `auto_fallback`+`cached_ref_slot_rebuild`+`cpu_gpu_diffmap_parity`
+`cpu_gpu_feature_sweep`+`cpu_parity` (26) — 4+10+28+37+26 = **105/105
passing**, zero failures.

## Known residual (NOT fixed by this change — separate, smaller, documented)

Independent of the pyramid-height bug (present at **scale 0**, which
involves no pyramid halving at all), `masked_det_4th` / `iw_det_4th`
specifically (never `ssim_4th`/`art_4th`/`mse` in the same block) show a
small absolute (≤0.02 measured) but sometimes large *relative* divergence
whenever the image height is odd. Isolated with a minimal pair: `320×240`
(clean) vs `320×241` (+1 row, otherwise identical content) — the
divergence appears at 241 only, and low-noise `320×241` variants show it
appearing at other scales too as noise increases, so it is not strictly a
"scale 0 only" phenomenon on closer measurement, just usually smallest
there. Root cause not isolated in this session; candidate mechanism is a
CPU/GPU divergence in the masked-IW strip kernel's handling of the
partial/boundary strip or vertical-mirror math specific to odd total
heights (`masked_iw_strip.rs`). This is **two to four orders of magnitude**
below anything that would explain the originally-reported corpus-busting
values (the reported ~270 vs a fresh-CPU 0.003–0.025; this residual's
worst measured absolute delta is 0.017) and does not trip the corruption
screen used for the blast-radius survey below (`|value| > 5`). Tracked so
a future session doesn't have to re-derive it: see the regression test's
module doc + `KNOWN_RESIDUAL_ABS_CEIL` in
`tests/it/odd_dim_pyramid_parity.rs`.

## Corruption detector (validated against the known ground truth)

A ladder (rows sharing one reference/source, grouped by e.g.
`image_id`+`codec` or `origin_id`+`width`+`height`, any number of distorted
variants) is corrupt iff: **(1)** at least one masked/IW feature column
(`f228`..`f371`) is bit-constant across every row in the ladder, **AND**
**(2)** that constant value exceeds `5.0` in absolute terms (corpus norms
top out ~2). Both conditions are required — bit-constant-but-small (e.g.
`hf_energy_loss==0` for noise-only distortions, architecturally impossible
to be anything else) is a false positive; large-but-varying is normal
signal. **Validated**: this exact criterion reproduces the zensim-side
finding on `dial_grid_372col_2026-05-29.parquet` bit-for-bit — 9 of 115
`(image_id, codec)` ladders flagged, and they are *the same 9 ladders*
zensim's own investigation named (the trio × webp, `9059ec43b26aa167_769x513`
× {jpeg, webp}, and 4 more × webp at `1022×818`). The quarantined sibling
(`dial_grid_372col_2026-05-29_quarantined.parquet`) correctly screens 0/106
remaining ladders.

## Blast radius — per-dataset verdict

| Dataset | Verdict | Detail |
|---|---|---|
| `dial_grid_372col_2026-05-29.parquet` (local + R2) | **CORRUPT — 9/115 ladders (known, pre-quarantined)** | Reproduces the zensim-side finding exactly (see detector validation above). Local↔R2 byte-identical (ETag `a3f4eb…-2` / md5 `cc7a4715…` for the un-quarantined file). |
| `dial_grid_372col_2026-05-29_quarantined.parquet` (local + R2) | **CLEAN (post-quarantine)** | 0/106 ladders flagged, 0 row violations. Local↔R2 byte-identical (ETag `1a8ef228…` / md5 `1a8ef228…`). |
| `corruption_grid_372col_2026-05-28.parquet` (local + R2) | **CLEAN + NOT-GPU-EXTRACTED** | 0/672 groups flagged, 0/2016 row violations. zensim's own pointer doc (`benchmarks/eval_grids_2026-05-29.pointer.md`) states this grid is "CPU-extracted via `extract_features_372col`" — consistent with the clean empirical result. Local↔R2 byte-identical (ETag `7d18e8ff…` / md5 `7d18e8ff…`). |
| `kadis700k_canonical_2026-06-30.parquet` (zensim-only KADIS) | **NOT-GPU-EXTRACTED** | `README.md`: "Chunk-mode sweep … pure-CPU config: `METRICS=zensim` (the CPU zensim crate has no GPU dep)". Not independently re-screened (verified via build-provenance doc, not by re-running the detector) — trustworthy by construction since the CPU crate doesn't share this code path at all. |
| `kadis700k_canonical_gpu_2026-07-01.parquet` (KADIS-GPU) | **CLEAN, despite GPU-extracted features** | `README_gpu.md`'s `METRICS=zensim-gpu,ssim2-gpu,…` confirms `feat_0..feat_371` IS extracted via the buggy `zensim-gpu` code path. Empirically screened 0/140,000 `source_id` ladders flagged, 0/700,000 row violations. Explained structurally: every persisted distorted image is a FIXED 512×384 (both `512 = 2^9` and `384 = 128·3` divide down cleanly through all 4 pyramid scales — 512→256→128→64, 384→192→96→48 — never odd), so this corpus never enters the buggy code path's failure mode regardless of the extractor used. |
| `kadis_cvvdp_val.parquet` (`strategy-fleet-2026-07-02/derived/`) | **CLEAN** | Same `source_id`/`severity_level` keys and 512×384 mechanism as KADIS-GPU above. Empirically screened 0/14,000 ladders, 0/70,000 row violations. |
| `kadis_cvvdp_train.parquet` (same dir, 927 MB) | **UNVERIFIABLE (not independently re-screened this session)** — presumed CLEAN by the same 512×384 structural argument as its val sibling, but not empirically confirmed for time/resource reasons. |
| `canonical-picker-2026-06-27/{zenjpeg_lossy, zenwebp_lossy, zenwebp_lossless, zenavif_lossy, zenpng_lossless, zenjxl_lossy, zenjxl_lossless}/train.parquet` (7 datasets) | **CLEAN + NOT-GPU-EXTRACTED (for features)** | `docs/CLEAN_PICKER_PROGRAM.md`: "metrics `ssim2`+`zensim`" (plain `zensim`, not `zensim-gpu`). Empirically confirmed clean on ALL 7 train splits despite substantial odd-dimension content (up to 21% of rows have odd width in `zenjpeg_lossy`; widths like 43, 171, 341, 85, 683, 427, 597 are all odd) — 0 flagged `(origin_id, width, height)` ladders, 0 row-level range violations, in every one of the 7 datasets. `val`/`test` splits not independently re-screened (same pipeline/corpus as `train`, same conclusion expected). |
| `/mnt/v/output/jxl-hqfill-A-2026-07-01/features/*.parquet` (local, zenjxl HQ-fill A re-sweep) | **CORRUPT** | Launcher script `scripts/sweep/hqfill_A_local.sh` line 60: `--metric zensim-gpu` (confirmed GPU-extracted). 56 of 1650 `image_path` ladders flagged (3.4%), 798 of 23,100 rows range-violated (3.5%). All flagged renditions have a height that becomes odd at some pyramid scale even where the top-level height itself is even (e.g. 238→119, 318→159, 738→369) — exactly the compounding mechanism this fix addresses. |
| `s3://zentrain/strategy-fleet-2026-07-02/derived/bigcodec_hqdedup_valdigits_2026-07-02.parquet` | **CORRUPT** | 5 of 2745 `ref_basename` ladders flagged (0.18%), 42 of 114,871 rows range-violated (0.037%). Shares at least one specific corrupted source (`o_1551.png.scale1024x769.png`) with the hqfill-A dataset above — same lineage. |
| `.../bigcodec_hqdedup_traindigits_2026-07-02.parquet` (5.3 GB) | **CORRUPT** | 82 of 4585 ladders flagged (1.79%), 1171 of 2,322,579 rows range-violated (0.05%). This is the `bigcodec` training group referenced by the ablation-bake manifests (e.g. `manifests/ab_base_s7.toml`'s `[[training.groups]] name = "bigcodec"` / `"bigcodec_val"`) — i.e. these corrupted feature rows have already been used as **training and validation data for the w11-era ablation bakes**. The w11 report's own characterization ("0.95% of `bigcodec_val` rows land below raw −2 … mostly honest-direction but absurd-magnitude … a linear-head property") should be revisited: at least some of that tail is now independently confirmed to be **extraction garbage**, not solely a linear-head extrapolation-sensitivity property. |
| `.../bigcodec_valdigits_2026-07-02.parquet` (non-`hqdedup`, 283 MB) | **CLEAN** | 0 of 1382 ladders flagged, 0 of 147,067 row violations. **Distinct dataset from `bigcodec_hqdedup_valdigits` above** despite the similar name — fewer unique `ref_basename` groups (1382 vs 2745), so this is not simply "the hqdedup file before deduplication." Do not assume clean/corrupt status transfers between `bigcodec_*` files based on naming alone — verified independently. |
| `.../bigcodec_traindigits_2026-07-02.parquet` (non-`hqdedup`, 5.5 GB) | **CLEAN (row-level)** | 0 of 2,946,036 rows range-violated. 2307 `ref_basename` groups — matches the canonical-picker datasets' `(origin_id, width, height)` group count exactly, confirming this file is drawn from the same regular picker-corpus (CPU) sweep, not the GPU HQ-fill pass. Full ladder bit-constancy check not run (row-level alone is 0, and the group-count match makes the CPU-extraction explanation solid) — if a future session wants full parity with the other entries, re-run `screen_bigcodec_train.py`-style grouping. |
| `.../bigcodec_hqfill_2026-07-02.parquet`, `bigcodec_hqfill_dedup_2026-07-02.parquet`, `bigcodec_hqfill_traindigits_2026-07-02.parquet`, `bigcodec_hqfill_valdigits_2026-07-02.parquet`, `bigcodec_mm6_traindigits_2026-07-02.parquet` | **NOT SCREENED this session, but LIKELY CORRUPT** (time/resource-bounded) | The `hqfill`/`hqdedup` naming (confirmed: `hqdedup` = "HQ-fill, deduplicated") ties these to the SAME GPU HQ-fill sweep pipeline as the confirmed-corrupt `bigcodec_hqdedup_*` pair and the confirmed-corrupt `/mnt/v/output/jxl-hqfill-A-2026-07-01/` local sidecars (both traced to `scripts/sweep/hqfill_A_local.sh`'s `--metric zensim-gpu`). Unlike the `valdigits`/`hqdedup_valdigits` naming collision (which turned out to be two genuinely different corpora), every file in THIS group shares the `hq*` prefix consistently, so the inference is stronger here — but still unverified. Sizes: 10.9 GB, 10.3 GB, 5.6 GB, 286 MB, 3.6 GB respectively. Re-run the same screen before using any of these for training. |
| `.../bigcodec_5p7M_2026-07-02.parquet` (10.8 GB) | **NOT SCREENED this session** (time/resource-bounded) | No `hq*` prefix, so by the pattern above it MIGHT be the clean regular-sweep family (like `bigcodec_traindigits`/`valdigits`) — but this is an inference, not a measurement. Verify before use. |
| `konfig_train_2026-07-02.parquet` (same `strategy-fleet-2026-07-02/derived/` family) | **CLEAN** | 0/20 `ref_basename` ladders, 0/1090 row violations. Source naming (`SRC01_PartA`-style) has no embedded scale/dimension tag, unlike the picker/bigcodec corpora — likely a fixed/controlled-resolution corpus (matches typical JPEG-committee CTC source conventions) or CPU-extracted; mechanism not independently confirmed, but the empirical result is unambiguous. |

## What this means for zensim-side docs (NOT edited here — zensim has its own session/marker)

The following zensim-repo docs reference this bug and should be updated
once this fix is confirmed landed on `zenmetrics` `master`:

- `benchmarks/eval_grids_2026-05-29.pointer.md` — currently says "the
  zensim-gpu odd-dim extraction bug ... needs its own session/marker
  (zenmetrics-side)" for the dial-grid-v2-rebuild blocker; can now note
  the root cause is fixed and the rebuild is unblocked (a v2 grid still
  needs to be built with the fixed extractor — that's a zensim-side data
  job, not something this fix does by itself).
- `benchmarks/linear_projections_2026-07-03.md` §w11 — same cross-reference
  ("zensim-gpu odd-dim extraction bug (zenmetrics-side, needs its own
  session/marker)").
- `benchmarks/provenance_best_results_2026-07-04.md` §f155 tail forensics —
  same cross-reference, plus it should separately note the **new
  finding this session**: the `bigcodec_hqdedup_{train,val}digits`
  parquets (used as ablation-bake training/val data) are independently
  confirmed to contain the SAME class of GPU-extraction garbage, not
  purely "real-input fragility" as previously characterized — see this
  doc's blast-radius table.
- `benchmarks/bsdr_shaping_forensics_2026-07-05.md` — the forensic
  timeline; could add a closing entry noting the zenmetrics-side fix
  landed same-day.
- `benchmarks/dial_reach_expanded_2026-05-29.md` — currently documents
  "~25% of cells drop to NaN on odd-dimension images (GPU zensim path)"
  as an accepted, separate pathology from the non-NaN-garbage bug this
  doc fixes. Worth a follow-up measurement (not done in this session):
  whether the NaN rate on odd-dim images changes now that the
  duplicate-boundary-row corruption is gone (plausible it drops, since a
  degenerate near-zero-variance situation that used to trigger a NaN
  might not occur once the extra corrupted row is gone — not verified).

None of the above were edited by this session — per the task boundary,
zenmetrics fixes zenmetrics; zensim's docs are zensim's own session's to
update.
