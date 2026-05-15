# Changelog

Workspace conventions per the global rules:

- One `[Unreleased]` section accumulates changes for the next release.
- Per-crate headings (`## cvvdp-gpu`, `## zen-metrics-cli`, …) sit under
  each version section since this repo ships multiple crates.
- `### QUEUED BREAKING CHANGES` accumulates breaks that need to land
  together — only cleared when the corresponding major (or minor for
  0.x) release ships.
- Every entry MUST include the short commit hash(es) that implemented
  it. Reference the merge or final commit for multi-commit features.

## [Unreleased]

### QUEUED BREAKING CHANGES

(none yet)

### Changed

#### cvvdp-gpu

- **`Cvvdp::score` and `Cvvdp::score_with_reference` now route
  through the GPU pipeline** (`compute_dkl_jod`), replacing the
  host-scalar reference path. Output matches the prior host
  path to f32 noise (verified by
  `compute_dkl_jod_matches_host_scalar` at ≤ 0.005 JOD) and the
  pycvvdp v1 R2 manifest to ≤ 0.005 JOD (verified by
  `shadow_jod_gpu`). The switch was explicitly pre-promised in
  `lib.rs` ("Switching `score` over to the GPU path is the
  remaining chunk of pipeline work") and was unblocked by tick 207's
  tightened manifest-parity tolerances. Callers that need the
  all-host path can still invoke
  `host_scalar::predict_jod_still_3ch` directly;
  cpu-runtime callers use `compute_dkl_jod_host_pool`.
  Also tightened `tests/pipeline_score.rs` `cvvdp_score_matches_v1_manifest`
  from 0.05 → 0.005 JOD (measured diffs 0.0000–0.0033).
- Removed the dead `reflect()` helper in `kernels/pyramid.rs` —
  superseded in tick 206 when `gausspyr_reduce_scalar` was
  rewritten to bug-compatible zero-pad + explicit boundary
  patches matching pycvvdp.
- **Manifest-parity tolerances tightened to 0.005 JOD across the
  v1 R2 corpus** (`tests/shadow_jod.rs`). Was a per-q schedule
  (0.5 JOD at q=1, 0.1 at q=5, 0.05 at q≥20 GPU; flat 0.05 host)
  before ticks 204/206 closed the chroma_shift and 73×91 odd-dim
  drifts. Measured diffs are now 0.0000–0.0031 JOD across all 6
  q levels (host + GPU) — well within the same 0.005 tolerance
  the other parity tests use.
- `pipeline_score.rs` host-vs-GPU corpus tests
  (`compute_dkl_t_p_bands_matches_host_on_corpus_256x256`,
  `compute_dkl_d_bands_matches_host_on_corpus_256x256`) updated
  to apply the tick-204 `CSF_BASEBAND_RHO` override in their
  host reference computation — caught when running the full
  suite after tightening shadow_jod tolerances.

### Added

#### cvvdp-gpu

- **`Cvvdp::compute_dkl_jod_host_pool`** — CPU-backend-compatible
  variant of `compute_dkl_jod`. Reads D bands back to host and
  pools them with the host-scalar `lp_norm_mean` instead of the
  GPU `pool_band_3ch_kernel` (which uses `Atomic<f32>::fetch_add`,
  unsupported by `cubecl-cpu`). Same JOD output as
  `compute_dkl_jod` to f32 noise (`diff = 0.000000` measured on
  the 32×32 odd-dim test pair); use it on the CPU backend or
  any runtime that lacks atomic f32 add. New
  `compute_dkl_jod_host_pool_matches_compute_dkl_jod` test pins
  the two paths together. Closes the standing CPU-backend
  blocker noted in `lib.rs`.
- **`tests/cpu_backend.rs`** — cpu-runtime smoke + parity tests
  exercising `compute_dkl_jod_host_pool` on `cubecl::cpu::CpuRuntime`.
  Validates the lib.rs claim that the cpu backend works:
    JOD finite + in [0, 10] on a 32×32 synth pair.
    cpu backend JOD vs host_scalar JOD: `diff = 0.000000`.
  All other test files gate themselves out of cpu-only builds; this
  file is the only place cpu-backend coverage lives.
  Run with `cargo test -p cvvdp-gpu --no-default-features --features cpu`.

#### cvvdp-gpu (docs)

- `Cvvdp::score` now has a `no_run` doctest example showing the
  canonical `Cvvdp::<CudaRuntime>::new` → `.score(&ref, &dist)`
  shape against a 64×64 byte-identical pair. Fills the only
  remaining doc gap on the crate's headline public entry point —
  the host-only and host-pool paths already had doctests via
  `host_scalar::predict_jod_still_3ch`, `compute_dkl_jod_host_pool`,
  and `compute_dkl_jod_host_pool_with_warm_ref`.

- **`crates/cvvdp-gpu/README.md`** — new crate-root README
  mirroring the peer GPU-metric crates' structure
  (`ssim2-gpu`, `zensim-gpu`, `dssim-gpu` all had one;
  cvvdp-gpu didn't). Covers the multi-vendor pitch
  (CUDA / WGPU / HIP / cubecl-cpu), single-image + cached-ref
  + warm-ref usage shapes, JOD 0..10 score interpretation
  (higher = better, matching pycvvdp convention), the
  `compute_dkl_jod_host_pool` workaround for cubecl-cpu and
  Metal (Atomic<f32>::fetch_add gotcha), the
  `CVVDP_COLUMN_NAME` / `CVVDP_IMPL_TAG` sweep-tooling story
  for parquet sidecars, the `parity-goldens` feature gate,
  and the standard build / license footer. Tick 285.

- README "Sweep tooling" section now links to
  `docs/CVVDP_SIDECAR_SCHEMA.md` (full identity-tuple +
  score-column + manifest spec) and `docs/BURN_PORT_PLAN.md`
  (scoping for the future `cvvdp_burn_v*` column that would
  land alongside `cvvdp_imazen_v*` and `cvvdp_pycvvdp_v054`).
  Closes the navigability gap a reader following
  `CVVDP_COLUMN_NAME` would hit. Tick 287.

### Fixed

#### cvvdp-gpu

- **Warm-ref state invalidation honored on all 6 dispatchers that
  overwrite `bands_ref`.** Tick 236 fixed the two weber-chain
  dispatchers (`compute_dkl_weber_pyramid`,
  `compute_dkl_t_p_bands`); tick 237 audits the rest and finds
  two more silent-stale-scalar holes through the Laplacian chain:
  `compute_dkl_laplacian_pyramid` and `compute_dkl_csf_weighted_bands`
  both run `_dispatch_laplacian_pyramid_gpu` which overwrites
  `bands_ref[k].planes[c]` with Laplacian bands (not the Weber
  bands the warm-ref state was built on). Pre-fix, a subsequent
  `compute_dkl_jod_with_warm_ref` would silently mix Laplacian
  bands against the cached Weber-baseband scalar. Both functions
  now clear `warm_ref_baseband_log_l_bkg` at entry; the
  `Cvvdp::warm_reference` docstring lists all 6 invalidators;
  and the regression test
  `warm_state_invalidates_after_each_documented_dispatcher`
  extends from 4 → 6 cases.

  Tick 238 closes the audit: the `warm_reference` docstring now
  also documents `Cvvdp::score` / `Cvvdp::score_with_reference`
  as transitive invalidators (via `compute_dkl_jod` since tick
  213) and `Cvvdp::set_reference` as an explicit non-invalidator
  (it only stashes host bytes). Regression test extends 6 → 8
  invalidator cases; new sibling test
  `set_reference_does_not_invalidate_warm_state` pins the
  non-invalidator contract — a future refactor that turned
  `set_reference` into an eager GPU dispatch would silently
  break batch-scoring callers and surface here.

### Changed (docs, tests, dedup — post-tick-238)

Many small docs / tests / dedup chunks landed under this bucket
during the ticks 239-273 maintenance run. They follow the
Keep-a-Changelog `Changed` semantics (no behavioural shift in the
public API; refactors, comment refreshes, regression-test pinning,
and helper extractions). Pre-tick-238 entries above stay in their
original Fixed/Added/Changed sections.

### Changed (post-tick-238)

#### cvvdp-gpu

- `crates/cvvdp-gpu/docs/PORT_STATUS.md` pipeline-stage table
  (line 15, "Per-band pooling" row) named the test-only
  `pool_band_kernel` as the GPU kernel consumed by
  `compute_dkl_jod`. Same stale-reference shape as tick
  319's README fix — the production dispatcher is the fused
  3-channel `pool_band_3ch_kernel`, per the tick-291 audit.
  Updated the row to: "GPU `pool_band_3ch_kernel` (fused
  3-channel, atomic f32 partials, one launch per pyramid band)
  consumed by `compute_dkl_jod`. Single-channel
  `pool_band_kernel` retained as a test-only entry point."
  The "Resolved tick 208" entry further down (line 112) keeps
  its `pool_band_kernel` reference — that's accurate
  historical context (tick 208 predates the tick 165 fusion
  to `pool_band_3ch_kernel`). Tick 320.

- `crates/cvvdp-gpu/README.md` "CPU backend" section had a
  stale reference to `pool_band_kernel` (the single-channel
  pool kernel) as the source of the `Atomic<f32>::fetch_add`
  that cubecl-cpu doesn't support. Tick 291's audit
  established that production dispatches the fused
  `pool_band_3ch_kernel` instead (one launch per pyramid band,
  3× fewer launches than the single-channel form); the
  `pool_band_kernel` symbol is retained only for the
  `tests/pool_scalar.rs::pool_band_kernel_matches_host_lp_norm_mean`
  unit-parity test. Updated the README to name the fused
  3-channel production kernel + the "one launch per pyramid
  band" descriptor. The cpu-runtime workaround (route through
  `compute_dkl_jod_host_pool`) is unchanged. Tick 319.

- `score_with_reference_errors_without_set_reference` (in
  `tests/pipeline_score.rs`) was using
  `format!("{err:?}").contains("NoCachedReference")` to verify
  the error variant. Substring matching on Debug output is
  brittle — a future variant rename that landed
  "NoCachedReferenceV2" or similar by accident would silently
  pass the substring check. Other tests in the same file
  (`invalid_image_size_surfaces_on_too_small_dims`,
  `dimension_mismatch_surfaces_on_wrong_size_inputs`) use
  proper `match err { Error::X => {}, other => panic! }`
  pattern matching on the variant via the public Error API,
  which pins identity directly. Switched this test to the
  same pattern. Test passes post-change. Tick 318.

- `Error::NoWarmReference` variant docstring listed two
  example invalidators (`compute_dkl_jod`,
  `compute_dkl_d_bands`) — fine when the list was short, but
  tick 314 grew the canonical invalidator list to 9
  (compute_dkl_jod, ..._host_pool, _d_bands, _weber_pyramid,
  _t_p_bands, _laplacian_pyramid, _csf_weighted_bands, score,
  score_with_reference). Maintaining a parallel example list
  on the Error variant is a duplicate-maintenance burden the
  variant doc would inevitably fall behind on — exactly what
  tick 314's audit caught.
  Removed the per-example list and instead pointed at
  `Cvvdp::warm_reference`'s docstring as the canonical
  invalidator source, plus a pointer to the
  `warm_state_invalidates_after_each_documented_dispatcher`
  regression test that pins each method to the contract. Also
  added the cpu-runtime variant
  (`compute_dkl_jod_host_pool_with_warm_ref`) to the
  "called without prior `warm_reference`" sentence — the
  variant doc previously only named
  `compute_dkl_jod_with_warm_ref` even though both methods
  return this variant from the same `.ok_or(NoWarmReference)`
  site. 14 pipeline_score tests still pass. Tick 317.

- `Error::InvalidImageSize` Display message was misleading.
  The variant is documented as dual-purpose (image too small
  for the configured pyramid OR GPU readback/dispatch failed,
  because cubecl's read errors aren't separable yet) but
  the Display impl only mentioned the image-size case:
  "image is too small for the configured pyramid". A user
  hitting a GPU readback failure would see this and
  investigate image dimensions instead of the actual backend
  failure. Updated to: "image too small for the configured
  pyramid, or GPU readback/dispatch failed (see the
  InvalidImageSize variant docs — cubecl's read errors aren't
  separable yet so both surface as this variant)". Also
  extended `error_display_messages_are_actionable` to pin
  the new dual-purpose hint by asserting the message contains
  "GPU"/"readback"/"dispatch" in addition to the existing
  "small"/"pyramid" check. Test passes. Tick 316.

- `gauss_chain_helpers_do_not_invalidate_warm_state` regression
  test extended from 2 → 3 non-invalidators. Adds
  `compute_dkl_jod_host_pool_with_warm_ref` — pinning the
  tick-314 docstring claim that this method only READS the
  cached scalar (`.ok_or(NoWarmReference)`) and must preserve
  warm state across calls. A refactor that accidentally
  cleared the cached scalar (e.g. moving the warm-ref
  host-pool path through `_dispatch_d_bands_into_scratch` by
  mistake) would silently break cpu-runtime batch scoring —
  this case catches it. Test passes. Tick 315.

- `Cvvdp::warm_reference` docstring's invalidator list was
  missing `compute_dkl_jod_host_pool` — it routes through
  `_dispatch_d_bands_into_scratch` →
  `_dispatch_ref_weber_pyramid_only` which clears
  `warm_ref_baseband_log_l_bkg`, same as the GPU jod path. A
  caller batch-scoring on cpu-runtime who mixed
  `compute_dkl_jod_host_pool` calls between
  `warm_reference` + `compute_dkl_jod_host_pool_with_warm_ref`
  would silently lose the warm state without the docstring
  warning. Added the missing entry; also noted explicitly that
  `compute_dkl_jod_host_pool_with_warm_ref` does NOT invalidate
  (it only reads the cached scalar).
  Extended the `warm_state_invalidates_after_each_documented_dispatcher`
  regression test from 8 → 9 invalidators to pin the
  `compute_dkl_jod_host_pool` contract directly. Test passes.
  Tick 314.

- New regression test:
  `debug_assert_fires_when_ppd_mismatches_geometry_on_warm_ref_path`
  in `tests/pipeline_score.rs`. Sibling to the existing tick-244
  test that pinned the tick-243 `debug_assert_ppd_matches_geometry`
  contract on `compute_dkl_jod`. All 6 public methods share the
  same assert-at-entry contract (`compute_dkl_jod` /
  `compute_dkl_d_bands` / `compute_dkl_t_p_bands` /
  `compute_dkl_jod_host_pool` /
  `compute_dkl_jod_host_pool_with_warm_ref` /
  `compute_dkl_jod_with_warm_ref`), but only `compute_dkl_jod`
  had a regression test. A refactor that dropped the assert
  from `compute_dkl_jod_with_warm_ref` specifically would have
  slipped through. The new test warms a reference, then calls
  `compute_dkl_jod_with_warm_ref` with a phone-resolution PPD
  (110.09 ≠ STANDARD_4K's 75.4) and expects the debug-only
  assert to fire. Both ppd-mismatch tests pass; the
  `#[cfg(debug_assertions)]` gate means release builds skip
  the test definition entirely (matches the existing pattern).
  Tick 313.

- Dropped dead `ppd: f32` parameter from two private GPU
  dispatchers (`_dispatch_d_bands_into_scratch` and
  `_dispatch_d_bands_dist_and_band_loop`) plus the redundant
  `let _ = ppd;` discard in `compute_dkl_t_p_bands`.
  All 6 public methods that take `ppd` validate it via
  `debug_assert_ppd_matches_geometry(ppd)` at entry; the value
  itself isn't consumed by the GPU stages (logs_row is
  pre-uploaded against the construction-time geometry at
  `Cvvdp::new` time, so the runtime band-loop kernel reads
  the cached rho-per-band instead of recomputing from ppd).
  The private helpers were threading ppd through dead until
  the `let _ = ppd;` discard at the bottom of each. Updated
  signatures and all 5 call sites
  (3× `_dispatch_d_bands_into_scratch` from
  `compute_dkl_jod` / `compute_dkl_d_bands` /
  `compute_dkl_jod_host_pool` and
  2× `_dispatch_d_bands_dist_and_band_loop` from
  `compute_dkl_jod_with_warm_ref` /
  `compute_dkl_jod_host_pool_with_warm_ref`).
  Public method signatures unchanged. 14 `pipeline_score`
  tests + `compute_dkl_jod_matches_pycvvdp_at_12mp_synth`
  all pass — JOD output is byte-identical. Tick 312.

- Dropped two unused dev-dependencies from
  `crates/cvvdp-gpu/Cargo.toml`:
  - `bytemuck` — zero references anywhere in `src/`, `tests/`,
    `examples/`, or `benches/`. The cubecl APIs we use
    (`f32::as_bytes` / `client.create_from_slice`) pull
    bytemuck transitively where needed.
  - `serde` — only `serde_json` is used directly; `serde_json`
    pulls the `serde` traits transitively.
  Pure dependency hygiene; cargo no longer asks rustc to link
  these two crates into the dev-build. `cargo build --tests
  --benches --examples` is clean under both `--features cuda`
  and `--features wgpu`. `cvvdp_score_matches_v1_manifest`
  still passes. Tick 311.

- `kernels::csf::interp1_clamped` binary-search midpoint switched
  from `(lo + hi) / 2` to `usize::midpoint(lo, hi)` (stable since
  Rust 1.85; workspace MSRV 1.93). Overflow-safe by construction
  and matches the canonical idiom — clippy `-W clippy::pedantic`
  was suggesting it. The shorthand can't overflow at our
  32-entry LUT sizes, but the explicit form documents intent
  and removes a pedantic-lint speed bump for anyone who flips
  `clippy::pedantic` on. `csf_scalar` parity tests
  (`sensitivity_matches_pycvvdp_v0_5_4`,
  `precomputed_band_weights_match_pointwise`,
  `flatten_band_weights_layout`,
  `sensitivity_is_finite_at_extremes`) still all pass. Tick 295.

- `kernels::pyramid::weber_contrast_pyr_dec_scalar` nested
  `fn build_pyr` helper was defined mid-body, after the
  `n_levels` resolution and `debug_assert!` statements, which
  clippy `-W clippy::pedantic`'s `items_after_statements` lint
  flags as confusing (items exist from the start of scope; the
  visual ordering implies a runtime dependency that doesn't
  exist). Moved the nested fn ahead of the statements without
  changing its body — pure reordering. `pyramid_scalar` parity
  tests (6 tests including `reduce_matches_pycvvdp` and
  `one_band_laplacian_matches_pycvvdp`) and the lib-internal
  `kernels::pyramid::tests` (`reduce_halves_dimensions`,
  `reduce_preserves_constant_signal`, `expand_*`, etc.) all
  still pass. Tick 296.

- `tests/pipeline_color.rs` had 9 identical
  `const TOLERANCE: f32 = 0.005;` declarations, each inside a
  separate test function and each tripping clippy
  `-W clippy::pedantic`'s `items_after_statements` (the const
  followed the per-test `let pycvvdp_golden_jod = ...` golden
  load). Hoisted to file scope as a single
  `const TOLERANCE: f32 = 0.005;` with a docstring tying the
  number to the tick 207 tolerance schedule. The 9 inner
  declarations are gone; their bodies still reference
  `TOLERANCE` via outer-scope lookup. All 31 pipeline_color
  tests still pass post-change (including the 12 MP parity
  pair that takes ~30 s/test). Tick 297.

- Closed out the lossless-cast cleanup arc with the remaining
  6 `cast_lossless` warnings:
  - `src/pipeline.rs:731` (`b as u32`) — sRGB-byte → u32 lane
    pack inside the persistent `src_u32_scratch` fill loop
    (warm-ref-amortised version of the per-call buffer pack).
  - `tests/color_kernel.rs:47` (`|&b| b as u32`) — same
    byte → u32 pack inside the color-kernel parity test setup.
    Also rewrote the closure as `.copied().map(u32::from)` so
    the `Copy` bound replaces the explicit deref pattern.
  - `src/pipeline.rs:2507` and `:2598` (`jod as f64`) — the
    public `Cvvdp::score` and `Cvvdp::score_with_reference`
    return-value widenings.
  - `examples/manifest_parity_probe.rs:100`
    (`r[(y*w + x)*3 + c] as i32`) — synth_noise_pair
    distortion construction; widens u8 to i32 for the +noise
    addition.
  - `tests/pipeline_color.rs:2075`
    (`ref_srgb[i + c] as i64`) — same noise-fixture pattern
    inside the 256² parity test.
  Lossless-widening warning count across cvvdp-gpu is now zero
  after ticks 300/301/302/304/305 (u16/i16/u64/u32/i32/i64/f64
  variants). Tests that exercise the changed code paths
  (`color_kernel::srgb_to_dkl_kernel_matches_host_scalar`,
  `pipeline_color::compute_dkl_jod_matches_pycvvdp_at_256x256_noise`)
  pass post-change. Tick 305.

- Two strict-equality `f32` `assert_eq!` calls in
  `tests/pool_scalar.rs` were tripping clippy
  `-W clippy::pedantic`'s `float_cmp` — but both are
  intentional bit-pattern equality checks
  (kernel-test invariants about untouched partial slots and
  fill-kernel output), not approximate-equality assertions.
  Switched to `.to_bits()` comparisons (`partials[i].to_bits()
  == 0.0_f32.to_bits()`, `v.to_bits() == value.to_bits()`)
  with a one-line comment per site explaining why bit-pattern
  equality is the correct test. `pool_scalar` test suite
  (8 tests) all pass post-change including
  `gpu::pool_band_kernel_matches_host_lp_norm_mean` and
  `gpu::fill_f32_kernel_writes_uniform_value`. Tick 304.

- Two `for x in container.iter()`/`for x in container.iter_mut()`
  sites switched to the more idiomatic `for x in &container`
  / `for x in &mut container` form (clippy
  `-W clippy::pedantic`'s `explicit_iter_loop`):
  - `tests/common/mod.rs:104` (hex-encode loop over `Sha256`
    finalize output, called by `manifest_sha256_hex` /
    `fetch`).
  - `tests/pipeline_color.rs:143` (host pyramid reduce
    loop over per-channel plane buffers).
  The `tests/common/mod.rs` site clears 3× because the file is
  consumed via `#[path]` from bench/example/test scopes (4
  total `explicit_iter_loop` warnings cleared across cvvdp-gpu).
  Also dropped two unnecessary trailing commas in
  `tests/pipeline_score.rs::dimension_mismatch_surfaces_on_wrong_size_inputs`'s
  `assert_eq!` macro calls (clippy `unnecessary_trailing_comma`).
  All affected tests
  (`dimension_mismatch_surfaces_on_wrong_size_inputs` plus
  the parity tests that reduce host pyramids) pass post-change.
  Tick 303.

- Six `u32 as u64` lossless widening casts switched to
  `u64::from(...)`. Sites:
  - `tests/common/mod.rs:409`
    (`v1_corpus_jod_golden` parameter comparison
    `Some(q as u64)`).
  - `examples/time_12mp.rs:132` per-pixel cost math
    (`(W as u64) * (H as u64)` for total-pixels divisor).
  - `examples/time_size_sweep.rs:104` per-bucket pixel
    count.
  - `benches/score.rs` twice — both `Throughput::Elements`
    calls in the GPU-JOD bench setup.
  Clippy `-W clippy::pedantic`'s
  `cast_lossless` flagged them with `an as cast can become
  silently lossy if the types change in the future` — the
  `u64::From<u32>` impl encodes the widening intent
  explicitly and would surface a hard compile error if a
  caller swapped `u32` for a wider type. Throughput-math and
  the corpus-q comparison evaluate to the same `u64`.
  `pipeline_score::cvvdp_score_matches_v1_manifest` (the
  primary consumer of `v1_corpus_qs` → `v1_corpus_jod_golden`)
  still passes. Tick 302.

- Seven `u8 as i16` widening casts in the
  `chroma_shift` synth-pair pattern
  (`(byte as i16 + 16).clamp(0, 255) as u8`) switched to
  `i16::from(byte) + 16` for the lossless widening.
  Six sites in `tests/pipeline_color.rs` (one per
  `chroma_shift`-family test, all using the same `.flat_map`
  closure) plus one in
  `examples/manifest_parity_probe.rs::synth_chroma_shift_pair`.
  Bit-identical arithmetic; all 9 `chroma_shift` parity tests
  (`compute_dkl_jod_matches_pycvvdp_at_256x256_chroma_shift`,
  `..._with_warm_ref`, plus the 7 per-stage shadow tests)
  still pass. Tick 301.

- Six `u8 as u16` widening casts (3 each in
  `tests/pipeline_color.rs::compute_dkl_jod_matches_pycvvdp_at_256x256_blur3x1`
  and `..._blur1x3`) and six more in
  `examples/manifest_parity_probe.rs::synth_blur3x1_pair` and
  `synth_blur1x3_pair` switched to `u16::from(...)`. Clippy
  `-W clippy::pedantic`'s
  `cast_lossless`/`unnecessary_cast` family flagged them
  because the `From<u8>` impl for `u16` is infallible — the
  bare `as` form works but is the wrong idiom for
  lossless widening conversions. The blur synth-pair fixtures
  (3-tap horizontal / 3-tap vertical mean over `u16`
  accumulator) produce identical pixel arithmetic; the 2
  blur parity tests
  (`compute_dkl_jod_matches_pycvvdp_at_256x256_blur3x1`,
  `..._blur1x3`) still pass. Tick 300.

- `tests/common/mod.rs` had 3 sites using Debug formatting
  (`{path:?}`, `{local:?}`) inside `panic!` for path-typed
  values, which clippy `-W clippy::pedantic`'s
  `unnecessary_debug_formatting` flags — Debug for
  `&Path`/`&PathBuf` renders with surrounding quotes and escape
  sequences (e.g. `"foo bar.png"`), while Display via
  `path.display()` shows the bare path (`foo bar.png`). For
  panic-context error messages the Display form is more
  readable. Switched all 3 sites
  (`fetch` cache-write panic at line 90,
  `load_rgb_bytes` open + decode panics at 449/451) to
  positional format args using `path.display()` /
  `local.display()`. The 3 unique warnings were each counted
  3× because `tests/common/mod.rs` is consumed via `#[path]`
  from the bench/example/test scopes (9 total pedantic
  warnings cleared). All 31 `pipeline_color` tests still pass
  post-change (the suite consumes `load_rgb_bytes` via
  `common::Backend` + image-corpus paths). Tick 299.

- `tests/common/mod.rs` had 4 sites using closure-wrapped
  method calls (`.and_then(|j| j.as_f64())`,
  `.and_then(|n| n.as_u64())`) that
  clippy `-W clippy::pedantic`'s
  `redundant_closure_for_method_calls` flags — the bare method
  pointer form is shorter and stylistically preferred. Each
  site (in `pycvvdp_synth_golden_jod`,
  `v1_corpus_jod_golden`, and `v1_corpus_qs`) now uses
  `.and_then(serde_json::Value::as_f64)` or
  `.and_then(serde_json::Value::as_u64)`. The 4 sites are each
  re-counted 3× because `tests/common/mod.rs` is consumed via
  `#[path]` from the bench/example/test scopes (12 total
  pedantic warnings cleared). `pipeline_score` corpus tests
  (`compute_dkl_jod_on_v1_manifest_corpus`,
  `score_with_reference_matches_score`,
  `cvvdp_score_matches_v1_manifest`, and 11 more) all still
  pass post-change. Tick 298.

### Fixed (post-tick-238)

#### cvvdp-gpu (docs)

- `kernels::csf::interp1_uniform` had a 14-line docstring whose
  opening 4 lines ("1-D linear interpolation in log-space along
  a monotonically increasing axis…") read as a generic
  intro that applied to either interpolator, blurring into the
  function-specific "Linear interp on a UNIFORMLY-spaced axis"
  description on line 78 with no separator. `interp1_clamped`
  underneath had no docstring at all.
  Rewrote both as standalone docs:
  - `interp1_uniform`: now opens with "uniformly-spaced axis
    via global-stride rescale" (the actual semantics), keeps the
    tick 199 / chroma-drift parity rationale, and adds an
    explicit "used for the outer L_bkg interp" pointer to
    `sensitivity_scalar` / `precompute_logs_row`.
  - `interp1_clamped`: gains a docstring explaining the
    binary-search bracket form, its applicability to any
    monotonically-increasing axis (uniform or not), and the
    pycvvdp-side rationale for using it on the inner rho axis
    (`torch.searchsorted` + linear interp) versus the L_bkg
    axis (`interp1q`). Cross-references `interp1_uniform`.
  Tick 294.

- `kernels/mod.rs` step 5 breadcrumb introduced in tick 291
  triggered 3 `clippy::doc_lazy_continuation` warnings — the
  new sentence about `pool_band_kernel` being test-only landed
  on continuation indentation (`//!    `) of bullet 5, which
  rustdoc/clippy now read as a malformed sub-list rather than
  body continuation. Split the breadcrumb out of the bullet
  into its own paragraph (blank `//!` separator) so it parses
  as flowing prose under the pipeline-order list. `cargo
  clippy --all-targets -W clippy::all` is back to zero
  warnings on both `--features cuda` and `--features wgpu`.
  Tick 293.

- `kernels::pool::pool_band_kernel` (single-channel) doc now
  explicitly notes it's not dispatched by
  `Cvvdp::compute_dkl_jod` — the production path uses the fused
  3-channel `pool_band_3ch_kernel`. Added a one-line breadcrumb
  pointing at the
  `tests/pool_scalar.rs::pool_band_kernel_matches_host_lp_norm_mean`
  parity test that justifies keeping the symbol public. A
  maintainer reading this kernel was previously left guessing
  why two near-duplicate `pool_band_*` kernels coexist; now the
  link from single → fused is symmetric (the 3ch docstring
  already cross-references the single-channel form as the
  base case). Tick 292.

- Pipeline-overview docstrings in `kernels/mod.rs` step 5,
  `pipeline.rs` step 6, `Cvvdp::compute_dkl_d_bands`'s "no
  readback" note, and `Cvvdp::compute_dkl_jod`'s ASCII pipeline
  diagram all referred to the GPU pool stage as `pool_band_kernel`.
  The production path dispatches `pool_band_3ch_kernel` (one
  fused 3-channel launch per band, ~3× fewer launches than the
  single-channel version) — `pool_band_kernel` survives only as
  a unit-test entry point in `tests/pool_scalar.rs`. Updated
  all 4 sites to name `pool_band_3ch_kernel` and noted the fused
  3-channel-per-launch property; added a one-line breadcrumb in
  `kernels/mod.rs` that `pool_band_kernel` is retained for the
  pool-scalar unit test. Tick 291.

- `cargo fmt --check` was failing on cvvdp-gpu — tick 298's
  `|n| n.as_u64()` → `serde_json::Value::as_u64` swap inside
  `v1_corpus_qs::filter_map` made the line too long for
  rustfmt's column limit, but the file wasn't reformatted at
  the time. Ran `cargo fmt -p cvvdp-gpu` to clean it up; the
  change splits the `filter_map` body across 4 lines (still
  consumed via `#[path]` from bench/example/test scopes).
  Also picked up several pipeline_color.rs `let (ref_srgb,
  dist_srgb) = common::synth_pair_*(...)` 2-line forms that
  now fit on a single line after ticks 278-280 shortened the
  helper names. `cargo fmt -p cvvdp-gpu --check`: clean.
  `pipeline_score::cvvdp_score_matches_v1_manifest` passes
  post-change. Tick 310.

- Added `#[must_use]` to 24 pure-return `pub fn`s where ignoring
  the return value is always a bug (clippy
  `-W clippy::must_use_candidate`). These are all
  host-scalar / kernel helpers and one method
  (`DisplayGeometry::pixels_per_degree`); the attribute is purely
  additive — callers that already use the return value are
  unaffected, and callers that drop it on the floor get a
  `unused_must_use` warning surfacing the bug at the use site.
  Breakdown:
  - `src/host_scalar.rs`: `predict_jod_still_3ch` (1)
  - `src/params.rs`: `DisplayGeometry::pixels_per_degree` (1)
  - `src/kernels/color.rs`: `srgb_byte_to_dkl_scalar` (1)
  - `src/kernels/csf.rs`: `sensitivity_scalar`,
    `sensitivity_corrected_scalar`, `precompute_logs_row`,
    `precomputed_band_weights`, `flatten_band_weights` (5)
  - `src/kernels/masking.rs`: `safe_pow`, `clamp_diff_soft`,
    `phase_uncertainty_no_blur`, `gaussian_blur_sigma3`,
    `phase_uncertainty_band`, `mask_pool_pixel`,
    `mult_mutual_pixel`, `mult_mutual_band` (8)
  - `src/kernels/pool.rs`: `lp_norm_mean`, `lp_norm_sum`,
    `met2jod`, `do_pooling_and_jod_still_3ch`,
    `pool_band_finalize` (5)
  - `src/kernels/pyramid.rs`: `band_frequencies`,
    `laplacian_pyramid_dec_scalar`,
    `weber_contrast_pyr_dec_scalar` (3)
  `must_use_candidate` warning count: 0 (was 24). The
  `#[cube(launch)]` GPU kernels are skipped because they return
  `()` and don't trigger the lint. `pool_scalar` (8) and
  `display_geometry` (2) test suites still pass post-change.
  Crate is `publish = false` so no semver implications.
  Tick 309.

- Added `# Panics` sections to 3 `pub fn` host-scalar /
  kernel helpers that can panic on out-of-spec input (clippy
  `-W clippy::missing_panics_doc`):
  - `host_scalar::predict_jod_still_3ch` — panics on
    `ref_srgb.len() != w*h*3` or `dist_srgb.len() != w*h*3`
    (the two `assert_eq!` calls at the top). Doc points at
    `Cvvdp::score` for the fallible `Result` variant routed
    through the same pipeline.
  - `kernels::pool::do_pooling_and_jod_still_3ch` — panics on
    empty `q_per_ch` (zero pyramid levels). cvvdp's pool stage
    is undefined on a zero-band input.
  - `kernels::pyramid::laplacian_pyramid_dec_scalar` — panics
    if the resolved level count is zero. Debug builds trip
    the `debug_assert!`; release builds reach the
    `gauss.pop().expect("at least one level")` line.
  `missing_panics_doc` warning count: 0 (was 3). 6 doctests
  still pass under the CI wgpu combo. Tick 308.

- Closed out the `# Errors`-section work started in tick 306.
  Added sections to the remaining 10 `Result`-returning public
  methods:
  - `compute_dkl_jod` and `compute_dkl_jod_host_pool` /
    `compute_dkl_jod_host_pool_with_warm_ref` — the canonical
    GPU and cpu-runtime JOD entry points documented in
    `lib.rs`. The warm-ref host-pool variant shares the
    tick-248 `DimensionMismatch`-before-`NoWarmReference`
    precedence rule with its all-GPU counterpart and the new
    `# Errors` section documents it.
  - 7 stage-debug helpers (`compute_dkl_planes`,
    `compute_dkl_gauss_pyramid`,
    `compute_dkl_laplacian_pyramid`,
    `compute_dkl_weber_pyramid`, `compute_dkl_t_p_bands`,
    `compute_dkl_d_bands`, `compute_dkl_csf_weighted_bands`) —
    each gets a uniform short `# Errors` section naming
    `DimensionMismatch` and `InvalidImageSize` with the
    specific stage chain that can fail.
  `clippy::missing_errors_doc` warning count: 0 (was 17 at
  start of tick 306). All 6 doctests still pass under the
  CI wgpu combo. Tick 307.

- Added `# Errors` sections to 7 user-facing public entry
  points (`Cvvdp::new`, `Cvvdp::new_with_geometry`,
  `Cvvdp::score`, `Cvvdp::set_reference`,
  `Cvvdp::score_with_reference`, `Cvvdp::warm_reference`,
  `Cvvdp::compute_dkl_jod_with_warm_ref`) — clippy
  `-W clippy::missing_errors_doc` was flagging all 17
  Result-returning public methods, but only the 7 user-facing
  ones really needed dedicated sections (the lower-level
  `compute_dkl_*_bands` helpers are exposed for testing /
  shadowing rather than primary use). Each new section
  enumerates the specific `Error` variants the method can
  return — including the tick-248 precedence audit detail
  that `compute_dkl_jod_with_warm_ref` returns
  `DimensionMismatch` *before* `NoWarmReference` when both
  conditions hold. All 6 doctests still pass under the CI
  wgpu combo (`cargo test --doc --features wgpu`). Cleared
  7 of 17 `missing_errors_doc` warnings. Tick 306.

- `host_scalar::predict_jod_still_3ch` had a stale comment
  claiming "weber_contrast_pyr path which we have NOT yet
  ported (vanilla Laplacian + linear DKL bands here vs.
  cvvdp's Weber-contrast Laplacian + log10(gauss) for L_bkg)".
  The Weber-contrast pyramid was ported in tick 24 (per
  `docs/PORT_STATUS.md`'s "Resolved tick 24" entry) and the
  surrounding code already calls
  `kernels::pyramid::weber_contrast_pyr_dec_scalar` — both
  `ref_weber` / `dis_weber` carry Weber-contrast bands and the
  log10-gauss `log_l_bkg`. Replaced the stale comment with an
  accurate description of the current baseband-bypass +
  non-baseband mult-mutual structure, the tick 204
  `CSF_BASEBAND_RHO = 0.1 cy/deg` override, and a
  forward-reference to the Weber-pyramid port history. Tick 290.

- `PoolingParams` scaffolding docstring referenced
  `BETA_CHANNEL` as the inlined `const` in `kernels::pool`, but
  the actual const there is `BETA_CH` (mirroring cvvdp's
  `beta_tch` field name). Grepping `BETA_CHANNEL` returned no
  hits, leaving a future maintainer reading the struct without
  a working pointer to the production value. Replaced
  `BETA_CHANNEL` with `BETA_CH` and added a one-line mapping
  note from the struct's `beta_channel` field to the const.
- `JodParams` docstring described `JOD = jod_a − jod_b · D^jod_c`,
  a 3-coefficient form the production code doesn't implement.
  `kernels::pool::met2jod` is a 2-coefficient piecewise function
  (`JOD_A`, `JOD_EXP`) with a linear extension below `Q = 0.1`
  joined continuously at the knee. Replaced the made-up
  3-coefficient formula with the actual piecewise definition,
  added the `JOD_A` (`≈ 0.0440`) and `JOD_EXP` (`≈ 0.9302`)
  numeric anchors, and noted that the struct's `jod_b` is unused
  (the formula has no separate `b` coefficient). Tick 288.
- `MaskingParams` struct-level docstring listed `MASK_P / MASK_Q
  / MASK_C / XCM_3X3` but omitted `D_MAX` (the clamp ceiling,
  separate from `MASK_C`'s phase-blur post-scale). Per-field
  docs claimed cvvdp `q` and `epsilon`/`k` semantics that don't
  match production: `MASK_Q` is `[f32; 3]` per-channel (the
  struct's scalar `q` is shape-mismatched) and there is no
  `MASK_K` / saturation-epsilon constant in `kernels::masking`
  (closest are `MASK_C` and `D_MAX`, both log10-encoded and
  semantically different). Updated to document the shape
  mismatch explicitly, flag `k` as reserved-no-current-mapping
  scaffolding, and note a future JSON-loader path would need to
  widen `q` to `[f32; 3]` and split `k`. Also expanded
  `CvvdpParams::PLACEHOLDER`'s "inlined consts" list to the full
  set (`IMAGE_INT`, `PER_CH_W`, `BASEBAND_W` in `kernels::pool`;
  `D_MAX`, `CH_GAIN`, `PU_BLUR_KERNEL_1D`, `PU_PADSIZE` in
  `kernels::masking`) so the docstring matches what
  `kernels::pool` and `kernels::masking` actually export.
  Tick 289.

#### cvvdp-gpu (doctests)

- **Doctest cpu-only feature combo also fixed** — tick 283's
  cuda+wgpu cascade left cpu-only `--features cpu` builds broken
  (3 GPU doctests fail compile since neither cuda nor wgpu is on).
  Added a third `# #[cfg(all(feature = "cpu", not(any(feature =
  "cuda", feature = "wgpu"))))] # type Backend = cubecl::cpu::CpuRuntime;`
  fallback so the cuda doctests now compile under cpu-only too.
  No-op for the rendered docs (the canonical cuda branch still
  renders). All 6 doctests now pass under: cuda-only, wgpu-only,
  cpu-only, and default (cuda+wgpu+cpu).
- **CI doctest pass under `--no-default-features --features wgpu`
  was broken** for the 5 GPU/cpu doctests added between ticks 225
  and 244. They hardcoded `cubecl::cuda::CudaRuntime` (3 doctests)
  or `cubecl::cpu::CpuRuntime` (2 doctests), which don't exist
  under the CI doctest invocation
  (`cargo test --workspace --no-default-features --features wgpu
  --doc --release` per `.github/workflows/ci.yml:173`).
  Each doctest now wraps its body in feature-gated cfg attrs:
  - CUDA doctests: `# #[cfg(feature = "cuda")] type Backend = ...;`
    + `# #[cfg(all(feature = "wgpu", not(feature = "cuda")))] # type
    Backend = cubecl::wgpu::WgpuRuntime;` (wgpu fallback). Rendered
    docs still show the canonical cuda form; the wgpu fallback is
    hidden via `# ` prefix but compiles when cuda isn't on.
  - CPU doctests: wrap the entire body in `# #[cfg(feature = "cpu")]
    { ... # }` so non-cpu builds skip the body. Rendered docs are
    unchanged.
  No regression on default-features builds (all 6 doctests still
  green); CI's wgpu-only doctest pass now compiles all 6.
  The CI was masked from this regression because the bug landed
  on `feat/cvvdp-gpu-scaffold` and CI triggers only on master/PR.

#### cvvdp-gpu (tests)

- `error_display_messages_are_actionable` — pins the user-facing
  `Display` strings for all 4 `cvvdp_gpu::Error` variants. Tests
  content (variant name hint, the actionable next step) rather
  than exact strings, so future context additions still pass.
  Pre-tick-282 a rename of the `Display` impl would have silently
  degraded the user experience for callers who `?`-bubble cvvdp
  errors through `anyhow::Error::to_string()` / `panic!`
  propagation.

#### cvvdp-gpu (tests + examples)

- Collapse the last two-line `synth_pair_odd_dim_ref + apply_offset_dist`
  pairs onto `common::synth_pair_odd_dim_with_offset_dist`:
  `tests/cpu_backend.rs::synth_pair` (was 3 lines) and
  `examples/manifest_parity_probe.rs::synth_odd_pair` (was 3 lines).
  Each collapses to a single tuple-returning call. Drops the
  now-unused `synth_pair_odd_dim_ref` import from
  `manifest_parity_probe.rs` since `synth_odd_pair` was its only
  consumer.

#### cvvdp-gpu (tests)

- New `common::synth_pair_odd_dim_with_offset_dist(w, h) -> (ref, dist)`
  pairs `synth_pair_odd_dim_ref` with `apply_offset_dist` for the
  73×91 pycvvdp golden's construction
  (`bench_12mp_cuda.py::synth_pair_odd_dim`). Replaces 7-of-8
  inline `synth_pair_odd_dim_ref + apply_offset_dist` pairs in
  `tests/pipeline_color.rs` with a single-line tuple destructure.
  Also migrated the two `synth_pair_ref + apply_offset_dist`
  pairs in `pipeline_color.rs` (12mp tests) onto the existing
  `synth_pair_with_offset_dist`. The cpu_backend `synth_pair`
  wrapper and the warm-ref idempotence test (dist_a + dist_b)
  intentionally keep the two-line form for clarity.

#### cvvdp-gpu (tests + examples)

- New `common::apply_offset_dist(ref_bytes: &[u8]) -> Vec<u8>`
  standalone helper for the canonical `(-8, -4, +12)` saturating
  offset distortion. Tick 278's `synth_pair_with_offset_dist`
  paired this with the regular `synth_pair_ref`; tick 279
  extracts the dist half so callers can pair it with either ref
  variant (regular or `synth_pair_odd_dim_ref`). Migrated 12 more
  inline copies:
  - 10 sites in `tests/pipeline_color.rs` (all of which use
    `synth_pair_odd_dim_ref` + the offset dist for stage-probe
    tests at 32×32 odd dims)
  - 1 in `tests/cpu_backend.rs::synth_pair`
  - 1 in `examples/manifest_parity_probe.rs::synth_odd_pair`
  `synth_pair_with_offset_dist` itself now delegates to
  `apply_offset_dist`. Total dedup across ticks 278-279: 16 sites
  consolidated.

#### cvvdp-gpu (tests + examples + benches)

- New `common::synth_pair_with_offset_dist(w, h) -> (ref, dist)`
  helper bundles the canonical `synth_pair_ref` + `(-8, -4, +12)`
  saturating offset dist that 16 sites across the crate were
  building inline:
  - `benches/score.rs::synth_pair`
  - `examples/time_12mp.rs::synth_pair`
  - `examples/time_size_sweep.rs::synth_pair`
  - `examples/manifest_parity_probe.rs::synth_pair_12mp`
  All four now collapse to a single `synth_pair_with_offset_dist`
  call; the per-site `synth_pair` wrappers stay since they pass
  `u32 → usize`. `tests/pipeline_color.rs` 12mp tests already
  use the equivalent inline pattern (untouched — each test
  decides whether to pre-cache the ref or inline). The cpu_backend
  synth_pair keeps its odd_dim ref version with a clarifying
  comment.

#### cvvdp-gpu (examples)

- `examples/time_12mp.rs` + `examples/time_size_sweep.rs` now
  consume `tests/common` via `#[path]` (matches ticks 275-276's
  pattern). Drops 4 duplicates total — 2× Backend cascade
  (~6 lines each) + 2× hand-inlined `synth_pair` (~20 lines each).
  Both examples kept their tiny wrapper that combines `synth_pair_ref`
  with the same saturating-sub/saturating-add dist construction —
  identical pattern to time_12mp.rs's bench in benches/score.rs.
  No behaviour change.

#### cvvdp-gpu (examples)

- `examples/manifest_parity_probe.rs` now consumes
  `tests/common/mod.rs` via
  `#[path = "../tests/common/mod.rs"] mod common;` (same shape as
  tick 275's bench dedup). Drops the example's local
  `synth_pair_ref`, `synth_pair_odd_dim_ref` (via `synth_odd_pair`),
  and `pycvvdp_synth_golden_jod` clones in favour of the common
  helpers. Closes ticks 266 (last hand-mirrored goldens in
  examples) by leveraging the bench-side discovery (tick 275) that
  examples + benches can both reach `tests/common` via `#[path]`.
  Probe still passes all 6 fixtures at ≤ 0.005 JOD; max
  measured |d_gpu| = 0.000172 (synth_256x256_blur3x1).
- Drop the now-redundant
  `#![allow(clippy::excessive_precision)]` since the goldens are
  no longer inline float literals.

#### cvvdp-gpu (benches)

- `benches/score.rs` now consumes `tests/common/mod.rs` via
  `#[path = "../tests/common/mod.rs"] mod common;`. Drops the
  bench's local `Backend` cascade + `load_rgb_bytes` + `synth_pair`
  in favour of `common::Backend`, `common::load_rgb_bytes`, and
  `common::synth_pair_ref` (with the bench's per-fixture dist
  builder inlined). Closes the last synth-pattern duplication
  outside the example file. The bench's `load_rgb_bytes(path)`
  wrapper preserves the 256×256-assert contract by passing the
  bench's `W_256` / `H_256` constants through.

#### cvvdp-gpu (docs)

- `lib.rs` Status section now cross-references the warm-state
  invalidation regression tests
  (`warm_state_invalidates_after_each_documented_dispatcher`,
  `set_reference_does_not_invalidate_warm_state`,
  `gauss_chain_helpers_do_not_invalidate_warm_state`) and points
  at `docs/PORT_STATUS.md`'s "Resolved ticks 236-249" audit-history
  entry. Surfaces the contract work in the crate-root docs that
  docs.rs renders first.

#### cvvdp-gpu (docs)

- `MaskingParams`, `PoolingParams`, `JodParams` docstrings now
  state they're unused scaffolding and cross-reference
  `CvvdpParams::PLACEHOLDER`. Previously only `CsfParams` had this
  note; the other three sub-bundles left it implicit. Same shape
  as tick 264's `Cvvdp::new` silent-ignored-fields docs — protects
  users who'd otherwise expect varying `p` / `beta_spatial` /
  `jod_a` to change the metric output.

#### cvvdp-gpu (tests)

- Migrated the last 2 inline `Backend` cascade copies onto
  `common::Backend` (tick 270 covered the 6 file-root cases):
  - `pool_scalar.rs::mod gpu` → `use super::common::Backend;`
    (paired with a new file-root `#[path = "common/mod.rs"] mod common;`
    gated on the same `any(cuda, wgpu, hip)` as the gpu submodule)
  - `shadow_jod.rs::shadow_jod_gpu_runs_and_is_close_to_manifest_on_corpus`
    → `use common::Backend;` at the function top (already had
    `mod common` at file root since tick 253). Drops the 4-line
    cascade inside the fn body.
  Backend cascade dedup now complete — 0 inline copies remain
  anywhere in `tests/`.
- New `common::Backend` type alias dedups the "first available GPU
  backend" cascade (`cuda` → `wgpu` → `hip`) that was hand-mirrored
  across 6 test files at file root: `color_kernel.rs`, `csf_kernel.rs`,
  `masking_kernel.rs`, `pyramid_kernel.rs`, `pipeline_color.rs`,
  `pipeline_score.rs`. Each now uses `use common::Backend;` after a
  `#[path = "common/mod.rs"] mod common;` at the file top. The
  alias is cfg-gated on the same `any(cuda, wgpu, hip)` so cpu-only
  builds (and the cpu_backend test's `CpuRuntime` alias) are
  unaffected. The inline-in-fn / inline-in-mod copies in
  `shadow_jod.rs` and `pool_scalar.rs` stay local for now (different
  scope; needs a `use super::common::Backend;` migration that's a
  separate chunk). 6 files × 6 lines of cascade = 36 lines deleted;
  all 63 tests across 6 binaries still green.

#### cvvdp-gpu (cleanup)

- `cargo fmt -p cvvdp-gpu` run across the crate. Multiple test
  files + examples had drift after the recent dedup refactors
  (mostly Cvvdp::<Backend>::new(...) and predict_jod_still_3ch()
  call sites that fit on one line post-helper-extraction).
  Alphabetised the masking_kernel.rs `use` import list while
  there. No behavioural changes — 6 masking_kernel + 31
  pipeline_color + 14 pipeline_score + 2 shadow_jod tests still
  green.

#### cvvdp-gpu (tests)

- `common::load_rgb_bytes` signature widened from `&PathBuf` to
  `&Path`. `&PathBuf` callers still work via auto-deref; `&Path`
  callers (e.g. `path.parent().unwrap()` returning `&Path`)
  newly work without an extra `PathBuf::from(...)`. Standard Rust
  API hygiene per the
  [pathbuf-vs-path nursery clippy](https://rust-lang.github.io/rust-clippy/master/index.html#ptr_arg).
- `tests/common::load_rgb_bytes` extracted. The 10-line PNG/JPEG
  decode + dimension-assert helper was hand-mirrored across
  `tests/pipeline_score.rs` and `tests/shadow_jod.rs`. Both call
  sites now use the common helper; the `image::ImageReader` +
  `std::path::PathBuf` imports drop out of both files.

#### cvvdp-gpu (examples)

- `examples/manifest_parity_probe.rs` no longer hand-mirrors the
  6 pycvvdp golden JODs as `golden: 9.xxx` fields in its fixture
  table. Loads from `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`
  at runtime via a local `pycvvdp_synth_golden_jod(name)` helper
  that mirrors `tests/common/mod.rs`'s identical function (examples
  can't easily import test modules, so the lookup logic is inlined).
  Closes the last hand-mirrored golden in the repo; a future
  `build_goldens.py` rerun propagates to the example with zero
  hand-editing. Probe verified end-to-end: all 6 fixtures pass at
  ≤ 0.005 JOD.

#### cvvdp-gpu (tests)

- `tests/cpu_backend.rs::compute_dkl_jod_host_pool_matches_pycvvdp_at_73x91_odd_on_cpu_backend`
  no longer hardcodes the `9.390370` pycvvdp golden as a `const`.
  Loads from `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`
  via the `common::pycvvdp_synth_golden_jod("synth_73x91_odd")`
  helper that the pipeline_color sibling tests already use. Last
  hardcoded synth golden in tests/; a build_goldens.py rerun now
  propagates without any hand-edited mirrors anywhere in tests/.
  The 6 hand-mirrored copies in examples/manifest_parity_probe.rs
  stay (examples can't easily import test modules).

#### cvvdp-gpu (docs)

- `Cvvdp::new` / `Cvvdp::new_with_geometry` docstrings now spell
  out that **only `params.display` is consumed** — the
  `csf`/`masking`/`pooling`/`jod` sub-bundles of `CvvdpParams` are
  silently ignored because the per-stage cvvdp v0.5.4 numbers are
  inlined as `const`s in the kernels module. `CvvdpParams::PLACEHOLDER`
  already documented this on the struct side; the constructor docs
  now point at it. Same shape as tick 243's silent-ignored-`ppd`
  docs surfacing — protects users who'd otherwise pass a custom
  `CvvdpParams` expecting the masking/pooling exponents to matter.

#### cvvdp-gpu (benches)

- `benches/score.rs` adds `gpu_compute_dkl_jod_with_warm_ref` to
  both bench groups (`bench_resolution` synth + `bench_at_quality`
  corpus). Captures the warm-ref batch-scoring fast path
  empirically — the lib.rs Status section quotes ~1.8× per-DIST
  throughput at 12 MP vs cold, but until now there was no
  bench that produced numbers for that path. Provides a
  regression-net for the warm-state work in ticks 236-240
  (warm-state invalidation + persistent dest Vecs + scratch
  buffers).

#### cvvdp-gpu (docs)

- Refreshed stale "0.40 JOD GPU-vs-host drift" / "where the
  q=1 JOD drift lives" docstrings on three corpus-scale parity
  tests in `tests/pipeline_score.rs`:
  - `compute_dkl_weber_pyramid_matches_host_on_corpus_256x256`
  - `compute_dkl_t_p_bands_matches_host_on_corpus_256x256`
  - `compute_dkl_d_bands_matches_host_on_corpus_256x256`
  These were stage-isolation probes during the tick 175 pyramid
  fix; ticks 175/204/206 closed every drift to f32 noise. The
  tests still serve a useful role (per-stage bit-stability pins
  at corpus scale) but the docstrings claimed an in-flight
  investigation that's been done for ~80 ticks.

#### cvvdp-gpu (tests + docs)

- `score_with_reference_matches_score` now iterates the full
  `common::v1_corpus_qs()` set (6 q-levels: 1, 5, 20, 45, 70, 90)
  instead of the hand-picked `&[1u32, 20, 90]` subset (3 levels).
  Doubles parity coverage on the cached-reference contract at
  the cost of ~6 extra corpus loads. Also updated the leading
  comment which still claimed the path was "currently a host-
  scalar pass-through" — that switched to GPU in tick 213.

#### cvvdp-gpu (tests)

- `tests/cpu_backend.rs::synth_pair` now uses
  `common::synth_pair_odd_dim_ref` instead of its own inline copy.
  Last unmigrated odd-dim synth site outside the example file;
  `pipeline_color.rs` + `cpu_backend.rs` both now go through the
  common helper. All 4 cpu_backend tests still green (incl. the
  73×91 pycvvdp parity test at 0.000001 JOD diff).
- New `common::synth_pair_odd_dim_ref(w, h)` helper for the
  alternate odd-dim synth pattern (`(x * 8) % 256` / `(y * 8) % 256`
  / `((x + y) * 4) % 256`). Migrated all 10 hand-inlined sites in
  `tests/pipeline_color.rs` onto it. Companion to tick 255-258's
  `synth_pair_ref` dedup. Bit-stable parity preserved on all 31
  pipeline_color tests (including 73×91 odd-dim cold + warm).
- Final 6 hand-inlined synth_pair_ref sites in
  `tests/pipeline_color.rs` migrated onto `common::synth_pair_ref`
  (stage-probe helpers for chroma_shift: `compute_dkl_planes`,
  `compute_dkl_t_p_bands`, `compute_dkl_weber_pyramid`,
  `spatial_pool`, `compute_dkl_d_bands`, plus an `_at_chroma_shift_sentinels`
  helper). 14 of 14 callers now consolidated; the inline modular-
  arithmetic ref construction no longer appears anywhere except the
  helper definition itself. Bit-stable parity preserved across all
  31 pipeline_color tests.
- Migrated the `blur3x1`, `blur1x3`, and `noise` 256×256 parity
  tests off the hand-inlined `synth_pair_ref` construction onto
  `common::synth_pair_ref`. 7 of 14 inlined sites in
  `tests/pipeline_color.rs` now consolidated. Bit-stable parity
  preserved on all three (still ≤ 0.005 JOD vs pycvvdp goldens).
- Migrated the two `compute_dkl_jod_*_pycvvdp_at_256x256_chroma_shift`
  tests (cold + warm-ref) off the inlined synth-pair construction
  onto `common::synth_pair_ref`. Same shape as tick 255's 12mp
  migration. 4 of 14 inlined sites in `tests/pipeline_color.rs`
  now use the helper. Bit-stable parity preserved (chroma_shift
  diff vs pycvvdp golden remains 0.0000).
- New `common::synth_pair_ref(w, h) -> Vec<u8>` helper builds the
  canonical synthetic-fixture reference image (the
  `(x * 17 + y * 5) % 251`-style modular pattern matching pycvvdp's
  `synth_pair_ref` in `bench_12mp_cuda.py`). Migrated the two
  largest fixture-using tests (`compute_dkl_jod_matches_pycvvdp_at_12mp_synth`
  and `compute_dkl_jod_with_warm_ref_matches_pycvvdp_at_12mp_synth`)
  off their hand-inlined copies; bit-stable output (JOD 9.4580
  matches pycvvdp golden to 0.0000 on both). The pattern was
  duplicated across 12 more sites in `pipeline_color.rs`; future
  ticks can migrate them opportunistically.

#### cvvdp-gpu (tests)

- New `common::v1_corpus_qs()` helper derives the q-list from the
  canonical `scripts/cvvdp_goldens/v1_corpus_jods.json` itself.
  Replaces the hand-mirrored `&[1, 5, 20, 45, 70, 90]` constant
  duplicated across 5 callers (3 in `pipeline_score.rs` + 2 in
  `shadow_jod.rs`). A goldens regen that adds (e.g.) q=2 to the
  manifest now propagates to every parity test without hand-editing.
  Same shape as tick 253's `v1_corpus_jod_golden(q)` dedup.
- `tests/shadow_jod.rs` no longer hardcodes the pycvvdp manifest
  JOD constants. Both tests now load via `common::v1_corpus_jod_golden(q)`
  from the canonical `scripts/cvvdp_goldens/v1_corpus_jods.json`
  (which the existing `tests/pipeline_score.rs::cvvdp_score_matches_v1_manifest`
  already used). Previously the same six `(q, expected_jod)` pairs
  were duplicated across three test files; a `build_goldens.py`
  rerun + JSON bump would have silently skipped the two shadow_jod
  copies until manual sync. Tick 253 dedup.

#### cvvdp-gpu (docs)

- `benches/score.rs` stale-comment cleanup:
  - `bench_score_q1` no longer claims the GPU drifts 0.4 JOD at q=1;
    that drift was closed to 0.0000 in ticks 204/206 (chroma_shift
    CSF + gausspyr_reduce parity-bug fixes). Comment now correctly
    states the historical drift and points at the regression-pin test.
  - `bench_at_quality`'s host_scalar group comment no longer says
    `Cvvdp::score` routes through it; tick 213 switched `score` to
    GPU `compute_dkl_jod`. Comment now correctly states the host
    path is a faster-to-debug reference exposed via
    `host_scalar::predict_jod_still_3ch`.

#### cvvdp-gpu (tests)

- `gauss_chain_helpers_do_not_invalidate_warm_state` — pins the
  inverse of `warm_state_invalidates_after_each_documented_dispatcher`:
  `compute_dkl_planes` and `compute_dkl_gauss_pyramid` write only
  to `gauss_ref` (per-call scratch, not warm state), so they MUST
  preserve the cached scalar. A future refactor that made either
  helper additionally emit bands into `bands_ref` (matching the
  symmetric `compute_dkl_weber_pyramid` interface) would need to
  invalidate warm state — this test would surface that. Sibling
  to `set_reference_does_not_invalidate_warm_state` (tick 238).

#### cvvdp-gpu (tests + docs)

- `set_reference_replaces_prior_cache` — pins the implicit
  cache-replace semantics of `Cvvdp::set_reference`. Test calls
  `set_reference(ref_a)`, then `set_reference(ref_b)`, then
  `score_with_reference(dist)`; expects the result to match
  `score(ref_b, dist)`, not `score(ref_a, dist)`. The contract
  was the natural cache-shape callers expect but had been
  documented-by-convention only — a refactor that no-op'd the
  second `set_reference` call wouldn't have surfaced in CI.
  `set_reference`'s docstring now explicitly states the replace
  semantics and cross-references this test + the tick-238
  non-invalidation test.

#### cvvdp-gpu

- **`compute_dkl_jod_with_warm_ref` / `compute_dkl_jod_host_pool_with_warm_ref`
  now check dim mismatch before `NoWarmReference`.** When a caller
  has BOTH a wrong-size dist buffer AND no warm state, the wrong-size
  buffer is the more actionable error — they need to fix the buffer
  regardless of whether warm state is set. Pre-tick-248 ordering
  reported `NoWarmReference` first, masking the dim mismatch. New
  test `compute_dkl_jod_with_warm_ref_reports_dim_mismatch_before_no_warm`
  pins the order so a future regression surfaces in CI.
- **Debug-assert `compute_dkl_csf_weighted_bands` weight-band-count
  matches construction-time `n_levels`.** The per-level
  `weight_band_kernel` loop reads `weight_idx = k * N_CHANNELS + c`
  into the host-flattened weights buffer for `k = 0..n_levels`. If
  the caller's `ppd` produces fewer band frequencies than the
  construction-time `n_levels`, the higher-k kernel launches read
  past `flat_weights.len()` and OOB the GPU buffer. Now
  `debug_assert_eq!(weights_per_level.len(), n_levels)` catches
  the precondition violation in debug builds with a message that
  spells out the fix: reconstruct against the new geometry. Release
  preserves silent OOB behavior since this is a documented
  precondition (per tick 246's docstring update). Tick 247 pairs
  the docstring warning with an enforceable check.
- **Revert misplaced tick-243 debug_assert + tick-245 docstring
  on `compute_dkl_csf_weighted_bands`.** Unlike the JOD-path
  helpers, this function genuinely consumes the caller-passed `ppd`
  (via `precomputed_band_weights(ppd, w, h, l_bkg)` which uses
  `band_frequencies(ppd, ...)` to compute per-band rho). The
  tick-243 audit assumed every public `ppd` parameter was a
  silent-ignored relic — true for the 6 JOD-path helpers but
  wrong for this Laplacian + per-band-weight helper.
  - Removed the `debug_assert_ppd_matches_geometry` call at entry
  - Docstring now correctly states `ppd` is consumed and warns
    that the caller must keep `band_frequencies(ppd, w, h).len()`
    consistent with the construction-time `n_levels` (otherwise
    the weights buffer mismatches the per-level kernel launches)
  No tests changed behaviour — all 30 pipeline_color + 12
  pipeline_score tests still green; existing call sites pass
  matching ppd so the spurious assert never fired in CI.

#### cvvdp-gpu (docs)

- Document the silent-ignored `ppd` argument on the 6 public
  methods that take it (`compute_dkl_jod`, `compute_dkl_d_bands`,
  `compute_dkl_t_p_bands`, `compute_dkl_jod_host_pool`,
  `compute_dkl_jod_host_pool_with_warm_ref`,
  `compute_dkl_jod_with_warm_ref`, plus
  `compute_dkl_csf_weighted_bands`). Each docstring now states that
  `ppd` is silently ignored — the GPU CSF LUT is pre-uploaded
  against the construction-time geometry — and points readers at
  `Cvvdp::new_with_geometry` for a different display geometry. Pairs
  with the tick-243 `debug_assert_ppd_matches_geometry` safety net
  by making the contract explicit in the docs that users read first.

#### cvvdp-gpu (tests)

- `debug_assert_fires_when_ppd_mismatches_geometry` — pins the
  tick-243 ppd-mismatch debug_assert. Builds Cvvdp with the
  default STANDARD_4K geometry (75.4 PPD), then calls
  `compute_dkl_jod` with the phone-shaped 110-PPD value; expects
  panic via `#[should_panic(expected = "ppd=")]`. Gated on
  `#[cfg(debug_assertions)]` so release builds skip the test
  (the assert compiles out there). A future refactor that drops
  the safety net would silently regress without this pin.

#### cvvdp-gpu (debug)

- **Surface silent-ignored `ppd` mismatches in debug builds.**
  6 public methods take a `ppd: f32` parameter that the
  implementation **silently ignores** — `logs_row` is pre-uploaded
  at construction time against `self.geometry.pixels_per_degree()`,
  so a caller who built `Cvvdp::new(client, w, h, p)` with the
  default `STANDARD_4K` (75.4 PPD) then called
  `compute_dkl_jod(ref, dist, phone_ppd)` (110 PPD) would get
  results scored against 75.4 PPD with no warning. Pre-tick-243
  there was no surfaced sanity check.
  - New `Cvvdp::debug_assert_ppd_matches_geometry(ppd)` helper:
    `debug_assert!((ppd - self.geometry.pixels_per_degree()).abs() < 1e-3)`
  - Wired into the 6 affected entries: `compute_dkl_jod`,
    `compute_dkl_d_bands`, `compute_dkl_t_p_bands`,
    `compute_dkl_jod_host_pool`,
    `compute_dkl_jod_host_pool_with_warm_ref`,
    `compute_dkl_jod_with_warm_ref`, and
    `compute_dkl_csf_weighted_bands`.
  - Release builds preserve silent-ignore (no public-API change);
    the parameter remains in the signatures for source compatibility.
    All 30 pipeline_color + 11 pipeline_score + 4 cpu_backend tests
    green — all existing call sites pass ppd consistent with geometry.

#### cvvdp-gpu (docs)

- Stale pre-tick-175 warm-ref throughput numbers updated across
  `warm_reference`, `set_reference`, the `CachedReference`
  struct doc, the `compute_dkl_jod_with_warm_ref` doctest, and
  `lib.rs`. All sites referenced `1.75× / 36.1 → 20.6 ns/px /
  42.9% saved` from tick 170; the tick-175 ceil-div correctness
  fix raised absolutes to ~62 / ~34 ns/px while keeping a similar
  ratio (~1.8×). Docstrings now cite `~1.8×` and defer to
  `lib.rs` "How we compare to the canonical reference" for the
  source-of-truth measurements. The "Resolved tick 170" entry in
  `PORT_STATUS.md` keeps the original numbers (accurate as-of-tick-170)
  plus a tick-175 update note explaining why the post-fix path is
  numerically slower (correct output vs broken pyramid).

#### cvvdp-gpu (tests)

- `invalid_image_size_surfaces_on_too_small_dims` — pins the
  `Error::InvalidImageSize` construction-time guard on `Cvvdp::new`
  and `Cvvdp::new_with_geometry`. Tests 6 sub-threshold cases
  (7×8, 8×7, 7×7, 4×4, 0×0, plus a `new_with_geometry` case)
  plus the 8×8 boundary success path. Pre-tick-241 a refactor
  that swapped the `width < PYRAMID_MIN_DIM * 2` check for
  `width < PYRAMID_MIN_DIM` (accepting 4×4 with no usable
  pyramid) would not have surfaced in CI.
- `dimension_mismatch_surfaces_on_wrong_size_inputs` — pins the
  `Error::DimensionMismatch` contract on every public entry that
  validates buffer length: `Cvvdp::score` (both arms),
  `set_reference`, `score_with_reference`, `warm_reference`, and
  `compute_dkl_jod_with_warm_ref`. Each is called with a buffer
  sized for `(w/2) × (h/2)` against a Cvvdp configured for `w × h`;
  the test asserts both that `DimensionMismatch` fires AND that
  the `(expected, got)` fields carry the right byte counts.
  Closes a real zero-coverage gap: a refactor that swapped `!=`
  for `<` (silently accepting smaller buffers and reading past
  `srgb.len()`) would not have surfaced in CI before this.
- `compute_dkl_jod_with_warm_ref_matches_pycvvdp_at_12mp_synth` —
  large-image warm-ref pycvvdp parity. Completes the warm-ref vs
  pycvvdp coverage grid: small same-parity (chroma_shift, tick 222)
  + small mixed-parity (73×91, tick 226) + large same-parity here
  (12 MP, full ~9-band pyramid). Exercises warm-state restoration
  across every weber_scratch level. Measured diff: 0.0000 JOD.
  Runtime ~6s on RTX-class CUDA — within parity-test budget,
  matches the existing cold-path 12mp test cadence.

#### cvvdp-gpu (performance)

- **Persistent `log_l_bkg_{ref,dis}_dests: Vec<Handle>` slots** —
  `Cvvdp::new_with_geometry` now pre-builds the destination-handle
  Vecs that `_dispatch_ref_weber_pyramid_only` and
  `_dispatch_dist_weber_pyramid_only` previously rebuilt per call
  via `weber_scratch.iter().map(|s| s.log_l_bkg.clone()).collect()`.
  Each dispatch now `mem::take`s the pre-built Vec, passes it as
  `&[Handle]` to `_dispatch_weber_pyramid_gpu`, then moves it back —
  zero heap allocation per JOD-side, replacing `(1 Vec alloc) +
  (n_levels - 1) handle ref-bumps` per call. Cold JOD pays this
  twice (REF + DIST), warm-ref JOD pays it once (DIST only). All
  30 pipeline_color + 10 pipeline_score + 4 cpu_backend tests
  green; manifest parity untouched.
- **Persistent `src_u32_scratch` host buffer** — `Cvvdp::new` pre-
  allocates a `Vec<u32>` of length `width * height * 3` once at
  construction time; `_dispatch_dkl_planes_gpu` now fills it in
  place via `iter_mut().zip(srgb.iter())` instead of allocating
  a fresh `Vec<u32>` per call via `.iter().map(|b| b as u32).collect()`.
  Removes ~`width × height × 12` bytes of host allocator round-trip
  per JOD-side dispatch — at 12 MP that's ~144 MB per side, paid
  twice per cold JOD and once per warm-ref DIST. The GPU buffer
  upload (`create_from_slice` of the scratch's bytes) still
  happens per call since cubecl 0.10 has no public "write into
  existing handle" API. All 27 pipeline_color + 9 pipeline_score
  + 4 cpu_backend tests green; manifest parity untouched.

#### cvvdp-gpu (docs)

- `Cvvdp::compute_dkl_jod_with_warm_ref` now has a `no_run` doctest
  example showing the canonical GPU batch-scoring pattern
  (warm REF once, score N DIST candidates against it). Mirrors
  the existing cpu-runtime example on
  [`Cvvdp::compute_dkl_jod_host_pool_with_warm_ref`] — completes
  the doctest coverage on the warm-ref API across both GPU and
  cpu-runtime paths.

#### cvvdp-gpu (cleanup)

- Cleared remaining 7 clippy warnings under `--all-targets`:
  - `tests/common/mod.rs`: collapsed nested `if let Ok(hex) = ...
    { if hex == sha256 { return ... } }` into a let-chain
    (`if let ... && hex == sha256`).
  - `tests/pipeline_color.rs`: dropped a redundant `(wu * hu) as usize`
    cast (wu/hu were already `usize`); added module-level
    `#![allow(clippy::needless_range_loop)]` for the 3 per-band
    `for k in 0..n_bands` loops (k indexes ref_tp[k] / d_bands[k]
    plus side metadata — enumerate is a wash). Mirrors the library's
    same allow.
  - `tests/cpu_backend.rs` + `examples/manifest_parity_probe.rs`:
    `#![allow(clippy::excessive_precision)]` for the pycvvdp
    golden literals — same rationale as the library-level allow:
    the 7-digit decimal documents the source value verbatim even
    though LLVM rounds at f32.
  Net: `cargo clippy -p cvvdp-gpu --features cuda --all-targets`
  is warning-clean. All 27 pipeline_color + 9 pipeline_score + 4
  cpu_backend tests still green.
- Fixed 8 clippy lints surfaced under MSRV 1.93:
  - 6× `manual_div_ceil` in `pipeline.rs` (`(x + 1) / 2` →
    `x.div_ceil(2)` in pyramid-level allocators)
  - 2× `manual_is_multiple_of` in `kernels/pyramid.rs` (`sh % 2 == 0`
    → `sh.is_multiple_of(2)` in `gausspyr_reduce_scalar`)
  Semantically equivalent rewrites — all 78 cuda + 4 cpu tests
  green; manifest parity untouched. `cargo clippy -p cvvdp-gpu`
  is warning-clean.

#### cvvdp-gpu (docs)

- `Cvvdp::score_with_reference` now has a `no_run` doctest example
  showing the canonical `set_reference` + `score_with_reference`
  batch pattern (one stashed REF, many DIST). Pairs with the
  `Cvvdp::score` doctest from tick 225 to cover both top-level
  public scoring entry points. Also notes the
  `Error::NoCachedReference` precondition explicitly in the
  doc body.
- Renamed `examples/chroma_shift_drift_probe.rs` →
  `examples/manifest_parity_probe.rs`. The file started life
  (tick 191) as a single-fixture probe while investigating the
  chroma_shift drift, but tick 210 expanded it to walk all 6
  manifest fixtures — the old name no longer reflected what it
  did. Internal doc header + run-with command updated; a note at
  the top of `docs/CHROMA_DRIFT_INVESTIGATION.md` flags the rename
  so historical references in that file (which describe past
  measurements) stay accurate. Active "See ..." pointer also
  updated to the new name. Probe verified end-to-end: all 6
  fixtures pass at ≤ 0.005 JOD vs pycvvdp goldens, max
  measured |d_gpu| = 0.000186 JOD on synth_256x256_blur1x3.

#### cvvdp-gpu (performance)

- `compute_dkl_d_bands` host readback init no longer pre-allocates
  `vec![0.0; n_px] × 3` per pyramid level only to immediately
  overwrite each entry with `f32::from_bytes(&bytes).to_vec()`.
  Now uses empty `Vec::new()` slots — matches `compute_dkl_gauss_pyramid`'s
  readback shape and drops `~3 × n_levels × n_px` floats of wasted
  host zero-fill per call. (`compute_dkl_d_bands` is a parity-test
  helper; production JOD path is unaffected since it pools on-GPU.)
- **Persistent `partials_h` atomic-pool buffer** — `Cvvdp::new`
  now allocates a single `n_levels × N_CHANNELS` partials buffer
  (≤ 144 bytes at MAX_LEVELS=9) and `_pool_and_finalize_jod` zero-
  fills it via `fill_f32_kernel` per call instead of allocating
  a fresh GPU buffer + uploading host zeros every JOD call.
  Removes one `create_from_slice` host alloc + Host→GPU copy per
  call from the JOD hot path; pattern mirrors the tick-168
  `baseband_log_l_bkg` migration. All 27 pipeline_color + 9
  pipeline_score + 8 pool_scalar tests green on CUDA, including
  manifest parity (`compute_dkl_jod_on_v1_manifest_corpus` at ≤ 0.005
  JOD) and the GPU-pool-vs-host-pool sentinel
  (`compute_dkl_jod_host_pool_matches_compute_dkl_jod`).

#### cvvdp-gpu (tests)

- `compute_dkl_jod_with_warm_ref_matches_pycvvdp_at_73x91_odd` —
  direct warm-ref pycvvdp parity on the mixed-parity 73×91 fixture.
  Pairs with the chroma_shift warm-ref test from tick 222: both pin
  the warm-state restoration path against canonical pycvvdp, but
  73×91 specifically exercises the tick-206 gausspyr_reduce
  parity-bug fix on REF (mixed-parity reduce levels 6×5 → 3×3 and
  46×37 → 23×19). Measured diff: 0.0000 JOD. Closes a transitivity
  gap: prior warm-ref pycvvdp coverage was same-parity only.

#### Workspace

- Pinned multi-tick task in `CLAUDE.md`: compute CVVDP scores for
  all zensim training data sets via vast.ai docker images, output
  as parquet sidecars with implementation-distinguished column
  names (e.g. `cvvdp_pycvvdp_v054`, `cvvdp_imazen_v0_0_1`). Survives
  context compaction; every `/loop` tick re-reads it.

#### zen-metrics-cli

- New `score-pairs` subcommand (feature-gated on `sweep`):
  consumes the pairs TSV that `sweep --pairs-tsv` produces and
  emits a parquet sidecar with the metric's versioned column name
  (e.g. `cvvdp_imazen_v0_0_1` for cvvdp). Schema matches
  `crates/cvvdp-gpu/docs/CVVDP_SIDECAR_SCHEMA.md` exactly:
  `image_path string`, `codec string`, `q int64`,
  `knob_tuple_json string`, `<metric> float64`. Zstd compression.
  Symmetric with `scripts/sweep/pycvvdp_worker.py score-pairs`.
  Initial n=4 sentinel: cvvdp-gpu vs pycvvdp parity within 0.03 JOD
  on q50/q90 zenjpeg-encoded 64×64 noise images.

#### zen-metrics-cli (sweep)

- `sweep` subcommand learns two new flags that pair off for
  external-scorer workflows (e.g. pycvvdp):
  - `--distorted-out-dir <DIR>`: every successfully-decoded cell
    writes its distorted image as a `Compression::Fastest` PNG
    into this directory. Filenames are deterministic and
    collision-resistant:
    `<src_stem>_<src_path_hash16>_<codec>_q<q>_<knob_hash16>.png`.
  - `--pairs-tsv <FILE>`: tab-separated companion to the main
    `--output` TSV with columns
    `image_path codec q knob_tuple_json ref_path dist_path` —
    one row per decoded cell. `dist_path` is empty when
    `--distorted-out-dir` is unset.
  - Smoke test: 2-image × 2-q sweep produced 4 PNGs + a 4-row pairs
    TSV that `pycvvdp_worker` then scored into a 4-row
    `cvvdp_pycvvdp_v054` parquet sidecar.

#### scripts/sweep

- `dual_impl_chunk.sh` — per-chunk dual-implementation runner.
  Drives one sweep + both cvvdp scorers (zen-metrics-cli score-pairs
  for cvvdp-gpu + pycvvdp_worker.py for canonical pycvvdp) and
  joins the two sidecars into a parity TSV. Local smoke test: 4
  cells joinable, mean |diff| 0.0245 JOD, max 0.0300 JOD on the
  synth zenjpeg q50/q90 corpus.
- `pycvvdp_worker.py` — canonical pycvvdp v0.5.4 scoring worker.
  Consumes a TSV of `(identity_tuple, ref_path, dist_path)` rows
  and writes a parquet sidecar with the `cvvdp_pycvvdp_v054`
  column per `crates/cvvdp-gpu/docs/CVVDP_SIDECAR_SCHEMA.md`.
  Verified end-to-end on a synth 64×64 pair: JOD 10.0 for identical
  inputs, 9.63 for chroma-shifted.
- `Dockerfile.pycvvdp` — image for the worker on vast.ai. Bases on
  `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` with pycvvdp
  0.5.4, pillow, numpy, pyarrow. CMD is help text; runners must
  pass an explicit `pycvvdp-worker score-pairs …` command.

#### cvvdp-gpu

- `CVVDP_COLUMN_NAME` const exposes a per-implementation column tag
  (default `cvvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>`, overridable via
  the `CVVDP_IMPL_TAG` build-time env var). Used by sweep tooling so
  multiple cvvdp variants land side-by-side in parquet sidecars
  without colliding.

#### zen-metrics-cli

- `MetricKind::Cvvdp::column_names()` now returns
  `cvvdp_gpu::CVVDP_COLUMN_NAME` when the `gpu-cvvdp` feature is
  enabled, so sweep TSV/parquet headers emit
  `score_cvvdp_imazen_v0_0_1` (or the override). The user-facing
  CLI flag `--metric cvvdp` stays stable.

#### cvvdp-gpu (new crate, v0.0.1)

- ColorVideoVDP (still-image) port matching pycvvdp v0.5.4 on the
  v1 R2 manifest within 0.006 JOD across q1–q90. Full pipeline:
  - Color: sRGB→DKLd65 host scalar + `srgb_to_dkl_kernel` (cuda
    parity ≤3e-5).
  - Pyramid: vanilla Laplacian + Weber-contrast variant
    (`weber_contrast_pyr_dec_scalar`) + 4 cubecl kernels
    (`downscale_kernel`, `upscale_v_kernel`, `upscale_h_kernel`,
    `subtract_kernel`, `weber_contrast_compute_kernel`).
  - CSF: 32×32×3 LUT bilinear interp host scalar +
    `csf_apply_per_pixel_kernel` (per-pixel L_bkg from achromatic
    Gaussian pyramid) + `weight_band_kernel`.
  - Masking: mult-mutual + xchannel + soft clamp.
    `mult_mutual_band` host scalar + 3 cubecl kernels
    (`min_abs_3ch_kernel`, `mult_mutual_3ch_no_blur_kernel`,
    `mult_mutual_3ch_with_blurred_kernel`), plus `pu_blur_h_kernel`
    + `pu_blur_v_kernel` for the σ=3 phase-uncertainty blur.
  - Pooling: 3-stage Minkowski + smooth `met2jod` piecewise JOD
    mapping. `pool_band_kernel` does per-pixel `safe_pow` +
    `Atomic<f32>::fetch_add` reduction.
  - Composed: `Cvvdp::score` and `host_scalar::predict_jod_still_3ch`
    are both v1-manifest-locked (≤0.006 JOD). `Cvvdp::new` defaults
    to `DisplayGeometry::STANDARD_4K`; `Cvvdp::new_with_geometry`
    accepts any cvvdp display geometry.
- Parity goldens at
  `s3://coefficient/cvvdp-goldens/v1/manifest.json`
  (public mirror: `https://coefficient.r2.imazen.org/...`).
- Test infrastructure: `parity-goldens` cargo feature gates the
  network-fetching integration test, keeping default `cargo test`
  offline. Per-stage parity tests (color, pyramid, csf, masking,
  pooling) all locked vs pycvvdp.
- **GPU-composed score path** — full pipeline up through D bands +
  masking runs on GPU; only the spatial pool + 3-stage Minkowski +
  `met2jod` are host. New `Cvvdp` helpers:
  - `compute_dkl_weber_pyramid` — color + Weber-contrast pyramid,
    returns `(bands, log_l_bkg)` per the `WeberPyramidGpu` type
    alias.
  - `compute_dkl_t_p_bands(ppd)` — Weber × per-pixel CSF S ×
    `CH_GAIN` × `band_mul`. `band_mul = 2.0` for non-edge levels,
    `1.0` at level 0 and baseband. Baseband sets `CH_GAIN_eff = 1.0`
    so callers can reproduce cvvdp's `|T_p - R_p|` baseband bypass.
  - `compute_dkl_d_bands(ref, dist, ppd)` — composes Weber + CSF +
    masking. Non-baseband bands use the GPU `mult_mutual_3ch_*`
    masker (with the `10^MASK_C` PU-blur scale applied via
    `weight_band_kernel`); baseband uses `|T_p_dis - T_p_ref|`.
    Uses the reference's `log_l_bkg` for both sides per cvvdp's
    `weber_g1` contract.
  - `compute_dkl_jod(ref, dist, ppd)` — full GPU score path
    returning a JOD scalar. Drift survey shows GPU matches host
    within 0.001 JOD for q ≥ 20; the 0.40 drift at q=1 is
    cumulative f32 noise compounding through `met2jod`'s steep
    slope region, not a parity bug.
- `Cvvdp::score_with_reference` is wired (previously returned a
  silent 0.0). Caches reference sRGB bytes and routes through
  `host_scalar::predict_jod_still_3ch` — exact-parity with
  `Cvvdp::score(ref, dist)`.
- Drift-survey tests document where GPU vs host diverges per
  stage: `compute_dkl_{weber_pyramid,t_p_bands,d_bands}_matches_host_on_corpus_256x256`
  + `compute_dkl_jod_vs_host_scalar_on_corpus` +
  `compute_dkl_jod_on_v1_manifest_corpus`.
- `zenbench` score-path benchmark (`benches/score.rs`) — first
  measured CPU vs GPU per-pixel numbers at 256×256 / 1 MP / 12 MP.
- `time_12mp` example (`examples/time_12mp.rs`) — fixed-iteration
  one-shot timer for compute_dkl_weber_pyramid / compute_dkl_d_bands
  / compute_dkl_jod at 12 MP. Per-phase breakdown surfaces where
  the GPU pipeline spends its time without the zenbench
  calibration loop's overhead at large image sizes.
- `CVVDP_TRACE=1` env-var-gated stderr instrumentation inside
  `compute_dkl_d_bands` — per-level CSF / masking / log_l_bkg
  upload timings. Zero-cost when unset.
- `CVVDP_TRACE_WEBER=1` env-var-gated stderr instrumentation
  inside `compute_dkl_weber_pyramid` splitting GPU dispatch from
  host readback.
- Direct kernel-level parity test for `csf_apply_3ch_kernel`
  in `tests/csf_kernel.rs` — sweeps the full log_l_bkg LUT axis
  with distinct per-channel ch_gain values (catches bugs the
  indirect d_bands test would miss).
- Consecutive-weber diagnostic block in `examples/time_12mp.rs`
  (`0a71bb22`) — calls `compute_dkl_weber_pyramid` twice on the
  same `ref_bytes` outside `compute_dkl_d_bands` to isolate
  whether the "weber(dist) is 2× weber(ref)" slowdown is
  position-dependent (consecutive-call overhead) or data-shape
  dependent. Result: standalone consecutive calls show no
  slowdown, ruling out cubecl warm-up / driver effects and
  pinning the cause to host memory pressure from holding the
  `ref_weber: Vec<Vec<f32>>` (~190 MB at 12 MP) alive across the
  second call inside the d_bands flow.
- `_dispatch_weber_pyramid_gpu` private helper (`072d9e43`)
  factored out of `compute_dkl_weber_pyramid` — takes a
  `&[Handle]` destination slice for the per-level `log_l_bkg`
  outputs. The bisect for tick 85's 5× regression revealed
  that this extraction itself does not regress, only the
  full 5-phase serial restructure did; the helper is kept so
  future experiments can swap the destination buffer set
  without re-wiring weber's body.

### Fixed

#### cvvdp-gpu

- **73×91 odd-dim residual closed (was 0.006 JOD).** Found a
  parity-check bug in pycvvdp's `gausspyr_reduce`: the
  horizontal-pass right-column patch uses `x.shape[-2]` (INPUT
  ROW parity) to pick its odd/even branch even though the
  comments say "columns" — `lpyr_dec.py:204-209`. For
  mixed-parity inputs (e.g. 6×5 → 3×3 at the 73×91 baseband)
  pycvvdp applies the wrong patch.
  - `host_scalar` `gausspyr_reduce_scalar`: rewritten to bug-
    compatible zero-pad + parity-aware patches.
  - GPU `downscale_kernel`: adds a delta correction at the right
    column when sw and sh parities differ.
  - New `compute_dkl_jod_matches_pycvvdp_at_73x91_odd` test
    passes at f32 precision (diff = 0.0000 vs pycvvdp golden).
  - All other corpus fixtures (256² + 4 MP, same-parity dims)
    unchanged — the bug-compat patches match pure reflection
    for all sw == sh parity inputs.

- **Chroma-shift drift closed (was 0.117 JOD).** pycvvdp overrides
  the baseband CSF rho to 0.1 cy/deg (`cvvdp_metric.py:628`),
  but our pipeline used the geometric value from
  `band_frequencies(ppd, w, h)` (0.190 at 256² standard_4k). Fixed
  by adding `kernels::csf::CSF_BASEBAND_RHO = 0.1` and applying it
  in both `host_scalar::predict_jod_still_3ch` and
  `Cvvdp::new`'s `logs_row` pre-upload. The
  `compute_dkl_jod_matches_pycvvdp_at_256x256_chroma_shift` test
  re-enabled at standard 0.005 JOD tolerance; chroma_shift now
  matches pycvvdp golden 9.664865 to f32 precision.

### Changed (performance)

#### cvvdp-gpu

After tick 70's per-band-allocation diagnosis, four scratch
hoists + one kernel fuse landed in succession:

- **Pre-allocate per-band CSF + masking scratch** on `Cvvdp::new`.
  `compute_dkl_d_bands` was alloc_zeros_f32-ing 18 buffers per
  non-baseband level per call (~1.5 GB worth at 12 MP). Moved
  to a `DBandsScratch` struct on the Cvvdp instance. Result:
  12 MP d_bands −25%, full jod −30%.
- **Pre-allocate per-band Weber pyramid scratch** — same shape
  for the expand/subtract/weber chain (l_bkg_fine, vscratch_a,
  log_l_bkg, per-channel vscratch_c/upscaled_c/layer_c).
  Result: 12 MP weber alone 5× faster (105 → 21.6 ns/px), full
  jod 2.4× faster (310 → 127 ns/px). **This crossed the milestone
  of beating fcvvdp single-thread** (214 ns/px on their bench).
- **Drop unused per-side GPU buffers** (`src_dis`, `gauss_dis`,
  `bands_dis`, `pool_partials`) that were allocated by
  `Cvvdp::new_with_geometry` but never read by any GPU helper.
  Saves ~13 MB per Cvvdp at 256×256.
- **Hoist `logs_row` uploads** to `Cvvdp::new_with_geometry`
  (24 small uploads of 128 B were happening per d_bands call,
  one per `(level, channel)`). Since `rho_k` is fixed per Cvvdp,
  the LUT rows are stable across calls.
- **Fuse 3-channel CSF apply** into a single kernel
  (`csf_apply_3ch_kernel`) that shares the per-pixel LUT bracket
  math across A/RG/VY channels. Cut L0 CSF time at 12 MP from
  420 ms (6 launches) to 170 ms (2 launches) — but the saved
  ~250 ms got absorbed by ~340 ms of unaccounted overhead
  (likely host Vec<Vec<f32>> alloc for the weber readback);
  median d_bands wall is unchanged.
- **`pow(10, x) → exp(x · ln(10))`** in CSF kernels for the
  mathematical identity. No measurable win on cuda (likely cubecl
  already compiles to similar PTX); kept for potential wgpu/hip
  payoff.
- **Dist-side CSF reads `self.bands_ref` handles directly**
  (`8b6f2776`) — `compute_dkl_d_bands` no longer uploads
  `dist_weber[k]` from host inside the per-band CSF apply. The
  dist-side handles are already resident in `self.bands_ref`
  after the `weber(dist)` call earlier in the band loop, so the
  CSF kernel reads them in place. REF-side still uploads since
  `bands_ref` has been overwritten with DIST data by band-loop
  time. Result on 12 MP cuda: weber 291 ms (baseline),
  d_bands 1.42 s (−3% from 1.46 s), jod 1.40 s (−7% from 1.50 s).
  Parity intact at 1.3e-3 band-relative on q=1 corpus. Critically,
  this also proves the handle-direct CSF pattern is **innocent**
  of tick 85's 5× weber regression — that regression was the
  5-phase serial restructure, not the handle access pattern.

The post-tick-87 fusion + structural-change wave (ticks 89–96)
took the d_bands per-band launch count from 27 → 14:

- **`weber_contrast_compute_3ch_kernel`** (`af994a87`) — fuses
  the per-pixel `layer/clamp(L_bkg)` math and the shared
  `log_l_bkg = log10(L_bkg)` write into one launch per
  non-baseband level. Was 3 separate
  `weber_contrast_compute_kernel` launches. log10 computed
  once per pixel instead of three times.
- **`subtract_weber_3ch_kernel`** (`39d6957f`) — drops the
  `layer_c` intermediate entirely. Reads `fine_c` and
  `upscaled_c` handles directly and writes `band[c] =
  clamp((fine_c − upscaled_c) / L_bkg)` for all three channels
  + shared `log_l_bkg` in one launch. Was 3 `subtract_kernel`
  launches + the (already-fused) weber compute. Frees ~36 MB
  of `WeberScratch.layer_c` at 12 MP per side.
- **`pu_blur_h_3ch_kernel` + `pu_blur_v_3ch_scaled_kernel`**
  (`78d951d1`) — fuses the masking-branch pu_blur into one
  h-pass + one v-pass for all 3 channels, AND folds the
  `* 10^MASK_C` post-scale into the v-pass output. Cuts the
  masking blur chain from 9 launches per non-baseband level
  (3× h + 3× v + 3× `weight_band_kernel`) to 2.
- **`csf_apply_6ch_kernel`** (`7bf02fae`) — fuses the
  REF + DIST CSF apply into a single launch sharing the
  per-pixel LUT bracket math. Per non-baseband level: 2
  `csf_apply_3ch_kernel` launches → 1 6-channel launch.
- **`diff_abs_3ch_kernel`** (`06d8e4a5`) — moves the
  baseband `|T_p_dis - T_p_ref|` bypass to GPU. Every level's
  D plane now lives in the same `d_scratch.d[k][c]` slot.
- **`pool_band_kernel` in `compute_dkl_jod`** (`5817a2e4`)
  — replaces host-scalar `lp_norm_mean` over the per-band D
  Vecs with GPU `pool_band_kernel(d_handle) → partials[k*3+c]`.
  Partials buffer is `n_levels × N_CHANNELS` floats (~144 bytes
  at 12 MP); the host fold operates on that tiny Vec.
- **Split `compute_dkl_d_bands`** (`ea632f87`) — extracted
  `_dispatch_d_bands_into_scratch` private helper that does the
  GPU dispatch only. `compute_dkl_jod` calls the helper
  directly and skips the per-band Vec readback that
  `compute_dkl_d_bands` was paying. **17% wall-time win** at
  12 MP (jod 122.4 → 101.8 ns/px); jod is now faster than
  d_bands because it skips the ~432 MB host readback. vs
  fcvvdp 8-thread at 360p, the gap narrowed from 1.48× slower
  (tick 89) to 1.18× slower.

Post-fuse housekeeping (ticks 97–107):

- **`examples/time_size_sweep.rs`** + benchmark snapshot
  (`134bc04a`) — covers tiny (64²), small (256²), medium
  (1024²), large (4000×3000) sizes with per-phase wall + per-
  pixel cost + naive OLS fit. Found per-pixel cost is
  **non-monotonic** in image size: medium (1 MP) is the
  cheapest at 53.7 ns/px JOD, large (12 MP) regresses to
  159 ns/px; weber alone shows the same shape (19 → 61 ns/px),
  so the regression is intrinsic to the dispatch, not pure
  readback bandwidth. Open investigation.
- **`shadow_jod_gpu`** manifest-parity test (`562ee924`) —
  pins the GPU JOD path directly against pycvvdp v0.5.4's
  published manifest values (not just against the host
  scalar via relative parity). q=1 tolerance is wider (0.5
  JOD) per the documented cumulative-f32 drift; q≥20 tol is
  0.05 (observed < 0.001).
- **`Cvvdp::level_dims`** helper (`efcdba76`) — drops 5 sites
  of duplicated `if k == 0 { width } else { width >> k }`
  boilerplate. The `if k == 0` branch was redundant since
  `>> 0` is a no-op.
- **Dropped `Cvvdp.ref_log_l_bkg` dead field** (`ba586480`)
  — was added in tick 85 for a regression bisect that
  confirmed the field was NOT the cause; kept around with
  `#[allow(dead_code)]` for "future use" that subsequent
  ticks went around. Frees ~190 MB of unused GPU memory per
  `Cvvdp::new` at 12 MP, drops 14 lines of allocation code.
- **`compute_dkl_t_p_bands` modernized** (`8e509807`) — uses
  the fused `csf_apply_3ch_kernel` and reads weber from the
  GPU-resident `bands_ref` handles instead of re-uploading
  from the host Vec. Per non-baseband level: 3 host uploads
  + 3 launches → 0 uploads + 1 launch.

Post-fuse housekeeping (ticks 108–124):

- **Tests + examples + benches now run under wgpu** (`a0473bf9`,
  `3c72a86d`, `70a62e63`) — `shadow_jod_gpu`, `time_12mp`,
  `time_size_sweep`, and `benches/score.rs` all switched from
  cuda-only to the `cfg(any(cuda, wgpu))` + `Backend` type-alias
  pattern. Machines without a CUDA SDK (macOS, AMD, Intel) can
  now run the manifest-parity anchor + per-phase timings under
  wgpu's Vulkan/Metal/DX12 backend.
- **`ch_gain_for_band(is_baseband, band_mul)` helper** (`f5c1df3c`)
  — replaces 6 lines of `if is_baseband { 1.0 } else { band_mul *
  CH_GAIN[c] }` boilerplate at two band-loop sites with a single
  destructuring bind.
- **Stack-allocated `compute_dkl_jod` partials zero-init**
  (`a4e019c0`) — replaces a 192-byte heap Vec with
  `[0.0_f32; MAX_LEVELS * N_CHANNELS]` sliced to the active
  prefix.
- **CHANGELOG catch-up + PORT_STATUS refresh + many small doc
  fixes** (`bcf3dfcc`, `0dc01ea5`, `b7686203`, `35a0b48d`,
  `6826c0eb`, `77908be7`, `fd1e2527`, `8cd803a9`, `ac1e21d3`,
  `067ba379`, `08c65040`, `45719dad`, `1b8b51ca`) — module-level
  pipeline overviews in `lib.rs`, `pipeline.rs`, and
  `kernels/mod.rs` updated to name the actual fused kernels;
  stale claims about which stages run host-side cleared;
  `compute_dkl_weber_pyramid` got its missing doc comment; the
  misleading α/β OLS fit dropped from `time_size_sweep`; and 9
  of 15 rustdoc warnings cleared (remaining 6 are macro-induced
  by `#[cube(launch)]`'s function-and-module duplication).
- **`Cvvdp::score` v1 manifest tolerance** still pinned by the
  CPU reference path (`shadow_jod`). The GPU composition path
  is parity-locked against pycvvdp directly via `shadow_jod_gpu`
  but with a wider q=1 tolerance (~0.4 JOD) per the documented
  cumulative-f32 drift through `met2jod`'s steep slope.

Host-memory-pressure relief (ticks 144–146):

- **Drop dist_weber host Vec immediately** (`02f37728`) —
  `compute_dkl_d_bands` was binding the `(dist_weber, _)` tuple
  from `compute_dkl_weber_pyramid(dist_srgb)` even though the
  dist-side CSF path reads `self.bands_ref` GPU handles
  directly (per tick 87). Changed to `let _ = ...` so the
  ~190 MB host Vec drops at the call site instead of
  surviving the band loop.
- **Per-band ref-side host Vec drops** (`913a7c5f`) — after the
  band-`k` CSF dispatch finishes its `create_from_slice`
  uploads, replace `ref_weber[k] = [Vec::new(); 3]` and
  `ref_log_l_bkg[k] = Vec::new()` so peak host residency scales
  with the remaining-bands sum, not the whole pyramid.

Together these two commits dropped 12 MP perf
(`benchmarks/time_12mp_tick145_2026-05-14.md`):
- weber pyramid: 26.4 → 30.6 ns/px (noise band)
- compute_dkl_d_bands: 106.6 → **82.1 ns/px** (−23%)
- compute_dkl_jod: 101.8 → **87.2 ns/px** (−14%)

The `d_bands − 2×weber` bucket (CSF + masking + IO) dropped
from 645 ms → 252 ms — a **2.5× speedup** on the non-weber
portion. vs fcvvdp's 8-thread number at 360p we crossed from
1.48× slower (tick 89) to 1.18× slower (tick 96) to **1.01×
tied** here.

- **DIST weber pyramid skips host readback entirely**
  (`8c5b96e0`, tick 150) — `compute_dkl_d_bands` was calling
  `compute_dkl_weber_pyramid` for the DIST side and
  immediately discarding the returned tuple. Tick 144 caught
  the unused tuple; tick 150 caught that the *wrapper* itself
  still allocated ~240 MB of host Vecs and issued
  `client.read_one` calls that wait for the GPU dispatch to
  complete before transferring bytes. Replaced with
  `_dispatch_weber_pyramid_gpu` (the dispatch-only private
  helper) — skips both the allocation AND the GPU→host
  transfer.

  Result on the next 12 MP run
  (`benchmarks/time_12mp_tick150_2026-05-14.md`):
  - compute_dkl_d_bands: 82.1 → **71.0 ns/px** (−14%)
  - compute_dkl_jod: 87.2 → **74.6 ns/px** (−14%)
  - `d_bands − 2×weber`: 252 ms → 156 ms (−38%)
  - vs fcvvdp 8-thread @ 360p: now **1.15× faster** (vs 1.01×
    tied pre-tick).

Perf trajectory through the recent fusion + host-pressure wave:

| tick | jod ns/px | vs fcvvdp 8t @ 360p |
| ---- | --------- | ------------------- |
| 64   | 444       | 5.16× slower        |
| 73   | 127       | 1.48× slower        |
| 89   | 122       | 1.42× slower        |
| 96   | 102       | 1.18× slower        |
| 145  |  87       | 1.01× tied          |
| 150  |  **75**   | **1.15× faster**    |

Host-memory-pressure relief continued + structural readback
elimination (ticks 151–160):

- **REF CSF reads `bands_ref` GPU handles directly** (tick 155,
  `d7c7322c`) — symmetrical to tick 87's DIST-side fix. The
  band-loop's REF CSF dispatch had been uploading `ref_weber[k]`
  from the host Vec; after tick 154's `bands_ref` / `bands_dis`
  split persisted both sides' data on GPU, the REF CSF kernel
  reads `self.bands_ref[k]` handles in place. Drops 3 host→GPU
  uploads per non-baseband level (~50 MB total at 12 MP).
- **REF weber pyramid skips bands readback** (tick 156, `2993c0a0`)
  — `_dispatch_d_bands_into_scratch` had been calling the public
  `compute_dkl_weber_pyramid(ref_srgb)` wrapper which read back
  ~190 MB of bands per call (`Vec<Vec<f32>>`). Replaced with a
  direct call to `_dispatch_weber_pyramid_gpu` + a manual
  `log_l_bkg`-only readback loop. 12 MP jod 70.3 → 60.2 ns/px
  (−14%), now 1.43× faster than fcvvdp 8t.
- **Dispatch-only split for `compute_dkl_planes` + `compute_dkl_gauss_pyramid`**
  (tick 157) — extracted private `_dispatch_dkl_planes_gpu` and
  `_dispatch_gauss_pyramid_gpu` siblings.
  `_dispatch_weber_pyramid_gpu` and `compute_dkl_laplacian_pyramid`
  switched off the public wrappers (was `let _ = ...`). Saves
  ~230 MB of wasted host transfer per weber call (36 MB level-0
  + ~190 MB pyramid). 12 MP jod 60.2 → 53.0 ns/px (−12%), now
  1.62× faster than fcvvdp 8t.
- **GPU baseband-divide** (tick 158, `3b78f847`) — adds
  `baseband_divide_3ch_kernel` (pyramid.rs). The weber baseband
  finishing step had been doing 3 channel readbacks + 3 channel
  reuploads + per-channel host divides; now does 1 GPU launch
  using host-computed `l_bkg_mean` as a scalar uniform. Sync
  drain count per weber side: 4 → 1.
- **Tested-and-regressed 3ch upscale fusion + laplacian dispatch-only split**
  (tick 159, `6495c462`) — `upscale_v_3ch_kernel` /
  `upscale_h_3ch_kernel` (same fusion pattern as
  `weber_contrast_compute_3ch`) regressed jod ~4% at 12 MP on
  RTX CUDA across two runs. Hypothesis: 3ch register footprint
  reduced warp-level latency hiding more than launch overhead
  was costing us. Left a breadcrumb in pyramid.rs so this isn't
  re-tried without a different angle (e.g. shared-memory tiling).
  Same commit also added `_dispatch_laplacian_pyramid_gpu` so
  `compute_dkl_csf_weighted_bands` no longer discards a full-
  pyramid host readback via `let _ = ...`.
- **Direct parity test for `baseband_divide_3ch_kernel`**
  (tick 160, `baf4878e`) — closes a coverage gap from tick 158.
  The kernel had been verified through the higher-level
  `compute_dkl_weber_pyramid_matches_host_on_corpus_256x256`
  integration test; the new unit test in `pyramid_kernel.rs`
  gives a fast regression gate with inputs that exercise
  negatives, large magnitudes, and 3 distinct channel patterns.

12 MP perf trajectory through this wave
(`benchmarks/time_12mp_tick{155,156,157,158}_2026-05-14.md`):

| tick | jod ns/px | weber 1-side | d_bands  | vs fcvvdp 8t |
| ---- | --------- | -----------  | -------- | ------------ |
| 150  | 74.6      | 29.0         | 71.0     | 1.15× faster |
| 155  | 70.3      | 31.8         | 73.5     | 1.22× faster |
| 156  | 60.2      | 29.2         | 52.0     | 1.43× faster |
| 157  | 53.0      | 25.5         | 45.2     | 1.62× faster |
| 158  | **52.9**  | **24.9**     | **43.7** | **1.63× faster** |

Continued perf wave + structural cleanup (ticks 162–166):

- **PORT_STATUS.md refresh** (tick 162, `621a5867`) — weber-
  contrast pyr row names `baseband_divide_3ch_kernel`, composed-
  pipeline row carries the tick 158 perf number, "Open tick 159"
  entry documents the 3ch upscale fusion negative result.
- **`compute_dkl_t_p_bands` skips bands readback**
  (tick 163, `8a6de7be`) — same tick-156 pattern applied to the
  test-only T_p path. Was discarding the bands portion of
  `compute_dkl_weber_pyramid`'s return tuple (~190 MB host
  alloc per call at 12 MP). Now dispatches via the private
  helper + log_l_bkg-only readback.
- **Size-sweep re-measurement** (tick 164, `d27c5194`) —
  documents the tick 150-158 wave's per-bucket impact:
  - tiny    jod 1835 → 527 ns/px (−71%)
  - small   jod  223 →  91 ns/px (−59%)
  - medium  jod   65 →  28 ns/px (−56%)
  - large   jod  145 →  39 ns/px (−73%)
  Most importantly the medium→large per-pixel regression open
  since tick 97 **narrowed from 2.2× to 1.36×** — falsifies the
  L2-cache-pressure hypothesis as dominant; most of it was
  host memory pressure all along. Small (256²) is now the most-
  expensive per-pixel bucket — launch overhead dominates at
  that thread count.
- **`pool_band_3ch_kernel` fusion** (tick 165, `df4dd106`) —
  3 per-channel pool launches per level → 1 fused 3ch launch.
  Total pool dispatch: `n_levels × N_CHANNELS = 24` → `n_levels
  = 8` launches per JOD. Unlike tick 159's upscale 3ch fusion
  (regressed via register pressure), pool kernel does only 3
  powfs + 3 atomic-adds per thread — register footprint stays
  small, fusion wins on launch-overhead reduction. 12 MP jod
  52.9 → 49.0 ns/px (−7%), 1.76× faster than fcvvdp 8t.

  **Decision rule for 3-channel fusion** extracted from
  tick 159 vs tick 165: fusion wins when per-thread arithmetic
  is tiny (atomics, pointwise math); loses to register pressure
  on medium-arithmetic kernels (5-tap convolutions, multi-read
  patterns). Future 3ch fusion attempts should respect this.

- **`log_l_bkg` roundtrip elimination** (tick 166, `7ce2bc24`)
  — adds `WeberScratch.log_l_bkg_dis` throwaway destination
  (parallel to tick 154's `bands_dis` split) so the DIST weber
  dispatch's log_l_bkg write doesn't clobber REF's data on
  `weber_scratch[k].log_l_bkg`. Per cvvdp's weber_g1 rule,
  both sides use REF's log_l_bkg, so DIST's value is computed-
  then-discarded. The band loop's CSF kernel now reads REF's
  log_l_bkg directly from the GPU-resident handle — no host
  roundtrip.

  Bytes saved per JOD at 12 MP: ~128 MB (64 MB readback +
  64 MB reupload of the same data). Sync drains saved: 7
  (one per non-baseband level). 12 MP jod 49.0 → **41.8 ns/px**
  (−15%). Now **2.06× faster than fcvvdp 8-thread @ 360p**.

12 MP perf trajectory through ticks 165-166
(`benchmarks/time_12mp_tick{165,166}_2026-05-14.md`):

| tick | jod ns/px | weber 1-side | d_bands  | vs fcvvdp 8t |
| ---- | --------- | -----------  | -------- | ------------ |
| 158  | 52.9      | 24.9         | 43.7     | 1.63× faster |
| 165  | 49.0      | 23.4         | 41.3     | 1.76× faster |
| 166  | **41.8**  | **22.2**     | **39.8** | **2.06× faster** |

Warm-ref API + last per-JOD host alloc removed (ticks 168–171):

- **`fill_f32_kernel` + `baseband_log_l_bkg` pre-alloc**
  (tick 168, `e0b6ca62`) — replaces the baseband band's per-JOD
  `vec![log_l_bkg_baseband; n]` host alloc + GPU upload with a
  single GPU fill launch into a pre-allocated buffer. Wallclock
  impact minimal (baseband is small), but closes the last
  per-JOD host alloc in the hot path. New parity test
  `fill_f32_kernel_writes_uniform_value` uses a sentinel-fill
  trick to catch off-by-one or short-write bugs.
- **Extract REF/DIST weber helpers + perf snapshot**
  (tick 169, `ea13bcf8`) — factors
  `_dispatch_ref_weber_pyramid_only` and
  `_dispatch_dist_weber_pyramid_only` out of
  `_dispatch_d_bands_into_scratch`. No behaviour change, sets
  up the warm-ref API. The tick 169 measurement landed at
  jod 38.0 ns/px (2.26× faster than fcvvdp 8t @ 360p) —
  the tick 166 reading at 41.8 was on the high end of its noise
  band.
- **Warm-ref batch-scoring API** (tick 170, `abe3599d`) —
  delivers the `score_with_reference` doc promise from v0.0.1:
  - `Cvvdp::warm_reference(ref_srgb)` dispatches REF weber once
    and stores `Some(log_l_bkg_baseband)` in
    `Cvvdp::warm_ref_baseband_log_l_bkg`. Any subsequent method
    that dispatches REF weber resets this to `None` —
    `_dispatch_ref_weber_pyramid_only` does the reset
    unconditionally so warm-reference is the only path that
    arms it.
  - `Cvvdp::compute_dkl_jod_with_warm_ref(dist_srgb, ppd)`
    skips the REF half of the JOD pipeline. Returns
    `Error::NoWarmReference` if the cache is cold.
  - Refactored band loop + pool into `_dispatch_d_bands_dist_and_band_loop`
    and `_pool_and_finalize_jod` so cold and warm paths share
    the post-REF tail.
  - Parity test `compute_dkl_jod_with_warm_ref_matches_unwarm_path`
    verifies: (1) warm/cold byte-for-byte match within 1e-5
    JOD, (2) state survives multiple warm-ref calls,
    (3) intervening cold calls invalidate correctly.
- **`time_12mp` measures warm-ref fast path**
  (tick 171, `8c7c5f96`) — adds phase 4 measuring per-DIST cost
  after one `warm_reference` per iter. 12 MP results:
  - jod (cold REF):       36.1 ns/px
  - jod_warm (cached REF): **20.6 ns/px**
  - Per-DIST saving: 42.9% (1.75× faster per call)
  - vs fcvvdp 8-thread @ 360p: **4.17× faster per DIST**

Warm path delivers below the naive 50% saving because the host
pool fold + band loop dispatch overhead run once per JOD
regardless of REF state. The amortization break-even is ~2
candidates per warmed reference — anything larger lands at
1.75× throughput.

| tick | jod cold (ns/px) | jod warm (ns/px) | vs fcvvdp 8t (cold / warm) |
| ---- | ----             | ----             | ----                        |
| 158  | 52.9             | —                | 1.63× / —                   |
| 166  | 41.8             | —                | 2.06× / —                   |
| 169  | 38.0             | —                | 2.26× / —                   |
| 171  | **36.1**         | **20.6**         | **2.38× / 4.17× faster**    |

The `d_bands − 2×weber` bucket (CSF + masking + IO) is sub-noise
since tick 156: 2×weber ≈ d_bands, meaning the band-loop overhead
is now bandwidth-tightly packed against the two weber pyramids.
The next remaining hot spot is the gauss-pyramid reduce (5×5
downscale, 25 src reads per output pixel), which a shared-memory
tiled rewrite could shrink — but the per-thread register
pressure observation from tick 159 means any fusion attempt
should change the memory access pattern, not just rearrange
launches.

### Tick 175–178 — ceil-div correctness wave (resolves tick 174 drift)

After tick 174 root-caused the 0.586 JOD drift vs pycvvdp at 12 MP
to floor-div vs ceil-div pyramid halving, the next ticks shipped
the fix and locked it with new tests.

- **Ceil-div pyramid + MAX_LEVELS = 9** (tick 175, `cee15d24`)
  — `build_pyramid` / `build_weber_scratch` /
  `build_d_bands_scratch` / `pyramid_levels` switched from
  `n / 2` to `(n + 1) / 2`. Order mattered: bumping MAX_LEVELS
  alone (tick 174 attempt) widened the drift to 1.54; ceil-div
  first then bump levels closed it to 0.0003.
  - 4000×3000 synth: ours **9.4583** vs pycvvdp **9.4580** —
    **drift 0.586 → 0.0003 JOD** (2000× more accurate).
  - All 67 existing parity tests stayed green (they run at
    power-of-2 sizes where floor == ceil at every level).
  - Trade-off: jod cold 36 → 62 ns/px, warm-ref 21 → 34 ns/px
    on the same RTX 5070. Open investigation — total pixel
    work is nearly unchanged, so the ~25% post-warmup slowdown
    must be a kernel-dispatch or boundary-branch interaction,
    not extra compute. Snapshot: `benchmarks/pycvvdp_parity_tick175_2026-05-15.md`.

- **`level_dims` reads `gauss_ref` shapes** (tick 176, `b9b5b71a`)
  — was computing `(bw, bh, n_px)` via `width >> k` (floor-div
  bit shift), which disagreed with the ceil-div allocator at
  odd-dim levels. Consequence: the band loop's CSF + masking +
  pool kernels dispatched fewer threads than the bands_ref /
  d_scratch buffers actually held — the last few tail pixels at
  each odd-dim level were written by weber but never processed
  downstream. 12 MP JOD output unchanged (tail values were
  near-zero so didn't move the pool), but the inconsistency
  was real and would matter on other inputs. Now reads
  `gauss_ref[k].w / .h` directly so all shape-using sites
  agree.

- **Odd-dim JOD parity test** (tick 177, `f2425dce`) — added
  `compute_dkl_jod_matches_host_scalar_on_odd_dims` at 73×91
  (the smallest source that diverges at ceil-vs-floor level 4+).
  Catches future floor-div regressions in either host_scalar
  or the GPU pyramid path. The other JOD parity tests all run
  at power-of-2 sizes where floor == ceil.

- **12 MP pycvvdp golden parity test** (tick 178, `cd61a217`)
  — added `compute_dkl_jod_matches_pycvvdp_at_12mp_synth`. The
  deterministic 4000×3000 synth pair from
  `examples/time_12mp.rs` runs through `compute_dkl_jod` and
  asserts the output matches pycvvdp v0.5.4's measured 9.4580
  golden within 0.005 JOD. Current observed diff: 0.0003.
  Would have failed at tick 173 (diff 0.586) and tick 174
  (diff 1.54); now gates the canonical-reference correctness
  in CI. Runtime ~5 s per call.

The pycvvdp parity matrix is now end-to-end:

| size      | test                                                              | tolerance | observed |
| ----      | ----                                                              | ----      | ----     |
| 32×32     | `compute_dkl_jod_matches_host_scalar`                            | 0.5 JOD   | < 0.1    |
| 73×91     | `compute_dkl_jod_matches_host_scalar_on_odd_dims`                | 0.5 JOD   | **0.0004** (post tick 181) |
| 256×256   | `compute_dkl_jod_matches_host_on_corpus_256x256` (drift sweep)   | 0.06 JOD  | < 0.05   |
| 4000×3000 | `compute_dkl_jod_matches_pycvvdp_at_12mp_synth`                  | 0.005 JOD | **0.0003** |
| 256×256 v1 manifest | `shadow_jod` (host scalar)                              | 0.01 JOD  | < 0.006  |

### Tick 179–182 — band-count alignment + pycvvdp goldens manifest

- **CHANGELOG / PORT_STATUS / lib.rs docs caught up to tick 175-178**
  (tick 179, `d7f8445f`) — the ceil-div correctness wave is now
  surfaced in user-facing docs. Corrected `lib.rs` to drop the
  misleading "2.58× slower than pycvvdp" framing (those numbers
  reflected a broken pyramid drifting 0.586 JOD); honest post-fix
  is 4.4× slower cold / 2.4× slower warm with correct output.

- **Extended pycvvdp bench script + goldens manifest**
  (tick 180, `b937401e`) — `scripts/cvvdp_goldens/bench_12mp_cuda.py`
  now produces a `pycvvdp_synth_goldens.json` manifest with the
  pycvvdp golden JOD for both the 4000×3000 12 MP fixture
  (9.4580) and a 73×91 odd-dim fixture (9.3904). The manifest
  schema lets future Rust parity tests load canonical reference
  values directly instead of duplicating hardcoded constants.

- **Surprise: host_scalar drifts ~0.6 JOD vs pycvvdp at 73×91**
  (tick 180 finding) — at sub-megapixel sizes our host_scalar
  reference produces 8.79 vs pycvvdp 9.39. The 256² v1 manifest
  fixtures hold ≤ 0.006 JOD, the 4000×3000 synth holds 0.0003,
  but 73×91 drifts ~0.6. Possible causes (open investigation):
  CSF interpolation at very small angular widths, band-mul rule
  difference for the small-band branch, or a display-geometry
  interpretation gap at sub-degree image sizes.

- **`pyramid_levels` defers to `band_frequencies` (tick 181, `e4951c15`)**
  — the GPU pipeline had a size-based cap (`cur >= 2 *
  PYRAMID_MIN_DIM`) that produced fewer bands than host_scalar
  at small inputs (4 vs 5 at 32², 5 vs 6 at 73×91, 7 vs 8 at
  256²). host_scalar already used `band_frequencies(ppd, w, h).len()`
  directly. Aligned the GPU side. Effect on the 73×91 GPU-vs-host
  parity test: **diff 0.092 → 0.0004 JOD** (235× better
  agreement). 12 MP pycvvdp gate still passes at 0.0003.

  Resolves the GPU↔host structural mismatch at small sizes.
  The remaining ~0.6 JOD drift at 73×91 is purely host_scalar
  vs pycvvdp (GPU now matches host within f32 precision).

### Investigation Notes (cvvdp-gpu, tick 174 — large-image drift)

After tick 173's pycvvdp v0.5.4 CUDA bench surfaced a **0.586 JOD
drift** between our `compute_dkl_jod` and pycvvdp on a 4000×3000
synthetic pair (ours 8.8726, pycvvdp 9.4580), tick 174 traced the
cause. Diagnostic scripts in `scripts/cvvdp_goldens/`:

- `bench_12mp_cuda.py` — pycvvdp CUDA timing + JOD output
- `diagnose_12mp.py` — pycvvdp metric internals
- `diagnose_pyramid.py` — pycvvdp band_freqs + height + pyr_shape
- `diagnose_freqs.py` — direct comparison of band frequencies
- `diagnose_decompose.py` — actual band tensor shapes via decompose()

**Two structural divergences from pycvvdp at large sizes:**

1. **n_bands cap**. Our `MAX_LEVELS = 8` caps the pyramid at 8
   levels. pycvvdp uses **9 bands** at 4000×3000 (one extra deep
   level). Bumping `MAX_LEVELS` alone is insufficient — see #2.

2. **Floor vs ceil division on pyramid sizes** (the dominant
   cause). pycvvdp uses **ceil-div** when halving level
   dimensions; we use floor-div. The bands diverge from level 4
   onward:

   | level | pycvvdp shape (ceil)  | cvvdp-gpu shape (floor) |
   | ---   | ---                   | ---                     |
   | 0     | 3000×4000             | 3000×4000               |
   | 1     | 1500×2000             | 1500×2000               |
   | 2     | 750×1000              | 750×1000                |
   | 3     | 375×500               | 375×500                 |
   | 4     | **188**×250           | **187**×250             |
   | 5     | 94×125                | 93×125                  |
   | 6     | **47×63**             | **46×62**               |
   | 7     | 24×32                 | 23×31                   |
   | 8     | 12×16 (baseband)      | (n/a — capped)          |

   Naively bumping MAX_LEVELS to 10 + adding level 8 INCREASED
   the drift (JOD 8.87 → 7.92) because the ceil-div mismatch
   compounds with every additional level. Reverted MAX_LEVELS
   to 8 until the ceil-div fix lands.

The 0.006 JOD parity tolerance our existing tests hit at 256×256
holds because at small sizes the ceil/floor difference is 0 or 1
pixel and most of pycvvdp's pyramid math rounds out. At 12 MP
the divergence stacks to ~0.6 JOD.

**Fix plan** (multi-tick):
- Switch pyramid `Level` allocator + `gauss_ref` chain to
  ceil-div (`(w + 1) / 2`).
- Update `downscale_kernel` boundary handling for the off-by-one
  case (currently floor-div semantics).
- Update upscale `back_v` / `back_h` math which assumes the
  parent floor-div shape.
- Bump MAX_LEVELS to 10 once ceil-div parity holds at 256×256.
- Add a 12 MP parity test driven by a pycvvdp golden so the
  drift is visible in CI.

**Goldens expansion (user ask, 2026-05-15):**

> pycvvdp needs to be the source of goldens and we have to sweep
> a larger distortion set

Current goldens at `v1/manifest.json` only cover 256×256 source
×6 JPEG quality levels. Planned expansion:
- Multi-resolution: 256², 1024², 4000×3000 (and 8K for sanity).
- More distortion types: Gaussian blur, Gaussian noise,
  contrast/saturation perturbations, downscale+upscale, color
  shifts, dithering, banding.
- Quality levels closer to perceptual JND than just JPEG-q.
- Sweep dimension: image content (photo, screen, line-art) so the
  golden corpus stratifies across the codec-corpus categories.

Goldens regenerator script (`build_goldens.py`) needs to grow a
distortion-config DSL + a multi-resolution + multi-image pipeline
before this expansion can land cleanly.

**cvvdp-gpu vs pycvvdp perf gap (cuDNN / Burn / cubek):**

User suggestion (2026-05-15):

> Burn is a libtorch alternative so we should be able to beat
> pycvvdp on GPU — maybe we didn't update to the latest cubecl
> 0.10 release or use the best algorithms in cubek?

Current state:
- cubecl pin: `0.10.0-pre.4` (per workspace Cargo.lock). The
  cubek (`tracel-ai/cubek`) high-level kernel library at
  `cubecl-kernels` exposes well-optimised matmul, conv, reduce.
- pycvvdp's hot path is the downscale/upscale Gaussian pyramid
  — pure depthwise separable convolution. PyTorch routes this
  via cuDNN, which has hand-tuned per-arch kernels.
- The cubek conv kernel (depthwise 5-tap, shared-memory tiled)
  would close the gap if it matches cuDNN. We currently do not
  use cubek conv — our `downscale_kernel` /
  `upscale_v_kernel` / `upscale_h_kernel` are hand-rolled 5-tap.

Investigation queued: try replacing the downscale/upscale
kernels with cubek-conv calls and re-measure. If cubek-conv
holds parity (separable filter, ceil-div boundaries) and lands
≤ pycvvdp at 12 MP, that's our path to "beat libtorch".

### Investigation Notes (cvvdp-gpu, post-tick-81)

These observations don't ship as code, but they document
findings that would otherwise be re-discovered:

- **Standalone weber(dist) is not slower than weber(ref)** —
  the consecutive-weber diagnostic in `examples/time_12mp.rs`
  shows two back-to-back `compute_dkl_weber_pyramid` calls on
  the same `ref_bytes` complete in nearly identical time. The
  "weber(dist) is 2× weber(ref)" effect observed inside
  `compute_dkl_d_bands` is therefore not algorithmic, not a
  cubecl warm-up artifact, and not driver thermal throttling.
  It is host memory pressure: ~190 MB of `ref_weber` Vec stays
  alive across the second call.
- **Tick 85's failed 5-phase d_bands refactor regressed
  standalone weber by 5×** (260 ms → 1300 ms) — the per-band
  bisect ruled out: (a) the new `self.ref_log_l_bkg` field
  itself (allocation-only does not regress), (b) the new
  `log_l_bkg_dest` parameter on `_dispatch_weber_pyramid_gpu`,
  and (c) the GPU memory-handle pattern (the dist-side CSF
  optimization above confirms this). The proven cause is the
  5-phase serial control-flow structure (all CSF(ref) bands →
  weber(dist) → all CSF(dist) bands → all masking), but the
  actual mechanism (cubecl sync barrier? memory-pool
  fragmentation? kernel-scheduler ordering?) remains unknown.
  Future attempts at the d_bands restructure should bisect a
  different axis (interleaved-per-level vs. phase-serial)
  rather than re-flatten the existing structure.

Net 12 MP performance trajectory (CUDA, RTX-class):

| metric                          | tick 64   | tick 73    | tick 171   |
| ----                            | ----      | ----       | ----       |
| weber pyramid (1 side)          | 103 ns/px | 21.6 ns/px | 18.7 ns/px |
| compute_dkl_d_bands             | 428 ns/px | 121 ns/px  | 33.7 ns/px |
| compute_dkl_jod (cold REF)      | 444 ns/px | 127 ns/px  | **36.1 ns/px** |
| compute_dkl_jod_with_warm_ref   | —         | —          | **20.6 ns/px** |

### Honest comparison against the canonical reference (tick 173)

The fcvvdp ratios cited in earlier rows compare against
`halidecx/fcvvdp` — a separate C+Zig fork, not the canonical
pycvvdp at `gfxdisp/ColorVideoVDP`. Direct pycvvdp v0.5.4
CUDA measurement on the same RTX 5070 host:

| metric                          | per-pixel  | vs pycvvdp CUDA |
| -----                           | ----       | ----            |
| **pycvvdp v0.5.4 (CUDA)**       | **14 ns/px** | baseline        |
| cvvdp-gpu cold                  | 36.1 ns/px | **2.58× slower** |
| cvvdp-gpu warm-ref              | 20.6 ns/px | **1.47× slower** |

pycvvdp benefits from cuDNN-optimised separable convolution on
the downscale/upscale pyramid; our cubecl kernels are hand-written
5-tap separable. cvvdp-gpu wins on portability (WGPU + HIP
backends, ~50 MB static binary vs ~3 GB PyTorch runtime, ~1 s
warm-up vs 1-13 s graph compile) but loses on raw CUDA throughput.

See `crates/cvvdp-gpu/benchmarks/pycvvdp_12mp_cuda_2026-05-14.md`
+ `scripts/cvvdp_goldens/bench_12mp_cuda.py` for the
reproduction recipe.

### vs fcvvdp (separate C+Zig fork, NOT the canonical reference)

fcvvdp's published 360p bench (i7-13700k):

| fcvvdp variant | per-pixel  | vs cvvdp-gpu cold @ 12 MP | vs cvvdp-gpu warm @ 12 MP |
| ----           | ----       | ----                       | ----                       |
| 1-thread       | 214 ns/px  | cvvdp-gpu **5.93× faster** | cvvdp-gpu **10.39× faster** |
| 8-thread       |  86 ns/px  | cvvdp-gpu **2.38× faster** | cvvdp-gpu **4.17× faster**  |

The fcvvdp comparison is real (numbers measured, ratios correct)
but **fcvvdp is not pycvvdp**. Use the pycvvdp row for the
canonical comparison.

### Fixed

#### cvvdp-gpu

- `host_scalar::predict_jod_still_3ch` index-out-of-bounds at
  image sizes where `band_frequencies` truncates below
  `ilog2(min(w, h))` (e.g. 1024×1024). The auto-pick now queries
  `band_frequencies(...).len()` instead of falling through to the
  `ilog2`-based default.

### Removed

#### cvvdp-gpu

- Dead `masked_diff_kernel` cubecl stub (always wrote 0.0; never
  launched).
- Dead `upscale_kernel` cubecl stub (replaced by the
  `upscale_v_kernel` + `upscale_h_kernel` pair).
- Empty `kernels::reduce` module (planned scope landed in
  `kernels::pool` instead).

#### zen-metrics-cli

- New `cvvdp` metric (`--metric cvvdp`). GPU bundle (`--features
  gpu`) now includes `gpu-cvvdp`. Sweep TSVs pick up the
  `score_cvvdp` column automatically.

### Workspace

- CI builds the new `cvvdp-gpu` crate alongside the existing four
  `-gpu` crates under `wgpu` (per-platform) and as part of the
  `i686-unknown-linux-gnu` cross-compile sanity job.
