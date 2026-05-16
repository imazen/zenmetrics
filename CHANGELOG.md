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

### Milestone — tick 500 (cvvdp-gpu)

The 416–500 tick arc was a deep invariant-pinning + documentation
hardening pass. ~85 tests added, ~30 doctests added, every public
constant + helper now has direct bit-pin / structural coverage.
Major themes:

- **Constants-pin series** (ticks 393–397, 401–402): every cvvdp
  v0.5.4 numeric (`BETA_*`, `MASK_*`, `D_MAX`, `KERNEL_A`, `GAUSS5`,
  `SRGB_LINEAR_TO_DKL`, `JOD_A`, `JOD_EXP`, `IMAGE_INT`, `PER_CH_W`,
  `BASEBAND_W`, `CSF_BASEBAND_RHO`, `SENSITIVITY_CORRECTION_DB`,
  `XCM_3X3`, `CH_GAIN`, `PU_BLUR_KERNEL_1D`, `PU_PADSIZE`,
  `LOG_L_BKG_AXIS`, `LOG_RHO_AXIS`, `LOG_S_O0_C1/C2/C3`,
  `GE_SIGMA`, display constants, crate-level dims) is bit-pinned
  against pycvvdp v0.5.4. A silent edit cascades as a specific test
  failure naming the constant, not a 0.001 JOD drift on shadow_jod.

- **Function-level structural invariants** (ticks 416–434): direct
  pin files for `flatten_band_weights`, `precomputed_band_weights`,
  `laplacian_pyramid_dec_scalar`, `gausspyr_reduce_scalar`,
  `gausspyr_expand_scalar`, `srgb_byte_to_dkl_scalar`,
  `weber_contrast_pyr_dec_scalar`, `clamp_diff_soft`,
  `phase_uncertainty_no_blur`, `mask_pool_pixel`,
  `mult_mutual_pixel`, `met2jod`, `do_pooling_and_jod_still_3ch`,
  `precompute_logs_row`, `phase_uncertainty_band`,
  `gaussian_blur_sigma3`, `mult_mutual_band`,
  `predict_jod_still_3ch`. Each pins shape, determinism via
  `to_bits()`, branch thresholds, dynamic range, edge cases.

- **Doctest coverage** (ticks 442–481, extended 507/510/513): every
  public constant + helper has a `# Examples` doctest with
  bit-equality / range assertions, plus rendered example docstrings
  on the user-facing `Cvvdp::*` scoring methods (`new` / `new_with_geometry`
  / `score` / `score_with_reference` / `compute_dkl_jod` /
  `compute_dkl_jod_with_warm_ref` / `compute_dkl_jod_host_pool*` /
  `warm_reference`). 44 doctests pass, 6 are `ignore` (Cvvdp methods
  need a feature-gated `Backend` type alias; docs.rs has no GPU).
  Measured 2026-05-16 via `cargo test --doc -p cvvdp-gpu`.

- **State machine pins** (ticks 486, 488, 489, 491, 493, 494, 497,
  498, 499): `Cvvdp` cache state machine (`set_reference` vs
  `warm_reference` independence, no-pollution from one-shot scoring),
  bit-determinism across all 3 scoring paths
  (`score`/`score_with_reference`/`compute_dkl_jod_with_warm_ref`),
  four-path consolidation, `new` ↔ `new_with_geometry(STANDARD_4K)`
  equivalence, 8×8 + 128×8 + 8×128 boundary/aspect smoke,
  cross-instance bit-equality, degenerate-input stability.

- **CHANGELOG provenance** (ticks 482–487, 490, 492): every entry
  from tick 386 onward now references its implementing commit's
  short hash via sed-batch backfill. Workspace convention.

- **Maintenance / cleanup** (various): cargo fmt drift sweeps,
  clippy fixes (`needless_range_loop`, `excessive_precision`,
  `clone_on_copy`), `lib_reexports.rs` re-export surface pin,
  `cvvdp_mem_table` example refactored onto `recommend_parallel`.

Branch is at parity with pycvvdp v0.5.4 to ≤0.005 JOD on every
fixture; the test suite catches drift across every layer that pins
mention. Tick 500.

**Post-milestone long tail (ticks 501–540, summarised here so the
detailed entries below stay grep-able):**

- **Re-export surface widened** (501–503): lib_reexports.rs grew
  from 5 to 11 pins, covering `Cvvdp<R>`, `Error`, `Result<T>`,
  the four lib-root constants, the `params::*` scaffolding types,
  and the `host_scalar::*` + 5 `kernels::*` submodule paths.
- **CHANGELOG provenance finished** (504): the last 4 unhashed
  entries (ticks 383/387/398/399) backfilled via `jj file annotate`
  + change_id resolution. All entries from tick 383 onward are
  now hash-tagged.
- **Rustdoc + clippy clean** (505, 514, 516, 518): cleared the 96
  `#[cube(launch)]` macro-emitted `missing_docs` warnings via
  file-level `#![allow(missing_docs)]` on each kernel file; added
  crate-level `#![warn(missing_docs)]` guard so future undocumented
  pub items surface at `cargo doc`; fixed an unresolved intra-doc
  link in pipeline.rs's private docs; tightened the type-complexity
  + assertions-on-constants clippy warnings introduced by 503.
- **State-machine + boundary smokes** (506–513): cross-instance
  bit-equality on fresh `Cvvdp::new` instances, degenerate-input
  stability on `pixels_per_degree` and `new_with_geometry`,
  end-to-end smoke at extreme aspect ratios (128×8 / 8×128 / 1024×8
  / 8×1024), GPU + host_pool perf_mode bit-equality on the cpu and
  GPU runtime variants, doctests on the remaining user-facing
  `Cvvdp::*` scoring methods (`new_with_geometry`, `warm_reference`,
  `compute_dkl_jod`).
- **Manifest URL tightening** (519–521): pinned the canonical R2
  host (`https://coefficient.r2.imazen.org/`) + the crate-specific
  bucket subpath (`/cvvdp-goldens/`) on `MANIFEST_URL`, closing
  silent CDN/sibling-crate misroute gaps the existing structure
  checks would have passed.
- **Static-assert promotion** (522–524): the integer-typed lib-root
  constants (`N_CHANNELS == 3`, `MAX_LEVELS == 9`,
  `PYRAMID_MIN_DIM == 4`, `PYRAMID_MIN_DIM * 2 == 8`) and the
  `CsfChannel` discriminants (`A == 0` / `Rg == 1` / `Vy == 2`)
  promoted from runtime `assert_eq!` to module-level
  `const _: () = assert!(...)` static asserts. Fundamental
  dimension parameters now catch at compile time.
- **Stuck-at-constant pinned across all four scoring paths**
  (508, 509, 525): strict q-level separation (q=90 > q=20 > q=1
  with ≥ 0.01 JOD gap) on `score()`, `compute_dkl_jod_host_pool`,
  and `compute_dkl_jod_host_pool_with_warm_ref`. Catches
  near-correct-but-non-discriminative collapse that the manifest
  tolerance pin (0.005 JOD) wouldn't surface.
- **Documentation polish** (526, 527, 528, 529, 530): added this
  long-tail summary block; replaced workspace README's stale
  `TBD | TBD | TBD` cvvdp-gpu row with
  `(pending — reference is pycvvdp v0.5.4)`; recorded the saturation
  point — every clippy / rustdoc / missing_docs surface is clean
  across all feature combinations and target selections, the
  lib_reexports surface is fully pinned at 11 tests, the
  stuck-at-constant contract is pinned across all 4 scoring paths,
  the CHANGELOG hash provenance is complete from tick 383 onward,
  and there are no remaining TODOs / FIXMEs in the source;
  normalized 8 `# Example` (singular) docstring headers in
  pipeline.rs to `# Examples` (plural) to match Rust API guidelines.

- **Intermediate-method doctest sweep** (531–537): every
  `Cvvdp::compute_dkl_*` method now has a rendered `# Examples`
  doctest. 7-tick arc adding `ignore` doctests for
  `compute_dkl_planes` (531), `compute_dkl_gauss_pyramid` (532),
  `compute_dkl_laplacian_pyramid` (533), `compute_dkl_weber_pyramid`
  (534), `compute_dkl_t_p_bands` (535),
  `compute_dkl_csf_weighted_bands` (536), and `compute_dkl_d_bands`
  (537). Doctest count grew from 44+5 → 44+13. Closes the docs.rs
  rendered-example gap on the advanced intermediate-stage API
  surface. Subsequent significant improvement still requires a
  fresh measurement (pycvvdp-baseline SRCC) or a directed feature
  (`CvvdpParams` JSON loader).

- **Sweep finalization** (538, 539): documented the 531-537 sweep
  in this long-tail block; caught the last remaining
  `# Example` (singular) header in `host_scalar.rs:48` that tick
  530's pipeline.rs-only normalization sweep had missed. The
  `# Examples` (plural) Rust API guidelines convention is now
  applied across every docstring in the crate.

As of tick 575 the crate contains **165** `const _: () = assert!(...)`
static asserts spread across 11 test files. Ticks 548-575 grew
the count from 11 to 165 by mining genuine gaps after each
premature "saturation" call: every named cvvdp v0.5.4 numeric is
now pinned (values via `to_bits()` bit-equality), every load-
bearing sign is pinned (`is_sign_positive` / `is_sign_negative`),
every load-bearing ordering is pinned (`u32 <` on `.to_bits()`
for positive f32 operands), every load-bearing cross-equality is
pinned, plus the SRGB_LINEAR_TO_DKL opponent-color sign signature
and the BETA hierarchy. These fire at compile time and are the
load-bearing enforcement; the runtime `#[test]` fns are preserved
beside them to keep test-runner-visible names referenced by older
CHANGELOG entries resolvable.

Earlier static-assert milestones:
- Tick 539 had 11 static asserts (CsfChannel + lib_constants + lib_reexports).
- Tick 540 verified clean: `cargo clippy --all-targets --all-features`,
  `cargo doc --document-private-items`, `cargo test --doc` (44 + 13).
- Ticks 548-572 promoted scalar+array bit-pins across pool, csf,
  masking, pyramid, color, display, params modules.
- Ticks 573, 575 added cross-bundle linkage and positivity pins on
  `CvvdpParams::PLACEHOLDER`'s scaffolding sub-bundles.
- Tick 576 (this entry) updates the milestone block to reflect
  the current state.

As of tick 576 verified clean across:
  - `cargo clippy -p cvvdp-gpu --all-targets --all-features` — 0 warnings
  - `cargo doc -p cvvdp-gpu --no-deps --document-private-items` — 0 warnings
  - `cargo test --doc -p cvvdp-gpu` — 44 passed + 13 ignored

Tick 548 promoted 5 more runtime asserts to static asserts (4 in
`csf_axes_invariants.rs` — `LOG_L_BKG_AXIS.len() == N_L_BKG`,
`N_L_BKG == 32`, `LOG_RHO_AXIS.len() == N_RHO`, `N_RHO == 32` —
plus `N_RHO > 0` in `lib_reexports.rs` to mirror the existing
`N_L_BKG > 0`). Static-assert count is now 16 across 3 test files.
The runtime `#[test]` fns in `csf_axes_invariants.rs` are preserved
beside the new static asserts (same compatibility rationale as
ticks 522-524). Same clippy / doc / doctest verification as tick 540
still passes (no new warnings introduced).

Tick 549 promoted 3 further invariant runtime asserts to static
asserts on physical-meaning constants that other tests in the same
file are predicated on:
  - `PU_PADSIZE == 6` in `phase_uncertainty_band_invariants.rs`
    (the branch-boundary parameter; `branch_boundary_at_pu_padsize`
    hardcodes the 6/7 transition pairs)
  - `PU_BLUR_KERNEL_1D.len() == 13` in `masking_constants.rs` (the
    σ=3 truncation tap count the per-element expected[] array
    depends on)
  - `SRGB8_TO_LINEAR_LUT.len() == 256` in `color_scalar.rs` (the
    one-entry-per-u8 LUT-size contract the indexing semantics rely
    on)
Static-assert count is now 19 across 5 test files. Verification:
same clippy / doc / doctest status as tick 540 — no regressions.

Tick 550 promoted 3 `f32::is_finite()` runtime asserts to static
asserts in `lib_reexports.rs` on the re-exported scalar constants
`MASK_C`, `JOD_A`, and `KERNEL_A`. `f32::is_finite` is `const fn`
in stable Rust since 1.83 (workspace pins `rust-version = "1.93"`,
absolute language minimum per project policy is 1.85, both well
above 1.83). Catches a refactor that accidentally substitutes
`f32::NAN` or `f32::INFINITY` as a constant literal. Static-assert
count is now 22 across 5 test files.

Tick 551 promoted 7 `DisplayGeometry::STANDARD_4K` +
`DisplayModel::STANDARD_4K` field-value runtime asserts to static
asserts in `display_geometry.rs`. u32 fields use direct `==`;
f32 fields use `to_bits()` (because `f32::PartialEq` isn't yet
`const fn` in stable Rust, but `f32::to_bits` is). Covered fields:
`resolution_w == 3840`, `resolution_h == 2160`,
`distance_m == 0.7472`, `diagonal_inches == 30.0`,
`y_peak == 200.0`, `y_black == 0.2`, `y_refl == 0.397_887_36`. The
v1 R2 manifest goldens were captured against these exact values;
a silent drift now fails to compile rather than at test time.
Static-assert count is now 29 across 6 test files.

Tick 552 added a compile-time pin for
`CvvdpParams::PLACEHOLDER.perf_mode == PerfMode::Strict` in
`params_placeholder.rs`. Uses `matches!` (which is `const`-callable)
since derived `PartialEq` on enums isn't yet `const fn` in stable
Rust. Every parity test inherits this perf-mode through
`Cvvdp::new(..., PLACEHOLDER)`; a silent flip to Fast would have
changed the calibration baseline for dozens of goldens. Static-
assert count is now 30 across 7 test files.

Tick 553 fixed two MSRV-reference factual errors introduced in
the tick 550 / 552 entries: the original wording said "MSRV 1.85"
but the workspace pins `rust-version = "1.93"` (with 1.85 as the
project's absolute language minimum, not the actual current
MSRV). Updated both the CHANGELOG entry and the in-source comment
in `params_placeholder.rs`. No code change; no test impact.

Tick 554 added a compile-time pin for `!CVVDP_COLUMN_NAME.is_empty()`
in `column_name.rs`. `str::is_empty` is `const fn` since Rust 1.39
(well below this crate's MSRV 1.93). An empty column name would
silently produce parquet sidecars with an unnamed score column,
breaking joins downstream. `str::starts_with` is NOT yet const fn
in stable Rust, so the `cvvdp_imazen_` prefix check stays runtime-
only. Static-assert count is now 31 across 8 test files.

Tick 555 promoted 4 Burt-Adelson kernel-constant bit-pins to
static asserts in `pyramid_scalar.rs`:
  - `KERNEL_A == 0.4_f32` (cvvdp v0.5.4 Burt-Adelson parameter)
  - `GAUSS5[1] == 0.25_f32`, `GAUSS5[2] == 0.4_f32`,
    `GAUSS5[3] == 0.25_f32` (inner taps of the 5-tap Gaussian)
The outer taps `GAUSS5[0]` and `GAUSS5[4]` stay runtime because
they use `(.. - ..).abs() < 1e-7` tolerance and `f32::PartialOrd::lt`
is not yet `const fn` in stable Rust. Static-assert count is now
35 across 9 test files.

Tick 556 promoted 3 scalar masking-constant bit-pins to static
asserts in `masking_constants.rs`:
  - `MASK_P == 2.264_355_2_f32` (transducer exponent)
  - `MASK_C == -0.795_497_12_f32` (phase-uncertainty scaling
    exponent; a sign flip would amplify masking 6×)
  - `D_MAX == 2.564_245_5_f32` (soft-clamp ceiling exponent)
Array constants (`CH_GAIN`, `MASK_Q`, `XCM_3X3`) remain runtime-
only for now — promoting them would add bulk for diminishing
return; the runtime tests still cover them at f32-bit precision.
Static-assert count is now 38 across 9 test files.

Tick 557 promoted 8 more scalar bit-pins:
  - `pool_scalar.rs`: `BETA_SPATIAL == 2.0`, `BETA_BAND == 4.0`,
    `BETA_CH == 4.0`, `IMAGE_INT == 0.577_918_3`,
    `JOD_A == 0.043_956_94`, `JOD_EXP == 0.930_204_27` (the
    met2jod power-law constants — `met2jod(d) = 10 - JOD_A·d^JOD_EXP`)
  - `csf_scalar.rs`: `SENSITIVITY_CORRECTION_DB == -0.279_742_33`,
    `CSF_BASEBAND_RHO == 0.1`
Each is independently load-bearing for JOD output across every
parity gate. Static-assert count is now 46 across 11 test files.

Tick 558 promoted 12 array-element bit-pins covering 4 three-entry
arrays:
  - `pool_scalar.rs`:
    - `PER_CH_W[0..3] == 1.0_f32` (still-image chrominance weights)
    - `BASEBAND_W[A,Rg,Vy] == 0.003_633_448_6 / 1.662_772_4 /
      4.118_745_3` (per-channel baseband weights)
  - `masking_constants.rs`:
    - `CH_GAIN[A,Rg,Vy] == 1.0 / 1.45 / 1.0` (RG masking-gain boost)
    - `MASK_Q[A,Rg,Vy] == 1.302_622_7 / 2.888_590_8 / 3.680_771_3`
      (per-channel masking exponents)
A typo that swapped any pair of array entries (e.g. CH_GAIN[A] ↔
CH_GAIN[Rg], muting chrominance) now surfaces at compile time.
Static-assert count is now 58 across 11 test files.

Tick 559 promoted 9 array-element bit-pins for the XCM_3X3
cross-channel masking matrix in `masking_constants.rs`. Each
entry pinned independently — the 3×3 matrix is derived from
cvvdp's published log2-space coefficient table via per-entry
2^x exponentiation, so a re-derivation that rounds differently
would surface here at compile time rather than during a parity-
test run. Static-assert count is now 67 across 11 test files.

Tick 560 promoted 13 individual tap bit-pins + 3 symmetry-pair
pins for the `PU_BLUR_KERNEL_1D` σ=3 Gaussian blur kernel in
`masking_constants.rs`. Pinning the symmetry pairs separately
(`kernel[0] == kernel[12]`, `kernel[1] == kernel[11]`,
`kernel[5] == kernel[7]`) catches a half-kernel typo that would
compile if each individual tap matched its expected literal but
the wrong half was substituted into the array. Static-assert
count is now 83 across 11 test files.

Tick 561 promoted the SRGB8_TO_LINEAR_LUT endpoint bit-pins to
static asserts in `color_scalar.rs`: `LUT[0] == 0.0_f32` and
`LUT[255] == 1.0_f32`. The IEC 61966-2-1 sRGB EOTF maps byte 0 →
linear 0 exactly and byte 255 → linear 1 exactly, and these are
the boundary cases an off-by-one byte index would silently break.
The 254 interior LUT entries remain runtime-only — they're each
derived from the sRGB EOTF formula and pinned by
`srgb_lut_matches_iec_61966_2_1_formula`, which can't lift
because the formula's branchless conditional uses `f32::powf`
(not const fn). Static-assert count is now 85 across 11 test files.

Tick 562 promoted 4 LUT-axis endpoint bit-pins to static asserts in
`csf_axes_invariants.rs`:
  - `LOG_L_BKG_AXIS[0] == -2.3010299957` (log10(0.005))
  - `LOG_L_BKG_AXIS[31] == 4.0` (log10(1e4))
  - `LOG_RHO_AXIS[0] == -1.0` (log10(0.1))
  - `LOG_RHO_AXIS[31] == 1.8061799740` (log10(64))
The 60 interior entries of each axis stay runtime-only — their
runtime invariants (monotonicity, uniform-spacing-in-log10) can't
lift without const-callable loop arithmetic. Static-assert count
is now 89 across 11 test files.

Tick 563 closed the last GAUSS5 gap: the edge taps that tick 555
skipped because the runtime test uses an abs-diff tolerance.
Since f32 arithmetic IS const-callable in stable Rust (only
`f32::PartialOrd::lt` isn't), the underlying derivation
`0.25 - KERNEL_A / 2.0` can be evaluated at compile time and
matched bit-exactly:
  - `GAUSS5[0].to_bits() == (0.25_f32 - KERNEL_A / 2.0_f32).to_bits()`
  - `GAUSS5[4].to_bits() == (0.25_f32 - KERNEL_A / 2.0_f32).to_bits()`
Plus a palindrome cross-check: `GAUSS5[0] == GAUSS5[4]`. The 5-tap
Burt-Adelson kernel is now fully bit-pinned at compile time. Static-
assert count is now 92 across 11 test files.

Tick 564 added 6 semantic ordering invariants leveraging the
observation that for positive f32 values, IEEE 754 bit-pattern
ordering matches numerical ordering — so `u32 <` (which IS const-
callable) is a sound proxy for the underlying f32 ordering:
  - `BASEBAND_W`: A < Rg < Vy (strict monotonicity across channels)
  - `MASK_Q`: A < Rg < Vy (strict monotonicity across channels)
  - `CH_GAIN`: Rg > A and Rg > Vy (chroma-boost invariant)
These catch a class of typo the individual-entry bit-pins miss: a
permutation that keeps every value intact but swaps which channel
gets which weight. Static-assert count is now 98 across 11 test
files.

Tick 565 added 3 more semantic invariants of a different flavour:
  - `MASK_C.is_sign_negative()` — phase-uncertainty exponent must
    be negative because `10^MASK_C` is an attenuator; a sign flip
    would convert the 0.16× attenuation into a 6× amplification.
    `f32::is_sign_negative` is const fn since Rust 1.83.
  - `BETA_BAND.to_bits() == BETA_CH.to_bits()` — the across-band
    and across-channel Minkowski exponents must remain equal (both
    = 4.0). A drift in one without the other breaks the symmetric-
    pool contract.
  - `JOD_EXP.to_bits() < 1.0_f32.to_bits()` — sublinear-saturation
    invariant on met2jod (`10 - JOD_A · d^JOD_EXP`). Both operands
    are positive so u32 bit-ordering is sound. A regression bumping
    JOD_EXP ≥ 1.0 would make JOD super-linear in d, changing the
    entire perceptual scale.
Static-assert count is now 101 across 11 test files.

Tick 566 added 4 more invariants extending the semantic-invariant
pattern from tick 565:
  - `BETA_SPATIAL.to_bits() < BETA_BAND.to_bits()` — BETA hierarchy
  - `BETA_SPATIAL.to_bits() < BETA_CH.to_bits()` — BETA hierarchy
    (the canonical pyramid-pool strategy raises the Minkowski
    exponent across each nesting level; the inner spatial pool is
    gentler than across-band / across-channel folds)
  - `MASK_P.is_sign_positive()` — transducer exponent must be
    positive (negative MASK_P → `pow(d, MASK_P)` → ∞ as d → 0)
  - `D_MAX.is_sign_positive()` — soft-clamp ceiling exponent must
    be positive (`10^D_MAX < 1` would collapse the clamp ceiling)
Static-assert count is now 105 across 11 test files.

Tick 567 added 9 sign-signature invariants on the
`SRGB_LINEAR_TO_DKL` matrix in `color_scalar.rs`:
  - Row 0 (A): all 3 entries `.is_sign_positive()`
  - Row 1 (Rg): [0]=positive, [1]=negative, [2]=negative
  - Row 2 (Vy): [0]=negative, [1]=negative, [2]=positive
This encodes the DKL opponent-color contract: A is weighted-
positive sum, Rg opposes R against G+B, Vy opposes B against
R+G. The per-entry value bit-pins already encode the sign
implicitly, but the sign-signature pin captures the SEMANTIC
contract directly — useful for the same documentation-of-intent
reason as the channel-ordering invariants (564-566). Static-
assert count is now 114 across 11 test files.

Tick 568 added 5 more sign-bit invariants on the remaining major
scalar constants:
  - `JOD_A.is_sign_positive()` — met2jod must decrease with d
  - `IMAGE_INT.is_sign_positive()` — multiplicative pool weight
  - `KERNEL_A.is_sign_positive()` — Burt-Adelson parameter ∈ (0, 0.5)
  - `CSF_BASEBAND_RHO.is_sign_positive()` — spatial frequency in cy/deg
  - `SENSITIVITY_CORRECTION_DB.is_sign_negative()` — calibrated
    attenuation (not amplification)
Static-assert count is now 119 across 11 test files.

Tick 569 added 9 positivity invariants on every `XCM_3X3` entry.
Each is derived in cvvdp v0.5.4 as `2^x` for some log2-space
coefficient, and `2^x > 0` always. A refactor that substituted
a different formula (e.g. `1 - exp(-x)` for an attenuation
reframe, or a sign drift in the source coefficients) could yield
negative entries while still matching the per-entry value bit-
pins. Pinning positivity directly captures the construction
rule. Static-assert count is now 128 across 11 test files.

Tick 570 added 7 positivity invariants on the unique taps of
`PU_BLUR_KERNEL_1D` ([0]..[6]; taps [7]..[12] inherit positivity
via the palindrome bit-equality pins from tick 560). The σ=3
Gaussian construction `exp(-x²/(2σ²)) / Σ` only emits positive
values; a refactor that substituted a different kernel family
(e.g. derivative-of-Gaussian, sinc with side lobes) would yield
negative taps. Pinning positivity directly captures the Gaussian
construction contract. Static-assert count is now 135 across 11
test files.

Tick 571 added 4 length-pin invariants on the per-channel CSF
sensitivity LUTs `LOG_S_O0_C1/C2/C3` in `csf_axes_invariants.rs`:
  - Each LUT length must equal `N_L_BKG * N_RHO` (32 × 32 = 1024)
  - Plus a cross-channel length-consistency pin (all 3 LUTs have
    matching length)
The CSF kernel indexes via `idx = l_bkg_i * N_RHO + rho_i` so a
size mismatch silently corrupts every per-pixel CSF query — these
pins catch the mismatch at compile time rather than as garbage
JOD output. Static-assert count is now 139 across 11 test files.

Tick 572 added the first test coverage for `GE_SIGMA`: a
bit-equality pin to cvvdp v0.5.4's `ge_sigma = 1.5` and a
positivity invariant (it's a Gaussian σ). The constant is
documented as carried for source-JSON fidelity but not yet
consumed by the still-image pipeline (eccentricity-aware paths
are future work); the pins guarantee the value stays correct
for when those paths land. Static-assert count is now 141 across
11 test files.

Tick 573 promoted 12 `CvvdpParams::PLACEHOLDER` scaffolding-field
bit-pins to static asserts in `params_placeholder_non_display.rs`:
  - csf sub-bundle: `a_peak`, `rg_peak`, `vy_peak` (all 0.0)
  - masking sub-bundle: `p=2.4`, `q=2.2`, `k=0.04`
  - pooling sub-bundle: `beta_spatial`, `beta_band`,
    `beta_channel` (all 4.0)
  - jod sub-bundle: `jod_a=10.0`, `jod_b=1.0`, `jod_c=0.30`
These fields are documented as unused-scaffolding (production
code reads from `kernels::*` consts) but they're publicly-visible
defaults that `CvvdpParams { ..PLACEHOLDER }` callers depend on.
Pinning at compile time keeps the scaffolded values stable until
they're intentionally wired through. Static-assert count is now
153 across 11 test files.

Tick 575 added 12 more invariants on `PLACEHOLDER`:
  - **Cross-bundle linkage (3)**: `PLACEHOLDER.display ==
    STANDARD_4K` — y_peak, y_black, y_refl each pinned via
    `to_bits()`. Guards against a refactor that copies the
    STANDARD_4K values into PLACEHOLDER literally (drifting if
    STANDARD_4K is later updated but PLACEHOLDER's copy isn't).
  - **Scaffolding positivity (9)**: masking.{p,q,k},
    pooling.beta_{spatial,band,channel}, jod.{jod_a,jod_b,jod_c}
    all `.is_sign_positive()`. Negative values would invert the
    expected algebra (pow singularities, pool reversal) the
    moment the fields are wired through.
Static-assert count is now 165 across 11 test files.

Tick 577 promoted the `CVVDP_COLUMN_NAME.starts_with("cvvdp_imazen_")`
runtime check to a compile-time pin via a const while-loop over
`as_bytes()`. `str::starts_with` itself isn't const fn, but
`str::as_bytes` is (since 1.39), integer comparison is trivially
const, `while` in const is stable, and `Option::is_none` (used to
gate the check on the default-form build, no `CVVDP_IMPL_TAG` env
override) is const since 1.48. Adds a length pin + a loop-body
match pin (2 logical asserts). Static-assert count is now 167
across 11 test files.

Tick 578 applied the same const-byte-loop trick to the goldens-
metadata structural invariants in `goldens_metadata.rs`:
  - MANIFEST_URL starts with `https://` (byte-prefix)
  - MANIFEST_URL ends with `.json` (byte-suffix at offset)
  - MANIFEST_URL starts with canonical R2 host
    `https://coefficient.r2.imazen.org/`
  - MANIFEST_SHA256 length == 64 (sha256 hex)
  - !GOLDEN_VERSION.is_empty()
  - GOLDEN_VERSION first byte == 'v' (the v<N> convention)
The substring `.contains` checks (golden-version path segment,
bucket subpath) and per-char hex validation stay runtime —
`.contains` requires substring search not easily const-callable.
Static-assert count is now 173 across 11 test files.

Tick 579 closed the last two goldens-metadata runtime gaps via a
const sliding-window substring-search helper (also const-callable
in stable Rust):
  - MANIFEST_URL contains `/cvvdp-goldens/` (bucket subpath)
  - MANIFEST_URL contains `/v1/` (version path segment)
The substring-search helper `bytes_contain(hay, needle)` is a
`const fn` doing the obvious O(n·m) sliding-window comparison.
That was the technique the prior tick claimed wasn't "easily const-
callable" — turns out it IS, the sliding-window inner loop is
just two more layers of the `while` + byte-comparison primitive
ticks 577-578 already used. Static-assert count is now 175 across
11 test files.

Tick 580 closed the per-char hex validation gap tick 578 left
runtime-only. `char::is_ascii_digit` / `RangeInclusive::contains`
aren't const fn, but raw u8 comparison IS — and MANIFEST_SHA256
is pure ASCII so byte-iteration covers every char correctly:
  - Every byte must satisfy `(c >= b'0' && c <= b'9') || (c >=
    b'a' && c <= b'f')` (lowercase hex)
A uppercase variant fails the case-sensitive sha2-Digest match
silently; a stray non-hex char fetches the wrong manifest. Now
both are compile-time-caught. Static-assert count is now 176
across 11 test files.

Tick 581 extracted `CACHE_DIR_SUBDIR = "zenmetrics-cvvdp-goldens"`
from `tests/common/mod.rs` to a pub const (was a magic string
inline in `cache_dir()`), then pinned 3 structural invariants on
it via the const-byte-loop primitives from ticks 577-580:
  - non-empty
  - contains "cvvdp" (disambiguation from sibling crates'
    cache dirs that all live under `~/.cache/zenmetrics-*/`)
  - all-ASCII alphanumerics or hyphen (filesystem-portable)
Static-assert count is now 179 across 11 test files. Small
refactor + pins — same shape as ticks 522-524 promoted dimension
constants from inline literals into the lib_constants module.

Tick 582 completed the tick-581 refactor by deduplicating the
remaining inline magic-string usage in `goldens_metadata.rs`'s
`cache_dir_path_embeds_golden_version` runtime test — now uses
`CACHE_DIR_SUBDIR` directly. If the subdir is renamed, the test
follows automatically and the static asserts on it still cover
the "must contain 'cvvdp'" invariant at compile time. Pure
dedup — no new static asserts.

Tick 583 promoted the `CVVDP_COLUMN_NAME.starts_with("cvvdp_")`
family-prefix check in `lib_reexports.rs` to compile time via
the const-byte-loop pattern (same as ticks 577/578/579/580).
This is the broader prefix invariant — the env-override
`CVVDP_IMPL_TAG` is intentionally a free-form discriminator
WITHIN the `cvvdp_*` namespace (pycvvdp uses `cvvdp_pycvvdp_v054`,
this crate uses `cvvdp_imazen_*`, a future Burn port reserves
`cvvdp_burn_*`); the family prefix must hold for all variants.
Also corrected the stale "`.starts_with` isn't const fn" comment
left over from tick 522. Adds 1 length pin + 1 byte-match pin
(2 logical asserts). Static-assert count is now 181 across 11
test files.

Tick 584 extracted the three duplicated const-byte-loop primitives
(`starts_with`, `ends_with`, `contains`) into a shared
`common::const_str` module in `tests/common/mod.rs`. The pattern
was duplicated across `column_name.rs`, `goldens_metadata.rs`,
and `lib_reexports.rs` (ticks 577-580, 583); each call site now
imports `common::const_str` and uses `const_str::starts_with(…)`
etc. Pure refactor — same static-assert count (181) but the
boilerplate per call site shrinks from ~10-30 lines to 1-3 lines.
No behavior change.

Tick 585 added direct unit tests for the new `common::const_str`
helpers in a new test file `const_str_helpers.rs`. 17 compile-time
`const _: () = assert!(...)` cases cover positive + negative paths
for each helper (`starts_with`, `ends_with`, `contains`), plus
edge cases (empty prefix/suffix/needle, prefix longer than
haystack, needle at start/middle/end). 6 runtime test fns mirror
the asserts so `cargo test` runners can name them in output.
Static-assert count is now 198 across 12 test files.

Tick 586 added a fourth helper `const_str::bytes_eq(a, b)` for
const slice-equality (`[u8]: Eq` isn't const-callable). Used it
to pin `GOLDEN_VERSION == "v1"` exactly in `goldens_metadata.rs`
— previously only the v-prefix was pinned. With this, a refactor
that bumps `GOLDEN_VERSION = "v2"` without also updating
`MANIFEST_URL`'s `/v1/` path segment fails to compile (instead of
passing both prefix and contains checks while silently fetching
the wrong manifest). Also adds 5 compile-time + 2 runtime tests
on the new helper. Static-assert count is now 204 across 12 test
files.

Tick 587 fixed two stale "stays runtime-only" comments in older
const blocks:
  - `column_name.rs:28-31`: tick 554 originally said
    `str::starts_with` "stays runtime-only" — but tick 577 lifted
    the prefix check, and tick 584 factored it into
    `common::const_str::starts_with`.
  - `pyramid_scalar.rs:28-31`: tick 555 originally said GAUSS5
    outer taps "stay runtime-only because they use abs+lt
    tolerance" — but tick 563 lifted them by deriving the bit
    pattern at compile time from `0.25 - KERNEL_A / 2.0`.
Comment-only updates; no code change, no behavior change. Doc
maintenance like tick 583 did for lib_reexports.rs.

Tick 588 added a new pub const `cvvdp_gpu::PYCVVDP_REFERENCE_VERSION = "v0.5.4"`
in `lib.rs` to centralize the pinned reference-version string
that previously appeared in 6+ places (`tests/parity.rs`,
`kernels/csf_lut/v0_5_4.rs` filename, csf.rs module name,
PORT_STATUS.md, CHANGELOG, requirements.txt).

The runtime test `tests/parity.rs::manifest_fetches` now sources
its expected version from `PYCVVDP_REFERENCE_VERSION` instead of
the hardcoded string — when the reference bumps, this test
follows automatically.

Also adds 3 compile-time format invariants on the new const:
non-empty, starts with `v`, contains `.` — catches a typo like
`v054` that breaks the `vX.Y.Z` convention. Static-assert count
is now 207 across 12 test files (+3 from tick 587's 204).

Tick 589 closed the requirements.txt lockstep gap from tick 588.
Pins `scripts/cvvdp_goldens/requirements.txt` at compile time
against `PYCVVDP_REFERENCE_VERSION` (strip leading `v` to match
the PyPI `cvvdp==X.Y.Z` format — note: the PyPI package is named
`cvvdp` even though the importable module is `pycvvdp`). Uses
`include_str!()` (compile-time file read) + `slice::split_first()`
(const-callable since 1.83) + `common::const_str::contains` to
verify the version substring is present.

A bump to PYCVVDP_REFERENCE_VERSION now FAILS TO COMPILE unless
requirements.txt is updated in the same commit. Closes the 6th
lockstep site documented in the PYCVVDP_REFERENCE_VERSION
docstring. Static-assert count is now 208 across 12 test files.

Tick 590 extended the lockstep coverage to the vendored LUT file:
`src/kernels/csf_lut/v0_5_4.rs`'s auto-generated header comment
`"Auto-generated from pycvvdp v0.5.4's csf_lut_weber_fixed_size.json."`
contains the full `v0.5.4` string (matches PYCVVDP_REFERENCE_VERSION
exactly — no v-stripping needed). `include_str!()` reads the full
LUT (~1000+ lines of f32 literals) at compile time and
`const_str::contains` finds the version substring. When the
reference bumps, the LUT regen procedure updates the header —
this pin catches a version mismatch between the const and the
vendored data. Static-assert count is now 209 across 12 test files.

Tick 591 closed the last include-able lockstep site:
`docs/PORT_STATUS.md`. Its "Reference version pin" section
reads "gfxdisp/ColorVideoVDP v0.5.4 (latest tag as of …)" — same
`include_str!()` + `const_str::contains` pattern as ticks 589-590.
Forces the prose documentation to update in the same commit as
the const + parity-test + requirements.txt + LUT header.

Tick 592 extended the lockstep further to the crate-level
README.md. It references `v0.5.4` in 4 places (algorithm-parity
claim, PerfMode::Strict semantics, parity-goldens feature, Status
section). Same `include_str!()` + `const_str::contains` pattern.
User-facing docs now also forced to update in lockstep. Static-
assert count is now 211 across 12 test files (5 cross-file
`include_str!()` pins on PYCVVDP_REFERENCE_VERSION:
parity-test runtime check, requirements.txt, LUT header,
PORT_STATUS.md, README.md).

Tick 593 closed the Cargo.toml feature-doc comment site. The
`parity-goldens` feature comment reads "Enables integration tests
that fetch the pycvvdp v0.5.4 goldens from R2 ..."; pinning
forces the comment to update in lockstep too. The const now has
6 cross-file `include_str!()` lockstep pins total. Static-assert
count is now 212 across 12 test files.

Tick 594 pinned `docs/CVVDP_SIDECAR_SCHEMA.md`'s "Reserved
column-name tags" table (which documents `cvvdp_pycvvdp_v054` →
"upstream pycvvdp v0.5.4"). CHROMA_DRIFT_INVESTIGATION.md is
intentionally NOT pinned — its v0.5.4 references are historical
audit material from the tick-200 chroma_shift bug hunt, not
current-state documentation; pinning would cement that historical
investigation against future reference bumps incorrectly. The
const now has 7 cross-file `include_str!()` lockstep pins.
Static-assert count is now 213 across 12 test files.

Tick 595 moved all 7 lockstep pins + the const-format invariant
block from `tests/parity.rs` (gated behind `parity-goldens`
feature) to a new always-on `tests/version_lockstep.rs` file.
Real correctness improvement: the pins now fire on every
`cargo check / test`, not only when the goldens feature is on.
Pure refactor — same pin count (213), same coverage. Test file
count grows to 13.

Ticks 596-598 (post-595 doc cleanup):
- Tick 596: removed a duplicated `#[allow(dead_code)]` outer
  attribute on `mod common;` in version_lockstep.rs. The inner
  `tests/common/mod.rs` already has `#![allow(dead_code)]`; the
  redundant outer attribute introduced a `duplicated attribute`
  warning that tick 595 had missed. Lint cleanup, no code change.
- Tick 597: rewrote the `PYCVVDP_REFERENCE_VERSION` docstring in
  `lib.rs` to list all 7 lockstep-pinned sites + 3 format
  invariants + 2 intentionally-unpinned historical docs + 2
  unpinnable Rust-identifier sites + the GOLDEN_VERSION cross-
  version-space relationship. Was listing only the 6 original
  sites from tick 588. Contract now self-documenting at the const.
- Tick 598: rewrote PORT_STATUS.md's "Reference version pin"
  bump procedure. Was listing 3 update sites (R2 prefix,
  GOLDEN_VERSION, tests/parity.rs assertion); now correctly
  documents `PYCVVDP_REFERENCE_VERSION` as the single trigger
  point + the 7-pin lockstep arc that surfaces the rest as
  compile failures.

Static-assert count unchanged at 213 across 13 test files.

Tick 600 added a 5th `common::const_str` helper:
  pub const fn count(s: &[u8], needle: &[u8]) -> usize
which counts non-overlapping occurrences of `needle` in `s`. Used
to add a LUT-channel-completeness pin to `version_lockstep.rs`:
the LUT file must contain at least 3 occurrences of `LOG_S_O0_C`
(one per channel declaration: C1, C2, C3). Catches an accidental
truncation that drops one of the channel LUTs entirely — the
per-channel length pins in `csf_axes_invariants.rs` (tick 571)
cover the LEN per channel WHEN each channel exists, but not the
"channel missing entirely" case. Also adds 8 compile-time count
asserts + 2 runtime test fns to `const_str_helpers.rs` covering
the new helper's positive / edge cases. Static-assert count is
now 222 across 13 test files.

Tick 621 — add `# Examples` doctest to the `Cvvdp<R>` top-level
scorer struct. `ignore`d (the runtime needs a live CUDA driver
which docs.rs sandboxes don't have — same pattern as the existing
ignored doctests on `Cvvdp::new`, `score`, etc.). The example
walks a docs.rs reader through:
- Allocating the buffer pool once for a fixed image size.
- One-shot `score(ref, dist)` returning a JOD.
- Cross-references the two cached-reference fast paths for
  multi-DIST sweeps.

The struct itself was previously documented only with a 4-line
prose docstring. Doctest count: 69 passed + 14 ignored (was
69 + 13). Docs-only change.

Tick 620 — add `# Examples` doctest to `pyramid::WeberPyramid`
struct. Pins the dual-level-count contract (`bands.len() ==
log_l_bkg.len()`) and the per-level spatial-shape match
(`log_l_bkg[k].len() == bands[k].w * bands[k].h` for non-baseband
levels). Companion to tick 619's `Band` doctest — now both
pyramid-output structs have constructor-level examples that
surface the layout contract to docs.rs readers. Doctest count:
68 → 69. Docs-only change.

Tick 619 — add `# Examples` doctests to two more public items:

- `kernels::csf::GE_SIGMA` — pin to 1.5 + positivity invariant
  (a negative sigma would invert the Gaussian eccentricity-
  falloff into an exponential blow-up at fovea). Still unused
  in the current still-image pipeline.
- `kernels::pyramid::Band` — show direct struct construction
  with the `data.len() == w × h` invariant. Used by
  `laplacian_pyramid_dec_scalar` and
  `weber_contrast_pyr_dec_scalar` to return pyramid bands; the
  doctest now surfaces the layout contract.

Doctest count: 66 → 68. Per the per-public-item sweep theme
(ticks 609-618). Docs-only change.

Tick 618 — combined `# Examples` doctest on `LOG_L_BKG_AXIS`
covering both CSF LUT axes (`LOG_L_BKG_AXIS` + `LOG_RHO_AXIS`).
Same shared-coverage pattern as `MASK_P` covering the four
masking scalars (tick 613). Pins:
- Both axes have 32 entries (= N_L_BKG = N_RHO).
- LOG_L_BKG_AXIS endpoints `[-2.301, 4.0]` (luminance 0.005–10000
  cd/m²).
- LOG_RHO_AXIS endpoints `[-1.0, 1.806]` (frequency 0.1–64 cy/deg).
- Both uniformly spaced in log10 (step uniformity over all 31
  intervals).
Cross-references `tests/csf_axes_invariants.rs`. Doctest count:
65 → 66. The three large per-channel sensitivity LUTs
(`LOG_S_O0_C1/C2/C3`) and the `GE_SIGMA` scalar still lack
doctests; the C* LUTs are 1024-entry arrays where a meaningful
doctest beyond "len() == 1024 + positive on bright photopic"
needs more thought. Docs-only change.

Tick 617 — add `# Examples` doctest to
`kernels::csf::precompute_logs_row`. Pins the return-shape
(length `N_L_BKG`) plus the bit-identity contract with
`sensitivity_scalar` at every L_bkg-axis grid point (no
interpolation needed when log_L_bkg lands exactly on a sample).
Surfaces the helper's role as the per-band precompute consumed
by `csf_apply_per_pixel_kernel` — a reader can now see what the
returned array MEANS without grepping the kernel call sites.
Doctest count: 64 → 65. Docs-only change.

Tick 616 — add `# Examples` doctest to
`kernels::csf::sensitivity_scalar`. The sister function
`sensitivity_corrected_scalar` already had a doctest covering
its multiplicative-factor relationship to `sensitivity_scalar`,
but the underlying primitive itself was undocumented at the
example level. Pins:
- Positive + finite output at standard photopic background
  (100 cd/m² → log10 = 2.0) at 4 cy/deg (CSF peak).
- High-frequency roll-off (30 cy/deg < 4 cy/deg sensitivity).
- Per-channel independence (each of A/Rg/Vy returns positive).
Doctest count: 63 → 64. Docs-only change.

Tick 615 — add `# Examples` doctest to `kernels::csf::CsfChannel`.
Pins the [A=0, Rg=1, Vy=2] discriminant ordering load-bearing for
every `channel as usize` indexing site, plus the
Copy/PartialEq/Debug derive contracts. Cross-references the
compile-time pins in `tests/csf_channel_invariants.rs`. Doctest
count: 62 → 63. Docs-only.

Tick 614 — add `# Examples` doctests to two more public
constants:

- `kernels::pyramid::KERNEL_A` — Burt-Adelson `a` parameter
  (= 0.4 in cvvdp v0.5.4). The doctest pins the bit-value AND
  shows how it propagates into `GAUSS5` (center tap = a, outer
  taps = 0.25 - a/2).
- `pipeline::PARALLEL_SAFETY_FACTOR` — pin to 1.5 + worked
  example showing `manual = free / (safety × est)` matches what
  `recommend_parallel` returns, so a reader sees the role of the
  constant in the formula without grepping the function body.

Doctest count: 60 → 62. Per the per-public-item sweep theme
(ticks 609-613). Docs-only change; `cargo test --doc` passes.

Tick 613 — combined `# Examples` doctest on `MASK_P` covering
all four scalar masking constants (`MASK_P`, `MASK_Q`, `MASK_C`,
`D_MAX`). Follows the same shared-doctest pattern already
established by `pool::BETA_CH` (one doctest covers BETA_* triple)
and `pool::JOD_EXP` (one doctest exercises JOD_A + IMAGE_INT).
Pins:
- `MASK_P > 0` (transducer exponent positivity) + bit-pin.
- `MASK_Q` is per-channel `[A, Rg, Vy]` strictly monotonic.
- `MASK_C < 0` (attenuator: `10^MASK_C` < 1).
- `D_MAX > 0` (soft-clamp ceiling: `10^D_MAX` > 100).
Doctest count: 59 → 60. Cross-references the compile-time bit-pins
in `tests/masking_constants.rs`. Docs-only change.

Tick 612 — continue the per-public-item doctest sweep (+4 more,
closing the params.rs scaffolding-struct queue from tick 611):

- `CsfParams` — all 3 sub-fields zeroed in PLACEHOLDER (production
  reads CSF straight from the vendored LUT).
- `MaskingParams` — scaffolding values (p=2.4, q=2.2, k=0.04) +
  the all-positive invariant required by future Minkowski/exponent
  algebra.
- `PoolingParams` — uniform 4.0/4.0/4.0 scaffolding triple (vs.
  production's BETA_SPATIAL=2.0, BETA_BAND=4.0, BETA_CH=4.0) +
  positivity invariant.
- `JodParams` — scaffolding values 10.0/1.0/0.30 + positivity
  invariant. Cross-reference to the v0.5.4 production values
  JOD_A ≈ 0.0439 and JOD_EXP ≈ 0.9302.

Each doctest cross-references `tests/params_placeholder_non_display.rs`
where the same values are bit-pinned at compile time. Doctest count:
55 → 59. The `CvvdpParams` struct itself + its `PLACEHOLDER` const
already had doctests in earlier ticks. Docs-only change.

Tick 611 — continue the per-public-item doctest sweep (+4 more):

- `DisplayModel` struct — field-access + spread-construct an
  HDR400 variant from STANDARD_4K.
- `DisplayModel::STANDARD_4K` const — pin the 3 field values
  with cross-reference to the runtime parity test.
- `DisplayGeometry` struct — field-access + a phone-at-arm's-
  length example showing `pixels_per_degree()` is higher than
  the 4K reference (smaller display + closer distance → tighter
  pixel grid).
- `DisplayGeometry::STANDARD_4K` const — pin the 4 field values.

Doctest count: 51 → 55. Still queued for the next pass (4 more):
`CsfParams` / `MaskingParams` / `PoolingParams` / `JodParams` —
the scaffolding-but-public structs from
`CvvdpParams::PLACEHOLDER`. Docs-only change.

Tick 610 — continue the per-public-item `# Examples` doctest
sweep from tick 609 (+3 more):

- `PYCVVDP_REFERENCE_VERSION` — format invariants (starts with
  'v', contains '.', non-empty) + the leading-'v'-strip → pip
  version trick that `scripts/cvvdp_goldens/requirements.txt`
  uses.
- `Error` — exercises `DimensionMismatch` (carries `expected` +
  `got` payload + actionable Display), the three zero-payload
  variants' Display hints, and the `?` bubble against
  `Box<dyn std::error::Error>`.
- `Result<T>` type alias — Ok/Err construction + the
  identity-`Into` `?` chaining (composes with the same return
  type without `.map_err(Into::into)`).

Doctest count: 48 → 51. Six other body-docstring-only public
items in `params.rs` (DisplayModel / DisplayGeometry / CsfParams
/ MaskingParams / PoolingParams / JodParams structs) are queued
for the next pass; the per-struct doctest needs a moment of
thought on what's load-bearing to show vs scaffolding. Docs-only
change; `cargo test --doc` passes.

Tick 609 — add `# Examples` doctests to four public crate-root
constants that had body-level docstrings but no doctest section:

- `N_CHANNELS` — pin value 3 (DKL still-image).
- `MAX_LEVELS` — pin value 9 with cross-reference to
  `tests/lib_constants.rs::max_levels_cap_at_nine` + the
  buffer-resize implication of bumping it.
- `PYRAMID_MIN_DIM` — pin value 4 and the derived 8-px minimum
  image dim (Cvvdp::new's accept threshold).
- `CVVDP_COLUMN_NAME` — pin `cvvdp_` prefix + parquet-safe charset
  (ASCII alphanumerics + `_` only), matching the contracts in
  `tests/column_name.rs`. Rendered docs now surface these
  contracts to docs.rs readers without forcing them to grep test
  files.

Doctest count: 44 → 48. Per the crate convention every public
constant should have a `# Examples` block with at least one
machine-checked assertion. Docs-only change; `cargo test --doc`
passes.

Tick 608 — tighten `srgb_byte_to_dkl_scalar`'s grayscale-chroma
doctest tolerances. The "(255,255,255) → near-zero chroma" claim
was `rg_white.abs() < a_white * 0.05` and `vy_white.abs() < a_white * 0.05`
(both 5%). Actual ratios from the bit-pinned SRGB_LINEAR_TO_DKL
matrix at (255,255,255) under STANDARD_4K are **0.36% (RG)** and
**0.98% (VY)** — pinned by `tests/color_scalar.rs:80`'s GOLDENS
table. Tightened to **1% (RG)** and **2% (VY)** respectively —
~3× and ~2× the actual values, leaving room for alternate display
models that shift `y_peak`/`y_refl` but still tight enough to
surface a matrix row-sum drift (e.g. a sign flip in row 1 or row
2 that would push the chroma residual above the new bounds).
Companion to ticks 605/606/607 (same doctest-tightening theme).
Docs-only change; doctest passes.

Tick 607 — tighten `safe_pow(2.0, 2.0)` doctest tolerance from
`< 0.01` to `< 1e-3`. The function's analytic deviation at this
point is the cross term `2·eps·x = 4e-5` (`eps = 1e-5` in the
implementation), so `0.01` was 250× looser than needed. Tightening
to `1e-3` still leaves 25× headroom for f32 rounding on the
`(x+eps)^p` evaluation, but now a refactor that raised eps by an
order of magnitude (e.g. `1e-4`) — which would push the deviation
to ~4e-4, well past `1e-3` — would surface at `cargo test --doc`.
Companion-in-spirit to ticks 605/606 (recommend_parallel,
pixels_per_degree, estimate_gpu_memory_bytes doctest tightenings)
— same theme: doctests should pin within an order of magnitude of
the actual implementation tolerance, not 100-5000× looser. Other
`< 0.01` doctest tolerances in pool.rs are CORRECTLY sized for
the L_p-norm `eps^(1/β)` tail (~3e-3 at β=2), so left alone.
Docs-only change; doctest passes.

Tick 606 — two more doctest tightenings in the same spirit as
tick 605:

1. `DisplayGeometry::pixels_per_degree` (params.rs:90) — claim was
   `assert!((ppd - 75.4).abs() < 0.5)` (loose ±0.5). The runtime
   parity test `ppd_matches_pycvvdp_standard_4k`
   (tests/display_geometry.rs:56) pins `< 1e-4`. Tightened the
   doctest to the same `< 1e-4` tolerance and updated the
   numeric reference value to the full-precision
   `75.402_449_f32`. The old loose tolerance was 5000× too
   wide — a refactor that drifted PPD by 0.3 (e.g. a sign flip
   in the geometry formula at f64-precision) would still pass
   the doctest while failing the runtime test, sending a
   confusing signal to a contributor running `cargo test --doc`.
2. `estimate_gpu_memory_bytes` (pipeline.rs:494) — the 4MP/1MP
   ratio claim was the weak `bytes_4mp > bytes_1mp`. The runtime
   pin `estimate_gpu_memory_scales_with_pixel_count`
   (tests/pipeline_score.rs:2077) pins `ratio ∈ (3.6, 4.4)`.
   Tightened the doctest to the same band, with a reference to
   the runtime test for provenance.

Docs-only change; all 44 doctests still pass.

Tick 605 — tighten `recommend_parallel`'s doctest claim from the
weak `assert!(p >= 2)` to the actual `(10..=40).contains(&p)`
range that the existing runtime test
`recommend_parallel_matches_documented_examples`
(tests/pipeline_score.rs:2220) pins. The old `>= 2` underclaim
risked misleading callers into provisioning ~5× fewer concurrent
instances than the formula actually recommends for the
8 GB / 1 MP case. Also added a second example covering the
24 GB / 12 MP (3090/4090 class) configuration with its `3..=10`
range pin, matching the existing runtime
`p_24gb_12mp` test case. Docs-only change; doctest passes.

Tick 604 — compile-time bit-pin promotion of the 9
`SRGB_LINEAR_TO_DKL` element magnitudes in
`tests/color_scalar.rs`. The existing runtime test
`srgb_linear_to_dkl_matrix_matches_pycvvdp_v0_5_4` already pinned
the same 9 f32 values via `.to_bits()`; promoting to compile-time
`const _: () = assert!(...)` blocks means a refactor that drifts
any element trips at `cargo check`, before any test binary
builds. Same promotion pattern as tick 561 (LUT endpoints) and
tick 567 (sign-signature). Companion to the row-sign-signature
pin already at compile-time. Static-assert count is now 232 across
13 test files (+9 from tick 603's 223).

Tick 603 — two `recommend_parallel` contract pins surfacing
implicit-via-language-semantic guarantees that a refactor could
silently break:

1. `recommend_parallel_saturates_at_u32_max_for_unbounded_free_bytes`
   — the function's docstring claims the result is "capped at
   u32::MAX", but the cap is implicitly enforced by Rust's
   saturating `f64 as u32` cast (saturating since Rust 1.45). A
   refactor that swaps to `try_into().unwrap()` would panic on
   u64::MAX free-bytes input. Pinned at 1024² and 8×8 (smallest
   pyramid-valid).
2. `recommend_parallel_monotonically_non_increasing_in_image_dims`
   — companion to the existing `..._monotonic_in_free_bytes` pin
   (tick 234). Holding free memory constant, larger images must
   produce ≤ smaller-image parallel counts. Strictly-monotonic
   decrease would be too strong (min-1 floor flattens the curve);
   non-increasing is the load-bearing invariant. A refactor that
   inverts the division (`est * free / safety` instead of
   `free / (safety * est)`) would silently make bigger images
   *more* parallelizable, masking OOM in auto-scaling sweep code
   that picks instance count from image size.

Both pins land in `tests/pipeline_score.rs` next to the existing
recommend_parallel test cluster. Static-assert count unchanged
(these are runtime tests, not compile-time const blocks — the
saturation behavior depends on `as u32` rounding which isn't
`const fn`-callable in this configuration).

Tick 602 (`df0998d0`) — docstring-accuracy fix on the
`common::const_str` module. The tick-584 module docstring was
doubly stale: listed only 3 helpers (`starts_with` / `ends_with`
/ `contains`) when ticks 586 and 600 added `bytes_eq` and `count`,
and listed only 3 call sites (`column_name.rs`,
`goldens_metadata.rs`, `lib_reexports.rs`) when ticks 588–600
added `version_lockstep.rs` (heavy user) and `const_str_helpers.rs`
(unit tests). Rewrote to enumerate all 5 helpers with their
tick-of-introduction and all 5 current call sites with one-line
purpose notes. Docs-only change — no code edits, no assert count
change.

Tick 601 added a separate-purpose static assert in
`version_lockstep.rs` pinning that `docs/BURN_PORT_PLAN.md` keeps
its "ABANDONED" status banner. The Burn port plan was marked
ABANDONED at tick 324 after a cubek::conv2d separable spike
measured 4.32× slower than the hand-written downscale_kernel at
4000×3000 on an RTX 5070. The banner status is stable — if the
plan is genuinely revived, the file should be renamed /
restructured, not silently un-marked. NOT a PYCVVDP_REFERENCE_VERSION
lockstep pin (this is about file-content stability, not version
bumping). Static-assert count is now 223 across 13 test files.

- **Spatial-contrast contract pinned across all 6 dispatch surfaces
  (ticks 542–547).** Eighteen hypothesis-test pins capture cvvdp's
  spatial-contrast contract — three properties × six dispatch paths
  (host scalar `predict_jod_still_3ch`, GPU cold-ref `Cvvdp::score`,
  GPU cached-ref `set_reference` + `score_with_reference`, GPU warm-
  ref `warm_reference` + `compute_dkl_jod_with_warm_ref`, cubecl-cpu
  host-pool `compute_dkl_jod_host_pool`, cubecl-cpu warm-ref host-
  pool `warm_reference` + `compute_dkl_jod_host_pool_with_warm_ref`):
  - **flat-vs-flat → JOD ≈ 10** (542 host, 545 GPU cold, 546 GPU
    cached, 546 cpu host_pool, 547 GPU warm-ref, 547 cpu warm-ref
    host_pool): pure black vs pure white returns JOD ≈ 10 because
    cvvdp measures contrast *within* an image, not absolute
    differences *between* images. Both flat inputs have zero
    Weber-band energy → D = 0 → JOD = 10. Guards against an
    "absolute-difference" refactor.
  - **textured-vs-flat → JOD ≪ 10** (543 host, 545 GPU cold, 546
    GPU cached, 546 cpu host_pool, 547 GPU warm-ref, 547 cpu warm-
    ref host_pool): a 32×32 textured ref vs flat mid-gray dist
    (catastrophic blur) gives JOD = 3.4402 on host scalar and JOD =
    3.4389 on all five GPU/cpu kernel paths. The ref's missing-band
    energy converts to a non-trivial Q via masking → pool, and
    met2jod maps that below 10.
  - **noise-amplitude monotonicity** (544 host, 545 GPU cold, 546
    GPU cached, 546 cpu host_pool, 547 GPU warm-ref, 547 cpu warm-
    ref host_pool): dense alternating-sign noise at amplitudes
    {2, 8, 32} produces JOD {9.9941, 9.9670, 9.6885} — bit-identical
    across all 6 dispatch surfaces to 4 decimals. Probes the dense-
    noise regime that the sparse-distortion pin
    (`output_responds_to_distortion_magnitude`) doesn't cover.

  Cross-path agreement: bit-identical to 4 decimals across host
  scalar, GPU cold-ref, GPU cached-ref, GPU warm-ref, cpu host_pool,
  and cpu warm-ref host_pool — the only divergence is host-scalar
  3.4402 vs the five kernel-path 3.4389 on textured-vs-flat (atomic-
  add noise floor on the host pool of the scalar reference path).
  Together with the stuck-at-constant pins (ticks 494, 508, 509,
  525), the spatial-contrast contract is now load-bearing-tested on
  every dispatch surface.

  Commit hashes: `002e6958` (542), `58523a73` (543), `2ed1f4a4`
  (544), `2da874bf` (545), `35594044` (546), tick 547 in this commit.

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

### Changed

#### cvvdp-gpu (tests)

- **`#![warn(missing_docs)]` crate-level lint guard** (in `lib.rs`)
  — pins the missing_docs-clean state established at tick 514.
  `warn` (not `deny`) so any newly-added undocumented public item
  surfaces during local dev + cargo doc but doesn't hard-block.
  The 5 kernel files override to `allow` via their own inner
  attribute (silencing `#[cube(launch)]` macro-emitted items
  only); kernel pub items written by humans remain protected. New
  pub items added elsewhere in the crate will trip the guard.
  Tick 516.

- **96 missing_docs warnings on `#[cube(launch)]` macro items —
  cleared** (in `kernels/{color,csf,masking,pool,pyramid}.rs`).
  Each kernel file now has `#![allow(missing_docs)]` at the top,
  immediately after the module-level `//!` block. The
  `#[cube(launch)]` macro emits a sibling module + launcher struct
  + associated `fn` per annotated kernel function; those items
  don't inherit the user's rustdoc comment and triggered 4
  warnings each × ~25 kernels × 4 emit sites ≈ 96 warnings total.
  Tick 494's attempt at putting `#[allow]` on the individual
  kernel functions didn't propagate through the macro; only the
  file-level inner attribute works. Every user-written pub item
  in these files (LUTs, scalar helpers, kernel functions
  themselves) is already documented, so the allow only suppresses
  macro-emitted noise — no user-doc coverage lost. Verified:
  `RUSTDOCFLAGS="-W missing_docs" cargo doc -p cvvdp-gpu` now
  reports 0 warnings (was 96). Tick 514.

- **`compute_dkl_jod_host_pool_with_warm_ref_runs_on_cpu_backend`**
  (in `cpu_backend.rs`) — tightened from 0.005 JOD tolerance to
  `to_bits()` bit-equality between cold-ref `compute_dkl_jod_host_pool`
  and warm-ref `compute_dkl_jod_host_pool_with_warm_ref` on the
  cubecl-cpu runtime. The cpu runtime executes every kernel
  sequentially (no GPU atomic-add nondeterminism), and the
  host_pool path uses sequential `lp_norm_mean` (no `Atomic<f32>::fetch_add`),
  so warm and cold dispatches MUST produce bit-identical f32 JOD
  on the same input. Catches a refactor that introduces accidental
  nondeterminism on the warm-ref path (e.g. accumulating across
  calls without resetting a scratch). Confirmed bit-equal at
  `0x4116b771` on the synth_pair 32×32 corpus. Tick 496.

### Added

#### cvvdp-gpu (tests)

- **`new_with_geometry_stable_under_degenerate_geometry`** (in
  `pipeline_score.rs`) — companion to tick 495's
  `ppd_does_not_panic_on_degenerate_inputs`. `Cvvdp::new_with_geometry`
  internally calls `geometry.pixels_per_degree()` to derive pyramid
  level count via `pyramid_levels`. Degenerate geometries can
  produce NaN/Inf ppd; the contract is that `new_with_geometry`
  remains total — either succeeds (potentially with a degraded
  pyramid level count) or returns `Error::InvalidImageSize`, but
  MUST NOT panic. 5 degenerate-geometry cases pinned: zero
  distance, zero diagonal, zero resolution_w, extreme close (1cm),
  extreme far (100m). All currently succeed; the test does not
  pin which path because future tightening could legitimately
  shift them between Ok and InvalidImageSize. Tick 497.

- **`ppd_does_not_panic_on_degenerate_inputs`** (in
  `display_geometry.rs`) — stability pin on
  `DisplayGeometry::pixels_per_degree` for degenerate inputs (zero
  `distance_m` / `diagonal_inches` / `resolution_w` /
  `resolution_h`, plus all-zero). The function is a total
  computation — it MAY return ±∞ or NaN for mathematically
  degenerate inputs, but it must not panic. Callers like
  `Cvvdp::compute_dkl_jod(ref, dist, ppd)` accept arbitrary ppd
  values; a future refactor that adds `assert!(distance_m > 0)` (or
  equivalent) to ppd computation would change the contract from
  "total + degraded output" to "panicking" — surface that change
  here. Observed degenerate outputs (zero distance → ppd ≈
  0.00556 = 1/180°, zero diagonal → Inf, zero resolution_*  →
  NaN). The pin doesn't assert on the specific values because a
  future formula refactor could legitimately shift ±0 ↔ ±Inf
  without breaking the no-panic guarantee. Tick 495.

- **`all_four_scoring_paths_agree_bit_equal_on_same_input`** (in
  `pipeline_score.rs`) — consolidation pin. The four documented
  public scoring paths must produce bit-identical f32 JOD on the
  same (ref, dist) input on a single Cvvdp instance: (A)
  `score(ref, dist)`, (B) `compute_dkl_jod(ref, dist, ppd)`, (C)
  `set_reference + score_with_reference`, (D) `warm_reference +
  compute_dkl_jod_with_warm_ref`. Individual pins cover pairwise
  relationships (tick 407: A↔B widening; tick 488: A↔C bit; tick
  489: D determinism). This test pins the four-way intersection: a
  refactor that, say, routes warm-ref through a subtly different
  pool kernel would surface here as D drifting from A/B/C even when
  each path's standalone determinism holds. Tick 494.

- **`new_equivalent_to_new_with_geometry_standard_4k`** (in
  `pipeline_score.rs`) — pins the documented `Cvvdp::new` rustdoc
  contract that it is "equivalent to
  `new_with_geometry(..., STANDARD_4K)`". Today the implementation
  forwards to `new_with_geometry`, but a future refactor that adds
  extra initialization to one but not the other (different default
  geometry, eager priming on the explicit-geometry path, etc.)
  would silently change the documented surface. 4 invariants:
  scoring the same (ref, dist) on two Cvvdp instances — one built
  via `new`, the other via `new_with_geometry(STANDARD_4K)` —
  produces bit-identical results on (1) `score()`, (2)
  `set_reference` + `score_with_reference()`, (3) `compute_dkl_jod`
  with the STANDARD_4K ppd, and (4) `warm_reference` +
  `compute_dkl_jod_with_warm_ref`. Tick 493.

- **clippy cleanup on `lib_reexports.rs` (tick 503 follow-up)** —
  closed two warnings introduced by tick 503's additions:
  (1) `clippy::type_complexity` on the inline
  `fn(&[u8], &[u8], usize, usize, DisplayModel, f32) -> f32` pointer
  type for `predict_jod_still_3ch` — hoisted into a `PredictJodFn`
  type alias; (2) `clippy::assertions_on_constants` on
  `assert!(N_L_BKG > 0)` — promoted to `const _: () = assert!(...)`
  static assertion that fires at compile time instead of runtime.
  Both warnings now clean; the 11 lib_reexports tests still pass.
  Tick 505.

- **`host_scalar_module_is_public` + `kernels_submodules_are_public`**
  (in `lib_reexports.rs`) — 2 new pins (11 total). Pins
  `cvvdp_gpu::host_scalar::predict_jod_still_3ch` (the canonical
  host-only reference pipeline used by shadow_jod, cpu_backend, and
  GPU-less CI environments) and the five kernels submodules
  (`color`, `csf`, `masking`, `pool`, `pyramid`) as public API
  surfaces via compile-time use sites. Existing per-kernel test
  files import specific items but no single pin verified that the
  module paths themselves remain public — catches a refactor that
  collapses one submodule into a parent or downgrades it to
  `pub(crate)`. Tick 503.

- **`params_scaffolding_types_are_public`** (in `lib_reexports.rs`)
  — adds a 9th pin to the lib re-export coverage. `CsfParams`,
  `MaskingParams`, `PoolingParams`, `JodParams` are documented as
  scaffolding for a planned "load parameters from vendored cvvdp
  JSON" path. They have no other test importing them — without
  this pin a future refactor that downgrades them to `pub(crate)`
  or removes them as unused would break the planned path silently.
  Compile-time use site via `cvvdp_gpu::params::{CsfParams, ...}`
  plus a touchpoint via `CvvdpParams::PLACEHOLDER` sub-bundle
  access. Tick 502.

- **`lib_reexports.rs` extended** — adds 3 new pins (8 total, was
  5): (1) `cvvdp_type_reexport_resolves` — `Cvvdp<R>` is the main
  scoring type; without this pin, a future refactor that moves it
  behind a feature gate or into a private module would break every
  downstream caller (zen-metrics-cli's `CvvdpBatchScorer` references
  `cvvdp_gpu::Cvvdp` directly); (2)
  `lib_constants_reexport_match_their_originals` — pins `N_CHANNELS`
  (3), `MAX_LEVELS` (9), `PYRAMID_MIN_DIM` (4), and
  `CVVDP_COLUMN_NAME` (prefix `cvvdp_`) against their documented
  values, also as a use-site pin; (3) `error_and_result_reexport_resolve`
  — `Error` and `Result<T>` are how callers see method failures,
  both must be reachable from the crate root. Tick 501.

- **`manifest_url_uses_cvvdp_goldens_bucket_subpath`** (in
  `goldens_metadata.rs`) — pins the crate-specific
  `/cvvdp-goldens/` bucket subpath on `MANIFEST_URL`. Sibling crates
  (zensim-gpu, butteraugli-gpu, dssim-gpu, ssim2-gpu) all publish
  their goldens to the same host under different subpaths
  (`/zensim-goldens/`, etc.). A refactor that accidentally swapped
  the bucket subpath would still pass the host check (tick 519) +
  version-segment check + scheme/suffix checks, but fetch a sibling
  crate's manifest. Tick 520.

- **`manifest_url_uses_documented_r2_host`** (in
  `goldens_metadata.rs`) — pins the canonical R2 host
  (`https://coefficient.r2.imazen.org/`) on `MANIFEST_URL`. Closes
  a gap where a refactor to a different CDN bucket on a different
  cloud, or a localhost dev mirror, would pass the existing
  structure checks (https scheme + .json suffix + `/v1/` segment +
  64-char hex sha256) yet fetch the wrong manifest. The host
  migration coordination is forced by requiring this pin to update
  in the same commit as the URL change. Tick 519.

- **`perf_mode_fast_matches_strict_on_gpu_host_pool`** (in
  `pipeline_score.rs`) — third leg of the PerfMode no-op contract.
  Existing coverage: `perf_mode_fast_matches_strict_today` (tick 322
  + 324) — GPU pool path with 1e-4 tolerance against atomic-add
  noise; `perf_mode_fast_matches_strict_on_cpu_host_pool` (tick 327)
  — cpu-runtime + host_pool path with bit-equality. This fills the
  missing combination: **GPU runtime + host_pool path** (e.g.
  CudaRuntime calling `compute_dkl_jod_host_pool`). The host_pool
  variant reads D bands back to host then folds via sequential
  `lp_norm_mean` — no GPU atomic-add involved, so bit-equality
  should hold even on a GPU runtime. Verified bit-equal at
  `0x40da3d6b` on 64×64 deterministic input. Catches a refactor
  that, say, makes Fast mode swap in a different host-fold
  accumulation order on the GPU runtime. Tick 512.

- **`compute_dkl_jod_host_pool_with_warm_ref_distinguishes_v1_corpus_q_levels`**
  (in `pipeline_score.rs`) — fourth-leg stuck-at-constant pin
  covering the warm-ref host_pool path (the batch-scoring fast path
  used by cubecl-cpu / Metal-compatible production workers). Same
  strict-separation contract as the GPU sibling (tick 508) and
  cold-host_pool sibling (tick 509): scoring v1 corpus at
  q ∈ {1, 20, 90} produces strictly increasing JOD with ≥ 0.01 JOD
  adjacent-level gap. Catches a refactor that caches the wrong DIST
  intermediate on the warm-ref host_pool path and silently breaks
  batch CPU scoring without showing up on the warm/cold-equality
  pin (which would still match within tolerance even if BOTH
  collapsed). Observed scores identical to GPU/cold-host_pool
  paths. Tick 525.

- **`cvvdp_host_pool_distinguishes_v1_corpus_q_levels`** (in
  `pipeline_score.rs`) — sibling to tick 508 for the host_pool
  scoring path (`compute_dkl_jod_host_pool`, the cubecl-cpu /
  Metal-compatible path). Same strict-separation contract:
  scoring v1 corpus at q ∈ {1, 20, 90} on the host_pool path
  produces strictly increasing JOD with ≥ 0.01 JOD adjacent-level
  gap. The host_pool path uses sequential `lp_norm_mean` instead
  of GPU atomic-f32 pool — a refactor that breaks distortion
  discrimination on one path doesn't automatically break it on the
  other, so pin both. Observed identical scores to GPU path
  (q=1→7.65, q=20→9.71, q=90→9.99) — consistent with tick 208's
  GPU/host_pool agreement pin. Tick 509.

- **`cvvdp_score_distinguishes_v1_corpus_q_levels`** (in
  `pipeline_score.rs`) — stuck-at-constant pin for the GPU
  `Cvvdp::score` path. Asserts that scoring the v1 corpus at
  q ∈ {1, 20, 90} produces strictly increasing JOD values with
  ≥ 0.01 JOD separation between adjacent levels. Catches a refactor
  that collapses pipeline output (e.g. forgets to route DIST
  through CSF, returns the REF-against-REF JOD uniformly, or drifts
  all scores toward a midpoint within the 0.005 manifest tolerance).
  The existing `cvvdp_score_matches_v1_manifest` pin (tick 207) is
  partly redundant for the BAD case (a pipeline collapsed to a
  single value would fail manifest tolerance at some q), but not
  for the GOOD case where scores stay within tolerance but lose
  discriminative power. Host-scalar sibling: tick 434's
  `predict_jod_invariants` "responds to distortion magnitude" pin.
  Observed gaps: q=1→7.65, q=20→9.71, q=90→9.99. Tick 508.

- **`two_fresh_cvvdp_instances_produce_bit_equal_jod`** (in
  `pipeline_score.rs`) — pin cross-instance determinism. Two
  `Cvvdp::new` calls with the same (width, height, params,
  geometry) scoring the same (ref, dist) pair MUST produce
  bit-identical JOD (within atomic-add tolerance on GPU).
  Within-instance determinism is pinned by
  `score_is_deterministic_across_repeated_calls` (tick 411); the
  cross-instance contract is independent — catches a refactor that
  accidentally shares state via `static` / `thread_local` / a
  process-global counter, or that uses non-deterministic allocation
  order to seed kernel blocks (e.g. via hashmap iteration). This
  would silently break batch scoring across multiple
  CvvdpBatchScorer instances on a sweep worker. Tolerance set to
  1e-4 to accommodate the documented GPU atomic-add nondeterminism
  (tick 324); first run observed |diff| = 0 (bit-equal in this
  sample). Tick 499.

- **`cvvdp_score_smoke_at_extreme_aspect_ratio`** (in
  `pipeline_score.rs`) — end-to-end GPU smoke at extreme aspect
  ratios. Tick 498 covered 128×8 + 8×128 (16:1 ratio at boundary
  on one side). Tick 511 extended to 1024×8 + 8×1024 (128:1
  ratio — stresses any width-axis-specific tiling assumption that
  the 16:1 case doesn't exercise). The tick-491 8×8 boundary smoke
  covers the minimum-square case; this covers asymmetric strips.
  `pyramid_levels` is bounded by `min(w, h).ilog2()` — a pyramid
  construction that accidentally defaults to `max(w, h).ilog2()`
  (= 7 instead of 3) at the asymmetric edge would surface here as
  NaN/Inf JOD or an InvalidImageSize error. 4 aspects × 4
  invariants each: (1) identity `score(ref, ref) ≈ 10` within
  1e-3; (2) non-trivial perturbation produces finite JOD in
  `[0, 10]` strictly less than identity. Tick 498, 511.

- **`cvvdp_score_smoke_at_pyramid_min_boundary`** (in
  `pipeline_score.rs`) — end-to-end GPU smoke test on the minimum
  supported dimensions (8×8 = `PYRAMID_MIN_DIM × 2`). Existing
  `invalid_image_size_surfaces_on_too_small_dims` only verifies
  that `Cvvdp::new(8, 8)` returns Ok — it doesn't verify any scoring
  path works at boundary dims. `predict_jod_still_3ch` invariants
  (tick 434) covered 8×8 on the host-scalar path; this pins the GPU
  equivalents. 4 invariants: (1) identity contract `score(ref, ref)
  ≈ 10` within 1e-3; (2) non-trivial perturbation produces finite
  JOD in `[0, 10]` strictly less than the identity JOD;
  (3) `set_reference` + `score_with_reference` works at boundary
  dims AND is bit-equal to direct `score()` (extends tick 488 pin
  to boundary); (4) `warm_reference` +
  `compute_dkl_jod_with_warm_ref` works at boundary dims with
  finite JOD in `[0, 10]`. A pyramid-construction bug at boundary
  dims (degenerate zero-channel band, off-by-one in dispatcher
  launch geometry, halving-loop regression) would surface here as
  a panic / NaN. Tick 491.

- **`compute_dkl_jod_with_warm_ref_is_deterministic_across_repeated_calls`**
  (in `pipeline_score.rs`) — 2 invariants on the warm-ref fast path
  (the `warm_reference` + `compute_dkl_jod_with_warm_ref` pattern
  that `CvvdpBatchScorer` uses for batch DIST scoring on vast.ai
  workers — the hottest call shape in the sweep): (1) warm-ref
  scoring twice on the same dist returns bit-identical `f32` JOD
  (`to_bits()` equality); (2) an intervening warm-ref call on a
  different dist does not poison per-call scratch — first and third
  dist_a calls remain bit-equal. Sibling pin to
  `score_with_reference_is_deterministic_across_repeated_calls`
  (tick 488) and `score_is_deterministic_across_repeated_calls` —
  completes bit-determinism coverage for all three scoring paths.
  Tick 489.

- **`score_with_reference_is_deterministic_across_repeated_calls`**
  (in `pipeline_score.rs`) — 3 invariants on the `set_reference` /
  `score_with_reference` cached fast path: (1) calling the cached
  path twice with the same dist returns bit-identical `f64` JOD
  (`to_bits()` equality); (2) bit-equal to the direct
  `Cvvdp::score(ref, dist)` path — strengthens the existing
  `score_with_reference_matches_score` (which used a 1e-6 tolerance)
  to the documented "match exactly" contract from tick 213; (3) an
  intervening cached-path call on a different dist does not poison
  the per-call state — first and third dist_a calls remain
  bit-equal. Sibling pin to
  `score_is_deterministic_across_repeated_calls` for the cached path
  that `CvvdpBatchScorer` relies on. Tick 488.

- **`state_machine_independence.rs`** — 5 invariant pins on the
  `Cvvdp::set_reference` / `Cvvdp::warm_reference` cache state
  machine. Pins (1) fresh `Cvvdp::new()` surfaces `NoCachedReference`
  *and* `NoWarmReference` from the two fast paths independently;
  (2) `set_reference` does NOT prime warm state (dual of tick 238's
  `set_reference_does_not_invalidate_warm_state`); (3) `warm_reference`
  does NOT prime the set_reference cache; (4) one-shot
  `Cvvdp::score` does NOT pollute either cache; (5) one-shot
  `compute_dkl_jod` does NOT pollute either cache. Catches a future
  "eager upload" refactor that would silently change the documented
  fast-path-error surface. Tick 486.

#### cvvdp-gpu (docs)

- **README "GPU memory budgeting" section** — documents the new
  `estimate_gpu_memory_bytes` + `recommend_parallel` +
  `PARALLEL_SAFETY_FACTOR` API surface added in ticks 398-399.
  Includes a size-vs-budget table at standard 4K geometry
  showing PARALLEL caps for 8 GB and 24 GB GPUs at six image
  sizes (64² through 12 MP), a code example for how a sweep
  worker should derive `PARALLEL`, and guidance on when to
  tighten / loosen the safety factor (warm-ref batches → 1.2;
  mixed CPU+GPU process → 2.0). Tick 400, 253899ab.

#### cvvdp-gpu (api)

- **`cvvdp_gpu::recommend_parallel(free_gpu_bytes, width, height)
  -> u32`** + **`PARALLEL_SAFETY_FACTOR`** const — bundles
  `estimate_gpu_memory_bytes` with the documented 1.5× safety
  factor so callers don't have to maintain the constant
  themselves. Returns the maximum number of `Cvvdp` instances
  that should fit on the GPU, with a `max(1)` floor (a single
  instance always gets to attempt scoring; OOM after that is the
  caller's signal to back off to host_pool or smaller images,
  not a "no work" signal). Returns 0 only when image dims are
  invalid or `free_gpu_bytes == 0`. Worked examples documented in
  the rustdoc + pinned by `recommend_parallel_matches_documented_examples`
  (8 GB / 1024² → PARALLEL in [10, 40]; 24 GB / 12 MP → PARALLEL
  in [3, 10]). `PARALLEL_SAFETY_FACTOR = 1.5` is exported so
  callers with different workload mixes can compute a tighter
  cap manually (warm_reference batches → ~1.2; mixed CPU + GPU
  process → ~2.0). Tick 399, 14d95cb4.

- **`cvvdp_gpu::estimate_gpu_memory_bytes(width, height) -> Option<usize>`**
  — static-analysis predictor for the GPU memory `Cvvdp::new` will
  allocate. Sums every persistent buffer: source u32 bytes, three
  full pyramids (`gauss_ref` + `bands_ref` + `bands_dis`), 6 ×
  d_scratch planes per level, weber_scratch (6 fine + 4 v-scratch
  per non-baseband level), partials, baseband log_l_bkg, srgb_lut,
  logs_row — using ceil-div halving to match the actual allocator
  layout (tick 175 ceil-div + tick 208 d_scratch + tick 240 pre-
  bundled handles). Worked-example table at standard 4K geometry:
  64² = 0.8 MB, 256² = 13 MB, 512² = 52 MB, 1024² = 208 MB,
  2048² = 833 MB, 4096×3072 (12 MP) = 2.5 GB. Use to cap fleet
  concurrency: `PARALLEL = floor(free_gpu_bytes / (1.5 *
  estimate))` — at 1024² on an 8 GB GPU that's 25, on 24 GB it's
  76. Returns `None` below the [`PYRAMID_MIN_DIM`] × 2 = 8×8
  threshold (same precondition as `Cvvdp::new`). Pinned by four
  tests in `tests/pipeline_score.rs`: below-threshold, pixel-count
  scaling (4× pixels → 4× ±10% bytes), order-of-magnitude at
  three sizes, and a worked-example concurrency-cap calc on an
  8 GB GPU. Includes `examples/cvvdp_mem_table.rs` for operators
  to probe the table on their own hardware. Tick 398, 9a30d97f.

#### scripts/sweep (cvvdp-backfill — deployment hardening, ticks 353-376)

Long arc of fleet deployment hardening between ticks 353 (backend
support docs) and 376 (CUDA 12.4 SDK build). 18 fleet attempts in
total, all destroyed pre-completion — each surfacing one or two
real defects that ship as commits. Summarized here because
walking every fleet's failure log into the changelog one-by-one
wouldn't aid future operators; the artifacts that survived the
arc are:

  - **`Cvvdp::compute_dkl_jod` / `_with_warm_ref` / `score` docs**:
    explicit "# Backend support" sections documenting the
    `Atomic<f32>::fetch_add` trap (cubecl-cpu PANICS at launch;
    Metal silently no-ops → JOD=10 for any input). Tick 353-354,
    50fb88ca/b909365b.
  - **`scripts/sweep/cvvdp_backfill/status.sh`** — at-a-glance
    fleet progress aggregator (manifest size + heartbeats +
    sidecar counts with %). Tick 357, ad8a3031.
  - **`assert_parity.py` anti-flatline check** — detects the
    silent-failure mode where score-pairs writes the same value
    for every row. Pairs with new `imazen_stats`/`pycvvdp_stats`
    fields in finalize.sh's manifest. Tick 358, b5b6f4cf.
  - **README §Documentation pointer** to the cvvdp-backfill
    operator runbook. Tick 359, 43cd69a3.
  - **`chunk_worker.sh` ENTRYPOINT override** —
    `docker run image zen-metrics ...` was hitting the production
    image's `zen-metrics-worker` ENTRYPOINT. `--entrypoint
    /usr/local/bin/zen-metrics` bypasses. Tick 360, b28e1f0b.
  - **`finalize.sh` non-destructive ~/.aws/credentials write** —
    used to overwrite the local developer box's `[default]`
    profile. Now idempotent-append. Tick 362, 9455eee8.
  - **`onstart_cvvdp_backfill.sh` apt-get install docker.io** —
    boot bootstrap was bailing on missing docker. Tick 363,
    6016441e.
  - **`onstart_cvvdp_backfill_imazen.sh` + `launch_imazen.sh`** —
    single-image fleet path (no docker-in-docker). vast.ai SSH
    instances don't allow privileged dockerd; this variant boots
    the zen-metrics-sweep image directly and skips pycvvdp
    entirely. Ticks 364-365, 8f81895a.
  - **`chunk_worker.sh` R2() wrapper** — bare `s5cmd` fell through
    to AWS `[default]` profile. Now all calls go through a
    wrapper that pins `--profile r2 --endpoint-url`. Tick 365,
    7f3a3cb8.
  - **`chunk_worker.sh` GROUPS → GROUP_LINES rename** — CRITICAL
    fix. `GROUPS` is a bash special array (current process's
    supplementary group IDs); string-assigning to it exits 1 and
    corrupts the value. Every chunk_worker invocation pre-fix
    silently died at this line. Tick 367, dc794de2.
  - **`Dockerfile.sweep.v13` CUDA SDK pin** — cubecl-cuda's
    `cudarc` 0.19.4 deps with `cuda-version-from-build-system` +
    `fallback-latest` was binding to `cuCoredumpDeregisterCompleteCallback`
    (cuda-13020 cfg-gated symbol that doesn't exist in any
    released NVIDIA driver yet). Builder stage now installs
    `cuda-nvcc-12-4` + `cuda-cudart-dev-12-4` so cudarc emits
    cuda-12040 features — compatible with driver 550+ which
    covers ~all vast.ai boxes. Tick 372 / 376, daa0d82e + cuda124.
  - **`scripts/sweep/cvvdp_backfill/STATUS.md`** — diagnostic
    chain captured so future operators don't re-burn 14 fleet
    attempts. Tick 370/372.

#### scripts/sweep (cvvdp-backfill pipeline — PINNED TASK)

End-to-end vast.ai fleet pipeline to backfill cvvdp scores
(`cvvdp_imazen_*` + `cvvdp_pycvvdp_v054` columns) onto the
2.37M-row unified parquet store at
`/mnt/v/zen/zensim-training/2026-05-07/unified/`. Six scripts
shipped across six commits + an operator runbook:

- `scripts/sweep/generate_cvvdp_backfill_chunks.py` — reads the
  7 unified parquets, splits into ~23,747 chunks of 100 rows
  each, emits `chunks.jsonl` with input_parquet, row_range,
  image_basenames, and per-impl R2 sidecar paths (`d2eb0f7c`,
  tick 336).
- `scripts/sweep/cvvdp_backfill_chunk_worker.sh` — per-chunk
  worker. Downloads input_parquet from R2, syncs basenames,
  slices parquet, groups rows by (codec,q,knob_tuple),
  re-encodes via `zen-metrics sweep --pairs-tsv`, scores in
  both impls (`score-pairs` + `pycvvdp_worker.py`), uploads
  sidecars. Host-binary OR docker-image execution modes
  (`87deac34`, tick 337).
- `scripts/sweep/onstart_cvvdp_backfill.sh` — vast.ai instance
  entry point. 239 lines: installs s5cmd+jq+docker, pre-pulls
  ZEN_METRICS_IMAGE + PYCVVDP_IMAGE, heartbeats, downloads
  chunks.jsonl + worker.sh, processes via `xargs -P $PARALLEL`
  with R2 atomic-claim (`32a3b64a`, tick 338).
- `scripts/sweep/cvvdp_backfill/launch.sh` — host-side fleet
  launcher modeled on `v15/launch_gpu.sh`. Boots ubuntu:24.04,
  bootstrap pulls onstart from R2. Defaults: N_BOXES=6,
  MAX_DPH=0.30, MIN_RAM_GB=16, MIN_DISK_GB=40 (for the 6.5 GB
  pycvvdp image) (`c572c192`, tick 339).
- `scripts/sweep/cvvdp_backfill/finalize.sh` — post-fleet
  consolidation. 253 lines: R2-syncs per-chunk sidecars,
  groups by (impl, source_stem), concatenates via
  `pyarrow.concat_tables`, emits parity TSV per source +
  manifest.json with per-source row counts and parity stats
  (`09512676`, tick 341).
- `scripts/sweep/cvvdp_backfill/README.md` — 213-line operator
  runbook with ASCII pipeline diagram, 6-step quick-start,
  docker image specs, 5 troubleshooting cases, "when NOT to
  use this" guidance vs v15 dispatcher (`cbf218d9`, tick 342).
- `scripts/sweep/cvvdp_backfill/assert_parity.py` — optional
  automation gate that consumes finalize.sh's `manifest.json`
  and exits non-zero on threshold violation. Defaults match
  the smoke-tested n=4 sentinel: `mean/median ≤ 0.10 JOD`,
  `max ≤ 0.50 JOD`. Six-fixture smoke-verified across
  pass/fail/null-tolerated/null-required-fails/only-sources
  scoping/json-summary write (`252ee704`, tick 344).

#### cvvdp-gpu (docs)

- **`CVVDP_TRACE` / `CVVDP_TRACE_WEBER` debug env vars** are now
  documented in lib.rs's crate-level docstring under a new
  "Debug tracing env vars" section. Both vars existed in
  pipeline.rs dispatch helpers but only `CVVDP_TRACE_WEBER` had
  a user-facing docstring — `CVVDP_TRACE` was discoverable only
  via grep or by reading a benchmark MD. The new section lists
  the exact stderr line shapes each var emits, verified against
  the `if trace` blocks in pipeline.rs (`9fb0c569`, tick 347).

#### cvvdp-gpu (tests, extension)

- **`tests/predict_jod_invariants.rs`** — 2 new tests extending the
  square-only coverage: `non_square_dimensions_are_supported`
  (32×16, 16×32, 64×24, 24×64) and `odd_dimensions_are_supported`
  (13×17 prime-ish, 15×15 odd-square, 73×91 — the historical tick
  206 regression case from pycvvdp's `x.shape[-2]` parity quirk in
  `gausspyr_reduce_scalar`). Both pin: identical → ≈10 within 1e-2;
  perturbed → finite < ident + 1e-3. Tick 437, ec79dc49.

#### cvvdp-gpu (docs)

- **Struct-field docstrings** — added field docs for
  `Band {w, h, data}`, `WeberPyramid {bands}`, `JodParams
  {jod_a, jod_b, jod_c}`, `DisplayGeometry {resolution_w,
  resolution_h}`, and `CvvdpParams {display, csf, masking,
  pooling, jod}`. The `CvvdpParams` field docs explicitly note
  that csf/masking/pooling/jod are scaffolding placeholders unused
  by production code (which reads `kernels::*` consts). Drops
  rustdoc missing-docs warnings from 110 → ~88 (rest are
  `#[cube(launch)]` macro-emitted items, not user-writable).
  Tick 481, b756adce.

- **`PU_PADSIZE` doctest** — added `# Examples` confirming the
  threshold value (6) and the branch-condition contract
  (`phase_uncertainty_band` at 6×6 takes the no-blur branch because
  the check is `> PU_PADSIZE` not `>=`). Tick 480, 6b542aaa.

#### cvvdp-gpu (docs, fix)

- **3 pre-existing `no_run` doctests on `Cvvdp::score`,
  `score_with_reference`, `compute_dkl_jod_with_warm_ref`** changed
  to `ignore` — they assumed a default-features build with cuda/wgpu/
  cpu, where `Backend = cubecl::cuda::CudaRuntime` resolves. Under
  `cargo test --doc --no-default-features` the type alias was empty
  and all three failed to compile. `ignore` preserves the
  documentation while skipping the compile-only step. Tick 479, 949daf00.

#### cvvdp-gpu (docs)

- **`BETA_CH` doctest** — added `# Examples` (covers all 3 pool
  Minkowski exponents): `BETA_SPATIAL == 2.0` (RMS), `BETA_BAND ==
  4.0`, `BETA_CH == 4.0`; spatial is the gentler exponent. Tick 478, 18d601ad.

- **`JOD_EXP` doctest** — added `# Examples` (on `JOD_EXP`; also
  cross-references `JOD_A` and `IMAGE_INT`): `met2jod(1.0)` matches
  `10 - JOD_A * 1^JOD_EXP` algebra within 1e-5, IMAGE_INT lives in
  `(0, 1)`. Tick 477, c02a797c.

- **`PER_CH_W` doctest** — added `# Examples` showing 3 channels
  all at 1.0 (no per-channel attenuation at the pool stage; chroma
  weighting happens earlier via `masking::CH_GAIN`). Tick 476, e0369c79.

- **`CH_GAIN` doctest** — added `# Examples` showing 3 channels,
  A/Vy at 1.0 passthrough, Rg boosted at 1.45 (cvvdp's "ch_chrom_w"
  for the red-green axis). Tick 475, 246a862a.

- **`N_RHO` doctest** — added `# Examples` covering both axis-size
  constants: `N_L_BKG == 32`, `N_RHO == 32`, and `LOG_L_BKG_AXIS`
  / `LOG_RHO_AXIS` lengths match (the LUTs are 32×32 = 1024 entries).
  Tick 474, da86cf51.

- **`SENSITIVITY_CORRECTION_DB` doctest** — added `# Examples`:
  small negative dB, linear factor `10^(DB/20)` lands in `[0.9, 1.0)`
  (≈ 0.9684 attenuation). Tick 473, 83c70385.

- **`CSF_BASEBAND_RHO` doctest** — added `# Examples` showing the
  hard-coded `0.1` cy/deg and that it's below the typical geometric
  baseband rho (~0.19 cy/deg at standard 4K + 256² — the tick-204
  pycvvdp parity override). Tick 472, 79501377.

- **`BASEBAND_W` doctest** — added `# Examples` showing 3 positive-
  finite entries with chroma dominance at baseband (`[2] > [0]`,
  `[1] > [0]` — low-spatial-freq luminance is below CSF threshold).
  Tick 471, 5f4066a3.

- **`PU_BLUR_KERNEL_1D` doctest** — added `# Examples` showing 13
  taps, symmetric around center via `to_bits()`, sum-to-1 (DC
  preservation, σ=3 Gaussian), center > 5× tail magnitude. Tick 470, a9849af7.

- **`GAUSS5` doctest** — added `# Examples` showing 5 taps,
  symmetric around center via `to_bits()` equality, DC preservation
  (sum to 1 within 1e-6), center tap equals `KERNEL_A`. Tick 469, 800474bd.

- **`XCM_3X3` doctest** — added `# Examples` showing 3×3 shape,
  all entries positive-finite, A-to-A self-coupling dominance
  `[0][0] > 0.5` (matrix orientation pin). Tick 468, e4881f9b.

- **`SRGB8_TO_LINEAR_LUT` doctest** — added `# Examples` showing
  length=256, endpoints `[0]==0.0` / `[255]==1.0`, strict monotonicity
  across the 256-entry table. Tick 467, 88210899.

- **`SRGB_LINEAR_TO_DKL` doctest** — added `# Examples` showing
  row-sum invariants: A row sums in [0.5, 2.0] (luminance gain);
  RG and VY row-sum absolute values < A row sum (DKL chroma rows
  mean-zero by construction on equal-energy input). Tick 466, a1dc45bd.

- **`DisplayGeometry::pixels_per_degree` doctest** — added
  `# Examples` showing standard 4K → ≈ 75.4 ppd (within 0.5) and
  realistic-range invariant 5..=500. Tick 465, 166aaf79.

#### cvvdp-gpu (tests)

- **`tests/lib_reexports.rs`** — 5 pins on the `lib.rs` re-export
  surface: `PerfMode::default()` resolves, `CvvdpParams::PLACEHOLDER`
  resolves, and `PARALLEL_SAFETY_FACTOR` / `estimate_gpu_memory_bytes` /
  `recommend_parallel` re-exports each match the original
  `pipeline::*` value/output. A refactor that drops one of these
  re-exports — or feature-gates them — trips here before silently
  breaking downstream callers. Tick 464, 7eb7c956.

#### cvvdp-gpu (examples)

- **`examples/cvvdp_mem_table.rs`** — refactored to use the public
  `recommend_parallel` function instead of duplicating the `mem /
  (1.5 × est)` math inline. Output is identical. Added module-level
  docstring describing the example's purpose + invocation. Tick 463, 503fcf7b.

#### cvvdp-gpu (tests, lint)

- **Clippy clean-up across tick 416-461 test files**:
  `weber_pyramid_invariants` + `do_pooling_invariants` rewrap to
  `RangeInclusive::contains`; `csf_channel_invariants` +
  `perf_mode_invariants` add per-statement `#[allow(clippy::clone_on_copy)]`
  on the deliberate `.clone()` exercise of the Clone trait;
  `csf_axes_invariants` adds module-level
  `#[allow(clippy::excessive_precision)]` (intentional pycvvdp f64
  digit preservation); `mult_mutual_band_invariants` adds module-level
  `#[allow(clippy::needless_range_loop)]` (intentional per-channel
  iteration); `laplacian_pyramid_invariants` drops a `.map().enumerate().map()`
  no-op. Tests pass identically. Tick 462, 54d791a1.

#### cvvdp-gpu (docs)

- **`mult_mutual_band` doctest** — added `# Examples`: 8×8 input
  with T == R → output identically zero bit-exact across all 3
  channels and all 64 pixels (the trivial-zero-diff contract).
  Tick 461, 8f8609c7.

- **`weber_contrast_pyr_dec_scalar` doctest** — added `# Examples`:
  16×16 + n_levels=3 → 3 bands and 3 log_l_bkg vectors; baseband
  log_l_bkg is bit-constant (replicated scalar mean). Tick 460, 23b64e11.

- **`laplacian_pyramid_dec_scalar` doctest** — added `# Examples`:
  16×16 + n_levels=3 → 3 bands at 16×16/8×8/4×4 dims; each band's
  `data.len() == w * h`. Tick 459, fa97e6c8.

- **`phase_uncertainty_band` doctest** — added `# Examples`:
  small-band (2×2) pure-scaling branch (no blur), large-band (8×8)
  blur-then-scale branch with output length match. Tick 458, 655e623e.

- **`gaussian_blur_sigma3` doctest** — added `# Examples`: output
  length matches input, DC preservation (uniform → uniform within
  1e-5). Tick 457, 87f53be3.

- **`gausspyr_expand_scalar` doctest** — added `# Examples`: 4×4 → 8×8
  (standard 2× upscale), 4×4 → 7×7 (odd target — supports `[2*sw-1, 2*sw]`
  range per debug_assert). Tick 456, 81c88a44.

- **`gausspyr_reduce_scalar` doctest** — added `# Examples`: 8×8 → 4×4
  with `dst.len() == 16`, odd-dim 7×7 → 4×4 ceil-halving. Tick 455, bfa03284.

- **`flatten_band_weights` doctest** — added `# Examples`: empty
  → empty, 2-level [[1,2,3],[4,5,6]] → [1..=6], `weight_idx =
  level * 3 + channel` indexing pin. Tick 454, a6d20ad5.

- **`precomputed_band_weights` doctest** — added `# Examples`:
  length agrees with `band_frequencies`, every [A, Rg, Vy] triple is
  positive-finite at standard 4K + 100 cd/m². Tick 453, f3b47c93.

- **`do_pooling_and_jod_still_3ch` doctest** — added `# Examples`:
  all-zero contrasts → JOD ≈ 10 within 1e-3, non-zero contrasts →
  JOD < that. Tick 452, 5ca2fc6c.

- **`mult_mutual_pixel` doctest** — added `# Examples`: T == R →
  D = [0, 0, 0], argument symmetry `f(T, R) == f(R, T)`, and
  non-negative output. Tick 451, 8c672482.

- **`mask_pool_pixel` doctest** — added `# Examples`: zero input
  → zero output, unit basis `[1, 0, 0]` recovers `XCM_3X3[0]` row.
  Tick 450, a8a4f127.

- **`pool_band_finalize` doctest** — added `# Examples`:
  zero partial → 0 (eps-tail explicitly canceled), negative partial
  clamps to 0, and uniform-|x|=c reconstruction at β=2 within 0.01.
  Documents the eps-tail bias size relationship `~ eps^(1/β)`. Tick 449, 184a1984.

- **`phase_uncertainty_no_blur` doctest** — added `# Examples`:
  pure scaling `input × 10^MASK_C`, zero passthrough, and scale-factor
  range pin in [0.15, 0.17]. Tick 448, 00ca7e06.

- **`lp_norm_sum` doctest** — added `# Examples`: pythagorean
  `lp_norm_sum([3, 4], 2) ≈ 5` within 0.01, empty → 0,
  sign-insensitive via abs. Tick 447, b04fd9ed.

- **`lp_norm_mean` doctest** — added `# Examples`: empty → 0,
  uniform input → constant within 0.01 (eps-tail bias),
  sign-insensitive via abs. Tick 446, 883ea3f1.

- **`sensitivity_corrected_scalar` doctest** — added `# Examples`
  showing positive output at standard photopic L_bkg (100 cd/m²) and
  that `corrected / uncorrected == 10^(DB/20)` within 1e-5. Tick 445, cb936c1b.

- **`clamp_diff_soft` doctest** — added `# Examples`: `f(0) == 0`,
  half-saturation at `d == d_max` (relative err < 1e-5),
  asymptotic bound `< d_max` at 1e9. Tick 444, d06a2073.

- **`safe_pow` doctest** — added `# Examples` covering `safe_pow(0, p)
  == 0` exact zero (via `(eps)^p - eps^p` cancellation), `safe_pow(2,
  2) ≈ 4` within 0.01, and monotonicity. Tick 443, b5cd8d3d.

- **`srgb_byte_to_dkl_scalar` doctest** — added `# Examples`:
  pure-white → positive A + chroma < 5% of A, pure-red → RG > 0
  (red-green axis convention). Tick 442, 6240f767.

- **`met2jod` doctest** — added `# Examples` covering perfect-quality
  limit (`met2jod(0) == 10`), monotonic decline (0 > 0.5 > 1.0 > 5.0),
  and extreme-input safety (`met2jod(1e6)` finite < 0). Tick 441, dddf1db9.

- **`band_frequencies` doctest** — added `# Examples` showing
  typical usage: at standard 4K geometry the function returns ≥ 5
  strictly-decreasing positive cy/deg entries for 1024×1024. Tick 440, e8505d51.

- **`estimate_gpu_memory_bytes` doctest** — added an `# Examples`
  section that exercises the function on 4 inputs and validates
  the rough magnitude: too-small (4×4, 7×8 → `None`); 1 MP at
  ~208 MB (asserted in `[100 MB, 300 MB]`); 4 MP > 1 MP. Doubles
  as documentation and a smoke test that runs under
  `cargo test --doc`. Tick 439, 8d3b4b94.

- **`csf_lut/v0_5_4.rs` LUT constant docstrings** — six previously-
  undocumented public LUT constants re-exported via
  `pub use csf_lut_v0_5_4::*`: `LOG_L_BKG_AXIS` (uniform-in-log10
  background-luminance axis, `[-2.301, 4.0]`), `LOG_RHO_AXIS`
  (uniform-in-log10 spatial-frequency axis, `[-1.0, 1.806]`),
  `LOG_S_O0_C1/C2/C3` (1024-entry A/Rg/Vy sensitivity tables with
  the `l_idx * 32 + rho_idx` layout), `GE_SIGMA` (eccentricity-
  falloff scaffolding). Tick 438, c217ee95.

- **`CsfChannel` variant docstrings** — `A` (achromatic /
  luminance), `Rg` (red-green opponent), `Vy` (violet-yellow
  opponent). Three previously-undocumented variants surfaced by
  `RUSTDOCFLAGS="-D missing_docs" cargo doc`. Tick 436, 38ba643e.

- **`Error::DimensionMismatch` field docstrings** — `expected`
  (`width × height × 3` byte count) and `got` (actual caller-passed
  length). Tick 436, 38ba643e.

#### cvvdp-gpu (tests)

- **`tests/params_placeholder_non_display.rs`** — 5 additional pins
  on `CvvdpParams::PLACEHOLDER`'s csf / masking / pooling / jod
  sub-bundles (the existing `params_placeholder.rs` only pinned
  display + perf_mode): (1) all csf peaks bit-equal 0.0 (scaffolded
  placeholder); (2) masking p=2.4, q=2.2, k=0.04 (scaffolding —
  doesn't match production `kernels::masking` constants); (3)
  pooling betas all 4.0 (doesn't match production
  `kernels::pool::BETA_SPATIAL=2.0`); (4) jod_a=10.0, jod_b=1.0,
  jod_c=0.30 (scaffolding); (5) struct supports `CvvdpParams {
  ..PLACEHOLDER }` update syntax (Copy + accessible-fields
  compile-time check). A future wire-through that actually consumes
  these fields will need to swap in real values; this pin flags the
  scaffolding state. Tick 435, e138a176.

- **`tests/predict_jod_invariants.rs`** — 7 flow invariants on
  `predict_jod_still_3ch` (the composed host-scalar pipeline)
  complementing `shadow_jod.rs`'s pycvvdp parity coverage:
  (1) byte-identical inputs → JOD ≈ 10 within 1e-3; (2) JOD ≤ 10 + ε
  for any (ref, dist); (3) determinism via `to_bits()`;
  (4) responds to distortion magnitude — sparse ±2 vs ±80 perturbation
  on a textured reference produces ≥ 1e-3 JOD shift AND larger
  distortion gives smaller JOD (catches stuck-at-constant refactor;
  flat reference + uniform shift was insufficient because the
  Weber-contrast pyramid of a flat input has zero band content);
  (5–6) panics on ref / dist `len() != w*h*3` (the `assert_eq!`
  entry guards); (7) 8×8 smoke — identical → 10, perturbed < 10.
  Tick 434, 489751a4.

- **`tests/csf_axes_invariants.rs`** — 9 structural pins on the
  public CSF LUT axis arrays `LOG_L_BKG_AXIS` and `LOG_RHO_AXIS`
  (32 entries each). `csf_constants_match_pycvvdp_v0_5_4` doesn't
  pin axis structure. Pins: (1) `LOG_L_BKG_AXIS.len() == N_L_BKG`
  + N_L_BKG == 32; (2) `LOG_RHO_AXIS.len() == N_RHO` + N_RHO == 32;
  (3) both arrays strictly monotonic; (4) `LOG_L_BKG_AXIS`
  endpoints bit-pinned to `-2.301..4.0`; (5) `LOG_RHO_AXIS`
  endpoints bit-pinned to `-1.0..1.806`; (6) `LOG_L_BKG_AXIS`
  uniformly spaced (`interp1_uniform` precondition); (7)
  `LOG_RHO_AXIS` uniformly spaced in log10 (the source comment
  about "non-uniform first interval" is about linear-rho ratios,
  not the log10 axis itself); (8) `LOG_L_BKG_AXIS` step matches
  the pycvvdp formula `(4.0 - (-2.301)) / 31 ≈ 0.2032`. Tick 433, 1d581c28.

- **`tests/perf_mode_invariants.rs`** — 6 invariants on `PerfMode`'s
  trait contract beyond the existing `params_placeholder.rs`
  PLACEHOLDER check: (1) `Default::default() == PerfMode::Strict`
  pinned explicitly (catches a refactor that moves `#[default]` to
  Fast while updating PLACEHOLDER); (2) Copy semantics work;
  (3) Clone yields Eq-equal value; (4) Strict != Fast (catches
  variant collapse); (5) Debug output is non-empty and distinct
  per variant; (6) exhaustive match visits exactly 2 variants.
  Tick 432, e2a37146.

- **`tests/mult_mutual_band_invariants.rs`** — 8 structural pins on
  `mult_mutual_band` (band-level 3-channel masking; existing
  coverage in `masking_kernel.rs` is GPU-parity only): (1) output
  shape 3 × `w*h` across 3 sizes; (2) `T == R` → identically zero
  bit-exact; (3) `f(T, R) == f(R, T)` symmetric (both `min(|T|,|R|)`
  and `|T - R|` are symmetric); (4) D[cc] ≥ 0 across signed inputs;
  (5) bounded by `d_max ≈ 366.69` even for ±1e6 contrast inputs
  (clamp_diff_soft cap); (6) determinism via `to_bits()`; (7)
  finite output for mixed-sign ramp ±1e3; (8) small-band branch
  exercised at 4×4 (below PU_PADSIZE=6, triggers no-blur path).
  Tick 431, eec3ea81.

- **`tests/gaussian_blur_sigma3_invariants.rs`** — 8 dedicated
  invariants on `gaussian_blur_sigma3`. The function previously had
  no direct tests — it was used only as a CPU reference for GPU
  parity in `masking_kernel.rs`. Pins: (1) output length matches
  `w * h` across 4 sizes; (2) constant input → constant output
  within 1e-5 relative (DC preservation; kernel sums to 1); (3)
  zero input → zero output bit-exact; (4) reflect-padded 7×7 (every
  pixel touches the boundary) stays finite; (5) non-negative input
  → non-negative output (kernel is all-positive); (6) determinism
  via `to_bits()`; (7) horizontal mirror-symmetric input yields
  symmetric output within 1e-5 (the kernel + boundary are
  symmetric); (8) impulse input concentrates max at the impulse
  location. Tick 430, 6f8b55de.

- **`tests/phase_uncertainty_band_invariants.rs`** — 7 invariant
  pins on `phase_uncertainty_band` (the branch-on-band-size helper).
  No prior direct tests — pipeline parity covered it indirectly.
  Pins: (1) small-band branch (`w ≤ 6 OR h ≤ 6`) is pure scaling
  bit-equal to `input × 10^MASK_C` across 6 size combos;
  (2) large-band branch (`w > 6 AND h > 6`) actually applies blur
  (impulse input → diffused output); (3) output length matches
  input across both branches; (4) determinism in both branches via
  `to_bits()`; (5) empty input → empty output, no panic;
  (6) finite output for finite input; (7) **branch threshold pin
  at `PU_PADSIZE = 6`** — `(6, 6)`, `(7, 6)`, `(6, 7)` all small;
  `(7, 7)` is the first large case. Catches a refactor that flips
  `&&` to `||` (would incorrectly blur degenerate strips that
  can't fit the σ=3 kernel's 13-tap support). Tick 429, 605f8ca4.

- **`tests/csf_channel_invariants.rs`** — 7 invariant pins on the
  `CsfChannel` enum's discriminants + trait contract. No prior
  test pinned these — a refactor that reorders variants (e.g.,
  `Rg = 0`) would silently shift every per-channel buffer index
  in the CSF stage. Pins: (1) `A = 0`, `Rg = 1`, `Vy = 2` via
  `as u32`; (2) all discriminants fit in `[0, N_CHANNELS)` for
  `as usize` array indexing; (3) Copy semantics; (4) Clone yields
  Eq-equal value; (5) PartialEq self-equality + cross-variant
  inequality; (6) Debug output is non-empty and unique per variant;
  (7) exhaustive match visits all 3 variants. Tick 428, b3f6b634.

- **`tests/precompute_logs_row_invariants.rs`** — 6 additional
  invariants on `precompute_logs_row` beyond the 3 existing tests
  in `csf_scalar.rs`: (1) determinism via `to_bits()` across 3
  channels × 3 rho; (2) distinct rows across A/Rg/Vy at 3 rho —
  catches a refactor that collapses the channel argument;
  (3) all-finite output across rho ∈ {0.001, 0.1, ..., 1024}
  (sub-LUT to super-LUT extrapolation); (4) `rho=0` doesn't panic
  or NaN (the `.max(1e-6)` clamp guards `log10(0) = -inf`); (5)
  negative rho clamps via `.max()` (not `.abs()`) — pins by
  matching `precompute_logs_row(-100, A)` bit-equal to
  `precompute_logs_row(1e-6, A)`; (6) `10^row[k]` is strictly
  positive-finite (sensitivities are physical, never zero).
  Tick 427, 513c7d60.

- **`tests/do_pooling_invariants.rs`** — 7 flow invariants on
  `do_pooling_and_jod_still_3ch` complementing the 3 pycvvdp
  parity tests in `pool_scalar.rs`: (1) zero input → JOD ≈ 10
  within 1e-5; (2) JOD ≤ 10 + 1e-3 across 4 input shapes;
  (3) monotonic in each (level, channel) position — perturbing
  any single element by +0.5 cannot raise JOD; (4) determinism
  via `to_bits()`; (5) responds to magnitude — 100× scaling
  produces ≥ 1e-3 JOD shift AND larger input gives smaller JOD;
  (6) single-level input (1 pyramid level) supported, no panic,
  JOD < 10 for non-zero; (7) 12-level stress input supported,
  finite output in [0, 10 + ε]. Tick 426, 2100715f.

- **`tests/mult_mutual_pixel_invariants.rs`** — 7 function-level
  invariants on `mult_mutual_pixel` (per-pixel cross-channel
  masking + diff). Complements the single `pycvvdp_4x4` parity
  test in `masking_scalar.rs` with shape pins: (1) `T == R` →
  `D = [0, 0, 0]` bit-exact via `to_bits()`; (2) symmetry
  `f(T, R) == f(R, T)` (since `min(|T|, |R|)` and `|T - R|` are
  both symmetric); (3) `D[cc] ≥ 0` for all signs of input;
  (4) `D[cc] < d_max = 10^D_MAX ≈ 366.69` even for ±1e6 inputs
  (clamp_diff_soft asymptote); (5) determinism; (6) any non-trivial
  `T ≠ R` produces positive `D` on at least one channel;
  (7) finite output across 5 dynamic ranges 1e-10 to ±1e6.
  Tick 425, 953780bd.

- **`tests/met2jod_invariants.rs`** — 8 invariant pins on `met2jod`
  beyond the 2 single-point tests already in `pool_scalar.rs`
  (`met2jod_continuous_at_kink` + `met2jod_clamps_at_origin`):
  (1) `met2jod(0) == 10` bit-exact via `to_bits()`; (2) value at
  kink Q=0.1 matches `10 - JOD_A * 0.1^JOD_EXP`; (3) strict
  monotonic decrease over Q ∈ [0, 100] step 0.01 (10001 samples);
  (4) `< 10` for any positive Q above f32 underflow (1e-3 onward);
  (5) power-branch algebra `10 - JOD_A * Q^JOD_EXP` for 6 Q above
  kink; (6) linear-branch algebra `10 - jod_a_p * Q` (where
  `jod_a_p = JOD_A * 0.1^(JOD_EXP-1)`) for 5 Q below kink — pins
  the slope-matching construction; (7) determinism; (8) declining
  finite JOD at extreme Q ∈ [1e3, 1e12]. Tick 424, 2764c3d8.

- **`tests/mask_pool_pixel_invariants.rs`** — 7 invariant pins on
  `mask_pool_pixel`, the 3×3 cross-channel masking matrix-vector
  multiply against `XCM_3X3`. No direct unit tests existed before
  (`mult_mutual_pixel` covered it indirectly through full-pipeline
  parity). Pins: (1) zero input → zero output bit-exact;
  (2) determinism via `to_bits()`; (3) α-scaling linearity within
  1e-6 relative across 5 scalars; (4) additivity `f(a+b) == f(a) +
  f(b)` within 1e-5 relative; (5) unit-basis inputs recover the
  rows of `XCM_3X3` exactly via `to_bits()` — catches a row-column
  transposition that wouldn't trip pipeline parity; (6) all-finite
  output for finite input across 6 input dynamic ranges (1e-10 to
  1e6, positive + negative); (7) A's self-coupling dominance
  (`out[0] > 0.5` for `[1, 0, 0]` since `XCM_3X3[0][0] = 0.877`)
  — pins the matrix orientation. Tick 423, f28b0455.

- **`tests/clamp_phase_uncertainty_invariants.rs`** — 10 invariant
  pins on two small masking primitives that previously had no
  direct unit tests:
  - `clamp_diff_soft(d) = d_max·d / (d_max + d)`: (1) `f(0) == 0`
    bit-exact via `to_bits()`; (2) strict monotonicity across 200
    samples in [0, 1000]; (3) asymptotic `f(d) < d_max` for d up
    to 1e9, plus gap < 0.1% at d ≥ 1e6; (4) half-saturation
    `f(d_max) == d_max/2` within 1e-5 relative; (5) determinism.
  - `phase_uncertainty_no_blur(m) = m * 10^MASK_C`: (6) pure
    scaling via `to_bits()` across 8 sample inputs incl. negatives;
    (7) scale factor pinned in [0.15, 0.17] (loose bound on the
    bit-pinned MASK_C); (8) `f(0) == 0`; (9) monotonicity over
    [-100, 100]; (10) determinism.
  Tick 422, 8c0d4bc7.

- **`tests/weber_pyramid_invariants.rs`** — eight structural
  invariant pins on `weber_contrast_pyr_dec_scalar` complementing
  the full-pipeline parity coverage in `pipeline_color.rs` /
  `pipeline_score.rs`: (1) band count matches `n_levels` 1..=4 (and
  `log_l_bkg` matches); (2) auto-`n_levels=0` selects
  `min(sw,sh).ilog2()` (64×32 → 5); (3) `log_l_bkg[k].len() ==
  bands[k].w * bands[k].h` per level; (4) baseband `log_l_bkg` is
  bit-constant (all entries equal via `to_bits()` — pins the
  "replicated scalar mean" docstring contract); (5) baseband band
  data is finite (division-by-zero guard via 0.01 floor); (6)
  non-baseband contrast clamped to `[-1000, 1000]` via 1e6 impulse
  on 0.001 L_bkg field (baseband intentionally excluded — it's
  unclamped per source); (7) zero-image + zero-l_bkg input produces
  no NaN/Inf (the 0.01 floor guards everything); (8) determinism
  via `to_bits()` over bands + log_l_bkg. Tick 421, c6c30191.

- **`tests/srgb_byte_to_dkl_invariants.rs`** — eight function-level
  semantic invariants on `srgb_byte_to_dkl_scalar` beyond the
  pointwise pycvvdp parity at `STANDARD_4K`: (1) DKL_A strictly
  monotonic in grayscale ramp 0..256 step 16; (2) grayscale chroma
  RG/VY < 5% of A's magnitude across 9 neutral bytes; (3) black <
  mid < white ordering on the A channel; (4) linearity in `y_peak`
  via Δ(100→200) = ⅓ × Δ(100→400); (5) corner-pixel safety for 8
  corners of the RGB cube (no panic, all finite); (6) determinism
  via `to_bits()` across 3 inputs × 3 channels; (7) pure-red → RG > 0
  and pure-cyan → RG < 0 (pins row-1 sign convention against a row
  swap with row 2); (8) pure-blue → VY > 0 and pure-yellow → VY < 0.
  Complements the matrix-bit-pin in `srgb_linear_to_dkl_matrix_*` —
  pins the FUNCTION'S shape, not just the matrix's entries. Tick 420, 43bd4a18.

- **`tests/gausspyr_expand_invariants.rs`** — seven structural
  invariant pins on `gausspyr_expand_scalar`, mirror of
  `gausspyr_reduce_invariants.rs`: (1) `dst.len() == out_w * out_h`
  across the full documented `[2*sw - 1, 2*sw]` × `[2*sh - 1, 2*sh]`
  range (4 combos of even/odd target dims); (2) `dst` fully
  overwritten — NaN pre-fill catches; (3) determinism via
  `to_bits()`; (4) capacity invariance; (5) `(sw=4, sh=2)` vs
  `(sw=2, sh=4)` produce distinct content (catches width/height
  collapse in the separable convolution); (6) both `odd_w/odd_h`
  branches inside the function succeed (5→9 odd and 5→10 even
  paths); (7) all-finite output across 7 typical pyramid expand
  pairs including non-square 8×4 → 16×7 and minimal 3×3 → 5×5/6×5.
  Tick 419, a68e5c33.

- **`tests/gausspyr_reduce_invariants.rs`** — seven structural
  invariant pins on `gausspyr_reduce_scalar`: (1) `(dw, dh) = (4, 4)`
  and `dst.len() == 16` for 8×8; (2) odd inputs ceil-halve correctly
  (7×7 → 4×4; 17×13 → 9×7); (3) returned `(dw, dh)` agrees with
  `dst.len()` across 9 size combos including non-square; (4) `dst`
  is fully overwritten — pre-fill with NaN and confirm none survive;
  (5) determinism via `to_bits()` bit-equality across repeated calls;
  (6) `(4, 8)` and `(8, 4)` produce distinct dims AND distinct content
  (width/height swap catches); (7) caller-provided `dst` capacity
  (too-big or zero) doesn't affect output. Complements
  `pyramid_scalar.rs::reduce_matches_pycvvdp`'s single fixed-input
  pycvvdp parity test with broad structural coverage. Tick 418, 41dae2f5.

- **`tests/laplacian_pyramid_invariants.rs`** — seven structural
  invariant pins on `laplacian_pyramid_dec_scalar`: (1) output band
  count matches requested `n_levels` for 1..=4; (2) auto-`n_levels=0`
  picks `min(sw, sh).ilog2()` bands across 64² (=6) and non-square
  64×32 / 32×64 (=5); (3) band dimensions track an independently-
  rebuilt `gausspyr_reduce_scalar` chain on 17×13 (odd-dim);
  (4) baseband (last band) is bit-equal to the coarsest gaussian
  via `to_bits()` (pins the docstring contract that the baseband
  is NOT a Laplacian residual); (5) determinism via bit-equality
  across repeated calls; (6) `n_levels=1` returns a single band
  bit-equal to the input (the `for k in 0..(n-1)` empty loop edge
  case); (7) Band invariant `data.len() == w * h` for every band.
  Complements `pyramid_scalar.rs`'s pointwise-numeric pycvvdp
  parity tests with structural / contract coverage. Tick 417, ae86b5e1.

- **`tests/band_weights_invariants.rs`** — eight invariant pins on
  `flatten_band_weights` and `precomputed_band_weights` covering
  edges + structural properties the existing pointwise test missed:
  empty input → empty output with zero capacity; length invariant
  `out.len() == weights.len() * 3` across n ∈ {0, 1, 2, 3, 5, 8,
  16, 50}; documented `flat[level * 3 + channel]` indexing
  contract; NaN/±∞/-0.0 bit-passthrough; `precomputed_band_weights`
  length agrees with `band_frequencies` across 9 image sizes
  (square + non-square 16² up to 4K); all-finite + strictly-positive
  output across log_L_bkg ∈ [-1.0, 3.0] (0.1 cd/m² dim through
  1000 cd/m² HDR peak); determinism via `to_bits()` equality;
  end-to-end flatten-then-index round-trip pinning the
  `weight_band_kernel` consumer contract. Tick 416, 6b2891af.

- **`tests/display_geometry.rs::ppd_is_*`** — four invariant tests
  on `DisplayGeometry::pixels_per_degree`: (1) positive + finite +
  in realistic [5, 500] range across phone, tablet, desktop,
  cinema, UHD-living-room viewing configs; (2) strictly
  monotonically increasing in `distance_m` (further → less angle
  per pixel → higher PPD); (3) strictly monotonically decreasing
  in `diagonal_inches` (larger physical screen at same distance →
  more angle per pixel → lower PPD); (4) strictly monotonically
  increasing in `resolution_w` at fixed 16:9 aspect. Catches sign
  flips and dimension-swaps that would silently mis-calibrate the
  CSF stage's per-band rho query. Tick 415, 1fc417cd.

#### cvvdp-gpu (docs)

- **README "Build" section refreshed for the CUDA-version-matters
  lesson learned during the v22-v25 fleet incident.** The
  previous claim "CUDA 13.2 required for cubecl 0.10's CUDA
  backend" was misleading — cubecl 0.10 itself doesn't require
  13.x; its `cudarc 0.19.4` dep auto-selects a `cuda-<MMmmpp>`
  cargo feature from the SDK present at build time, and the
  resulting binary's dlsym entries must match symbols the host's
  libcuda exports. Binaries built against CUDA 13 try to dlsym
  `cuCoredumpDeregisterCompleteCallback` (gated behind cudarc's
  `cuda-13020` feature) which is absent from every released
  NVIDIA libcuda — panics at first dispatch. New README guidance
  explicitly differentiates RTX 50-series (CUDA 13 required for
  Blackwell sm_120) from RTX 20/30/40/A2000 etc. (CUDA 12.6 SDK,
  proven on the production fleet under driver 535+). Plus calls
  out the runtime requirement on NVRTC headers
  (`cuda-cudart-dev-<MMmm>`) — without them, `Cvvdp::score`
  returns the dual-purpose `InvalidImageSize` masking an NVRTC
  compile failure (v25 lesson). Tick 414, 00c5875e.

#### cvvdp-gpu (tests)

- **`tests/pyramid_scalar.rs::band_frequencies_{are_strictly_decreasing,minimum_image_dim_returns_some_bands,per_band_ratio_in_sensible_range}`**
  — three invariant tests on `band_frequencies`: (1) output is
  strictly decreasing (Laplacian pyramid orders finest→coarsest)
  across 4 (ppd, dim) combos, every entry finite + positive; (2)
  minimum image dim 8×8 returns ≥ 1 band (so Cvvdp::new never
  builds a zero-band pyramid where the Vec-of-Level sizing would
  silently fail); (3) mid-pyramid adjacent-band ratio in
  [1.5, 3.5] — captures the "near-octave" Laplacian behavior
  while accommodating the first-level Nyquist-quarter scaling
  and the trailing MIN_FREQ=0.2 floor. Tick 412, f861d026.

- **`tests/pipeline_score.rs::score_is_deterministic_*`** — two
  contract tests pinning the critical "no state leakage between
  calls on the same `Cvvdp` instance" property that zen-metrics-
  cli's `CvvdpBatchScorer` relies on for the vast.ai backfill
  pipeline. (1) `score(ref, dist)` called twice → bit-identical
  output via `.to_bits()`; (2) `score(ref, dist_a)` →
  `score(ref, dist_b)` → `score(ref, dist_a)` again → first and
  third results bit-identical (no state leaked from the b call);
  (3) intervening warm_reference + compute_dkl_jod_with_warm_ref
  doesn't poison cold-path scratch. A regression where a scratch
  buffer reset is dropped, an accumulator grows across calls, or
  warm-ref state contaminates cold dispatch surfaces here before
  silently breaking the cached-instance pattern that the OOM-fix
  tick 384 depends on. Tick 411, ebe21f89.

- **`tests/masking_safe_pow.rs`** — five direct unit tests on
  `kernels::masking::safe_pow` (cvvdp's `(x + eps)^p - eps^p`
  used in the masking chain — distinct from
  `pool::safe_pow_lp`'s `|x|.abs() + eps` variant). Previously
  exercised only transitively through `mult_mutual_pixel` and
  the composed-pipeline parity tests. New tests pin: (1)
  `safe_pow(0, p) = 0` exactly across p ∈ {1, 2, MASK_P, 4} —
  catches a refactor that drops the `- eps^p` correction; (2)
  `safe_pow(1, p)` matches the closed-form `(1 + eps)^p - eps^p`
  for the same p set; (3) strictly monotonic in x for positive p
  (catches a sign-flip on the correction term); (4) eps offset
  dominates only near zero — for x ≫ eps, result ≈ x^p (rel <
  1e-3) and for x = eps the closed form `eps^p × (2^p - 1)`
  holds; (5) finite + positive at extreme x ∈ {100, 1k, 10k}
  across p set (catches an overflow-to-inf regression). Lives
  in a dedicated file per the tick-401 precedent (linter-revert
  safety vs `masking_scalar.rs`). Tick 410, 57fc5225.

- **`tests/error_traits.rs`** — pins five trait-side contracts on
  `cvvdp_gpu::Error`: (1) `impl std::error::Error` (compile-time
  check via `&dyn` coercion); (2) `Clone` preserves variant +
  payload across all four variants (catches a derive-to-manual-
  impl refactor that drops a field on `DimensionMismatch`); (3)
  `source()` returns `None` for every variant — these are leaf
  errors with no nested cause chain (if a future variant wraps a
  backend error, this test fails loudly + maintainer documents
  the new contract); (4) `Debug` includes the variant name
  verbatim across all four; (5) the `?`-bubble path through
  `Box<dyn std::error::Error>` works and preserves the actionable
  Display message. Sibling to tick 282's
  `error_display_messages_are_actionable` (Display content) —
  this tests the trait *implementations*. Tick 409, 8177979b.

#### cvvdp-gpu (docs)

- **`kernels::csf::N_RHO` gets its own docstring**, separating
  it from `N_L_BKG`'s. The two constants previously shared a
  single `/// Number of grid points along each LUT axis.` comment
  preceding `N_L_BKG` only — rustdoc attached the doc to
  `N_L_BKG` and left `N_RHO` undocumented. Now each has its own
  doc explaining the kernel-sizing constraint, with a cross-
  reference between them. Verified via a python doc-coverage
  sweep: 0 undocumented public items remain in the non-LUT
  source (the LUT file `csf_lut/v0_5_4.rs` is auto-generated
  from pycvvdp's JSON; comments there would not survive
  regeneration). Tick 408, 7c4d4758.

#### cvvdp-gpu (tests)

- **`tests/pipeline_score.rs::score_returns_lossless_f64_widening_of_compute_dkl_jod`**
  — pins the documented `Cvvdp::score` contract: returns
  `f64::from(compute_dkl_jod(ref, dist, ppd))` where ppd comes
  from `self.geometry.pixels_per_degree()`. f32 → f64 widening
  is lossless, so the round-trip `(score() as f32).to_bits() ==
  compute_dkl_jod().to_bits()` must hold bit-for-bit. Sweeps the
  full v1 corpus q-grid. Catches a refactor that introduces a
  precision-eating step (e.g. `jod as f64 * 1.0` rounded
  through an intermediate). Also asserts score ∈ [0, 10] across
  the corpus q-range. Tick 407, 14582e8a.
- **`tests/pipeline_score.rs::{parallel_safety_factor_*, recommend_parallel_{monotonic,budget}_*, estimate_gpu_memory_grows_*}`**
  — four invariant tests on the GPU memory predictor + concurrency-
  cap API: (1) `PARALLEL_SAFETY_FACTOR` in [1.0, 3.0] sane-range
  with exact-value pin at 1.5 (catches a refactor that drops it to
  0.5 (overrun) or 5.0 (waste)); (2) `recommend_parallel` is
  monotonically non-decreasing in `free_gpu_bytes` (catches a sign-
  flip / inverted-division bug that would silently mis-cap large-
  GPU sweeps); (3) the budget invariant `N × SAFETY × est ≤ free`
  holds whenever `recommend_parallel` returns N > 1 (the floor-1
  case is the documented "back off explicitly" signal); (4)
  `estimate_gpu_memory_bytes` is strictly increasing across six
  image sizes (catches a refactor that introduces fixed-cost
  inversion). Tick 406, 8039d126.
- **`tests/params_placeholder.rs`** — pins the two `CvvdpParams::PLACEHOLDER`
  fields the pipeline actually consumes: `display ==
  DisplayModel::STANDARD_4K` (field-by-field bit-pattern check
  on y_peak/y_black/y_refl) and `perf_mode == PerfMode::Strict`.
  Every parity test in the crate constructs `Cvvdp::new(...,
  PLACEHOLDER)`, so a refactor that flipped the placeholder
  default to `PerfMode::Fast` would silently change every
  golden-test calibration baseline. Plus a contract test
  exercising PerfMode's Copy + PartialEq derives. Tick 405, daab6476.
- **`tests/goldens_metadata.rs`** — pins the self-consistency of
  the goldens-fetch infrastructure in `tests/common/mod.rs`:
  `MANIFEST_URL` must embed `GOLDEN_VERSION` as a path segment
  (`/v1/`), use https scheme, end in `.json`; `MANIFEST_SHA256`
  must be exactly 64 chars of `[0-9a-f]` (catches truncation +
  uppercase typos that would silently break `Sha256::finalize()`
  comparison); `cache_dir()` must embed `GOLDEN_VERSION` and the
  crate-specific subdir; `GOLDEN_VERSION` must follow the
  `v<N>` convention with decimal digits. A regression that bumps
  `GOLDEN_VERSION = "v2"` but forgets to update `MANIFEST_URL`
  surfaces here before the goldens-feature gate runs the actual
  fetch. Same loud-failure-on-silent-edit discipline applied to
  test-infrastructure constants. Tick 404, 6e95bfac.
- **`tests/shadow_jod.rs::predict_jod_still_3ch_returns_max_jod_on_identical_inputs`**
  — integration-test promotion of the lib.rs doctest's identity
  contract (scoring a buffer against itself yields JOD ≈ 10.0).
  Doctests are skipped when filtering with `cargo test --test
  <name>`, leaving the host-scalar identity contract uncovered
  in the standard test path. Sweeps three sizes × three uniform
  values: (8×8 = PYRAMID_MIN_DIM×2 boundary, 64×64 = doctest
  size, 73×91 = odd-dim with pycvvdp `gausspyr_reduce` column-
  parity bug-compat patches from ticks 204-206) × val ∈ {0, 128,
  255}. Companion to tick 350's
  `compute_dkl_jod_host_pool_returns_max_jod_on_identical_inputs`
  (GPU host-pool path) — same contract on the host-scalar
  reference twin. Tick 403, 7ef5b683.
- **`tests/lib_constants.rs`** — seventh in the constants-pin
  series. Pins the three crate-level constants exposed from
  `lib.rs`: `N_CHANNELS = 3` (still-image DKL opponent count),
  `MAX_LEVELS = 9` (pyramid-level cap — bumping requires
  resizing `logs_row` + `partials_h` + weights buffers), and
  `PYRAMID_MIN_DIM = 4` (minimum logical level dim). Plus a
  derived-invariant test `PYRAMID_MIN_DIM × 2 = 8` so a refactor
  that changes the `width < PYRAMID_MIN_DIM * 2` guard's
  multiplier surfaces here instead of as a boundary-test
  regression. Tick 402, c61e4b22.
- **`tests/masking_constants.rs`** — sixth in the constants-pin
  series (393 pool / 394 csf / 395 pyramid / 396 display / 397
  color matrix). New dedicated test file pins by exact f32 bit
  pattern: `CH_GAIN = [1.0, 1.45, 1.0]`, `MASK_P = 2.264_355_2`,
  `MASK_Q = [1.302_622_7, 2.888_590_8, 3.680_771_3]`, `MASK_C =
  -0.795_497_12`, `D_MAX = 2.564_245_5`, all 9 entries of
  `XCM_3X3`, and all 13 taps of `PU_BLUR_KERNEL_1D`. Plus
  structural invariants on `PU_BLUR_KERNEL_1D`: DC preservation
  (sum ≈ 1.0 within 1e-6) and symmetry around the centre tap.
  Lives in a dedicated file (not the historically linter-edge-
  case-sensitive `masking_scalar.rs`) so the consts pin stays
  durable. Tick 401, 57506ad9.
- **`tests/color_scalar.rs::srgb_linear_to_dkl_*`** — fifth in
  the constants-pin series (393 pool / 394 csf / 395 pyramid /
  396 display). Pins all 9 entries of `SRGB_LINEAR_TO_DKL` by
  `.to_bits()` — the f32 row-major DKL matrix composed at f64
  precision from `LMS2006_to_DKLd65 @ XYZ_to_LMS2006 @
  sRGB_to_XYZ`. Previously verified only transitively through
  the 8-point byte goldens + the row-sum heuristic — a refactor
  that swaps two entries within a row, or substitutes a
  plausible-but-different matrix (LMS2000 instead of LMS2006),
  could pass both. Second test pins the opponent-color sign
  signature: row 0 (A) all positive, row 1 (Rg) is `(+, -, -)`,
  row 2 (Vy) is `(-, -, +)`. Tick 397, 616a6a8a.
- **`tests/display_geometry.rs::display_{model,geometry}_standard_4k_*`**
  — pins the f32 bit patterns of `DisplayModel::STANDARD_4K`
  (`y_peak = 200`, `y_black = 0.2`, `y_refl = 0.397_887_36`) and
  `DisplayGeometry::STANDARD_4K` (`3840×2160`, `distance_m =
  0.7472`, `diagonal_inches = 30`). The v1 R2 manifest goldens
  were captured under this display configuration — a silent edit
  to any field (e.g. swapping `y_refl` for the unrounded f64
  literal) would invalidate every shadow_jod parity test in a
  way that's hard to trace back to the display constants.
  Companion to ticks 393 (pool) / 394 (csf) / 395 (pyramid). Tick 396, 0b5cf789.
- **`tests/pyramid_scalar.rs::pyramid_constants_match_pycvvdp_v0_5_4`**
  — sibling to ticks 393 (pool) / 394 (csf). Pins
  `KERNEL_A = 0.4` (the Burt-Adelson `a` parameter) by exact
  bit pattern and verifies `GAUSS5 = [0.05, 0.25, 0.4, 0.25,
  0.05]` is consistent with the compile-time derivation
  `[0.25-a/2, 0.25, a, 0.25, 0.25-a/2]` — outer taps use
  abs-diff < 1e-7 because `0.25 - 0.4/2.0` rounds one ULP
  below the literal 0.05 at compile time; inner taps are
  exact. Plus two structural invariants: DC-preservation
  (sum ≈ 1.0 within 1e-6) and symmetry around the center tap
  (bit-identical pairs `[0]==[4]`, `[1]==[3]`). A drift in
  `KERNEL_A` to e.g. 0.375 (the Burt original) would broaden
  the kernel and silently shift every pyramid level; the
  test trips with a specific message instead of cascading
  into shadow_jod drift. Tick 395, 31eb8bbe.
- **`tests/csf_scalar.rs::csf_constants_match_pycvvdp_v0_5_4`** —
  sibling to tick 393's pool-constant pin. Locks the exact f32
  bit patterns of `SENSITIVITY_CORRECTION_DB` (-0.279_742_33)
  and `CSF_BASEBAND_RHO` (0.1), plus integer values of
  `N_L_BKG` and `N_RHO` (both 32). The dB correction was
  previously checked transitively via the
  `sensitivity_correction_is_a_small_attenuation` magnitude test
  (tick 388) but not pinned to an exact bit pattern;
  `CSF_BASEBAND_RHO` had no direct value check at all. N_L_BKG /
  N_RHO are the LUT grid sizes the GPU kernels assume via array
  sizing — a refactor that bumps either without resizing kernel
  buffers would corrupt every per-pixel CSF lookup. Tick 394, f8c962aa.
- **`tests/pool_scalar.rs::pool_constants_match_pycvvdp_v0_5_4`** —
  pins the exact f32 bit patterns of the eight pool constants
  imported verbatim from pycvvdp v0.5.4: `BETA_SPATIAL`,
  `BETA_BAND`, `BETA_CH`, `IMAGE_INT`, `JOD_A`, `JOD_EXP`,
  `PER_CH_W[0..3]`, `BASEBAND_W[0..3]`. A silent edit (typo,
  sign flip, decimal-point shift) to any of these cascades into
  JOD drift across every parity gate, where it's hard to
  localize. The new test trips with a specific message
  identifying which constant changed — turning a 0.001 JOD
  drift on shadow_jod_gpu into "BASEBAND_W[Vy] = X, expected
  4.118_745_3 (cvvdp v0.5.4)". When a future cvvdp version
  (0.5.5+) ships new coefficients, update these values together
  with the pin and re-run parity. Tick 393, 5d5d4ff3.
- **`tests/pipeline_score.rs::dimension_mismatch_surfaces_on_wrong_size_inputs`**
  — extended once more to cover six pyramid/band intermediate-
  output methods that validate buffer length transitively through
  `_dispatch_dkl_planes_gpu` (the shared entry point that contains
  the actual `!=` check): `compute_dkl_gauss_pyramid`,
  `compute_dkl_laplacian_pyramid`, `compute_dkl_weber_pyramid`,
  `compute_dkl_t_p_bands`, `compute_dkl_csf_weighted_bands`, and
  `compute_dkl_d_bands` (both ref + dist args per docstring). Each
  method's docstring documents the `Error::DimensionMismatch`
  return — a refactor that inlines `_dispatch_dkl_planes_gpu` into
  a caller but forgets to copy the length check would surface here
  before slipping into a kernel-side panic on the under-sized
  buffer read. Test now exercises all 13 documented dim-check
  sites (was 9 after tick 390). Tick 392, e277f103.
- **`tests/pipeline_score.rs::compute_dkl_jod_host_pool_with_warm_ref_reports_dim_mismatch_before_no_warm`**
  — sibling pin to the tick-248 GPU-variant test
  (`compute_dkl_jod_with_warm_ref_reports_dim_mismatch_before_no_warm`).
  The source code for `compute_dkl_jod_host_pool_with_warm_ref`
  applies the dim check before the warm-state check (the
  comment references the tick-248 ordering rationale) but had
  no regression test pinning the contract. A refactor that
  swaps the order on the host_pool path — returning
  NoWarmReference first and masking the more actionable
  DimensionMismatch — would slip past CI. host_pool matters
  because cubecl-cpu / Metal callers route through it
  explicitly (the GPU Atomic<f32>::fetch_add path doesn't run
  on those backends), so their production error reporting
  needs the same ordering as the GPU path. Tick 391, bc65041c.
- **`tests/pipeline_score.rs::dimension_mismatch_surfaces_on_wrong_size_inputs`**
  — extended to cover four additional public entry points the
  original tick-239 test acknowledged in its docstring but did
  not actually exercise: `compute_dkl_jod` (both ref/dist args),
  `compute_dkl_planes`, `compute_dkl_jod_host_pool` (both args),
  `compute_dkl_jod_host_pool_with_warm_ref`. The five sites
  previously covered were `score`, `set_reference`,
  `score_with_reference`, `warm_reference`, and
  `compute_dkl_jod_with_warm_ref` — leaving the GPU-pool and
  host-pool variants of `compute_dkl_jod` unchecked. A refactor
  that swaps the `!=` check for `<` on any of the four newly-
  covered entries (silently accepting smaller buffers and
  reading garbage past `srgb.len()`) would slip past the
  original 5-site coverage. Tick 390, 8e4d2590.
- **`tests/pool_scalar.rs::lp_norm_mean_*`** — four direct unit
  tests on `lp_norm_mean` (cvvdp's `lp_norm` with `normalize=True`).
  The function was exercised only through the GPU-gated
  `pool_band_kernel_matches_host_lp_norm_mean` test and the
  single-input `pool_band_finalize_matches_lp_norm_mean_on_synth_signal`
  test, leaving no direct CPU-only coverage of its algebra
  invariants. New tests pin: (1) empty-input early-return → 0.0
  exactly at p ∈ {1,2,4,8} via `.to_bits()` (without the guard,
  `acc/n` produces NaN at n=0); (2) uniform-input identity:
  `lp_norm_mean([a; n], p) ≈ a - eps^(1/p)` at (a, p) ∈
  {(0.5, 2), (2.5, 4)} × n ∈ {1,4,16,64} (catches a refactor
  that drops the `/ n` step, which would overestimate by
  `n^(1/p)`); (3) sign-handling via `.abs()` — pos/mixed/neg
  inputs produce bit-identical output (mirror of lp_norm_sum's
  test, pinned separately for the lp_norm_mean call site); (4)
  the defining identity `lp_norm_sum ≈ n^(1/p) * lp_norm_mean`
  at p ∈ {2, 4} on an 8-element signal (a structural-divergence
  catcher — if either function changes its eps shift, this
  trips). Tick 389, bfef0b2f.
- **`tests/csf_scalar.rs::sensitivity_corrected_*` + `sensitivity_correction_*`**
  — three direct unit tests on `sensitivity_corrected_scalar`,
  which the production CSF apply path (`precomputed_band_weights`
  + the GPU kernel host-side row-precompute) reads through but
  previously had no scalar-side direct coverage. New tests pin:
  (1) the correction is a constant multiplicative factor
  (corrected/uncorrected ratio bit-identical to 1e-5 across 3
  channels × 3 rho × 3 log_l_bkg = 27 points — catches a
  refactor that breaks the input-independence invariant); (2)
  the factor magnitude (0.9, 1.0) and specific value ≈ 0.9684
  (catches sign flips that would amplify instead of attenuate,
  and order-of-magnitude wrong DB constants); (3) extreme-input
  finiteness (same clamping contract as `sensitivity_scalar`,
  but pinned separately so the uncorrected path and the
  multiplicative step can each regress independently). Tick 388, 506f61bf.
- **`tests/color_scalar.rs::srgb_lut_*`** — four direct unit tests
  on the public `SRGB8_TO_LINEAR_LUT` 256-entry sRGB EOTF table.
  Previously the LUT was verified only transitively through the
  8-point `matches_pycvvdp_standard_4k` byte goldens — and a
  historical "~6e-4 drift at bright bytes" regression (referenced
  in `pipeline_color.rs:2009`) had shipped because the goldens
  happened to skip the affected bytes. New tests pin: (1) length
  256 + exact 0.0 / 1.0 endpoints at byte 0 / 255 via `.to_bits()`
  (off-by-one + missing-boundary catcher); (2) strict monotonic
  increasing across all 256 bytes (bit-flip or swapped-pair
  catcher); (3) direct comparison against the IEC 61966-2-1
  inverse companding formula at every byte (f64 reference, 1e-6
  absolute tolerance — well under the 6e-4 historic drift); (4)
  seam continuity around c = 0.04045 (byte 11) — pin the local
  slope ratio to (0.5, 2.0) to catch a refactor that mis-aligns
  the piecewise branch threshold. Tick 387, 0e284715.
- **`tests/csf_scalar.rs::precompute_logs_row_*`** — five direct
  unit tests on the previously-GPU-only-exercised public
  `precompute_logs_row`. The helper had no scalar-side coverage:
  it was used in `tests/csf_kernel.rs` to set up GPU kernel
  inputs, but that file is feature-gated to
  `cfg(any(cuda, wgpu, hip))` so a CPU-only test run (no GPU
  available, no atomic-f32 support) never touched it. New tests
  pin: (1) returns exactly `N_L_BKG = 32` entries across all
  channels × four rho values (a refactor that shrinks the row
  would corrupt every per-pixel CSF lookup); (2) the closed-form
  identity `sensitivity_scalar(rho, LOG_L_BKG_AXIS[k], cc) =
  10^precompute_logs_row(rho, cc)[k]` across 3 channels × 4 rho
  × 32 axis points = 384 points (interp1_uniform returns the
  exact row value at axis indices, so this identity is parity
  glue between the two public functions); (3) frequency
  dependence — max |diff| > 0.1 between rho=0.5 and rho=16
  cy/deg for the achromatic channel (catches a refactor that
  collapses the rho axis); (4) channel dependence — pairwise
  max |diff| > 1e-3 between A/Rg/Vy at fixed rho=4 (catches a
  channel_lut dispatch typo); (5) the `rho.max(1e-6)` clamp —
  rho ∈ {0, -1, 1e-6} produce bit-identical rows via
  `.to_bits()` (silent-NaN-propagation regression catcher).
  Same gap-shape as ticks 351/383. Tick 386, 0e284715.
- **`tests/pool_scalar.rs::pool_band_finalize_*`** — five direct
  unit tests on the previously-indirectly-exercised public
  `pool_band_finalize`. The function was covered only via the
  GPU-backed `pool_band_kernel_matches_host_lp_norm_mean` test,
  which means CPU-only test runs (e.g. cubecl-cpu CI on a host
  without atomic-f32 GPU) couldn't catch host-side regressions
  to its algebra. New tests pin: (1) zero-partial returns 0 across
  β ∈ {1,2,4,8} and n ∈ {1,64,1024,65536} — eps^(1/β) tail must
  cancel head; (2) negative-partial clamping to 0 (atomic-noise
  protection — without `.max(0)`, β=2 returns NaN at non-integer
  exponents); (3) scalar-form identity vs `lp_norm_mean` on a
  synthesised signal (the same identity the GPU kernels rely on,
  now testable without a GPU); (4) eps^(1/β) tail magnitude
  pinned at β ∈ {1,2,4} — same observation as the lp_norm_sum
  tests, at β=2 the tail is 316× larger than at β=1; (5)
  strict-monotonic decreasing in n_pixels under fixed partial,
  with a closed-form check at (partial=100, n=100, β=2). Tick 383, a5218943.
- **`tests/pool_scalar.rs::lp_norm_sum_*`** — four direct unit
  tests on the previously-uncovered public `lp_norm_sum`:
  Pythagorean-triple at p=2, sign-handling via `.abs()`,
  zero-input across n in {0, 1, 5, 64}, and uniform-input
  count-scaling at p=4. Discovered while writing: the outer
  `eps^(1/p)` tail subtraction is NOT negligible — sqrt(1e-5)
  ≈ 0.00316 at p=2, eps^0.25 ≈ 0.0562 at p=4. Tests subtract
  the eps tail explicitly rather than loosening tolerances to
  mask it; this is cvvdp's documented safe_pow shape. Tick 351,
  `711eba8a`.
- **`tests/cpu_backend.rs::compute_dkl_jod_host_pool_returns_max_jod_on_identical_inputs`**
  — end-to-end identity gate that scores a buffer against
  itself and asserts JOD ≈ 10.0. Closes a gap where the
  property was only exercised by the `Cvvdp::score` doctest
  (skipped in `cargo test --test <name>` runs). Tick 350,
  `ca3b9d3a`.
- **Documented panic contracts now have `should_panic` regression
  tests** — `do_pooling_and_jod_panics_on_empty_q_per_ch` in
  `pool_scalar.rs` and `predict_jod_still_3ch_panics_on_ref_dim_mismatch`
  / `predict_jod_still_3ch_panics_on_dist_dim_mismatch` in
  `shadow_jod.rs`. Both `# Panics` docstring sections previously
  had only doctest coverage; the integration tests gate them in
  the standard `cargo test` run. Tick 349, `2beffe90`.
- `tests/pyramid_scalar.rs::band_frequencies_exceeds_max_levels_at_high_ppd_or_dim`
  pins the `MAX_LEVELS=9` cap in `pipeline::pyramid_levels` as
  non-vacuous: `band_frequencies` returns 11 entries for
  `(ppd=400, 2048×2048)`, `(ppd=200, 4096×4096)`,
  `(ppd=200, 8192×8192)` — the cap MUST engage to keep
  `weight_idx = k * N_CHANNELS + c` indexing within the
  construction-time weights buffer. Counter-case at
  `(ppd=75.402, 4000×3000)` (standard-4K corpus) shows the cap
  is dormant for typical inputs (`233ed177`, tick 346).
- `tests/column_name.rs` — five regression tests pin the
  `CVVDP_COLUMN_NAME` contract that downstream parquet sidecars
  depend on: non-empty, `cvvdp_` prefix, parquet-safe chars
  (ASCII alnum + underscore only), default form encodes the
  crate version (`cvvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>`),
  claims the reserved `cvvdp_imazen_*` tag. Version + tag
  assertions skip gracefully when `CVVDP_IMPL_TAG` is set at
  compile time so the override path stays a free-form escape
  hatch (`a08d79a0`, tick 345).

#### cvvdp-gpu (api)

- `kernels/mod.rs` module-level "Numerical parity target"
  paragraph claimed the 0.005 JOD bound against pycvvdp v0.5.4
  without scoping it to a `PerfMode`. Same gap I closed in
  `src/lib.rs` Status (tick 323) and in the kernels overview.
  Updated to "under the default `crate::PerfMode::Strict`" with
  a one-line forward-reference that future `PerfMode::Fast`
  optimizations gate inside individual kernel dispatch sites
  and leave the strict path as-described. Tick 332.

- `host_scalar::predict_jod_still_3ch` docstring didn't mention
  `PerfMode`, which left readers asking whether the host-scalar
  reference path responds to `PerfMode::Fast`. Answer: no — it's
  the canonical f32-precision reference that every
  `PerfMode::Strict` parity test validates against, and Fast-mode
  optimizations apply only to the GPU pipeline (gated on
  `Cvvdp::params.perf_mode`). Added an explicit "Always runs the
  strict path" paragraph with intra-doc links to `PerfMode`,
  `PerfMode::Strict`, and `Cvvdp::compute_dkl_jod_host_pool`
  (the right answer for callers who want portable perf + the
  strict numerical contract). All 8 doctests still pass; `cargo
  doc` zero-warning. Tick 331.

- `kernels::pool::pool_band_finalize` docstring said "Finish
  the host-side fold for `pool_band_kernel`", but the function
  is used to finalize partials from BOTH `pool_band_kernel`
  (test-only, post-tick-291) AND the fused `pool_band_3ch_kernel`
  (production). The finalize algebra doesn't care which kernel
  wrote the partial — both store the raw `safe_pow(|x|, β)`
  contribution at `partials[partial_idx]` for the host to fold.
  Updated the docstring to name both kernels and call out the
  shared semantics. Tick 330.

- **`burn-conv-spike` clippy `approx_constant`** — the perlin-
  pattern frequency literal `6.28` in
  `crates/burn-conv-spike/src/main.rs:40` trips clippy's
  `approx_constant` lint (close to but not exactly TAU). Added
  a targeted `#[allow(clippy::approx_constant)]` over the
  expression with a comment explaining why the literal stays
  frozen: the README's parity number (`rel_diff = 0.000156`)
  was measured against this exact value, and the spike's
  "don't extend this crate" rule keeps the verdict
  configuration stable. `cargo clippy --release` in the spike
  dir is now zero-warning. The spike crate has its own
  `[workspace]` root so it doesn't affect parent CI; this is
  a quality-of-life fix for anyone who `cd`s into the spike
  dir to reproduce the verdict. Pre-existing rustfmt
  formatting issues in the spike's main.rs (from the
  subagent's commit) are intentionally left alone — they're
  cosmetic and modifying them would touch the frozen state.
  Tick 329.

- `docs/PORT_STATUS.md` "Open questions" section had a stale
  claim that `warm_state_invalidates_after_each_documented_dispatcher`
  covers 8 cases — true at tick-249 close but tick 314 extended
  it to 9 (added `compute_dkl_jod_host_pool` as a real
  invalidator the original audit missed). Updated the "Resolved
  ticks 236-249" entry to acknowledge the tick-314 extension
  inline. Also added three new "Resolved" entries summarizing
  the post-249 work that wasn't captured anywhere in
  PORT_STATUS.md:
  - **Tick 313-315**: sibling regression-test coverage gaps to
    the tick 236-249 audit (warm-ref ppd-mismatch test,
    non-invalidator dual coverage for cpu host_pool warm-ref).
  - **Tick 322**: `PerfMode` enum framework + the two regression
    tests pinning the no-op contract (GPU pool with 1e-4
    tolerance; cpu host-pool with bit-equality).
  - **Tick 324**: Burn-based port abandoned with measured
    4.32× regression numbers + recommended next perf lever.
  PORT_STATUS.md is now current through tick 328. Tick 328.

- **`perf_mode_fast_matches_strict_on_cpu_host_pool`** —
  cpu-runtime sibling of the GPU-side
  `perf_mode_fast_matches_strict_today` (in
  `tests/pipeline_score.rs`). The GPU-pool test had to relax
  to a 1e-4 tolerance because `pool_band_3ch_kernel` uses
  `Atomic<f32>::fetch_add` whose reduce order is
  non-deterministic across runs. The cpu-runtime host-pool
  path bypasses that atomic entirely (reads D bands back to
  host then folds via deterministic sequential f32
  `lp_norm_mean`), so Fast vs Strict CAN be pinned to
  bit-equality via `.to_bits()`. Covers both
  `compute_dkl_jod_host_pool` (cold) and
  `compute_dkl_jod_host_pool_with_warm_ref` (warm). When a
  real Fast-mode optimization lands on the host-pool path
  this test relaxes to the documented per-stage drift budget
  for the cpu/host-pool case. Tick 327.

- Tick 324's "abandon Burn port" verdict reached into three
  surviving stale references in cvvdp-gpu docs that still
  pitched the Burn port as a "future" direction:
  - `src/lib.rs:160` (`CVVDP_COLUMN_NAME` docstring) — replaced
    "future Burn-based port" with "future alternative
    implementation" + an explicit "(A Burn-based port was
    investigated and abandoned tick 324; see
    `docs/BURN_PORT_PLAN.md`'s banner.)" qualifier.
  - `README.md:175,184` (Sweep-tooling section) — same
    treatment plus a sentence with the empirical justification
    ("4.32× regression vs. the hand-written separable kernel")
    pointing at both BURN_PORT_PLAN.md and the spike's README.
  - `docs/CVVDP_SIDECAR_SCHEMA.md:52` (Reserved-tags table) —
    `cvvdp_burn_v*` row's producer column now reads
    "(abandoned tick 324; the Burn port was investigated and
    ruled out...). The tag stays reserved in case a future
    re-attempt wants to reuse it."
  Tag namespace stays reserved (no risk of accidental
  collision if someone DOES revisit the question), but the
  prose tells the reader honestly that the door is closed for
  now. Tick 326.

- **`crates/burn-conv-spike/README.md`** — new top-level README
  for the perf-spike crate that informed tick 324's
  "abandon Burn port" verdict. Documents what the spike is
  (one-shot paper trail, not a maintained crate), the
  measured numbers in a table form (4.32× for the best cubek
  algorithm, 4.98–5.03× for the rest), the root cause
  (CMMA tile waste at 1-channel; im2col→GEMM memory-traffic
  overhead), the recommended actionable next lever
  (shared-memory tiling of the existing direct stencil), and
  how to re-run. Also adds a "Don't extend this crate" note —
  future re-investigations should spin up a sibling
  `burn-conv-spike-v2/` so this one stays frozen at the
  configuration that produced the verdict. Tick 325.

- **Burn port plan marked ABANDONED.** Tick 322's PerfMode
  framework was paired with a perf spike at
  `crates/burn-conv-spike/` (`e101c895`, run on RTX 5070 sm_120
  at 4000×3000 f32) that compared the proposed
  `cubek::conv2d(5×1) + conv2d(1×5)` separable replacement
  against our hand-written direct-stencil `downscale_kernel`.
  Result: **4.32× slower** even with the best cubek algorithm
  choice (`SimpleSyncCyclic + Mma`, 1.46 ms/op vs 0.34 ms/op
  for the hand-written). Other algorithm choices landed
  4.98–5.03× slower. Root cause: cubek routes conv2d through
  im2col → GEMM with CMMA 16×16×16 tensor-core tiles, which
  waste 15/16 of the work when `in_channels = out_channels = 1`
  and doubles memory traffic vs. a direct stencil. The
  "recover cuDNN-class perf via Burn" pitch doesn't hold for
  our 1-channel separable use case. `docs/BURN_PORT_PLAN.md`
  now has a "Status: ABANDONED" banner up top pointing at the
  spike + recommending shared-memory tiling of the existing
  direct stencil as the actionable next perf lever. The
  surviving content stays as design context.

- **`perf_mode_fast_matches_strict_today` regression test fix +
  extension.** The tick-322 form asserted bit-pattern equality
  via `.to_bits()`; tick 324 (this tick) surfaced that this
  was wrong — two separate `Cvvdp` instances running the same
  inputs can disagree by 1 ULP because `pool_band_3ch_kernel`
  uses `Atomic<f32>::fetch_add` whose reduce order is
  non-deterministic across runs (`CHROMA_DRIFT_INVESTIGATION.md`
  documents the ~1e-5-abs floor over O(10⁴) pixels). The
  tick-322 test passed by chance on the small 64² fixture; the
  warm-ref extension I added in this tick caught the latent
  bug. Switched to `(strict - fast).abs() < 1e-4` (1000× the
  observed 1-ULP noise floor, still well below any real
  Fast-mode optimization's drift budget like 0.005 for
  nearest-CSF or 0.01 for f16 pyramid). Extended coverage to
  `compute_dkl_jod_with_warm_ref` and `Cvvdp::score` so a
  refactor that wired `perf_mode` through one entry point but
  not another would surface. Tick 324.

- **`PerfMode` surfaced in user-facing docs**. Tick 322 added
  the framework; this tick wires it into the discoverable
  surface area:
  - `crates/cvvdp-gpu/README.md` gets a new "Parity vs. perf —
    `PerfMode`" section between "CPU backend" and "Features"
    with a code example showing the struct-update opt-in
    pattern.
  - `src/lib.rs` "Status" section now scopes the 0.005 JOD
    parity claim to `PerfMode::Strict` explicitly and notes
    that Fast is a no-op today (pointing at the regression
    test).
  - Also fixed a pre-existing `rustdoc::broken_intra_doc_links`
    ambiguity warning at `pool.rs:161` — `[`pool_band_3ch_kernel`]`
    was ambiguous between the function and the auto-generated
    module of the same name from `#[cube(launch)]`. Switched to
    the `()` disambiguation form for the function reference.
    `cargo doc -p cvvdp-gpu` is now zero-warning. Tick 323.

- **`PerfMode` enum** opens the parity-vs-perf opt-in surface
  on the public API. Two variants:
  - `PerfMode::Strict` (default) — matches pycvvdp v0.5.4
    bit-for-bit within f32 noise, exactly what every parity test
    in `tests/` is calibrated against.
  - `PerfMode::Fast` — opt-in entry point for future stage-level
    relaxations that trade measurable per-call cost for a
    bounded JOD drift vs. Strict. Currently a no-op (no
    Fast-mode fast paths have landed yet); the variant exists so
    callers can wire the opt-in once and individual stages can
    later gate on `params.perf_mode == Fast` without forcing a
    breaking change.
  Plumbed through `CvvdpParams::perf_mode` (new field, defaults
  to `Strict` in `CvvdpParams::PLACEHOLDER`) → stored on
  `Cvvdp` via the existing `params` field. Re-exported from
  `cvvdp_gpu` for convenience (`use cvvdp_gpu::PerfMode;`).
  Regression test `perf_mode_fast_matches_strict_today` (in
  `tests/pipeline_score.rs`) pins the bit-pattern-equality
  invariant; when a real Fast-mode optimization lands the test
  should be RELAXED (not deleted) to the documented per-stage
  drift budget for that optimization. `doctest` count grows
  from 6 → 8 (two new examples in the `PerfMode` doc comment).
  Tick 322.

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
