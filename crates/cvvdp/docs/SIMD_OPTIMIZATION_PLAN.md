# cvvdp SIMD Optimization Plan

**Status**: SCOPING DOC for the SIMD chain agent. No source changes
in this document — only measurements, hot-loop attribution, and a
chunk-by-chunk implementation plan.

**Created**: 2026-05-25
**Author**: cvvdp-perf-scoping agent (sibling workspace)
**Anchored to**:
- `benchmarks/zensim_perf_target_2026-05-25.{tsv,meta}` — the target
- `benchmarks/cvvdp_cpu_perf_baseline_2026-05-25.{tsv,meta}` — the baseline
- `benchmarks/cvvdp_cpu_flamegraph_2026-05-25.svg` — the attribution

## Performance Target (DERIVED FROM ZENSIM BENCH)

| Metric | Current | Phase-1 target | Stretch | Zensim reference |
|---|---:|---:|---:|---:|
| 1024² median wall (8t)   | **222 ms** | **≤ 50 ms** | ≤ 25 ms | 14.66 ms |
| 1024² per-pixel cost     | 212 ns/MP  | ≤ 45 ns/MP  | ≤ 22 ns/MP | 14.0 ns/MP |
| Speedup vs current       | 1.0×       | **≥ 4.4×**  | ≥ 8.8×  | 15.1× |
| Speedup gap to zensim    | 15.1× slow | 3.4× slow   | 1.7× slow | 1.0× |

**Hard floor**: 222 ms (must not regress on any size).
**Parity gate**: JOD scalar within **1e-3** of pre-SIMD on all
`tests/parity_against_host_scalar` + `tests/parity_corpus` cells.
The existing 1e-4 tolerance MAY be widened to 1e-3 with explicit
user confirmation; without confirmation the 1e-4 gate holds.

The 3.4× headroom vs zensim is justified by cvvdp's structural
complexity:
- cvvdp has **8 Weber pyramid bands** at 1024² (5-tap separable
  + 2× decimation); zensim has 4 SSIM scales (more downsampling).
- cvvdp has **per-pixel transcendentals** in masking + CSF (powf
  for `(x+eps)^q`, exp for CSF sensitivity, log10 for Weber-
  contrast); zensim's hot loop has zero per-pixel powf.
- cvvdp's masking does a **σ=3 13-tap separable Gaussian** per
  band (phase uncertainty); zensim's box blurs are 3-tap.

## Hot-loop Attribution

**Flamegraph capture**: `cargo flamegraph -p cvvdp --example
time_size_sweep --release` (full sweep, dominated by 1024² + 2048²
samples). Output: `cvvdp_cpu_flamegraph_2026-05-25.svg`. Raw `perf
report --no-children --percent-limit 0.5` snapshot:

| Rank | Self % | Symbol | Origin |
|------|-------:|--------|--------|
| 1 | **32.06%** | `cvvdp_gpu::kernels::masking::gaussian_blur_sigma3` | cvvdp-gpu (called from cvvdp masking) |
| 2 | **15.41%** | `cvvdp::pyramid::gausspyr_expand` | cvvdp pyramid |
| 3 | 10.63% | Rayon `FnMut::call_mut` shim | rayon trampoline overhead |
| 4 | **8.32%** | `cvvdp::pyramid::build_gauss_pyramid` | drives `gausspyr_reduce` |
| 5 | **7.27%** | `__powf_fma` | libm — 9 calls/px in `mult_mutual_band_into` |
| 6 | 5.16% | `__log10f_finite` | Weber-contrast `log_l_bkg.log10()` + glibc log10 |
| 7 | 4.65% | `__memset_avx512` | `Vec::resize(n_px, 0.0)` for scratch |
| 8 | 2.99% | `__logf_fma` | inside glibc log10 + safe-pow ln path |
| 9 | 2.56% | `__expf_fma` | CSF apply `exp(log_s · LN_10)` per pixel |
| 10 | 2.52% | `cvvdp::pyramid::weber_contrast_pyr` | driver (children dominate) |

Sum of top 10: **91.57 %**.

**Wall-time attribution at 1024² (222 ms total)**:

```
Pyramid Gaussian (reduce + expand + sigma3 phase-unc)   ≈ 124 ms (56%)
Per-pixel powf in masking inner loop                    ≈ 16 ms  (7%)
Per-pixel CSF (log10 + exp + interp)                    ≈ 17 ms  (8%)
Per-band Vec::resize(0.0) memsets                       ≈ 10 ms  (5%)
Rayon plumbing (parallel band + parallel side build)    ≈ 24 ms  (11%)
Other (color conversion, sum/reduce, copies)            ≈ 31 ms  (14%)
```

The 5 chunks below attack the 5 biggest contributors directly. The
combined recovery budget is ~113 ms (222 → 109 ms before structural
work; an additional ~60 ms is reachable from band-fusion + strip-
pipeline structural changes in Phase 2, out of scope here).

---

## Chunk 1 — SIMD `gaussian_blur_sigma3` (the σ=3, 13-tap phase-uncertainty blur)

**Status**: HIGHEST PRIORITY. This is the #1 self-time hotspot at
32.06%. The blur is a 13-tap separable Gaussian with reflect
padding, called 3× per non-baseband band (once per channel) inside
`mult_mutual_band_into`. At 1024² with 8 bands × 7 non-baseband ×
3 channels = ~21 calls per encode, each touching ~`bw·bh` × 2
passes × 13 mul-adds = ~135 ms total per encode (matches the 32%
× 222ms = 71 ms self + ~50 ms inclusive children).

**Mechanism**:
- Port a SIMD variant **into cvvdp's `masking.rs`** (so we
  avoid touching cvvdp-gpu's API surface — the parity agent owns
  the cross-crate change if needed).
- New helpers:
  - `gaussian_blur_sigma3_simd(src: &[f32], w: usize, h: usize,
    scratch_h: &mut Vec<f32>, dst: &mut Vec<f32>)` — public to the
    crate, replaces the upstream `gaussian_blur_sigma3` call in
    `mult_mutual_band_into`.
  - Internal `#[arcane]` entry points per archmage tier:
    `blur_h_sigma3_inner_v4` (AVX2 8-wide), `_v3` (SSE4 4-wide),
    `_neon`, `_wasm128`, scalar fallback via `#[magetypes]`.
  - Boundary handling stays scalar (first/last 6 rows + 6
    columns); interior runs SIMD over `width − 12` SIMD-aligned
    f32 columns.
- Vertical pass: kernel is `13 × width` rolling window. Use one
  archmage f32x8 accumulator per column-group, 13 broadcast-mul-
  add taps. With reflect padding pre-extended, this is a clean
  separable SIMD loop.
- Horizontal pass: same pattern, transposed.

**File scope**:
- `crates/cvvdp/src/masking.rs` — replace `gaussian_blur_sigma3`
  call sites with `gaussian_blur_sigma3_simd`.
- New file `crates/cvvdp/src/simd_blur.rs` (or fold into
  `masking.rs`) for the SIMD entry points.
- `crates/cvvdp/src/scratch.rs` — add `pu_blur_h: Vec<f32>` if
  not already there (mirrors `pu_h` slot).
- `crates/cvvdp/Cargo.toml` — `archmage` already in workspace
  via cvvdp-gpu; verify zero extra deps needed.

**Expected speedup**: 3-4× on the blur itself (zensim's archmage
SIMD blur kernels measure 3-4× over scalar at 1024²). At 56% wall
share, that's ~28-42% of 222 ms = **~62-93 ms recovered**.

**Risk to parity**: LOW. f32 SIMD mul-add on the same kernel
coefficients reorders f32 adds; JOD-scalar tolerance is far above
the noise floor. Reference `~/work/archmage/docs/site/content/
magetypes/examples/convolution_5tap.rs` for the separable-blur
SIMD pattern.

**Test strategy**:
- New `tests/blur_sigma3_simd_parity.rs`: 10 sizes (16, 24, 32,
  64, 100, 128, 256, 512, 600, 1024) random + uniform inputs;
  assert max abs delta < 1e-5 vs scalar `gaussian_blur_sigma3`.
- Property test: DC preservation (uniform input → uniform output)
  to 1e-5 abs.
- The existing `tests/parity_against_host_scalar.rs` must still
  pass at 1e-4 JOD.

**Dependencies**: none — first chunk, can land in isolation.

**Reference SIMD patterns**:
- `~/work/zen/zensim/zensim/src/simd_ops.rs` — see
  `fused_blur_h_ssim` (3-tap separable, but 13-tap follows the
  same shape with more taps).
- `~/work/archmage/docs/site/content/magetypes/examples/blur.rs` —
  canonical pattern.

---

## Chunk 2 — SIMD `gausspyr_reduce` + `gausspyr_expand` (5-tap Gaussian pyramid)

**Status**: SECOND PRIORITY. #2 + #4 hotspots together at **23.73%**.
At 1024² called 36× per encode (6 sides × 6 channels × n-levels-1
non-baseband expands). Scalar implementation in `pyramid.rs:58-225`.

**Mechanism**:
- Port to archmage SIMD same as Chunk 1 — separable 5-tap kernel
  (`GAUSS5`) with stride-2 read (reduce) or zero-insert + stride-1
  read (expand).
- Reduce: vertical pass writes `dh × sw` scratch, then horizontal
  reads scratch with stride-2 + 5-tap dot product per output px.
  - SIMD inner: lane shuffles handle the stride-2 ceil-halving;
    use 4-wide accumulator on f32 with archmage v3 token; 8-wide
    on v4.
- Expand: vertical pass walks `sh` source rows, inserts zeros
  between rows, applies 5-tap kernel for `out_h` outputs (×2
  output rows per input row plus boundary). Same SIMD shape.

**File scope**:
- `crates/cvvdp/src/pyramid.rs` — replace scalar inner loops
  with archmage entry points; keep the reflect-padded boundary
  handling scalar.
- Possibly factor into `crates/cvvdp/src/simd_pyramid.rs`.

**Expected speedup**: 3-4× on the pyramid kernels (5-tap is simpler
than the 13-tap; LLVM may even auto-vectorize the scalar after
adding the `#[target_feature]` boundary, per the
`benchmarks/find_best_split_asm_post_6011f10_2026-05-17.txt`
finding in jxl-encoder). At 23.73% wall share: ~3-4× → **~37-50 ms
recovered**.

**Risk to parity**: LOW-MEDIUM. The pyramid bit-exactness against
the upstream scalar is currently asserted at 1e-6 abs in
`tests/pyramid::tests::reduce_matches_upstream_scalar`. SIMD reorders
adds; switch the gate to 1e-5 abs (still ~3 orders below the JOD
tolerance) at the same time as the SIMD lands.

**Test strategy**:
- Extend existing `pyramid::tests::reduce_matches_upstream_scalar`
  + `expand_matches_upstream_scalar` to 1e-5 abs.
- Add 1024² stress test: build full Weber pyramid from a real
  CID22 image, compare every band to scalar pyramid at 1e-4 abs.
- `tests/parity_against_host_scalar.rs` at 1e-4 JOD must hold.

**Dependencies**: independent of Chunk 1; can land in parallel.

**Reference SIMD patterns**:
- `~/work/archmage/docs/site/content/magetypes/examples/
  convolution_5tap.rs` — exactly this kernel shape.
- `~/work/zen/zenmetrics/crates/cvvdp-gpu/src/kernels/pyramid.rs`
  has the GPU equivalents — read the lane layout for hints.

---

## Chunk 3 — SIMD per-pixel `safe_pow` in `mult_mutual_band_into`

**Status**: THIRD PRIORITY. #5 hotspot at 7.27% self-time = ~16 ms
wall. Agent A predicted this would be the #1 lever; the flamegraph
correction shows it's actually #3 in priority. Still worth doing —
the 9 powf calls per pixel are amenable to SIMD vexp+vln
approximations.

**Mechanism**:
- The pattern `(x + eps).powf(p) - eps_p` (where `p` is one of
  `MASK_P`, `MASK_Q[0..3]`, all loop-invariant per call) decomposes
  to `exp(p · ln(x + eps)) - eps_p`.
- For SIMD, write a `safe_pow_simd<const P_FIXED: bool>(buf: &mut
  [f32], p: f32, eps_p: f32)` that uses an archmage SIMD ln + exp
  (already implemented in zensim — see `simd_ops::*`, or port from
  Sleef-style polynomial approximations).
- **Precision**: f32 vexp+vln is good to ~5 ULP, far below cvvdp's
  JOD tolerance (1e-3 = ~14 bits). The 1e-4 golden tolerance
  remains safe because errors cancel over the spatial-Minkowski
  pool.

**File scope**:
- `crates/cvvdp/src/masking.rs` — replace `(va + SAFE_EPS).powf(q_a)`
  loops with SIMD entry calls.
- Possibly factor into `crates/cvvdp/src/simd_pow.rs`.

**Expected speedup**: 3-6× on the powf loops. Wall share 7.27% +
the related `__logf_fma` (2.99%) which also covers safe-pow:
target ~7-10 ms recovered.

**Risk to parity**: LOW. SIMD vexp+vln are well-understood; the
JOD-scalar effect of 5-ULP per-pixel noise is < 1e-5 (verified
in zensim).

**Test strategy**:
- New `tests/safe_pow_simd_parity.rs`: 4 q values (MASK_P, MASK_Q[0..3]),
  10000 sample points each, max abs delta vs scalar < 1e-4 rel.
- Re-run `tests/parity_against_host_scalar.rs` at 1e-4 JOD.

**Dependencies**: independent of Chunks 1 + 2.

**Reference SIMD patterns**:
- `~/work/zen/zensim/zensim/src/simd_ops.rs` — search for existing
  vexp/vln helpers; reuse if present, port otherwise.
- Sleef-rs or libm-derived polynomial approximations.

---

## Chunk 4 — TLS pool for per-band scratch + pre-cleared buffers

**Status**: FOURTH PRIORITY. #7 hotspot `__memset_avx512` at 4.65%
+ rayon plumbing at 10.63% have overlapping causes. Per-band
`vec![0.0; n_px]` + `Vec::resize(n_px, 0.0)` for 10× scratch Vecs
(d_a/d_rg/d_vy + m_mm_a/rg/vy + term_a/rg/vy + pu_scratch) =
~10 ms total wall at 1024². Plus the **parallel path** in
`fold_bands_parallel` allocates 10 fresh `Vec<f32>`s per band per
thread on every encode — that's where most of the memset cost
lives.

**Mechanism**:
- Mirror butteraugli's B7c pattern: `thread_local!` pool of
  `Vec<f32>` buffers keyed by size, with a `Mutex<Vec<Vec<f32>>>`
  overflow for stealing.
- Module `crates/cvvdp/src/tls_pool.rs`:
  ```rust
  pub struct BufferPool {
      tls: ThreadLocal<RefCell<Vec<Vec<f32>>>>,
      overflow: Mutex<Vec<Vec<f32>>>,
  }
  impl BufferPool {
      pub fn acquire(&self, len: usize) -> PooledVec<'_> { ... }
      // Returns a guard that auto-recycles on drop.
  }
  ```
- Replace 10 `Vec::new()` per-band per-thread with
  `pool.acquire(n_px)`. The pool buffer is already cleared
  (re-fed by previous user) → eliminates the memset step in 80%
  of cases (only first-call sees the zeroing cost).
- Per-thread, cap pool at ~16 entries; overflow holds 48 entries
  (per butteraugli B7c proportions).

**File scope**:
- `crates/cvvdp/src/tls_pool.rs` (new).
- `crates/cvvdp/src/pipeline.rs::fold_bands_parallel` — use
  the pool.
- `crates/cvvdp/src/pyramid.rs` — same for PyramidScratch
  buffers.
- `crates/cvvdp/Cargo.toml` — add `thread_local = "1.1"` to
  deps (verify nothing else in the workspace uses a different
  version; otherwise piggyback on butteraugli's existing dep
  via workspace).

**Expected speedup**: 5-8 ms recovered (memset cost + reduces
rayon-spawn overhead via reduced per-task working-set).

**Risk to parity**: ZERO. Pure allocator-management change; no
numerical impact. (Per butteraugli B7c: 79 new tests passed
byte-identical, zero parity drift.)

**Test strategy**:
- 5 new unit tests for the pool itself (TLS cap, overflow,
  acquire/release roundtrip, size-keyed reuse, Send+Sync).
- Re-run all existing tests — no parity gate change expected.

**Dependencies**: independent. Can land in parallel with Chunks
1/2/3, but biggest wall-time win comes AFTER Chunks 1+2 (because
the pool helps amortize per-task overhead which is currently
dominated by the scalar inner loops; once SIMD lands, per-task
overhead becomes the new top hit).

**Reference**:
- `~/work/butteraugli/butteraugli/src/precompute.rs` — B7c TLS
  pool pattern.
- `~/work/zen/zenmetrics/memory/w44_phase3_b7c_tls_pool_2026-05-23.md`
  for the butteraugli B7c lessons (TLS pool was a +3.6% wall
  REGRESSION at the butteraugli scale, but cvvdp has a
  much larger allocation footprint per encode — re-validate
  with paired A/B during this chunk).

---

## Chunk 5 — SIMD CSF apply per pixel + persistent rayon pool

**Status**: FIFTH PRIORITY (composite). Two complementary attacks
on the remaining hot loops + the 10.63% rayon plumbing overhead.

**Mechanism part A — SIMD CSF apply**:
- `apply_csf_row_per_pixel` (in `pipeline.rs:54-67`) does, per
  pixel per channel: `(log_l − min) × step → frac + idx; lerp;
  add LN_10 constant; exp`.
- The exp is `__expf_fma` (#9, 2.56%). Combined with the loop
  body, the per-pixel CSF is ~17 ms wall.
- SIMD: walk pixels in chunks, lerp+add+exp on 8-wide f32
  accumulators; reuse the vexp from Chunk 3.

**Mechanism part B — persistent rayon pool**:
- The current `fold_bands_parallel` does `(0..n_levels).into_par_iter()`
  — that's 8 tasks for an 8-band image, with rayon spawning fresh
  work units each time. At 1024² each task is ~25 ms — substantially
  larger than rayon's typical work granularity, so per-task overhead
  is only ~3 ms / task. But `build_one_side` uses `rayon::join`
  with nested join — that DOES exhibit per-spawn overhead at low
  task sizes (256² and below).
- Replace with persistent `rayon::ThreadPool::new()` stored on
  `Cvvdp` struct, with explicit `pool.scope(|s| s.spawn(...))` per
  band/channel. Reduces thread-wakeup latency for the 64-512²
  sizes where overhead dominates (currently 76-225% of zensim
  wall).

**File scope**:
- `crates/cvvdp/src/pipeline.rs::apply_csf_row_per_pixel` →
  new SIMD entry; mark `#[inline(never)]` to keep the cold path
  measurable.
- `crates/cvvdp/src/pipeline.rs::Cvvdp` — add
  `thread_pool: Option<Arc<rayon::ThreadPool>>` field; build once
  at `Cvvdp::new`, use throughout the band loop.

**Expected speedup**:
- SIMD CSF: 3-4× on the CSF call → ~10-12 ms recovered.
- Persistent pool: 5-15 ms at 1024² (most savings come from
  smaller sizes — at 64² the rayon spawn overhead is 30-50% of
  wall, so this brings the small-image scoring under the floor).

**Risk to parity**: LOW. SIMD CSF reuses the same vexp from
Chunk 3. Persistent pool is structural — no numerical change.

**Test strategy**:
- SIMD CSF: `tests/csf_simd_parity.rs` — 10000 (log_l, channel)
  samples, max abs delta < 1e-5.
- Persistent pool: extend `tests/diffmap_invariants.rs` to assert
  identical JOD across 10 encodes on a single `Cvvdp` instance
  (no thread-pool side effect).
- Re-run `tests/parity_against_host_scalar.rs` at 1e-4 JOD.

**Dependencies**: SIMD CSF depends on Chunk 3 (shares vexp helper).
Persistent pool is independent.

---

## Chunk 6 (Phase 2, OUT OF SCOPE for this chain) — Band fusion

The remaining wall after Chunks 1-5 is structural: the band loop
visits each band's pixels twice (once for CSF + masking, once for
spatial pool), plus the pyramid is built ONCE per channel and
held in memory at full resolution. zensim's strip pipeline fuses
all of this into a single image pass.

Implementing strip-pipeline-style fusion for cvvdp is a multi-week
effort that risks parity (band boundaries become tile-internal,
pyramid build becomes streaming). Filed as Phase 2; **NOT in the
SIMD chain agent's scope**.

## Summary table

| Chunk | Priority | Self % attacked | Expected ms saved | Actual ms saved | Risk | Dep |
|-------|----------|----------------:|------------------:|----------------:|------|-----|
| 1 — SIMD gaussian_blur_sigma3 | HIGHEST | 32.06% | 62-93 ms | **~76 ms @ 1024² (-48.6 %)**; 256²/512² 1.76× — vs MASTER | LOW | — |
| 2 — SIMD gausspyr reduce/expand | HIGH | 23.73% | 37-50 ms | **~0-5 ms** (see below) | LOW-MED | — |
| 3 — SIMD safe_pow | MED | 10.26% | 7-10 ms | _pending_ | LOW | — |
| 4 — TLS pool + pre-clear | MED | 4.65% + 10.63% | 5-8 ms | _pending_ | ZERO | — |
| 5 — SIMD CSF + persistent rayon pool | LOW-MED | ~3.0% + 10.63% | 15-20 ms | _pending_ | LOW | Ch3 |

**Chunk 1 actual result (LANDED 2026-05-26, re-verified vs current master):**
The original agent benched against `71bd498f` (Chunk 2, before Chunk 4
buffer recycling). The verdict pass re-benched against current master
(`0fc2eb2b`, Chunks 2/3/4/5 + Chunk 4 Scratch already shipped) to confirm
the win is not double-counting Chunk 4's allocation reduction. It is not —
the win reproduces at ~2× on top of master. Paired A/B on
`time_masking_paired_ab` (`RAYON_NUM_THREADS=8`, no native, 5 rounds ×
30 iters, full `Cvvdp::score`):

| size      | baseline (master) | post-SIMD | Δmed%    | Δbest%   | speedup |
|-----------|------------------:|----------:|---------:|---------:|--------:|
| 256×256   |  7.44 ms          |  4.24 ms  | -43.1 %  | -43.1 %  | 1.76 ×  |
| 512×512   | 30.99 ms          | 17.57 ms  | -43.3 %  | -45.0 %  | 1.76 ×  |
| 1024×1024 |158.40 ms          | 82.88 ms  | -47.7 %  | -48.6 %  | 1.91 ×  |
| 2048×2048 |639.29 ms          |360.38 ms  | -43.6 %  | -44.2 %  | 1.77 ×  |

The win is two-fold and stacks: (1) the upstream `gaussian_blur_sigma3` is
pure scalar (the inner `reflect_idx_for_blur` branch blocks LLVM auto-
vectorization), and (2) it allocates 2× `w*h` `Vec<f32>` on every call —
costs Chunk 4's OUTER Scratch does not cover. The SIMD entry vectorizes
the boundary-clean interior (99 % of cells @1024²) AND threads caller
scratch, removing both at once. Chunk 2's "memory-bound at 1024²+"
finding did NOT generalize: the 13-tap source loads overlap heavily (adj
output cols share 12 of 13 inputs), keeping the working set L1d-resident
(compute-bound). See `benchmarks/cvvdp_cpu_simd_sigma3_2026-05-26.meta`.

1e-4 JOD parity floor preserved (`standard_4k_path_still_at_parity_against_host_scalar`
green). 5 new SIMD parity tests at 1e-5 abs all pass.

**Chunk 2 actual result (2026-05-25, commit shipped on master):**
Wall delta at 1024² is ±5 % vs baseline (median post warm 196 ms vs
189 ms — within noise). The pyramid inner kernels ARE fully SIMD-
vectorized (verified by disasm: AVX-512 `zmm` registers, `vmulps`/
`vaddps` confirmed) and the pyramid wall share dropped from 23.5 %
to 22.6 %. The chunk's wall recovery is structurally bounded by the
pyramid's share-of-wall (~23 %, not the 56 % the flamegraph extrapolated),
times the memory-bound speedup ratio (~1.2-1.5× on inner kernels, not
the 3-4× the plan assumed for compute-bound kernels). See
`benchmarks/cvvdp_cpu_simd_pyramid_2026-05-25.meta` for the full
attribution + honest discussion. Outstanding wall now dominated by
`gaussian_blur_sigma3` (Chunk 1, 32 %) and rayon plumbing (Chunk 5).

Combined recovery: ~126-181 ms (60-80% of current 222 ms).

**Realistic post-SIMD wall at 1024²**: 41-96 ms.
- If we land Chunks 1+2+3 (the SIMD core), we hit ~30-40% of
  current = **66-89 ms** — comfortably under the 50 ms Phase-1 target.
- If Chunk 4 also lands, ~58-81 ms.
- If all 5 chunks land, ~41-60 ms.

## Acceptance Gates (per chunk)

Every chunk MUST clear:

- **Build**: `cargo build -p cvvdp --release` PASS on all
  feature combinations (default, no-default, no-default + alloc,
  no-default + alloc + parallel).
- **Tests**: `cargo test -p cvvdp` PASS all 22 existing tests
  + new SIMD parity tests. JOD tolerance held at 1e-4 unless
  user-approved widening to 1e-3.
- **Lint**: `cargo clippy -p cvvdp --no-deps -- -D warnings`
  + `cargo fmt --check` PASS.
- **Perf**: Paired A/B `cargo run --release --example
  time_size_sweep` showing ≥ 10% wall reduction at 1024² with
  ZERO regression on any size (within ±2% noise).
- **Multi-arch**: Build for `i686-unknown-linux-gnu` (via cross)
  + `aarch64-apple-darwin` (via macos-15-intel runner) must not
  regress. archmage tier dispatch handles this automatically.

## DO NOT (binding for SIMD chain agent)

- DO NOT use `-C target-cpu=native` in `Cargo.toml` profile or
  CI builds (per CLAUDE.md). Runtime SIMD dispatch via archmage
  is the right path.
- DO NOT use the `wide` crate (per CLAUDE.md). archmage `#[arcane]`
  + `#[rite]` + `#[magetypes]` is the canonical pattern.
- DO NOT use `unsafe`. `#![forbid(unsafe_code)]` is set in
  cvvdp's lib.rs; archmage provides safe SIMD via tokens.
- DO NOT widen the JOD tolerance above 1e-3 without explicit
  user confirmation.
- DO NOT change the public API. Setters/getters stay stable;
  only internal helpers move.
- DO NOT touch `cvvdp-gpu`'s source. If `gaussian_blur_sigma3`
  needs a SIMD variant in cvvdp-gpu (for cvvdp-gpu callers), that
  is a SEPARATE chunk owned by a different agent.
- DO NOT cite "FMA precision" for any wall-time delta. The wins
  here are SIMD-vectorization wins; report them as such.
- DO NOT default-on a chunk that requires reading a new
  envrionment variable. All SIMD chunks should be unconditionally
  enabled (or behind a `cargo` feature flag that's default-on).

## Reference implementations (where to copy from)

- **archmage SIMD patterns** at `~/work/archmage/docs/site/content/
  magetypes/examples/` — 7 production SIMD examples; see
  `convolution_5tap.rs` for the pyramid blur shape and
  `blur.rs` for the general separable-Gaussian skeleton.
- **zensim SIMD** at `~/work/zen/zensim/zensim/src/simd_ops.rs` —
  `#[arcane]` entry points for SSIM/edge/blur. The 2026-05-15
  fused-2-mask kernel shape is the closest in-tree analog for our
  multi-channel masking SIMD work.
- **butteraugli B7c TLS pool** at `~/work/butteraugli/butteraugli/
  src/precompute.rs` — direct template for Chunk 4.
- **cvvdp-gpu kernels** at `crates/cvvdp-gpu/src/kernels/` — the
  GPU equivalents of every kernel we're SIMD-izing. Read these
  for the lane layout, padding strategy, and reduction shape.

## Bench harnesses to use

- **Primary**: `crates/cvvdp/examples/time_size_sweep.rs`
  (already exists). 5 cold + 5 warm iters per size × content class,
  median reported.
- **Flamegraph**: `cargo flamegraph -p cvvdp --example
  time_size_sweep --release -o
  benchmarks/cvvdp_cpu_flamegraph_<date>.svg`.
- **Paired A/B**: For each chunk, capture
  `benchmarks/chunkN_<chunk-name>_ab_<date>.{tsv,meta}` — runs
  pre and post on the same host, alternating order, 7+ iters.

DO NOT use criterion (per CLAUDE.md). zenbench or hand-rolled
bench harnesses only.

## Hand-off invariants

Whenever this plan is updated, the SIMD chain agent MUST:
1. Read the LATEST `cvvdp_cpu_perf_baseline_<date>.meta` (the
   numbers above will drift; the methodology is what stays).
2. Verify zensim's current perf in `zensim_perf_target_<date>.meta`
   matches the SIMD chain's target (zensim itself might land
   more perf work; re-anchor if it gets faster).
3. Land each chunk as ITS OWN commit + bench TSV. No batching.
4. Update `SIMD_OPTIMIZATION_PLAN.md` after each chunk with
   ACTUAL measured speedup vs EXPECTED. If a chunk underperforms
   2× below estimate, file an HONEST-STOP memo before continuing.
