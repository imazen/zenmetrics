# `iwssim-gpu` vs standalone `zensim` IW machinery — fusion / port analysis

**Author**: zensim-fusion-review agent
**Date**: 2026-05-22
**Worktree**: `/home/lilith/work/zen/zenmetrics--zensim-fusion-review/`
**Repos surveyed**:
- `iwssim-gpu` and `zensim-gpu` in this worktree
- `~/work/zen/zensim/` (read-only)

## TL;DR

**Recommendation: KEEP THE CRATES SEPARATE. Do NOT fuse, do NOT port
algorithms.** The two IW machineries solve different problems with
different math and different intent. There are **no portable
optimizations** in `zensim::iw_pool` worth lifting into `iwssim-gpu`
because (a) `iw_pool.rs` is dead code in zensim production and (b)
its weight formulas are deliberately different from Wang & Li 2011
IW-SSIM. The only fusion-shaped move that pays off is **leaving them
separate and unifying their CLI / parquet column naming** — already
done via `IWSSIM_COLUMN_NAME` and `cvvdp_imazen_v*` patterns.

## Inventory

### What `iwssim-gpu` actually computes (paper-faithful Wang & Li 2011)

`crates/iwssim-gpu/src/lib.rs:1-47` is explicit about provenance: this
is a port of the MATLAB / PyTorch reference from
`ece.uwaterloo.ca/~z70wang/research/iwssim/`. The pipeline at
`crates/iwssim-gpu/src/pipeline.rs:1-30`:

1. **5-level Laplacian pyramid** with pyrtools `binom5`
   (`sqrt(2)·[1,4,6,4,1]/16`), `reflect1` boundary
   (`crates/iwssim-gpu/src/pipeline.rs:49-98`).
2. **11×11 Gaussian σ=1.5** contrast-structure map per scale
   (`crates/iwssim-gpu/src/kernels/gauss11.rs`).
3. **Information-content weight via the GSM model**:
   - 3×3 box statistics → `g_buf` and `vv_buf`
     (`crates/iwssim-gpu/src/kernels/box3.rs`).
   - `imenlarge2(LP[s+1])` parent band
     (`crates/iwssim-gpu/src/kernels/imenlarge2.rs`).
   - **9×9 or 10×10 covariance accumulation**
     (`crates/iwssim-gpu/src/kernels/cov.rs:1-22`) — per-thread
     partial outer products of the 9- or 10-neighbor patch.
   - **CPU eigendecomposition + matrix inverse** of `C_u`
     (`crates/iwssim-gpu/src/eig.rs`) — one-shot per scale.
   - **Per-pixel quadratic form** `ss = (Y · C_u_inv · Yᵀ) / N`
     followed by the paper's mutual-information weight
     `infow = Σ_j log₂(1 + ((vv + (1+g²)σ²)·ss·λ_j + σ²·vv) / σ⁴)`
     (`crates/iwssim-gpu/src/kernels/infow.rs:1-17` for the
     reference formula and lines 24-345 for the 9- and 10-neighbor
     kernels).
4. Per-scale pool `wmcs_j = Σ(cs_j · w_j) / Σ(w_j)`; final
   `score = Π |wmcs_j|^β_j` with paper-exact
   `β = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`
   (`crates/iwssim-gpu/build.rs:84-91`,
   `crates/iwssim-gpu/src/pipeline.rs:2354-2379`).

Parity gate: `tests/parity_lock.rs:71-76` asserts within 0.5 %
relative of the published Python reference on the canonical
`images/Ref.bmp / Dist.jpg` pair (expected 0.803405).

### What the standalone `zensim` CPU crate has under `iw_pool.rs`

`~/work/zen/zensim/zensim/src/iw_pool.rs:1-100` defines a research
playground that is **explicitly NOT what zensim production uses**:

- `IwWeightKind::LocalVariance` — variance in a (2k+1)² window, scalar
  (line 40-43, `compute_local_variance` at lines 194-225).
- `IwWeightKind::LocalGradL1` / `LocalGradL2` — 3-tap centered gradient
  L1/L2 norm (lines 44-48 + `compute_gradient` at 350-372).
- `IwWeightKind::SteerablePyramidLogGsm` — 2026-05-15 spike approximating
  the paper's directional sensitivity via 4 oriented 3×3 gradient kernels
  (0°, 45°, 90°, 135°) + per-orientation local variance + max across
  orientations (lines 49-69 + `compute_directional_max_variance` at
  227-315).
- `IwWeightConfig.info_log_sigma_e_sq` — optional Wang 2011-style
  `log₂(1 + σ²_p / σ²_e)` post-transform (lines 85-105, applied at
  173-181).
- `WeightedPool::mean / l2 / l4` — weighted spatial pools
  (lines 379-427).
- `IwSsimFeatures` struct — 6-tuple of pooled `(ssim, art, det, mse)`
  per (scale, channel) (lines 433-500).

**Status, line 29-35 of `iw_pool.rs`**:

> "Initial implementation. Designed for offline experimentation —
> correctness over performance. Once the V0_20a sweep shows the
> Wang 2011 paper claim is reproduced for our corpus, the pool
> integration migrates into the SIMD streaming loop..."

**That migration never happened.** The 2026-05-15 perf-hotspot doc
(`~/work/zen/zensim/benchmarks/iw_perf_hotspots_2026-05-15.md:124-130`)
states verbatim:

> "`iw_pool.rs::compute_local_variance`, `compute_iw_weights`,
> `IwSsimFeatures::pool_from_maps`, `WeightedPool::mean/l2/l4` are
> all DEAD CODE in the production path (verified by cargo build
> warnings: `function never used`). The production IW pool uses
> the streaming-loop blurred activity, not the `iw_pool.rs`
> implementation. Skip."

The only external consumer of `iw_pool` is
`zensim-validate/examples/iw_pyramid_ab.rs`, a research A/B harness
for measuring Pearson correlation between scalar variance and the
steerable-pyramid approximation
(verified via `grep -rn "iw_pool\|IwSsimFeatures\|compute_iw_weights"
~/work/zen/zensim/ --include="*.rs"`).

### What `zensim` production CPU actually computes for "IW features"

`~/work/zen/zensim/zensim/src/streaming.rs:1320-1380` shows the live
path used by the `compute_iw_features = true` (372-feature) regime:

```
activity_raw[y,x] = |src[y,x] - h_blur_src[y,x]|
activity[y,x]    = blur_1pass(activity_raw)       # box blur, radius=5
iw_w[y,x]        = 1 + k_iw * activity[y,x]       # k_iw = 4.0
```

That's **`iw_w = 1 + 4·activity`** — a texture-emphasising scalar
weight derived from a local-mean residual, NOT the Wang 2011 GSM
mutual-information weight. The masked-features path is the same
expression with a different formula:
`mask = 1 / (1 + k_mask · activity)`.

### What `zensim-gpu`'s `WithIw` regime computes

`crates/zensim-gpu/src/kernels/masked_iw.rs:22-25,69-70,237-242` is a
**verbatim GPU port of the streaming-loop activity-based IW**:

```
activity = blur(|src - mu1|)
mask_w   = 1 / (1 + K_MASK * activity)   # K_MASK = 4.0
iw_w     = 1 + K_IW * activity            # K_IW   = 4.0
```

`docs/FEATURE_PARITY.md` lines 137-142 confirms: `iw_weight = 1 + 4 *
activity`. This is bit-parallel to the CPU streaming loop, not to the
paper.

## Algorithmic comparison — table

| Aspect | `iwssim-gpu` | `zensim::iw_pool` (dead) | `zensim::streaming` (production CPU) | `zensim-gpu::masked_iw` (production GPU) |
|---|---|---|---|---|
| Paper source | Wang & Li 2011 IW-SSIM | Wang & Li 2011 + cheaper proxies | None (zensim heuristic) | None (zensim heuristic) |
| Weight estimator | GSM σ²_p via 9×9 / 10×10 covariance + eigendecomp | local variance / gradient / 4-orient max-variance | `1 + 4 · blur(\|src − blur(src)\|)` | identical to CPU streaming |
| Per-scale work | Laplacian pyramid (5 levels, binom5) | optional pyramid via `SteerablePyramidLogGsm` (3×3 kernels only) | Gaussian pyramid (zensim's own, 4 scales) | same as CPU streaming |
| Pyramid construction | pyrtools `binom5`, `reflect1` | none (single-scale) | zensim's 4-tap binomial | same |
| Pooling | `Σ(cs · w) / Σ(w)`; product `Π |wmcs|^β` | weighted mean/L2/L4 | weighted mean/L2/L4 per scale (228+72 layout) | identical to CPU |
| Per-scale combine weights | `β = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]` | n/a (research) | learned MLP weights | learned MLP weights |
| Output | scalar IW-SSIM ∈ [0,1] | weights only | 372-feature vector | 372-feature vector |
| Production status | shipping | DEAD CODE in production | shipping | shipping |

**The four implementations are computing different things**:

1. `iwssim-gpu` produces the scalar Wang-2011 IW-SSIM score.
2. `zensim::iw_pool` is a research scaffold for IW *weights* — never
   used for scoring in production.
3. `zensim::streaming` + `zensim-gpu::masked_iw` produce 72 IW
   *features* (per scale × per channel × 6 pooled stats) that feed
   the zensim MLP. The "IW" here is a texture-emphasising heuristic
   tuned to the MLP's training signal, not paper IW-SSIM.

## Answers to the brief's questions

### Q1 — what IW algorithms does standalone zensim implement?

Three variants in `iw_pool.rs`, but **all three are dead code** in the
production zensim CPU path:

- `LocalVariance` (default in the research code): 5×5 spatial variance.
- `LocalGradL1` / `LocalGradL2`: 3-tap centered gradient norms.
- `SteerablePyramidLogGsm` (2026-05-15 spike): 4-orientation gradient
  → per-orientation local variance → max across orientations →
  optional `log₂(1 + σ²/σ²_e)` post-transform. Empirically Pearson
  ≈ 0.84 vs `LocalVariance` at 5×5 (decorrelated enough to suggest
  trainable signal, per
  `~/work/zen/zensim/benchmarks/iw_pyramid_ab_results_2026-05-15.md`).

**Adjustments vs paper**: the steerable variant trades the paper's
5-level Simoncelli pyramid for a 4-orientation 3×3 gradient
approximation (~200 LOC saved). Documented as paper-faithful only at
"zeroth order" (`iw_pool.rs:54-62`). The default `LocalVariance` is
not paper-faithful at all — it's the Wang 2011 §III-B "practical
approximation" caveat the authors themselves describe.

### Q2 — relationship between zensim's IW pool and iwssim-gpu's IW pool?

**They are computing different things.** Not "same thing, different
boundary handling" — different math, different intent:

- `iwssim-gpu` computes a per-scale **score** via the paper's full
  GSM mutual-information weight, 9×9/10×10 patch covariance,
  eigendecomposition, and the paper's `log₂(1 + (…)/σ⁴)` summed over
  eigenvalues. Output is one f64 per pair.
- `zensim::iw_pool` computes per-pixel **weights** via cheap proxies
  (variance, gradient, or 4-orientation directional max) and uses
  them in `Σ(w · v^p) / Σ(w)` to pool **research feature vectors**.
  It never reaches a Wang-2011 score because there's no pyramid
  scaling, no β combination, no `log(…/σ⁴)`.

The closest equivalence: `iw_pool::SteerablePyramidLogGsm + info_log_sigma_e_sq=Some(σ²_e)`
yields `log₂(1 + σ²_p/σ²_e)` weights that are *philosophically* aligned
with paper §II's intent, but the σ²_p comes from gradient variance
not steerable-pyramid GSM, and the resulting weights feed feature
pooling not IW-SSIM scoring.

### Q3 — optimizations or correctness fixes in zensim's IW that iwssim-gpu lacks?

**No.** The `iw_perf_hotspots_2026-05-15.md` improvements explicitly
state `iw_pool.rs` is dead code and out of scope. The perf work
happened in `streaming.rs` (fused 2-mask SIMD kernels, sigma-reuse,
inline IW weight computation) — that's the zensim-gpu `masked_iw_kernel`
porting target, NOT the iwssim-gpu paper-faithful path.

Notable findings recorded by zensim that **could** seem relevant but
aren't actually portable to iwssim-gpu:

- `info_log_sigma_e_sq` saturation transform (zensim
  `iw_pool.rs:85-105`): this is the paper's log saturation, but
  iwssim-gpu already implements the paper's formula directly inside
  `infow_kernel` (the `log₂(1 + (…)/σ⁴)` sum). Not a missing
  optimization — it's the same math expressed at a different layer.
- Steerable-pyramid 4-orientation spike (zensim
  `iw_pool.rs:227-315`): this is a deliberate *cheaper* approximation
  to the paper. iwssim-gpu already does the paper-faithful version
  (full 9×9/10×10 covariance + eigendecomp). Porting the cheaper
  variant **would be a regression** away from parity.
- Strip / streaming optimizations (zensim `streaming.rs`): these
  target zensim's masked+IW heuristic, not paper IW-SSIM. The
  fused-2-mask SIMD kernels assume `mask` and `iw` are
  scalar-weighted by the same activity; iwssim-gpu's `iw` is the
  GSM `infow` map, which is per-eigenvalue summed. Wrong shape.

### Q4 — port or fuse?

**Neither. Keep separate.**

The case against fusion:

1. **The IW concepts coincide in name only.** zensim-gpu's `WithIw`
   regime emits a 72-feature vector for an MLP; iwssim-gpu emits a
   single Wang-2011 score. Bundling them couples release cadence
   with no algorithmic overlap.
2. **No shared kernel surface.** zensim-gpu's `masked_iw_kernel` and
   iwssim-gpu's `cov_accum_*` + `infow_*` kernels share zero
   instructions. Both build pyramids but the pyramids are different
   (zensim's 4-scale Gaussian binomial vs iwssim's 5-level Laplacian
   binom5), boundary conventions differ (zensim uses period-mirror
   periodic boundary in `masked_iw.rs:177-194`; iwssim uses
   pyrtools `reflect1`), and statistics differ (zensim's 11-tap
   box vs iwssim's 11×11 Gaussian σ=1.5).
3. **Parity gates would block each other.** iwssim-gpu's lock test
   is "within 0.5 % of Python piq". zensim-gpu's lock test is
   "within FMA tolerance of CPU zensim". Mixing the crates would
   require maintaining both lock surfaces from one Cargo.toml; any
   refactor breaking one gate fails CI for both. The current
   per-crate `*_COLUMN_NAME` convention already gives the sidecar
   joining what fusion would provide.
4. **Different memory profiles.** iwssim-gpu maintains a
   `MIN_NATIVE_DIM = 176` floor (`lib.rs:81`); zensim-gpu runs to
   8×8. Tile/strip strategies don't compose cleanly.

The case against porting `iw_pool` algorithms into iwssim-gpu:

1. iwssim-gpu already does the *fuller* paper-faithful path. Adding
   `LocalVariance` or `SteerablePyramidLogGsm` would either be (a)
   dead code or (b) a new `IwssimVariant::Approximate` enum
   parameter that nobody asked for and that would dilute the parity
   contract ("which variant should I use to match Python piq?").
2. The steerable-pyramid spike's main lesson — Pearson 0.84 vs scalar
   variance — is for *training* a 372-feature MLP, not for scoring
   IW-SSIM. iwssim-gpu doesn't train.
3. The `info_log_sigma_e_sq` knob is already baked into
   `infow_kernel` via the `sigma_nsq` parameter (set to 0.4 at the
   call site, `pipeline.rs:2918`).

The only thing that resembles fusion already exists: the parquet
sidecar layer. `iwssim-gpu::IWSSIM_COLUMN_NAME = "iwssim_imazen_v<ver>"`
(lib.rs:174-184) and zensim-gpu's `WithIw`-regime feature columns
both land in the same parquet via `zen-metrics-cli score-pairs`.
Consumers join on `(image_path, codec, q, knob_tuple_json)` to get
both columns. No code-level fusion needed.

### Q5 — concrete portable improvements

Given the analysis above, there are **no high-value algorithm ports**
from `zensim::iw_pool` into `iwssim-gpu`. The honest answer to "what
should iwssim-gpu adopt from zensim?" is "nothing — they solve
different problems."

That said, here are five **non-algorithmic** improvements iwssim-gpu
could pick up from the broader zensim ecosystem (these are
infrastructure / quality-of-life, not IW math):

1. **Multi-vendor backend feature-flagging pattern**. zensim-gpu's
   `opaque.rs` shim (referenced in
   `crates/iwssim-gpu/src/opaque.rs:1-3`) is already mirrored. Audit
   that the iwssim-gpu opaque API matches zensim-gpu byte-for-byte
   on the `Backend::{Cuda, Wgpu, Cpu}` enum so callers can pick
   identically. Low effort, high consistency value.
2. **Versioned column-name discipline**. zensim-gpu's
   `cvvdp_gpu::CVVDP_COLUMN_NAME` env-override pattern is already
   ported (`iwssim-gpu/src/lib.rs:174-184`). Verify the production
   `score-pairs` path uses it — quick grep on
   `crates/zen-metrics-cli/`.
3. **Strip-mode VRAM budgeting**. zensim-gpu has the same
   `MemoryMode::{Full, Strip, Auto, Tile}` enum. iwssim-gpu's
   `STRIP_PROCESSING.md` notes the cached-reference fast path
   doesn't yet exist for strip mode (`lib.rs:213-214` —
   `CachedRefNotSupportedInStripMode`). That gap is shared with
   zensim-gpu? Worth confirming; if both crates plan a strip cached
   ref, share the design.
4. **Pinned-host-memory upload path**. The zenmetrics root CLAUDE.md
   ("Pinned-vs-pageable host memory" section) flags 4× upload
   speedup. If iwssim-gpu's grayscale upload (`pipeline.rs:~2960`)
   goes through `create_from_slice` rather than the lilith/cubecl
   `create_from_slice_pinned` fork's pinned path, that's a free win
   independent of any IW machinery decision.
5. **R2 sidecar schema mirroring**. Document iwssim-gpu's parquet
   sidecar schema (column dtype + nullability) the same way
   `cvvdp-gpu/docs/CVVDP_SIDECAR_SCHEMA.md` documents cvvdp's.
   Anchor downstream consumers (zentrain, zenpicker) so a future
   `iwssim_burn_*` column drop-in is mechanical.

None of these involve touching `zensim::iw_pool`. The zensim CPU
crate's IW machinery is, on inspection, not a treasure trove of
hidden optimizations — it's a research scaffold for *zensim's*
MLP-fed feature heuristic, structurally separate from the paper
algorithm `iwssim-gpu` ports.

## Honest risks / open follow-ups

- **I did not run any code.** All findings are from reading source +
  benchmark docs. The claim "`iw_pool` is dead code" rests on the
  perf doc's verbatim statement + an exhaustive grep — both checked
  twice. If there's a non-Rust consumer (Python notebook scoring,
  vast.ai script invoking `WeightedPool` directly via FFI) I missed
  it; grep was Rust-only.
- **The zensim "steerable pyramid" spike could in principle be
  ported to iwssim-gpu** as an alternative `IwssimVariant`, but
  doing so contradicts iwssim-gpu's parity-with-Python-piq lock.
  Not recommended; flagged here only because the brief asked
  whether techniques are portable. Technically yes; practically no.
- **There may be IW-SSIM improvements elsewhere in zensim** I didn't
  surface — e.g., the iwssim-target training docs
  (`v0_22_iw_methodology_2026-05-16.md`) describe using IW-SSIM as
  an MLP supervision target via Python piq, not improving the
  scoring path. That's a training-side decision, not a kernel-side
  port target. Flagged as out-of-scope; happy to chase if asked.
- **Fusion could revisit if iwssim-gpu starts emitting feature
  vectors.** If a future workstream wants per-scale `wmcs_j`
  exposed as features (analogue to zensim's 6-pooled-stats-per-scale
  output), the question reopens. Today neither crate is structured
  that way.

## Action items if you DO want to port (against this rec)

Skeleton only; do not start without explicit go-ahead.

1. Add `IwssimVariant::{Paper, ZensimSteerable}` enum to `IwssimConfig`
   (`lib.rs:120-159`). Default `Paper`.
2. Port `compute_directional_max_variance` from
   `~/work/zen/zensim/zensim/src/iw_pool.rs:251-315` into a new GPU
   kernel `kernels/steerable_w.rs`. Substitute the `infow_kernel`
   call when `variant == ZensimSteerable`.
3. Add a parity test against `zensim_validate::iw_pyramid_ab` numbers.
4. Suffix `IWSSIM_COLUMN_NAME` with `_steerable` for non-default variants.
5. Update `STRIP_PROCESSING.md` + `lib.rs` rustdoc.

Effort: 1-3 days. **Strong recommendation: don't.**

## Migration plan if fusing (also against this rec)

Skeleton only.

1. Move iwssim-gpu's `kernels/*.rs` under
   `zensim-gpu/src/kernels/iwssim/`. Promote shared modules to public.
2. Add `ZensimFeatureRegime::WithPaperIwssim` exposing 5 new floats
   at indices `372..377` (per-scale `wmcs_j`). Total = 377.
3. Unify `score_from_features` so MLP weights either zero-weight the
   new scalars (back-compat) or learn to weight them.
4. Merge parity tests: CPU-parity gate + Python-piq gate in one CI.
5. Pick a release cadence — fusion forces co-release of every
   IW-SSIM numerics change and every zensim feature-pipeline change.
6. Update `zen-metrics-cli` `score-pairs --metric iwssim` to dispatch
   to the fused crate; deprecate standalone iwssim-gpu over one release.

Effort: 2-4 weeks including sidecar schema migration + downstream
consumer audits.

## Files cited (absolute paths)

zensim (read-only):
- `~/work/zen/zensim/zensim/src/iw_pool.rs`
- `~/work/zen/zensim/zensim/src/streaming.rs:1320-1380`
- `~/work/zen/zensim/zensim-validate/examples/iw_pyramid_ab.rs`
- `~/work/zen/zensim/benchmarks/iw_perf_hotspots_2026-05-15.md:124-130`
- `~/work/zen/zensim/benchmarks/iw_pyramid_ab_results_2026-05-15.md`

zenmetrics (this worktree, prefix `crates/`):
- `iwssim-gpu/src/lib.rs:1-184`
- `iwssim-gpu/src/pipeline.rs:1-30,2354-2379,2900-2937`
- `iwssim-gpu/src/kernels/infow.rs:1-345`
- `iwssim-gpu/src/kernels/cov.rs:1-22`
- `iwssim-gpu/build.rs:84-91`
- `iwssim-gpu/tests/parity_lock.rs:60-95`
- `zensim-gpu/src/lib.rs:100-180`
- `zensim-gpu/src/kernels/masked_iw.rs:22-25,69-70,103-407`
- `zensim-gpu/docs/FEATURE_PARITY.md:115-145`
