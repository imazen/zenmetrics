# Strip processing in cvvdp-gpu

> **Status (task #79, 2026-05-26): JOD-preserving Mode E lands in
> phases.** Task #79 reintroduces `MemoryMode::Strip { h_body }` as a
> mode-E variant: the reference-side state lives in dedicated full-
> image buffers, and the dist side runs the standard pipeline against
> them. **JOD output is bit-stable with Full mode** (within the
> Atomic<f32> reduction-order band, ≤ 1e-4 abs JOD), unlike the
> rolled-back capped-pyramid variant. The supported `MemoryMode` is
> now `{ Auto, Full, Strip { h_body } }`. Phase 1+2 ships dedicated
> ref-cache + snapshot/restore plumbing; Phase 3 (per-strip dist
> walker that shrinks the dist working set) is multi-day follow-on
> work — see "Phase status" below.

> **Earlier rollback (task #77, 2026-05-26):** the previous
> `MemoryMode::Strip { capped_levels: Some(k) }` variant was rolled
> back because **capping the pyramid depth changes the JOD value at
> any k < 9**. Methodology trace from the capped-pyramid sweep lives
> under `archived/`. Task #79's Strip variant is structurally
> different: it preserves the full pyramid and only partitions the
> *working set*, not the *algorithm*.

## Why cvvdp doesn't strip

cvvdp's spatial decomposition is a **9-level Weber-contrast pyramid**
(at the standard 4K viewing geometry; `band_frequencies` caps it).
Each non-baseband band feeds an σ=3 phase-uncertainty (PU) blur over
the masking chain, then a 3-channel multi-mutual mask and a
3-stage Minkowski pool.

A true strip walker — one that processes the image in horizontal
slabs and stitches partial results — would have to:

1. Reproduce the PU-blur halo across strip boundaries without drift
   in `log10(L_bkg)`, `T_p`, or the masking-chain intermediates.
2. Accumulate per-band Minkowski partials across strips and finalize
   them after the last strip lands (the pool is non-linear, so naive
   summing of partials gives the wrong answer).
3. Stay bit-stable with the canonical full-image path — every other
   metric crate in this workspace pins parity against its own
   reference, and cvvdp's parity gate is ≤ 0.005 JOD vs pycvvdp
   v0.5.4. A strip walker that drifted outside that band is not
   shippable.

That's a major redesign, not a small refactor.

## Why the capped-pyramid Strip was rolled back

A previous tick shipped `MemoryMode::Strip { capped_levels: Some(k) }`
as a stopgap. It wasn't a true strip walker — it was a Full pipeline
with the pyramid depth clamped to `k`, which shrinks the σ=3 PU-blur
halo at non-baseband bands proportionally to `6 × 2^(k-2)` rows. The
intent was to fit 24 MP square inside a smaller VRAM cap.

**The problem:** capping depth changes the metric output. At any
k < 9 the JOD value drifts from the canonical full-pyramid result.
The sweep data in `archived/cvvdp_capped_levels_2026-05-22.csv`
showed cap=8 sometimes fit the ≤ 0.005 JOD parity gate, but
**cap=8 still produced a different JOD than uncapped Full** — just a
smaller diff. Different JOD = different metric. A code path that
silently changes the metric value is exactly the kind of
"sometimes-correct" surface this workspace has zero tolerance for.

There is no panorama use case in the production corpus to justify
the redesign cost of a true strip walker, so the variant was
removed entirely rather than left as an opt-in landmine.

## What `Auto` does when Full doesn't fit

If `estimate_gpu_memory_bytes(width, height)` exceeds the VRAM cap
(env `ZENMETRICS_VRAM_CAP_BYTES`, else `nvidia-smi` probe, else
8 GB default), `resolve_auto` returns
`Error::TooBigForFull { needed, cap }`. Callers can then:

- Raise `ZENMETRICS_VRAM_CAP_BYTES` if the GPU has headroom.
- Pick a different metric (zensim / iwssim / dssim / ssim2 / butter
  all have working Strip paths).
- Split the image at the application layer — score sub-regions
  independently and report per-region JOD. **The host-side aggregation
  is the caller's responsibility**, and per-region JOD is not
  equivalent to whole-image JOD (the pool is non-linear), so this is
  a fallback only when an approximate per-region score is what the
  caller wants.

## What's in `archived/`

| File | Date | Purpose |
|---|---|---|
| `cvvdp_capped_levels_2026-05-22.csv` | 2026-05-22 | Cap-vs-JOD drift sweep (4 fixtures, cap ∈ [1..9]) — confirms cap < 9 always changes JOD |
| `cvvdp_capped_perf_2026-05-22.csv` | 2026-05-22 | Perf-vs-cap sweep — confirms cap had no meaningful perf benefit |
| `capped_levels_sweep_run.log` | 2026-05-22 | Run log for the now-deleted `capped_levels_sweep` example |

These survive as methodology trace: if a future redesign reconsiders
strip-like memory savings for cvvdp, the prior sweep data documents
which caps were measured and what drift each produced. Do not treat
them as an active configuration surface.

## Mode E (task #79): ref-full + dist-strip cached-ref

Mode E shrinks the *working set* without changing the algorithm. The
reference-side state stays full-image on device; only the dist side
walks the image in strips. Per-band atomic-pool sums are associative
across strips, so the final JOD equals Full-mode JOD within the
documented Atomic<f32> reduction-order noise band (~1e-4 abs JOD on
CUDA).

**JOD preservation invariant.** The dist-side band loop reads the
full ref state at every band — no per-strip approximation of the ref
pyramid is taken. This is the structural difference vs the rolled-
back capped-pyramid variant.

**Scope: cached-ref only.** Mode E only applies to the
`warm_reference` + `compute_dkl_jod_with_warm_ref` (and host-pool /
diffmap variants) code path. One-shot `score()` is still Full-only
because its memory profile *is* the dist working set that Mode E
shrinks (and we can't shrink it without the strip walker, which is
meaningful only across many DIST candidates).

### Phase status

| Phase | What it ships | Status |
|---|---|---|
| 1 — Enum + surface | `MemoryMode::Strip { h_body }`, `new_strip`, umbrella `From` mapping, `STRIP_H_BODY_DEFAULT`, `STRIP_ALIGN` | **Landed (task #79)** |
| 2 — Dedicated ref cache | `RefFullState` struct + `_snapshot_ref_state_to_full` on `warm_reference` + `_restore_ref_state_from_full` on `compute_with_warm_ref`. Cached state survives intervening one-shot dispatches; `has_warm_reference()` correctly reports it. | **Landed (task #79)** |
| 3 — Per-strip dist walker | Shrinks the dist working set to a `(h_body + halo)` strip. Requires per-band σ=3 PU-blur halo bookkeeping at every of the 9 pyramid levels + halo-aware dist-side Weber pyramid build. | **Multi-day follow-on**: not yet wired (see "Phase 3 design notes" below) |
| 4 — Parity tests | `crates/cvvdp-gpu/tests/strip_mode_e_parity.rs` (11 tests, 1e-4 JOD tol) + `cached_ref_cvvdp_strip_n_distortions` in the umbrella. | **Landed (task #79)** |
| 5 — Estimator + docs | `estimate_gpu_memory_bytes_strip(w, h, h_body)` exists (conservative for Phase 2 — returns Full footprint + ref cache delta); doc updates in this file. | **Landed (task #79)**, tightens with Phase 3 |

### Memory profile

| Mode | At 12 MP (4000×3000) | At 16 MP (4096×4096) | At 24 MP (6000×4000) |
|---|---|---|---|
| Full | ~700 MB | ~5.5 GB | OOM on 6 GB |
| Strip (Phase 2 today) | Full + ~50 MB ref cache | Full + ~70 MB ref cache | OOM on 6 GB |
| Strip (Phase 3 target) | ~700 MB ref state + ~50 MB strip dist | ~700 MB ref state + ~50 MB strip dist | ~700 MB ref state + ~50 MB strip dist |

The Phase 2 memory profile does **not** yet shrink the dist working
set — that's Phase 3 work. Today's Mode E delivers:

- **API surface**: the umbrella's `MetricCache::set_reference_unsupported`
  flag no longer fires for cvvdp Strip mode (`has_cached_reference()`
  returns `true` after `set_reference_srgb_u8`).
- **Cached-ref durability**: the cached state survives intervening
  `score()` calls because it lives in dedicated buffers (Phase 2's
  observable behaviour change beyond the API).
- **Foundation for Phase 3**: the `RefFullState` storage is the
  permanent home for the ref-full data the strip walker will read
  from per strip.

### Phase 3 design notes — architectural deep-dive (2026-05-26)

**Status: investigation complete, walker not shipped.** The
2026-05-26 deep-dive confirmed the design notes' "multi-day work"
estimate. Documented below are the load-bearing constraints found
during that investigation so the next agent does not re-trace them.

#### Where the memory actually lives

Per `estimate_gpu_memory_bytes` measured 2026-05-26:

| Size       | Estimate | Breakdown (approx)                                    |
|------------|---------:|-------------------------------------------------------|
| 1024×1024  | 199 MB   | `d_scratch ≈ 60%`, pyramids `30%`, weber `9%`         |
| 2048×2048  | 795 MB   | same proportions                                      |
| 4096×4096  | 3179 MB  | same proportions                                      |
| 4900×4900  | 4549 MB  | same proportions                                      |

The dominant buffer at every realistic image size is `d_scratch`
(6 buffer types × 3 channels × **sum_level_pixels** × f32). At 4 MP
that's ~480 MB. Pyramids (gauss_ref + bands_ref + bands_dis) come
second at ~240 MB. Weber scratch is ~70 MB. To meet the task's
"< 70% of Full" target at 4096² the strip walker has to shrink
roughly all three to per-strip footprints — none of them alone
gives enough headroom.

#### Why a clean strip walker is multi-day

The pipeline's strip-blocking properties are not symmetric across
levels. Investigated 2026-05-26:

1. **9 pyramid levels with per-level halo accumulation.** The dist
   Weber pyramid build at level k uses `gauss[k+1]` to subtract
   upscaled-coarser from fine — every level needs its halo-padded
   sibling at the level above. Halo at base ≈ `6 × (2^max_level - 1)`
   ≈ 1500 rows at 9 levels for the σ=3 PU-blur context.
2. **Per-band σ=3 PU blur halo.** At each non-baseband band, the
   masking chain runs `pu_blur_h_3ch_kernel` +
   `pu_blur_v_3ch_scaled_kernel` — a 13-tap separable Gaussian. The
   strip-side dist band needs `±6` rows halo *at that band's
   resolution*, which is `±6 × 2^k` at base resolution.
3. **Pyramid kernels are not strip-aware today.** `downscale_kernel`,
   `upscale_v_kernel`, `upscale_h_kernel`, `subtract_weber_3ch_kernel`,
   and the masking PU blur kernels (`pu_blur_v_kernel`,
   `pu_blur_v_3ch_scaled_kernel`) all take `(width, height)`
   parameters and apply **reflection at the array edges**. The
   reflection helpers (`reflect_pu_idx`, the inline `2*sh_i -
   (cy + 2) - 1` lines in `downscale_kernel`) read the buffer's
   declared height, not a logical image height.
4. **`gauss_ref` is shared scratch between REF and DIST sides.**
   The dist weber pyramid dispatch reuses `self.gauss_ref` to build
   the dist gauss pyramid (clobbering it for that dispatch), then
   immediately consumes it during the dist Weber subtract chain.
   Mode E Phase 2 (`RefFullState`) restores ref bands before the
   dist dispatch begins; the dist gauss build can run on a strip-
   sized `gauss_ref` only if every other code path that reads from
   `gauss_ref` (e.g. the baseband path consuming `gauss_ref[last]`
   for `inv_l_bkg_mean`) is migrated to read from `ref_full_state`
   exclusively. This cuts across the cached-ref API surface.

#### The deep-band problem

At 4096² with the canonical `STRIP_H_BODY_DEFAULT = 512` rows:

| Level | Strip body height | PU blur halo (rows at that level) | Strip vs halo |
|------:|-------------------:|-----------------------------------:|----|
| 0     | 512  | ±6   | strip >> halo (OK) |
| 1     | 256  | ±6   | strip >> halo (OK) |
| 2     | 128  | ±6   | strip > halo (OK)  |
| 3     | 64   | ±6   | strip > halo (OK)  |
| 4     | 32   | ±6   | strip ≈ halo (marginal) |
| 5     | 16   | ±6   | strip ≈ halo (broken) |
| 6     | 8    | ±6   | strip < halo (broken) |
| 7     | 4    | ±6   | strip < halo (broken) |
| 8     | 2    | ±6   | strip << halo (broken) |

At levels k ≥ 4 the PU blur halo is comparable to or larger than
the strip body height at that level — the strip stops being a
"strip" and effectively needs the whole band. **For deep bands the
walker has to fall back to full-image processing**, which means
a hybrid dispatch (shallow bands per-strip, deep bands full-image)
with separate code paths for the two regimes.

Deep bands are small in absolute terms (level 8 at 4096² is 16×16
= 256 pixels — negligible memory). The full-image fallback there
costs nothing structurally; it just adds dispatch complexity.

#### Approach options (none ship in this push)

(A) **Modify all pyramid + PU blur kernels** to take `body_offset` +
   `body_height` parameters and reflect at logical image edges.
   Touches `downscale_kernel`, `downscale_tiled_kernel`, three
   upscale kernels, `subtract_weber_3ch_kernel`,
   `baseband_divide_3ch_kernel`, both PU blur 1ch + 3ch variants.
   Each kernel needs new parity tests pinning the strip-aware path
   matches the legacy path on full-image inputs. Estimated 2-3 days
   of careful kernel work + verification.

(B) **Allocate strip buffers with enough halo that legacy kernels work.**
   Per strip, the buffer is `(body_h + 2 × max_halo)` rows. Max halo
   at level 0 is `6 × 2^4 = 96` rows at base (the level-4 PU blur
   reflected back through pyramid scaling). Strip buffer = 512 + 192
   = 704 rows ≈ 1.4× body. Allocator wise this means each strip's
   dist buffers are 1.4× the body size; the savings vs Full come
   from N_strips × body_pixels << Full_pixels only at large heights
   (e.g. 4096 / 512 = 8 strips, total ≈ 1.4 × 4096 / 8 ≈ 0.7×
   Full per-strip — modest). Estimated 1-2 days, less kernel
   modification but more host-side state management.

(C) **Hybrid: shallow bands per-strip via (A) or (B); deep bands
   full-image.** The shallow bands consume most memory, the deep
   ones are small. Splits at K_SPLIT where `body_h / 2^K_SPLIT >= 12`
   (twice the PU blur radius). For body=512, K_SPLIT=5.
   Most memory wins, most architecture cost. Estimated 3-4 days.

#### Why I (the 2026-05-26 agent) did not ship Phase 3

Failure-mode clause invoked: the structural complexity of the kernel
boundary handling, combined with the deep-band problem requiring a
hybrid dispatch, places this firmly in the "multi-day refactor"
band the original notes warned about. Pushing a half-correct walker
would either drift JOD outside the 1e-4 tolerance (forbidden by
the JOD-preservation invariant) or pad strips so heavily the memory
reduction disappears.

The user-visible consequence remains as documented in the original
Phase 3 notes: cvvdp at > 16 MP on small-VRAM boxes falls back to
`Error::TooBigForFull` until the walker lands. The Phase 2
foundation (RefFullState + snapshot/restore) is in place and ready
to be the per-strip reader once the walker is built.

### Phase 3 Approach B incremental landing (2026-05-26 follow-on)

**Status: pool stage strip-aware. CSF + masking + dist weber pending.**

The follow-on session shipped the first strip-aware kernel and
walker:

- New kernel `pool_band_3ch_offset_kernel` (in
  `kernels/pool.rs`). Identical math to `pool_band_3ch_kernel` but
  takes a `start_offset` so the host can dispatch on a row-slab of
  a larger d-plane.
- New host walker `_pool_and_finalize_jod_strip` (in
  `pipeline.rs`). Partitions each band's per-pixel pool into row-
  strips sized `strip_h_body >> k` and dispatches the offset kernel
  per slab. Atomic-adds are associative across slabs, so JOD is
  bit-exact against `_pool_and_finalize_jod` within the same
  Atomic<f32> ordering noise band that Full produces on repeated
  calls.
- `compute_dkl_jod_with_warm_ref` and
  `score_from_linear_planes_with_warm_ref` route through the strip
  pool when in Mode E (`strip_config.is_some()`).
- Test-only `Cvvdp::strip_dispatch_counter()` accessor exposed via
  `#[doc(hidden)]`. Tests assert N >= 2 strip iterations at 1024²
  with `h_body=512`, proving the walker actually partitions.
- 5 new parity tests in `tests/strip_mode_e_phase3.rs`:
  - `phase3_pool_strip_matches_full_at_64x64` (degenerate strip,
    JOD bit-exact)
  - `phase3_pool_strip_matches_full_at_1024x1024` (L0 partitions
    into 2 strips, JOD bit-exact)
  - `phase3_strip_walker_dispatches_n_strips_at_1024` (counter >= 2)
  - `phase3_pool_strip_repeats_deterministically` (no walker-side
    non-determinism)
  - `phase3_full_mode_counter_stays_zero` (counter is gated on
    strip mode, doesn't leak into Full callers)
- 11 existing `strip_mode_e_parity.rs` tests still pass — the pool
  walker is a drop-in for the existing dispatchers.

**Memory impact**: zero so far. Only the pool stage iterates in
strips; d_scratch, bands_ref, bands_dis, weber_scratch all remain
full-image-sized. The pool stage is a tiny fraction of the working
set. This landing proves the walker is correct end-to-end (atomic
associativity + per-strip iteration + counter visibility); the
memory wins are gated on the kernel-port work below.

#### Next chunks (each ~1 day of focused kernel work + parity test)

The chunks below port the rest of the cvvdp pipeline to strip-aware
kernels. Each is small enough to land alone, ship a strip-aware
parity test, and incrementally reduce the d_scratch /
bands_dis / weber_scratch peak footprint:

1. **Strip-aware pu_blur kernels** — add `(body_offset_y, logical_h)`
   to `pu_blur_v_3ch_scaled_kernel` and `pu_blur_h_3ch_kernel`. The
   horizontal kernel is trivial (no vertical halo); the vertical
   needs the same reflection-at-logical-edge fix as the pyramid
   downscale plan.
2. **Strip-aware CSF apply** — `csf_apply_3ch_kernel` /
   `csf_apply_6ch_kernel` are per-pixel; trivial offset
   parameterisation.
3. **Strip-aware masking chain** — `mult_mutual_3ch_*` /
   `subtract_kernel` / etc. are per-pixel; trivial.
4. **Per-strip d_scratch slab allocator** — sized
   `(strip_h + 2 × halo_at_base) × width` per band, where
   `halo_at_base = 6 × 2^min(k, K_HYBRID)` and K_HYBRID is the cut
   level below which we still run full-image (see the deep-band
   problem table). For `h_body = 512`, `K_HYBRID = 4` lets bands
   0-3 run strip-aware and bands 4..n_levels run full-image.
5. **Strip-aware downscale / upscale_v / upscale_h /
   subtract_weber_3ch** — the four pyramid kernels. Each needs the
   `(body_offset_y, logical_h)` reflection treatment. The downscale
   kernel has the pycvvdp parity-bug compat delta at the right
   column — keep using `logical_sw % 2` / `logical_sh % 2` for the
   delta condition.
6. **_dispatch_dist_weber_pyramid_only_strip** — walker that calls
   the new kernels with per-strip slab views. The
   `_run_d_bands_band_loop` already in Phase 3 is what consumes the
   per-strip bands_dis; the loop body needs the same
   `(body_offset, logical_h)` plumbing for the masking chain.

When all six chunks ship, the per-strip d_scratch + bands_dis
footprint shrinks to `n_strips × per_strip_pixels << full_pixels`,
hitting the original `<= 70% of Full` Phase 3 target. The
JOD-preservation contract stays tight (each chunk's parity test
pins bit-exact match on full-image inputs vs strip-aware path).

#### Specific implementation hints for the next agent

If approach (A) (kernel modification) is chosen:

- The simplest signature change is to add `dst_offset_y` +
  `logical_h` to each pyramid/PU kernel; the kernel computes
  `global_y = dst_y + dst_offset_y` and reflects against
  `logical_h` instead of `h`. The dispatched `n_px` stays the
  strip-body count; the buffer height stays the strip-buffer
  height.
- Add a `tests/pyramid_kernel_strip_aware.rs` that constructs a
  full-image kernel result, slices a strip out, and runs the
  strip-aware kernel on a strip buffer asserting bit-exact match
  at the body rows.
- The downscale kernel has a pycvvdp bug-compat delta at the
  right column (see `downscale_kernel` lines 851-886). That delta
  uses `sw % 2`/`sh % 2` parity — both of which refer to the
  LOGICAL image dims, not the strip dims. The strip-aware path
  must pass `logical_sw` + `logical_sh` so the parity delta fires
  identically. Easy to miss; pin in a strip-parity test.

If approach (B) (halo extension) is chosen:

- The dssim-gpu pattern (see `dssim-gpu/src/pipeline.rs` —
  `compute_with_reference`'s strip walker) is the right template.
  The difference is cvvdp's per-level halo is `6 × 2^k` not
  `2 × 2^k` like dssim — so the per-strip buffer must size for
  the **worst** halo (at level 0, since 2^k grows faster than
  the strip body shrinks).
- `copy_rows_kernel` analogue: cvvdp doesn't have one yet. Will
  need a new kernel.

#### What's wired and ready

- `RefFullState` (ref bands, log_l_bkg, baseband gauss) at full
  resolution: ready for per-strip slab reads.
- `StripConfig { h_body }` known at construction time.
- `_warm_ref_baseband_log_l_bkg_for_dispatch` is the integration
  point — currently `restore_ref_state_from_full` runs a full
  restore. Phase 3 replaces that with a per-strip slab restore
  inside the strip walker.
- `_run_d_bands_band_loop` is the per-band masking + pool dispatch.
  Phase 3 strip walker calls this once per strip with strip-sized
  inputs.
- `_pool_and_finalize_jod` consumes `partials_h` which atomic-
  accumulates across band+channel slots. **Atomic adds are
  associative** — the same `partials_h` accumulates correctly
  across both bands AND strips, so the pool finalizer needs zero
  changes for Phase 3.

`tests/strip_mode_e_parity.rs` pins the JOD value contract;
Phase 3 must keep all 11 tests passing AND add a
`strip_walker_dispatches_n_strips` test that asserts N > 1 strip
iterations occur at sizes large enough to require partitioning.

## What `Auto` does today

For `(width, height)` whose Full footprint exceeds the VRAM cap,
`resolve_auto` now picks `Strip { h_body: STRIP_H_BODY_DEFAULT }`
instead of surfacing `TooBigForFull`. Phase 2 (today) gives the
caller a working cached-ref path with full ref-state durability;
Phase 3 will shrink the dist working set so the same picker decision
actually fits a smaller VRAM cap.

If even the Phase 2 estimate (Full + ref cache) exceeds the cap,
`resolve_auto` still surfaces `Error::TooBigForFull { needed, cap }`.

`Auto` does NOT pick `StripPair` (Mode B); that variant is opt-in
only — callers explicitly choose between `Strip` (cached-ref for
batch workloads) and `StripPair` (one-shot for CLI use). See the
Mode B section below.

## Mode B (Round 2): one-shot pair stripwise (StripPair)

Mode B (`MemoryMode::StripPair { h_body }`) is the one-shot pair
strip-walker variant: **both ref and dist** walk in strips together
with no full-ref cache on device. Designed for CLI / one-shot
scoring on large images where the user only needs a single JOD
value and doesn't reuse the reference across many distorted
candidates.

### When to pick Mode B vs Mode E

| | Mode E (`Strip`) | Mode B (`StripPair`) |
|---|---|---|
| Ref cache on device | Yes (RefFullState, full-image) | No |
| Per-DIST work | Dist pyramid + masking only | Full ref+dist pyramid + masking |
| Peak memory (target) | RefFullState + 1 strip dist | 2 strips (ref + dist) |
| Best for | Batch workflows (sweep workers, many DISTs per REF) | One-shot scoring (CLI, single-pair) |
| Construction | `Cvvdp::new_strip` or `MemoryMode::Strip` | `Cvvdp::new_strip_pair` or `MemoryMode::StripPair` |

### Round 2 status (2026-05-26)

What shipped in Round 2 (commits c2cbcdfd, 8ac4597c, 1aa46182):

- **API surface**: `MemoryMode::StripPair { h_body }`,
  `Cvvdp::new_strip_pair` / `new_strip_pair_with_geometry`,
  `is_strip_pair_mode()` predicate, `estimate_gpu_memory_bytes_strip_pair`.
- **Internal**: `StripMode` enum (`CachedRef` for Mode E, `Pair` for
  Mode B) carried in `StripConfig`. The `new_with_memory_mode`
  dispatcher routes `MemoryMode::StripPair` through the new
  constructor.
- **Estimator**: returns Full footprint (conservative bound — Mode
  B does NOT pay the `RefFullState` delta that Mode E does, so this
  estimate is strictly tighter than Mode E's at equal `h_body`).
- **Tests**: 6 Mode B parity tests pinning the JOD contract; all
  11 Mode E tests still pass.
- **JOD invariant**: Mode B `score()` returns the same JOD as Full
  at 64×64 within `1e-4 abs` (because today's walker routes through
  the existing Full pipeline — see "what's NOT shipped" below).

What did NOT ship in Round 2 (multi-day follow-on work):

- **Per-strip dist+ref buffer allocation**. Currently
  `new_strip_pair` allocates at full-image dimensions — same as
  Full — so there is no memory reduction yet. The dist scratch
  (`bands_dis`, `d_scratch`, `weber_scratch`) needs to be sized
  for `(h_body + 2 × halo)` strips, not full-image.
- **Strip walker** that dispatches the Round 1 strip-aware kernels
  per strip iteration. Mode B's walker shape mirrors Mode E's
  (`_dispatch_strip_body(...)`) but uploads fresh REF data per
  strip instead of reading from RefFullState.
- **Hybrid K_SPLIT dispatch** for the deep-band problem
  (shallow=strip, deep=full-image). K_SPLIT=5 for `h_body=512`.
- **4096² / 24 MP nvidia-smi delta measurements**.

These items are gated on the same multi-day kernel + walker work
documented in the Mode E Phase 3 design notes above. The Round 1
strip-aware kernels (downscale, upscale_v, upscale_h,
subtract_weber_3ch, pu_blur_h_3ch, pu_blur_v_3ch_scaled, pool) are
the shared building blocks both Mode E and Mode B walkers will
dispatch — Mode B reuses the entire Round 1 kernel surface.

### Why Mode B sits next to Mode E

The two walkers share most of the per-strip dispatch body. A
future `_dispatch_strip_body(strip_idx, ...)` helper will be
called by both Mode E (which sources REF from RefFullState row
slabs) and Mode B (which uploads ref strip bytes per strip
iteration alongside the dist strip bytes). The variants differ
only in:

- **Pre-strip setup**: Mode E restores REF state from
  RefFullState into the strip-sized bands_ref scratch; Mode B
  runs `srgb_to_dkl` + gauss pyramid + weber pyramid on the
  ref-strip bytes.
- **Construction**: Mode E allocates RefFullState (full-image
  ref bands + log_l_bkg + baseband gauss); Mode B does not.
- **Memory profile**: Mode B's per-strip working set is double
  Mode E's (ref+dist both strip-sized), but Mode B saves the
  RefFullState delta (~340 MB at 4096²).

For images small enough to fit Full mode, both Mode E and Mode B
add overhead (per-strip dispatch is slower than a single full-image
dispatch on backends with high launch cost). For large images that
don't fit Full, both variants restore feasibility — pick Mode E
for batch, Mode B for one-shot.

## Phase 1: structural strip-major walker — investigation 2026-05-26

> **Status: investigation complete, structural blocker identified.
> Phase 1 deliverable is the analysis below + the precise Phase 2
> recipe to unblock. No new dispatch path lands in this push because
> a partial walker either (a) ships wrong JOD or (b) reduces to the
> already-level-major dispatch the codebase has today.**

### What Phase 1 was supposed to deliver

A new dispatch path for `MemoryMode::StripPair` (Mode B) where the
**outer loop is strip-major**: for each strip s in `0..n_strips`,
run ref weber pyramid + dist weber pyramid + band loop + baseband
bypass + per-band JOD-partial accumulate, then discard per-strip
work and advance to strip s+1. Full-image buffers stay in place;
the walker reads/writes them as `Handle::offset_start`-style strip
windows. The goal is a dispatch-order change only — Phase 2 then
replaces the full-image buffers with strip-sized buffers (where
"strip-sized" includes halo) to deliver the actual memory win.

### Why the simple strip-major inversion does not work

Today's Mode B dispatch is **level-major outer, strip-major inner**:

```text
ref weber pyramid (level k = 0..n_levels, strip-walks per level)
dist gauss pyramid (level k = 0..n_levels, strip-walks per level)
dist weber baseband (one-shot)
band loop (for k in 0..n_levels):
  if not baseband and use_blur:
    fused dist-weber+csf strip walker for level k (strip-walks)
    masking strip walker for level k (strip-walks; pool inline)
  else:
    full csf + full masking + inline pool
```

Inverting this to a strip-major outer loop requires every kernel
that reads halo rows to find correct data at those halo rows. Two
kernel chains have non-trivial cross-strip reads:

1. **Pyramid build (gauss + weber).** The downscale kernel reads
   `±2` source rows around `2 · dst_y_logical`; the separable
   upscale reads `±1` source rows around `dst_y / 2`. At each
   level k the halo at level 0 doubles, so the deepest level's
   halo at base resolution is `±2^k` rows. For a 9-level pyramid
   that is `±256` rows at level 0 — much larger than typical
   strip bodies.

   In a strip-major walk, when strip `s` runs level-1 reduce, it
   needs level-0 rows in a `±2`-row neighbourhood around
   `2 · body_offset_y_at_1`. The neighbourhood extends BELOW into
   strip `s − 1`'s body (already written — OK) and ABOVE into
   strip `s + 1`'s body region (not written yet — STALE DATA).

   The full-image buffers don't hold zeroes either. `gauss_ref` is
   shared between the REF pass and the DIST pass; when the DIST
   pyramid runs strip-major and overwrites strip `s`'s level-0
   rows, the rows above (= strip `s + 1`'s region) still contain
   **the REF pass's level-0 data from minutes ago**. A strip-major
   DIST gauss reduce that reflects against `logical_h = full_h`
   silently mixes its own strip's DIST output with the previous
   call's REF residue at the halo, producing wrong DIST data —
   and wrong JOD.

2. **Masking V-blur.** Each non-baseband band runs
   `pu_blur_v_3ch_scaled_strip_aware_kernel`, a 13-tap separable
   Gaussian with `±6` rows of vertical reflection. At level k the
   halo is `±6` rows at the band's resolution, i.e., `±6 · 2^k`
   rows at base. For `h_body = 512` and `n_levels = 9`, level-4
   halo at base is `96` rows, level-5 is `192`, etc.

   The existing `_run_band_masking_strip_walker` already handles
   this WITHIN a level: the halo-padded window over `t_p_*[k]`
   reads `[body − 6, body + body_h + 6]` rows of the band's
   `t_p_*` buffer. That works today because by the time the
   masking strip walker runs at level k, the **fused dist-weber +
   csf strip walker has fully populated `t_p_*[k]` for every
   strip of level k** (the inner strip-walks). The halo rows are
   real data.

   Inverting to strip-major outer breaks this. When strip s runs
   the masking chain for level k, only strip s's body rows of
   `t_p_*[k]` are populated; the halo rows above (= strip s+1's
   body region) contain stale data (either zeroes from
   construction or residue from a prior call). The V-blur silently
   mixes valid + stale data and produces wrong masking output.

Both blockers boil down to the same root cause: **strip-major
outer dispatch reads halo rows that future strips will eventually
populate**. There is no read ordering that satisfies the halo
dependency without either (a) computing the halo redundantly per
strip from the original input bytes, or (b) carrying halo data
through buffers explicitly sized to hold body + halo.

### Why "just compute the halo redundantly" doesn't fit Phase 1

Approach (a) — strip `s` recomputes its own halo at every level
from the original sRGB input bytes — is mathematically correct.
DKL is per-pixel (trivially halo-extendable). Each pyramid level
strip recomputes for `body + ±halo_at_level_0` rows. The kernel
output for the halo rows is bit-identical to what a Full-mode
dispatch would have written there, because the kernel is
deterministic and the input data (sRGB bytes) is fully available
at every strip.

**But Phase 1's constraint is "no buffer changes."** Approach (a)
requires every per-level buffer (`gauss_ref[k].planes[c]`,
`bands_dis_strip[k]`, `t_p_*[k]`, `m_raw/m_mid/m_blur`) to hold
the strip's body + halo at that level. Today's `bands_dis_strip`
is already strip-only-body-sized (the prior Phase-1d shrink). To
make strip-major work, `bands_dis_strip` must grow to
`(body_h + 2 · halo_at_level) · fine_w · 4 · 3 channels`. That is
a buffer size change. Phase 1 explicitly forbids it.

The other option — leaving today's full-image buffers in place and
having strip `s` write its halo redundantly into the full image —
would technically work, but it also DEFEATS Phase 2's memory goal.
If full-image buffers are kept everywhere, no memory reduction
ever happens. Phase 1 was supposed to be a dispatch-order change
that prepares the way for Phase 2 to substitute strip-sized
buffers for the full-image buffers — but the dispatch-order
change itself requires the buffer substitution to be correct.

**Phase 1 and Phase 2 are structurally entangled.** The dispatch
order cannot be inverted without per-strip halo-padded buffers;
the per-strip halo-padded buffers serve no purpose without the
inverted dispatch order. Either both ship together (a multi-day
change) or neither ships.

### What the current dispatch already does

The current code is **level-major outer + strip-major inner**,
with the strip-major-inner loops handling all the per-level halo
correctly. Specifically:

- `_reduce_gauss_pyramid_strip_walker` strip-walks each level's
  reduce. Cross-strip data flow is captured by level-major outer
  (level k-1 is fully written before any strip of level k runs).
- `_finalize_weber_pyramid_strip_walker` strip-walks each level's
  weber. Same level-major outer invariant.
- `_dispatch_dist_weber_csf_strip_walker_for_level` strip-walks
  the fused dist Weber + csf for one level k. Reads full-image
  ref bands + log_l_bkg (written by ref pyramid before band loop)
  and writes per-strip `bands_dis_strip[k]` + body rows of full-
  image `t_p_*[k]`.
- `_run_band_masking_strip_walker` strip-walks the masking chain
  for one level k. Reads halo-padded window of `t_p_*[k]` (fully
  populated by the prior dist+csf walker), writes body rows of
  `d_scratch[k].d_strip`, and pools inline.
- `_pool_and_finalize_jod_strip` finalizes the baseband pool.

This is already as strip-aware as the dispatch can be **while
buffers stay at their current sizes**. The pool stage runs strip-
major within each level (atomic associativity makes the strip-vs-
level ordering irrelevant for partials_h); the band-internal
masking chain is halo-aware. Re-ordering the outer loop to be
strip-major would NOT change anything observable except introduce
the halo blocker described above.

### Phase 2 recipe (unblocks Phase 1's dispatch-order change)

To actually invert the outer loop to strip-major, the buffer plan
must change. The precise recipe:

1. **Grow `bands_dis_strip[k]`** from `(strip_h_at_k · fine_w)` to
   `((strip_h_at_k + 2 · halo_at_k) · fine_w)` for shallow levels
   (`k < K_SPLIT`). At deep levels (k ≥ K_SPLIT) the band is small
   enough to keep full-image-shape (or strip = full because
   `strip_h_at_k` collapses to 1).

2. **Shrink `bands_ref[k]`** the same way (or keep full-image
   for ref, since Mode B doesn't cache ref across calls — the
   alloc cost lives once per `score()`). The simpler move: shrink
   bands_ref to the same `body+halo` shape as bands_dis_strip and
   have ref weber pyramid run strip-major-outer alongside dist.

3. **Shrink `t_p_*[k]`** to body+halo. Today they are full-image
   transients (allocated inside `DBandsTransient::new`); making
   them per-strip body+halo halves their footprint at any given
   strip, and only ONE strip's transients live on device at a
   time (the next strip's transients overwrite them).

4. **Shrink `m_raw / m_mid / m_blur`** similarly. They are already
   per-band transients (`DBandsTransient`); making them per-strip
   body+halo is mechanical.

5. **Shrink `weber_scratch[k].log_l_bkg` + `log_l_bkg_dis`** to
   body+halo. Today both are full-image. Strip-major writes them
   per strip; the band-loop CSF reads them in the same strip
   iteration.

6. **Shrink `weber_scratch[k].l_bkg_fine`** the same way. Today
   full-image.

7. **Shrink `gauss_ref[k]`** to body+halo for shallow levels. Deep
   levels keep their full-image footprint (they're small anyway —
   level 8 at 4096² is 16×16 pixels).

8. **Shrink `d_scratch[k]` non-`.d` fields** (`csf_pyr`, masks,
   `d_p_ref`, `d_p_dis`) to body+halo. Today these are part of
   the per-band scratch.

9. **Compute `halo_at_k`** as `±2^k · 2` (pyramid-level reads) +
   `±6 · 2^k` (PU-blur halo at level k, converted to level 0 by
   the per-level scale). For `k = 0`, halo at level 0 is `±14`
   rows (pyramid: 2, PU: 6, times 2 for the level-1 reduce's
   halo back-projected through doubling). Halo grows with k until
   the K_SPLIT cutoff where the body+halo strip is wider than the
   full-image; at that point the level falls back to full-image
   processing.

10. **Define K_SPLIT** per the existing `mode_b_k_split` helper
    (already implemented). At `h_body = 512`, `n_levels = 9`,
    K_SPLIT = 6 — levels 0..5 run strip-mode, levels 6..8 run
    full-image-mode (cheap: level 8 at 4096² is 256 pixels total
    across all three channels). At `h_body = 256` K_SPLIT = 5.

11. **Re-dispatch order**: after all buffer plumbing, the outer
    loop becomes
    ```text
    for s in 0..n_strips_at_level_0:
      DKL on strip body + halo from src bytes
      gauss reduce for levels 0..K_SPLIT (per-strip, body+halo)
      weber finalize for levels 0..K_SPLIT (ref then dist; per-strip body)
      band loop for levels 0..K_SPLIT (csf + masking + pool inline)
    for k in K_SPLIT..n_levels:
      level-major full-image dispatch (existing path)
    baseband bypass + pool (existing path)
    finalize JOD from partials_h
    ```

This is the full Phase 2 implementation. Estimated 3-5 days of
careful kernel + walker work, with parity tests at every step.

### What Phase 1 ships in this push

Given the structural blocker above, Phase 1 ships **the precise
analysis and recipe**. A test scaffold (the existing
`mode_b_walker_parity.rs` tests) already pins the JOD-bit-identity
gate that Phase 2 must hold against. The Phase 2 work plan above
is the concrete next move; it is NOT in this push because the
correctness gate (JOD = Full ± 1e-4) requires the buffer-shape +
walker-order changes to land together.

### Lessons for the next agent

- **Read this section first.** A strip-major outer loop with
  today's buffers is a one-line refactor that produces a four-
  hour debug session ending in wrong JOD. The halo blocker is
  not visible until you run a real image through it.
- **The existing strip walkers already do everything that's safe
  to do with full-image buffers.** Look at `_reduce_gauss_pyramid_strip_walker`,
  `_finalize_weber_pyramid_strip_walker`,
  `_dispatch_dist_weber_csf_strip_walker_for_level`, and
  `_run_band_masking_strip_walker` — they are the level-major-outer
  + strip-major-inner pattern that hides the halo dependency.
- **K_SPLIT is the key.** The `mode_b_k_split` helper already
  computes it correctly for `(h_body, n_levels)` pairs. Phase 2
  uses it directly.
- **bands_dis_strip is the canary.** When you see code that
  treats `bands_dis_strip[k]` as body-only (no halo), and you
  want to invert the dispatch order, that buffer's shape is the
  first thing to change.

## Phase 2 — investigation 2026-05-27 (PRE-IMPLEMENTATION ANALYSIS)

> **Status: read-only analysis. No walker / kernel / buffer code
> was touched in this push. This section is a refinement of the
> Phase 2 recipe at lines 683-787 — written after a full read of
> the existing strip walkers and the per-level kernel reflection
> semantics. The analysis identifies three concerns the brief's
> recipe under-specifies (halo back-projection, REF/DIST clobber
> ordering, walker rewrites), decomposes the work into ten
> commit-sized increments each gated by JOD parity, and recommends
> a low-risk pre-work pass that the next session can ship without
> taking on the full canary surgery.
>
> The numbers in §1 are first-pass calculations from the kernel
> reflection semantics; the next session MUST verify them by
> reading each `*_strip_kernel` / `*_strip_aware_kernel` against
> the actual reflection bounds before sizing buffers off them.**

### 1. Halo math — VERIFIED 2026-05-27 (P2.0)

The pre-implementation analysis above hypothesized different
formulas (`8·2^k` from the recipe, `max(body+halo, 2·next+2)` from
§1's earlier pass arriving at 830 rows). Reading
`downscale_strip_kernel` (`crates/cvvdp-gpu/src/kernels/pyramid.rs:928`)
directly gives the verified semantics:

- The downscale kernel reads source rows at logical positions
  `{2·dy_logical − 2, −1, 0, +1, +2}` per output row, with
  reflection against logical bounds at image edges.
- A strip needs halo on BOTH sides of the body (PU blur ±6 reads
  both directions; reduce stencil reads ±2 above AND below).
- Producing `H_dst` valid level-(k+1) rows from level-k source
  needs `2·H_dst + 4` valid level-k source rows.
- The level-k buffer must satisfy BOTH its own band-loop body+halo
  (`body_k + 2·halo_band` with `halo_band = 8`) AND the next-level
  reduce's source-row appetite (`2·R_{k+1} + 4`).

Recursion (deepest-shallow → shallowest):
```
R_{K−1} = body_{K−1} + 16
R_k     = max(body_k + 16, 2·R_{k+1} + 4)   for K−1 > k ≥ 0
```

For `h_body = 512, K_SPLIT = 6`:

| k | body_k | R_k                          |
|---|--------|------------------------------|
| 5 | 16     | 32                           |
| 4 | 32     | max(48, 2·32+4) = 68         |
| 3 | 64     | max(80, 2·68+4) = 140        |
| 2 | 128    | max(144, 2·140+4) = 284      |
| 1 | 256    | max(272, 2·284+4) = 572      |
| 0 | 512    | max(528, 2·572+4) = **1148** |

Level-0 strip = **1148 rows** (not 528 from the optimistic helper,
not 830 from the pre-impl analysis). At 4096² fine_w that's
`1148·4096·4 = 18.8 MiB` per channel per buffer (vs 64 MiB full-
image — **−70.6%** per buffer at level 0).

The verified formula ships as `mode_b_strip_h_at_level(k, h_body,
k_split)`. The estimator clamps `min(R_k, image_h_at_level_k)` at
each level to handle the degenerate case where back-projected strip
exceeds full-image (e.g. `1024² h_body=512` back-projects to 1148 >
1024 at level 0 — degenerates to full-image storage, signaling the
Auto resolver to pick Full).

### 2. Estimator — UPDATED 2026-05-27 (P2.0)

`estimate_gpu_memory_bytes_strip_pair` now uses
`mode_b_strip_h_at_level` with the per-level clamp. Verified ratios:

| size  | h_body | ratio (StripPair / Full)  | savings    |
|-------|--------|---------------------------|------------|
| 1024² | 256    | 58.7%                     | −41.3%     |
| 1024² | 512    | 100% (degenerate)         | 0% — pick Full |
| 4096² | 256    | **19.8%**                 | **−80.2%** |

The 4096² result is excellent — the deep-level full-image floor is
tiny (level 8 at 4096² is 16×16 px). Today's measured nvsmi delta
of −22.7% will move to roughly the −80% estimator target once the
per-strip buffers actually shrink (Phase 2's 7 buffer shrinks).

`tests/mode_b_walker_parity.rs` asserts:
- `1024² h_body=256 → ratio < 0.65` (P2.0: passes at 0.587)
- `1024² h_body=512 → 0.99 ≤ ratio ≤ 1.05` (degenerate fallback)
- `4096² h_body=256 → ratio < 0.25` (P2.0: passes at 0.198)

JOD parity preserved (bit-identical |diff|=0.0 at 128² and 1024²).

### 3. The REF/DIST gauss_ref clobber forces a per-strip reorganization

Today's pipeline runs:

```text
_dispatch_ref_weber_pyramid_only(ref_srgb):
  _dispatch_gauss_pyramid_gpu(ref_srgb)   // writes gauss_ref[0..n_levels]
  _finalize_weber_pyramid_after_gauss()    // reads gauss_ref, writes bands_ref + log_l_bkg

_dispatch_dist_weber_pyramid_only(dist_srgb):
  _dispatch_gauss_pyramid_gpu(dist_srgb)  // CLOBBERS gauss_ref with DIST data
  _finalize_weber_pyramid_after_gauss()    // reads gauss_ref (DIST), writes bands_dis_strip

_run_d_bands_band_loop()                   // reads bands_ref + bands_dis_strip
```

The `gauss_ref` buffer is shared — REF then DIST. The recipe's
strip-major outer dispatch implicitly requires (for k < K_SPLIT):

1. Per strip s, build REF gauss for body+halo at level 0 → write
   gauss_ref's strip rows.
2. Per strip s, reduce REF gauss through all shallow levels.
3. Per strip s, finalize REF weber for all shallow levels → writes
   bands_ref strip rows + log_l_bkg strip rows.
4. Per strip s, build DIST gauss (clobbers gauss_ref).
5. Per strip s, reduce DIST gauss through all shallow levels.
6. Per strip s, finalize DIST weber + csf + masking + pool for
   all shallow levels.

For steps 1-3 to coexist with later strips' steps 4-6 without
clobbering REF data, **bands_ref needs to be allocated per-strip
sized AND the REF state from prior strips must not be reused for
the current strip's band loop**. The latter is satisfied because
each strip's band loop only reads its own bands_ref rows; the
former requires a per-strip bands_ref buffer.

The alternative — keeping gauss_ref full-image but separate from
gauss_dis — adds one full-image pyramid (3 channels × sum_level_pixels
× 4 bytes ≈ 256 MiB at 4096²) and breaks the memory budget. The
correct path is per-strip bands_ref, per-strip gauss_ref for shallow
levels.

### 4. The existing strip walker helpers are level-major-outer-only

All four existing strip walker helpers
(`_reduce_gauss_pyramid_strip_walker`,
`_finalize_weber_pyramid_strip_walker`,
`_dispatch_dist_weber_csf_strip_walker_for_level`,
`_run_band_masking_strip_walker`) iterate strips internally with
**level-major outer** semantics. They read full-image previous-level
buffers via reflection-against-logical-image-bounds. Strip-major
outer requires either:

- New helpers that take a fixed strip index `s` and run all-shallow-
  levels for that strip, OR
- Refactoring the existing helpers to accept an `(s, k)` tuple and
  iterating from the outer caller.

### 5. Decomposed Phase 2 increment plan

The brief's "canary + 7 shrinks" decomposition front-loads the
buffer-shape change AND the dispatch-order change into the canary
commit. The honest decomposition (each step is its own commit +
JOD parity gate at 128² / 1024² / 4096²):

**P2.0 — Verify the halo back-projection math.**
- Read each strip-aware kernel's reflection bounds and write a
  table mapping `(level k, kernel)` → halo. Settle on either the
  simple `8·2^k` from the recipe or the iterative table from §1.
- Document the result in this section before P2.1.
- This is read-only; no code change.

**P2.1 — Add strip-major outer dispatch with FULL-IMAGE buffers.**
- New method invoked from `compute_dkl_jod` only for `StripMode::Pair`.
- Iterates `s in 0..n_strips_at_level_0`.
- Per strip: runs REF gauss + REF weber + DIST gauss + DIST weber
  + band loop + pool for that strip's body+halo rows at each
  shallow level.
- BUFFERS REMAIN FULL-IMAGE. The strip writes its body+halo rows
  into full-image buffers; overlap zones get overwritten by neighbour
  strips (deterministic — kernel is pure).
- Levels `k >= K_SPLIT` run via the existing level-major path AFTER
  all strips complete the shallow-level work.
- JOD parity gate: `|jod_strip_major - jod_level_major| ≤ 1e-4` at
  128² / 1024² / 4096².
- No memory change yet. This isolates the dispatch-order
  correctness question from the buffer-shape question.

**P2.2 through P2.8 — Shrink one buffer at a time.**
- Each commit shrinks ONE buffer to body+halo-sized per the
  verified back-projection table:
  - P2.2 → `bands_dis_strip` (already body-sized; widen to
    back-projected H_k)
  - P2.3 → add `bands_ref_strip` per-strip-sized sibling
  - P2.4 → `t_p_*` in `DBandsTransient`
  - P2.5 → `m_raw / m_mid / m_blur` in `DBandsTransient`
  - P2.6 → `log_l_bkg / log_l_bkg_dis / l_bkg_fine`
  - P2.7 → `gauss_ref` for k<K_SPLIT
  - P2.8 → `d_scratch[k]` non-`.d` fields
- Each commit must hold the JOD parity gate AND maintain Full
  mode unchanged (per MODE_SELECTION.md §"Wall-time benchmark"
  and §"Auto resolver").

**P2.9 — Wall-time bench + perf-aware resolver.**
- Extend `examples/mem_mode_b_vs_full.rs` with wall-time capture
  (n=20, p50/p25/p75 per cell).
- Commit `benchmarks/cvvdp_mode_b_wallclock_2026-05-27.csv`.
- Wire `pipeline::strip_perf_ratio_for_size` lookup table per
  MODE_SELECTION.md §"Auto resolver".

### 6. Why this agent honest-stopped before P2.1

The recipe at lines 683-787 lays out the eleven steps that the
canary commit must land *together* to avoid breaking JOD. The
honest read of the current pipeline is:

- The four existing strip walkers must each grow a strip-major-outer
  twin OR be refactored to accept the outer `s` index. Either
  shape is a substantial walker rewrite (~400 LOC each, across
  four functions).
- The CSF + masking pipeline must run per-strip with halo-aware
  buffer reads that don't yet exist for `bands_ref` (only
  `bands_dis_strip` has a per-strip variant today).
- The REF/DIST gauss_ref clobber order requires the REF-side
  walker (currently atomic) to be sliced per strip and interleaved
  with the DIST side.
- The halo back-projection table (§1) must be verified against
  the actual kernels' reflection bounds before the buffer allocator
  can be sized correctly.

In one session this is too many uncertainties stacked together for
a JOD-parity gate to land cleanly. The risk: ship a canary with
subtle drift that's not caught until a downstream parity sweep
runs on real corpora.

The decomposition in §5 sequences the work so each commit's risk
is bounded by one parity gate. P2.0 + P2.1 together should be one
session's worth; the seven shrinks can then proceed at one or
two per session.

### 7. What's safe to do RIGHT NOW (low-risk pre-work)

If a follow-up session has 1-2 hours and wants to make progress
without taking on P2.1's full surgery:

- **Verify the halo back-projection** by reading each
  `downscale_strip_kernel` / `upscale_*_strip_kernel` /
  `pu_blur_v_*_strip_aware_kernel` and documenting the actual
  reflection-row range per call.
- **Add `mode_b_strip_h_at_level(k, h_body, k_split)`** helper that
  returns the back-projected level-k strip height (body + 2·halo),
  alongside the existing `mode_b_halo_at_level` (which keeps the
  band-resolution semantics it was originally meant to have).
- **Add an estimator variant** `estimate_gpu_memory_bytes_strip_pair_back_projected`
  that uses the new helper.
- **Add a side-by-side delta print** in `mem_mode_b_vs_full` so
  the optimistic vs realistic numbers ship together.
- **Update tests** in `mode_b_walker_parity.rs` to expect the
  realistic ratio (~`< 0.35` at 4096² rather than `< 0.25`) — this
  may surface that the recipe's "-88%" target was based on the
  optimistic model and needs to be revised to "-65 to -75%".

That work is mechanical and JOD-neutral (it changes only memory
prediction, not the runtime dispatch). It positions the next
session to start P2.1 with a verified halo budget.

## Phase 2 — P2.1 implementation analysis (2026-05-27)

> **Status: investigation complete, second honest-stop. The P2.1
> canary remains structurally entangled with kernel/walker rewrite
> work that exceeds a single session's safe surgical scope.** This
> section documents the precise dispatch-order semantics, the
> halo-handling requirement that surfaces inside the masking
> walker, and a refined decomposition that breaks the canary into
> three smaller commits — each gated by JOD parity at every
> production size. The decomposition trades one large commit for
> three small ones; each shippable in ~half a session, with the
> first two delivering zero memory change and the third hitting
> the −70% bands_dis_strip+upscaled_c_strip footprint shrink.

### 1. Why "strip-major outer with full-image buffers" is NOT a no-op

The §5 P2.1 plan above proposes "strip-major outer dispatch with
FULL-IMAGE buffers" and asserts buffer overlap is "deterministic
because the kernel is pure". A careful read of the masking walker
(`_run_band_masking_strip_walker` at pipeline.rs:5167) reveals an
order-sensitive cross-strip read that the §5 plan does not address:

The masking V-blur (`pu_blur_v_3ch_scaled_strip_aware_kernel`)
dispatches over a halo-padded window of `t_p_*[k]`:
```
top_global = body_offset_y.saturating_sub(HALO=6)
bot_global = (body_offset_y + body_h + HALO=6).min(bh)
```
and reads m_mid rows in `[top_global, bot_global)`. m_mid was
written by H-blur on the same window of m_raw which was written
by min_abs from t_p_*[k] over the same window. So masking strip s
**reads t_p_*[k] rows above and below its own body** — those rows
belong to strip s-1 (above) and strip s+1 (below).

In today's level-major-outer + strip-major-inner pattern, by the
time level k's masking strip walker fires, the csf walker (one
loop earlier in the band loop body) has fully populated t_p_*[k]
for ALL strips of level k. So the halo reads find valid data
written by sibling strips' csf passes.

Inverting to strip-major outer breaks this. When strip s's
masking runs for level k, strip s+1's body of t_p_*[k] has NOT
yet been written (we're still iterating s). The V-blur reads
stale data (zeros from construction or leftover from a prior
call) and the JOD output diverges.

The fix is to **extend the csf walker so each strip writes
body+halo rows of t_p_*[k]** (not just body rows). Each strip
then computes the halo redundantly; when masking strip s reads
its halo, the data is valid (computed by this strip's own csf
pass or overwritten by the next strip's csf with the same
deterministic result).

### 2. The extend-csf-to-body+halo requirement cascades to bands_dis_strip

Looking at `_dispatch_dist_weber_csf_strip_walker_for_level` at
pipeline.rs:4859, csf reads three per-pixel inputs:
- `band_ref_*` (full-image bands_ref[k]) — slice at offset, no shape change
- `log_l_bkg` (full-image weber_scratch[k].log_l_bkg) — slice at offset, no shape change
- `bands_dis_strip[k]` (strip-local, fine_w × strip_h_at_k) — **READS first n_strip_fine = body_h × fine_w elements, indexed from buffer row 0**

For csf to read bands_dis_strip body+halo (n_strip_window = strip_window_h × fine_w elements), bands_dis_strip must be sized fine_w × strip_window_h (= fine_w × `mode_b_strip_h_at_level(k, ...)`), not fine_w × strip_h_at_k.

This is what P2.2 in the §5 plan calls for: grow bands_dis_strip
to the back-projected R_k. But P2.2 alone is meaningless without
P2.1's extended csf dispatch — they ARE the same atomic change.

### 3. The bands_dis_strip growth path requires a bigger weber strip dispatch

Stage 3 of `_dispatch_dist_weber_csf_strip_walker_for_level`
(`subtract_weber_3ch_strip_kernel`) writes bands_dis_strip body
rows. To write body+halo, the dispatch needs:
- Input `gauss_ref[k].planes[c]` sliced at `top_global * fine_w * 4`
  (full-image; slice is valid).
- Input `upscaled_c_strip[c]` body+halo data. **upscaled_c_strip is
  also strip-local (fine_w × strip_h_at_k today)** — must also grow.
- Input `l_bkg_fine` sliced at `top_global * fine_w * 4` (full-image; OK).
- Output `bands_dis_strip[c]` body+halo (sized fine_w × R_k now).
- Output `log_l_bkg_dis` sliced at `top_global * fine_w * 4` (full-image; OK).

Stages 1-2 (separable upscales) similarly write upscaled_c_strip
body rows; to write body+halo, the upscale_v_strip + upscale_h_strip
dispatches grow correspondingly.

So the canary really needs three buffer changes (bands_dis_strip,
upscaled_c_strip, and one vscratch sibling) **plus** the kernel
dispatch shape changes (extend each per-strip launch from
n_strip_fine = body_h × fine_w to n_strip_window = strip_window_h
× fine_w).

### 4. The strip-major outer loop itself is the smallest piece

Once stages 1-4 of the csf walker dispatch over body+halo, the
existing `_run_band_masking_strip_walker` works unchanged — its
halo reads find valid data per strip. The strip-major outer
reordering is then a straightforward refactor of `_run_d_bands_band_loop`:

```rust
let n_strips_at_0 = self.height.div_ceil(h_body);
let k_split = mode_b_k_split(h_body, n_levels);

// Shallow levels strip-major-outer.
for s in 0..n_strips_at_0 {
    for k in 0..k_split {
        // One-strip csf walker (body+halo dispatch) for (s, k).
        // One-strip masking walker for (s, k).
    }
}
// Deep levels level-major (unchanged from today).
for k in k_split..n_levels {
    // existing per-level path
}
// Baseband bypass + pool.
```

The "one-strip csf walker" requires factoring `_dispatch_dist_weber_csf_strip_walker_for_level`'s inner `for s in 0..n_strips`
loop into a `_dispatch_dist_weber_csf_strip_s_for_level(s, k)`
helper. Similarly for the masking walker.

### 5. Refined 3-commit decomposition

Splitting the canary into three commits that each ship JOD bit-
identical + a single mechanical change keeps the surgery
bounded:

**P2.1a — Factor per-strip helpers (JOD bit-identical).**
- Extract `_dispatch_dist_weber_csf_strip_s_for_level(s, k, ...)`
  from the current `_dispatch_dist_weber_csf_strip_walker_for_level`.
  The existing function becomes a thin wrapper that loops `for s in
  0..n_strips` and calls the per-strip helper.
- Same for `_run_band_masking_strip_walker` → factor into a
  `_run_band_masking_strip_s_for_level(s, k, ...)` plus the loop wrapper.
- No dispatch order change, no buffer change. JOD identical to the
  current strip walker. Sets up the architecture for P2.1b.
- Risk: very low. JOD parity must hold bit-exact at 128² / 1024².

**P2.1b — Extend per-strip helpers to dispatch over body+halo.**
- Modify the new per-strip helpers from P2.1a to dispatch over the
  halo-padded window (n_strip_window = strip_window_h × fine_w),
  not the body-only window. Write to bands_dis_strip / t_p_*
  rows 0..strip_window_h (with src_strip_offset = top_global).
- Buffer sizes don't change yet (bands_dis_strip / upscaled_c_strip
  stay at strip_h_at_k rows). This means at strips near the bottom
  of the image, body_h is small and the body+halo window may not
  fit — clamp to bh, fall back to the existing body-only dispatch
  for those.
- Need to verify: when csf writes redundant halo rows of t_p_*[k]
  (each strip re-computes its halo), the result is bit-identical
  to today's level-major (where each strip's csf is computed once
  per t_p_* row).
- Strategy: pin a small test that captures one strip's t_p_*[k]
  body+halo output vs the same window from the full-image dispatch.
  Must be bit-exact (deterministic CSF, same inputs).
- Risk: moderate. JOD parity must hold bit-exact.

**P2.1c — Grow bands_dis_strip + upscaled_c_strip to R_k and invert
the outer loop.**
- Change `build_weber_scratch` to size bands_dis_strip and
  upscaled_c_strip at fine_w × R_k where R_k =
  `mode_b_strip_h_at_level(k, h_body, k_split)`.
- Replace `_run_d_bands_band_loop`'s level-major outer with strip-
  major outer for k < k_split (using the per-strip helpers from
  P2.1a+b).
- The deep level-major path (k >= k_split) stays unchanged.
- Risk: low (the order-correctness was proved bit-exact by P2.1a
  per-strip factorization + P2.1b halo dispatch). Memory savings:
  bands_dis_strip + upscaled_c_strip grow vs body-only, but vs
  full-image they shrink. Realistic ratio: vs full-image Mode B
  buffers ≈ 0.15-0.25 at 4096².

The total work is the same as P2.1+P2.2 in the original brief, but
each commit's risk is bounded by one parity gate, and the first
commit (P2.1a) is a JOD-bit-identical factorization that lands the
architecture without any behavioural change. P2.1b is where the
correctness work concentrates; P2.1c is mechanical once P2.1b is
green.

### 6. Why this session did not ship P2.1a–c

The agent landed P2.0 (verified back-projected halo helper) at
ff6c733d and had the canary decomposition above in scope. The
honest scope read:

- P2.1a (factor per-strip helpers): tractable in this session,
  but JOD-bit-identical refactors only land value when the
  follow-ons (P2.1b + P2.1c) also land. P2.1a alone is dead
  weight in the source tree.
- P2.1b (extend per-strip helpers to body+halo dispatch): requires
  new parity tests pinning the per-strip body+halo output against
  the full-image dispatch on a per-band basis. Building that test
  scaffold + iterating until bit-exact is the bulk of the work.
- P2.1c (buffer growth + outer loop inversion): mechanical once
  P2.1b is green.

The current cvvdp-gpu test suite gates only `compute_dkl_jod`
output (the final JOD scalar). A per-strip per-band parity test
that captures t_p_*[k]'s body+halo window does not yet exist —
adding it is the actual canary work.

The agent recommends the next session start with the per-strip
parity test (a new `tests/strip_csf_halo_parity.rs` that compares
one strip's t_p_*[k] body+halo from the per-strip helper against
the same window of t_p_*[k] from the full-image csf), and only
then start P2.1a. That ordering puts the verification scaffold in
place BEFORE the refactor, so the refactor's correctness is
machine-checked at every step.

### 7. Recap of the architectural constraints

- `bands_dis_strip[k]` is the canary buffer. Body-only today;
  must grow to R_k for body+halo dispatch.
- `upscaled_c_strip[k]` is the second canary buffer. Must grow
  to R_k in lockstep with bands_dis_strip.
- `t_p_*[k]`, `m_*[k]` stay full-image in P2.1; subsequent
  commits (P2.4-P2.5) can shrink them once strip-major outer is
  in place because each strip writes/reads only its body+halo
  range.
- `bands_ref[k]`, `weber_scratch[k].log_l_bkg`, `gauss_ref[k]`,
  `l_bkg_fine` remain full-image — they're read at per-strip
  body+halo slices by the per-strip helpers.
- Deep levels (k >= k_split) continue using the level-major
  dispatch with full-image storage. Their absolute pixel count
  is small (level 8 at 4096² = 256 pixels) so the memory hit is
  negligible.

## Phase 2 — P2.1b LANDED (2026-05-27)

### Summary

P2.1b extends the per-strip CSF walker
(`_dispatch_dist_weber_csf_strip_s_for_level`) to dispatch stages 1-4
over a **body+halo window** at level k for shallow levels (k <
k_split). The strip-local buffers `bands_dis_strip` and
`upscaled_c_strip` grow from body-only (`fine_w × strip_h_at_k`) to
back-projected **R_k** (`fine_w × mode_b_strip_h_at_level(k, h_body,
k_split)`) to accommodate the halo writes.

Under the existing level-major-outer caller, adjacent strips overlap
at halo rows of `t_p_*[k]`. The CSF kernel is deterministic on the
same inputs (band_ref + log_l_bkg are full-image; bands_dis_strip is
recomputed per strip from full-image gauss + the strip-local upscale,
which is itself a function of the global row index via the strip-
aware kernels' `zy_base = local_y + body_offset_y` math). So overlap
rows receive identical values from both strips that touch them, and
the final t_p_*[k] state — and the JOD scalar — is bit-identical to
today's body-only dispatch.

### JOD parity gates (all bit-identical |diff|=0.0)

- `mode_b_walker_jod_matches_full_at_128` — 128² h_body=32 (4 strips
  at L0). JOD 9.456145, |diff|=0.000e0, strip_dispatch_counter=506.
- `mode_b_walker_jod_matches_full_at_1024` / `_h_body_256` — 1024²
  h_body=256. JOD 9.458330, |diff|=0, counter=716.
- `p21b_csf_halo_parity_256_h_body_128` — 256² h_body=128 (2 strips).
  JOD 9.457394, |diff|=0.
- `p21b_csf_halo_parity_512_h_body_128_4_strips` — 512² h_body=128.
  JOD 9.457932, |diff|=0.
- `p21b_csf_halo_parity_1024_h_body_128_8_strips` — 1024² h_body=128
  (8 strips at L0, 7 inter-strip halo overlaps). JOD 9.458330, |diff|=0.
- `p21b_csf_halo_deterministic_across_calls` — two consecutive
  `score()` calls produce identical JOD (no halo-row state leakage).
- `p21b_csf_halo_parity_single_strip_degenerate` — single-strip
  degenerate (h_body=h). JOD 9.457394, |diff|=0.

### Measured memory (nvidia-smi delta during compute_dkl_jod)

| size  | Full       | StripPair(256) | Δ vs Full |
|-------|------------|----------------|-----------|
| 1024² | +385 MiB   | +353 MiB       | **−8.3%** |
| 4096² | +4225 MiB  | +3457 MiB      | **−18.2%** |

P2.1b's per-strip buffer growth (body-only → R_k) is the reason the
4096² nvsmi delta moved from today's −22.7% to −18.2% — strictly more
strip-side memory now allocated for halo coverage. This is the
**necessary** cost of body+halo dispatch; the user-visible memory
win requires the follow-on commits (P2.4-P2.8) that shrink the
full-image transients (t_p_*, m_*) per-strip — only possible AFTER
the outer-loop inversion in P2.1c.

### Parity test scaffold

`tests/strip_mode_b_csf_halo_parity.rs` pins five `(size, h_body,
n_strips_at_L0)` combinations that stress the inter-strip halo
overlap. All currently bit-identical to Full mode JOD.

### What P2.1b does NOT do

- Does NOT change the outer-loop order: the caller
  `_dispatch_dist_weber_csf_strip_walker_for_level` still iterates
  strips inside the level-major band loop. P2.1c flips this.
- Does NOT shrink t_p_*[k] / m_*[k] — those stay full-image. Those
  shrinks land in P2.4-P2.5 (after P2.1c enables them by inverting
  the outer loop).
- Does NOT change deep-level (k >= k_split) dispatch — those keep
  body-only sizing because `mode_b_halo_at_level` returns 0 for deep
  levels and the helper's halo extension reduces to body-only there.

### Why ship P2.1b separately from P2.1c

The buffer-growth + dispatch-extension can be JOD-bit-verified under
the existing level-major caller (P2.1b). The outer-loop inversion is
then a pure scheduling refactor (P2.1c) — no kernel changes, no
buffer changes. Splitting commits keeps each commit's risk bounded
by one parity gate.
