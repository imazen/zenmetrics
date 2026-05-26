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
