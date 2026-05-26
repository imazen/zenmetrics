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

### Phase 3 design notes (for the next agent)

The cvvdp pipeline is structurally more complex than dssim or
zensim's strip walkers:

1. **9 pyramid levels with per-level halo accumulation.** The dist
   Weber pyramid build at level k uses `gauss[k+1]` to subtract
   upscaled-coarser from fine — every level needs its halo-padded
   sibling at the level above. Halo at base ≈ `6 × (2^max_level - 1)`
   ≈ 1500 rows at 9 levels for the σ=3 PU-blur context.
2. **Per-band σ=3 PU blur halo.** At each non-baseband band, the
   masking chain runs `pu_blur_h_3ch_kernel` +
   `pu_blur_v_3ch_scaled_kernel` — a 13-tap separable Gaussian. The
   strip-side dist band needs `±6` rows halo *at that band's
   resolution*.
3. **Pyramid kernels are not strip-aware today.** `downscale_kernel`,
   `upscale_v_kernel`, `upscale_h_kernel`, and
   `subtract_weber_3ch_kernel` all take `(width, height)` parameters
   and apply reflection boundary handling at the array edges. A strip
   walker either (a) extends them to take a `body_offset` +
   `body_height` parameter that selects which slab of an oversized
   buffer is "the strip", or (b) reuses them on a strip-sized buffer
   with explicit halo rows pre-populated from the prior/next strip.

**Path of least resistance**: option (b) — keep the kernels
unchanged, allocate strip-sized dist buffers, and copy halo rows
from adjacent strips before each Weber pyramid pass. This is what
dssim-gpu does (see `dssim-gpu/src/pipeline.rs::compute_with_reference`
and the `copy_rows_kernel` it uses). The complication for cvvdp is
that the halo per level is **larger** than dssim's (`6 × 2^(k-2)` vs
dssim's `2 × 2^k`), so the strip buffer must be sized for the
worst-case halo at every level.

**Where the new code goes**: introduce a `_dispatch_dist_strip_band_loop`
helper in `pipeline.rs` that mirrors the Full-mode
`_dispatch_d_bands_dist_and_band_loop` but walks the dist side in
strips, reading ref state directly from `ref_full_state`. The pool
finalizer (`_pool_and_finalize_jod`) stays unchanged — atomic-pool
sums accumulate across strip iterations the same way they accumulate
across blocks in a single Full dispatch.

`tests/strip_mode_e_parity.rs` already pins the JOD value contract;
Phase 3 must keep all 11 tests passing.

## What `Auto` does today

For `(width, height)` whose Full footprint exceeds the VRAM cap,
`resolve_auto` now picks `Strip { h_body: STRIP_H_BODY_DEFAULT }`
instead of surfacing `TooBigForFull`. Phase 2 (today) gives the
caller a working cached-ref path with full ref-state durability;
Phase 3 will shrink the dist working set so the same picker decision
actually fits a smaller VRAM cap.

If even the Phase 2 estimate (Full + ref cache) exceeds the cap,
`resolve_auto` still surfaces `Error::TooBigForFull { needed, cap }`.
