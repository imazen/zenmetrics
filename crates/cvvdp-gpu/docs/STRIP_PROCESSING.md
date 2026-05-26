# Strip processing in cvvdp-gpu

> **Status (task #77, 2026-05-26): cvvdp does not strip.** The
> previous `MemoryMode::Strip { capped_levels: Some(k) }` variant was
> rolled back because **capping the pyramid depth changes the JOD
> value at any k < 9**. The supported `MemoryMode` is `{ Auto, Full }`
> only. Methodology trace from the original capped-pyramid sweep
> lives under `archived/`.

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
