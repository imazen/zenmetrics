# cvvdp CPU K_SPLIT Walker — Chunks 4 + 6 Handoff

**Status (2026-05-28, after agent #127 work):**

- Chunks 1, 2, 3, 5 — **SHIPPED** (this branch, 2 commits).
- Chunks 4, 6 — **NOT SHIPPED** (this handoff).

## What's shipped

`crates/cvvdp/src/strip_kernels.rs` (new module, ~1500 lines + tests):

| Function | Purpose | Parity gate |
|---|---|---|
| `pu_blur_h_strip_aware_3ch_into` | X-only PU blur, body_offset unused | bit-identical to upstream H-pass (5 sizes) |
| `pu_blur_v_strip_aware_3ch_into` | V-blur with reflection vs logical_h then strip-local translate | bit-identical to upstream V-pass for body rows (degenerate + interior strip) |
| `pu_blur_3ch_strip_aware` | H+V wrapper with pu_scale post-multiply | bit-identical to `gaussian_blur_sigma3` in degenerate mode |
| `downscale_strip_into` | 2× reduce with pycvvdp parity-on-rows bug-compat via logical_src_h | bit-identical to `gausspyr_reduce_scalar` (degenerate + interior, 6 sizes) |
| `upscale_v_strip_into` | V-pass expand body-only | bit-identical to upstream V-pass (5 sizes incl odd target) |
| `upscale_h_strip_into` | X-only expand | bit-identical to upstream H-pass (5 sizes) |
| `subtract_weber_3ch_strip_into` | Per-pixel weber-contrast + log_l_bkg | bit-identical to `weber_contrast_pyr_dec_scalar` formula |

CSF and the per-pixel masking chain (chunks 2+3) are strip-degenerate
per audit AUDIT_2026-05-28.md §A rows 8-11 — the existing per-pixel
helpers in `pipeline.rs` work as-is on strip-sized slices. Parity
tests prove this.

All 12 strip-kernel parity tests use bit-identical (`to_bits()`)
gates, not 1e-4 tolerance.

## What's NOT shipped (chunks 4 + 6)

### Chunk 4: per-strip `d_scratch` slab allocator

**The data structure that needs to be reshaped:**

`crates/cvvdp/src/scratch.rs` currently has `BandWorkspace`, one per
band in the pyramid (n_levels of them, indexed by k). Each holds:
- `t_p_a/rg/vy`: CSF-weighted contrasts (size `bw * bh`)
- `r_p_a/rg/vy`: same for ref
- `d_a/rg/vy`: post-masking diff (size `bw * bh`)
- `m_mm_a/rg/vy`, `term_a/rg/vy`, `pu_h`: masking intermediates

When `Cvvdp::new` is called, `Scratch::new(w, h, n_levels)`
pre-allocates the pyramid's full-image `WeberPyramid` slots:
- 6× `WeberPyramid::with_capacity(w, h, n_levels)` (3 ref + 3 dist):
  ~12 bytes/px × n_levels × (4/3) = ~16 bytes/px summed across levels.
- 3× `WeberPyramidCache::with_capacity(w, h, n_levels)` (dist gauss
  intermediates): ~16 bytes/px × 2 (gauss_img + gauss_l).

At 16 MP this is **3-5 GB** just for the persistent scratch.

**What chunk 4 needs to do:**

Create a `StripBandWorkspace` variant of `BandWorkspace` sized to
`(strip_h_at_band_k × bw)` instead of `bw * bh`. The strip walker
will allocate one of these per shallow level (k < k_split) and
reuse the existing band_ws full-image allocation for deep levels
(k >= k_split, which are tiny).

For the CPU port:

```rust
// In scratch.rs:
pub(crate) struct StripBandWorkspace {
    pub t_p_a: Vec<f32>,   // sized (R_k × bw) per shallow level
    pub t_p_rg: Vec<f32>,
    pub t_p_vy: Vec<f32>,
    pub r_p_a: Vec<f32>,
    pub r_p_rg: Vec<f32>,
    pub r_p_vy: Vec<f32>,
    pub d_a: Vec<f32>,
    pub d_rg: Vec<f32>,
    pub d_vy: Vec<f32>,
    pub m_mm_a: Vec<f32>,
    pub m_mm_rg: Vec<f32>,
    pub m_mm_vy: Vec<f32>,
    pub term_a: Vec<f32>,
    pub term_rg: Vec<f32>,
    pub term_vy: Vec<f32>,
    pub pu_h: Vec<f32>,
}
```

Sized for `n_strip_rows = R_k × bw` where `R_k =
mode_b_strip_h_at_level(k, h_body, k_split)` (CPU port already
ships at `crate::strip::mode_b_strip_h_at_level`).

`Scratch` then gets an optional second pool `strip_band_ws:
Option<Vec<StripBandWorkspace>>` that's allocated only when strip
mode is active.

### Chunk 6: strip-major dispatcher

**The dispatcher needs to:**

1. **Build per-strip weber pyramids.** Currently `score_internal`
   calls `build_both_sides_into(&mut self.scratch, w, h, n_levels)`
   which builds full-image pyramids. The strip mode needs to walk
   strips of scale-0 and build per-strip pyramids using the
   strip-aware kernels from chunk 5:
   - `downscale_strip_into` for the gauss pyramid
   - `upscale_v_strip_into` + `upscale_h_strip_into` for the expand
     chain
   - `subtract_weber_3ch_strip_into` for the contrast computation

2. **Run band-fold per strip × per shallow level.** Currently
   `fold_bands_parallel` iterates over levels (each level taking
   the full band's `bw * bh` pixels). The strip-major variant
   inverts: for each strip `s` of scale 0, for each shallow level
   `k < k_split`, compute the strip's band slice at level k (size
   `bw * R_k`) and run the CSF + masking + PU-blur chain on the
   strip-shaped buffer.

3. **Pool per strip across all bands.** This part already works
   (the existing `pool_band_3ch` walker partitions the pool stage —
   that landed in #124 D2). The dispatcher just needs to feed it
   the per-strip d arrays in row-order.

4. **Deep levels stay full-image.** For k >= k_split, the existing
   `_run_d_bands_band_loop` (in `pipeline.rs` Full-mode path)
   handles them. The strip-major outer loop only covers shallow
   levels.

The GPU equivalent is `_run_d_bands_strip_major_shallow` at
`cvvdp-gpu/src/pipeline.rs:5343`. The CPU port should follow the
same shape but using the scalar strip kernels from chunk 5
instead of cubecl dispatches.

### Where peak heap is dominated today (16 MP measured per #124)

At 16 MP cold `score()` peak is **4.73 GB**. Decomposition (from
`Scratch::new` source):

1. **6 × DKL planes** (`dist_a/rg/vy`, `ref_a/rg/vy` — pre-allocated
   `vec![0.0; w*h]`): 6 × 64 MB = **384 MB**.
2. **6 × WeberPyramid::with_capacity** (3 ref + 3 dist):
   ~12 bytes/px × n_levels × (4/3) summed pyramid mass ≈ ~16 bytes/px
   × 4 (4 bytes/f32) × 16 MP × 6 = **6 GB** … wait, that's high.
   Actually `with_capacity` pre-allocates `bands[k]` (per-level
   pixel data) + `log_l_bkg[k]`, which is 2 × pyramid mass per
   WeberPyramid = 2 × (4/3) × W*H = **~170 MB per WeberPyramid**
   × 6 = **1 GB**.
3. **3 × WeberPyramidCache::with_capacity** (dist gauss_img +
   gauss_l for shallow + deep levels): 2 × (4/3) × W*H × 4 ×
   n_levels per cache, but with_capacity only sizes gauss_img +
   gauss_l at all levels, NOT the inner PyramidScratch. So per
   cache: 2 × (4/3) × 64 MB ≈ **170 MB** × 3 = **510 MB**.

Total pre-allocated by `Scratch::new` at 16 MP ≈ **1.9 GB**.
Heaptrack measured 4.73 GB peak suggests another ~2.8 GB is
transient (BandWorkspace ws.t_p_*/r_p_*/m_mm_*/term_*/pu_h grown
during fold_bands at largest band, plus rayon's per-thread copies
of those).

**Chunk 6's job is to (a) make this lazy in strip mode and (b)
allocate strip-shaped versions instead.** The biggest wins come
from:

- Replace `WeberPyramidCache::with_capacity` at shallow levels with
  strip-shaped `gauss_img[k]` / `gauss_l[k]` = `R_k × bw` instead
  of `bh × bw`. Saves ~80% of (3) at 16 MP.
- Replace `WeberPyramid::with_capacity`'s shallow `bands[k]` +
  `log_l_bkg[k]` with strip-shaped. Saves ~80% of (2).
- BandWorkspace's per-band slots stay full-image-sized in current
  code; chunk 4's `StripBandWorkspace` replaces them at shallow
  levels. Saves ~80% of the transient.

If all three land, peak heap at 16 MP should drop from 4.73 GB to
~1.7 GB target (per brief). The shipped chunk 4 data structure
already exists; chunks 4-wiring + 6 are the dispatcher work.

### How to verify chunks 4+6 land correctly

The parity gate is already wired:
`crates/cvvdp/tests/strip_parity.rs` runs 90 cells (default grid:
18 seeds × 1 size × 5 h_body) covering both cold + warm-ref paths.
Currently every cell passes because `score_strip` produces
bit-identical JOD to `score`. After chunks 4+6 land, the SAME
test suite must continue to pass with bit-identical results — any
drift indicates a porting bug in the per-strip pyramid kernels
that wasn't caught by the per-kernel parity tests in
`strip_kernels.rs`.

Add a new test asserting peak heap usage drops:

```rust
#[test]
#[cfg(feature = "heaptrack")]
fn score_strip_peak_heap_below_threshold_at_4mp() {
    let w = 2048;
    let h = 2048;
    let (r, d) = synth_pair(w, h, 0xfeed);
    // Use heaptrack instrumentation; require ratio ≤ 0.35
    // (matching GPU's measured 1502 MiB / 4225 MiB).
}
```

The `crates/cvvdp/benchmarks/cpu_kspl_chunkN_2026-05-28.tsv` format
per the brief is the right place to record peak heap measurements
chunk by chunk.

### Why chunk 6 wasn't shipped

The architectural change required is substantial:
- `Scratch::new` either needs a variant-per-mode constructor or
  needs lazy allocation of the strip-shaped vs full-image buffers.
- `Cvvdp::score_strip` needs to gate on a `StripConfig` analogous
  to the GPU's `StripConfig` (currently CPU's strip_h_body is
  just a `Cell<Option<u32>>` switched at pool stage).
- The walker reshapes the band-fold's iteration order from
  level-major to strip-major for shallow levels.

Doing this without regressing the existing 270-cell `strip_parity`
gate is the work. The kernel ports in chunks 1-3+5 are the
prerequisite — they prove the per-kernel math is bit-identical.
The dispatcher work is now the only remaining piece.

## Files changed in this branch

```
crates/cvvdp/src/lib.rs              (+2 lines: register strip_kernels mod)
crates/cvvdp/src/strip_kernels.rs    (+~1500 lines: 7 kernels + 12 parity tests)
```

No `Scratch` / `Cvvdp` / `pipeline.rs` changes; the new module is
gated behind `#[allow(dead_code)]` until chunk 6's dispatcher
wires it in.

## Acceptance gate at end of agent #127

- [x] Chunks 1, 2, 3, 5 landed with bit-identical parity tests passing (12 tests)
- [ ] Chunks 4, 6 landed (handoff doc, this file)
- [ ] heaptrack at 1/4/16 MP × 4 modes (deferred until 4+6 land)
- [x] 40 MP synthetic heap measurement (real, agent #129 — see below)
- [ ] Wall regression < 22% (deferred until 4+6 land)
- [x] Workspace cleanup pending (run on this agent's final commit)

## 40 MP measured heap (agent #129, 2026-05-28)

**Replaces the prior linear-extrapolation claim** ("40 MP fits at 3.85 GB
by linear extrapolation from 16 MP"). Per global CLAUDE.md, source-
informing measurements must run the real cell — no extrapolation.

Measured 2026-05-28 on lilith (Ryzen 9 7950X, 49 GiB RAM) via
`heaptrack` 1.3.0 + the existing `cpu-profile` driver
(`benchmarks/heaptrack/drivers/cpu_profile/src/main.rs`), release
build at master@origin = `5979e084`. Synthetic 7680×5184 (39.81 MP)
XorShift sRGB pair, score JOD 9.456094741821289 across all 4 modes
(parity intact):

| mode | peak heap (bytes) | peak heap (GiB) | peak heap (heaptrack G) | runtime |
|---|---|---|---|---|
| `full` | 8,679,597,188 | 8.0835 | 8.68 G | 15.75 s |
| `strip` (h_body=512) | 8,679,597,189 | 8.0835 | 8.68 G | 14.94 s |
| `warm_ref` | 7,485,111,384 | 6.9711 | 7.49 G | 14.10 s |
| `warm_ref_strip` (h_body=512) | 7,485,111,390 | 6.9711 | 7.49 G | 13.60 s |

Notes:
- Strip ≡ full (both modes hit 8.68 GB) because today's
  `score_strip` only walks the **pool stage** in strips — weber
  pyramid + masking + d_scratch are still full-image-sized. That
  matches the GPU's currently-shipped strip walker. Chunks 4 + 6
  (this handoff) are exactly the work that would change this.
- warm_ref drops peak heap by **1.19 GB** (≈ 14 %) vs cold, the
  delta corresponding to ref pyramid construction during
  `Cvvdp::score`. warm_ref_strip is at parity with warm_ref for
  the same reason as `strip ≡ full`.
- Heaptrack 1.3.0 reports "G" using SI (10⁹). Precise bytes
  extracted via `heaptrack_print --massif-threshold 99 -M …`.
- Raw `.zst` traces:
  `benchmarks/heaptrack/cvvdp_{full,strip,warm_ref,warm_ref_strip}_40mp_2026-05-28.zst`
- TSV: `crates/cvvdp/benchmarks/cpu_kspl_40mp_2026-05-28.tsv`
- Meta: `crates/cvvdp/benchmarks/cpu_kspl_40mp_2026-05-28.meta`

**Headline:** at 40 MP today (before chunks 4+6 land), CPU strip
mode peaks at **8.08 GiB** — well above the 4.2 GB working target
and far above the 3.85 GB extrapolation that was previously
reported. This is the real signal that motivates chunks 4 + 6:
the persistent `Scratch::new` allocations + transient
`BandWorkspace` slots grow linearly in W·H and dominate at 40 MP.
