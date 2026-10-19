# NIQE + PIQE canonical-speed rework, 2026-10-19

Same disease as BRISQUE (scalar convs + FMA-libcall baseline x86), same fix:
archmage `#[attune]`/`attuned!` kernels with magetypes generic vectors —
**bitwise-identical** on every tier, verified corpus-wide.

## NIQE (`crates/niqe`)

- `conv_gauss` (was ~47% of e2e): both 7×7 MSCN convs (x and x²) run as one
  fused `f64x4`/`f64x8` kernel over 3-px edge-replicated f64 pad planes —
  `pad_edge3_pair` widens each f32 sample (and its square) once, not per
  tap. The oracle's *reversed* tap order (`r+3−a`, `c+3−b`) is preserved
  as padded offsets `r+6−a`, `c+6−b`; f64 `mul+add` stays two roundings on
  every backend (Rust never contracts to FMA) → bitwise on all tiers.
- `pairwise_sum_f32/f64` leaf (≤128 elems): the numpy 8-accumulator unroll
  is now an `f32x8` / 2×`f64x4` fixed-lane accumulate — identical per-lane
  add order, ~4× fewer instructions on the leaf bulk.
- `aggd`: unconditional `|v|`/`v²` streams take indexed writes
  (elementwise-exact, auto-vectorizes); the `np.argmin((r_gam−x)²)` scan
  (9801 iters/call) is a nearest-point `partition_point` on the **strictly
  increasing** r_gam table — proven bitwise-equivalent, guarded by the
  `r_gam_strictly_increasing` unit test.
- `compute_feature`: `np.roll` materializes via per-row rotation copies
  (≤2 contiguous runs), then `aggd` computes `v = a·b` inline — the full
  `prod` plane is gone. All 5 stat workspaces hoist to one `AggdScratch`.
- MSCN normalize: `f32x8`/`f32x16` elementwise (all correctly-rounded).
- Luma: three u8→product tables (`24.966·nf(b)` etc. — each entry the
  identical f64 product) remove 3 divs + 3 converts + 3 muls per pixel.
- `resample_rows`: per-row `aug` buffer hoisted (was 480 allocs).

## PIQE (`crates/piqe`)

- `gauss7` (was ~43%): separable 7-tap f64 blur runs as
  interior-vectorized `f64x4`/`f64x8` kernels — **no pad planes** (the
  first draft's pad planes added ~25 MB traffic and LOST to scalar).
  Interior outputs (x ∈ [3, w−3) etc.) need no clamps at all; the 3-px
  borders keep the scalar clamped-index path. The `im²` conv is fused
  into the row pass — `v*v` per tap IS the materialized `sq` element's
  rounding — removing one whole input plane.
- `sigma`/`norm` computed as a whole-plane `f64x4` pass (elementwise-exact).

## Correctness evidence

- **All 2016 konjnd-pairs images: zero score-bit diffs, both metrics**
  (`/tmp/nr-ab`, base `6038a9f3`).
- Full 176-row golden parity gates pass for both crates
  (`cargo test -p niqe/piqe --test parity`); all behavior tests pass.
- ARM NEON (`zen-arm-xl`, Neoverse-N1): score bit-identical to x86 AVX2
  on the golden image; NEON `.2d` kernels verified in the binary.
- `no_std`, wasm32, i686, aarch64, and the opt-in `avx512` feature build.

## Timing (r7900x, single thread, e2e incl. decode-free pipeline)

| metric | size | before | after | speedup |
|---|---|---|---|---|
| niqe | 640×480 | 34.08 ms | 17.31 ms | 1.97× |
| niqe | 1024×1024 | 108.33 ms | 66.49 ms | 1.63× |
| piqe | 640×480 | 8.14 ms | 5.31 ms | 1.53× |
| piqe | 1024×1024 | 45.97 ms | 22.10 ms | 2.08× |
| piqe | 640×480 (NEON) | — | 12.9 ms | — |

Residual hotspots are oracle-order-bound: niqe's `aggd` keeps numpy's
filtered materializations + pairwise tree (now lane-vectorized), piqe's
`var1`/segment stds are sequential f64 folds, and `resample_rows`' 8-tap
dot chains are sequential `fmaf` order per output (gather-vectorization
is not profitable on Zen 4). Scores on both test images are bit-identical
to the pre-change implementation (niqe `3.7578777869343147`,
`2.8298792112369173`; piqe `27.821832123338076`, `31.71664026438561`).
