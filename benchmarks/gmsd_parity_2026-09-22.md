# GMSD port parity vs libgmsd — 2026-09-22

The `gmsd` crate (`crates/gmsd`) against its C reference, **libgmsd**
(<https://github.com/clunietp/libgmsd>, commit
`de646c9a957a892e4b9d78ff94601b83b363c158`, MIT, by Tom Clunie). The reference
was built only as a differential oracle, outside this repository
(`~/tmp/devin/gmsd-ref/`), and is not committed or used by anything else.

## Result

| quantity | measured |
|---|---|
| pairs | **64** real image pairs, 26×64 … 1024×768 (after even crop) |
| GMS map | **64/64 bit-identical** to libgmsd (max \|Δ\| = 0) |
| GMSD score | max \|Δ\| = **9.5e-15**, max relative = **4.0e-13** |
| score range covered | 0.00105 … 0.1289 |
| identity pair | exactly 0 (unit test `identity_is_exactly_zero`, 9 sizes) |

Per-pair values: [`gmsd_parity_2026-09-22.tsv`](gmsd_parity_2026-09-22.tsv).

**Tolerance stated in advance:** the map is expected bit-identical, because
the port reproduces libgmsd's float evaluation order term for term with no
FMA (`src/kernel.rs` header); the score is expected within f64 rounding
(≤ 1e-12 relative), because the port pools in one pass with a shifted f64
accumulator where libgmsd runs a two-pass mean/variance. Both held.

**Re-checked after `c59cb81e`** (SIMD sRGB8→gray fused into the bands): same
64 pairs, maps still 64/64 bit-identical, score deltas unchanged; the
zensim-lane GMSD scores on 3,561 KADID/KonFiG pairs re-computed bit-identical
to the pre-change scalar conversion.

## Why the score is not bit-identical

libgmsd computes `mean = Σq / n` then `Σ (q − mean)² / (n − 1)` in f64,
summing sequentially. The port accumulates `e = 1 − q` and `e²` per row in
four f64 SIMD lanes, adds rows in order, and forms
`(Σe² − Σe·mean_e) / (n − 1)`. Same quantity, different summation order: the
observed 4e-13 relative is that order difference, not a formula difference.
The port's result is identical at every thread count (row sums are combined
in row order regardless of banding).

## Inputs

64 pairs drawn with seed 20260922 from the joint-core-v2 **TRAIN** list
(`/mnt/v/output/zensim/zgeom-2026-09-21/extract/z2_train.tsv`), one per
reference, spread over the 14 distortion directories (zenjpeg, zenavif,
zenjxl, zenwebp, aom, libjxl, mozjpeg, Cloudinary AVIF/HEIC/JP2/WebP renders,
safesyn PNGs, …). Decoded through the zenmetrics CLI's zen-codec path
(`decode::decode_image_to_rgb8`), converted with `gmsd::rgb8_to_gray`
(libgmsd's CLI luma, `round(0.299R + 0.587G + 0.114B)`), cropped to even
width/height, and fed to both implementations as the same raw f32 planes.

Even crop is required, not a convenience: libgmsd's `downsample_2x2` sizes
its output `rows/2 × cols/2` but its loop writes one more row/column on odd
input — a heap overflow. The port uses the documented `⌊w/2⌋ × ⌊h/2⌋` grid on
odd input (dropping the trailing row/column) and is therefore not
parity-comparable there by construction.

## Constants

libgmsd and the authors' `GMSD.m` use `c = 170` on 0..255 intensities; the
paper states `c = 0.0026` for 0..1 intensities (`0.0026 · 255² = 169.065`).
The port follows the reference implementations (170). Prewitt scale is
libgmsd's `(float)(1/3)`.

## Vectorisation check (disassembly)

`objdump -d` of the release `zenmetrics` binary, symbol
`gmsd::kernel::__arcane_gmsd_band_v3` (the AVX2 tier's `#[arcane]` entry):
the per-row GMS helper is inlined — the loop body is 8-wide `ymm`
arithmetic with 2× `vsqrtps`, 1× `vdivps`, 2× `vcvtps2pd` and `vaddpd`/
`vmulpd` accumulation, and **no call instruction inside the loop**. The
decimation helper `downsample_row_v3` stays an out-of-line call **once per
row** (vectorised with `ymm` shuffles internally). The only other calls in
the entry are the ring-buffer allocations and edge-row `memset`s. A first
build also carried six per-iteration bounds-check branches from the
ten-wide window slicing; they were hoisted by re-slicing each padded row to
its exact length (`src/kernel.rs::gms_row`). After `c59cb81e` the sRGB8→gray row helper `gray_row_v3` is inlined into
the same entry (no symbol of its own; the entry's `ymm` count rose to 702)
and the standalone plane converter `__arcane_gray_plane_v3` is 4-wide f64
(`vcvtdq2pd` → `vmulpd`/`vaddpd` → `vroundpd` → `vcvtpd2ps`).

## Reproduce

```
# oracle (outside the repo)
git clone https://github.com/clunietp/libgmsd && cd libgmsd && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && make   # gcc 15.2, -O3
gcc -O3 -std=gnu99 -I../include gmsd_raw.c liblibgmsd.a -lm -o gmsd_raw         # raw-plane driver
# Rust half
cargo build --release -p zenmetrics-cli --example gmsd_parity_dump
gmsd_parity_dump pairs.tsv run/          # writes <i>.ref.f32 / .dist.f32 / .rust_map.f32 + rust.tsv
for each i: gmsd_raw run/<i>.ref.f32 run/<i>.dist.f32 W H run/<i>.c_map.f32
```

The driver `gmsd_raw.c` (16 lines: read two raw f32 planes, call `gmsd()`,
write the map, print the score) and the comparison script live with the lane
scratch (`~/tmp/devin/gmsd-ref/driver/`, `~/tmp/devin/gmsd_parity_compare.py`).
Binary hashes: `gmsd_raw` `a85f97c9…`, `liblibgmsd.a` `e5ac455b…`,
`gmsd_parity_dump` `5e133608…`. libgmsd's own test suite passes on this build
(5/5 against its MATLAB-derived expected values at 4 decimals).
