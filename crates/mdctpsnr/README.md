# mdctpsnr

Pure-Rust CPU reimplementation of **mDCT-PSNR** — the masked-DCT perceptual
image quality metric by Thomas Richter (University of Stuttgart), "An
Autoregressive Multi-DCT Domain Image Quality Metric" (QoMEX 2009 lineage;
the metric the JPEG AIC-4 benchmark publishes in its `mDCT-PSNR` column).

The port follows the author's reference implementation
[`thorfdbg/mDCTpsnr`](https://github.com/thorfdbg/mDCTpsnr) (the `dctpsnr`
CLI), which is zlib-licensed; its notice is retained at the head of
[`src/lib.rs`](src/lib.rs). This crate is an altered reimplementation and is
not the original work.

## Algorithm

1. sRGB-8 → sRGB-linear RGB (exact EOTF LUT) → **linear** BT.601 YCbCr;
   only Y is re-encoded with the sRGB gamma knee, Cb/Cr stay linear
   (signed, ±0.5).
2. Per component, a sliding 8×8 windowed DCT at every position: each block
   is windowed toward its own mean (`b·w + avg·(1−w)`, Hamming-family
   `window_table`) then transformed with a two-pass scaled AAN float DCT.
3. Per (band, component) a 5×5 separable visibility mask:
   `mapped = |c·vis|^expon`, cosine-normalized kernel,
   `mask = 0.08 / (0.08 + Σ5×5/25)`.
4. Per measured position, `errorline[j]` multiplies the per-band
   non-detection probabilities `exp(−(err·max(mask)·vb)^3.5)` over 195
   terms (64 bands × 3 components; the Y DC band is measured on a 4-way
   5×5 low/high extended filter split instead of its own coefficient).
5. `score_dB = −20·log10(error / (w·h·comps))` with `w = W−13`,
   `h = H−13`, `comps = 3` — note the denominator's `H−13` vs the `H−12`
   measured lines; that is the reference's own normalization, replicated
   deliberately.

`+inf` on identical inputs, as the reference computes it. Minimum size is
14×14 (`ImageTooSmall` below that; the reference has unchecked UB there).

## API

```rust
let score = mdctpsnr::mdct_psnr_srgb8(&ref_rgb8, &dist_rgb8, w, h)?; // f32 dB
let jnd   = mdctpsnr::mdct_psnr_jnd(score);                        // JND map
```

- Input is packed sRGB8 (`w*h*3` bytes). Feature `std` on by default;
  `no_std` + `alloc` supported.
- `mdct_psnr_jnd` is the reference CLI's optional `-jnd` remap
  (`0.000475997802·(80−dB)^2.3182`, clamped at 0) — the AIC-4 column is the
  plain dB score.

## Parity with the compiled reference

The parity target is the **compiled binary**, not the source's apparent
semantics: built with GCC `-O3 -ffast-math` on x86_64 + glibc, the reference
reassociates reductions, contracts multiplies to FMAs, replaces the mask
division with `vrcpps` + a Newton step, and routes `powf`/`expf`/`cosf`
through libmvec's vectorized entry points (`_ZGVdN8vv_powf`,
`_ZGVdN8v_expf`). On x86_64 GNU/Linux this crate reproduces that codegen —
including calling the real libmvec symbols — so scores match to ~4e-6 dB
(bit-exact errorline on verified blocks/lines). Elsewhere the same
algorithm runs on scalar libm; identical algorithm, ~1ulp/step drift.

Validation (see `DIVERGENCES.md` for the full record):

- Synthetic goldens in the test suite, including all eight `(W−13) % 8`
  pooling-width residue classes (the vectorized reduction's tail guard
  `rem ≥ 4` is the subtle one — `rem == 3` never reads a slack lane).
- AIC-4 `mDCT-PSNR` column over the 53-pair stratified subset:
  median |Δ| 4.0e-7 dB, max 1.13e-5 dB.

## Not the same name, not the same metric

`mDCT-PSNR` (this crate — Richter's masked-DCT PSNR) is unrelated to the
AIC-4 `proposal-mDCTPSNR` submission column, which is a private contest
entry and cannot be reproduced by design.

## License

AGPL-3.0-only OR LicenseRef-Imazen-Commercial (crate). The ported algorithm
and the retained notice belong to the zlib-licensed reference; the notice
text is at the top of `src/lib.rs`.
