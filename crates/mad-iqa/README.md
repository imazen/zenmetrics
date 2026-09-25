# mad-iqa

Pure-Rust CPU port of **MAD — Most Apparent Distortion** (E. C. Larson
& D. M. Chandler, *Most apparent distortion: full-reference image
quality assessment and the role of strategy*, JEI 19(1), 2010).

MAD is a full-reference **distance** metric: `0` = identical, higher =
more distorted, unbounded above. It models two HVS strategies and
blends them geometrically:

- **HI** (`hi_index`) — *visible distortion / appearance masking*:
  luminance transform `k·x^(2.2/3)`, Mannos–Sakrison CSF weighting in
  the DFT domain, blocky local statistics (the release's `ical_std`
  C-mex: stride-4 16×16 windows replicated into 4×4 tiles, plus an
  8×8 min-pooled reference std), a contrast-threshold detection mask,
  and a 16×16 `imfilter`-`'same'` local MSE. Result ×10.
- **LO** (`lo_index`) — *near-threshold statistical deviation*: a
  5-scale × 4-orientation Kovesi log-Gabor bank (minWaveLength 3,
  mult 3, sigmaOnf 0.55, dThetaOnSigma 1.5), per-subband `ical_stat`
  maps (std / skewness / kurtosis), weighted
  `|Δstd| + 2|Δskw| + |Δkrt|` with scale weights
  `[0.5 0.75 1 5 6]/13.25`.
- Combine (JEI paper): `sig = 1/(1 + b1·HI^b2)`,
  `MAD = HI^sig · LO^(1−sig)`, `b1 = exp(−2.55/3.35)`,
  `b2 = 1/(ln10·3.35)`.

## API

```rust
use mad_iqa::{mad_plane_f32, mad_rgb8, MadScore};

// f32 luma planes, 0–255 scale, explicit stride.
let s: MadScore = mad_plane_f32(&reference, &distorted, w, h, stride)?;
// sRGB RGB8 packed rows (MATLAB rgb2gray luma: 0.2989/0.5870/0.1140,
// unrounded).
let s = mad_rgb8(&ref_rgb, &dst_rgb, w, h, 3 * w)?;
// s.hi, s.lo — the two strategy indices; s.mad — the blend.
```

Images with `min(w,h) < 34` leave no samples after the reference's
`mp(17:end-17)` edge kill and score `NaN`, as the reference does.
Identical and constant inputs score exactly `0.0`.

## Fidelity

Validated against the authors' MATLAB release (`hi_index.m` /
`lo_index.m`, distributed in the STMAD_2011 package archived at
`Netflix/vmaf`) run under GNU Octave, with `ical_std.m` /
`ical_stat.m` shims ported verbatim from the release's C mex sources
(`mkoctfile` was unavailable — the shims reproduce the C semantics
exactly). 13 goldens in `src/tests.rs`: worst relative deltas
`hi` 2.4e-7, `lo` 1.8e-6, `mad` 1.3e-6. Regenerate with
`validation/gen_goldens.m` (see `validation/README.md` for
provenance and the uint64-type pitfall the generator works around).

## Floating-point discipline

`f32` planes end-to-end; one-time grids (CSF, log-Gabor) and all
block-stat numerators / final norms accumulate `f64`. Every SIMD tier
(scalar / v3 / v4-avx512 / neon / wasm128) and every `parallel`
thread count produces **bit-identical** output — per-element
expressions have fixed order independent of lane count.

## Features

- `std` (default) — off for `core`+`alloc` only.
- `parallel` — Rayon across the ref/dst work pairs (CSF filters,
  per-image Gabor banks, stat maps); bit-identical.
- `avx512` — opt-in `f32x16` tier (like the sibling crates).
- `_dev` — exposes per-tier entry points for parity tests /
  disassembly. Not API.

## License

`AGPL-3.0-only OR LicenseRef-Imazen-Commercial`, matching the
workspace. The reference MATLAB files are *not* redistributed here;
`validation/README.md` documents provenance.
