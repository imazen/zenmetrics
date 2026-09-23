# cvvdp ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenmetrics/cvvdp.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/cvvdp?style=flat-square) ![lib.rs](https://img.shields.io/crates/v/cvvdp?style=flat-square&label=lib.rs&color=blue) ![docs.rs](https://img.shields.io/docsrs/cvvdp?style=flat-square) ![License](https://img.shields.io/crates/l/cvvdp?style=flat-square)

Pure-Rust CPU port of [ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP)
(still-image and video scoring). Built on top of the canonical pycvvdp v0.5.7
algorithm (still-image path identical to v0.5.4), designed as a drop-in perceptual metric for the JPEG XL
encoder's iterative quantization loop where the GPU backend's
host-to-device upload latency exceeds CPU compute time.

Companion to [`cvvdp-gpu`](../cvvdp-gpu/) — both crates produce
scalar JOD values within `≤ 1e-3` of each other and of the pycvvdp
v0.5.7 reference.

## What it does

```rust
use cvvdp::{Cvvdp, CvvdpParams};

let (w, h) = (256, 256);
let ref_srgb: Vec<u8> = vec![/* w*h*3 bytes */];
let dist_srgb: Vec<u8> = vec![/* w*h*3 bytes */];

let mut scorer = Cvvdp::new(w as u32, h as u32, CvvdpParams::default())?;
let jod: f32 = scorer.score(&ref_srgb, &dist_srgb)?;
// JOD ∈ [0, 10]; 10 = imperceptible difference.
```

Buttloop-style hot path (same reference, many distorted candidates):

```rust
scorer.warm_reference(&ref_srgb)?;
for candidate in candidates {
    let jod = scorer.score_with_warm_ref(&candidate)?;
    // ...
}
```

Per-pixel diffmap for spatial error localization (JPEG XL
quantization refinement):

```rust
let mut diffmap = Vec::new();
let jod = scorer.score_with_diffmap(&ref_srgb, &dist_srgb, &mut diffmap)?;
// diffmap.len() == w * h, row-major, contiguous.
// Non-negative; zero where ref == dist; concentrates spatially
// on the distorted region.
```

Video scoring (streaming; holds only the temporal-filter window of
frames, not the whole clip):

```rust
use cvvdp::{DisplayGeometry, VideoScorer};

let mut v = VideoScorer::new(1920, 1080, 30.0, CvvdpParams::default(),
                           DisplayGeometry::STANDARD_FHD)?;
for (ref_frame, dist_frame) in ref_frames.iter().zip(dist_frames.iter()) {
    v.push_frame(ref_frame, dist_frame)?;   // &[u8] sRGB-8, w*h*3 each
}
let jod = v.finish()?;
```

A whole-clip convenience `cvvdp::score_video(&ref_frames,
&dist_frames, w, h, fps, params, geometry)` wraps the same
push/finish loop. `cvvdp::score_video_with_stats` additionally
accepts a `FrameLayout` (`Interleaved`/`Planar`, the `dim_order`
analog) and returns `VideoStats` — the `(Q_jod, stats)` pair
pycvvdp's `predict` returns, including `loss()` = `10 − JOD`.
`VideoScorer::with_options` takes `VideoScorerOptions`
(`layout`, `temp_padding`, `low_memory` — a source-sample ring window
that cuts peak RSS by ~⅓ at 1080p for a few percent CPU; scores
bit-identical).

Stills and video accept three display-encoded sample types, matching
pycvvdp's `video_source_array` dtypes: `u8` (`v/255`), `u16`
(`v/65535` — `score_u16`, `push_frame_u16`, `score_video_u16`) and
`f32` (`[0,1]` as-is — `score_f32`, `push_frame_f32`,
`score_video_f32`). Under an HDR display model the u16/f32 path is
genuine nit-domain scoring, not an 8-bit upscale — verified on a
committed real HDR10/PQ clip (`video_parity` u16 cells: max
|Δ| = 2e-6 JOD video, 1.2e-5 stills vs pycvvdp v0.5.7). Mixing
sample types within one scorer errors (`Error::MixedSampleTypes`).

A 1-frame clip routes through the still path, bit-identical to
`Cvvdp::score`. See `docs/VIDEO.md` for the port
design, measured parity, and benchmarks — including the
fast-ssim2-per-frame comparison (wall, user+sys CPU, peak RSS, 1t vs
8t; ~230 ms/frame at 1080p/8t — faster than ssim2 per-frame —
~1.0 GB peak RSS, ~0.7 GB with `low_memory`; the streaming bound is
the temporal-filter window, not a small footprint).

## Scope

- Still-image and video scoring (temporal channels, transient
  achromatic channel, causal `replicate` padding — pycvvdp's
  defaults).
- DKLd65 opponent + Weber-contrast pyramid + castleCSF + mult-mutual
  masking + Minkowski pooling (spatial β=2, per-band/channel β=4,
  frames β=2 for video).
- Matches pycvvdp v0.5.7 within `≤ 1e-3 JOD`: stills on synthetic
  fixtures 16²–512² (`tests/parity_against_host_scalar.rs`); video on
  44 cells (11 situations × 4 displays) at 24/30/60 fps
  (`cvvdp-conformance::video_parity`, measured max |Δ| = 3e-6 JOD).

## Why a CPU port

`cvvdp-gpu` running on CUDA is ~14 ns/px on an RTX 5070. Per-iteration
host-to-device upload at 12 MP is `~12 MB × 2` ≈ 24 MB, which on
PCIe 4.0 at ~20 GB/s is ~1.2 ms even pinned. For JPEG XL's buttloop
at 1024×1024, the GPU compute time is ~30 ms; CPU compute time is
roughly comparable on a 7950X. When compute and upload are at parity
the CPU path wins by eliminating the upload entirely (and the cold
cubecl JIT compile on first call).

This crate is the CPU twin so we can drop the GPU dependency in
contexts (CI, container deploys, embedded) where shipping a CUDA
runtime is impractical.

## Features

| flag         | default | effect |
|--------------|---------|--------|
| `std`        | on      | enable `std`-dependent paths |
| `alloc`      | on      | use the `alloc` crate (always required) |
| `parallel`   | on      | per-band rayon parallelism (requires `std`) |
| `pixels`     | off     | `zenpixels::PixelSlice` integration |

`no_std + alloc` builds work; `parallel` implies `std`.

## License

Dual-licensed under either:

- AGPL-3.0-or-later (`LICENSE-AGPL3` at the workspace root)
- A commercial license (contact <support@imazen.io>)

## Parity reference

Pinned against [`pycvvdp v0.5.7`](https://github.com/gfxdisp/ColorVideoVDP/releases/tag/v0.5.7)
(v0.5.4 until 2026-09-23; the still-image code path is numerically identical).
The pinned reference version constant is re-exported as
`cvvdp::PYCVVDP_REFERENCE_VERSION`.
