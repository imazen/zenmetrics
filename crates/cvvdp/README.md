# cvvdp ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenmetrics/cvvdp.yml?style=flat-square&label=CI) ![crates.io](https://img.shields.io/crates/v/cvvdp?style=flat-square) ![lib.rs](https://img.shields.io/crates/v/cvvdp?style=flat-square&label=lib.rs&color=blue) ![docs.rs](https://img.shields.io/docsrs/cvvdp?style=flat-square) ![License](https://img.shields.io/crates/l/cvvdp?style=flat-square)

Pure-Rust CPU port of [ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP)
(still-image scoring). Built on top of the canonical pycvvdp v0.5.4
algorithm, designed as a drop-in perceptual metric for the JPEG XL
encoder's iterative quantization loop where the GPU backend's
host-to-device upload latency exceeds CPU compute time.

Companion to [`cvvdp-gpu`](../cvvdp-gpu/) — both crates produce
scalar JOD values within `≤ 1e-3` of each other and of the pycvvdp
v0.5.4 reference.

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

## Scope

- Still-image scoring (no temporal channels).
- DKLd65 opponent + Weber-contrast pyramid + castleCSF + mult-mutual
  masking + 3-stage Minkowski pool.
- Matches pycvvdp v0.5.4 within `≤ 1e-3 JOD` on synthetic fixtures
  16² through 512² (gated by `tests/parity_against_host_scalar.rs`).

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

Pinned against [`pycvvdp v0.5.4`](https://github.com/gfxdisp/ColorVideoVDP/releases/tag/v0.5.4).
The pinned reference version constant is re-exported as
`cvvdp::PYCVVDP_REFERENCE_VERSION`.
