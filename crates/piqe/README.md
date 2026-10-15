# piqe

PIQE — Perception-based Image Quality Evaluator — as a pure-Rust, CPU,
no-reference metric for zenmetrics. Score is in `[0, 100]`; **higher is
worse** (`ScoreKind::Distortion`).

This is a port of [pypiqe](https://github.com/michael-rutherford/pypiqe) 1.2
(MIT, Copyright (c) 2023 Michael Rutherford — notice kept in
`LICENSE-pypiqe`), a Python conversion of MATLAB's `piqe` that the author
tested against the MATLAB implementation. There are no learned parameters.

## Algorithm

1. Convert to gray: OpenCV `COLOR_BGR2GRAY`, i.e. the exact 15-bit
   fixed-point `Y = (3735·B + 19235·G + 9798·R + 16384) >> 15`.
2. Pad the image bottom/right with `numpy` `symmetric` (edge-inclusive
   mirror) to a multiple of 16; rescale to `[0, 255]` via
   `round(255·v/max)` (ties to even).
3. Normalize locally: `imnorm = (im − mu)/(sigma + 1)` where `mu`, `sigma`
   are the 7-tap Gaussian (σ = 7/6, replicate borders) mean/std in f64.
4. For each 16×16 block of `imnorm`: skip unless `var > 0.1` (spatially
   active). Active blocks add `impaired·(1 − var) + noisy·var` to the
   distortion score, where *impaired* = any of the 44 overlapping 6-pixel
   edge segments has std < 0.1, and *noisy* = `blockSigma > 2·blockBeta`
   from the center/surround deviation ratio.
5. `score = (distortion + 1)/(NHSA + 1)·100`; a uniform image scores 100.

Deliberate divergence from the oracle: pypiqe scores inputs smaller than
one block by symmetric-padding them up; this port returns
`ErrorKind::TooSmall` below 16×16 instead.

## Usage

```rust
use piqe::Piqe;
use zenmetrics_abstractions::NoReferenceMetric;

let mut m = Piqe::new();
// m.evaluate_image(image, opts) -> ScalarEval { "piqe": score }
```

The CLI exposes it as `--metric piqe` (feature `nr`) through
`PairedNoReference`, so a `batch`/`score` run reports `piqe` for the
distorted image and `piqe_ref` for the reference.

## Paper

> Venkatanath N., Praneeth D., Maruthi Chandrasekhar Bh., Channappayya S. S.,
> Medasani S. S., "Blind Image Quality Evaluation Using Perception Based
> Features", *2015 Twenty First National Conference on Communications
> (NCC)*, pp. 1–6. doi:10.1109/NCC.2015.7084843

## Parity

`tests/golden_piqe.py` regenerates `tests/goldens_piqe.tsv` (ladder PNGs +
64 KonJND images) with `pypiqe==1.2` on `opencv-python`; `tests/parity.rs`
scores the same images and asserts each golden is within `JND/1000`, printing
the worst |Δ| in absolute and JND units.
