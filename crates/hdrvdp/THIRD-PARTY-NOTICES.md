# Third-party notices — `hdrvdp`

This crate is a Rust port of HDR-VDP-2.2. It carries three provenances.

## 1. HDR-VDP-2 core (ISC) — ported

The metric's equations and calibration constants are ported from the MATLAB
`hdrvdp-2.2.x` release. The following upstream files carry the notice reproduced
below, and the Rust modules derived from them reproduce it in their headers:

`hdrvdp.m`, `hdrvdp_visual_pathway.m`, `hdrvdp_ncsf.m`, `hdrvdp_mtf.m`,
`hdrvdp_joint_rod_cone_sens.m`, `hdrvdp_rod_sens.m`, `hdrvdp_parse_options.m`,
`hdrvdp_pix_per_deg.m`.

> Copyright (c) 2011, Rafal Mantiuk <mantiuk@gmail.com>
>
> Permission to use, copy, modify, and/or distribute this software for any
> purpose with or without fee is hereby granted, provided that the above
> copyright notice and this permission notice appear in all copies.
>
> THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
> WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
> MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
> ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
> WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
> ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
> OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

Cite the metric as: Rafał Mantiuk, Kil Joong Kim, Allan G. Rempel, Wolfgang
Heidrich, *"HDR-VDP-2: A calibrated visual metric for visibility and quality
predictions in all luminance conditions"*, ACM TOG 30(4), SIGGRAPH 2011; and for
the 2.2 quality calibration, Manish Narwaria, Rafał Mantiuk, Mattheiu Perreira Da
Silva, Patrick Le Callet, *"HDR-VDP-2.2: A calibrated method for objective
quality prediction of high dynamic range and standard images"*, J. Electronic
Imaging 24(1), 2015.

## 2. Three upstream helpers that are NOT under that grant — NOT ported

`create_cycdeg_image.m`, `fast_conv_fft.m` and `load_spectral_resp.m` carry
`"(C) Rafal Mantiuk. This is an experimental code for internal use. Do not
redistribute."` — no permission grant.

**No line of those three files is reproduced here.** Their jobs (a radial
cycles-per-degree grid over the DFT lattice; convolution in the Fourier domain
with post-padding; loading a comma-separated spectral table and resampling it)
are standard operations, and `src/fft.rs` / `src/spectral.rs` implement them
independently from that one-line description. Where the independent
implementation is knowably *different* from upstream's, the difference is
documented at the site (see `fft::cycles_per_degree_grid`).

## 3. Steerable pyramid (MIT) — to be ported in chunk 2

The multi-scale decomposition is Simoncelli & Freeman's steerable pyramid, whose
`sp3Filters` taps come from `matlabPyrTools`
(<https://github.com/LabForComputationalVision/matlabPyrTools>, MIT). Cite:
E. P. Simoncelli & W. T. Freeman, *"The Steerable Pyramid: A Flexible
Architecture for Multi-Scale Derivative Computation"*, ICIP 1995; and
A. Karasaridis & E. P. Simoncelli, *"A Filter Design Technique for Steerable
Pyramid Image Transforms"*, ICASSP 1996.

## 4. Spectral data in `data/`

- `log_cone_smith_pokorny_1975.csv` — Smith & Pokorny (1975) cone fundamentals,
  log₁₀ sensitivity, as distributed with the ISC-licensed hdrvdp release.
- `cie_scotopic_lum.txt` — CIE scotopic luminous efficiency V′(λ).
- `d65.csv` — CIE standard illuminant D65 relative spectral power.
- `emission_spectra_ccfl-lcd.csv` — measured emission spectra of a CCFL-backlit
  LCD, as distributed with the hdrvdp release (the upstream default display).

The first three are published colorimetric standards; all four are redistributed
here under the ISC grant above.
