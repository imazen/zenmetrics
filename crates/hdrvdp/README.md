# hdrvdp

A pure-Rust `f64` CPU reference port of **HDR-VDP-2.2** (Mantiuk, Kim, Rempel &
Heidrich, SIGGRAPH 2011; quality recalibrated by Narwaria et al., JEI 2015).

HDR-VDP-2 predicts two things for a pair of images given in **absolute
luminance**: *visibility* — the per-pixel probability that a human notices a
difference — and *quality* — a single MOS correlate. It is the field's reference
HDR metric, and the bar every HDR feeding experiment in this repo has been
measured against:

| dataset | HDR-VDP-2 | best in this workspace |
|---|--:|--:|
| AIC-HDR2025 (HDR compression JND) | **0.936** | SSIMULACRA2 0.906 |
| UPIQ (380 HDR compression pairs, JOD) | **0.812** | cvvdp faithful 0.758 |

## Status — chunk 1 of 6, **not yet a metric**

This crate lands in chunks (tracked in [imazen/zenmetrics#50]). Today it holds
the **front of the visual pathway**, complete and unit-tested (52 tests):

- `display` — the five colour encodings → cd/m² (`luminance`, `luma-display`,
  `srgb-display`, `rgb-bt.709`, `xyz`), plus the "you passed relative values"
  detector.
- `spectral` — Smith–Pokorny cone fundamentals, CIE scotopic V′, display
  emission spectra, and the `channels × LMSR` mixing matrix.
- `csf` — the eye's optical MTF, the neural CSF, and the rod / cone
  contrast-versus-intensity curves.
- `photoreceptor` — the luminance → JND-space response, built by integrating
  sensitivity so that **equal distance means equal discriminability**.
- `fft` — a dependency-free complex FFT (radix-2 + Bluestein), the radial
  cycles-per-degree grid, and zero-phase convolution with post-padding.
- `pathway` — stages 1–5 assembled: MTF → LMSR → adapting luminance →
  photoreceptor non-linearity → separate cone/rod DC removal → achromatic
  response.

**Still to come:** the steerable-pyramid decomposition and contrast masking
(chunk 2), probability pooling and the `Q` / `Q_MOS` quality correlate (chunk 3),
end-to-end scoring and UPIQ validation (chunk 4), umbrella wiring (chunk 5), and
a CubeCL GPU port gated against this `f64` reference (chunk 6).

**Do not report any number out of this crate as an HDR-VDP-2 score** until chunk
4 lands and its measured UPIQ SROCC is recorded in `benchmarks/`.

## Units

Everything is **absolute luminance in cd/m² (nits)** — the same currency
`zenmetrics_api::hdr` speaks. Feeding relative or normalised values scores at the
wrong adaptation level and silently produces wrong answers;
`display::looks_relative` catches the common form of that mistake.

## Licensing

Ported from the **ISC-licensed** HDR-VDP-2.2 MATLAB release. Three upstream
helpers carry no permission grant ("internal use, do not redistribute") and are
therefore **not** ported — they are reimplemented from their descriptions in
`fft.rs` and `spectral.rs`, with the one knowable behavioural difference
documented at the site. The steerable pyramid (chunk 2) comes from the MIT
`matlabPyrTools`. Full detail, including the notices that must travel with this
code: [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).

[imazen/zenmetrics#50]: https://github.com/imazen/zenmetrics/issues/50
