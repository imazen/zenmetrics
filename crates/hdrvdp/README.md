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

## Status — chunks 1–3 of 6: **it scores a pair, it is not yet validated**

Landed in chunks (tracked in [imazen/zenmetrics#50]). `hdrvdp::hdrvdp()` now takes
two images in absolute luminance and returns `Q_MOS` (0–100, 100 = best), the raw
`Q`, and the per-pixel visibility maps `P_map` / `C_map`. 100 tests (98 unit +
2 end-to-end on real corpus pixels) and **zero runtime dependencies** — the FFT,
interpolation and quadrature are in-crate, so i686 / wasm / windows-arm builds
carry nothing extra. The only dev-dependencies are an image decoder and the
shared corpus, used by the end-to-end test and example.

| stage | module |
|---|---|
| the five colour encodings → cd/m², plus the "you passed relative values" detector | `display` |
| optical MTF, neural CSF, rod/cone contrast-versus-intensity | `csf` + `fft` |
| cone fundamentals, scotopic V′, display emission, the LMSR mixing matrix | `spectral` |
| luminance → JND-space photoreceptor response | `photoreceptor` |
| MTF → LMSR → adapting luminance → non-linearity → cone/rod DC removal | `pathway` |
| `sp3` steerable pyramid (4 orientations), build + reconstruct | `spyr` |
| the `[1, 4×H, 1]` band list + base-band CSF filtering | `bands` |
| MATLAB-compatible `imresize` (bicubic, antialiased) | `resize` |
| mutual masking, the three masking terms, the transducer, per-band `D` | `masking` |
| `S_map` → `P_map` / `P_det` / `C_max`, and `Q` → `Q_MOS` | `pool` |
| the end-to-end entry point | `metric` |

**Still open:** UPIQ validation (chunk 4), umbrella wiring as
`MetricKind::Hdrvdp` (chunk 5), and a CubeCL port gated against this `f64`
reference (chunk 6).

**Do not publish any number from this crate as an HDR-VDP-2 score yet.** The
pipeline is complete and behaves like a luminance-aware metric — quality falls
monotonically along a distortion ladder, the visibility map localises, and the
same relative distortion is measurably less visible at ~0.04 cd/m² than at
~80 cd/m² — but "behaves correctly" is not "matches the reference". Chunk 4
measures UPIQ SROCC against the published **0.812** and records it in
`benchmarks/`; until that lands, treat the output as a work in progress.

### Two things to know before reading a number

- **After spatial pooling, `C_max` is a total, not a peak.** `S_map ·= sum/max`
  makes the new maximum equal the old sum, so a `C_max` of 400 does not mean
  "400× threshold". `P_map`'s spatial shape is unaffected.
- **An identical pair scores 99.9998, not 100** — every band's error is 0, so
  each quality term is `log(0 + 1e-12)`, and that epsilon is finite. Upstream
  behaviour, documented in its own 2.1.2 ChangeLog.

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
