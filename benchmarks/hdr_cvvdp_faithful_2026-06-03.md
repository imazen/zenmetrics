# Faithful HDR metrics — linear-planes paths (chunk 4: cvvdp + butteraugli)

**Date:** 2026-06-03 · **host:** lilith (RTX 5070, CUDA/WSL) · **branch:** `feat/hdr-metrics`
**Binary:** `cargo build --release -p zen-metrics-cli --no-default-features --features "png,jpeg,cpu-metrics,gpu-cvvdp,gpu-butteraugli,gpu-cuda,hdr"`

## The bottleneck this removes

Chunks 2–3 score HDR by PU21-encoding luminance → **u8**. PU21 maps 100 cd/m² →
~256, so any highlight above ~80 cd/m² clamps at 255 — the SDR metrics go **blind
to highlight differences** (the user-named "u8 PU highlight-clamp bottleneck").

cvvdp is the metric designed for HDR (display model + CSF + masking on absolute
luminance). It already exposes a native **linear-planes** entry that takes f32
RGB directly — no u8 round-trip. Chunk 4a routes HDR cvvdp through it.

## What changed (Layer-3 only — no kernel changes)

The pure-Rust `cvvdp` crate and `cvvdp-gpu` already have everything: `Eotf::Linear`,
`STANDARD_HDR_LINEAR`, BT.2020/P3 primaries, and `Cvvdp::score_from_linear_planes`
(6 tight `W×H` f32 planes, "unit-scaled" display-relative `[0,1]`; the GPU applies
`y_peak`/DKL). So chunk 4a is plumbing in the CLI:

- `hdr::to_cvvdp_linear_planes(&NitsImage) -> (r,g,b)` — split nits into
  display-relative `[0,1]` planes (`v = nits / HDR_DISPLAY_PEAK_NITS`, clamped).
- `DisplayTarget::hdr(peak)` — a `peak` cd/m² display, `Eotf::Linear`, BT.709
  primaries (same color treatment as the sRGB SDR path → the HDR win is isolated
  to **luminance range**; wide-gamut primaries are a follow-up), `STANDARD_4K`
  geometry.
- `CvvdpBatchScorer::score_from_linear_planes(...)` — mirrors the cached sRGB
  scorer but feeds the planes. GPU runtimes only (cubecl-cpu lacks the atomic
  pool kernel; it errors cleanly).
- `score --hdr` and `batch --hdr` route cvvdp through this path when `gpu-cvvdp`
  is built; otherwise cvvdp falls back to the sRGB8 path (which errors without a
  GPU, as SDR cvvdp already does).

`HDR_DISPLAY_PEAK_NITS = 1000` (common HDR mastering target). Content brighter
than the peak clips, as a real 1000-nit display would.

## Verified — sane, discriminating JOD (10 = imperceptible)

| pair | `score --hdr` | `batch --hdr` |
|---|---|---|
| UltraHDR-JPEG identity | 10.000000 | 10.000000 |
| UltraHDR-JPEG vs q15 re-encode | 7.143168 | 7.143168 |
| HEIC vs its own corpus EXR (cross-path) | 10.000000 | 10.000000 |
| EXR different content (32F76D88 vs 94067DD9) | 3.508498 | — |

score and batch agree to the digit (same scorer path).

## Value proof — faithful path sees what u8-PU cannot

A synthetic pair (`examples/synth_highlight_pair`) differing ONLY in a highlight
patch — **2000 cd/m² (ref) vs 200 cd/m² (dist)**, identical 80 cd/m² background:

| metric | score | verdict |
|---|---|---|
| **cvvdp (faithful HDR linear planes)** | JOD **4.43** | **strongly detects** the 10× highlight crush |
| ssim2 (u8 PU-clamp) | **100.0** | **completely blind** — both patches PU-clamp to 255 |
| butteraugli (u8 PU-clamp) | **0.0 / 0.0** | **completely blind** |

Identity sanity: cvvdp 10.0, ssim2 100.0 (both correct). This is the bottleneck
made concrete: above the u8 clamp, the SDR-PU path reports "identical" for a
visibly different highlight; the faithful cvvdp path scores it at JOD 4.43.

## Chunk 4b — faithful butteraugli HDR

butteraugli also has the foundation: `butteraugli-gpu`'s `internals` linear-planes
API (`set_reference_from_linear_planes` / `compute_with_reference_from_linear_planes`)
skips the sRGB→linear LUT + the u8 pack, and its opsin stage scales linear input by
`intensity_target` (cd/m² that maps to plane-value 1.0). So feeding the same
display-relative `[0,1]` planes (`nits / peak`) with `intensity_target = peak` gives
butteraugli the full HDR range with **zero kernel changes**.

Wiring is CLI-only: enabling `butteraugli-gpu/internals` via cargo feature
unification (a direct dep behind `gpu-butteraugli`) exposes the methods on the
re-exported `zenmetrics_api::butter::Butteraugli`; `butter_pnorm3::score_both_from_
linear_planes` uploads the 6 f32 planes → handles → set-ref + compute-with-ref.
`score --metric butteraugli-gpu --hdr` and `batch` route through it; CPU
`butteraugli --hdr` stays on the u8-PU path.

Verified (intensity_target = 1000): identity 0.0/0.0, q15 51.2/30.2, EXR-different
319.6/136.8, score==batch. **Highlight A/B** (2000 vs 200 cd/m²): faithful
butteraugli-gpu **205.9 / 90.6** (strongly detects) vs u8-PU butteraugli **0.0 / 0.0**
(blind) — same bottleneck, same resolution as cvvdp.

So **two** metrics now have faithful HDR (cvvdp + butteraugli-gpu), both GPU.

## Scope / follow-ups

- **ssim2 / dssim / iwssim faithful HDR** — unlike cvvdp/butteraugli (which have
  native HDR-luminance handling), these have no linear-light input contract that
  cleanly absorbs HDR. Feeding PU values would mis-transform through their own
  sRGB→working-space color science. Would need a per-metric linear-f32 input path
  (real kernel work) — not pursued.
- **Wide-gamut primaries** — `DisplayTarget::hdr` uses BT.709 to isolate the
  luminance win; our HDR sources are P3. Tracking source primaries (P3/BT.2020)
  into the DKL/opsin matrices is a clean follow-up (chroma-channel refinement).
- **Per-pair / configurable display peak** — fixed 1000 cd/m²; a `--hdr-peak` /
  `--display-model` knob + per-pair content-peak option is a follow-up.

GPU smoke harness: `scripts/hdr/hdr_cvvdp_faithful_smoke.sh` (10 assertions, needs a
`gpu-cvvdp,gpu-butteraugli` build + a CUDA GPU; the CPU `hdr_metric_smoke.sh` covers
the u8 paths).
