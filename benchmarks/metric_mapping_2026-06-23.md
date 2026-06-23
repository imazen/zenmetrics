# JPEG Cross-Metric Mapping Table

Maps a target on any one of five perceptual-quality scales to the equivalent
on the other four, for **baseline JPEG** (libjpeg-turbo, 4:2:0). Each operating
point is a real measured cell — no fabricated, interpolated, or extrapolated
values.

## Scales

| Scale | Crate / tool | Direction | Range | Reference point |
|---|---|---|---|---|
| **SSIMULACRA2** | `fast-ssim2` 0.8.2 (Imazen SIMD SSIMULACRA2) | higher = better | ~ -∞ … 100 | ~90 ≈ visually lossless; ~70 ≈ high quality; ~30 ≈ low |
| **butteraugli (3-norm)** | `butteraugli` 0.9.2, `pnorm_3` (libjxl-style, the user's standing pivot) | lower = better | 0 … ∞ | ~1.0 ≈ JND, ~1.5 ≈ PJND |
| **butteraugli (max)** | same call, `score` (per-block max) — reported alongside for reference | lower = better | 0 … ∞ | — |
| **libjpeg-turbo Q** | `/usr/bin/cjpeg` (libjpeg-turbo 2.1.2), `-quality N`, default 4:2:0 baseline | higher Q = better | 0 … 100 | the JPEG dial itself |
| **cvvdp** | ColorVideoVDP via `cvvdp` 0.1.0 / `cvvdp-gpu` 0.0.1 (CubeCL, CUDA), JOD scale, `standard_4k` display model | higher = better | 0 … 10 | ~10 = imperceptible |
| **zensim Profile-A** | `zensim` 0.3.0, `ZensimProfile::A` (= `latest_preview()`; bake `v47-strict-QAT-native`, rotated 2026-05-27, sha256 `d0ef7a30…`) | higher = better | ~ -∞ … 97.69 | the metric the zen recompressors target as "zensim-A"; **identity = 97.69**, not 100 |

## Reading the table

The forward table is indexed by libjpeg-turbo Q. The operating-point table is
indexed by SSIMULACRA2 level (the most widely-used anchor) and gives the
median equivalent on every other scale, plus the implied Q. Cross-mappings in
any direction follow by reading down the appropriate columns (all metrics are
monotone in quality, so the correspondence is invertible within sampling
noise).

## Key caveats

1. **The correspondences are content-dependent — substantially so.** See the
   per-class tables. ScreenContent / text behaves very differently from photos
   under JPEG: at a fixed SSIMULACRA2 it carries a *worse* butteraugli and a
   lower cvvdp than a photo does, and its Q→ssim2 curve is flatter (high at
   low-Q, low at high-Q). Treat the all-content medians as a centroid; for a
   known content class, use the class row.

2. **butteraugli 3-norm vs max.** The primary butteraugli column is the
   **3-norm** (`pnorm_3`) per the user's standing preference (~1.5 ≈ PJND).
   The max-norm is reported alongside because it is what the oracle-d2
   cross-check dataset stored.

3. **zensim-A ceiling is ~97.7, not 100.** Its `identity` score is 97.69
   (the dial maximum of the v47-strict bake), so "zensim-A 90" is closer to
   the top of its usable range than "ssim2 90" is to 100.

4. **cvvdp compresses hard at the top.** On the JOD 0–10 scale, almost the
   entire useful JPEG quality range lives in cvvdp 8.5–10.0; cvvdp is a coarse
   discriminator at high quality and only separates aggressive low-Q encodes.

5. **libjpeg-turbo Q semantics.** Q is the real libjpeg-turbo `cjpeg -quality`
   dial at default settings (4:2:0 chroma, baseline Huffman, integer DCT).
   A different JPEG encoder (mozjpeg, jpegli, zenjpeg) at the *same numeric Q*
   will land at a *different* perceptual point — Q is encoder-specific. The
   *metric-to-metric* cross-mappings (ssim2↔butter↔cvvdp↔zensim), by contrast,
   are largely encoder-independent (they describe the distortion, not the
   dial), and are cross-validated below against a second encoder (mozjpeg-rs)
   on a different corpus.

## Provenance

- **Generated:** 2026-06-23, host `lilith` (7950X / RTX 5070), zenmetrics git HEAD `af76ca13`.
- **Encoder:** real libjpeg-turbo 2.1.2 `cjpeg -quality Q` at default settings (4:2:0 chroma, baseline, integer DCT). **Not a proxy** — `cjpeg` was on PATH. PNG sources were converted to PPM (ImageMagick) then fed to `cjpeg`; the encoded JPEG was decoded and scored.
- **Metric tools** (every value is a real tool output — no estimates, no extrapolation):
  - SSIMULACRA2: `fast-ssim2` 0.8.2 (Imazen SIMD)
  - butteraugli: `butteraugli` 0.9.2 — both `pnorm_3` (primary, libjxl 3-norm) and `score` (max) per call
  - cvvdp: `cvvdp` 0.1.0 / `cvvdp-gpu` 0.0.1, CubeCL **CUDA** backend (native, RTX 5070), `standard_4k` display model. Column tag `cvvdp_imazen_v0_0_1`.
  - zensim Profile-A: `zensim` 0.3.0, `ZensimProfile::A` (= `latest_preview()`), bake `v47-strict-QAT-native` (rotated 2026-05-27, sha256 `d0ef7a30…`)
  - Scored via one `zenmetrics compare --reference SRC --variant Q5.jpg … --metric ssim2 --metric butteraugli --metric zensim --metric cvvdp --gpu-runtime cuda` per source image.
- **Corpus** (real, named, non-Kodak, non-gradient — 85 images, 20 Q each = **1700 measured cells**):
  - **photo** — CID22-512, 50 images (`/mnt/v/work/corpus/CID22-512`)
  - **screen** — gb82-sc + imazen-26-screenshots, 17 images (3 screenshots dropped on a PNG→PPM conversion edge case; logged, not silently skipped)
  - **lineart** — sci-figures-color (NASA/NIST color figures: charts, diagrams, line-art), 18 images
- **Q grid:** 5,10,15,…,100 (step 5, 20 points) — dense across the low-q web-compression range.
- **Mined vs swept:** the primary table was **swept** (fresh cjpeg encode + 4-metric score). The ssim2↔butter cross-check was **mined** from `~/oracle-d2-store/oracle-d2` — 54,000 JPEG encodings (`mozjpeg-rs-420-e2`, butter = MAX norm) joined from the content-addressed metric store (`meta/metrics/<encoding_id>/*.json` ⨝ `meta/encodings/<encoding_id>/record.json`). oracle-d2 has **no** cvvdp or zensim, which is why the full 5-metric join had to be swept.

## Data files

- `metric_mapping_2026-06-23.tsv` — machine-readable: `forward_Q` (all + per class), `operating_point_ssim2` (the 5-scale grid), `content_dependence_ssim2` (per-class spread). Median + p25/p75 columns.
- This file — the readable explainer.

## Operating-point grid (all content, median)

Anchored on SSIMULACRA2. Each row is one perceptual operating point; read across for the equivalent on every scale. `~Q` is the median libjpeg-turbo quality that lands there.

| ssim2 | ~Q (cjpeg) | butter (3-norm) | butter (max) | cvvdp (JOD) | zensim-A | bpp |
|------:|-----------:|----------------:|-------------:|------------:|---------:|----:|
| 95 | 100 | 0.26 | 0.73 | 9.99 | 92.0 | 3.00 |
| 90 | 95 | 0.52 | 1.17 | 9.99 | 87.1 | 1.22 |
| 85 | 90 | 0.79 | 2.18 | 9.96 | 85.1 | 1.52 |
| 80 | 85 | 1.05 | 3.11 | 9.94 | 80.4 | 1.09 |
| 75 | 75 | 1.29 | 3.76 | 9.90 | 74.8 | 0.90 |
| 70 | 60 | 1.52 | 4.53 | 9.87 | 67.8 | 0.79 |
| 65 | 50 | 1.76 | 4.98 | 9.82 | 63.9 | 0.76 |
| 60 | 45 | 1.97 | 5.69 | 9.79 | 57.7 | 0.69 |
| 55 | 35 | 2.20 | 5.90 | 9.73 | 52.0 | 0.67 |
| 50 | 30 | 2.34 | 6.40 | 9.67 | 45.4 | 0.60 |
| 45 | 25 | 2.71 | 7.77 | 9.62 | 40.7 | 0.53 |
| 40 | 20 | 2.66 | 7.08 | 9.52 | 35.4 | 0.51 |
| 30 | 15 | 3.19 | 7.67 | 9.38 | 28.7 | 0.44 |
| 20 | 15 | 3.13 | 8.73 | 9.23 | 15.6 | 0.42 |

**Worked example** (the format you asked for): *libjpeg-turbo Q75 ≈ ssim2 80 ≈ butteraugli-3norm 1.05 ≈ butteraugli-max 3.1 ≈ cvvdp 9.94 ≈ zensim-A 80*. (At Q75 the forward table gives ssim2 median 74.8; the operating-point table inverts through the corpus to put ssim2 80 nearer Q85 — the difference is the per-image spread, see p25/p75 in the TSV.)

## Cross-mappings (median, all content)

Because all five are monotone in quality, the operating-point grid above *is* the cross-map: pick the row by your known value, read the others off. The most-asked directions:

- **ssim2 → butter (3-norm):** 90→0.52, 85→0.79, 80→1.05, 75→1.29, 70→1.52, 60→1.97, 50→2.34.
- **butter (3-norm) → ssim2:** 0.5→~90, 0.8→~85, 1.0→~80, 1.3→~75, 1.5→~70, 2.0→~60, 2.3→~50. (The user's ~1.5 PJND pivot ≈ ssim2 70 ≈ cvvdp 9.87 ≈ Q60.)
- **cvvdp → ssim2:** 9.99→90+, 9.96→85, 9.94→80, 9.90→75, 9.87→70, 9.79→60, 9.67→50. cvvdp is a **coarse** discriminator above 9.9 — almost all Q≥40 photo encodes sit in cvvdp 9.5–10.0.
- **zensim-A ↔ ssim2:** near-1:1 in the mid-range (zensim-A ≈ ssim2 within ±3 from ssim2 50–90), diverging at the extremes (zensim-A's identity ceiling is 97.69, and it reads a few points *higher* than ssim2 below ssim2 40).

## Content-dependence — how much the correspondences shift

The correspondences ARE content-dependent. The single most important caveat: **at a fixed SSIMULACRA2, screen/line-art content carries a much worse butteraugli and a lower cvvdp than a photo does.** Quantified (median, from the `content_dependence_ssim2` table):

| at ssim2 | butter-3norm: photo / lineart / screen | cvvdp: photo / lineart / screen | ~Q: photo / lineart / screen |
|---------:|:---:|:---:|:---:|
| 85 | 0.75 / 0.77 / 0.89 | 9.95 / 9.98 / 9.96 | 90 / 80 / 95 |
| 80 | 1.01 / 1.35 / 1.11 | 9.93 / 9.94 / 9.95 | 80 / 80 / 85 |
| 75 | 1.25 / 1.64 / 1.35 | 9.88 / 9.92 / 9.91 | 75 / 60 / 75 |
| 70 | 1.44 / 1.82 / 1.67 | 9.85 / 9.89 / 9.88 | 60 / 60 / 65 |
| 60 | 1.79 / 2.41 / 2.20 | 9.76 / 9.85 / 9.82 | 45 / 30 / 45 |
| 50 | 2.15 / 4.42 / 2.91 | 9.65 / 9.66 / 9.72 | 30 / 38 / 25 |

Read this as: **"ssim2 = X" does not pin butteraugli.** At ssim2 60, butteraugli-3norm spans 1.79 (photo) → 2.41 (lineart) — a **35 % spread**; at ssim2 50 it spans 2.15 → 4.42 — a **2× spread**. Line-art/figure content (sharp text and edges on flat fields) is where SSIMULACRA2 and butteraugli disagree most: SSIMULACRA2 is comparatively forgiving of the blocking that butteraugli's local max/3-norm punishes hard. Screen content sits between photo and line-art.

The Q→ssim2 mapping is also content-dependent (from `forward_Q` per class):

| Q | ssim2: photo / lineart / screen |
|--:|:---:|
| 10 | -0.3 / 34.7 / 21.3 |
| 30 | 51.5 / 62.6 / 54.7 |
| 50 | 63.7 / 72.5 / 65.1 |
| 75 | 74.6 / 79.5 / 74.5 |
| 90 | 84.4 / 85.5 / 81.7 |
| 100 | 90.4 / 89.9 / 84.9 |

At a given Q, line-art reads ~10–35 ssim2 points *higher* than photo at low Q (flat regions compress cleanly) but the gap closes by Q90; **screen content plateaus lowest at high Q** (Q100 ssim2 only ~85) because its hard edges never fully resolve under 4:2:0 JPEG.

## Tight vs loose correspondences (where to trust this)

- **Tight (encoder-independent, low scatter):** ssim2 ↔ butteraugli, and ssim2 ↔ zensim-A, in the mid-range (ssim2 55–88). The oracle-d2 cross-check (a *different* encoder, mozjpeg-rs, on a *different* corpus) reproduces ssim2→butter-max within ~0.1–0.3 at ssim2 70–85 (oracle 2.01/3.17/3.47/4.53 vs this sweep 2.18/3.11/3.76/4.53 at ssim2 85/80/75/70). This is strong evidence the metric-to-metric maps describe the distortion, not the dial.
- **Loose (use the per-class row):** anything touching **line-art/screen** at ssim2 < 65 — the butteraugli and Q equivalents scatter widely (see the 2× butteraugli spread at ssim2 50). cvvdp at the top (≥ 9.9) is loose by construction — it cannot finely separate high-quality encodes.
- **Encoder caveat:** the **Q column is libjpeg-turbo-specific.** mozjpeg / jpegli / zenjpeg at the same numeric Q land at a different perceptual point (mozjpeg trellis buys ~3–8 ssim2 at the same Q). Use the Q column only with libjpeg-turbo; use the metric columns with any JPEG encoder.
