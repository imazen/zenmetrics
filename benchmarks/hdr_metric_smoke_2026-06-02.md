# HDR-metric smoke + sanity — zenmetrics `score --hdr` (chunk 2)

**Date:** 2026-06-02 · **host:** lilith · **rustc:** 1.96.0 · **branch:** `feat/hdr-metrics`
**Harness:** `scripts/hdr/hdr_metric_smoke.sh` (self-checking, 16 assertions, all PASS)
**Binary:** `cargo build --release -p zenmetrics-cli --features hdr` (CPU metrics, no GPU)

## What was wired (chunk 2)

`zenmetrics score --hdr` now decodes HDR sources to absolute luminance (cd/m²)
and preps per metric before the existing CPU kernels:

- **SDR metrics** (ssim2 / dssim / butteraugli / zensim): PU21-encode (banding_glare,
  100 cd/m² → ~256) → clamp to sRGB8. Highlights above ~80 cd/m² clamp at 255 (the
  u8 contract; faithful f32 input is a later chunk).
- **cvvdp** (GPU-only in the CLI score path): peak-normalize → sRGB-encode → sRGB8.

New surface: `crates/zenmetrics-cli/src/hdr.rs` (decode + prep), the `hdr` build
feature, and `ScoreArgs::--hdr`. Decode mirrors `zenhdr-corpus`; PU21/transfer
mirror `zensim::{pu21,transfer}` / `zenmetrics_api::hdr`.

Decode paths:
| Input | How |
|---|---|
| `.exr` | `image` EXR reader; values already absolute cd/m² |
| `.heic`/`.heif` | `heic` base (P3/sRGB) + ISO 21496-1 `tmap` gain map → `ultrahdr_core::apply_gainmap` (LinearFloat ×203 nits) |
| `.jpg`/`.jpeg` | `ultrahdr-rs` `decode_hdr(4.0)` — Google-format UltraHDR only (`hdrgm:` XMP, primary or MPF-secondary) |

## Results (all assertions PASS)

### UltraHDR JPEG (`ultrahdr/test_ultrahdr.jpg`, 256×256)
| metric | identity | q15 re-encode (distorted) |
|---|---|---|
| ssim2 | 100.000000 | 78.498318 |
| dssim | 0.000000 | 0.002148 |
| butteraugli (max / pnorm3) | 0.0 / 0.0 | 3.843829 / 2.052711 |
| zensim | 100.000000 | 84.330082 |

Distorted JPEG produced by `examples/make_distorted` (decode → re-encode @ base_q=15,
gainmap_q=15 via `ultrahdr-rs::Encoder`).

### HEIC (gain-map, ISO 21496-1 tmap)
| pair | metric | score |
|---|---|---|
| `FE0C…HEIC` identity | ssim2 | 100.000000 |
| `FE0C…HEIC` identity | dssim | 0.000000 |

### EXR
| pair | metric | score |
|---|---|---|
| `IMG_1509.exr` identity | ssim2 | 100.000000 |
| `32F76D88.exr` vs `94067DD9.exr` (different content, matched dims) | ssim2 | −326.561476 |

### Cross-path consistency (decode determinism)
The 16 corpus EXRs were decoded from HEIC by `zenhdr-corpus`. Scoring a HEIC against
its own EXR confirms the CLI's HEIC path reproduces them **bit-for-bit**:
| pair | metric | score |
|---|---|---|
| `IMG_1420.HEIC` vs `IMG_1420.exr` | ssim2 | 100.000000 |
| `IMG_1420.HEIC` vs `IMG_1420.exr` | butteraugli (max/pnorm3) | 0.0 / 0.0 |

### Fleet worker: `batch --hdr` (chunk 3)

`zenmetrics batch --hdr` is the fleet-scale primitive — a chunk TSV of
`(ref_path, dist_path, …)` rows, each decoded via the same HDR path and scored,
columns appended. A launcher hands each fleet box a chunk + `--hdr`; mixed input
types are fine in one chunk. The u8 PU path reuses the orchestrator's existing
`TaskData::Srgb8` lane, so **no orchestrator change is needed** for HDR at scale
(`TaskData::LuminanceF32` is only for the faithful f32 path — chunk 4).

Mixed-input chunk (one TSV, `--metric ssim2 --hdr`):
| row | ref → dist | ssim2 |
|---|---|---|
| jpeg_identity | `test_ultrahdr.jpg` → itself | 100.000000 |
| jpeg_distorted | `test_ultrahdr.jpg` → q15 re-encode | 78.498318 |
| heic_vs_exr | `IMG_1420.HEIC` → `IMG_1420.exr` | 100.000000 |

## Notes / scope

- **UltraHDR JPEG support is Google-format only.** Apple-namespace gain-map JPEGs
  (`apple_gainmap_*.jpg`) and libultrahdr *encoder-input components*
  (`fullColor-fullRes-IDEAL.jpg`, `luminosity-lowRes.jpg`) report "No gain map
  metadata" — they lack the `hdrgm:` XMP `ultrahdr-rs` keys on. Apple JPEG gain
  maps would need an Apple-namespace path in `ultrahdr-rs` (out of chunk-2 scope;
  the HEIC path already handles Apple's ISO 21496-1 tmap).
- **cvvdp / iwssim** are GPU-only in the CLI `score` path (`gpu-cvvdp`/`gpu-iwssim`);
  the HDR branch routes cvvdp through the peak-normalized sRGB prep but needs a GPU
  build + a peak-matched display model for faithful absolute calibration (follow-up).
- **u8 bottleneck:** SDR metrics ingest sRGB8, so PU highlights above ~80 cd/m²
  clamp. Faithful f32/u16 HDR metric input is deferred (chunk 4 — touches the
  per-metric kernels).
