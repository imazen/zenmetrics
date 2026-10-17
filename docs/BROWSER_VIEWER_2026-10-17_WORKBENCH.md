# Browser workbench, 2026-10-17

The viewer at https://metrics.imazen.org/ became a workbench: it lists every
CLI metric with its browser status, loads metric code on demand, adds the
three no-reference ports, sweeps JPEG XL distances for any metric, and
opens from links. The link contract is in
[BROWSER_WORKBENCH_LINKS.md](BROWSER_WORKBENCH_LINKS.md). Earlier history:
[2026-10-13](BROWSER_VIEWER_2026-10-13.md),
[loop player](BROWSER_VIEWER_2026-10-15_LOOP_PLAYER.md).

## Coverage

`web/metrics/coverage.json` (served at `/coverage.json`) has one entry per
CLI `MetricKind` name, 115 names. `check-pages.py` derives
every status from what the browser did in that run and fails when the
shipped file disagrees:

- `score+map` / `score-only`: the module scored the metric on the check's
  256×256 fixture (1024×1024 for metrics that refuse the smaller size), and
  for `score+map` also returned the metric's own native map. The listed map
  names are the ones that evaluation published.
- `unavailable`: the module was asked to score the name and refused it; the
  reason quotes that refusal.

Counts at this commit: 19 score+map, 64 score-only,
32 unavailable (learned metrics are covered below).

## Native comparison of every browser score

`browser_oracle` now also scores every browser-scored metric natively, through
the native build of the same entry (`score_json`; Zensim through
`Reference`), on the same fixture bytes. `check-pages.py` records each
column's absolute difference (`native_scalar_delta` in its report). On i265
(Chromium 153, local build of this commit): 83 metrics compared, 46 bit
identical, 37 differ by 1.4e-17 to 2.1e-5. The largest are
`butteraugli-original` 2.1e-5, `butteraugli` max 9.1e-6, `dssim` 4.0e-7,
`ssimulacra1` 2.9e-7, `flip-v17` 2.8e-7, `iwssim-piq` 5.1e-8, `flip` 3.0e-8 and
`iwssim` 1.7e-8; the rest are at most 9.1e-13. These are recorded, not
gated: no tolerance is approved for them. The newly added PIQE and BRISQUE
match exactly; NIQE's distorted score matches exactly and its reference score
differs by 2.8e-14.

## Dynamic loading

The module is built twice from one crate:

| Build | Features | Contents | wasm bytes | gzip -9 |
| --- | --- | --- | ---: | ---: |
| before (one module) | default | every CPU metric, all decoders | 13,558,404 | 3,265,209 |
| `pkg/` core | `core` | JPEG decoder and encoder, Zensim | 4,713,343 | 1,517,758 |
| `pkg-full/` full | `full` | every CPU metric, PNG/WebP/JPEG XL decoders, NR | 13,940,979 | 3,482,433 |

The page starts the core module only. The first non-Zensim metric, a PNG,
WebP or JPEG XL upload, or the demo's first score (after which it is fetched
in the background, so later choices and offline use need no fetch) imports
the full module into the same coordinator worker with its own memory and
Rayon pool. Each request runs in the module that compiles its metric; the
score names the module. Cancellation words are posted per module, and the
page signals all of them. The demo's JPEG XL loop starts after the first
comparison, so its encoder module's compile does not delay the first score.

`scripts/checkpoint/measure-pages-load.py` measured the first load, 9 fresh
contexts each, served from localhost, Chromium 153 on i265; first score is
navigation to the demo's score as polled from the page:

| Site | Bytes requested before the first score | Median first score | at 20 Mbit/s, 20 ms |
| --- | ---: | ---: | ---: |
| before | 13,628,706 | 263 ms | 5,878 ms |
| after | 4,841,042 | 192 ms | 2,285 ms |

Localhost runs are bimodal on this host (about 190 ms and 350–390 ms, in
both builds), so the medians are coarse. The emulated network rate uses
uncompressed transfer; Cloudflare's own compression is not measured here.

## JPEG XL impact

The loop player (2026-10-15) remains the step-by-step view of how a metric's
map steers jxl-encoder. It needs a spatial map. The new "JPEG XL impact"
panel works for every browser metric, maps or not: it encodes the reference
at each listed distance (`roundtrip_distance`, new in the JPEG XL codec
module; jxl-encoder `LossyConfig::new(distance)` as before), decodes it,
scores it with the selected metric, and plots score against bits per pixel.
Choosing a row or point loads that encode as the distorted image, so the
viewer shows its map. The Images panel also takes an explicit JPEG XL
distance, and `?codec=jxl&distance=2` encodes the demo at distance 2.

## Links

The URL contract and the coverage schema are in
[BROWSER_WORKBENCH_LINKS.md](BROWSER_WORKBENCH_LINKS.md). The metric bar
links the selected metric's dashboard model card, speed and agreement views,
and the coverage file, and shows its status and reason. The check opens
`?metric=psnr-y&codec=jxl&distance=2`, verifies the rewritten address and
back-links, runs a two-distance sweep and loads one row, opens a GPU-only
name and requires its reason with no score, and requires refusals naming
`metric=nope` and `q=500` while reading the fragment form `#codec=webp`.

## Coverage table

| CLI name | Status | Module | Maps | Reason |
| --- | --- | --- | --- | --- |
| `vmaf` | score-only | full | – | – |
| `vmaf-neg` | score-only | full | – | – |
| `vmaf-4k` | score-only | full | – | – |
| `vmaf-v1` | score+map | full | speed_u, speed_v, aim_0_h (+40) | – |
| `vmaf-studio601` | score-only | full | – | – |
| `vmaf-neg-studio601` | score-only | full | – | – |
| `psnr` | score+map | full | squared_error | – |
| `psnr-y` | score-only | full | – | – |
| `psnr-y601` | score-only | full | – | – |
| `psnr-y-studio601` | score-only | full | – | – |
| `psnr-y-libvmaf` | score-only | full | – | – |
| `psnr-ycbcr611` | score-only | full | – | – |
| `psnr-cb` | score-only | full | – | – |
| `psnr-cr` | score-only | full | – | – |
| `ciede2000` | score-only | full | – | – |
| `ssimc` | score-only | full | – | – |
| `flip` | score-only | full | – | – |
| `flip-v17` | score-only | full | – | – |
| `cw-ssim` | score-only | full | – | – |
| `cambi` | score-only | full | – | – |
| `butteraugli-original` | score-only | full | – | – |
| `ssimulacra1` | score-only | full | – | – |
| `ws-hv` | score-only | full | – | – |
| `ssim` | score-only | full | – | – |
| `nlpd` | score+map | full | diffmap, level_0, level_1 (+4) | – |
| `nlpd-iqa` | score-only | full | – | – |
| `lpips-alex` | unavailable | – | – | Browser refusal: Error: metric 'lpips-alex' is disabled (rebuild with `--features nn`) |
| `lpips-vgg` | unavailable | – | – | Browser refusal: Error: metric 'lpips-vgg' is disabled (rebuild with `--features nn`) |
| `lpips-squeeze` | unavailable | – | – | Browser refusal: Error: metric 'lpips-squeeze' is disabled (rebuild with `--features nn`) |
| `dists` | unavailable | – | – | Browser refusal: Error: metric 'dists' is disabled (rebuild with `--features nn`) |
| `stlpips-alex` | unavailable | – | – | Browser refusal: Error: metric 'stlpips-alex' is disabled (rebuild with `--features nn`) |
| `stlpips-vgg` | unavailable | – | – | Browser refusal: Error: metric 'stlpips-vgg' is disabled (rebuild with `--features nn`) |
| `deepdc` | unavailable | – | – | Browser refusal: Error: metric 'deepdc' is disabled (rebuild with `--features nn`) |
| `ckdn` | unavailable | – | – | Browser refusal: Error: metric 'ckdn' is disabled (rebuild with `--features nn`) |
| `afine` | unavailable | – | – | Browser refusal: Error: metric 'afine' is disabled (rebuild with `--features nn`) |
| `dreamsim` | unavailable | – | – | Browser refusal: Error: metric 'dreamsim' is disabled (rebuild with `--features nn`) |
| `arniqa` | unavailable | – | – | Browser refusal: Error: metric 'arniqa' is disabled (rebuild with `--features nn`) |
| `maniqa` | unavailable | – | – | Browser refusal: Error: metric 'maniqa' is disabled (rebuild with `--features nn`) |
| `liqe` | unavailable | – | – | Browser refusal: Error: metric 'liqe' is disabled (rebuild with `--features nn`) |
| `musiq` | unavailable | – | – | Browser refusal: Error: metric 'musiq' is disabled (rebuild with `--features nn`) |
| `hyperiqa` | unavailable | – | – | Browser refusal: Error: metric 'hyperiqa' is disabled (rebuild with `--features nn`) |
| `paq2piq` | unavailable | – | – | Browser refusal: Error: metric 'paq2piq' is disabled (rebuild with `--features nn`) |
| `pieapp` | unavailable | – | – | Browser refusal: Error: metric 'pieapp' is disabled (rebuild with `--features nn`) |
| `wadiqam-fr-live` | unavailable | – | – | Browser refusal: Error: metric 'wadiqam-fr-live' is disabled (rebuild with `--features nn`) |
| `wadiqam-fr-tid` | unavailable | – | – | Browser refusal: Error: metric 'wadiqam-fr-tid' is disabled (rebuild with `--features nn`) |
| `wadiqam-nr-live` | unavailable | – | – | Browser refusal: Error: metric 'wadiqam-nr-live' is disabled (rebuild with `--features nn`) |
| `wadiqam-nr-tid` | unavailable | – | – | Browser refusal: Error: metric 'wadiqam-nr-tid' is disabled (rebuild with `--features nn`) |
| `wd` | unavailable | – | – | Browser refusal: Error: metric 'wd' is disabled (rebuild with `--features nn`) |
| `ahiq` | unavailable | – | – | Browser refusal: Error: metric 'ahiq' is disabled (rebuild with `--features nn`) |
| `dmm` | unavailable | – | – | Browser refusal: Error: metric 'dmm' is disabled (rebuild with `--features nn`) |
| `ssim2` | score+map | full | diffmap, ssim_y_0, ssim_x_0 (+52) | – |
| `ssim2-gpu` | unavailable | – | – | GPU implementation (CubeCL on CUDA or native wgpu); this page has no CubeCL runtime. Score the CPU metric ssim2 here. Browser refusal: Error: Browser image pairs use SDR CPU metrics; GPU and absolute-nits HDR requests are unavailable |
| `butteraugli` | score+map | full | diffmap | – |
| `butteraugli-gpu` | unavailable | – | – | GPU implementation (CubeCL on CUDA or native wgpu); this page has no CubeCL runtime. Score the CPU metric butteraugli here. Browser refusal: Error: Browser image pairs use SDR CPU metrics; GPU and absolute-nits HDR requests are unavailable |
| `dssim` | score+map | full | diffmap, ssim_0, ssim_1 (+3) | – |
| `dssim-gpu` | unavailable | – | – | GPU implementation (CubeCL on CUDA or native wgpu); this page has no CubeCL runtime. Score the CPU metric dssim here. Browser refusal: Error: Browser image pairs use SDR CPU metrics; GPU and absolute-nits HDR requests are unavailable |
| `iwssim-gpu` | unavailable | – | – | GPU implementation (CubeCL on CUDA or native wgpu); this page has no CubeCL runtime. Score the CPU metric iwssim here. Browser refusal: Error: Browser image pairs use SDR CPU metrics; GPU and absolute-nits HDR requests are unavailable |
| `zensim` | score+map | core | diffmap | – |
| `zensim-gpu` | unavailable | – | – | GPU implementation (CubeCL on CUDA or native wgpu); this page has no CubeCL runtime. Score the CPU metric zensim here. Browser refusal: Error: Browser image pairs use SDR CPU metrics; GPU and absolute-nits HDR requests are unavailable |
| `cvvdp` | score+map | full | diffmap | – |
| `cvvdp-gpu` | unavailable | – | – | GPU implementation (CubeCL on CUDA or native wgpu); this page has no CubeCL runtime. Score the CPU metric cvvdp here. Browser refusal: Error: Browser image pairs use SDR CPU metrics; GPU and absolute-nits HDR requests are unavailable |
| `iwssim` | score+map | full | cs_0, cs_1, cs_2 (+7) | – |
| `iwssim-piq` | score-only | full | – | – |
| `gmsd` | score+map | full | gms | – |
| `mdsi` | score-only | full | – | – |
| `ms-gmsd` | score-only | full | – | – |
| `ms-gmsdc` | score-only | full | – | – |
| `hdrvdp` | unavailable | – | – | Scores absolute-luminance HDR input; this page decodes SDR sRGB8 only. Browser refusal: Error: Browser image pairs use SDR CPU metrics; GPU and absolute-nits HDR requests are unavailable |
| `hdrvdp3` | unavailable | – | – | Scores absolute-luminance HDR input; this page decodes SDR sRGB8 only. Browser refusal: Error: Browser image pairs use SDR CPU metrics; GPU and absolute-nits HDR requests are unavailable |
| `psnrhvs` | score+map | full | csf_error, masked_error, csf_error_r (+5) | – |
| `psnrhvs-y` | score-only | full | – | – |
| `haarpsi` | score+map | full | similarity | – |
| `haarpsi-y` | score-only | full | – | – |
| `fsim` | score+map | full | similarity | – |
| `ssim-pyiqa` | score-only | full | – | – |
| `ssim-pyiqa-f32` | score-only | full | – | – |
| `ssim-pyiqa-f32-k2` | score-only | full | – | – |
| `ssim-pyiqa-f32-k3` | score-only | full | – | – |
| `ssim-pyiqa-f32-k4` | score-only | full | – | – |
| `ssim-pyiqa-f32-k5` | score-only | full | – | – |
| `ssim-pyiqa-f32-k6` | score-only | full | – | – |
| `ssimc-pyiqa-f32` | score-only | full | – | – |
| `ssimc-pyiqa-f32-k2` | score-only | full | – | – |
| `ssimc-pyiqa-f32-k3` | score-only | full | – | – |
| `ssimc-pyiqa-f32-k4` | score-only | full | – | – |
| `ssimc-pyiqa-f32-k5` | score-only | full | – | – |
| `ssimc-pyiqa-f32-k6` | score-only | full | – | – |
| `ms-ssim-pyiqa` | score-only | full | – | – |
| `ms-ssim-pyiqa-f32` | score-only | full | – | – |
| `ms-ssim-pyiqa-f32-k2` | score-only | full | – | – |
| `ms-ssim-pyiqa-f32-k3` | score-only | full | – | – |
| `ms-ssim-pyiqa-f32-k4` | score-only | full | – | – |
| `ms-ssim-pyiqa-f32-k5` | score-only | full | – | – |
| `ms-ssim-pyiqa-f32-k6` | score-only | full | – | – |
| `fsim-k2` | score-only | full | – | – |
| `fsim-k3` | score-only | full | – | – |
| `fsim-k4` | score-only | full | – | – |
| `fsim-nodownscale` | score-only | full | – | – |
| `fsim-y` | score-only | full | – | – |
| `vsi` | score+map | full | quality | – |
| `vsi-k2` | score-only | full | – | – |
| `vsi-k3` | score-only | full | – | – |
| `vsi-k4` | score-only | full | – | – |
| `vsi-nodownscale` | score-only | full | – | – |
| `msssim` | score+map | full | ssim, diffmap | – |
| `vif` | score+map | full | diffmap, info_loss_0, info_loss_1 (+10) | – |
| `vifvec` | score+map | full | diffmap, info_loss_l1_b1, info_loss_l1_b4 (+22) | – |
| `mad` | score+map | full | hi, lo | – |
| `ssim-libvmaf` | score-only | full | – | – |
| `msssim-libvmaf` | score-only | full | – | – |
| `psnrhvs-daala` | score-only | full | – | – |
| `mdctpsnr` | score+map | full | detection | – |
| `piqe` | score-only | full | – | – |
| `niqe` | score-only | full | – | – |
| `brisque` | score-only | full | – | – |
