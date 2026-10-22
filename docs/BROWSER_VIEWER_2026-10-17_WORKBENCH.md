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
- `nn-webgpu` / `nn-cpu-wasm`: the learned metric ran on that provider and
  met its card's parity gate (below).
- `unavailable`: the module was asked to score the name and refused it (the
  reason quotes the refusal), or, for a learned metric, the reason states
  its weights licence class or the missing route.

Counts at this commit: 20 score+map, 63 score-only, 14 nn-webgpu, 1 nn-cpu-wasm and 17 unavailable.

`iwssim-piq` serves the map of the metric its score names
(`Iwssim::with_params(.., IwssimParams::piq_luma())`); a native test requires
that evaluation's score to equal the browser's `iwssim-piq` score bit for bit.
`fsim-y` and `haarpsi-y` stay score-only. Their crates publish luma scores
(`fsim_luma8`, `haarpsi_luma8`) but no luma map. The public plane map routes
are not the same arithmetic: `haarpsi_luma8` averages each 2×2 block in f64
RGB before the luma dot product, while `haarpsi_plane_f32_with_diffmap`
averages an f32 luma plane; `fsim_luma8` fuses luma into its f32 decimation
window. A browser map built from either plane route would come from a
different computation than the score it sits beside. Serving one needs a luma
map entry point in those crates.

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

### Extra modules (2026-10-21)

A deployment can add wasm-bindgen modules without rebuilding the site. An
`extra-modules.json` beside `worker.js` (absent on the public site, where the
worker's fetch of it fails quietly) lists them; each is imported the first
time one of its metrics is scored or mapped:

```json
{"modules": [{"name": "m", "label": "Menu group", "url": "pkg-m/m.js",
  "metrics": [{"name": "m-metric", "maps": ["diffmap"], "cli": "m score {reference} {distorted}"}],
  "loopDefaults": [{"metric": "m-metric", "map": "diffmap", "scale": 1, "exponent": 1, "r": 1}]}]}
```

A module answers the calls the worker makes on core and full
(`module_name`, `available_metrics`, `available_spatial_maps`, `JobControl`,
`score_rgba`, `spatial_map`, `cached_spatial_map`, `loop_map`); `url` is
relative to `worker.js`, and `module_name()` must return `name`. A module
with `initThreadPool` starts like core and full; one without runs
single-threaded on its own unshared memory, so no cancellation words are
posted for it and a running call finishes before the next. Its metrics form
their own menu group and get no dashboard links; `cli`, when present,
replaces the CLI snippet, and `loopDefaults` join the loop's defaults.

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

## Learned metrics on WebGPU

The page runs 15 of the CLI's 24 learned metrics with
[onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) 1.30.0
(MIT), on its WebGPU provider unless the user picks the WebAssembly CPU
provider. `scripts/checkpoint/vendor-ort.py` fetches the npm tarball, checks
its sha512 integrity value, and ships only the JSPI build: its wasm is
16,758,545 bytes, under Cloudflare Pages' 25 MiB file limit, while the
asyncify build behind `ort.webgpu` (26,781,914 bytes) and the JSEP build
(28,312,028 bytes) are over it.

**Weights** are the qualified exports in the frozen sets
`models-df35ce6e87bbf1af`, `models-0fa5b37cc1c9f860` (WaDIQaM) and
`models-5c30c9e175858c15` (WD), uploaded unchanged to R2 (bucket `zentrain`,
public domain `zentrain.r2.imazen.org`, key
`zenmetrics-web/nn/<sha256>/<file>`, `Cache-Control: immutable`, CORS `*`),
not to GitHub. Only models whose card says `licence_class = "permissive"`
are published. `web/metrics/nn-models.json` lists each graph's URL, size and
sha256; the page checks the sha256 after download and keeps the bytes in
Cache Storage.

**Inputs** are built in the full module by zenmetrics-nn's own
`preprocess.rs`, `resize.rs` and `tfresize.rs`, included by path
(`crates/zenmetrics-browser/src/nn_prep.rs`), with each model's eval
transform restated from its source file (cited per model). The graphs are
the models' whole-metric `*_full` exports, except WaDIQaM (trunk and head
over ≤2000-patch chunks, the upstream concat and `Σya/Σa` pooling in the
page) and WD (its VGG16 trunk, then the statistics head from `wd.rs`,
restated in `nn_prep.rs` and run per tap). No-reference models score the
distorted image and then the reference under `<column>_ref`, as the CLI's
`PairedNoReference` does.

**Parity.** `crates/zenmetrics-browser/examples/nn_oracle.rs` scores the
same two fixed pairs (the 256×256 scalar fixture and the 480×352 map
fixture) natively on the rten engine (trunk graphs and Rust heads, an
independent path from the browser's fused graphs); its output is committed
as `benchmarks/browser_nn_2026-10-17/native-rten.json` (two runs were bit
identical). `check-pages.py` runs every model on both providers and compares
with the card gate, `tolerance_jnd = 1e-3` JND per column (`jnd_scale.tsv`).
A WebGPU result outside the gate is rerun once (i265's GPU flips bits) and
both runs are kept. On i265's RTX 4080 (Vulkan adapter `nvidia lovelace`,
headless Chromium 153 with `--enable-unsafe-webgpu --enable-features=Vulkan
--use-vulkan=native --use-angle=vulkan`), worst deviation per model in JND:

| Metric | Graph bytes | WebGPU | WASM CPU | Status |
| --- | ---: | ---: | ---: | --- |
| lpips-alex | 9,898,333 | 8.4e-08 | 8.4e-08 | nn-webgpu |
| lpips-vgg | 58,886,773 | 5.9e-08 | 3.0e-06 | nn-webgpu |
| lpips-squeeze | 2,942,223 | 6.8e-08 | 2.1e-08 | nn-webgpu |
| stlpips-alex | 9,923,995 | 3.2e-07 | 1.6e-06 | nn-webgpu |
| stlpips-vgg | 58,943,547 | 2.9e-07 | 8.4e-07 | nn-webgpu |
| dists | 58,944,709 | 6.9e-06 | 1.5e-04 | nn-webgpu |
| deepdc | 80,205,613 | 2.8e-06 | 1.0e-03 (9.99e-4) | nn-webgpu |
| arniqa | 93,983,732 | 2.8e-06 | 3.3e-06 | nn-webgpu |
| maniqa | 456,965,919 | 5.6e-06 | 3.3e-05 | nn-webgpu |
| musiq | 108,645,665 | 1.22e-03 | 1.7e-05 | nn-cpu-wasm |
| wadiqam-fr-live | 25,154,281 | 5.4e-07 | 4.9e-07 | nn-webgpu |
| wadiqam-fr-tid | 25,154,281 | 4.3e-07 | 2.3e-06 | nn-webgpu |
| wadiqam-nr-live | 20,960,992 | 3.9e-08 | 5.7e-08 | nn-webgpu |
| wadiqam-nr-tid | 20,960,992 | 6.8e-06 | 3.1e-06 | nn-webgpu |
| wd | 30,545,316 | 2.4e-07 | 1.6e-07 | nn-webgpu |

MUSIQ's WebGPU result on the 480×352 pair is 1.22e-3 JND from native on
every run (rerun identical, ORT graph optimisation `all`, `basic` and
`disabled` identical), while its WASM-provider result is within 1.7e-5 JND,
so the preprocessing matches and the difference is in the WebGPU kernels.
The gate is not widened: the page runs MUSIQ on the CPU unless WebGPU is
chosen, and its coverage entry says why. Chromium's SwiftShader adapter
(`--enable-unsafe-webgpu` without a GPU, the CI configuration) gives the same
classification for all 15 models.

**CI.** GitHub's runners have no GPU, so the Pages workflow runs the
learned metrics on Chromium's SwiftShader WebGPU adapter. SwiftShader is
slow: the whole check with all 15 models took 12 min 44 s on i265 (20
threads), the default CI subset (LPIPS-alex and -squeeze, ST-LPIPS-alex,
MUSIQ, WaDIQaM FR and NR live, WD: every model kind) 242 s. The full set
runs on manual dispatch (`nn_models=all`); its GPU receipt is committed. The
post-deploy check of the live site runs LPIPS-squeeze end to end.

**Placement.** The WebGPU provider can leave nodes it does not implement on
ORT's CPU kernels. The page profiles each session's first run and reports
how many nodes each provider ran; the check requires WebGPU-run nodes in
every WebGPU session and none in WASM sessions. On the RTX 4080, three
WebGPU sessions also ran nodes on ORT's CPU kernels: DeepDC (450 WebGPU, 150
CPU), ARNIQA (205, 7) and MUSIQ (624, 92); the other twelve ran every node on
WebGPU. Receipt: `benchmarks/browser_nn_2026-10-17/parity-i265-rtx4080.json`.

**Never a silent fallback.** A session gets only the requested provider. In
Chromium without a WebGPU adapter the request fails with "WebGPU
unavailable: requestAdapter() returned no adapter"; the WASM provider is a
separate choice (`ep=wasm`, or the menu), and the result names the provider,
the adapter and the node placement.

**Checked failures and recovery:** transparent input, an image below the
model's minimum and an unknown provider are refused; an aborted weight
download ("Weight download failed … Failed to fetch") and a truncated one are
reported, and the same worker downloads the weights afterwards; cancelling
an in-flight inference emits no score, and the new worker gives the same
score again.

**Not in the browser (measured reasons in `coverage.json`):** CKDN, DreamSim,
LIQE, HyperIQA and AHIQ have weights licence class `unknown`; A-FINE,
PaQ-2-PiQ and PieAPP are `noncommercial`. Their weights are not
redistributed. DMM's weights are permissive, but its score needs a 16×16 SVD
per patch over two VGG16 taps (`dmm.rs`, a LAPACK port whose residual floors
are owner-gated); the browser does not restate that solver.

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
| `lpips-alex` | nn-webgpu | nn | – | – |
| `lpips-vgg` | nn-webgpu | nn | – | – |
| `lpips-squeeze` | nn-webgpu | nn | – | – |
| `dists` | nn-webgpu | nn | – | – |
| `stlpips-alex` | nn-webgpu | nn | – | – |
| `stlpips-vgg` | nn-webgpu | nn | – | – |
| `deepdc` | nn-webgpu | nn | – | – |
| `ckdn` | unavailable | – | – | Weights licence class "unknown" ([weights] in crates/zenmetrics-nn/models/ckdn/card.toml): the browser does not redistribute these weights. |
| `afine` | unavailable | – | – | Weights licence class "noncommercial" ([weights] in crates/zenmetrics-nn/models/afine/card.toml): the browser does not redistribute these weights. |
| `dreamsim` | unavailable | – | – | Weights licence class "unknown" ([weights] in crates/zenmetrics-nn/models/dreamsim/card.toml): the browser does not redistribute these weights. |
| `arniqa` | nn-webgpu | nn | – | – |
| `maniqa` | nn-webgpu | nn | – | – |
| `liqe` | unavailable | – | – | Weights licence class "unknown" ([weights] in crates/zenmetrics-nn/models/liqe/card.toml): the browser does not redistribute these weights. |
| `musiq` | nn-cpu-wasm | nn | – | The WebGPU provider differs from the native rten score by 0.00122 JND at unified480x352/musiq (rerun 0.00122), over the card gate of 0.001 JND; the WASM provider is within 1.74e-05 JND, so the page runs this metric on the CPU unless WebGPU is chosen. |
| `hyperiqa` | unavailable | – | – | Weights licence class "unknown" ([weights] in crates/zenmetrics-nn/models/hyperiqa/card.toml): the browser does not redistribute these weights. |
| `paq2piq` | unavailable | – | – | Weights licence class "noncommercial" ([weights] in crates/zenmetrics-nn/models/paq2piq/card.toml): the browser does not redistribute these weights. |
| `pieapp` | unavailable | – | – | Weights licence class "noncommercial" ([weights] in crates/zenmetrics-nn/models/pieapp/card.toml): the browser does not redistribute these weights. |
| `wadiqam-fr-live` | nn-webgpu | nn | – | – |
| `wadiqam-fr-tid` | nn-webgpu | nn | – | – |
| `wadiqam-nr-live` | nn-webgpu | nn | – | – |
| `wadiqam-nr-tid` | nn-webgpu | nn | – | – |
| `wd` | nn-webgpu | nn | – | – |
| `ahiq` | unavailable | – | – | Weights licence class "unknown" ([weights] in crates/zenmetrics-nn/models/ahiq/card.toml): the browser does not redistribute these weights. |
| `dmm` | unavailable | – | – | Permissive weights ([weights] in crates/zenmetrics-nn/models/dmm/card.toml), but the score needs a 16×16 SVD per patch over two VGG16 taps (crates/zenmetrics-nn/src/dmm.rs, a LAPACK port whose residual floors are owner-gated); the browser has no graph or route for it and does not restate the solver. |
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
| `iwssim-piq` | score+map | full | cs_0, cs_1, cs_2 (+7) | – |
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

## Local-file learned routes (2026-10-18)

`nn-local` means conditional CPU WASM scoring with user-supplied weights.
CKDN and HyperIQA use the exact exported graphs named and SHA-256-pinned in
`web/metrics/nn-models.json`. Select the graph file and accept that model's
weight terms explicitly before loading it. A wrong file, wrong hash, or absent
consent refuses loading. These routes require browser JSPI support.

The selected weights stay in the coordinator worker's memory. The viewer does
not fetch them from a URL or put them in browser persistent storage. They are
not bundled in the public website. Reloading the page requires selecting
and consenting again. The current page retains the selected File and consent
so it can reload a replacement worker. The route uses the WASM CPU provider, and does not
establish WebGPU availability. The existing MUSIQ route is unchanged.

The separate qualification command requires the model directory and a native
RTEN baseline supplied by the caller:

```sh
just browser-local-weights-check SITE MODEL_DIRECTORY NATIVE_JSON REPORT_JSON
```

It checks both fixture pairs against the existing 0.001 JND gate, repeated
score identity, refusal before loading/consent, wrong-file/hash controls,
absence of ONNX network requests, and the actual local-file UI flow. Public
CI does not acquire restricted weights for this check. Its ordinary Pages
regression still checks catalog classification and unloaded-route refusal.

The updated catalog has 122 entries: 20 `score+map`, 63 `score-only`,
14 `nn-webgpu`, 1 `nn-cpu-wasm`, 2 conditional `nn-local`, and 22
`unavailable`. These categories describe available routes and their gates,
not a claim that every metric has been benchmarked at every image size.
