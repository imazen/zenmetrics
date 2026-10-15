# Encoder loop player, 2026-10-15

The viewer's "Encoder loop player" panel runs jxl-encoder's perceptual
quantization loop on the reference image inside the page and steps through
it: play, pause, step back and forward, or scrub through every scored
iteration of every seed. The demo scene loads when the page starts, and its
loop encodes straight away. While it encodes, each iteration appears as soon
as it is scored, for both kinds of steering: the steering map over the
reconstruction.

## Steering

The loop is steered either by jxl-encoder's own butteraugli, as in a normal
encode, or by the map selected in the Metric panel: any metric with a
spatial map in this build, and any of its maps (its fused `diffmap` by
default, or a per-scale map). One metric choice drives the score, the map
and the steering. The metric and map menus sit in a bar that stays at the
top of the page. The Map menu and the loop's Steer menu list every metric's
maps, grouped by metric, so choosing another metric's map in either one
switches the metric.

jxl-encoder's loop was calibrated on butteraugli. It compares each 8×8
block's 16-norm, times 1.2 (libjxl's `TileDistMap`), with the target
distance, and when it picks among seeds it compares the map's maximum, the
score, with the target. So another metric's map is first put on
butteraugli's scale:

1. The map is read as error: `1 − v` for similarity maps, the value itself
   for difference and detection-probability maps. Feature maps (pooling
   weights, information terms) have no error direction and are refused.
2. It is resampled to one value per pixel: each pixel takes the native cell
   that owns it; pixels no cell owns read as zero error.
3. Then `scale · error^exponent`. The score is the result's maximum.

The default `scale` and `exponent` for every map come from
`crates/zenmetrics-browser/examples/loop_norm_calibrate.rs`. Results are in
[benchmarks/loop_norm_calibration_2026-10-15.tsv](../benchmarks/loop_norm_calibration_2026-10-15.tsv),
and they are compiled into `src/loop_map.rs`. The fit encodes 17 CID22
training images at seven distances (0.5 to 6, effort 7) and compares, block
for block, the metric's normalized block distances with butteraugli's on
the same encodes. For each exponent on a grid from 0.1 to 4, the scale is
the mean log ratio over blocks where butteraugli sees at least 0.05. The
exponent with the smallest log RMSE wins. The panel shows each default's
log-space correlation with butteraugli on 19 held-out images (8 CID22
validation photos and 11 gb82-sc screenshots). Both fields can be edited.

Each metric's default map (all 124 maps are in the TSV):

| Metric · map | Scale | Exponent | Held-out log r | Held-out log RMSE |
| --- | ---: | ---: | ---: | ---: |
| butteraugli · diffmap | 1 | 1 | 1.000 | 0.000 |
| cvvdp · diffmap | 0.562 | 0.35 | 0.793 | 0.651 |
| dssim · diffmap | 16.68 | 0.45 | 0.892 | 0.427 |
| fsim · similarity | 4.115 | 0.35 | 0.851 | 0.491 |
| gmsd · gms | 4.53 | 0.35 | 0.842 | 0.508 |
| haarpsi · similarity | 3.783 | 0.4 | 0.849 | 0.499 |
| iwssim · diffmap | 5.884 | 0.35 | 0.784 | 0.620 |
| mad · hi | 1.169 | 0.1 | 0.448 | 0.774 |
| mdctpsnr · detection | 3.18 | 0.25 | 0.867 | 0.464 |
| msssim · diffmap | 8.361 | 0.4 | 0.896 | 0.421 |
| nlpd · diffmap | 0.5149 | 0.45 | 0.765 | 0.628 |
| psnr · squared_error | 5.755 | 0.25 | 0.764 | 0.649 |
| psnrhvs · csf_error | 0.389 | 0.25 | 0.825 | 0.573 |
| ssim2 · diffmap | 0.7351 | 0.25 | 0.671 | 0.746 |
| vif · diffmap | 1.268 | 0.3 | 0.644 | 0.733 |
| vifvec · diffmap | 1.492 | 0.35 | 0.696 | 0.726 |
| vmaf-v1 · speed_u | 0.7671 | 0.1 | 0.195 | 1.028 |
| vsi · quality | 0.7458 | 0.1 | -0.119 | 1.058 |
| zensim · diffmap | 8.402 | 0.35 | 0.849 | 0.533 |

On the held-out images, 12 of the 19 default maps track butteraugli's block
distances at log r 0.76–0.90. SSIMULACRA2, VIF and vifvec reach 0.64–0.70.
Butteraugli itself fits at scale 1, exponent 1 exactly.
Exponents below 1 compress the other maps' wider dynamic range to
butteraugli's. Three maps track it poorly, and steering with them follows
their own judgment rather than butteraugli's: MAD's `hi` (r 0.45),
VMAF v1's chroma-only `speed_u` (r 0.19) and VSI's `quality` (r −0.12). VSI's
map is a saliency-weighted quality contribution, not an error. The panel
shows each map's r. The fit matches block distances only; whether a metric
steers to better files at equal size is a separate measurement this does
not make.

The jxl-encoder side is `EncodeRequest::with_external_metric`
(imazen/jxl-encoder #130, merged with #129's `LoopTrace`). The loop
keeps its seeds, iteration count, block reduction and quant-field update,
and only the scorer changes. Its test plugs butteraugli in through the same
interface and gets the built-in loop's bytes, scores, diffmaps and block
distances exactly.

## Layers

All layers are at the image's resolution:

| Layer | Shows |
| --- | --- |
| Reconstruction + map (default) | the steering map over the reconstruction: `t = √(min(1, v/(2·target)))` on the palette, blended with weight `0.9·t³`, so the palette's middle is the target distance and pixels well below it stay readable |
| Reconstruction | the image this iteration scored (sRGB8) |
| Metric diffmap | the steering map, after normalization, one square-root scale across all frames (99.5th percentile) |
| Block distance vs target | each 8×8 block's distance (16-norm of the map times 1.2); the middle of the palette is the target, and blocks above it get finer quantization next |
| Quant field | the float quant field this iteration encoded with, one scale across frames |
| Quant field change | the loop's adjustment for the next iteration, `log2(next/current)`: red finer, blue coarser |
| Source | the input |

The info line gives:
- the seed and iteration;
- the score against the target;
- how many blocks are above the target;
- how many blocks the loop made finer or coarser;
- the final encoded size;
- what steered the loop.

"Save trace" writes the run as a `.ltrace` file, and "Saved trace" opens
one. jxl-encoder can also write one natively:

```
cargo run --release --example loop_trace_dump -- in.png out.ltrace [distance] [effort] [iterations]
```

Format (`JXLTRACE1`, little-endian):
1. The magic `JXLTRACE1\n`.
2. A `u32` header length, then the JSON header: `width`, `height`,
   `distance`, `effort`, `iters`, `bytes`, `xsize_blocks`, `ysize_blocks`,
   an optional `steering` label, and
   `frames: [{seed, iter, metric, score, target, has_next}]`.
3. The source as packed sRGB8.
4. Per frame: the quant field, the next quant field (when `has_next`), the
   tile distances and the diffmap, all as `f32`, then the reconstruction as
   packed sRGB8.

### Why the last frames repeat

jxl-encoder follows libjxl's schedule. Iterations 0 and 1 rescale every
block's quant field by `(tile/target)^cur_pow`. From iteration 2,
`resolved_cur_pow` is 0, and only blocks above the target get finer. When
no block is above the target, the field stops changing, and later
iterations encode and score the same image. The demo at distance 1 is such
a case. Both of its traces (Zensim-steered and built-in butteraugli) change
the field on every block in iterations 0 and 1, and their next/current
ratio is exactly 1 from iteration 2 on. The info line marks a repeated
frame. Higher distances, or images with blocks above the target, keep
changing.

Built-in steering scores through the same interface as an external
metric. The JPEG XL module plugs in butteraugli with jxl-encoder's CPU
parameters, which gives the built-in loop's bytes
(`builtin_scoring_keeps_the_bytes`). That is how its iterations can stream
to the page too.

## Encoder modules

JPEG XL, AVIF and WebP live in `web/codecs`, a Cargo workspace of their own
with its own `Cargo.lock`. The worker imports each module the first time it
is used. All three are single-threaded WebAssembly with SIMD128:

| Module | Encoder | Size (wasm) |
| --- | --- | --- |
| `jxl` | jxl-encoder with `__external_metric`, decoded by zenjxl-decoder | 5.1 MB |
| `avif` | zenavif (zenravif backend, speed 6), decoded by zenavif | 4.2 MB |
| `webp` | zenwebp lossy, decoded by zenwebp | 0.8 MB |

JPEG stays in the main module (zenjpeg). The Images panel encodes the
reference with the chosen encoder and quality. Changing either encodes it
again, and "Save encoded file" saves the result.

The modules build apart from the zenmetrics workspace and its sibling pins
for two reasons. The JPEG XL module needs jxl-encoder's external loop
metric, which the CLI's pinned jxl-encoder predates. Moving that pin would
also change the encodes in the benchmark sweeps. WebP and AVIF use the
CLI's pinned revisions (`ci/sibling-pins.tsv`), so browser and sweep encodes
match. jxl-encoder is pinned to its main
at `a8c11520`, the merge of #130.
`scripts/checkpoint/build-codecs.sh` builds the modules, and
`build-browser.sh` calls it.

## Validation

`check-pages.py` runs in headless Chromium:
- The demo scene loads by default (JPEG at quality 40, scored and mapped),
  and its live loop encodes, steered by Zensim's map through the external
  metric.
- Through the worker protocol:
  - MS-SSIM and DSSIM list their maps with `diffmap` first, and a second
    map read from the evaluation cache equals the same map computed fresh;
  - a score-only metric returns no map;
  - JPEG, WebP, AVIF and JPEG XL round-trip with the right file magic;
  - an MS-SSIM-steered loop returns a trace and scores reconstructions;
  - a zero scale is refused.
- A synthetic two-frame trace checks the info line (blocks above target,
  finer and coarser counts, the score-only final pass), exact pixels of the
  reconstruction and quant-field-change layers, and refusal of a malformed
  file.
