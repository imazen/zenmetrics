# Browser viewer validation, 2026-10-13

The current viewer implements the native SDR CPU catalog and Zen image codecs.
This advances [issue 70](https://github.com/imazen/zenmetrics/issues/70); it does
not close all acceptance gates. There is no benchmark backend fallback.

## Measured browser checks

Commands run in a CI-pinned snapshot, with Chromium 153.0.8010.12 and two Rayon
workers:

```
cargo run --locked --release -p zenmetrics-browser --example browser_oracle
just viewer-check <browser-build> <native-oracle.json> <browser-results.json>
```

All 38 selectable scalar metrics produced results on the textured 256×256
fixture. Zensim codec-target score and raw map match the native fixture exactly.
The maximum absolute native/WASM raw-map differences on the textured 64×64 pair
were:

| Native map | Maximum absolute difference |
| --- | ---: |
| Butteraugli | 0.000021696090698242188 |
| CVVDP | 0.0000002384185791015625 |
| DSSIM scale-0 SSIM | 0.000010013580322265625 |
| GMSD similarity | 0 |
| PSNR normalized squared error | 0 |

These are measured differences, not newly approved parity tolerances. The
fixture does not establish all-image parity. Raw map samples and dimensions
remain unchanged by visualization; similarity maps display `1 - similarity`.

One 1920×1080 SSIM2 run took 196.57000000029802 ms for scoring and
211.73000000044703 ms for the complete score-plus-PSNR-map worker request.
This is one interactive measurement, not a fleet benchmark or a stable latency
qualification. The browser check records both timings rather than omitting map
work from the complete request.

The compiled all-metric/all-codec module was 11,160,520 bytes uncompressed and
2,735,932 bytes with Python gzip. It exceeds issue 70's 2 MB / 650 KB gates.
The later direction to include native codecs supersedes that issue's proposed
browser decoding path; no smaller alternate module is silently selected.

The optimized WASM build completed in 21 seconds with run-heavy peak RSS
0.59 GiB, four build jobs and a 12 GiB hard memory cap. Sources/tools were cached;
this is not a completely cold toolchain installation measurement.

The upload controls, metric switching and map switching also passed with the
browser context offline and zero observed network requests during those actions.
An independently encoded transparent PNG retained its alpha and was refused for
opaque-only scoring.

## Remaining acceptance work

SSIMULACRA2 has no native spatial-map API in the current pinned fast-ssim2 port.
It remains available as a scalar metric; the viewer explicitly names a separate
map metric. No other map is labeled as SSIMULACRA2.

The native CLI does not currently expose issue 70's alpha-policy flag. The viewer
preserves alpha and refuses non-opaque scoring. Noise/minimum-white-black alpha
policies and exact matching Python snippets remain unimplemented. The CLI
snippet uses verified existing flags and pins zensim to profile B.

Animated spatial steering remains separate from generating global-quality JPEG
candidates. A codec must expose a qualified spatial-control input before that
interaction can be implemented without inventing a control contract.

## Reproducible deployment

The Pages workflow builds the matching source commit, locked dependencies and
23 sibling pins. Zenflate main is pinned at
`71ee10b6ce5317de74b2e9b03c78b39fcaf805df`; the compatible zenpng pin is
`c461fde90bffeb4c750abc892296e8a4f333a6e5`. The direct zenflate patch also covers
PNG's dependency, rather than leaving an older registry copy in the decoder.

The browser module requires relaxed SIMD128, shared memory and initialized
Rayon workers. Missing isolation or unsupported WASM capabilities produce an
error. Browser scores never satisfy fleet benchmark acceptance by themselves.
