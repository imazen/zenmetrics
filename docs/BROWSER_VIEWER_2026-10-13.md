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

SSIMULACRA2's native spatial terms come from fast-ssim2's
`compute_ssimulacra2_maps` (fast-ssim2 PR #12, pinned at its branch head until it
merges). The `ssim2` map choice shows `ssim_y_0`, the full-resolution Y-channel
SSIM error term. The score pools 54 such terms (6 scales × X/Y/B × SSIM error,
edge artifact, detail lost) by L1 and 4-norm, so this is one component, not a
per-pixel score. The browser keeps scoring SSIMULACRA2 through zenmetrics-api.

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

## Live deployment observation

Source `a94d4767be43cdccf9c08b96f512c9826a87ef9b` was deployed and the
browser checks passed on both the Pages origin and https://metrics.imazen.org/
on 2026-10-13. The custom domain was active at that observation. Resolve its
current behavior with `just site-check-live <assembled-site> https://metrics.imazen.org/`;
read `/site-source.json` to identify the currently deployed source and hashes.

Local native browser tests passed 38 tests with two existing ignores unchanged.
The native CLI compile check and targeted browser clippy completed. The wider
advisory quality kit was absent on the build host; its wrapper could not run.
Changed Rust files passed format checks, and the tracked-file hygiene check
reported no address/identifier hits. Remote CI was started after the push;
this observation does not claim every remote job was already green.

## Large-image completion and job control follow-up

The earlier small fixtures missed a deterministic trap in the live demo:
768×512 JPEG encoding/decoding and zensim reference preparation succeeded,
but comparison trapped in zensim's native background deallocation thread.
`96c3c9bb` in zensim frees buffers synchronously on wasm32; Rayon computation
and the metric definition remain unchanged.

The revised viewer publishes the scalar before computing the optional map.
Its `JobControl` uses how-far 0.1.0 `Stop` and `Report` over four shared atomic
words. Completed input rows are actual work units. Native kernel totals are
unknown: the UI shows the current stage and elapsed time with an indeterminate
bar, rather than turning cancellation polls into a percentage.

Zensim, PNG decoding and JPEG encoding receive cooperative Stop policies.
Other native metric adapters have no browser Stop parameter. Cancel signals
shared memory immediately and, after 250 ms, terminates and recreates the
coordinator and its worker pool. A 120-second request deadline also prevents
an unresolved promise from leaving the UI busy indefinitely. Runtime traps
invalidate the worker. No alternate algorithm or backend supplies a score.

Chromium 153.0.8010.12 passed all 38 scalar routes, exact existing zensim
score/map oracle assertions, full-HD zensim score/map, the actual demo,
atomic cancellation with no score, subsequent comparison and UI cancellation
with fresh-worker recovery. `just viewer-check` now includes these regressions.
The test run used run-heavy: peak RSS 0.17 GiB, minimum available 26579 MiB,
peak load 1.64. This is browser verification, not benchmark acceptance.

Workspace metrics use archmage/magetypes 0.9.31-beta.1 at published source
revision `da0ed5deca40ac60df32dcf4f6e7602658b2f959`, verified from both crates'
VCS metadata. Cargo rejects the registry prerelease alongside pinned siblings'
registry `^0.9.30` constraints; the distinct git source preserves those stable
requirements. The lock is resolved against all 23 pinned siblings.

Native browser tests passed 39 tests with the same two existing ignores.
Targeted browser clippy (`--no-deps -- -D warnings`) and formatting passed.
A broader dependency clippy invocation stopped on an existing VMAF doc-comment
lint; it did not establish workspace-wide lint cleanliness. The all-feature
module now measures 12,734,524 bytes, or 2,980,985 bytes with Python gzip; the
issue 70 size gate remains unmet.

The follow-up source `a7803994280dc25d298d0a98ee620b556076504b` was deployed
to https://metrics.imazen.org/ and the expanded browser checks passed there on
2026-10-13, including the actual demo and cancellation/recovery. The live
verification used run-heavy: peak RSS 0.18 GiB, minimum available 26633 MiB,
peak load 0.50. Remote workspace CI was still running at this observation.

## Upgrade isolation and browser readiness

A reported loader/module mismatch provided 25 pages to a module requiring 42.
The current generated glue requests 42. An undersized-memory negative control
produces the same WebAssembly LinkError signature; the matching generation
passes initialization and the full viewer suite.

Pages now places the app, coordinator, generated glue, WASM and nested Rayon
helpers under `assets/<source-commit>/`. The browser tests block historical
`/pkg/` URLs and still require the real demo, all 38 CPU scalar routes,
full-HD zensim and cancellation/recovery. Root entry/source/worker files use
no-store; immutable asset cache rules do not overlap those paths. Cloudflare
[joins duplicate header values across matching rules](https://developers.cloudflare.com/pages/configuration/headers/),
so a broad no-store rule would interfere with immutable caching.

Brave users see the instruction to turn Shields off for this site. Actual
relaxed-SIMD validation and Rayon pool initialization remain required.
WebGPU is tested with adapter/device creation, compute dispatch and readback;
missing adapters are reported explicitly. Headless Chromium on the validation
host reported no WebGPU adapter. This is an honest readiness diagnostic, not
GPU metric execution or hardware benchmark qualification; this browser build
still selects native CPU metrics.

The versioned local build passed the expanded checks under run-heavy: peak RSS
0.18 GiB, minimum available 26555 MiB, peak load 0.52. The advisory quality kit
was absent; JavaScript syntax, Python compilation and tracked-file hygiene
checks passed.

Source `e54713a7b2b5fffab06120f72173a25d627b6d96` was deployed and the expanded
suite passed on https://metrics.imazen.org/ on 2026-10-13. Live responses
reported `Cache-Control: no-store` for the entry page and
`public, max-age=31536000, immutable` for versioned glue. The ready message
reported eight Rayon workers and SIMD128 plus relaxed SIMD. The live suite
used run-heavy: peak RSS 0.18 GiB, minimum available 26636 MiB, peak load 0.22.
The separate live-header check used peak RSS 0.13 GiB, minimum available
27501 MiB, peak load 0.21. Remote workspace CI remained in progress.

## Map visualization scale follow-up

The deployed 768×512 JPEG demo exposed a visualization defect: a shared fixed
maximum of 1 clipped 65.92508951822916% of Butteraugli samples and
97.01156616210938% of CVVDP samples, while the Zensim maximum was
0.026854952797293663. Native values use different metric units.

The viewer now starts with square-root visualization at each map's 99.5th
percentile. The legend shows its per-image range. Fixed, maximum and linear
percentile scaling remain selectable; fixed scaling is the option for a shared
calibration across images. This changes display colors only: raw downloads,
scalar scores, native map calculations and benchmark acceptance are unchanged.
CVVDP here exposes the CPU port's masked-error map, not its video raw heatmap.

The Pages check now renders the actual demo's Zensim, Butteraugli and CVVDP maps
and requires more than 32 colors with no single color covering half the image.
This catches both a missing map and the previous widespread saturation, beyond
the existing finite-array and native-output checks. The local Chromium check
passed with run-heavy peak RSS 0.18 GiB.

Source `01b087df57db76472cc23639f962e4bf41939997` was freshly built from
the pinned snapshot, deployed to https://metrics.imazen.org/ and checked live.
The build used run-heavy peak RSS 0.62 GiB; the live Pages check used 0.18 GiB.
Live screenshots showed localized map detail. The demo's native map minimum,
median, 99.5th percentile and maximum were unchanged before/after this display
fix. Browser qualification still checks all 38 scalar routes, native fixture
outputs, full-HD completion, upload recovery and cancellation. Subsequent
metric commits had reinstated the older unresolved lock; the pinned snapshot
regeneration was retained again in `01b087df`.

## Explicit Zensim model selection

The viewer defaults explicitly to generation B, the pinned library's current
codec-target SDR profile. Its model selector also offers candidate C, fast D,
legacy A and the legacy linear previews v0.1/v0.2. One selection controls the
Zensim scalar and, where supported, its Zensim map; results name the selection. Changing models
invalidates the prepared reference cache, and the CLI snippet names the model.
The existing browser constructor remains equivalent to B; `Reference.with_profile`
is the additive browser entry point for explicit SDR selection. Unknown models
and HDR profiles are refused through this opaque sRGB8 ingress.

The CLI built-in resolver also accepts D and includes it in its model inventory,
so the viewer's generated D command selects the same native model.

The pinned C diffmap route fails the native feature-vector contract. C is labeled
scalar-only: its canonical full-pair scalar fold uses no live stop token or cached-reference
route, and cancellation
disposes the worker. Requesting a Zensim C map fails explicitly; another map
metric requires the user's own selection. B/D/A/previews retain native maps.

On the textured 64×64 fixture, all six browser scalars matched the direct native
oracle exactly. All five supported model maps also matched exactly. Worker/UI
checks cover switching models back to B, naming the score/map model, explicit
C-map refusal and recovery, and refusal of HDR models on SDR ingress. The CLI
D resolver and existing invalid-model controls passed four native tests.

Targeted library/oracle clippy passed with warnings denied. All-target clippy
still finds existing test-only lints in the shared CLI sources (test-module
placement and constant `chunks_exact` calls); this change does not claim those
pre-existing lints are resolved.

Source `9e45737c04a84fc213f5d5edaae940c1e79ba307` was deployed and its
model-selection/native-oracle controls passed on https://metrics.imazen.org/.
The cached-reference C scalar route also failed the feature-vector contract;
C uses the direct native full-pair scalar route. No other model supplies its
score or map. Native D resolver tests used run-heavy peak RSS 1.01 GiB;
the live browser check used 0.18 GiB. Featureless browser compilation passed.

## Startup versus comparison timing

Worker startup is measured on the main thread from worker creation until its
ready message arrives. It includes the worker module loading/bootstrap and
message transport. Separate worker-clock intervals record WASM load/compile/
initialization and Rayon pool initialization; these are startup subintervals.
They are shown independently and never added to a comparison's compute time.

The compute tile measures the comparison inside the ready worker: reference
preparation or cache verification, scalar scoring and spatial-map construction.
The visible phase breakdown names all three. Map time includes the native map
API's own work (including any repeated scalar calculation in that API).
Request wall time is measured on the main thread from sending the comparison
until receiving its final result, including data cloning, scheduling and message
transport; it excludes subsequent canvas rendering. Upload decoding and JPEG
encoding are separate jobs and are not included in comparison time.

Controls require finite nonnegative durations, phase intervals contained in
the comparison total, cold preparation versus warm cache verification, and
separate startup/compute text after cancellation and model changes. These
interactive durations do not qualify fleet benchmark acceptance.

Source `614994a5d4c391d568e58d4b06244c7d7d849b32` was freshly built,
deployed and checked on https://metrics.imazen.org/. Its live controls passed,
including cold versus warm reference timing and model/cancellation recovery.
The build used run-heavy peak RSS 0.62 GiB and the live check 0.18 GiB.

The earlier model-selection CI run 38386150704 failed exact native/WASM equality
for legacy Preview v0.1, despite the local fixture matching. Timing changes
keep that assertion intact and report the scalar values and maximum map delta
on failure. The cause and cross-host envelope are not established by the local
checks; no parity tolerance has been broadened.

## Unified map exports, 2026-10-14

The browser map selector reads the compiled module's supported-map catalog.
FSIM/FSIMc luminance similarity, HaarPSI similarity, VSI quality contribution
and MS-SSIM level-0 SSIM are exposed alongside the existing maps. The common
`Metric::evaluate()` / `Evaluation::map()` surface supplies raw samples, polarity and cell geometry
for these metrics and GMSD. Samples retain their native dimensions and values.
FSIMc's native map is the luminance quantity shared with FSIM; it does not
include the scalar's chroma factors.

Rendering follows each cell's declared origin and step. MS-SSIM's valid-window
map starts at (5,5), leaving a five-pixel border without samples; map view marks
that region, overlay preserves its image pixels, and hover reports no sample.
Subsampled maps keep their native ownership grid, including odd image edges.
Legacy exports without unified geometry retain their existing placement.

VSI's native `qualitymap` includes saliency weights. The 257×253 native identity
fixture spans exactly 0 to 1 across 65,021 samples. Visualization shows weighted
quality contribution directly and labels higher values accordingly. Inverting
it as `1 - similarity` would incorrectly paint low-saliency identity regions as
errors. Raw downloads and metric scalar calculations remain unchanged.

HaarPSI's local map divides by its native window weights. Zero-weight windows
produce NaN even when the pooled scalar is valid. Those samples remain NaN in
raw downloads; map view hatches them, overlay preserves source pixels, and hover
reports an undefined native map value. Infinities and other metrics' NaNs still
refuse the map.

The Pages check exercises all five new selector choices on the real demo,
odd-size cell geometry, the cropped MS-SSIM border and mixed flat/textured
HaarPSI windows. FSIM, FSIMc, GMSD, HaarPSI, MS-SSIM and VSI maps match the native
oracle exactly on the 64×64 fixture; existing score/model, cancellation,
worker isolation and timing checks also pass. This validates CPU browser
scoring, independently of benchmark resource admission and GPU acceptance.
