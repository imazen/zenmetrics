# libjxl adapter: `rgb-input` timed scope

`libjxl_compute.cpp` takes an optional trailing **scope** argument:

```
iqa-jxl <metric> <w> <h> <ref.rgb> <dist.rgb> <threads> <mode> <samples> [scope]
```

- `native-buffer` — default (`argc 9`, or passed explicitly). Unchanged
  original staging: RGB file read, `ImageMetadata`, worker pool,
  `ImageBundle`s, both `BufferToImageBundle` conversions and the
  `JxlButteraugliComparator` are built once before the loop; the per-call
  clock covers the metric plus `ImageF` release.
- `rgb-input` — opt-in (`argc 10`). Only the raw RGB vectors and the
  reusable `ThreadPoolInternal` remain outside the clocks. Each iteration
  (warmup included) times, inside one RAII block: `ImageMetadata` setup,
  both `ImageBundle` constructions, both `BufferToImageBundle`
  conversions, a per-call `JxlButteraugliComparator` (butteraugli only —
  ssim2 never uses one), the identical metric call, and the release of
  the map and every per-call object before `steady_clock::now()` stops.

Unknown scope strings are rejected; `argc` other than 9 or 10 is a usage
error. Both checks run before any file is opened.

## Output contract

- `VERSION libjxl-v0.12.0` — same literal in both scopes (it names the
  library, not the scope). Emitted by a shared `emit_header`:
  `native-buffer` calls it at the original point — after staged
  metadata/bundles/conversions/comparator, before the sample loop (so a
  staging failure still prints nothing, as before); `rgb-input` calls it
  before its loop.
- `SCOPE rgb-input` — emitted only under `scope=="rgb-input"`, inside
  `emit_header`, after VERSION. The `native-buffer` cohort output gains
  no new line; it is not relabeled.
- `SAMPLE <ns> <score>` — unchanged shape; warmup (`i=0`) suppressed in
  both scopes. SAMPLE carries measured times, so outputs are never
  byte-identical across runs; no cross-scope output or score
  equivalence is claimed.
- `UNSUPPORTED_MAP` — `ssim2` + `score+map` still returns 0 before reads
  in both scopes.

## What `rgb-input` adds to the clock (and what it does not)

Timed now: metadata init, bundle construction, both conversions, the
per-call comparator (butteraugli), the metric, map + per-call object
destruction. Still untimed: process start, RGB file reads, pool creation,
`ButteraugliParams`/`JxlGetDefaultCms`/`NoMemoryManager` binding, and the
pool's own lifecycle. `ImageBundle` holds a borrowed `ImageMetadata*`
(`lib/jxl/image_bundle.h`), so metadata is declared before bundles in the
same block — destruction order keeps the borrow valid and inside the clock.

Metric invocation, intensity target (255 ssim2 / 80 butteraugli),
`hf_asymmetry=1`, `xmul=1`, `intensity_target=80`, thread policy and the
shared `run_metric` lambda are identical in both scopes.

## Source/build pin for this extension

- Adapter source: `benchmarks/iqa_compete_2026-10-16/libjxl_compute.cpp`
- libjxl source: pinned v0.12.0 tree (`source.tar.gz` sha256
  `03e9be69a30be4011f559da75328b6d7cea8ad921fabfbd551ce10bf45cdc992`,
  tag `v0.12.0`, codeload.github.com/libjxl/libjxl)
- Build: `build_libjxl.sh` → CMake Release, `-O3 -DNDEBUG -std=gnu++17`,
  gcc 11.4.0, Ninja; static jxl/jxl_threads/jxl_extras/jxl_gauss_blur.
- First build under this extension produced `iqa-jxl` sha256
  `523507698cb484bf6f5499ccb0f924cd376abcf3beaca7a47de13604c2b0df61`
  (control build, `~/tmp/swe2opt/native-jxl-evid-20261019T0827/build/`).

## Explicitly not claimed

Only that RGB/bundle preparation is now *inside* the measured region for
the `rgb-input` scope. Full cold/warm equivalence, numerical parity,
backend parity, `native-buffer`↔`rgb-input` cross-scope comparability,
pool-lifecycle attribution and caller-side staging costs remain
**unqualified** — this patch is a timing-boundary extension, not a
matched/qualified cohort. No timing or score data is produced or claimed
here.
