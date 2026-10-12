# Native and browser checkpoints

A checkpoint is an immutable source commit plus target-specific executables,
locked dependencies, build commands and SHA-256 manifests. Native packages ship:

- `zenmetrics`, with the CPU IQA registry and the selected GPU backend compiled in;
- `zenbench`, pinned independently in `ci/checkpoint-profiles.json`;
- `zenfleet-worker` and `zenfleet-ctl` for the existing fleet job protocol;
- `drivers/latency` and `drivers/memory`, plus their separate build identities;
- the observation adapter, Windows shim, compiled metric catalog, lockfiles and sibling pins.

The memory driver contains allocation instrumentation. Use the ordinary driver
for timing. The full profiles enable codecs, native HDR routes, orchestration and
feature jobs. CUDA and WGPU are separate binaries. The CPU-only i686 package has
no GPU route. `--all-features` combines software, experimental and oracle features;
the reviewed shipping profiles enumerate the intended capability set instead.
GPU libraries/drivers, external model assets and platform runtime dependencies
remain runtime requirements. Compilation is not hardware qualification.

## Build and release

The `Checkpoint binaries` workflow tests and builds Linux x64/ARM64, Windows
x64/ARM64, macOS ARM64 and i686 Linux through cross. It runs CLI all-targets
and doc tests, then the IQA study functional controls. Browser CI runs Chromium
workers and compares a textured raw-map fixture exactly with the native library.
Each package records the immutable source, compiler, features and dependency
hashes. The workflow publishes CI artifacts; it does not publish a release.

Manifest file paths use forward slashes on every host, matching tar archive
members. Archive controls exercise Windows path conversion, nested-file
verification and refusal of changed payloads before packaging.

For a local native build, use the CI-pinned snapshot and the pinned runner:

```sh
python3 scripts/checkpoint/package.py --profile full-wgpu \
  --target x86_64-unknown-linux-gnu --source FULL_SOURCE_SHA \
  --runner-source /path/to/pinned/zenbench --out /path/to/new/checkpoint
```

After CI succeeds, download and verify the exact release assets:

```sh
python3 scripts/checkpoint/prepare-release.py --run CHECKPOINT_RUN_ID \
  --source FULL_SOURCE_SHA --out /path/to/new/release-assets
```

The preparer refuses PR-only commits, missing or expired targets, failed or
skipped checkpoint jobs, a source absent from the default branch, and non-green
CI/native-protocol runs on that source. It verifies the archive and every
manifest-bound file without rebuilding, tagging or publishing.

Run heavy builds with the workspace resource wrapper. Output directories must be
new. No `target-cpu=native` flags are accepted. The package has runtime dispatch.
After every platform is green on the release commit, review the README and the
checkpoint manifests before approving a checkpoint release. Tag only that tested
commit. Release assets must preserve executable permissions and carry SHA-256
checksums; use the archived CI artifacts rather than rebuilding after approval.

## Deploy IQA comparisons

Install the target-matching checkpoint through the existing fleet deployment.
Set `ZENBENCH_BIN` to its pinned runner and `ZEN_EXEC` to the packaged adapter
(on Windows, its `.cmd` shim). Supply Python, scoped object-store credentials,
`s5cmd` for remote transfer, durable scratch and a shared host coordination mount.
Run native architecture and adapter controls before enrolling a worker.

Use [BENCHMARK_STUDIES.md](BENCHMARK_STUDIES.md) to expand the algorithm, input,
size, stride, reference-cache and thread matrix into immutable cases. Point the
study's latency/memory artifacts at the packaged binaries and build JSON files.
Generate correctness contracts before staging the bundles. Then follow
[BENCHMARK_FLEET.md](BENCHMARK_FLEET.md): stage by content hash, declare every
machine/domain/replicate, execute through the existing worker, collect every
attempt and analyze accepted observations. A captured busy/refused observation
is retained evidence, not an invitation to select a faster retry.

GPU cases require the requested hardware adapter and backend. Software adapters,
CPU execution and adapter mismatch are refused before metric work. Shipping all
implementations never permits a benchmark to switch implementations silently.
Functional protocol CI uses declared placement suitable for its fixtures; real
resource-admission validation runs separately on an identified target with
unchanged acceptance criteria. Both retain refusal outcomes.

## Threaded browser package

The release checkpoint provides zensim codec-target scoring and raw maps.
The current website additionally selects the native SDR CPU metric catalog and
decodes JPEG, PNG, WebP and JPEG XL with Zen codecs. It generates JPEG candidates
with zenjpeg and supports split, blink, pan, zoom, Turbo/Viridis overlays, hover
inspection, and raw-map download. PNG/WebP/JXL alpha is preserved; scoring rejects
non-opaque pixels. ICC metadata is not applied: use composited sRGB sources.

The module requires a secure context, COOP `same-origin`, COEP `require-corp`,
SharedArrayBuffer, SIMD128 and relaxed SIMD. Unsupported configurations fail.
No alternate metric backend is substituted. CVVDP names its display explicitly.
The zensim cached-reference route names its `codec_target` profile. Native
DSSIM scale-0 SSIM and GMSD maps retain their sampling grids and similarity
values; only visualization applies `1 - similarity`. Butteraugli and CVVDP
return native error maps. PSNR returns per-pixel normalized RGB squared error.
SSIMULACRA2 has no spatial-map export in the current pinned port.

Browser controls compare native map outputs on a textured 64×64 fixture,
require exact zensim score/map repeatability and PSNR/GMSD map parity, exercise
all selectable metrics on 256×256 inputs, decode an independently verified Zen
JPEG, reject malformed JPEGs, recover on the next upload, preserve PNG alpha,
and refuse non-opaque scoring and unisolated worker startup. Measurements and
limits are in [the viewer validation record](BROWSER_VIEWER_2026-10-13.md).
They do not establish all-image parity or fleet benchmark acceptance.

Animated spatial steering needs a codec route that consumes spatial control,
not merely a global quality slider. The inspected JXL route offers an internal
zensim feedback loop; the reviewed wrapper did not expose an external map input.
JPEG and AVIF need separate verification of spatial input and encoded output.
No steering or realtime claim is made by this checkpoint. Qualification should
bind each frame's source, controls, codec revision, decoded pixels, score/map and
measured end-to-end latency before enabling an animated steering mode.

## Local qualification record (2026-10-12)

The Linux x64 WGPU package at source
`1e1d1b40637be0e0ac9037d48fcefbac18afa3ce` passed CLI all-targets and doc tests,
38 SDR and two native HDR functional variants, and both shipped-runner IQA
bundle controls. The package build used `run-heavy --mem 12G --jobs 4`; its
reported peak RSS was 4.90 GiB. Archive manifests and mismatched-source refusal
also passed. These are local packaging controls, not fleet performance results.

The threaded browser control used Chromium 153.0.8010.12 and two actual Rayon
workers. Its textured fixture matched the native raw map and scalar exactly;
repeatability, alpha refusal and missing isolation refusal passed. Its qualified
source was `648d5e7a14a7b2371765f8b585a0374e1608196e`. Final release artifacts
must be rebuilt by CI at the reviewed release source.

[Native protocol CI run 38257659558](https://github.com/imazen/zenmetrics/actions/runs/38257659558)
refused a macOS Intel fixture as `busy` before execution. The fixture declared
unrestricted CPU placement and a two-second admission deadline; unrestricted
placement still preserves admission. The refusal does not establish a protocol
failure. Retain the receipt, packet and event stream when diagnosing this class;
never turn a refused observation into an accepted benchmark. The checkpoint
release requires a green native-protocol run under the current matrix.
macOS Intel was removed from all matrices at owner direction; macOS ARM64
remains a required checkpoint target.

For i686, `cross` mounts the common ancestor containing the complete pinned
sibling workspaces. Mounting only dependency member directories loses inherited
workspace metadata. `just checkpoint-i686-vmaf-check` exercises that mount and
VMAF’s portable SIMD generation on the 32-bit target. This is a compile check;
it does not establish numerical parity on i686.

Native packaging requires the caller to declare `ZENMETRICS_TEST_GPU=present`
or `absent`. GPU-present hosts retain both butteraugli parity assertions at
5e-2. GPU-absent hosts require failure, no stdout score and a backend-unavailable
diagnostic. This functional test contract does not qualify hardware GPU
benchmark observations or alter resource admission.

## Attach files to GitHub Releases

Run the manual
**Publish checkpoint release** workflow from `master`. Supply the successful
Checkpoint binaries run ID, its full source SHA, a new
`vX.Y.Z-checkpoint.ID` tag. README sign-off is required for crates.io
publication, not for these GitHub checkpoint attachments.

The publisher repeats all green-CI and archive
verification gates before tagging. It attaches all eight archives, their eight
SHA-256 files, and `release-assets.json` to a GitHub prerelease. The published
asset index identifies archives by their release filenames. Existing tags
and assets are never overwritten. It does not publish a crate or a container.
Use `publish-release.py --dry-run` with the same inputs to download and verify
the assets and print the publication command without creating a tag or release.

The i686 test builder includes Python for the existing distortion-worker
protocol fixtures and mounts the caller’s durable `TMPDIR`. Its base image is
pinned by digest. Linking is serialized on this target to bound build memory;
all existing functional and numeric assertions still execute. The build image
is a CI tool, while distributed checkpoint files remain GitHub Release assets.

The separately pinned `runner-source` checkout is excluded from this Cargo
workspace. It keeps its own dependency lock and does not join the metric graph.

## Browser website

The interactive viewer runs on https://zenmetrics-site.pages.dev/; its custom
domain is https://metrics.imazen.org/ (activate with a proxied CNAME to
zenmetrics-site.pages.dev). `.github/workflows/site.yml` builds the matching
frontend and shared-memory WASM from the same commit and CI-pinned siblings,
including zenflate main. Cargo and wasm-bindgen tool caches avoid repeated
installation and dependency compilation. `site-source.json` records the source
commit and artifact hashes. Immutable release archives remain supported by
`prepare-pages.py --archive`; development website builds use `--build`.

`web/site/_headers` supplies COOP `same-origin`, COEP `require-corp`, and public
asset CORS. These headers enable SharedArrayBuffer and Rayon workers directly.
The browser check requires a real two-thread comparison, finite raw map values,
repeatable results, and refusal on an unisolated control server. It runs before
and after deployment; benchmark acceptance remains unchanged.

GitHub deployment requires the encrypted `CLOUDFLARE_API_TOKEN` secret with
Cloudflare Pages Edit permission for the configured account. R2 remains separate
artifact storage. No rotating Wrangler login credential belongs in CI.

Without that secret, the workflow builds and verifies the viewer but
reports that deployment is not configured. Production deployment and its live
browser check run only when the credential is supplied.
