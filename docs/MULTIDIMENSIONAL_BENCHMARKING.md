# Comparative metric benchmarking across size and hardware

Source audit: 2026-10-12. Design proposal, not a completed performance study.
Audited zenmetrics revision `d666f20d` and zenbench revision
[`ac675d3a`](https://github.com/imazen/zenbench/tree/ac675d3a3f5334f4be6749a61b37995c9f3d34fb).
No benchmark campaign, target-host enrollment or metric build was run for this
investigation. Counts below come from source or proposed manifest arithmetic;
there are no new timing or memory measurements.

## Missing capabilities and decisions first

| Gap | Consequence / next action |
|---|---|
| CLI has 46 names; fleet IQA driver has 29 cold SDR CPU names and four warm names | Generate a capability inventory and explicit unsupported cells; extend the existing driver, rather than treating 29 as all metrics. |
| Fleet observations are descriptive; no paired-plan zenfleet adapter | Integrate `zenbench pair` through the existing benchmark job lifecycle before distributed Confirm. Do not reinterpret `fleet-diff` as inference. |
| Effective metric SIMD tier is not observed | Label current runs runtime-dispatched. Tier-specific studies need actual dispatch evidence and a supported forcing interface per metric. |
| HDR, GPU, decode-inclusive and research workloads have no common fleet driver contract | Add separate versioned contracts with color/viewing/output/phase semantics. Do not reuse the RGB8 label. |
| Requested heap, RSS, private commit and GPU memory measure different quantities | Keep separate counters and instrumented passes; no single cross-platform “peak RAM” column. |
| External pair/fleet results cannot use the suite chart exporters | Add a packet-aware plotting input and renderer with actual intervals, missing cells and provenance. |
| Numeric-axis fits are unimplemented | Implement the separate [zenbench issue 39](https://github.com/imazen/zenbench/issues/39) analysis contract, with an independent numerical oracle and replay gates. |
| Physical hardware and study permissions are not enrolled | AMD Zen 3/4/5, Intel, M4 Pro and Windows ARM64 are requested testbed classes, not an available fleet claim. Resolve inventory and permission at enrollment. |
| Hosted CI resource load can prevent functional fixtures from executing | Separate deterministic protocol tests from native resource-admission validation; preserve production refusal rules. |

Decisions to freeze before measurement: the representative input pairs, allowed
testbeds/time windows, practical A/B margins and timing-error bound, whether
research metrics are in scope, and the exact output products to retain. Existing
PSNR names are inventoried below, but they are not selected for the initial
perceptual-metric study. None of these decisions blocks building the proposed
inventory, manifest validator or plotting reader.

## Source inventory and correctness contracts

The CLI authority is [`MetricKind`](../crates/zenmetrics-cli/src/metrics/mod.rs)
and `run_metric_display`, not a README count. Its enum has **46 explicit names**:
40 CPU names, including two HDR-only names, and six GPU twins. The umbrella's
[`MetricKind`](../crates/zenmetrics-api/src/metric.rs) has eight kinds and is
not the full CLI registry. Availability depends on the binary's Cargo features.
The current CLI [`cpu-metrics`](../crates/zenmetrics-cli/Cargo.toml) bundle
includes native CVVDP, IW-SSIM and HDR-VDP along with the classical ports; older
four-metric bundle descriptions do not describe this audited manifest.

In the following table, RGB8 means the CLI's decoded, packed, encoded-sRGB
three-channel input pair. Conversion to the metric's internal representation
belongs to its dispatch call. Oracle names identify validation sources, not
claims that this investigation reran or certified them. Exact fixture gates
and reference revisions remain in [METRIC_PROVENANCE](METRIC_PROVENANCE.md),
[DIVERGENCES](../DIVERGENCES.md) and each crate's tests.

| CLI names (complete inventory) | Input and output contract | Current implementation / oracle |
|---|---|---|
| `ssim2`, `ssim2-gpu` | RGB8; SSIMULACRA2 score, higher better | `fast-ssim2`; C++ SSIMULACRA2 reference. GPU twin has same-metric parity tests. |
| `butteraugli`, `butteraugli-gpu` | RGB8; max distance and p-norm 3, lower better | Sibling `butteraugli`; libjxl reference. CLI can return both; current fleet driver keeps only primary max score. |
| `dssim`, `dssim-gpu` | RGB8; dissimilarity, zero identity | `dssim-core` called directly on CPU; GPU parity against CPU. |
| `zensim`, `zensim-gpu` | RGB8; learned score, higher better; profile/bake matters | Sibling `zensim` defines the metric; GPU same-profile parity. No external author oracle. |
| `cvvdp`, `cvvdp-gpu` | RGB8 plus declared display; still-image JOD | Native `cvvdp` / GPU twin; pycvvdp 0.5.7, `crates/cvvdp/tests/it/upstream_parity_extended.rs`. Video is a distinct temporal workload. |
| `iwssim`, `iwssim-gpu`, `iwssim-piq` | RGB8 → luma; similarity; rounded BT.601 versus unrounded YIQ ingress for piq | `iwssim`; Python-IW-SSIM reference in `tests/parity_python.rs`; direct author MATLAB validation remains absent. GPU coverage is backend-specific. |
| `hdrvdp`, `hdrvdp3` | Absolute linear RGB nits; v2 quality/detection products, v3 Q_JOD; explicit viewing conditions | `hdrvdp` and `hdrvdp::v3`; official 2.2.2 / 3.0.7 fixtures in `tests/golden_2_2_2.rs` and `tests/v3_reference.rs`. Reject SDR RGB8. |
| `gmsd` | RGB8 → BT.601 luma, half-resolution gradient map; deviation, lower better | `gmsd`; libgmsd / authors' GMSD design. |
| `mdsi` | RGB8 → L/H/M; distance, lower better | Paper-derived `gmsd` module; `src/mdsi_tests.rs::author_score_gate` checks external author scores under an explicit caller contract. Gate data and execution were not verified here. |
| `ms-gmsd`, `ms-gmsdc` | RGB8 → multiscale luma / chroma; distance | Paper-derived `gmsd` modules; author-software parity absent. |
| `haarpsi`, `haarpsi-y` | RGB8 → YIQ / unrounded BT.601 Y; similarity | `haarpsi`; authors' HaarPSI.m goldens. |
| `fsim`, `fsim-y` | RGB8 → YIQ / luma; FSIM and FSIMc / FSIM-Y similarities | `fsim`; authors' FR_FSIMc.m goldens. |
| `vsi` | RGB8 → Lab/opponent and saliency pipeline; similarity | `vsi`; authors' VSI.m goldens. |
| `msssim`, `msssim-libvmaf`, `ssim-libvmaf` | RGB8 → metric-specific luma; similarity; libvmaf variants use studio BT.601 and different pyramid/pooling conventions | `msssim`; MATLAB and vendored libvmaf FFI, `tests/ffi_libvmaf.rs`. |
| `ssim` | RGB8; Gaussian-window similarity averaged over R/G/B | CLI `metrics/classical.rs`; author-software parity absent. |
| `nlpd`, `nlpd-iqa` | RGB8 → RGB planes / quantized BT.709 Y; different pyramid/pooling conventions; distance | `nlpd`; author PyTorch / IQA_pytorch configuration reference. These are distinct quantities. |
| `vif`, `vifvec` | RGB8 → luma; pixel-domain / wavelet vector-GSM information fidelity | `vif`; authors' vifp_mscale.m / vifvec.m + matlabPyrTools goldens. These are distinct algorithms. |
| `mad` | RGB8; distance plus high/low strategy indices | `mad-iqa`; author MATLAB/C-mex goldens. |
| `mdctpsnr` | RGB8; dB plus JND conversion surface in crate | `mdctpsnr`; author's compiled dctpsnr tool; platform-specific floating-point oracle limits apply. |
| `psnrhvs`, `psnrhvs-y`, `psnrhvs-daala` | RGB8 → RGB / luma / studio YUV; HVS and masked dB products or Daala variant | `psnrhvs`; authors' psnrhvsm.m / libvmaf FFI. |
| `vmaf`, `vmaf-neg`, `vmaf-4k`, `vmaf-v1` | RGB8 → studio BT.709 YUV420; distinct model scores; even dimensions required | In-tree `vmaf`; libvmaf 3.2.1 feature/frame/NEG parity, `tests/ffi_fusion.rs`. |
| `psnr`, `psnr-y`, `psnr-y601`, `psnr-y-studio601`, `psnr-y-libvmaf` | RGB8; RGB / full BT.709 / full BT.601 / studio BT.601 / studio BT.709 dB | CLI `metrics/classical.rs`; analytic definition, identity yields infinity. Inventory only in initial proposal. |

CPU feature gates are `cpu-metrics`, `cpu-cvvdp`, `cpu-iwssim`,
`cpu-hdrvdp`, `cpu-gmsd`, `cpu-psnrhvs`, `cpu-haarpsi`, `cpu-fsim`,
`cpu-vsi`, `cpu-msssim`, `cpu-vif`, `cpu-vifvec`, `cpu-iwssim-piq`,
`cpu-mad`, `cpu-mdctpsnr`, `cpu-vmaf`, as wired in the CLI manifest.
The six twins require their `gpu-*` metric gate and a runtime gate such as
`gpu-cuda` or `gpu-wgpu`; HIP/CubeCL CPU feature declarations are not proof
of working metric/platform combinations. Native CPU fallback must be detected
and labeled as a changed backend, never merged into a GPU cell. Auto selection
is unsuitable for a study that intends to compare explicit implementations.

Research scope is separate: [`margarine`](../crates/margarine/README.md) is an
excluded standalone workspace, not a CLI name. Its frozen vendor oracle,
feature experiments and provenance need their own bundle. `dvifmish` is not a
member/package at the audited checkout: its historical mention does not make
it an executable benchmark target. No neural-runtime LPIPS/DISTS/TOPIQ target
was found in this registry. Maintain an explicit absent/unsupported inventory.

### Benchmark entry points actually present

| Entry | What it measures / limit |
|---|---|
| [`new-metrics-wall`](../benchmarks/heaptrack/drivers/cpu_profile/src/bin/new_metrics_wall.rs), `METRICS`, `score_once` | 29 cold SDR CPU names via real `run_metric_display`; primary column only. Suite `lat` versus `par1/par4/par8` includes pair-level concurrency, not merely metric-internal threading. Synthetic/photo content; photo mode mirror-tiles, so it is not a faithful size-resampling study. |
| [`fleet_case::run`](../benchmarks/heaptrack/drivers/cpu_profile/src/fleet_case.rs) | Exact operations, fresh process, RGB8 packed/padded input, explicit Rayon threads; cold list as above; warm only ssim2/butteraugli/dssim/zensim. Packing/pool creation are preparation. Heap build marks `timing_usable:false`. |
| `cpu-wall`, `loop-wall` in [cpu-profile manifest](../benchmarks/heaptrack/drivers/cpu_profile/Cargo.toml) | Selected six-metric packed ingress versus encoder-search planar linear-f32 warm-reference boundary; different workloads. |
| [`cpu-metrics-1t8t`](../justfile), [`cpu_metrics_1t8t.sh`](../scripts/cpu_metrics_1t8t.sh) | Existing historical CPU wall/RSS study; useful execution/analysis precedent, not the full hardware matrix. |
| Per-crate benches: `cvvdp/{csf_wall,kernel_tiers}`, `iwssim/kernel_tiers`, `vmaf/{backends,arm_stages}`, `nlpd/nlpd`, `hdrvdp/pipeline` | Algorithm/kernel-specific suites; boundaries and fixture identity must be normalized before comparing cost. HDR bench covers v2; no common v3 fleet route. |
| GPU examples/benches under the six twin crates | Warm, strip/full, batching and backend-specific probes; no unified phase/input/fleet contract. |
| Orchestrator `bench.rs` and `bench_worker` / `multiwarm_bench` | Runtime chooser/calibration workloads. The study must control backend instead of allowing performance-based selection. |

The fleet driver's missing CPU names are `vmaf-neg`, `vmaf-4k`, `vmaf-v1`,
`psnr-y601`, `psnr-y-studio601`, `psnr-y-libvmaf`, `psnrhvs-y`,
`haarpsi-y`, `fsim-y`, `hdrvdp`, `hdrvdp3`; all six GPU names are also absent.
Its finite-primary-score contract additionally cannot represent legitimate
identity infinity or degenerate NaN products. Correctness controls must describe
those cases explicitly instead of coercing them into a timing value.

Correctness has two layers. First qualify each implementation against its
declared oracle, full products and supported layout/size contract. Then qualify
latency versus memory builds and A/B outputs on the actual study inputs. Exact
score bits suit a same-target same-code instrumentation control. CPU/GPU or
cross-tier equality may need an existing documented numerical parity gate,
with full score/diffmap checks; do not widen gates to admit a cell. The current
fleet exact `score_bits` marker requires target-specific expected values or a
new validated checker route for legitimate cross-platform differences. Markers
derived from the same buggy executable establish repeatability, not accuracy.

Unlike metrics must never compare outputs by byte equality. Cross-metric cost
means time/resources for specified products on a shared pair; it does not mean
equivalent perceptual accuracy. A quality/accuracy study needs held-out subjective
labels and its own protocol. This investigation proposes no quality ranking.

## Purposeful initial matrix, then expansion

Start with six explicit CPU implementations: `ssim2`, `butteraugli`, `dssim`,
`zensim`, `cvvdp`, `iwssim`. Choose four textured reference/distorted pairs:
one photo, one screen/text, one line-art and one mixed-content pair, drawn from
hash-verified corpus manifests. This is a pilot, not a content-population sample.
Do not use smooth-gradient or tiled-photo performance results as representative
content. Choose sources large enough for the largest derivative, never upscale;
freeze resampler/version/kernel, crop rectangle and distortion provenance.

Initial size grid: 256², 512², 1024², 2048², 4096²; packed RGB8 sRGB;
one thread; explicit cold pair scoring; primary output product. The 256² floor
accommodates IW-SSIM's `min(width,height) >= 176` contract. Add 32² and 64²
only for metrics that admit them, with explicit unsupported IW-SSIM rows.
Choose an even-size grid for VMAF. Add padded and odd-size cells as correctness
controls first, not a second full performance Cartesian product.

Run the pilot on **one authorized quiet native testbed**, then expand the same
frozen cases to these six hardware/OS classes. Physical enrollment is pending;
the local source audit only checked `hostname`/`nproc`, identifying an Intel
audit host with 20 logical CPUs. It made no remote availability check.

| Requested factor | Enrollment and capability requirement |
|---|---|
| AMD Zen 3, Zen 4, Zen 5 / Linux x86-64 | Three separate testbeds; capture CPU model, native ISA, logical/physical cores, SMT, RAM and actual cache domains. Zen generation alone is insufficient. On asymmetric-cache CPUs measure explicit domains separately. |
| Intel / Linux x86-64 | Retain exact model and P/E/cache domain; one-core baseline uses a verified selected class. |
| Intel / Windows x64 | Prefer the same physical CPU used on Linux for an OS contrast; record firmware/power/runtime differences. Otherwise this is a combined hardware+OS contrast. |
| Apple M4 Pro / macOS ARM64 | Verify native architecture and performance-level/QoS policy. Current Apple placement is a hint, not guaranteed P-core confinement; retain that limitation. |
| Windows ARM64 | Native binary and OS topology oracle required; never measure x64 emulation under an ARM64 label. Current zenbench native ARM64 class discovery exists, but this does not certify every IQA crate on that target. |

For counting, the first cross-platform panel below uses six testbeds: AMD
Zen 3/4/5 Linux, Intel Linux, M4 Pro macOS, Windows ARM64. Intel Windows is
an additional explicitly declared panel, ideally a matched physical machine.
No machine address, serial or private inventory is part of the public manifest.
Resolve capabilities using `zenbench fleet capabilities` and native OS inventory
at enrollment; archive a sanitized CPU/cache/OS observation with its date and
command. Do not infer availability from historical documentation or CI labels.

Expand in this order: remaining supported CPU metric families; four warm APIs;
internal threading at 1, 2, 4 and a recorded class-limited physical-core count;
equal-pixel wide/tall rectangles; dense size points around observed cache
transitions; alternate luma/model products; linear-f32/HDR; GPU full/strip/warm
and device-resident compute. Compile feature-off and feature-on threading
artifacts separately. `RAYON_NUM_THREADS=8` cannot activate absent `parallel`
features; [cpu-profile `metrics-parallel`](../benchmarks/heaptrack/drivers/cpu_profile/Cargo.toml)
is an existing opt-in. Record nested pools and opaque FFI workers independently.

Each row needs metric+implementation revision, all output products, input SHA,
origin/content/derivative identity, width/height/pixels/aspect ratio, color
encoding/bit depth/channels/stride, backend, reuse, operations, thread policy,
SIMD policy and observed tier, build/target/toolchain/lock/features, testbed,
CPU/cache domain, OS/runtime/driver and measurement mode. “Unknown” is data;
it must not become an assumed dispatch tier or native-architecture claim.

## Measurement boundaries and peak-memory meanings

Measure direct library dispatch inside the child. Process isolation enables
fresh allocator/pool state; it does not justify timing CLI startup as metric
latency. Report outer launch/admission/cleanup wall time separately.

1. **Decoded-input score:** the present fleet timer wraps `score_once` or warm
   `Warm::score`; metric-internal conversion/allocation is included. File reads,
   strided packing and global pool startup are preparation. This is the first
   fair cross-metric cost boundary available.
2. **Prepared compute:** preconverted metric-native planes, persistent state and
   prescribed outputs. Requires direct per-crate adapters. Different native
   formats remain visible; this is not the same boundary as decoded RGB8.
3. **Warm reference:** separately report reference preparation, warmup and
   per-candidate scoring. Model a declared N-candidate workflow from measured
   components and validate an actual replay; do not hide preparation amortization.
4. **Decode-to-result:** real files, decoder/color management/orientation/alpha,
   conversion and scoring. Name the decoder revision and storage/cache policy.
   This requires another driver; the current raw-RGB8 route excludes decoding.
5. **GPU:** cold context/JIT, transfers, steady score+readback, and event-timed
   device kernels are separate boundaries. Synchronize completed work; host
   submission duration is not kernel runtime. Record driver, device, runtime,
   memory mode and batching. GPU VRAM and shared unified memory are separate views.

Cold means no retained reference between operations, not necessarily cold OS
page cache, first-ever JIT or empty CPU caches. The current cold operation vector
has no explicit score warmup; its first score may differ. For steady cold-pair
latency add and declare within-child warmup; retain a separately labeled
first-call case. Warm mode currently does one untimed score warmup. Calibrate
operations during exploration and freeze them before Confirm; do not adapt
the work budget to a favorable sample. Resolve short timers using repeated fixed
work, and report the error bound. Use latency per image plus MP/s and ns/pixel;
never report only ns/pixel when fixed overhead or size regimes matter.

| Quantity | Existing coverage | Interpretation / exclusions |
|---|---|---|
| Requested live Rust heap entry/peak/exit and lifetime peak | Zenbench [`HeapProfiler` / `HeapPhase`](https://github.com/imazen/zenbench/blob/ac675d3a/src/heap.rs); all Rust global-allocator threads, including preexisting Rayon pools and cross-thread frees | Absolute live requested sizes; retained input/ref allocations remain visible. Excludes allocator metadata/fragmentation, hidden realloc overlap, direct mappings, bypassing native/FFI allocators, children and GPU memory. Instrumentation serializes allocations and changes timing. |
| Phase peak minus entry | Derived from the above | Additional live-byte watermark, not ownership-based scratch attribution; freeing old memory can mask later allocations. Never sum phase peaks. |
| Allocation traffic | `HeapPhase` requested bytes/counts | Successful allocation and realloc-new sizes; not live heap or RAM. |
| Timing allocator statistics | [`AllocProfiler`](https://github.com/imazen/zenbench/blob/ac675d3a/src/alloc.rs), thread-local snapshots | Does not automatically aggregate workers or measure a process peak. Do not use it as multithreaded peak memory. |
| Linux intercepted heap | Fleet `heaptrack` backend, retained trace and integer global-peak stack contributions | Intercepted Rust/native dynamic allocations; static/custom allocators and mappings may bypass it. Individual stack maxima are not additive; use contributions at the same global peak. |
| Resident high-water | Fleet `os_peak`: Linux GNU time ru_maxrss ×1024, macOS time -l bytes | Whole child OS counter, includes more than heap; not a simultaneous sum over a process tree and not a resettable phase peak. |
| Windows working-set / private-commit high-water | `PeakWorkingSetSize` / `PeakPagefileUsage` queried before child handle closes | Distinct resident and private-commit quantities, not requested heap. Query failure is null with reason. Job-object commit is another quantity, not interchangeable with child commit. |
| GPU allocated/resident/peak | No unified metric fleet instrument found | Add runtime-specific counters with scope/tool version, or null+reason. Driver pools/shared memory obscure ownership; never add host and device counters as total RAM. |

Use fresh processes per declared memory observation. Keep input preparation,
reference preparation, warmup, scoring and reference release phases, joining all
workers at boundaries. Whole-process OS counters still include preparation;
they cannot be subtracted into a scoring RSS peak. A separate process that
stops after preparation gives a separate observation, not a reset of the first
process's high-water counter. Preserve native allocations and mappings through
OS/heaptrack views. For every absent counter retain quantity, null bytes,
unavailable reason, backend/version, coverage and scope. Zero means measured zero.
Never divide a memory peak by operations. Timing passes use the ordinary build;
memory instrumentation passes never supply native latency claims.

## Concrete manifest and transport design

The following is a **proposed study schema**, not accepted input to current
`declare-benchmarks`. Resolve symbolic SHA values and expand this description
into existing bundle specs and observation specs before execution.

```json
{
  "schema": "zenmetrics-study-proposal-1",
  "study": "cpu-size-pilot",
  "inventory_revision": "d666f20d",
  "metrics": ["ssim2", "butteraugli", "dssim", "zensim", "cvvdp", "iwssim"],
  "pairs": ["photo-1", "screen-1", "line-art-1", "mixed-1"],
  "sizes": [[256,256], [512,512], [1024,1024], [2048,2048], [4096,4096]],
  "testbeds": ["amd-zen3-linux", "amd-zen4-linux", "amd-zen5-linux",
               "intel-linux", "apple-m4pro-macos", "native-winarm"],
  "pixel_contract": {"format":"rgb8_srgb", "channels":3, "layout":"packed"},
  "execution": {"backend":"native_cpu", "reuse":"cold", "threads":1,
                "simd":"runtime_dispatch", "products":"primary_score"},
  "observations": {"replicates":3, "passes":["wall", "os_peak", "driver"]},
  "latency_ab_followup": {"blocks":5, "candidates":1, "aa":true,
                          "case_selection":"freeze_after_pilot"},
  "artifacts": {"inputs":"resolve_by_sha256", "builds":"resolve_by_sha256",
                "runner":"resolve_by_sha256", "retain_all_attempts":true}
}
```

These three fleet passes capture supervisor wall, process memory and requested
heap. The ordinary driver's internal `operation_ns` lives in stdout; `wall` is
not its latency. A proposed ingest reader may extract those timings for
descriptive plots, always labeling the boundary and instrumented status.
Statistical A/B follows the separate paired plan. Fleet OS-peak could also retain
ordinary internal timings, but combining observations must not double-count
the same invocation as independent latency evidence.

An existing driver case is executable after supplying real bundle inputs:

```json
{"schema":1,"metric":"ssim2","width":256,"height":256,
 "stride_bytes":768,"reference":"reference.rgb","distorted":"distorted.rgb",
 "pixel_format":"rgb8_srgb","reuse":"cold","threads":1}
```

Bundle `plan.program` names the native driver; args are `["fleet","case.json"]`;
set `RAYON_NUM_THREADS=1`, fixed operations, backend and timeout. Include case,
both images, executable, runtime libraries where needed and symbols for heap
attribution. `expected_correctness` contains the independently qualified
primary score-bit vector for that artifact/case/operations contract.

Expand to [`JobKind::Benchmark`](../crates/zenfleet-core/src/job.rs):
bundle SHA, runner SHA, experiment, machine, domain, planned replicate, backend,
execution timeout and admission budget. Wrap with resource hints and pass to
[`declare-benchmarks`](../crates/zenfleet-ctl/src/bin/benchmark_declare/mod.rs).
The [existing adapter](../scripts/jobsys/benchmark_exec.py) stages hash-verified
bundle files into the content store, capability-routes the machine/domain/runner,
materializes a fresh attempt, executes `fleet-run`, and uploads packet, streams
and receipt. Use existing worker leases/admission and baked images, with native
Windows `.cmd` packaging and macOS binaries. Do not build or install at boot.

Bundle hashes bind bytes and plan, but not automatically source-to-binary
provenance. Add a build manifest: source SHA/dirty state, resolved dependency
revisions and registry checksums, lock/sibling-pin SHA, compiler/LLVM/target,
features, rustflags, profile/LTO/codegen units and executable/library SHA.
Keep the installed runner's byte identity separate from the driver's zenbench
dependency: the CPU harness pins `77707b68`, native fleet CI pins `4f152ef9`,
while this audit of latest renderer/placement source used `ac675d3a`.
Do not attribute latest-main capabilities to every pinned deployment.

Proposed normalized observation row (integer bytes, no fictitious data):

```json
{"study_sha":"SHA","cell_sha":"SHA","job_id":"SHA","attempt_id":"UUID",
 "replicate":0,"packet_sha":"SHA","bundle_sha":"SHA","runner_sha":"SHA",
 "build_sha":"SHA","testbed_sha":"SHA","input_pair_sha":"SHA",
 "metric":"ssim2","backend":"native_cpu","width":256,"height":256,
 "pixels":65536,"aspect_ratio":1,"threads":1,"reuse":"cold",
 "timing_boundary":"decoded_input_score","timing_usable":true,
 "latency_vector_ns":null,"latency_unavailable_reason":"not ingested",
 "quantity":"requested_live_heap_peak","bytes":null,
 "unavailable_reason":"ordinary latency build","scope":"score",
 "status":"declared","inference":"none"}
```

Ingest immutable receipts into Parquet views keyed by study/cell/planned
replicate/attempt, retaining raw invocation/block vectors and correctness.
`Done` means durable observation, not valid performance. Report declared,
unsupported, busy, wrong-output, cancelled, timed-out, transport-failed and
complete counts. Keep a manifest-to-observation coverage table. Retry is a new
attempt of the same planned job, not another declared statistical replicate.
Predeclare attempt selection; never choose fastest. Confirm's failed/incomplete
study cannot be completed by pooling resumed attempts.

Reuse [RUNNING_JOBS](RUNNING_JOBS.md), [PLAN_SWEEPS](PLAN_SWEEPS.md) and
[BENCHMARK_FLEET](BENCHMARK_FLEET.md). Codec plan sweeps can generate immutable
distorted inputs upstream; no re-encode is needed for a metric cost study.
[`benchmark_gc.rs`](../crates/zenfleet-worker/src/bin/support/benchmark_gc.rs)
already expands bundle/runner/nested receipt evidence before latest-wins folding.
Provide original attempt ledgers and archive roots: lossy latest-wins views
cannot reconstruct discarded history. Deduplicate binaries, symbols and pairs
by hash; retain traces outside Git. Commit only a selected compact study record
and pointers/hashes, under the generated-data size policy.

## Inference, busy hosts and CI

For same-metric implementation A/B, use fresh serial children in randomized
blocks on the same compatible machine/domain/input/settings cell. Different
builds are arms; preserve matching format, output products and correctness.
The [external paired estimator](https://github.com/imazen/zenbench/blob/ac675d3a/docs/pair-inference.md)
uses each invocation's arithmetic mean of internal timings, then the block
log(candidate/baseline) ratio. Its reported ratio is the exponentiated mean
log ratio. Internal operations are not independent process blocks; multiple
derived sizes of one origin are not independent images.

Confirm freezes blocks, identities, margins, timing-error bound, family and
accept rule. Include an A/A baseline clone and require equivalence within the
control margin; failure to reject zero is insufficient. Current pair intervals
use Student-t uncertainty over block log ratios with Bonferroni family correction
including selected candidates/cases and A/A, plus the declared bounded timing
error. Record invalid, incomplete and inconclusive outcomes. No adaptive stopping,
resumed pooling or rerun-until-pass. Family correction applies to the predeclared
claim family across a study, not whichever charts were selected after seeing data.

Independent blocks must reasonably satisfy the model; randomization and A/A do
not prove independence or exclude candidate-specific interference. Code layout,
different workload duration, cache carryover or correlated disturbances that
affect only B can pass A/A while biasing B. Inspect order/residuals/raw resources;
repeat a separately declared validation study or independent build layout for
broader claims. Current real-codec acceptance validates protocol plumbing on
small fixtures, not representative IQA performance or universal validity.

Hardware/OS comparisons are descriptive experimental factors, not paired
same-host A/B. Different core counts/cache capacities change the workload's
resource budget. Show both one-thread latency and class-limited throughput
without implying that a 16-core result equals an 8-core result. To isolate OS,
use the same physical testbed and randomized independent session order, recording
power/thermal/firmware/compiler/runtime differences. If machine and OS both
change, identify the contrast as confounded. A single machine per CPU generation
does not estimate a population-wide architecture effect.

Source distinction: current [`fleet_run::execute`](https://github.com/imazen/zenbench/blob/ac675d3a/src/bin/support/fleet_run.rs)
admits at CPU load ≤0.8 and available RAM ≥1 GiB, with boundary observations
and `cleanliness_certified:false`. This is **descriptive observation admission**,
not Confirm's strict before/during/after policy. Pair Confirm requires CPU ≤5%,
no foreign heavy processes, RAM ≥512 MiB and temperature ≤85°C when available;
it requires a complete during-child observation and retains attribution gaps.
Sampling cannot certify the absence of interference between samples. Shared
coordination protects participants, not unrelated software.

Use Iterate for one selected case during edits; its default four blocks with
one candidate+A/A launch 12 children. Confirm's minimum five blocks launch 15.
Long enough fixed internal work is needed for during-child resource observation,
but must preserve the intended call boundary. Busy agent hosts may produce
mostly refusals: reserve an authorized quiet window or idle testbed, retain
busy observations, and report usable-evidence yield. Never relax acceptance to
obtain a verdict. Memory runs can use descriptive fleet admission, visibly
labeled, and cannot certify native latency.

### Functional CI is not a quiet benchmark laboratory

The supplied log for [zenbench CI run 38220119799](https://github.com/imazen/zenbench/actions/runs/38220119799)
was inspected. Its macOS ARM64 `fleet_cli` wrong-output scenario returned
`busy` before execution: resource admission expired with `cpu_load:1.0` and
nine heavy processes. The assertion at `tests/fleet_cli.rs:201` expected
`wrong_output`. The later OS-counter scenario failed acquiring the poisoned
serialization mutex at line 223. That is a resource-dependent protocol fixture
failure followed by a cascade, not a chart-rendering failure. The workflow's
still-running ARMv7 job and temporary rerun refusal were reported by the parent;
this investigation did not independently query that transient state.

Proposed separation, with **no acceptance relaxation**:

- Deterministic tests drive admitted/busy/cancelled observations through a
  test-only observation boundary or controlled fake monitor. Require the
  wrong-output fixture actually execute before asserting its rejection; test
  busy as a separate expected refusal. No test-only hook in release workers.
- Native subprocess/OS-counter/parent-lifetime controls use explicit functional
  mode and emit `not_performance_evidence`. They test real execution, hash
  recovery and cancellation without claiming a quiet measurement. Arrange
  isolated processes so one assertion cannot poison unrelated scenarios.
- Dedicated native admission tests keep real thresholds and require known
  busy-load refusal plus clean-host admission on a provisioned testbed. CI
  environment inability to supply the clean precondition is visible incomplete
  validation, not a graceful success or threshold change.
- Release benchmark Confirm retains current strict resources, placement,
  correctness and completion requirements. Hosted test success is neither
  metric-platform qualification nor production enrollment.

The existing [native fleet workflow](../.github/workflows/benchmark-fleet.yml)
covers Linux, Windows x64/ARM64 and macOS Intel/ARM64 using small synthetic
adapter/worker controls. It does not build and score the complete IQA matrix.
Windows ARM64 topology discovery is now implemented in zenbench's
[`windows_topology.rs`](https://github.com/imazen/zenbench/blob/ac675d3a/src/platform/windows_topology.rs)
and ARM64 branch of `placement.rs`; independent native core/cache/mask controls
are recorded in [CPU placement verification](https://github.com/imazen/zenbench/blob/ac675d3a/docs/cpu-placement-verification.md).
Multiple Windows processor groups and Apple hard P/E binding remain unsupported.
The brief's older ARM discovery limitation is superseded by this source.

## Size curves, fits and chart output

Plot raw per-image latency versus pixels first, with width/height/aspect ratio
available as fields and hover/filter dimensions. Show ns/pixel and MP/s as
secondary views. Use separate panels by backend, boundary, reuse, threads,
build, hardware/OS and cache class. For the same origin, compare equal-pixel
wide/tall inputs to expose row/pyramid effects. Do not pool unrelated settings.

Issue 39 remains open, and no numeric case-axis fitting implementation was
found in the audited pair source. A proposed descriptive fit is
`time_ns = intercept_ns + slope_ns_per_pixel * pixels`, within a predeclared
size regime and fixed factors. Retain the included/excluded cell inventory,
residuals, leverage and prediction error. Negative intercepts or systematic
residuals invalidate a universal fixed-cost interpretation. Cache transitions,
FFT/pyramid size steps and threading thresholds can require separate regimes;
choose boundaries before confirmatory fitting, not by maximizing an effect.
Dense log-spaced points and held-out intermediate sizes are needed before using
fits to predict costs. The five-size pilot is descriptive only.

Repeated sizes from one image and repeated process blocks are clustered.
For exploratory cross-content summaries, use origin-level resampling with nested
process blocks or a declared hierarchical random-intercept/slope model; report
between-origin versus process uncertainty separately. Four pilot origins cannot
support a broad content-population fit. Regression uncertainty must be derived
from the raw clustered observations, not per-cell t interval endpoints. Family
correction/model checks require an independently validated inference contract;
until then label fits descriptive and report no acceptance verdict from slopes.

Acceptance plan for the fit implementation:

1. Independent Python NumPy/SciPy least-squares oracle with checked units,
   analytic constant/linear datasets and fixed-overhead+scaling controls.
   Check coefficients, residuals and included inventory against Rust output;
   synthetic numbers are labeled test data, never hardware measurements.
2. Refuse missing/nonfinite/duplicate axes unless declared distinct replicates;
   refuse mismatched identity/settings, invalid correctness cells and zero
   spread. Show exclusions; preserve aspect ratio as an independent field.
3. Test piecewise/cache-step/aspect-ratio and correlated-origin controls that
   expose bad pooling. Independently validate any cluster uncertainty and its
   coverage/family correction before enabling inferential output.
4. Replay a retained real encoder size grid as required by issue 39, plus a
   real metric size grid. Verify inputs/builds/settings, compare numerical
   results with the oracle, predict held-out sizes and retain failures. No
   encoder campaign is authorized by this document.

Reuse zenbench's existing SVG/HTML themes and suite exports:
`zenbench data read PACKET --format html`, `--save-charts DIR`, and with
`charts`, `--publish-charts DIR --chart-theme light|dark --vertical`.
[`charts::build_series`](https://github.com/imazen/zenbench/blob/ac675d3a/src/charts.rs)
uses suite mean bars. These bars do not draw confirmation intervals; embedded
MAD comments are not visible uncertainty. The packet reader explicitly refuses
pair HTML/SVG, and fleet has its separate JSONL exporter. Do not squeeze external
observations into fabricated `SuiteResult` statistics.

Proposed plot-ready tables: cell inventory; per-invocation observations;
block contrasts with estimator/interval/margin/family; named memory quantities;
fit/residual records; artifact/build/testbed provenance. Every plot spec includes
source-table hashes, selected attempt IDs, units, filters, normalization,
estimator, interval type/confidence, axis transform, theme and renderer revision.
Regeneration verifies hashes and needs no remeasurement.

Produce slide-ready standalone SVGs with restrained colors and readable labels:
latency and throughput size curves; separate requested-heap/RSS/commit size
curves; hardware/OS small multiples; compatible A/B ratio curves with the
equivalence band and corrected intervals. Expose linear/log axes explicitly.
Zeros cannot be silently removed from log plots. Missing/invalid points are
marked gaps with reasons, not zeros or connecting lines. Descriptive quantiles
must not be labeled confidence intervals. Do not display example benchmark
numbers invented for attractive graphics.

An offline explorable HTML view can filter metric, input, OS, testbed, boundary,
reuse, threads, tier and quantity; hover reveals raw units, uncertainty and
packet hashes. Link to included cell inventory and validity details. All views
derive from the same immutable tables, with sanitized metadata suitable for
sharing. No external chart service is needed for publication or regeneration.

## Exact launch counts, time and storage

For explicit eligible cell set E, planned observation replicates R(c) and
measurement passes P(c), fleet observation launches are exactly
`sum(c in E, R(c) * |P(c)|)`. Unsupported declarations remain inventory rows
but schedule no launch. Execution retries add attempts, separately counted.
For compatible paired cells A with B(c) blocks, K(c) candidate arms and
AA(c) in {0,1}, fresh child launches are exactly
`sum(c in A, B(c) * (1 + K(c) + AA(c)))`. This excludes separately declared
pilot/calibration/checker processes. Dry-run expansion must reproduce the count
and enforce zenbench's per-plan 10,000-invocation/retained-vector limits.

The proposed one-testbed pilot has `6 * 4 * 5 = 120` cells. Three replicates
and three observation passes give **1,080 launches**. Expanding to the six
testbeds gives **720 cells / 6,480 observation launches**. Adding Intel Windows
adds **120 cells / 1,080 launches**. If all 120 pilot cells were separately
selected for five-block one-candidate+A/A Confirm, that would be **1,800 paired
child launches**; prefer a frozen smaller follow-up family. These are arithmetic
budgets, not promised wall times, statistical power or available machines.

Actual calendar cost needs pilots of each metric/size/backend regime and target:
successful child duration, admission/refusal wait, checker/monitor/startup cost,
instrumentation slowdown, upload/download volume/time, and safe serialized
concurrency. Compute serial budget from measured components over declared
launches; scheduler critical path and authorized quiet windows determine elapsed
calendar time. Do not infer wall time from one host or scale one size linearly.
For paid resources, estimate cost only from resolved current rates and measured
occupied hours; this design launches none.

Storage is exactly the sum of distinct content-addressed object lengths, plus
non-deduplicated ledger/index/plot bytes. Pilot trace length versus operations,
metric/size/native allocations and instrumentation before estimating. Shared
input/build/runner objects are stored once; attempts preserve unique streams,
packets and traces. A compact selected Git record never replaces raw evidence.
Resumable observation transport may fill the declared coverage table; immutable
Confirm studies remain independent. Keep original attempts, including busy and
transport-failed ones, so selection and retry costs remain auditable.

## Prioritized implementation sequence and gates

| Step | Concrete location/interface | Acceptance |
|---|---|---|
| 1. Inventory and manifest dry-run | CLI `metrics/mod.rs`, existing `cpu-profile` driver, proposed private tooling `scripts/jobsys/build_benchmark_study.py`; no new public Rust API | Extract all 46 CLI names; report 29 cold/four warm driver coverage; feature/size/format/target unsupported rows; exact counts, duplicate refusal, deterministic cell IDs and hash binding. |
| 2. Qualify native driver/builds | `cpu_profile/src/fleet_case.rs`, its manifest, `check_iqa_fleet.py`, `check_iqa_bundle.py`, `check_iqa_heap_oracle.py` | Separate ordinary/memory binaries; release build provenance; actual target fixture parity, packed/padded, odd/even/minimum sizes, supported threading and full products. Extend HDR/GPU through explicit new case versions. Keep all current expectations. |
| 3. Separate protocol CI and real admission controls | Zenbench `tests/fleet_cli.rs`, runner admission boundary, `.github/workflows/ci.yml`; zenmetrics native fleet workflow | Deterministic admitted/wrong-output/busy/cancel paths; no release bypass; independent native OS counter and parent-lifetime controls; real quiet/busy admission gate tests and retained packets. All target builds require their actual IQA smoke matrix. |
| 4. Paired studies through zenfleet | Existing `benchmark_exec.py`, `benchmark_declare/mod.rs`, `JobKind::Benchmark` contract and capability tokens | Versioned paired-plan transport; bind all arm/input/runner bytes; fresh full-study attempt, cancellation, lease renewal, all packet evidence recovery/GC roots. Refuse unsupported old adapters. Tiny native controls only before real pilot. |
| 5. Ingestion and plotting | Existing fleet JSONL export plus proposed study reader/chart adapter; reuse zenbench themes | Hash-verified tables, integer memory/null reasons, complete coverage and no retry selection; real pair/fleet packet SVG/HTML, visible intervals/margins/invalid cells, reproducible plot spec, offline interaction. |
| 6. Descriptive size fits | Zenbench pair postprocessor, issue 39 | Independent numerical oracle, malformed/identity/cluster controls and real encoder+metric size-grid replay; inference disabled until separately validated. |
| 7. Small authorized pilot, then hardware expansion | Existing zenfleet enrollment/image builder/worker and content store | On-target native architecture/class/runner/workload probes; one successful receipt and recovery/GC proof before scaling; report missing cells and measured calendar/storage budgets. Expand only declared contrasts. |

This sequence reuses the implemented scheduler, bundle verification, observation
adapter, all-thread heap instrumentation, native OS counters and fixed-block
paired inference. The remaining work is a study layer and explicit workload/
deployment/analysis coverage; another scheduler would add no missing capability.
