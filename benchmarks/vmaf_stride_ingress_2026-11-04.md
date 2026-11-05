# VMAF packed ingress follow-up, 2026-11-04

The PR 109 packed path now borrows the caller's frame descriptor slice through
`TightFrames::Borrowed`. Strided inputs retain packing and exact-minimum extent
support. V0/v1 kernels and score arithmetic are unchanged. The existing four
stride tests and 17 library tests pass with unchanged assertions. The complete
26-test libvmaf parity panel, two CAMBI map tests, 170 CVVDP tests and 86
CVVDP doctests also pass. Strict library Clippy passes for VMAF and CVVDP.

The retained workload is `crates/vmaf/examples/stride_cost.rs`; the profiler
runner is `scripts/ci/profile-vmaf-ingress.sh` (also exposed in the justfile).
It compares PR head `739a0aa0` with VMAF source from `ef6c166f`, both at default
release settings, with no function-alignment flags or target-cpu override.
The same example was copied into the exported old revision before building it.
Both builds use the pinned sibling snapshot. This is r7900x/Zen 4; Zen 5
and GPU performance were not measured.

## Measured allocation control

Heaptrack observes ten whole calls at 320x240. Each scorer is constructed once;
process/runtime startup and fixture allocation remain included on both sides.

| Model | Before allocation calls | After allocation calls | Removed |
| --- | ---: | ---: | ---: |
| V0 Standard | 644 | 604 | 40 |
| V1 Standard1080p | 1137 | 1097 | 40 |

The original trace includes the `Vec<TightFrame>` normalization stacks; the
corrected packed path has no such stacks. The observed reduction is four
allocations per call for this workload. This does not remove kernel scratch or
claim a lower peak memory footprint.

## Counter controls

Three alternating-order process pairs each execute 100 whole calls per model.
Every before/after stdout matches exactly, including score bits:
V0 `405851b6f92274b9`, V1 `4058f3a2e3d333ac`.

| Model | Round | Instructions before | Instructions after | Cycles before | Cycles after |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0 | 1 | 15531120311 | 15529300812 | 3511505842 | 3522574811 |
| v0 | 2 | 15530200291 | 15529525478 | 3509065195 | 3510619163 |
| v0 | 3 | 15529742334 | 15530218523 | 3521627544 | 3514907645 |
| v1 | 1 | 5724964001 | 5724980481 | 1190637942 | 1183062246 |
| v1 | 2 | 5724762746 | 5724577364 | 1158343745 | 1175381503 |
| v1 | 3 | 5725027252 | 5725078317 | 1152731679 | 1182730337 |

This counter panel establishes neither a wall-time speedup nor default-build
non-inferiority. The historical PR note's default-build slowdown remains unqualified by a passing wall-time acceptance
run. The retained confirmation below is invalid on resource observations. Do
not use forced-alignment measurements as acceptance for ordinary builds.

Full artifacts remain under `$HOME/tmp/zenmetrics-pr-followup-20261104/` on the
review machine, including binary hashes, raw counters and heaptrack traces in
`profiles-r2/`. The first profiler attempt stopped because heaptrack selected
`.zst` rather than `.gz`; it remains preserved in `profiles/`.
The corrected runner handles both formats and fails on missing traces.

Resource wrapper receipt for the corrected profiling panel:
`run-heavy: done rc=0 6s | peak-RSS 0.09GiB | min-avail 16190MiB | peak-load 0.78`.
This is the wrapper's host-process footprint, not GPU memory or kernel-only RSS.

Reproduce: build the retained example at both revisions via the CI-pinned
snapshot, preserve each binary, then run `just vmaf-stride-cost-profile
BEFORE AFTER OUTPUT` under run-heavy. Arguments are literal paths. The runner
records binary hashes, checks exact output identity and retains all samples.

## Focused software-WGPU stride controls

With the pinned dependency graph and current merged source, all newly added
stride modules pass: CVVDP 2/2, Butteraugli 2/2, Zensim 2/2 and the umbrella
API 3/3. They exercise exact tight/strided score/map identity, minimum extents,
invalid geometry and applicable warm paths. `vulkaninfo --summary` identifies
the available device as llvmpipe (LLVM 21.1.8, CPU device). These are software
WGPU controls; they do not qualify CUDA or discrete GPU performance.

Commands and full output are retained in `pr109-gpu-stride.log` and
`pr109-other-gpu-stride.log` beside the profiling artifacts. The latter wrapper
receipt is `run-heavy: done rc=0 86s | peak-RSS 2.15GiB | min-avail 15735MiB | peak-load 5.14`.

## Default-release wall confirmation

A frozen zenbench `56d5a77` confirmation compared master `b50fea22` to the
packed-path correction, at 320x240 for v0 and v1. Both binaries use the same
retained workload; only the absent `strides` fields were removed for master.
Each fresh process creates one scorer and executes 1000 calls; outer wall time
includes fixture setup, launch and completion polling. Five blocks include
baseline/candidate/A-A for both cases: all 30 invocations completed and all
exact stdout/score-bit checks pass.

The frozen non-inferiority margin is 1%, A/A control margin 1%, family
confidence 95%, and seed 11. The declared two-nanosecond timing resolution
covers the measured one-nanosecond CLOCK_MONOTONIC resolution and float
conversion; polling overhead remains part of the explicit measured operation.

**Not accepted.** 26 of 30 invocations include unclean resource observations.
Zenbench returns exit 2 and every candidate/control verdict is `invalid`.
The concurrent CPU observations and unresolved A/A equivalence do not qualify
the default packed path as non-inferior. Do not promote the diagnostic latency
estimates or replace the unchanged acceptance conditions with an alignment flag.

The [selected CSV](vmaf_stride_wall_2026-11-04.csv) retains these invalid
verdicts. The [portable plan](vmaf_stride_wall_2026-11-04.plan.json) substitutes
relative binary paths; copy the two frozen binaries to its `binaries/` directory
before reproduction. The executed absolute-path plan and complete packet remain
in `$HOME/tmp/zenmetrics-pr-followup-20261104/wall-confirm/` on the review host.
The original packet SHA-256 is `bcf42f119b42600e33cccc725019b8e1905d8f5cc2853dcc92fecd09874a11d4`
and experiment fingerprint is `8b948fa6f4af4f5e8a8ad077e98d1f3c5d75ff43f5caf1551c1b8bee2f257bb3`.
Run the portable plan through run-heavy with `zenbench pair PLAN --mode confirm
--output NEW_DIRECTORY --wait 60`; changing paths produces a new identity.
The retained workload later receives only an equivalent integer-divisibility
style correction for strict Clippy; measured binaries remain bound to the
recorded source revisions and hashes.

Resource receipt: `run-heavy: done rc=2 154s | peak-RSS 0.04GiB | min-avail 18972MiB | peak-load 3.28`.

PR 109 remains draft until an eligible default-build confirmation passes.
CUDA execution also remains unqualified; the software-WGPU tests above are
exact stride-equivalence controls, not CUDA numerical qualification.
