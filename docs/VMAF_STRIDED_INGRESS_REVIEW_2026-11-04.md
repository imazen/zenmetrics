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
non-inferiority. The historical PR note's default-build slowdown remains
unqualified by a retained zenbench A/B wall-time acceptance run. Do not use its
forced-alignment measurements as acceptance for ordinary builds.

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
