# Minimal re-benchmark since f7668c7

The original audited combined tree was `b216412c`, descended from upstream master
`7ed7b106`. This is a scope manifest, not a successful build or timing result.
Use the final landed commit and its pinned siblings for the new measurements.

[existing_current41.txt](existing_current41.txt) contains **37 existing CPU CLI names**;
[new.txt](new.txt) contains **4 added names** for first measurements.
[scope.json](scope.json) preserves the original 27-route audit. The current
41-route scope also includes the eight PSNR routes revised in `07972529` and
`bcadda73`: `psnr`, `psnr-y`, `psnr-y601`, `psnr-y-studio601`,
`psnr-y-libvmaf`, `psnr-cb`, `psnr-cr` and `psnr-ycbcr611`.
`psnrhvs-daala` also reaches the revised `studio601_yuv444` preprocessing;
its omission from the previous 40-name selection is corrected.
FLIP, FLIP-v17 and WS-HV are also revised by `f5523a4b` and `7d27d22b`.
Base `hdrvdp` and `hdrvdp3` gained the upstream sRGB-display route in `4f89155c`.
The five fixed-factor MS-SSIM variants need score revalidation as well as timing.
VIF and VIFvec need their existing parity gates retained.

`fsim-y` is an optional unchanged-output control: its existing Auto path gained
a delegating wrapper for an additive API. Other FSIM CLI paths are unchanged.
`zensim-gpu` needs a refusal-behavior check (panic became an error), not timing.
No learned-model or GPU-kernel changes were found in this diff. The standalone
umbrella API's newly enabled IW-SSIM AVX-512 feature matters to API-only consumers;
the CLI already enabled it at the baseline.

The AVX-512 dependency edits do not require re-running all twelve consumer
metrics: eleven already had the feature enabled in the baseline CLI. VMAF and
CAMBI are included because their CLI dependency lacked that feature.

Keep the original baseline measurements and source hashes. Compare the same
host, inputs, thread budgets and timing mode. Use the existing benchmark harness
and its qualification rules; do not reinterpret historical rows as measurements
of the new executable. Add one unchanged CPU control when comparing whole
binaries to detect host or toolchain drift.

The earlier MS-SSIM gate experiment includes a 4K multithreaded slowdown in
[msssim-libvmaf results](../opt_msssim-libvmaf_2026-10-18/summary.tsv).
Re-measure that cell explicitly; the gate is not established as a win at every
size. Snapshot publication additionally requires green checkpoint tests and CI.

Run `just bench-iqa-changed` with the explicit build/input/provenance options in
[the command guide](../../docs/IQA_CANONICAL_COMMANDS.md) for the current 41-name
scope. Both native selectors use [the measured local order](order_current_local41_2026-10-20.tsv)
instead of sorting names alphabetically. Its [receipt](local41-order-2026-10-20.json)
binds one common 1152×1152 S67 photo crop on Ryzen 9 7950X / WSL2, source
`ea767ba1`, one thread and at least ten warm observations. This size admits
corrected MS-SSIM k6. The original forty-route run took 309s, peak RSS 0.76 GiB, minimum available
14205 MiB, peak load 2.16. Daala was measured separately on the identical
input/configuration: 2s, peak RSS 0.11 GiB, minimum available 15156 MiB,
peak load 2.93. All observed noisy rounds remain retained. This ordering is
a scheduling hint for other workloads. The original [40-route table](order_current_local40_2026-10-20.tsv)
and [membership list](existing.txt) remain unchanged for their frozen plans.
The older [38-route table](order_current_after_flip_2026-10-20.tsv) remains unchanged.
For selected sizes, use `just bench-iqa cid22,4k changed` with the same options.

The original [27-route table](order.tsv) remains unchanged for frozen comparison
plans. `just bench-iqa-canonical-changed MANIFEST NEW_OUTPUT` runs canonical
comparisons in the manifest's declared `order`; omitting that field retains
the legacy table. Its new plans preserve every source case's numerical bounds.
