# Minimal re-benchmark since f7668c7

The audited combined tree is `b216412c`, descended from upstream master
`7ed7b106`. This is a scope manifest, not a successful build or timing result.
Use the final landed commit and its pinned siblings for the new measurements.

[existing.txt](existing.txt) contains **23 existing CPU CLI names**;
[new.txt](new.txt) contains **4 added names** for first measurements.
[scope.json](scope.json) gives the reason and landing status for every name.
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

Run `just bench-iqa-changed --host <label> --corpus <AIC2026-img-dir> --skip-native`
for this exact 27-name scope. The coordinator retains the historical fastest-first
order in [order.tsv](order.tsv) for each size/thread budget instead of sorting
names alphabetically. The table cites historical 1024×1024 single-thread timing
sources; it does not claim that the integrated implementation has that ranking.
For selected sizes, use `just bench-iqa cid22,4k changed` with the same options.
