# VMAF rerun: AVX-512 feature correction

The owner correctly identified the missing feature in the frozen benchmark.
The original source e2d414c3 did not enable VMAF AVX-512. The feature propagation
fix is present in the newly built revision 2f96c9ff. Repeating the old executable
on an AVX2-only CPU does not test that fix.

## Completed

- Three fresh paired blocks at both 2048x1080 and 3840x2160 on a 5900XT,
  using the historical Rust and native executable bytes: Rust/native geometric
  mean latency ratios 1.753 and 1.743. This is an AVX2 diagnostic only.
- Fresh pinned-source build at 2f96c9ff6e258876c8f9b11c22098fe46b3a401a.
  All 23 sibling pins and locked resolution passed. Built cpu-profile/iqa-compete
  with `compete,metrics-parallel`, thin LTO, and no target-cpu=native flags.
- Resolved `cargo tree -e features -i vmaf` explicitly includes `vmaf/avx512`
  through both cpu-profile and zenmetrics-cli.
- Binary SHA256: `af5d51220df709ac0f99f8761510ca45cffeafa0caefca26fe79793820bbd20f`.
  run-heavy build: rc0, 98s, peak RSS 1.15 GiB, minimum available 50027 MiB,
  peak load 4.98, cap 16 GiB, jobs12.
- The new executable requires glibc2.43; the target has glibc2.35. Staged an
  owned private loader/library bundle, shared by all three planned arms, with
  no system library changes. The new executable's inventory command runs there.

## Not completed

The 7950X measurement did **not start**: the initial 45-second admission wait
and subsequent 180-second wait for the corrected experiment both expired behind
existing heavy work. No production process was interrupted. There is no new
AVX-512 timing, speedup claim, or runtime proof that a VMAF AVX-512 kernel ran.

The prepared experiment compares libvmaf, historical Rust, and fixed Rust in
three randomized blocks per size, with A/A controls, identical inputs and the
shared private runtime. Native still receives prepared BT709 YUV420 while Rust
receives RGB; that workload distinction remains. Exact code/build changes and
runtime changes prevent attributing any future difference solely to the feature
without the corresponding evidence. Existing numeric tolerance stays 0.02.

Private evidence: `/mnt/v/aic2026-evidence-private/vmaf-rerun-20261019T140803/`.
Includes completed AVX2 attempts, fixed build/feature receipts, the prepared
7950X plan and both failed admission attempts. The next needed resource is a
benchmark slot on the 7950X, not another source audit or another old-binary run.
