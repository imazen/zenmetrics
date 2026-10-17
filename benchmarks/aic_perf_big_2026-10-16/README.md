# CAMBI and original Butteraugli CPU optimization

All required gates pass. All 61 AIC scores for each metric and all 61 complete
Butteraugli error maps match the preserved baselines exactly. Published-unit
and JND residual bits are unchanged.

## Original Round 14 measurements (before review follow-up)

Host: i265, release builds, no `target-cpu=native`. One-worker runs pin CPU 2;
16-worker-budget runs use the available CPUs. Each cell runs base, new, base,
new in separate processes; each process averages five calls. Values below are
means of the two process means per role. These procedural-fixture kernel
observations have no confidence intervals and cover this host only.

| Rust metric | Size | Thread budget | Base ms/call | New ms/call | Base/new |
|---|---:|---:|---:|---:|---:|
| [CAMBI](abab-cambi-1024-1t.log) | 1024×1024 | 1 | 22.208 | 22.203 | 1.00× |
| [CAMBI](abab-cambi-1024-16t.log) | 1024×1024 | 16 | 26.029 | 8.953 | 2.91× |
| [CAMBI](abab-cambi-3355-1t.log) | 3355×2516 | 1 | 160.759 | 160.828 | 1.00× |
| [CAMBI](abab-cambi-3355-16t.log) | 3355×2516 | 16 | 164.801 | 63.282 | 2.60× |
| [Butteraugli-original](abab-butter-1024-1t.log) | 1024×1024 | 1 | 1034.395 | 308.292 | 3.36× |
| [Butteraugli-original](abab-butter-1024-16t.log) | 1024×1024 | 16 | 1074.443 | 313.392 | 3.43× |
| [Butteraugli-original](abab-butter-3355-1t.log) | 3355×2516 | 1 | 7958.670 | 3005.396 | 2.65× |
| [Butteraugli-original](abab-butter-3355-16t.log) | 3355×2516 | 16 | 7970.544 | 3015.221 | 2.64× |

CAMBI keeps the original one-worker path. With `parallel` enabled, its column regions use
the existing Rayon pool at per-scale areas of at least 512×512. Original
Butteraugli remains serial within one image; its 16-worker-budget rows
measure the same serial implementation under that requested budget. VMAF
frame and streaming scorers keep CAMBI serial with the default/one-thread
configuration; configured private pools also own single-frame CAMBI work.
A nested-pool regression checks this routing and serial/parallel score bits.

[All observations, counters, RSS and load averages](measurements.tsv);
[audited summary and gate receipts](summary.json). Each linked raw cell log
contains its run-heavy resource footer and all four `/usr/bin/time -v` records.

## Changes and ordering

- CAMBI (`0ffb7674`): independent column regions retain each histogram cell’s
  original row/pixel update order. Eight-column alignment leaves eight scalar
  columns at the end of every non-final region; those columns used SIMD in
  the full-width traversal. Results retain their original pixel positions
  before unchanged spatial pooling. The original serial histogram, point
  evaluator and pooling functions remain text-identical to `849ea8da`.
- Butteraugli-original (`6ced5b29`, `50acada6`): 32×32 convolution tiles retain
  each pixel’s tap order and original border weights. Eight independent
  interior Malta pixels retain their neighbor/path order; border pixels keep
  the original traversal. f32 planes and selected f64 expressions retain
  their types, grouping and identity multiplications.

[Generated-code evidence](butter-kernels.json) records separate packed
`mulps`/`addps` with no FMA in the inspected Butteraugli kernels.
[CAMBI evidence](cambi-ordered-kernels.json) identifies the unchanged point
evaluator. Full disassembly remains at the hashed external paths.

The original column-region implementation added temporary c-value storage.
At 3355×2516 and budget
16, measured process peak RSS changes from 105,824 to 140,060 KiB. The
Butteraugli one-worker process peak is 1,437,516 KiB for both versions.
Corresponding large-image resource scopes:

- `run-heavy: done rc=0 2s | peak-RSS 0.13GiB | min-avail 24187MiB | peak-load 1.68`
- `run-heavy: done rc=0 110s | peak-RSS 1.37GiB | min-avail 22989MiB | peak-load 5.33`

The [512×512 boundary control](cambi-threshold-512.log) retains score bits
and exercises the parallel dispatch boundary. Strict strided map tests
compare both visibility presets and native windows 17/65 on 1/2/4-worker
pools against the original ordered implementation ([receipt](cambi-presets.log)).

## Correctness and required gates

CAMBI retains nine explicit geometry refusals in the 27-case expanded
ladder. Its 18 supported ladder cases match pinned libvmaf numerically;
all 61 AIC pairs produce scores. Maximum libvmaf differences are 0 on the
supported ladder and 6.938893903907228e-18 on AIC. Original Butteraugli’s
ladder maximum is 4.790954593048014e-7, below the unchanged 4.8e-7 bound.
The original C++/libvmaf builds and output-precision convention are recorded
in the [existing oracle provenance](../aic_more_2026-10-15/validation.meta).

[61-pair bit panel](aic-parity.tsv) / [raw run](aic-parity.log);
[common residual bits](residuals.tsv) / [input hashes and derivation](residuals.json).
The final oracle run uses fresh output files for all four original panels,
including new pinned CAMBI references for the six added non-square cases.

`just aic-more-ci` passes CLI/vmaf tests, all-member fmt, exact CI workspace
clippy and inventory. [All 12 CLI feature combinations pass](feature-matrix.log).
The advisory quality kit is absent; explicit required checks pass.

## Reproduction and retained controls

Use a dedicated `LOCK_SNAP_DIR`. Baseline and interleave orchestrators run
outside flock; every bounded step acquires/releases the shared heavy lock.
Source/binary/function hashes: [qualification metadata](qualification.meta.json).

- `just aic-perf-big-baseline OUTPUT "cambi butter"`
- `just aic-perf-big-interleave CAMBI_BASE BUTTER_BASE CANDIDATE OUTPUT`
- `just aic-perf-big-parity BASE CANDIDATE OUTPUT_TSV FRESH_ARTIFACT_DIR`
- `just aic-perf-big-oracles FRESH_OUTPUT ORACLE_PYTHON`
- `just aic-perf-big-summary SCRATCH OUTPUT_PREFIX thread`

The first noise scout produced zero CAMBI; the primary banded fixture
extends the original-author pinned 10-bit staircase. Baseline source is
`849ea8da`; binaries and perf traces remain preserved outside Cargo output.
Early hybrid-PMU scouts retain both counter lines. Final comparisons pin
perf itself to performance-core CPU 2 and select `cpu_core` counters.

An integer-run experiment is superseded because it regrouped histogram
updates despite bit identity. Its receipts are labelled as experiments;
its helper/assertions are test-only. The final production change preserves
the original updates. The first oracle attempt stopped on missing system
OpenCV; final runs explicitly use the existing complete oracle environment.
No expectation, tolerance or golden was relaxed.

## Review follow-up

The merged review tree keeps scalar CAMBI inside parallel frame workers while
retaining column work for single-frame and streaming private-pool calls.
Direct row controls cover active out-of-band values, vector groups and scalar
tails; both new controls fail before the fix and pass afterward. Column regions
borrow disjoint output-row segments with fallible descriptor allocation.
Both Malta path banks have compile-time four-pixel halo checks.

[Fresh local gate receipts](review-gates.tsv) cover `just aic-more-ci`, all
12 CLI feature combinations, complete scalar and SIMD/parallel VMAF suites,
and strict library clippy. [The new bit panel](review-panel.tsv) matches the
committed original panel: 61/61 CAMBI scores, 61/61 Butteraugli-original scores
and 61/61 complete Butteraugli map hashes are unchanged.

The [RSS comparison](review-memory.tsv) runs fresh base/new/base/new processes,
five calls each, with 16 Rayon workers. Values below are the maximum of two
OS peak-RSS observations per version, from `/usr/bin/time -v`.

| Standalone CAMBI | Before KiB | After KiB |
|---|---:|---:|
| 1024×1024 | 19,808 | 16,656 |
| 3355×2516 | 140,488 | 107,624 |

Source: baseline `7585b851`, reviewed tracked tree `d317b64b` (identical to
exported working copy `293a4027`); binary hashes and individual wall/RSS
observations are retained in the TSV. Release features are
`parallel,simd,avx512`, without `target-cpu=native`. Execution is AVX2 on i265;
AVX-512 code was compiled and was not executed.

Large-image comparison resource receipt:
`run-heavy: done rc=0 1s | peak-RSS 0.13GiB | min-avail 23269MiB | peak-load 1.45`.
Full bit-panel receipt:
`run-heavy: done rc=0 90s | peak-RSS 1.37GiB | min-avail 22318MiB | peak-load 1.64`.

An additional strict all-target clippy probe remains red on the original and
reviewed trees with the [same 49 diagnostics](review-lint.tsv): one unused
mutable test binding, two constant-size chunk lints, and 46 FFI safety-comment
lints. All affected files are byte-identical to `7585b851`. The required CI
clippy command and the strict library clippy command pass; no lint allowances
or test expectations changed.
