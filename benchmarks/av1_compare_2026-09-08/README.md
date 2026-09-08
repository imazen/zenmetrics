# AV1 still RD and timing comparison — 2026-09-08

**Completed locally:** 90 pilot encodes, then 3,012 expanded encodes (1,004
cells × three rounds). All expanded outputs decoded successfully. Repeated
encodes were byte-deterministic. A separate 42-cell lossless matrix verified
native sample equality across supported 8/10/12-bit and 420/422/444/mono paths.

This is a two-source diagnostic, not a general routing calibration. Sources:
CLIC2025 training `097cb426910ba8ce2525dd8bb7fb1777.png` and gb82-sc
`terminal.png`, resized with Lanczos3 to maximum edges 256 and 512. Source
hashes and exact executable/backend identities are in the saved rows. The
photo sizes are 188×256 and 376×512; screenshot sizes 256×164 and 512×330.
Input images are 8-bit SDR: high-depth arms preserve precision through RGB→YUV
conversion; these are **not native HDR tests**.

## Full training scout status

[The population scout is stopped](IMAZEN26_SCOUT_STATUS.md), with two complete
population artifacts verified and retained (468 timed encodes). Canonical ledger
recovery and remote fleet checkout inspection are pending. The report includes
bytes/bpp, quality and timing; population calibration is not complete.

## Canonical restoration-unit experiment

[The completed restoration ablation](IMAZEN26_RESTORATION_UNITS.md) adds
144 fleet encodes with automatic reconstruction verification. Legal smaller
units are implemented and reachable, but the two changed photo cells became
slightly larger and scored slightly lower. Screenshot restoration was bypassed.
The experiment remains opt-in. The report includes bytes/bpp, perceptual scores,
three-round timing, common-budget estimates and separate CPU provenance.

## Canonical imazen-26 research follow-up

[The completed research baseline](IMAZEN26_RESEARCH_BASELINE.md) adds150 encodes
on canonical training origins1000/8100. Rust SVT -1 matches C on10/10 distinct
cells; all50 cells decode and repeat deterministically. It exposes a substantial
Rust -1 screenshot timing gap and five libaom/Rust AOM screenshot differences.
This is a baseline for broader scouting, not the requested minimum corpus set.

## Method

- Static Linux x86-64 executable; Intel Core Ultra 7 265K, default CPU target,
  runtime SIMD dispatch, one heavy job, nice/ionice through `run-heavy`.
- In-memory public APIs, fresh encoder lifecycle. Timer includes encoder setup,
  required input representation copies and teardown. It excludes source I/O,
  shared RGB→YUV conversion, independent decode, scoring, hashing and artifact I/O.
- Native presets 3/6/9; native quantizers 0,10,15,20,25,30,35,40,45,50,55,60,63
  for 8-bit 420. zenrav1e uses four times those quantizers (its native 0..255
  scale); equal quantizers are never treated as equal achieved quality.
- Additional preset-6 arms: 8-bit 444, 10-bit 420/444, 12-bit 444, each at native
  QP 0/20/30/40 (zenrav1e ×4). Unsupported SVT format cells are not declared.
- Threads=1; C SVT interprets this as lp1, not literally one OS thread. Arm
  order rotates across rounds. Local run was not CPU-affinity pinned.
- Full RGB SSIMULACRA2 via fast-ssim2 0.8.2, with one shared BT.709 limited-range
  conversion and libaom decoder. Conversion-only and codec-only scores are also
  recorded. Sizes below are **AV1 OBU payload bytes**, not AVIF container sizes.
- Every OBU is saved by SHA256. `analyze.py` emits measured cells plus log-linear
  matched-quality estimates between neighboring quantizers. No extrapolation;
  reversals are reported and excluded from interpolation. Estimated timings
  interpolate per-cell medians, not subprocess elapsed times.

The first wrapper draft accidentally exposed zenavif-style SVT speed mapping
while describing it as native speed. This was corrected to the raw pipeline
**before either real-source run**. The original checkerboard smoke is not RD
calibration. V2 adds aligned C input copies and format support; keep its binary
and protocol identities separate from V1.

## Filled presets: equal time budgets at SSIMULACRA2 80

The 2026-09-08 follow-up completed **2,016 missing-preset encodes** plus
**480 local high-quality encodes**. The four C/Rust SVT/AOM arms now cover every
normal preset 0..9, including SVT 0/1/2, at maximum edge 512. Added QPs are
5/8/10/15/20/25/30/35/40/45/50/55/60/63, with prior QP0 anchors at 3/6/9.
The zenrav1e baseline remains presets 3/6/9. Three rounds per cell; all decoded
and repeated byte-identically. The same local CPU and timing method were used.

The table selects the smallest bracketed payload under each common budget,
independently choosing each backend's preset. **AV1 payload bytes**, 8-bit 420:

| Budget | SVT photo | AOM photo | SVT screenshot | AOM screenshot |
|---|---:|---:|---:|---:|
| 30 ms | 13,288 | 14,359 | 14,955 | 24,002 |
| 100 ms | 12,357 | 13,012 | 13,222 | 14,028 |
| 200 ms | 11,327 | 13,012 | 12,829 | 14,028 |
| 500 ms | 10,821 | 11,097 | 12,008 | 11,927 |
| 1,000 ms | 10,634 | 10,712 | 11,906 | 10,990 |
| 2,000 ms | 10,634 | 10,609 | 11,136 | 10,990 |
| 5,000 ms | 10,634 | 10,609 | 11,136 | 10,490 |

SVT/AOM in this table mean **zenav1-svt / zenav1-aom**. The complete five-backend
selection, actual estimated milliseconds, chosen presets and quantizer brackets
are in [time_budgets_ssim2_80.tsv](time_budgets_ssim2_80.tsv). Presets are discrete:
there is no interpolation across preset numbers. Budget cutoffs use estimated
median time, not a guarantee of a runtime deadline.

At the photo's slow end, Rust SVT preset 0 is **10,634 B / 680 ms**, essentially
tying Rust AOM's smallest tested point, preset 2 at **10,609 B / 1,500 ms**.
For the screenshot, Rust AOM preset 3 is **10,990 B / 602 ms**, versus Rust SVT
preset 1 at **11,906 B / 701 ms** (AOM 7.7% smaller and 14% faster). The smallest
normal-preset screenshot points are SVT p0 **11,136 B / 1,589 ms** and AOM p0
**10,490 B / 2,295 ms** (AOM 5.8% smaller at 1.44x time).

The C baselines retain practical value: screenshot libaom p0 estimates
**10,490 B / 965 ms**, versus C SVT p0 **11,136 B / 1,084 ms**. The Rust AOM
runtime gap to its C baseline is separate from an AOM-vs-SVT algorithm choice.

**Supersedes the earlier coarse photo interpolation:** adding QP5/QP8 shrank
large QP0-to-QP10 brackets. For example SVT p3 changed from 12,836 to 11,327 B
at score 80; that is an estimate correction, not a new encoder improvement.
The revised preset-6 slice is:

| Backend | Photo bytes | Photo ms | Screenshot bytes | Screenshot ms |
|---|---:|---:|---:|---:|
| C SVT | 13,565 | 13.40 | 14,955 | 16.66 |
| zenav1-svt | 13,581 | 21.37 | 14,955 | 24.29 |
| libaom | 13,469 | 51.55 | 12,774 | 53.25 |
| zenav1-aom | 13,012 | 85.08 | 14,028 | 77.17 |
| zenrav1e | 12,599 | 638.39 | 17,590 | 489.68 |

## C SVT research preset -1

The signed benchmark adapter now measures C's public preset -1 and rejects
negative presets for the Rust backends. This C build rejects -2/-3 despite
those names being present in its header enum. A fresh test encodes/decodes -1
and checks all these refusal boundaries.

A further **120 real-source encodes** compared -1 and 0, three rounds per
cell. All decoded and were deterministic; all 20 preset-0 control cells match
the pre-change executable byte-for-byte. At SSIMULACRA2 80:

| C SVT preset | Photo bytes | Photo ms | Screenshot bytes | Screenshot ms |
|---|---:|---:|---:|---:|
| -1 | 10,608 | 736.98 | 10,831 | 2,494.64 |
| 0 | 10,509 | 435.90 | 11,136 | 1,082.68 |

Research mode is 2.7% smaller on the screenshot at 2.3x time, but slightly
larger and 1.7x slower on the photo. The unsigned Rust API's missing -1 is a
coverage gap; it is not evidence that -1 should become the default. The full
research ladder changes multiple tools, so this does not isolate SGR's gain.

## Zenfleet supplement

The canonical zenfleet worker completed **20/20 comparison jobs, 960 encodes**,
three rounds each, on three Nomad-managed remote workers. QPs 0/2/5/8 were
covered at every normal preset 0..9 for both sources and all four C/Rust
SVT/AOM arms. Every persisted OBU was downloaded and SHA/length-verified.
The worker capability setting was corrected after an initial launcher error
(resource classes differ from executor capability tokens); no encoding jobs
failed. All three successful allocations drained and exited.

Storage: `s3://zentrain/jobs/av1-preset-fill-20260908/` contains the manifest,
exact static executable/worker/corpus bundle, per-job output bundles, Parquet
ledger and complete snapshot. **Remote timings are not pooled into the local
budget table.** Each remote source/preset comparison shares one worker; worker
identity is retained in the ledger. All 160 local QP5/QP8 overlapping cells (480 rounds) match the fleet
outputs byte-for-byte across CPUs, independently of timing.

See [AOM_ADOPTION.md](AOM_ADOPTION.md) for source-backed adoption candidates
and the important distinction between missing C research-mode support,
already-implemented tools, and possible SVT extensions.

## Direct high-fidelity format result

On the **512×330 screenshot, 10-bit 444, preset 6**, measured QP20 (zenrav1e Q80):

| Backend | Bytes | SSIM2 | Median ms |
|---|---:|---:|---:|
| libaom | 15,953 | 87.69 | 71.38 |
| zenav1-aom | 16,057 | 87.25 | 114.50 |
| zenrav1e | 20,658 | 87.21 | 756.72 |

Rust AOM is 22% smaller and 6.6× faster than this zenrav1e point, at essentially
the same measured quality. This is a concrete reason to evaluate AOM for the
444 requests outside SVT's envelope. It is not evidence to remove SVT extensions.

At lossless native-plane coding on the 256×164 screenshot, conversion ceilings
were 89.19 SSIM2 (8-bit 420), 91.57 (8-bit 444), 92.53 (10-bit 420), 98.34
(10-bit 444), and 100 (12-bit 444). Full-range RGB identity is a distinct path
not tested here; "lossless YUV" must not be presented as lossless RGB.

## Regression witnesses, preserved

At 512×330, 8-bit 420, preset 9, both C SVT and Rust SVT produced:

| QP | Bytes | SSIM2 |
|---|---:|---:|
| 15 | 24,228 | 73.897 |
| 20 | 21,178 | 75.770 |

QP20 dominates QP15. This is a measured reversal shared with C, not evidence of
a translation-only flaw. No expectation, threshold or encoder tool was disabled.
The analyzer reports both rows and excludes this segment from interpolation.

Some larger/10-bit SVT cells are not byte-identical to C despite identical
lossless decoded planes; these need separate parity investigation. Do not infer
universal byte identity from the six matching V1 pilot cells.

## Evidence, validation and remaining work

The original pilot and V2 runs, the preset fill, local high-quality supplement,
research/control run, all OBU files, request grids, analyses and validation logs
are retained in:

- `s3://zentrain/benchmarks/av1-compare/2026-09-08/evidence.tar`
- SHA256: `47c7fae8466889c3ebf3fab6bb398735ab6e604d77d861ae5c4f5380e0a35dd6`
- Artifact/build manifest: `s3://zentrain/benchmarks/av1-compare/2026-09-08/artifacts.json`
- Fleet manifest/ledger/bundles: `s3://zentrain/jobs/av1-preset-fill-20260908/`

The manifest distinguishes the exact fleet-measured executable from the final
harness binary. Two needless-borrow Clippy cleanups followed the research run;
the measured research `measure.rs` is retained separately, and measured binary
hashes remain in every row. Do not relabel old rows with the final binary hash.

This continuation completed **3,576 additional real-image encodes**: 2,016
preset-fill + 480 local high-quality + 120 research/control + 960 remote.
All repeats were deterministic and all emitted outputs decoded. Five benchmark
library tests pass, including the 42-format native-lossless matrix, plus scoped
library/benchmark-binary Clippy with warnings denied. Existing warnings in
codec dependencies and their wider workspace gates remain separate. Budget
selection was checked at exact boundaries, against dominated presets, and for
refusal of incomplete rounds. New report files pass repository hygiene checks.

`pilot.tsv` retains the historical 30 V1 cells. The dense normal-preset table
supersedes the coarse photo estimates; it does not change the encoders.
Do not use two sources or sparse high-depth brackets to fit production
optimality. Denser high-depth curves, IQ/SCM arms, more perceptual metrics,
larger images and held-out sources remain further validation work. C/Rust
parity witnesses and Rust's missing research preset remain open.

## Corrected intra-edge ablation

[Completed canonical native-1 intra-edge experiment](IMAZEN26_INTRA_EDGE.md):
120 encodes,40 deterministic cells,20 exact reconstruction replays. Filtering
costs more size/time at the two bracketed quality targets; remains opt-in.
The report supersedes the pre-fix run and preserves its correctness finding.
