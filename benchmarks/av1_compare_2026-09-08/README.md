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

## Matched quality: 512-pixel sources, 8-bit 420, preset 6

Log-linear estimates at **SSIMULACRA2 80**, bracketed by measured quantizers:

| Backend | Photo bytes | Photo ms | Screenshot bytes | Screenshot ms |
|---|---:|---:|---:|---:|
| C SVT | 14,702 | 13.49 | 14,955 | 16.66 |
| zenav1-svt | 14,708 | 20.23 | 14,955 | 24.29 |
| libaom | 14,337 | 52.62 | 12,774 | 53.25 |
| zenav1-aom | 13,826 | 86.69 | 14,028 | 77.17 |
| zenrav1e | 12,599 | 638.39 | 17,590 | 489.68 |

At this point Rust AOM saves about 6% over Rust SVT at 3.2–4.3× the time.
Zenrav1e saves 14% on the photo at about 32× the time; on the screenshot it
uses both more bytes and more time. Preset numbers across encoder families
are not equivalent effort, so this table is a slice, not the overall Pareto front.

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

## Evidence and remaining work

`pilot.tsv` contains the 30 V1 cells. Full V2 rows, 1,004-cell summaries,
matched-quality brackets and encoded artifacts are retained as a run bundle;
the storage pointer is added after upload. Commands and request grids are kept
with that bundle. Do not use two sources or sparse high-depth QP brackets to
fit a production optimality model. Fleet expansion, denser high-depth curves,
additional perceptual metrics, explicit tune/SCM arms and larger image sizes
are the next validation steps.
