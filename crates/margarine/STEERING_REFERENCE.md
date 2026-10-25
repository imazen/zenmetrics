# Steering reference audit (2026-10-25)

The selected command and frozen teacher are unchanged. Native-linear steering
meets the calibration and JPEG XL steering bars below. Full rank,
encoder-choice and resource acceptance remain unqualified.

## Identical pixels, historical implementations

`reference-audit/` pins three Butteraugli sources independently:

- Encoder loop: [`d2466a4`](https://github.com/imazen/butteraugli/tree/d2466a4eaad67f8008838c0caa452484300214ca).
- Frozen teacher: [`13c49cbc`](https://github.com/imazen/butteraugli/tree/13c49cbcab6e2b2bb65f60e3cb28344463bacbf7).
- Current Fast and LibjxlExact: [`554a42f8`](https://github.com/imazen/butteraugli/tree/554a42f832ceffe7a70303d43f3cddcd5a8cd275).

On all 324 saved 512-square calibration pairs, the first two and current Fast
produce bit-identical maps on identical RGBA8 pixels. LibjxlExact differs from
Fast by at most 0.0005548000335693359 in any map pixel. These results rule out
the historical source revision as the explanation of the observed bulk bias
on these pairs; they do not establish universal equivalence.

The comparison against the saved encoder block targets is:

| Mode | Photo bulk log bias | Screenshot bulk log bias | Screenshot worst-block log r |
|---|---:|---:|---:|
| Historical loop / frozen teacher / current Fast | -0.1606055365 | -0.9313701598 | 0.9911185373 |
| LibjxlExact | -0.1610879770 | -0.9327599421 | 0.9911125241 |

Bulk uses positive candidate blocks with target >= 0.05. Tail uses positive
maxima on both sides: 111 of 120 screenshot pairs, as the prior calibration
instrument does. Nine rounded screenshot reconstructions produce zero map
maxima; their exclusion is explicit, not evidence of successful matching.

The target comes from the encoder's floating-point reconstruction before
rounding or clamping. The saved distorted image and external steering server
instead round and clamp that reconstruction to RGBA8. Source:
`zensim-workbench/loop-converge/src/bin/loop_pairs.rs` and
`src/lib.rs::{srgb8,recon_rgba,ServerScorer}` in the companion zensim workspace.
Comparing the resulting maps therefore compares different pixel inputs.

LibjxlExact is an arithmetic mode in recent Butteraugli, not an encoder
strategy. The loop's pinned `d2466a4` predates that mode. Both Zenjxl and Libjxl
strategies use the CPU Butteraugli backend and 80-nit SDR parameters; changing
encoder strategy does not select LibjxlExact arithmetic. The old
`PARITY_PLAN.md` describes an earlier audit. Current numerical parity evidence
is in [libjxl-exact.md](https://github.com/imazen/butteraugli/blob/554a42f832ceffe7a70303d43f3cddcd5a8cd275/benchmarks/libjxl-exact.md)
and the [2026-10-24 diffmap comparison](https://github.com/imazen/butteraugli/blob/554a42f832ceffe7a70303d43f3cddcd5a8cd275/benchmarks/libjxl_parity_2026-10-24/README.md).

## Reproduce and artifacts

From zenmetrics, under the session's compute lock/resource supervisor:

```sh
just margarine-reference-build
just margarine-reference-maps PAIRS_DIR NEW_MAPS_DIR
just margarine-reference-summary PAIRS_DIR MAPS_DIR NEW_SUMMARY.json
```

Observed on dev: `/mnt/v/output/margarine/steering-2026-10-25/reference-maps/`
contains four full-resolution f32 maps per pair and the input index. The strict
JSON summary is `reference-summary-v2.json` in its parent, SHA256
`1615142850cf58ff392b8f1f37deab266667bd4c2c59f853c3cae7c034294b2c`.
The earlier summary is retained but superseded because its tail diagnostic
included zero maxima and emitted nonfinite JSON. Input:
`/mnt/v/zenmetrics-private/aic4/loop-pairs/2026-10-24-hq/`.

Build and clippy passed. Map receipt: `rc=0 17s | peak-RSS 0.07GiB |
min-avail 35287MiB | peak-load 1.62`. Summary/clippy receipt:
`rc=0 6s | peak-RSS 0.40GiB | min-avail 35342MiB | peak-load 0.81`.
No labels were read. No runtime or RSS comparison between implementations was
measured by this audit; these receipts describe only the audit processes.

## Native-linear candidate

`linear-steering` enables the separate `margarine-steer` binary. It retains the
selected row-Malta kernels and geometry. `--server` advertises
`input: linear-rgb-f32` in the JSON header, then accepts R/D messages containing
u32 width/height and tightly packed interleaved linear-sRGB RGB f32le values.
The reply is max score f64le plus a tightly packed native-resolution f32le map.
Finite values outside [0,1] are preserved. Partial messages and nonfinite samples
are errors. `--linear W H REF.f32 DIST.f32 --diffmap NEW.f32` scores saved native
pixels without an image-codec conversion. This command is unselected.

`loop-replay/` re-encodes the frozen calibration references at each recorded
distance. The JXL bitstreams and linear reconstructions are retained. All 324
RGBA8 reconstructions reproduce byte for byte. Native maps reduce to the saved
block targets within 6.584309018009549e-7 using the Python reduction. The replay
requires the read-only jxl-encoder sibling at `19f165bb`; it does not modify it.

| Selected RGBA8 / native-linear | Photo bulk bias | Screenshot bulk bias | Screenshot worst-block log r |
|---|---:|---:|---:|
| Selected, original calibration instrument | -0.1627384826 | -0.8549528266 | 0.9902534190 |
| Native-linear, same kernels | -0.0068962510 | +0.0207127390 | 0.9992006773 |

Native-linear screenshot bulk log r is 0.9941449365; worst-block log bias is
-0.0208095328. All 120 screenshot maxima are positive; the original calibration
uses 111/120 positive pairs. No power/scale/weight was fitted. This meets the
calibration target, but does not establish steering RD or quality/resource
acceptance. Native maps still differ from native Butteraugli by up to
2.3433563709259033 at an individual pixel: this remains an approximation.

Artifacts on dev under the same root: `linear-replay/` (pixels, bitstreams,
native Butteraugli maps), `linear-calibration/` (candidate maps and summary),
`selected-calibration.json`. No human labels were read. Frozen rank/encoder-choice
records are not new native-ingress qualification; no expectation is relaxed.

Replay receipt: `rc=0 36s | peak-RSS 0.10GiB | min-avail 30602MiB | peak-load 3.55`.
Native calibration: `rc=0 4s | peak-RSS 0.16GiB | min-avail 36118MiB | peak-load 2.33`.
Selected calibration: `rc=0 30s | peak-RSS 0.80GiB | min-avail 35084MiB | peak-load 1.51`.

Independent readback (`just margarine-replay-oracle PAIRS_DIR REPLAY_DIR
NEW_OUTPUT.json`) decoded all 324 retained bitstreams with djxl v0.11.1 as
RGB8 at the expected dimensions. It records decoder/JXL/pixel hashes and
diagnostic differences from the calibration pixels. Those pixels are loop
iteration zero; the final bitstream uses the updated quantization field, so
that diagnostic is not a reconstruction-parity check. The calibrated replay's
iteration-zero RGBA reproduction is checked separately by `loop-replay/`.
Native debug/release tests, clippy, release build and readback completed with
`rc=0 14s | peak-RSS 0.33GiB | min-avail 43719MiB | peak-load 2.70`.

## Native-linear steering result

The coordinator's zensim record `9cb02f1b`,
`benchmarks/byconstruction_2026-10-22.md`, section "JPEG XL steering on the
loop's float reconstruction", reports the native-linear `margarinelin` arm.
The runtime is v2 SHA256
`baf0500c1368eda12de21c6e13a29e3aea91e43d917d986f03829b13c275d89a`,
one Rayon thread per steering worker, scale 1/exponent 1. R/D tags are unchanged;
`input: linear-rgb-f32` negotiates the float pixel representation. A first
attempt selected an uncommitted lowercase-tag experiment and failed before
scoring; the results below use the calibrated R/D runtime.

Same 78 TRAIN images, effort 7, four iterations, distances
0.3/0.45/0.6/1/1.6/2.5/4/6.4/10. Positive values mean bytes saved at equal
judged score: mean over images of each image's median saving, with 95%
bootstrap intervals over images. The two steering judges are SSIMULACRA2 and
CVVDP. The following paired results come directly from the retained tables.

| Class (images) | Versus Butteraugli: SSIMULACRA2 % | Versus Butteraugli: CVVDP % |
|---|---:|---:|
| All (78) | +0.00 [-0.13, +0.13] | +0.16 [+0.04, +0.28] |
| Photo (38) | -0.02 [-0.13, +0.09] | +0.15 [+0.08, +0.22] |
| Illustration (22) | -0.16 [-0.54, +0.17] | +0.11 [-0.06, +0.30] |
| Screen (18) | +0.26 [+0.03, +0.48] | +0.24 [-0.19, +0.68] |

| Class | Versus selected RGBA8 arm: SSIMULACRA2 % | Versus selected RGBA8 arm: CVVDP % |
|---|---:|---:|
| All | +1.11 [+0.51, +1.82] | +1.31 [+0.73, +2.03] |
| Photo | +0.37 [-0.00, +0.91] | +0.55 [+0.17, +1.19] |
| Illustration | +0.68 [-0.27, +1.98] | +1.08 [+0.34, +1.99] |
| Screen | +3.18 [+1.28, +5.31] | +3.16 [+1.26, +5.51] |

The native-linear arm meets the brief's positive-or-tied bar under both judges
in every class. Its earlier screen deficit against the Butteraugli loop is
absent on the float inputs. Together with the unchanged kernels and identical
historical RGBA8 reference maps, this identifies ingress rounding/clipping as
the cause of that deficit in this study. No weight or content detector changed.
This does not qualify unrelated rank-loss, harmful-choice, speed or RSS gates,
or establish behavior beyond the measured TRAIN study.

Independent controls: Bcheck2's 702 bitstream hashes match B exactly. All
14,898 earlier rows reproduce under each of SSIMULACRA2, Butteraugli, zensim,
CVVDP and DVIFM, excluding only DVIFM's row-number field. The coordinator also
verified the six recorded 8-bit tail-fit rows before reading linear results.

On dev, `steering-rd-9cb02f1b/` under the artifact root above retains the two
paired tables, current/prior judge tables, loop controls, corpus/run metadata
and all 702 native-arm JXL bitstreams with verified encoded SHA256s. Its
`controls.json` and `MANIFEST.json` preserve the independent checks and hashes.
No sibling source was modified by this lane. The selected command and vendor
check remain unchanged.

## Existing source-verification failure

The dedicated CI lint job fails `verify_sources.py` on the untouched base
`8641dff3` as well as this work: [45e46e998](https://github.com/imazen/zenmetrics/commit/45e46e9986056a0175a92be3d830e56afa6bcc42)
renamed v4 dispatch to v4x in the vendored `blur.rs` and `malta.rs` without
updating the source record. Six modules still match their frozen hashes.
Reversing only `v4x`/`V4x` to `v4`/`V4` reproduces the original hashes of both
changed files exactly. The commit's purpose was to keep one AVX-512 tier:
v4-only processors then fall back to AVX2. This is a dispatch-policy change,
not a perceptual-weight change.

The established check remains unchanged and failing, by coordinator direction.
No source bytes were restored and no additional representation was accepted.
Restoring the two original files from the migration commit passes the existing
verifier in an isolated copy; it would also restore their older v4 dispatch
policy, so that is a selected-behavior decision for the owner. Build, map and
performance effects of that restoration have not been measured here.

A proposed alternative preserves all eight upstream hashes, records the exact
post-rename hashes of these two files, and checks both their exact patched
bytes and the reconstructed upstream bytes. It would accept only those six
unmodified files and those two exact declared dispatch patches; arithmetic
changes or unknown patch kinds would still fail. That proposal has not been
applied and requires owner approval.
