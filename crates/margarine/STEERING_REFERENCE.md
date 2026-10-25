# Steering reference audit (2026-10-25)

The selected command and frozen teacher are unchanged. Native-linear steering
qualification is in progress; no quality or resource acceptance is claimed.

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
