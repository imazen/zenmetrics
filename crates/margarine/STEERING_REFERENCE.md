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
