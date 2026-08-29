# HDR-VDP-2.2 port — first end-to-end scores on real pixels (2026-08-28)

First run of `crates/hdrvdp` (the `f64` CPU reference port, imazen/zenmetrics#50)
over a real image and real codec distortion. Data:
[`hdrvdp2_corpus_ladder_2026-08-28.tsv`](hdrvdp2_corpus_ladder_2026-08-28.tsv).

## What this is — and, more importantly, what it is not

**It is** proof that the whole pipeline runs on real pixels and behaves like a
quality metric: display model → optical MTF → photoreceptor spectral mixing →
JND-space non-linearity → steerable pyramid → contrast masking → pooling →
`Q_MOS`.

**It is NOT** a validation of the port. Nothing here compares against the
reference MATLAB implementation, so the *magnitudes* below are unverified — only
their ordering is. The number that settles whether this port is HDR-VDP-2 is the
UPIQ HDR SROCC against the published **0.812**, and that measurement has not been
made (see "Blocker" below). **Do not quote a `Q_MOS` from this table as an
HDR-VDP-2 score.**

## Setup

| | |
|---|---|
| crate | `hdrvdp` 0.0.1, commit `5f597a3e` (chunk 3) |
| harness | `cargo run --release -p hdrvdp --example score_corpus_ladder` |
| source | `zenmetrics-corpus` `source.png`, 256×256 RGB |
| distortions | the corpus' own JPEG ladder, q1 / q5 / q20 / q45 / q70 / q90 |
| viewing | 24″ 1920×1200 at 0.5 m → **33.510 pixels/degree** |
| SDR feed | HDR-VDP-2's `srgb-display` model: sRGB EOTF onto 99 cd/m² peak + 1 cd/m² black |
| HDR feed | the same content presented at a **1000 cd/m² peak**: sRGB EOTF → display-linear → ×1000 cd/m², fed as `rgb-bt.709` in absolute units |
| host | Apple aarch64, macOS 26.5.2, rustc 1.98.0 |

The "HDR" column is an SDR photograph shown at HDR display luminances — not a
native HDR capture. It exercises the absolute-luminance path on real pixels and
shows the verdict moving with presentation luminance; it is not a substitute for
scoring real HDR content.

## Results

| JPEG q | SDR `Q_MOS` | SDR `C_max` | HDR `Q_MOS` | HDR `C_max` |
|---:|---:|---:|---:|---:|
| 1 | 60.42 | 16 210.9 | 65.12 | 146 000.5 |
| 5 | 81.83 | 5 735.3 | 80.53 | 51 194.2 |
| 20 | 93.14 | 943.2 | 91.52 | 6 403.5 |
| 45 | 96.37 | 300.1 | 94.72 | 2 152.0 |
| 70 | 97.39 | 149.0 | 95.72 | 992.9 |
| 90 | 98.97 | 27.2 | 98.68 | 178.3 |

`P_det` is 1.000000 in every cell and is omitted from the table — see below.

## Four observations, all of them checkable

1. **`Q_MOS` is strictly monotone in JPEG quality on both feeds**, across two
   decades of distortion, with no inversions. That is the minimum bar for a
   quality metric and it is met on real codec artefacts, not synthetic gratings.
2. **`C_max` falls monotonically** and spans ~600× from q1 to q90 — the
   difference magnitude tracks the distortion over the whole ladder rather than
   saturating.
3. **`P_det` saturates at 1.0 everywhere**, which is expected and not a defect:
   spatial pooling multiplies the map by `sum/max`, so the pooled maximum equals
   the *un-pooled sum*. Any distortion with real spatial extent therefore drives
   `P_det` to 1. `P_det` discriminates near the visibility threshold, not across
   a supra-threshold codec ladder; `Q_MOS` is the axis to read here.
4. **The HDR and SDR curves cross.** At q1 the HDR presentation scores *higher*
   (65.1 vs 60.4) and at q90 *lower* (98.68 vs 98.97). The metric is not applying
   a constant offset for luminance — the model's adaptation and masking terms
   respond to presentation luminance differently at different distortion levels.
   Whether that crossing matches the reference implementation is exactly the kind
   of thing the UPIQ measurement has to settle.

## Blocker on the real validation

The UPIQ HDR subset (380 pairs + JOD ground truth) lives at
`/mnt/v/datasets/upiq_extracted/` on the Linux dev box. The machine this port was
written on is an aarch64 laptop with no `/mnt/v`, so chunk 4's SROCC measurement
must run where UPIQ lives (or the dataset must be mirrored first). The harness to
run there already exists: `scripts/hdr/upiq_corr.py`.

## Reproducing

```sh
cargo run --release -p hdrvdp --example score_corpus_ladder
```

A reduced 128×128, three-rung version of the same ladder runs as a normal test
(`crates/hdrvdp/tests/corpus_ladder.rs`, ~6 s in debug) and gates the
monotonicity on every CI run.
