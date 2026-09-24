# hdrvdp × UPIQ HDR real-corpus validation — 2026-09-24

**Result: closed.** The crate reproduces the published per-condition
HDR-VDP-2.2 scores on the full UPIQ HDR subset, and its pooled JOD
correlation clears the published 0.812 bar.

**Host:** r5900xt — AMD Ryzen 9 5900XT (16c/32t), Linux, rustc stable,
workspace `release` profile, **no** `-C target-cpu=native`. Build:
default features (serial kernels; the example's own worker pool
parallelises across pairs, `HDRVDP_JOBS` default = available cores = 32).

**Corpus:** UPIQ dataset (`upiq_dataset.zip`, ~2.4 GB, from tower NFS
export `v-datasets-archives-2026-07-22/upiq/`), extracted at
`/home/lilith/tmp/upiq/upiq_dataset`. HDR subset = **380 conditions**
(narwaria 140, korshunov 240), absolute-luminance EXR pairs
(1920×1080), joined per `condition_id` between
`upiq_subjective_scores.csv` (JOD truth) and `upiq_objective_scores.csv`
(official per-condition `HDRVDP2_2`).

**Command** (all 380 pairs, 0 failures, 2m03s wall — ~9.8 s CPU/pair):

```sh
cargo build -p hdrvdp --release --example upiq_score
HDRVDP_PPD=30 ./target/release/examples/upiq_score \
    /home/lilith/tmp/upiq/upiq_dataset out.tsv luminance
python3 scripts/hdr/upiq_hdrvdp_report.py out.tsv
```

Raw rows: [`hdrvdp_upiq_2026-09-24.tsv`](hdrvdp_upiq_2026-09-24.tsv).

## Reproduction protocol (the part that was not obvious)

The released `HDRVDP2_2` column was generated with:

- **luminance input** — BT.709 Y of the EXRs' absolute RGB (`luminance`
  encoding), NOT `rgb-bt.709` (rank-identical but ~0.17 further off
  absolutely);
- **fixed `pix_per_deg = 30`** — NOT the subjective CSV's `pix_per_deg`
  column (56.5487 narwaria / 60.3186 korshunov — feeding it lands ~12
  points low). 30 ≈ UPIQ's own ~50 ppd viewing geometry applied to
  half-scale effective pixels, and equals HDR-VDP's canonical default;
- **full resolution** — downsampling the images ×0.5 (the competing
  hypothesis for the apparent ~0.5 ppd factor) is falsified: it scores
  SROCC 0.944 vs official, full-res 0.996;
- default `surround_l` (1e-5).

Falsified en route: CSV `pix_per_deg` direct (−11.7 mean delta),
`0.5×`/`0.55×` CSV ppd (fits one dataset, biases the other — the two
datasets want 0.531 and 0.497, i.e. the constant is 30, not a ratio),
half-resolution input, `rgb-bt.709` feeding, `surround_l =` geometric
mean of the reference (no material effect).

## Headline numbers (n = 380)

| comparison | SROCC | PLCC |
|---|---:|---:|
| **ours vs official `HDRVDP2_2`** | **0.9962** | **0.9984** |
| **ours vs JOD** | **0.8203** | 0.7729 |
| official `HDRVDP2_2` vs JOD (same join) | 0.8117 | 0.7681 |
| published HDR-VDP-2.2 bar | 0.812 | — |

Per-image delta (ours − official): **+0.046 ± 0.363**, min −1.200,
max +2.216. The official column vs JOD reproduces the published 0.812
(0.8117 measured) on this join, so the harness and the CSV join are
right; the residual is per-image protocol noise, not a systematic port
bias.

Ours landing *above* the official column vs JOD (0.8203 vs 0.8117) is
expected rather than suspicious: the ~±0.36 JOD-scale deltas re-rank a
few near-tied pairs, and pooled across two datasets with different JOD
scales the pooled SROCC is offset-sensitive.

## Per dataset

| subset | n | ours vs off. SROCC | ours vs off. PLCC | delta mean±sd | ours vs JOD | off. vs JOD |
|---|---:|---:|---:|---:|---:|---:|
| korshunov | 240 | 0.9999 | 0.9999 | −0.020 ± 0.094 | 0.9484 | 0.9485 |
| narwaria | 140 | 0.9916 | 0.9950 | +0.159 ± 0.567 | 0.8745 | 0.8857 |

Korshunov is essentially exact — median |delta| 0.013, 238/240 within
±0.2 (worst outlier −1.2). Narwaria: median |delta| 0.020, 127/140
within ±0.2; the residual concentrates in the `n-i07` family — a run of
~+2.0 deltas on one reference — plausibly a per-image protocol detail in
the upstream score generation (e.g. a slightly different effective
geometry for that batch); it moves the pooled mean, not the rank
structure.

## What this proves vs the golden tests

- Golden tests (`tests/golden_2_2_2.rs`): implementation parity vs
  official 2.2.2 on 24 *synthetic* cases — the correctness gate.
- This run: parity vs official scores on *real* corpus content **plus**
  the feeding convention (luminance, ppd 30, full-res) that makes the
  published number reproducible — the chunk-4 gate. Both are now green.

## Reproduce

The scorer is `crates/hdrvdp/examples/upiq_score.rs`; analysis is
`scripts/hdr/upiq_hdrvdp_report.py` (ours-vs-JOD, official-vs-JOD,
ours-vs-official, per-image deltas). Probe knobs retained for
regression: `HDRVDP_PPD` (fixed), `HDRVDP_PPD_SCALE` (×CSV ppd),
`HDRVDP_IMG_SCALE` (image downsample), `HDRVDP_SURROUND=geom`,
`HDRVDP_LIMIT`/`HDRVDP_SKIP`, `HDRVDP_JOBS`, encoding arg.
