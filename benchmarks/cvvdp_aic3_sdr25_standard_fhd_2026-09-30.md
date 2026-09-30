# CVVDP rescore at standard_fhd — AIC-3 and SDR25 — 2026-09-30

## Result

Both corpora scored with our native-CPU CVVDP port at the JPEG AIC organisers'
`standard_fhd` display (37.843 pixels/degree) — the same display-configuration
fix validated on AIC-4 in `cvvdp_aic_discrepancy_2026-09-22.md`. This closes
the "AIC-3 and SDR25 were not re-scored" gap that record's note left open.

| corpus | n pairs | \|SROCC\| | \|PLCC\| |
|---|---|---|---|
| AIC-3 (`aic3_pairs_ab.tsv`) | 600 | **0.8246** | 0.7830 |
| SDR25 (`sdr25_pairs.tsv`) | 50 | **0.9464** | 0.9556 |

Both human-score columns are the corpora's own **signed JND** scale
(`build_aic3`/`build_sdr25` in `zensim/scripts/canonical_corpus/build_fr_corpus_pairs.py`
document "signed JND; verdict uses \|SROCC\| so orientation is fine" — same
convention as AIC-4). Raw Spearman ρ was +0.8246 for AIC-3 and −0.9464 for
SDR25; the sign differs between the two corpora's own JND zero-points, which
is expected and immaterial under the \|ρ\| convention this repo already uses
for every signed-JND corpus (AIC-3, AIC-4, KonJND).

No organisers' reference CVVDP column exists for AIC-3 or SDR25 in this corpus
(unlike AIC-4, which ships one) — this record reports our port's own SROCC at
the correct display, not a discrepancy-vs-organisers comparison.

## 1. Data

| item | path | pairs |
|---|---|---|
| AIC-3 CTC held-out gate pairs | `/mnt/v/output/zensim/v2-ab-2026-07-19/aic3_pairs_ab.tsv` | 600 |
| SDR25 (JPEG-AI-SDR25 scoreable subset) pairs | `/mnt/v/output/zensim/v2-backfill-2026-07-20/sdr25_pairs.tsv` | 50 |

## 2. Method

```
zenmetrics score-pairs --metric cvvdp --display-model standard_fhd \
  --pairs-tsv <corpus>_pairs*.tsv --out-parquet <corpus>_cvvdp_standard_fhd.parquet
```

- Binary: `zenmetrics-cli` built `--release --target x86_64-unknown-linux-musl
  --no-default-features --features sweep,png,jpeg,webp,avif,jxl,cpu-metrics`
  from the `scripts/ci/lock.sh`-generated CI-pinned sibling snapshot (23
  siblings at their `ci/sibling-pins.tsv` commits), at zenmetrics master
  `0a61830db89a`.
- 0 failed pairs on both corpora (600/600 and 50/50 scored). Score column
  `cvvdp_cpu_imazen_v0_1_0_standard_fhd`.
- SROCC/PLCC computed against each TSV's `human_score` column, positionally
  joined 1:1 against `score-pairs`' output rows (row-order preserved; verified
  0 mismatches between output `image_path` and input `ref_path` on both
  corpora before trusting the join).
- This is a direct rescore, not a fleet job: 650 pairs total is well inside
  single-box CPU capacity (AIC-3: 143s wall, peak RSS 1.33 GiB; SDR25: 2s,
  peak RSS 0.14 GiB — both under `run-heavy --mem 24G`), so it ran locally
  rather than through zenfleet's declare/claim/ledger path. The same build
  also produced a static executor image
  (`ghcr.io/imazen/zenfleet-worker:exec-cvvdp-0a61830d`, smoke-tested with
  `jobexec --help`) for the larger CID22/UPIQ CVVDP-everywhere lane
  (`paper-cvvdp`), which does need the fleet.

## 3. Provenance

- Command log: `/var/tmp/cvvdp-rescore/rescore-run.log`.
- Raw outputs: `/var/tmp/cvvdp-rescore/{aic3,sdr25}_cvvdp_standard_fhd.parquet`
  (not committed — regenerate with the command in §2; both runs took under
  2.5 minutes combined).
- SROCC/PLCC computation: ad hoc `scipy.stats.spearmanr`/`pearsonr` against
  the positionally-joined parquet + TSV; not yet a committed script (the
  corpus is small enough that this was a one-off — worth promoting to a
  reusable tool if a third signed-JND corpus needs the same treatment).
