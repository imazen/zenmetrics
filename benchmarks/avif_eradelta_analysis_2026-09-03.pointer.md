# AVIF era-delta wave — outputs — pointer

Record: [`avif_eradelta_analysis_2026-09-03.md`](avif_eradelta_analysis_2026-09-03.md).
Nothing here is in git (>30 KB rule).

**Locations**

| what | where |
|---|---|
| analysis outputs (local) | `/mnt/v/output/zensim-avifdoe-eradelta/` |
| analysis outputs (LAN store) | `s3://zentrain/analysis/avif-eradelta-2026-09-03/` |
| raw score blobs | `s3://zentrain/jobs/avifdoe-svt-eradelta-sf-cpu-20260903/blobs/` |
| encode bitstreams | `s3://zentrain/jobs/avifdoe-svt-eradelta-{a1,b1,c1}-20260903/blobs/` |
| the OLD-era runs this differences against | `s3://zentrain/jobs/avifdoe-svt-{a1,ag}-20260901/blobs/`, `s3://zentrain/jobs/avifdoe-svt-t1d-20260902/blobs/` |
| reference corpora | `s3://codec-corpus/avif-doe-1024-2026-09-01/` (crop — a1, b1) and `s3://codec-corpus/avif-subsample-2026-09-01/` (native — c1) — **same 32 filenames, different pixels on 13 of them; never conflate** |

**Regeneration** (all scripts in `zenmetrics/scripts/jobsys/`)

```sh
# 0. score-side declaration + a worker. The run->refs-prefix map is the hazard:
#    a1/b1 are the CROP corpus, c1 is the NATIVE one.
ZEN_DOE_SFRUN=avifdoe-svt-eradelta-sf-cpu-20260903 \
ZEN_DOE_RUNS="avifdoe-svt-eradelta-a1-20260903=s3://codec-corpus/avif-doe-1024-2026-09-01/ \
              avifdoe-svt-eradelta-b1-20260903=s3://codec-corpus/avif-doe-1024-2026-09-01/ \
              avifdoe-svt-eradelta-c1-20260903=s3://codec-corpus/avif-subsample-2026-09-01/" \
  avifdoe_score_gapfill.sh
lan_score_launch.sh lilith@192.168.50.27 avifdoe-svt-eradelta-sf-cpu-20260903 edsf cpu \
  ghcr.io/imazen/zenfleet-worker:exec-avifdoe-b6-6b3c41fe

# 1. harvest (era-delta wave ONLY — never pooled with Stage-A rows)
avifdoe_harvest.py --score-dir <blobs> --sizes <sizes.tsv> \
    --pairs a1=… --pairs b1=… --pairs c1=… --out eradelta_scored_2026-09-03.parquet

# 2. BD-rate + the paired matched-q read, IN-RUN control (same instrument as stagea_inrun/)
avifdoe_stagea_analyze.py --scored eradelta_scored_2026-09-03.parquet \
    --crop-manifest /mnt/v/output/avif-doe-1024-2026-09-01/crop_manifest_2026-09-01.tsv \
    --control inrun --outdir eradelta_inrun \
    --parity-check ~/work/zen/zenavif/scripts/rd_gap/bd_arm.py

# 3. cross-era identity + scorer-drift + effect stability (arm-set A replication)
avifdoe_era_compare.py --old a1=<stageA a1 pairs> --new a1=<eradelta a1 pairs> \
    --old-scored /mnt/v/output/zensim-avifdoe/doe_scored_2026-09-02.parquet \
    --new-scored eradelta_scored_2026-09-03.parquet \
    --effects-old /mnt/v/output/zensim-avifdoe/stagea_inrun/main_effects.tsv \
    --effects-new eradelta_inrun/main_effects.tsv \
    --label-old ef0b122b --label-new 2ca060f4 --outdir era_a1

# 4. arm-set C vs the superseded t1d block
avifdoe_era_compare.py --old c1=<t1d pairs> --new c1=<c1 pairs> \
    --label-old ef0b122b --label-new 2ca060f4 --outdir era_c1

# 5. c1's paired bd10-vs-8bit read, with AG supplying the native 8-bit control
avifdoe_harvest.py --score-dir <merged c1+AG blobs> --sizes <c1+AG sizes> \
    --pairs ag=<ag pairs> --pairs c1=<c1 pairs> --pairs t1d=<t1d pairs> --out c1_ag_scored.parquet
avifdoe_stagea_analyze.py --scored c1_ag_scored.parquet \
    --crop-manifest /mnt/v/output/avif-doe-1024-2026-09-01/crop_manifest_2026-09-01.tsv \
    --control inrun --paired-control-run ag --outdir c1_paired
# the registered n = 13 restriction is then a column filter:
awk -F'\t' 'NR==1 || $NF=="crop-native"' c1_paired/paired_per_q.tsv
```

**Instrument note.** Step 2 uses `--control inrun`, matching
`/mnt/v/output/zensim-avifdoe/stagea_inrun/` — **not** `stagea_a0r/`. The A0R
dense 29-q control does not exist in this era, and differencing an in-run-9q
effect against an a0r-dense one would confound instrument with era.

<!--TABLE:FILES-->
