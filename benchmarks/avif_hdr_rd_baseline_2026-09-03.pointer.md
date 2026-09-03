# AVIF HDR-10 RD baseline (Track T2) — outputs — pointer

Record: [`avif_hdr_rd_baseline_2026-09-03.md`](avif_hdr_rd_baseline_2026-09-03.md).
Nothing here is in git (>30 KB rule).

**Locations**

| what | where |
|---|---|
| analysis outputs (local) | `/mnt/v/output/zensim-avifhbd-t2/` |
| analysis outputs (LAN store) | `s3://zentrain/analysis/avif-hbd-t2-2026-09-03/` |
| raw score blobs | `s3://zentrain/jobs/avifhbd-t2-sf-cpu-20260902/blobs/` |
| encode bitstreams (svt) | `s3://zentrain/jobs/avifhbd-t2a-fix-20260902/blobs/` |
| encode bitstreams (zenrav1e) | `s3://zentrain/jobs/avifhbd-t2b-20260902/blobs/` |
| HDR PQ-PNG references (16, K=16 picks) | `s3://codec-corpus/avif-hbd-t2-refs-2026-09-02/` |
| K=16 picks + primaries/content cross-tab source | `benchmarks/avif_hdr_t2_picks_k16_2026-09-02.tsv` (in git) |
| ⛔ the pre-#18-fix svt run — **invalid, never read, never pooled** | `s3://zentrain/jobs/avifhbd-t2a-20260902/` (paused at 120/3,248) |

**Regeneration** (scripts in `zenmetrics/scripts/jobsys/`)

```sh
# 0. score declaration. ZEN_DOE_SCORE_HDR=1 IS LOAD-BEARING: an HDR wave declared
#    without --hdr poisons every job with `encoder_panic` (46 lost that way once).
ZEN_DOE_SFRUN=avifhbd-t2-sf-cpu-20260902 ZEN_DOE_SCORE_HDR=1 \
ZEN_DOE_RUNS="avifhbd-t2a-fix-20260902=s3://codec-corpus/avif-hbd-t2-refs-2026-09-02/ \
              avifhbd-t2b-20260902=s3://codec-corpus/avif-hbd-t2-refs-2026-09-02/" \
  avifdoe_score_gapfill.sh
ZEN_CORPUS_BUCKET=codec-corpus ZEN_CORPUS_PREFIX=avif-hbd-t2-refs-2026-09-02 \
  lan_score_launch.sh root@tower avifhbd-t2-sf-cpu-20260902 t2sfb cpu \
  ghcr.io/imazen/zenfleet-worker:exec-avifhbd-t2fix-64252836

# 1. harvest — the HDR knob tuples are single-dial ({"preset":N} / {"speed":N});
#    avifdoe_harvest.py labels them and fills `dial` / `dial_kind`.
avifdoe_harvest.py --score-dir <blobs> --sizes <sizes.tsv> \
    --pairs t2a=<t2a-fix pairs.tsv> --pairs t2b=<t2b pairs.tsv> --out t2_scored.parquet

# 2. the Q5/Q6 reduction
avifhbd_t2_analyze.py --scored t2_scored.parquet \
    --picks benchmarks/avif_hdr_t2_picks_k16_2026-09-02.tsv \
    --outdir t2_analysis --parity-check ~/work/zen/zenavif/scripts/rd_gap/bd_arm.py
```

**G5 route proof (reproduce it before citing any number)** — inside the scoring
image, on any T2 pair:

```sh
docker exec <scoring-container> bash -c '
  cd /tmp && printf "ref_path\tdist_path\n<ref.png>\t<dist.avif>\n" > pairs.tsv
  for T in pq pu-rescale; do
    zenmetrics score-pairs --metric ssim2 --pairs-tsv pairs.tsv --hdr \
      --hdr-transfer $T --out-parquet out_$T.parquet
  done'
```

`--hdr-transfer` must be **inert** (|Δ| exactly 0) — that is the faithful f32
route. The u8 shell's own test asserts `|pu − pq| > 1e-9`, so a non-zero delta
means the run was shelled and its ssim2/zensim only order encodes coarsely.
Scoring the same pair **without** `--hdr` must be **refused loudly** (the
2026-09-02 `e9e2ef71` tripwire), not silently narrowed to 8 bits.

<!--TABLE:FILES-->
