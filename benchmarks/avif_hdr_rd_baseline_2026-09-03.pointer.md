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

mirrored 12 objects to s3://zentrain/analysis/avif-hbd-t2-2026-09-03/

| file | bytes | sha256 |
|---|--:|---|
| `t2_analysis/t2_arm_summary.tsv` | 728 | `88bde486dfe096422b7abfeb780794e8490a247dc6425d06ceb050a41e34775c` |
| `t2_analysis/t2_bd_matrix.tsv` | 9936 | `da483dee22d3f6bd896da1989d9d474e1ffa19898cf48e1fcd81e660cbb7df7d` |
| `t2_analysis/t2_bd_per_image.tsv` | 249134 | `229f01bedfba7f370201969832d3296a8d87cf1e7d02040f9573260c8fd275ff` |
| `t2_analysis/t2_envelope_by_class.tsv` | 825 | `c1337d60c7d06096a0f0a2f7fe821eee3d11a4a65ae4a8bebbf5f282e6d0f86e` |
| `t2_analysis/t2_envelope_contrast.tsv` | 253 | `0e5e7b26363314d6ec571e9533ea3309be080aaa4a175a32a902714a3d5055a9` |
| `t2_analysis/t2_envelope_per_image.tsv` | 4866 | `9aec7821ee767bd7d63fd0c444fff66c7dfce4ceff45c565b1b2b1bc18118664` |
| `t2_analysis/t2_primaries_content_crosstab.tsv` | 183 | `59895222c7a219cd5b9df284e409c9e8f7fa27786fa5246dc05a28518c90ad34` |
| `t2_analysis/t2_rd_baseline.tsv` | 17245 | `34811ed12bd4047139b24fd80ddaa116e6e98f2f3a8bb293ec9861092b913bf9` |
| `t2_analysis/t2_rd_points.tsv` | 711392 | `cc9b4782ac10b1b3676a9712a4d0f31bcbd456fd3ef087e4e03ee6edb8812af9` |
| `t2_analysis/_t2_summary.json` | 475 | `73ae82dab27d738117b597af36bb552379ba0d1dceb5311eaecfb9e8c7338f79` |
| `t2_preera_blobs.txt` | 3120 | `c59f5bedd35d2f0ba1621e60e303322ff7b3ba9ebc9b5c0db1036cf7edc470fa` |
| `t2_scored_2026-09-03.parquet` | 255776 | `d60a3bec40a2a1653d6bbbdc1f22bdadac9deb56144acf2a46c019db6f0e21e8` |