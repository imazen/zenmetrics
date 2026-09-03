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
| the OLD-era runs this differences against | `s3://zentrain/jobs/avifdoe-svt-{a1,a2,ag}-20260901/blobs/`, `s3://zentrain/jobs/avifdoe-svt-t1d-20260902/blobs/` |
| the OLD-era SCORE runs read for those | `s3://zentrain/jobs/avifdoe-svt-sf-cpu-20260902/blobs/` (Stage-A) and `s3://zentrain/jobs/avifdoe-svt-t1-sf-cpu-20260902/blobs/` (the superseded `t1d` half of §4.4) |
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

# 2. BD-rate + the paired matched-q read, IN-RUN control (same instrument as
#    stagea_inrun/). ONE RUN AT A TIME: a1 and b1 share 3,456 cell identities,
#    so pooling them would enter each of those images twice.
for R in a1 b1; do
  avifdoe_stagea_analyze.py --scored eradelta_scored_2026-09-03.parquet --runs "$R" \
      --crop-manifest /mnt/v/output/avif-doe-1024-2026-09-01/crop_manifest_2026-09-01.tsv \
      --control inrun --outdir "eradelta_inrun_$R" \
      --parity-check ~/work/zen/zenavif/scripts/rd_gap/bd_arm.py
done

# 3. cross-era identity + scorer-drift + effect stability (arm-set A replication)
avifdoe_era_compare.py --old a1=<stageA a1 pairs> --new a1=<eradelta a1 pairs> \
    --old-scored /mnt/v/output/zensim-avifdoe/doe_scored_2026-09-02.parquet \
    --new-scored eradelta_scored_2026-09-03.parquet \
    --effects-old /mnt/v/output/zensim-avifdoe/stagea_inrun/main_effects.tsv \
    --effects-new eradelta_inrun_a1/main_effects.tsv \
    --label-old ef0b122b --label-new 2ca060f4 --outdir era_a1

# 3b. arm-set B: its s4 legs joined against Stage-A a1, its s6 legs against a2,
#     both identity joins feeding ONE verdict table (its s7 legs have no
#     Stage-A counterpart and come out NOT-MEASURED-old, which is correct).
avifdoe_era_compare.py \
    --old s4=<stageA a1 pairs> --new s4=<eradelta b1 pairs> \
    --old s6=<stageA a2 pairs> --new s6=<eradelta b1 pairs> \
    --effects-old /mnt/v/output/zensim-avifdoe/stagea_inrun/main_effects.tsv \
    --effects-new eradelta_inrun_b1/main_effects.tsv \
    --label-old ef0b122b --label-new 2ca060f4 --outdir era_b1_stability

# 4. arm-set C vs the superseded t1d block
avifdoe_era_compare.py --old c1=<t1d pairs> --new c1=<c1 pairs> \
    --label-old ef0b122b --label-new 2ca060f4 --outdir era_c1

# 5. c1's paired bd10-vs-8bit read, with AG supplying the native 8-bit control
avifdoe_harvest.py --score-dir <merged c1+AG blobs> --sizes <c1+AG sizes> \
    --pairs ag=<ag pairs> --pairs c1=<c1 pairs> --pairs t1d=<t1d pairs> --out c1_ag_scored.parquet
avifdoe_stagea_analyze.py --scored c1_ag_scored.parquet \
    --crop-manifest /mnt/v/output/avif-doe-1024-2026-09-01/crop_manifest_2026-09-01.tsv \
    --control inrun --runs c1,t1d --paired-control-run ag --outdir c1_paired
# the registered n = 13 restriction is then a column filter:
awk -F'\t' 'NR==1 || $NF=="crop-native"' c1_paired/paired_per_q.tsv
```

**Instrument note.** Step 2 uses `--control inrun`, matching
`/mnt/v/output/zensim-avifdoe/stagea_inrun/` — **not** `stagea_a0r/`. The A0R
dense 29-q control does not exist in this era, and differencing an in-run-9q
effect against an a0r-dense one would confound instrument with era.

| file | bytes | sha256 |
|---|--:|---|
| `c1_ag_scored_2026-09-03.parquet` | 88562 | `db659c0804d3916d883d4a0e27266e199fd6b5520f7680b0ac74051d2d0a6c27` |
| `c1_paired/arm_byte_identity.tsv` | 575 | `7f7d53548d55926fcf1deaff8a78adabaffcf724aeb80c7095c09b32eaa3bed5` |
| `c1_paired/bd_per_image.tsv` | 61 | `f3f17ab0101e2b2bc95528fa35119718bad3ee2665ea91ff55272ca9469940ce` |
| `c1_paired/interactions_per_image.tsv` | 48 | `b743fa896f83fc54bd50def694e9c208e37873a5a94baf5014cf74ce951feec7` |
| `c1_paired/interactions.tsv` | 77 | `d41f0ecc49c47aace4c282bf58309d45f3fbee9322be17c8fb2147e767fa01f5` |
| `c1_paired/main_effects_by_class.tsv` | 49 | `848444223a0a5f29517d2db7e35d6a4a47639ac4a7530d911f7bdf53caed08a3` |
| `c1_paired/main_effects.tsv` | 76 | `9b1e736c889a52b415fbfc5dfef112562d25ed3526571c06b09d3fae5b0e9525` |
| `c1_paired/paired_per_q.tsv` | 21937 | `591d5dbcc1d2419a941bbff0a52569ec48e89bad660a27cc70af8559fe621baa` |
| `c1_paired/paired_summary_by_transform.tsv` | 1088 | `b81bd1862e806c5b3df8b5f2ac59e27dd7dc42290986880a6eabdf9233c4e482` |
| `c1_paired/paired_summary.tsv` | 534 | `03697cba2eaea07ec3d65baf54bd8f34667748e037e35579f292b3bd6f0c93f3` |
| `c1_paired/_summary.json` | 108 | `3e2fcd243a2fdf53255f5a441dbde474b5270a6b85533f6f19cea4f210b16ba8` |
| `era_a1/era_cell_diffs.tsv` | 45 | `e1f81a6bfee340c16fbbf4f8e710f13db04a1241ebb45e2e02c0dd0757df7b12` |
| `era_a1/era_effect_stability.tsv` | 2685 | `b4313ab43b92b3a2819cbf31ee688cf46d2e39db73b263254bc0abfbf3fbc67c` |
| `era_a1/era_identity_summary.json` | 887 | `1d405d24790ca9c4168468460fb544516292337e9cdc72a074a8dbe7677b7ced` |
| `era_a1/era_score_drift.tsv` | 63 | `44a6b6301ba3cc4fd588dbcd2fb38e87fa632470cbc1e7e8578cd6d2f0316999` |
| `era_a1/era_stratum_identity.tsv` | 972 | `d4d22b33ef3c1e7d0e9cb8f4877d50efb31351c4e3f5849e11d56925454af71f` |
| `era_b1_stability/era_cell_diffs.tsv` | 45 | `e1f81a6bfee340c16fbbf4f8e710f13db04a1241ebb45e2e02c0dd0757df7b12` |
| `era_b1_stability/era_effect_stability.tsv` | 3145 | `cae0c707795bc8f3046f40aa5e1313c0ab3a7cba52312e475b44d9dd08b7fe34` |
| `era_b1_stability/era_identity_summary.json` | 1300 | `c6ad5d78789ae385b06a9550b44bd36da8cff235502c427f3d2028aea9f1fd03` |
| `era_b1_stability/era_stratum_identity.tsv` | 928 | `73462b08b98e47267ff4b36c0beeb1c862aa71aba26358187867819af39c936d` |
| `era_b1_vs_a1/era_cell_diffs.tsv` | 45 | `e1f81a6bfee340c16fbbf4f8e710f13db04a1241ebb45e2e02c0dd0757df7b12` |
| `era_b1_vs_a1/era_identity_summary.json` | 615 | `6eb50509d03c9fb07841ec04ed6550313a237969aa2f4e375603ddba743fba77` |
| `era_b1_vs_a1/era_stratum_identity.tsv` | 529 | `46cef3ffd06567201896b5c9001916e4782545b89968ef5bfb08bb7e10c9c3ad` |
| `era_b1_vs_a2/era_cell_diffs.tsv` | 45 | `e1f81a6bfee340c16fbbf4f8e710f13db04a1241ebb45e2e02c0dd0757df7b12` |
| `era_b1_vs_a2/era_identity_summary.json` | 622 | `ae084b0705367b4808bd9c9ae7c538abd7a9947556844be7d1dc3acc8aeb2369` |
| `era_b1_vs_a2/era_stratum_identity.tsv` | 463 | `a74884cec2ac2d70c076ce51827a39cc0b05d413d5f59dc931008cd5cdfe3593` |
| `era_c1/era_cell_diffs.tsv` | 1965 | `1603218548025978b3989fd924ed1d6f8b96538f46d41f8fd982b7f7b3fa0895` |
| `era_c1/era_identity_summary.json` | 882 | `74df49c88974c3d92b109a08a010f40b99a4b251905403a03428dad8b84efae0` |
| `era_c1/era_stratum_identity.tsv` | 101 | `9f9dc3b5245441c5056ea0b5eed5e1f1cf624f1cc15b5b3cc28738e5737e4b74` |
| `eradelta_inrun_a1/arm_byte_identity.tsv` | 630 | `a2e05085acc729ba053f73b2bf1f7cfd2c87ea9bf213f90d7248b16dafcc54cf` |
| `eradelta_inrun_a1/bd_per_image.tsv` | 51990 | `38dcbf8757739d38fae2c4517f29a0e6115323024aedb476bb147beea03781c6` |
| `eradelta_inrun_a1/interactions_per_image.tsv` | 48 | `b743fa896f83fc54bd50def694e9c208e37873a5a94baf5014cf74ce951feec7` |
| `eradelta_inrun_a1/interactions.tsv` | 77 | `d41f0ecc49c47aace4c282bf58309d45f3fbee9322be17c8fb2147e767fa01f5` |
| `eradelta_inrun_a1/main_effects_by_class.tsv` | 4708 | `d5c53bfd705f334c2a5792df7228b7784def4f3b5d6499722e3f2369caad14c6` |
| `eradelta_inrun_a1/main_effects.tsv` | 1457 | `33f688d291b32befc71001429b8b920831dd0b5765a1a15b1df7a5a7b6a2d0da` |
| `eradelta_inrun_a1/paired_per_q.tsv` | 552585 | `4a1df70a9ea3392e13d8b575e6ccfadca3ae29f35251632665622c2d598089b0` |
| `eradelta_inrun_a1/paired_summary_by_transform.tsv` | 23988 | `b0a926b9cca8b370d7ff8aa3a34798dc399faae772d029fe799c79ffdb9f9f32` |
| `eradelta_inrun_a1/paired_summary.tsv` | 10634 | `6d33e92b35169b1aa6916d70cbe94ee2b932f76a7db940f6658e98bd64b1a388` |
| `eradelta_inrun_a1/_summary.json` | 133 | `a71bf637607e6f07884b9188884e7eba980801e7fb353b5b0f894f2aefe5a17a` |
| `eradelta_inrun_b1/arm_byte_identity.tsv` | 989 | `1d13ff742e655d5e7c8f5a41f95bb1ad18bb6a3d0a54897d1202b997f10b597b` |
| `eradelta_inrun_b1/bd_per_image.tsv` | 85882 | `3b65920821e311e92cff10c79e7de7a2b4388ea8123edb0b505166c3e2ae405b` |
| `eradelta_inrun_b1/interactions_per_image.tsv` | 48 | `b743fa896f83fc54bd50def694e9c208e37873a5a94baf5014cf74ce951feec7` |
| `eradelta_inrun_b1/interactions.tsv` | 77 | `d41f0ecc49c47aace4c282bf58309d45f3fbee9322be17c8fb2147e767fa01f5` |
| `eradelta_inrun_b1/main_effects_by_class.tsv` | 7689 | `7d62ff7639b1ae222bb677b7e1f826f567e224d15fc0b4b1e8f895596d465f4f` |
| `eradelta_inrun_b1/main_effects.tsv` | 2340 | `bf47d4aaff2dc3e45b379d375fe0ee687ff927543e540a9e10d8eaef2a5e259a` |
| `eradelta_inrun_b1/paired_per_q.tsv` | 885287 | `ef33dfadee5cc44cfc5707d94bcf2c2962ee802e69498e108e298899219b04a2` |
| `eradelta_inrun_b1/paired_summary_by_transform.tsv` | 38770 | `079910bfe2147a8c92fbe4af60d3a782da237ab95fe851994388118de14273a0` |
| `eradelta_inrun_b1/paired_summary.tsv` | 17148 | `2a45630cfaac8bedd0f7aaf25274796c28babd5d50e007858e3fa98d70808978` |
| `eradelta_inrun_b1/_summary.json` | 133 | `166b71dcd3ec444f6ec108edfe4034493c9544da0e55a974a24e0809e463242e` |
| `eradelta_scored_2026-09-03.parquet` | 546134 | `85191268dc09b4108257840b9727c814855374101dda8f96c48fdedb6e9d25ba` |
