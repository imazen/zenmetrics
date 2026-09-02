# AVIF DOE Stage-B6 outputs — pointer

Record: [`avif_doe_stageB6_analysis_2026-09-02.md`](avif_doe_stageB6_analysis_2026-09-02.md).
Nothing here is in git (>30 KB rule).

**Locations** (identical bytes, two copies of the analysis outputs; the raw
score blobs and bitstreams live only in the LAN store, which is their canonical
home). *Tower was NOT mirrored this session — `/mnt/tower` returned a stale NFS
handle. Recorded as unavailable, not as done.*

| what | where |
|---|---|
| analysis outputs (local) | `/mnt/v/output/zensim-avifdoe-b6/` |
| analysis outputs (LAN store) | `s3://zentrain/analysis/avif-doe-stageB6-2026-09-02/` — 20 objects, 1.77 MB |
| raw score blobs (5,907 NDJSON) | `s3://zentrain/jobs/avifdoe-svt-b6-sf-cpu-20260902/blobs/` |
| encode bitstreams (23,489) | `s3://zentrain/jobs/avifdoe-svt-b6-20260902/blobs/` |
| native defaults control, encode | `s3://zentrain/jobs/avifsub-svt-enc-20260901/blobs/` |
| native defaults control, score | `s3://zentrain/jobs/avifsub-svt-sf-cpu-20260901/blobs/` |
| reference corpus (**NATIVE**) | `s3://codec-corpus/avif-subsample-2026-09-01/` — the same 32 filenames as the 1024² crop corpus, **different pixels on 13 of them; never conflate** |
| Stage-A outputs differenced against | `/mnt/v/output/zensim-avifdoe/doe_scored_2026-09-02.parquet` |

**Regeneration** — see the record's section 9. All three scripts in
`zenmetrics/scripts/jobsys/`: `avifdoe_harvest.py`,
`avifdoe_stageb6_analyze.py`, and (for the Stage-A leg)
`avifdoe_stagea_analyze.py`.

**Every `.tsv` below is also written as a zstd `.parquet` with the same stem.**

| file | bytes | sha256 |
|---|--:|---|
| `b6_scored_2026-09-02.parquet` | 1,155,963 | `c6b4a7a7afd7ee6e0d554594d9b6d0958f48c0c97f98636db1551792ba4b8d13` |
| `naive_native_control_scored_2026-09-02.parquet` | 463,352 | `e0595a38a2a8c6b83d8d42322d7ef8a30f36a6cd8a458de7bd8f6da3b2e6281c` |
| `native_dims.tsv` | 1,095 | `aa6232449d189d073d97d45daaf5b14910248093a50b534a7d65b0a2ce5885e9` |
| `stageb6/b6_bd_per_image.parquet` | 10,409 | `d673f596113984763e9c9bfc49492b7d29bfa1bbd361d3816e855254aefea7b9` |
| `stageb6/b6_bd_per_image.tsv` | 61,997 | `c22b9697a8eba81a468d942bdb47e5ebfe47dfa73fc526bd542e68cc87f12029` |
| `stageb6/b6_budget_vs_native_per_image.parquet` | 11,536 | `84f131899978abd2c34059dae1c367d95d6c02e18869c4ad08772115d8696654` |
| `stageb6/b6_budget_vs_native_per_image.tsv` | 23,105 | `bef51d3f6ee042a60c381d19bcb2dbf88a0651ff103eec3c6d92e373dec4ebf8` |
| `stageb6/b6_control_robustness.parquet` | 2,815 | `7bc14f996bcec7900276f7d5cea18046cd99b983568f284c932782e3fe3bed4f` |
| `stageb6/b6_control_robustness.tsv` | 890 | `9256407500289afd71abf09e6317c7bd54d2116fda866d725807ff836cb83463` |
| `stageb6/b6_main_effects_by_class.parquet` | 10,954 | `d289251f6c2d6e8f31d09af39bf8756a0f422a8fb6b22ddef6b6395bf9a7d79f` |
| `stageb6/b6_main_effects_by_class.tsv` | 8,513 | `254b3292c8383143420dc9e9adb544f5f25bb8caf7fe00b692f2ac2ea4e16bda` |
| `stageb6/b6_main_effects.parquet` | 6,222 | `89e28358a4747411e61d18d34d1952d339cdcb6270b69ed06fda01c96c754057` |
| `stageb6/b6_main_effects.tsv` | 2,058 | `3df523d9cc871bd81f142150ed923e5f1e8d1f42cb83d1781bac647735ab9e41` |
| `stageb6/b6_transfer_gate_full.parquet` | 4,009 | `999e1802b8692acb61041aa61dc7707af4198cb683ea4a1a164f61a0f65531a5` |
| `stageb6/b6_transfer_gate_full.tsv` | 893 | `0650ef1471b21539f9727a94ce2b6f6d8d09a904281f14e2ba6e31ec2c70ee27` |
| `stageb6/b6_transfer_gate_qmatched.parquet` | 3,962 | `578c1158412f2601868cfcbba7ae7b0fd17452f8b9794003ea2004dd43d4a183` |
| `stageb6/b6_transfer_gate_qmatched.tsv` | 840 | `9ec4022da6d676ef211fbc7495a0d111378b49ce3050f9220554fd5083d0b74b` |
| `stageb6/gate_cross_run_identity.parquet` | 2,351 | `fa6e3dd939b03b39b049359133736ead11ce4824776165520cdb8b92eb8668c2` |
| `stageb6/gate_cross_run_identity.tsv` | 384 | `996add643dd3a09e55a6a2919224d79f416ee80d62989822f0ff9087c617fe49` |
| `stageb6/_summary.json` | 1,371 | `402ef7122ba5d47121c3cb55b407112e045bfdb3b4b4b47ef1a823d7d7c53cfb` |
