# AVIF DOE Stage-A outputs — pointer

Record: [`avif_doe_stageA_2026-09-02.md`](avif_doe_stageA_2026-09-02.md). Nothing here is in git (>30 KB rule).

**Locations** (identical bytes, three copies of the analysis outputs; the raw
score blobs live only in the LAN store, which is their canonical home):

| what | where |
|---|---|
| analysis outputs (local) | `/mnt/v/output/zensim-avifdoe/` |
| analysis outputs (LAN store) | `s3://zentrain/analysis/avif-doe-stageA-2026-09-02/` — 27 objects, 4.13 MB |
| raw score blobs (3.4 GB, 17k NDJSON) | `s3://zentrain/jobs/avifdoe-svt-sf-cpu-20260902/blobs/` |
| encode bitstreams | `s3://zentrain/jobs/avifdoe-svt-{a1,a2,a0r,ag}-20260901/blobs/` |
| reference corpora | `s3://codec-corpus/avif-doe-1024-2026-09-01/` (crop) and `s3://codec-corpus/avif-subsample-2026-09-01/` (native) — **same 32 filenames, different pixels on 13 of them; never conflate** |

**Regeneration** (all three scripts in `zenmetrics/scripts/jobsys/`):

```sh
avifdoe_harvest.py --score-dir <blobs> --sizes <sizes.tsv> \
    --pairs a1=… --pairs a2=… --pairs a0r=… --pairs ag=… --out doe_scored.parquet
avifdoe_stagea_analyze.py --scored doe_scored.parquet --crop-manifest <crop_manifest.tsv> \
    --control a0r --outdir stagea_a0r \
    --parity-check ~/work/zen/zenavif/scripts/rd_gap/bd_arm.py
avifdoe_stagea_gates.py --scored doe_scored.parquet --crop-manifest <…> \
    --native-sizes <a0native_sizes.tsv> --native-dims <native_dims.tsv> \
    --main-effects stagea_a0r/main_effects.tsv --interactions stagea_a0r/interactions.tsv \
    --bd-per-image stagea_a0r/bd_per_image.tsv --outdir stagea_a0r
```


| file | bytes | sha256 |
|---|--:|---|
| \`doe_scored_2026-09-02.parquet\` (49,120 × 18) | 2449165 | \`3c7c49f436429ac8cf42847e7183cbf827327d108cb1076c6e9606af940db6fb\` |
| \`stagea_a0r/ag_identity_violations.tsv\` | 37 | \`1460bab5c3149a1cb8d83c022fc3800eb23a33e95f91917a5a5df75086cf7ab5\` |
| \`stagea_a0r/ag_transfer_gate.tsv\` | 1481 | \`dd39014a7ef7b99bab789de55176458847573b8fddc74f3c632071d43d83d3dd\` |
| \`stagea_a0r/arm_byte_identity.tsv\` | 5848 | \`59ae84b108f22cd39ea4c3db74e74c5efcf26dda71a8d8ed9ed8045c99fa8bce\` |
| \`stagea_a0r/bd_per_image.tsv\` | 440253 | \`389b2904c124321bc6c4c1747bf46d66f6e338deaf6aad25201e094cc5e640db\` |
| \`stagea_a0r/bytes_alpha_beta.tsv\` | 231590 | \`c4d22acdf235d4696197f67079dc5bdcd12ed9378b28b9d9d76703adf52f863e\` |
| \`stagea_a0r/interactions_per_image.tsv\` | 213883 | \`4216a9229a7948661d9d259f4a4a93e1cfe1c647a5d4f2769211e49e5cca33d7\` |
| \`stagea_a0r/interactions.tsv\` | 4903 | \`05696cf5f5eac4a0c5ab0d105656d90d70383e6f8d4fa62e480d62abf82d1b84\` |
| \`stagea_a0r/main_effects_by_class.tsv\` | 9127 | \`56d51f7d14c6f84521563bd5853327ac70514666ebed83999d5d246c30c90535\` |
| \`stagea_a0r/main_effects.tsv\` | 2761 | \`9b6925a26dc05d15e465493b1082b706a535d42b355397e9277ccaed5d424424\` |
| \`stagea_a0r/stage_b_triggers.tsv\` | 8191 | \`a564ec6f1029847c6ee5b5e65c5a3c40556c77950b51875b63b95abc1de6bbd8\` |
| \`stagea_a0r/bytes_model_meta.json\` | 117 | \`c9b9ebb36921af52f005f15944caa6216cbc85e3d7cb6987c4d651a102a18d94\` |
| \`stagea_a0r/stage_b_budget.json\` | 222 | \`d121f10cb8a8e94d7516e24ca05d3eb7a68d5a171e741181d3364bd5718f2cd4\` |
| \`stagea_a0r/_summary.json\` | 116 | \`d713a22d2aef43e019a6a19663e45716d296fc709cae477dff7ad99445d08d13\` |
| \`stagea_inrun/ag_identity_violations.tsv\` | 37 | \`1460bab5c3149a1cb8d83c022fc3800eb23a33e95f91917a5a5df75086cf7ab5\` |
| \`stagea_inrun/ag_transfer_gate.tsv\` | 975 | \`74674b49977837d46e01c8cf4f30c808d75a4bcdb6c5e3a3d24445ceae502644\` |
| \`stagea_inrun/arm_byte_identity.tsv\` | 5848 | \`59ae84b108f22cd39ea4c3db74e74c5efcf26dda71a8d8ed9ed8045c99fa8bce\` |
| \`stagea_inrun/bd_per_image.tsv\` | 440119 | \`3fe3d29456ef20d7f320e4b3a9676f59c493993b7c5bb920553c4d04f0889ee9\` |
| \`stagea_inrun/bytes_alpha_beta.tsv\` | 75217 | \`59becbfeff31da97f1667c78d90f0a6a3011f96982ff9edf8726e6cdb685625c\` |
| \`stagea_inrun/interactions_per_image.tsv\` | 213076 | \`cd7a5dfa79a66f23b71e1e51e405b584efb59bf9f632835a34cb6982beb4329e\` |
| \`stagea_inrun/interactions.tsv\` | 4829 | \`939280030839f1ecccd80b35690b644bcfd3bdc58e11474e01bcd2d203493d0c\` |
| \`stagea_inrun/main_effects_by_class.tsv\` | 9120 | \`77f1077d081bece64604c3dbabf73d7bfa976bf63cc4b24699b70df42aacfe91\` |
| \`stagea_inrun/main_effects.tsv\` | 2763 | \`0e096b81594156e401e139a5f610e8932c27ff417cb6d044fac53d8936c57102\` |
| \`stagea_inrun/stage_b_triggers.tsv\` | 6173 | \`3d6fb1c3ff1f32f1476b7ead474ac1c21730a672578c22df975b321c4ab97f5c\` |
| \`stagea_inrun/bytes_model_meta.json\` | 116 | \`c2be4d66fc61af3e62745b3af00c2340a58745948c0ec9a066f07cf70399f15c\` |
| \`stagea_inrun/stage_b_budget.json\` | 194 | \`fb30d63f5c93838cead4045b2bf5b3e88170d4d738f54404e3ccd4d128d4f13c\` |
| \`stagea_inrun/_summary.json\` | 116 | \`5e2f443ec823e7de844a36d3736f4bdb89470697c31e3bfe8b154932987322ed\` |
