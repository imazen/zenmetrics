# Pointer — `avif-backend-2026-09-03` (Stage-B remainder analysis outputs)

Record: [`avif_backend_selection_2026-09-03.md`](avif_backend_selection_2026-09-03.md) ·
Registration: [`avif_stageB_remainder_2026-09-03.md`](avif_stageB_remainder_2026-09-03.md)

**Not in git** (>30 KB rule, and these are generated tables). Canonical copies:

| location | path |
|---|---|
| local | `/mnt/v/output/avif-backend-2026-09-03/` — 42 objects, 2,090,262 B |
| LAN store | `s3://zentrain/analysis/avif-backend-2026-09-03/` — **VERIFIED 42 objects / 2,090,262 B** |
| Tower | `/mnt/user/coefficient/output/avif-backend-2026-09-03/` — **VERIFIED 42 files / 2.1 MB, sha256 spot-matched** |

⚠ The Tower copy was written **over SSH**, not through the NFS mount: `/mnt/tower`
on the dev box returned `StaleNetworkFileHandle` (code 116) at the time of
writing and is not in `/etc/fstab`. Tower itself was healthy (array 43 % full,
load 4.6). When the mount is repaired the same bytes appear at
`/mnt/tower/output/avif-backend-2026-09-03/`.

**Era pins:** `2026-09-03` — zenavif `56179fcb` / zenav1-svt `2ca060f4` / zenrav1e `e4883037`; fleet image `ghcr.io/imazen/zenfleet-worker:exec-avifhbd-eradelta-e015344f`. Producer commit `f9d890ad`. **Never join across eras.**

**Score coverage: 12,467 / 12,467 = 100.00 %** (brnat 7,488/7,488, brsdr 4,979/4,979). The 8 `brsdr` cells outstanding at first harvest landed during the analysis and everything was **re-run at full coverage**: every Q1 table byte-identical, sole delta one scan image's BD-rate at 0.009 pp. These files are the 100 %-coverage run.

**Producers (both canonical owners, extended not forked):**

- `scripts/jobsys/avifdoe_harvest.py (extended: SDR backend arms)`
- `scripts/jobsys/avifdoe_brem_analyze.py (new; imports every statistic from the Stage-A owner)`

**Inputs:**

- `brnat_ledger` — `s3://zentrain/jobs/avifdoe-svt-brnat-20260903/ledger/`
- `brsdr_ledger` — `s3://zentrain/jobs/avifdoe-rav-brsdr-20260903/ledger/`
- `score_blobs` — `s3://zentrain/jobs/avifdoe-br-sf-cpu-20260903/blobs/ (2706 NDJSON)`
- `stagea_scored` — `/mnt/v/output/zensim-avifdoe/doe_scored_2026-09-02.parquet`
- `crop_manifest` — `/mnt/v/output/avif-doe-1024-2026-09-01/crop_manifest_2026-09-01.tsv`
- `native_dims` — `/mnt/v/output/zensim-avifdoe-b6/native_dims.tsv`
- `speed_instrument` — `/mnt/v/output/avif-speed-instrument-2026-09-03/ (build_commit d1928710)`

## Files (41, 2.08 MB)

| file | bytes | sha256 |
|---|--:|---|
| `br_scored_2026-09-03.parquet` | 529,175 | `447d50cebf06725c06e6ae5f4e91cd6b0434ce72c84d7a6ee7dc28880b1373b8` |
| `brnat_pairs.parquet` | 445,260 | `77ccdc1e633a20aa6c7f568c80f4321d8af329a01136c29d518ffed90aa7ff40` |
| `brsdr_pairs.parquet` | 350,214 | `a13baf3ed857db38d6b855380f95508ae8e4ab904ab041f83a29ec2e6313982f` |
| `chroma_census_2026-09-03.parquet` | 42,584 | `38c2d5b9b21606b3b2fe9a99b350b30ea783ff44a8050584014bc1e5a5a33626` |
| `parquet/arm_effects_budget.parquet` | 5,717 | `75114e9b1ce1883ac84a8d0d529ac8f36633323ce4472a0baf94733254716205` |
| `parquet/arm_effects_native.parquet` | 5,738 | `15c165f0713eeae973efe338e1f92b03c171d5287612fb2a77a974b728b6cc91` |
| `parquet/backend_iso_time.parquet` | 13,002 | `ac9911b7735efd6150c59f4c8b64120a5696a12fbb97b66f6078e3cf2649ec06` |
| `parquet/backend_per_image.parquet` | 13,391 | `92e0d92a1b1969eca74a6330460327a707ad99055879aecf3536355000212f0f` |
| `parquet/backend_per_image_speed.parquet` | 20,391 | `85dbaf6f7ca7fba087395b9e16dcf07387ac88d662fe4c25f5745d9042137a98` |
| `parquet/backend_speed_cost.parquet` | 9,578 | `7c6fe46c4390e9ccd0f0a4ca5203e48aeac4ae836f3bdb32632786922c4cd084` |
| `parquet/backend_summary.parquet` | 5,413 | `0d643c6c464a46acda94a81c883e26cffc7cd465a16664fd5d6dc017d24b1085` |
| `parquet/backend_three_axis.parquet` | 8,694 | `ff9060eedbc933a4d70f256a04c070c8d7c9fd465c0b1aef604f1a44a8dbb0f1` |
| `parquet/bd_per_image.parquet` | 10,984 | `247c925e329e954c3a06b91caef46cf6ba41e13e427a257c5159c2d3dda73e46` |
| `parquet/brnat_inert_per_stratum.parquet` | 2,105 | `52ce7db758667bbc9040c0632727146130f88163f52d581c6eb4a9c39441fa24` |
| `parquet/brnat_inert_per_stratum_image.parquet` | 2,709 | `3402850a3901cd426fa398d5f508e43af9129b36f61b3a42b270def52dfaf89c` |
| `parquet/factorial_pooled.parquet` | 4,213 | `ee5134deeeff954ed4e1cb8ee15119a9d083e0e14b5449721513637855061beb` |
| `parquet/interaction_size_ab.parquet` | 4,842 | `ab011afd80ec7cf463173456244a5286e21f643567691dae0e7ed86bbec94bb8` |
| `parquet/interaction_size_ab_per_image.parquet` | 10,319 | `a75b3755b063bcfef895a782dc7c1be3e29be1ad44cf5de6a3f644f44f6d20cd` |
| `parquet/interactions_budget.parquet` | 5,819 | `d7f385da3b4ce17046b89fd24d4cb5296f9e5dbec1aac45976c3f3b18ca2bdd9` |
| `parquet/interactions_native.parquet` | 5,792 | `eba95992798e1718d9b61c5a5770a365ca1a1573866056c543c95269ece9fa33` |
| `parquet/interactions_native_by_class.parquet` | 4,450 | `b4c10b4a72420ded3c523f80f21f082f1207852487c661fbabc2b4020fc2b34b` |
| `parquet/interactions_per_image.parquet` | 22,777 | `0c9410e19c9482fbda7f30074b28ba0f1180c0ebab5614c7e9ce934d84531aea` |
| `tables/arm_effects_budget.tsv` | 2,418 | `80efdae1825953f61949578e7ab367d2c581de891654403d11bb38cbad65a614` |
| `tables/arm_effects_native.tsv` | 2,434 | `7b74c5459e6f6b4156be301dee520a1cb20fe447cabf37f4c4af9eb22633aa3b` |
| `tables/backend_iso_time.tsv` | 56,744 | `ae8ce114c43f99867cc502f75d3f479d6c99ca94696ada3489bbbf86b875c526` |
| `tables/backend_per_image.tsv` | 6,346 | `11036f1ffbf7e99cb5f3c7e1bff6d827bc6e3781ae91c42e69e5c387d7cc44ab` |
| `tables/backend_per_image_speed.tsv` | 47,110 | `96a367604d97bb100ee101cfe2ba270c14f382173dd92497166204b483fb3139` |
| `tables/backend_speed_cost.tsv` | 121,328 | `2a24742f44634f684642052306c6ef8e59f5b70c2d0c9eec8fd321a69fe46fd5` |
| `tables/backend_summary.tsv` | 3,277 | `03865a8c492ab8a3b9301f86a585c182362111bb535361941493329ce791f7ba` |
| `tables/backend_three_axis.tsv` | 13,827 | `461c637b7b543e18ffc906db0106eada33d120e1155f728c375b20f0a7e45da2` |
| `tables/bd_per_image.tsv` | 127,220 | `04cccf846b33d4daa3745be7ff7e7e976f465ee70983006a0636700185345fb8` |
| `tables/brnat_inert_per_stratum.tsv` | 988 | `3954690ae7dd33900acd0304d173f34f1adf783ff2ec7758e35149e3e57d3c13` |
| `tables/brnat_inert_per_stratum_image.tsv` | 50,585 | `fd900909de7f304eeca1520801e7a60411fe7387035cd272218f89067f45b4a8` |
| `tables/factorial_pooled.tsv` | 1,286 | `285eeff751f4fe669bb42545a6c73d6777283cff46bc4bfdfc56d524aa6a827c` |
| `tables/interaction_size_ab.tsv` | 3,946 | `1f0fff7cd9ecf1d38e7d6d2633f08c6aed8f27a493c243163366fa24a9308317` |
| `tables/interaction_size_ab_per_image.tsv` | 30,785 | `00fdb4b9d2c297cf4554c909aa1951d2e679e7025ec9a998ec4c62c502ffda95` |
| `tables/interactions_budget.tsv` | 1,565 | `d57144a3450b3b4b13b6fe6c159178f9b0d65228b6d041552816086c0c393b29` |
| `tables/interactions_native.tsv` | 1,560 | `635b4c599d233d808f73b7a35deda8cfd87ec8aa661bc8d0cf5b68513e2d8bb2` |
| `tables/interactions_native_by_class.tsv` | 1,994 | `2c99b78eab1f03406ef8cdc21149718a76b7748fa600ca1af62ef62af41c343f` |
| `tables/interactions_per_image.tsv` | 84,653 | `c54b8e6525f88d513c718a99a24cffce4210b8485b4db4b49e1d0dab10cf75e9` |
| `tables/notes.json` | 1,253 | `e84612daddab0daa41f8aaf886f9bc1204f4ce9033386062260c4216a58f6972` |

### What each table is

- **`br_scored_2026-09-03.parquet`** — the harvested tidy table — 12,467 cells x 18 cols, one row per CELL (not per bitstream), `metrics_present` carried, 8 rows null-scored
- **`brnat_pairs.parquet / brsdr_pairs.parquet`** — `zenfleet-ctl pairs` ledger reductions (7,488 / 4,979 DONE cells)
- **`chroma_census_2026-09-03.parquet`** — the av1C chroma census — 1,114 bitstreams, `seq_profile`/`sub_x`/`sub_y`/`chroma`; 0 exceptions
- **`tables|parquet/interactions_native.tsv`** — the 4x3 QM x sharpness interaction surface at NATIVE (residual, bootstrap CI, sign test)
- **`tables|parquet/interactions_budget.tsv`** — the same at BUDGET size (A2), for the size A/B
- **`tables|parquet/interaction_size_ab.tsv`** — native-minus-budget delta, scoped cropped-13 / passthrough-19 NULL CONTROL / all-32-diluted
- **`tables|parquet/arm_effects_native.tsv`** — per-arm main effects vs the in-run control
- **`tables|parquet/brnat_inert_per_stratum.tsv`** — byte-identity-to-control census per stratum (`tn0` is 288/288 inert)
- **`tables|parquet/backend_per_image.tsv`** — THE per-image backend table — BD-rate (full + banded), achieved spans, per-band bytes
- **`tables|parquet/backend_iso_time.tsv`** — the three-axis view at 4 wall-time budgets, with reach-vs-budget failures kept distinct
- **`tables|parquet/backend_speed_cost.tsv`** — the speed-instrument join, per (image, backend, speed), flagged per-source vs pooled and q45-ONLY
