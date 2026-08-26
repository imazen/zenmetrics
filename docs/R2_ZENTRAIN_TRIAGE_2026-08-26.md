# R2 zentrain triage — inventory + LAN-mirror evidence (2026-08-26)

R2 `zentrain` totals **1,811 GiB / ~7.6M objects** across 113 top-level prefixes
(full table: sized by rclone, ~/tmp/zentrain_inventory.tsv archived below). Tower
NVMe cache (SeaweedFS home) has only **454G free**, so full migration is
structurally impossible on the cache; the Unraid array has ~22T free.

## Top prefixes
| size | objects | prefix |
|---|---|---|
| 607.2 GiB | 2019 | jxl-lossy |
| 368.7 GiB | 3108446 | jobs |
| 299.4 GiB | 2815229 | canonical |
| 199.7 GiB | 701564 | kadis-700k-gpu |
| 55.6 GiB | 37 | ext720-canonical-2026-07-22 |
| 50.0 GiB | 36 | ext924-canonical-2026-07-27 |
| 43.7 GiB | 141233 | kadis-700k |
| 29.3 GiB | 16 | tbig-944-2026-08-02 |
| 26.5 GiB | 85 | ext944-canonical-2026-08-01 |
| 22.7 GiB | 657805 | cvvdp-v15rc-2026-05-18 |
| 17.6 GiB | 2 | tbig-720-2026-07-22 |
| 11.6 GiB | 36 | strategy-fleet-2026-07-02 |
| 11.1 GiB | 118897 | omni-multi-codec-2026-05-19 |
| 10.4 GiB | 7119 | refs |
| 7.7 GiB | 26685 | jxl-lossy-hqfill-A |
| 5.2 GiB | 171 | dualmodel-2026-06-28 |
| 5.2 GiB | 408 | fill4-6codec-2026-07-01 |
| 4.8 GiB | 185 | hdr |
| 3.1 GiB | 3 | kadis-720-2026-07-24 |
| 3.1 GiB | 7 | kadis-944-2026-08-01 |

## Mirror evidence (spot-checked 2026-08-26)
- **Fleet-active data: already on SeaweedFS** (jobs/hdrgrid-*, jobs/avifgen-*, refs/ — byte-verified).
- **Current-era canonical TABLES have local /mnt/v mirrors** (ext924 27G, ext944 53G,
  tbig-924 24G, kadis700k/canonical views 1.8G, canonical-picker views ~9G each) —
  but the local copies are the parquet VIEWS, not the artifact trees.
- **R2 is the ONLY complete copy of the artifact bulk**: `jxl-lossy` 607 GiB
  (local dirs are 115M/6M/1.5M summaries; tower jxl-all is 1.5M), `canonical`
  299 GiB (encoded variants), `kadis-700k-gpu` 200 GiB (distorted PNGs =
  rescore-from-links), plus old `jobs/` runs ~300 GiB.

## Proposed tiers (user decision pending)
1. **SeaweedFS (hot, ≤ ~200G)**: current-era canonical tables + eval grids + refs —
   the working set for training/eval on the LAN.
2. **Unraid array (cold, ~1.1-1.4 TB)**: the artifact bulk (jxl-lossy, canonical
   variants, kadis-700k-gpu PNGs, old jobs) — preserves rescore-from-links and the
   persist-everything contract without cloud cost; slow is fine for cold reads.
3. **Drop candidates** (user-gated, after array copy verifies): May-era sweep-v*
   scratch prefixes, smoke/demo prefixes — a few GiB total, mostly noise value.
R2 deletion/retirement remains user-gated in all cases.
