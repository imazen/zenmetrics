# REV2 RECALCULATION — 372 corpora LAN staging (pointer)

Block storage: `/mnt/v/zen/zensim-training/rev2-lan-stage-2026-09-06/`
(`_STAGE_INDEX.json` at the root; `manifests/<corpus>_MANIFEST.json` per
corpus with sha256 per referenced file; `pairs/<corpus>_pairs_lan.tsv` —
each corpus's `ref_path`/`dist_path` pairs rewritten to `s3://` keys).

Resolves `docs/PLAN_REV2_RECALC_2026-09-06.md` §7.3's blocker: *"the 372
corpora are NOT on the LAN store — their pixels live at local WSL paths no
fleet node can see."* Full record + the row-order finding for `live`: see
the §7 amendment in that plan doc.

## What is staged, where

All 8 corpora the postC 372 root's own manifest marks `"source": "fresh:*"`
(re-extractable) are synced — WHOLE directory, path-mirrored, via `s5cmd
sync` through `scripts/lib/s3env.sh`'s resolved endpoint — to
`s3://codec-corpus/eval372-rev2-2026-09-06/<corpus>/`:

| corpus | rows | referenced files | referenced bytes | pairs source |
|---|--:|--:|--:|---|
| cid22 | 4,292 | 4,341 | 1.47 GB | `cid22val_pairs_ab.tsv` (verified) |
| kadid | 10,125 | 10,206 | 3.07 GB | `kadid_pairs_ab.tsv` (verified) |
| tid | 3,000 | 3,025 | 1.06 GB | `tid_pairs_ab.tsv` (verified) |
| csiq | 866 | 896 | 0.38 GB | `csiq_pairs.tsv` (EXACT — the file the postC build read) |
| live | 779 | 808 | 0.87 GB | `live_r2_pairs.tsv`, `.bmp` paths (EXACT) |
| aic3 | 600 | 610 | 1.52 GB | `aic3_pairs_ab.tsv` (verified row-for-row) |
| konjnd | 1,008 | 2,016 | 0.60 GB | generated from `subjective_ratings.csv` (verified row-for-row) |
| pipal | 23,200 | 23,400 | 5.61 GB | generated from `Train_Label/*.txt` (**superset — see caveat**) |
| **total** | **43,870** | **45,302** | **14.57 GB referenced** (24.96 GB / 124,742 objects actually synced — whole dirs, not just referenced files) | |

The 6 byte-copy corpora (aic4, nonphoto, imazen26, sdr25, hfnlproxy,
hf_nearlossless) are **not staged** — they are copied from the old root
and not re-extractable on this box, so there is nothing to stage.

**PIPAL caveat.** The full PIPAL train set (`Train_Label`/`Train_Ref`/
`Distortion_1..4`) has 23,200 pairs (200 refs x 116 distortions each,
verified by direct enumeration); the postC root's `pipal` table has 21,800
(200 x 109). Every reference loses exactly 7 entries in the stored table,
with no selection rule found in `zensim-validate::load_pipal` (which returns
the un-truncated 23,200) or in `--max-images` (default 0 = all; the build
script does not pass it). This staging covers the FULL 23,200-pair pixel set
— a strict superset, nothing missing — and its pairs TSV carries all 23,200.
Reconciling the exact 21,800-row historical selection is wave-2
DECLARE-time work, registered here, not resolved.

## Verified, not just staged

- **Executor reachability, end-to-end, no code change**: `resolve_source` /
  `resolve_feature_input` (`crates/zenmetrics-cli/src/jobexec.rs`) already
  fetch `s3://…` in-process for both the reference and every distorted input
  — built for exactly this recalculation. `scripts/jobsys/
  verify_lan_stage_reachability.sh <corpus>` declares + runs one Feature job
  against the staged `s3://` pairs TSV and the SAME pairs via their local
  paths, and asserts the feature vectors are bit-identical. Run against
  csiq, live (BMP), pipal, konjnd on 2026-09-06 — **0 cells differ** in every
  case.
- **Cross-node read**: `r7900x` (192.168.50.27, idle at the time, no `zen*`
  container running) fetched 3 randomly-picked referenced files (aic3,
  kadid, csiq) by their manifest sha256 over SSH + `aws s3 cp` against the
  LAN store — all 3 hashes matched exactly.
- **Disk budget**: LAN store NVMe cache (`/mnt/cache` on tower, SeaweedFS
  `zen-lanstore` container) had 925 GB free before staging, 912 GB after
  (the corpora are ~25 GB — small, per the task's own framing; tower's
  underlying array has 8.9–19 TB free regardless).

## Regenerating

`scripts/jobsys/stage_eval372_corpora_lan.py [--corpus NAME]` — idempotent
(re-running re-syncs, re-hashes, and overwrites the manifest/TSV; `s5cmd
sync` only transfers changed objects). Needs `s5cmd`, `aws`, and the LAN
store env (`scripts/lib/s3env.sh`'s default resolution, or
`~/.config/zen/lanstore.env`).
