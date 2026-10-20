# AIC July/drop CSV reconciliation (2026-10-20)

Both cropped and full-resolution tables contain the same 9,618 image keys in July and in the supplied drop files. All 66 shared columns agree numerically, and reference-image identities agree after stripping only the `PTC_` crop prefix. There are no numeric conflicts in these shared columns. Formatting-only differences occur in 4,697 cropped cells and 4,766 full-resolution cells; the union preserves the drop file's original strings for shared values.

July has 80 columns and drop has 83. The union contains 97 source columns plus `view` and `ptc_subset` (99 total). New and older variant names remain separate; name similarity does not establish equivalent metric conventions. `WS-HV` occurs in all three drop CSVs, but not the July tables.

The 3,417-row `metrics_PTC.csv` is an exact, string-for-string subset of the drop cropped table. It adds no image rows. Its membership is marked only on cropped union rows. Cropped and full-resolution values are never joined into one observation.

## Combined artifacts

Measured on Ryzen 9 7950X / WSL2. Resolve the output directory before reuse; this host's paths differ from other machines. Generated CSVs stay outside git. Original supplied files and July copies are preserved.

Output root: `/mnt/v/output/zenmetrics/aic-csv-combined/2026-10-20-union/`.

| File | Rows | Bytes | SHA256 |
|---|---:|---:|---|
| `union_cropped.csv` | 9618 | 13187821 | `6c5f3c86825e915289dedef206a9e040f69279ec1347bc9a11c0af00d0089147` |
| `combined_cropped.csv` | 9618 | 23424678 | `485204b6cb7570d90483146911b47bf473070dce3d70042e348e98e407f4afcf` |
| `union_fullres.csv` | 9618 | 13112286 | `99f8387e9eacacff916a2d040cacab9aea47ad81691e3cfe886b2080070242e1` |
| `combined_fullres.csv` | 9618 | 23275092 | `3148b927eb38311fbc7cd621aca26cf8f70a3543933f69c64568e74d2c0f58fb` |
| `combined.csv` | 19236 | 26299104 | `91d7c2df86f6ac973288f511b1bb625890d8b4a2c35fbe8b3199d28c76836866` |
| `combined.csv.gz` | — | 11629498 | `167e45e14e670c642be083fc5cf19f5297faabac82d67226d1bcf3055d097818` |

`combined.csv` is the convenient 19,236-row union with an explicit `view` column. `union_cropped.csv` and `union_fullres.csv` separate the views. `combined_cropped.csv` and `combined_fullres.csv` are the lossless audit tables retaining every field twice under `july::` and `drop::`; every original string was checked after re-reading these files. Gzip uses level 9 and mtime=0; only original source strings are copied, never rounded/recalculated.

## Inputs

July: AIC2026 dataset [DOI 10.18419/DARUS-6156](https://doi.org/10.18419/DARUS-6156), copied from the existing dataset cache on dev. Drop: the user-supplied `metrics_PTC.csv`, `metrics_cropped.csv` and `metrics_fullres.csv`; these names do not establish their release date. The source hashes below bind this observation. Local paths and every per-column comparison remain in `comparison.json` beside the outputs.

| Input | SHA256 |
|---|---|
| `july_cropped` | `3fb7208165c039cdc4bdea2e1244ea52020b60019c9c96f4b00a9b9967a7ad99` |
| `july_fullres` | `786e1647d2a8618672b5ed2336b4892a21ba278da47d12722d662e22ba010951` |
| `drop_cropped` | `bd879b813db72568d52302afe3e66ac7352b539a8e9216b9c919cc1162bfdd88` |
| `drop_fullres` | `5d8db954e36caac49ba71e0b87f934f1e52051e5a09dfd2a12a5ebed128e20ba` |
| `ptc` | `26e250acc9006fb915d56a128ae552f20566a01d431a62e410d3b7388d5d48ee` |

## Column differences

July-only column names: `PSNR-HVS`, `JND_PSNR-HVS`, `VMAF-neg`, `JND_VMAF-neg`, `proposal-DVIFM`, `proposal-DVIFM-0.2-use_chroma`, `VMAF`, `PSNR-HVS-Y`, `PSNR-Cb`, `PSNR-Cr`, `PSNR-HVS-Cb`, `PSNR-HVS-Cr`, `VIF_vmaf`, `CKDN`.

Drop-only column names: `PSNR-HVS-vmaf`, `JND_PSNR-HVS-vmaf`, `proposal-DVIFM-2d`, `proposal-DVIFM-min`, `WS-HV`, `PSNR-HVS-Y-vmaf`, `VMAF v1`, `VMAF v0`, `VMAF-neg v0`, `Butteraugli-max`, `Butteraugli-3norm`, `SSIMULACRA1`, `Butteraugli-original`, `VIF-vmaf-scale0`, `VIF-vmaf-scale1`, `VIF-vmaf-scale2`, `VIF-vmaf-scale3`.

## Reproduction

```sh
python3 scripts/complexity/aic_csv_reconcile.py --july-cropped JULY/metrics_cropped.csv \
  --july-fullres JULY/metrics_fullres.csv \
  --drop-cropped DROP/metrics_cropped.csv \
  --drop-fullres DROP/metrics_fullres.csv --ptc DROP/metrics_PTC.csv \
  --output NEW_OUTPUT_DIRECTORY
```

The command refuses existing output directories, duplicate normalized keys, reference mismatches, ambiguous PTC membership and conflicting union values. Comparisons distinguish numeric changes from representation-only differences using decimal parsing; no float-based equality tolerance is introduced. It does not compare image pixels or validate metric implementations. This is published-input reconciliation, distinct from native score regeneration.

Resource receipt: `run-heavy: done rc=0 12s | peak-RSS 0.71GiB | min-avail 10437MiB | peak-load 2.33`. This covers the full comparison, union writing, gzip and source-string checks, not per-metric memory or speed.
