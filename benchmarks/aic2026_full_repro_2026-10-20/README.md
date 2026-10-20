# AIC2026 full-set reproduction — cropped set, fast tier (2026-10-20)

All 9,618 distorted images of the July 2026 AIC2026 cropped set (840×944)
scored with zenmetrics and compared against the published oracle columns.
Same binary, same routes throughout. The assembled `metrics_cropped.zenmetrics.csv`
(identical header + row order, nine metadata columns copied, `JND_*` recomputed from our
raw values via the site-config mapping) is committed here gzipped:
[`metrics_cropped.zenmetrics.csv.gz`](metrics_cropped.zenmetrics.csv.gz), with
[`provenance_cropped.json`](provenance_cropped.json). It holds only our own values: columns
produced by unpublished implementations (dvifmish, private ports) stay empty, which
`repro_check.py` enforces. Per-metric score tables and the fullres CSV are in `metrics.pointer.md`.

Licence: derived from AIC2026 (DaRUS, doi:10.18419/DARUS-6156), released under CC BY-SA 4.0;
this CSV is distributed under the same licence. Source-image attribution:
`AIC2026_sources.csv` in that release.

Tools (`scripts/aic2026/`): `assemble.py` builds the CSV from per-metric batch TSVs (moved from
the lane's scratch; column table in `aic_columns.py`), `july_diff.py` diffs any assembled CSV
against the published `metrics_<set>.csv`, and `repro_check.py` checks the committed files
(`just aic-repro-check`; add `--reference combined.csv` to re-derive `compare_cropped.tsv` exactly).

Reference: the July/drop union `combined.csv` (sha256 `91d7c2df…`,
https://zentrain.r2.imazen.org/aic2026/csv/july-drop-2026-10-20/combined.csv,
[reconciliation](../aic_csv_reconcile_2026-10-20/README.md)), cropped view. Its July columns agree numerically
with the July `metrics_cropped.csv` the lane first compared against: re-deriving against it changes no statistic,
only CAMBI's exact-string count (2,503 → 1,050, the union keeps the later export's decimal strings).
`compare_cropped.tsv` is that re-derivation; it adds two `nodata` rows (HDR_VDP_2/3, not scored in this set).
These values came from the CLI's six-decimal batch text; full-precision rescoring uses `just aic-rescore`
([docs/AIC_RESCORE.md](../../docs/AIC_RESCORE.md)).

Binary: `zenmetrics` (zenmetrics-cli) v0.6.0 built from the CI-pinned
sibling snapshot of master `4d833195` + the `aic_vmaf_features` example
(ADM2 / VIF_vmaf columns, which `batch` does not expose).

Run: `zenmetrics batch --metric <m> --pairs pairs.tsv --output <m>.tsv
--jobs 16 --group-by-ref` (cvvdp adds `--display-model standard_fhd`).

## Result: 31 of 37 computed columns inside the 1e-3 median site-JND gate

Regenerated 2026-10-20 on `master@origin` `22bff298` with the settled
routes `proposal-butteraugli`, `cambi-aic2026` and `cw-ssim-pyiqa`, plus the
newly scored `HDR_VDP_2`/`HDR_VDP_3` columns. All six rescores on the
changed-code routes (`flip-v17`, `psnr`, `psnr-cb`, `psnr-cr`,
`psnr-y-studio601`, `psnr-ycbcr611`) came back byte-identical to the
values in this CSV.

`compare_cropped.tsv` has the full per-column table (n, exact-equal count
at published precision, median/max |Δraw|, median/max |Δ site-JND|,
Pearson, Spearman). JND deltas map the site-config fit onto BOTH our value
and the published raw value; the published `JND_*` columns use different
fits, so they are compared informationally only.

Exact-level columns (median |Δ| at float-representation scale):
`SSIMULACRA2` 2.5e-7, `FLIP` 2.5e-7, `mDCT-PSNR` 5e-7, `proposal-mDCTPSNR`
(derived) 7.8e-8, `SSIMc` 2.6e-7, `SSIM-pyiqa` 4.7e-7, `MS-SSIM-pyiqa`
2.8e-7, `CW-SSIM` 2.5e-7, `CAMBI` 2.2e-7, `proposal-Butteraugli` 6.4e-7,
`GMSD` 4.4e-7, `DSSIM` 2.6e-7.

## The seven misses, diagnosed

Every miss is a small raw residual amplified by a near-vertical site-JND
curve, or a documented variant difference — none are route/option errors.
Worst-five rows per column: `july_diff.py` writes them next to its report
(not committed: they contain published values).

| column | route | median \|dJND\| | diagnosis |
|---|---|---:|---|
| `VMAF` | `vmaf-studio601` | 0.00107 | Borderline (gate 1e-3). Raw median 6.5e-3/100, Pearson 1.00000 — libvmaf-version-level noise; worst rows are top PNG-ladder rungs |
| `NLPD` | `nlpd-iqa` | 0.0355 | Documented residual: `docs/AIC2026_REPRODUCTION_STATUS.md` records "median 2.6e-3, 2.6% relative (f32 order and bilinear edges)". Our 2.3e-3 matches that scale |
| `FSIM` | `fsim-y` | 0.0105 | Raw median 4.7e-5; site JND slope ≈ −172/unit at score 0.99 where the column clusters. All diffs at >0.96, ours systematically higher; doc noted "one JPEG tail pair" on the earlier probe |
| `FSIMc` | `fsim` | 0.0098 | Same mechanism, shared phase-congruency core |
| `VSI` | `vsi` | 0.0074 | Raw median 1.5e-5; slope ≈ −270/unit at 0.99. Previously verified "reproduced" at ≤3.4e-6 residual; over gate purely from amplification |
| `HDR_VDP_3` | `hdrvdp3-aic2026` | 0.00153 | Raw median 1.56e-3, Spearman 0.99998 — near miss; the native HDR-VDP 3.0.7 port is slightly off the authors' build |
| `HDR_VDP_2` | `hdrvdp-aic2026` | 0.00477 | Systematic near-match: raw median 0.0281 Q, ours slightly low. Matlab-vs-native and Octave-vs-native split checks leave a residual that tracks the authors' own build; under investigation (see status doc) |

## Timing (dev, nice-19, 16 jobs per metric batch)

Fast metrics 20–200 s each for 9,618 pairs; heavy: `cw-ssim` 28 min,
`ciede2000` 18 min, `flip-v17` ~60 min, `butteraugli` ~8 min at 12 jobs,
`mdctpsnr` ~50 min. `aic_vmaf_features` (all three VMAF models + feature
dump) 30 s for the full set.

Full-resolution pass streams the 23 GB archive one source at a time
(extract → score → delete); see `compare_fullres.tsv` when complete.

## Canonical reference extension (execution in progress)

The separate canonical study schedules 60 CLI variants without exact July
columns: 43 classical, three no-reference and 14 learned routes. After the
classical group, a separate PyIQA 0.1.16 FSIM/FSIMc/VSI panel tests which
implementation agrees with the July columns. Original-author CW-SSIM,
f64-pooled FLIP and original nine-crop AHIQ are included. Published July
routes, GPU twins, zensim and dvifmish are outside this extension.

`canonical_campaign.py` runs the groups in that order from a supplied local
manifest. `canonical_run.py` serializes each native/reference call through the
shared admission locks and resource wrapper. `canonical_compare.py` joins by
distorted-image identity, verifies the nine original metadata fields, and
retains missing scores as empty fields. The supplied native binary is the
same `4d833195` build described above; its ordinary batch export rounds to
six decimal places. Comparisons therefore include that export rounding.
Existing numerical bounds remain unchanged; an absent bound is `no-gate`,
and a zero-comparison row does not establish a numerical mismatch or match.

Full-cohort canonical results are pending. External score exports and
`provenance.canonical.json` belong under the dated reproduction output root
in `metrics.pointer.md`. The provenance writer records exact calls, source
bindings, installed package metadata, coverage and retained attempt receipts.
Per-image third-party reference values remain external; this directory will
receive aggregates and the compressed CSV of native values after validation.

The provenance recipe accepts `--source-bindings` and `--reference-methods`
JSON inventories. It rechecks every listed source hash before exporting their
contents and records each inventory's own hash. This binds ancillary Octave
helpers and distinguishes normalized float32 PyIQA inputs from VSI's explicit
internal float64 conversion without changing the frozen scoring attempt.

`diagnose_psnr_rounding.py` compares the declared studio-709 comparator,
the frozen native normalization order, and the existing integer-numerator
definition control. On one retained crop, the two floating-point orders
produce different rounded luma at one reference pixel; neither equals the
integer definition at every pixel. This explains that control's residual
beyond batch rounding. It does not establish full-cohort behavior or an
acceptance bound; the scoring routes remain unchanged.

A separate twelve-worker MAD native control completed 11 of 12 rows; one
worker returned `Cannot allocate memory (os error 12)`. Its wrapper recorded
2 seconds, 0.10 GiB peak RSS and 13,911 MiB minimum available memory under
the 12 GiB hard cap. No cap-fired exit was recorded, and the cause is
unproven. Production retains three single-thread workers per classical
bundle, releasing shared admission between bundles. The failed control
does not establish a performance comparison.

Local manifests list shared `heavy_locks` in acquisition order. Maintained
drivers require unique absolute paths before starting resource admission.

`canonical_input_hashes.py` inventories every unique image named by the pair
list with a single read stream and 1 MiB reads. It retains the inventory and
receipt externally, retries errno-12 reads from the beginning, and refuses
changed or missing inputs during verification. The provenance writer accepts
that inventory with `--input-bindings` and rechecks its bytes before export.

Scheduling successors use `just aic-canonical-successor` after the prior
controller reaches a completed bundle boundary. The new manifest retains
the full cohort, metric specifications, scorer bytes and explicit scoring
environment. Completed reference bundles carry their receipt/output hashes
into the successor; their identities are imported without re-scoring.
Unstarted work uses 32 rows per worker and the same ordered heavy locks.
Original manifests, plans and results remain preserved. The original July
metadata CSV is copied and hash-bound for later aggregation.

The 2026-10-20 scheduling transition retained all 9,618 native MAD rows and
2,304 completed reference rows. Its first new bundle completed 96 new
identities, exactly 32 per worker, without re-scoring any retained identity.
That scope recorded `run-heavy: done rc=0 84s | peak-RSS 0.68GiB |
min-avail 12706MiB | peak-load 4.39`. This is a scheduling observation on
that bundle; no full-cohort canonical comparison is qualified yet.
