# CVVDP vs the JPEG AIC organisers on AIC-4 — 2026-09-22

## Result

The gap between our CVVDP and the organisers' CVVDP on the AIC-4 sample comes
from the **display configuration**, not from a port defect.

- The organisers run ColorVideoVDP with `-d standard_fhd` (37.84 pixels per
  degree). Our comparator ran pycvvdp's default, `standard_4k` (75.40 pixels per
  degree), because the CPU `cvvdp` path had no way to pick a display.
- Under the same display, our CPU port agrees with reference pycvvdp to
  ≤ 0.00023 JOD on all 300 pairs.
- `zenmetrics batch|score-pairs --metric cvvdp --display-model standard_fhd`
  now reproduces the organisers' column: **SROCC 0.9609 (theirs 0.9609)**, all
  five per-source SROCCs equal to three decimals, max |Δ| 0.00027 JOD.
- Side finding: `scripts/sweep/pycvvdp_worker.py`, the scorer in the
  `pycvvdp-scorer` image, passed reference and distorted to pycvvdp in the
  wrong order. Fixed; see §4.

## 1. Data

| item | path | sha256 (first 16) |
|---|---|---|
| 300 AIC-4 PTC crop pairs, 620×800 portrait, 8-bit RGB, no colour chunks | `/mnt/v/output/zensim/v2-backfill-2026-07-20/aic4_pairs.tsv` | `955a9601e94c0877` |
| organisers' metric column (`CVVDP`) + human JND | `~/work/zen/zensim/site/data/parquet/aic4_sample.parquet` (`score_cvvdp`, from `JPEG-AIC_metric_scores.csv`) | `c32dc6a66a3e37f7` |
| our board table (`peer_cvvdp`, aic4 axis) | `/mnt/v/output/zensim/reports/refmetrics/aic4_cvvdp.tsv` (`cvvdp_cpu_imazen_v0_1_0`) | `4042d578c0f15b89` |

SROCC is Spearman against the human JND (reported as |ρ|; JND is a distortion,
so the raw sign is negative).

## 2. Where the AIC display settings come from

| source | what it says |
|---|---|
| **AIC-4 Common Test Conditions v2.0**, wg1n101246 §4 (local copy `~/tmp/papers-dvifm/wg1n101246-108-ICQ-Common_Test_Conditions_on_Objective_Quality_Assessment_v2_0.md`, line 188) | ColorVideoVDP **v0.4.2**. SDR: `cvvdp -d standard_fhd`, 37.84 pixels per degree, peak 200 cd/m², black 0.2 cd/m², reflected 0.3979 cd/m². HDR: 56.55 pixels per degree, 1000 cd/m², black 0.001, reflected 0.007958. JND mapping `f(x) = 3.1889 · max(0, 10 − x)^1.0129`. |
| AIC-4 CTC v1, wg1n101156 (same folder, `work/jpegmd/`) | Same SDR display (`standard_fhd`, 37.84 pixels per degree, 200 / 0.2 / 0.3979). Older linear mapping `3.1420 · (10 − x)`. |
| pycvvdp `vvdp_data/display_models.json` (identical in 0.4.2 and 0.5.4) | `standard_fhd`: 1920×1080, 24", 0.6 m, 200 cd/m², contrast 1000, 250 lux. `standard_4k`: 3840×2160, 30", 0.7472 m, same photometry. The two differ **only in geometry**. |
| AIC2026 readme (`/mnt/v/datasets/aic2026/readme_AIC2026.md`) and `encoding_recipes.md` | No display settings; only "CVVDP-based estimates" for choosing distortion levels. |
| `aic4_sample.parquet` metadata | pandas schema only; no display information. |
| AIC-4 dataset README | JND mapping form only; no display information. |

## 3. Measurements (300 pairs)

Reference pycvvdp runs used `scripts/sweep/pycvvdp_worker.py` **after the
argument-order fix** (§4), on CPU torch 2.14.0: pycvvdp 0.5.4 from PyPI, and
0.4.2 from git tag `v0.4.2` (`3a746fd4`), since 0.4.2 is not on PyPI.

| configuration | SROCC | PLCC | JOD sd | max / mean \|Δ\| vs organisers | rank agreement with organisers | 00002 | 00006 | 00007 | 00009 | 00010 |
|---|---|---|---|---|---|---|---|---|---|---|
| organisers (AIC CSV) | 0.9609 | 0.9599 | 0.2810 | — | 1.0000 | 0.969 | 0.984 | 0.976 | 0.906 | 0.974 |
| board row `peer_cvvdp` (ours, CPU, standard_4k) | 0.8906 | 0.8248 | 0.1175 | 0.7837 / 0.2872 | 0.9499 | 0.972 | 0.987 | **0.796** | 0.898 | 0.980 |
| pycvvdp 0.4.2, standard_4k | 0.8906 | 0.8247 | 0.1175 | 0.7837 / 0.2871 | 0.9499 | 0.972 | 0.987 | 0.796 | 0.898 | 0.980 |
| pycvvdp 0.5.4, standard_4k | 0.8906 | 0.8247 | 0.1175 | 0.7837 / 0.2871 | 0.9499 | 0.972 | 0.987 | 0.796 | 0.898 | 0.980 |
| **pycvvdp 0.4.2, standard_fhd** | **0.9609** | 0.9599 | 0.2810 | **0.00005** / 0.00002 | 1.0000 | 0.969 | 0.984 | 0.976 | 0.906 | 0.974 |
| pycvvdp 0.5.4, standard_fhd | 0.9609 | 0.9599 | 0.2810 | 0.00016 / 0.00003 | 1.0000 | 0.969 | 0.984 | 0.976 | 0.906 | 0.974 |
| ours `batch --metric cvvdp` (no flag) | 0.8906 | 0.8248 | 0.1175 | 0.7837 / 0.2872 | 0.9499 | 0.972 | 0.987 | 0.796 | 0.898 | 0.980 |
| ours `--display-model standard_4k` | 0.8906 | 0.8248 | 0.1175 | 0.7837 / 0.2872 | 0.9499 | 0.972 | 0.987 | 0.796 | 0.898 | 0.980 |
| **ours `--display-model standard_fhd`** | **0.9609** | 0.9599 | 0.2809 | **0.00027** / 0.00005 | 1.0000 | 0.969 | 0.984 | 0.976 | 0.906 | 0.974 |

Per-pair agreement:

| pair of scorers | max \|Δ\| JOD | mean | pairs > 1e-3 |
|---|---|---|---|
| ours (no flag) vs board table | 0 | 0 | 0 |
| ours `--display-model standard_4k` vs ours (no flag) | 0 | 0 | 0 |
| ours standard_4k vs pycvvdp 0.5.4 standard_4k | 0.00018 | 0.00002 | 0 |
| ours standard_fhd vs pycvvdp 0.5.4 standard_fhd | 0.00023 | 0.00004 | 0 |
| pycvvdp 0.4.2 vs 0.5.4, standard_4k | 0.00072 | 0.00000 | 0 |
| pycvvdp 0.4.2 standard_fhd vs organisers | 0.00005 | 0.00002 | 0 |

The 0.4.2 → 0.5.4 version change is immaterial here; the organisers' CSV is
exactly pycvvdp 0.4.2 at `standard_fhd`, up to its four-decimal rounding.

**Why the display matters this much.** At 37.84 pixels per degree each pixel
covers twice the visual angle it does at 75.40, so compression artefacts sit at
lower, more visible spatial frequencies. The JOD spread therefore widens 2.4×
(sd 0.117 → 0.281), and the between-codec ordering within a source changes.
Source 00007 is where that reordering hurts most (0.796 → 0.976).

## 4. Side finding: pycvvdp worker had reference and test swapped

pycvvdp's signature is `cvvdp.predict(test_cont, reference_cont, dim_order)`.
`scripts/sweep/pycvvdp_worker.py` called `predict(ref, dist)`. It surfaced here
as an apparent port drift of up to 0.016 JOD: a one-pair probe calling
`predict(dist, ref)` gave 9.420868 for `PTC_00007_JPEG-XL_10`, where our port
gives 9.421022 and the worker gave 9.436721. Swapped vs correct, same pycvvdp
build, 300 pairs:

| display | max \|Δ\| | mean | pairs > 1e-3 |
|---|---|---|---|
| standard_4k | 0.0159 | 0.00124 | 90 |
| standard_fhd | 0.0173 | 0.00200 | 158 |

The conformance goldens (`build_goldens.py`, `build_conformance_goldens.py`)
already used the correct order and are unaffected. The same swap was in the
JODs printed by `bench_{1024_all_paths,all_paths,4096_vs_pycvvdp}.py`; fixed.
Any `cvvdp_pycvvdp_v054` column the worker wrote before this fix is
swapped-argument data. `docs/CVVDP_HISTORY.md` records the column's production
run as never completed, and `docs/SCORING_DATA_2026-06-24.md` reports none in
that corpus. I did not audit R2 for stray sidecars.
Regression test: `scripts/sweep/test_pycvvdp_worker.py`. It uses a recording
stand-in for pycvvdp; reverting the fix makes it fail.

## 4b. pycvvdp 0.5.7 and `--temp-padding` (checked on request)

The port is pinned to **pycvvdp 0.5.4**. 0.5.6 added `--temp-padding=symmetric`,
and 0.5.7 fixed it for short videos and set the default back to `replicate`
(release notes v0.5.6 / v0.5.7). For single-frame input, pycvvdp 0.5.7 takes
the `is_image = (N_frames==1)` branch of `cvvdp_metric.py`, which never reaches
the temporal-padding code. Measured on the 300 pairs (CPU torch 2.14.0,
correct argument order):

| scorer | standard_4k SROCC | standard_fhd SROCC | max \|Δ\| vs pycvvdp 0.5.4 | max \|Δ\| vs our port |
|---|---|---|---|---|
| pycvvdp 0.5.7, `temp_padding="replicate"` | 0.8906 | 0.9609 | **0** (both displays) | 0.00018 / 0.00023 |
| pycvvdp 0.5.7, `temp_padding="symmetric"` | 0.8906 | 0.9609 | **0** (both displays) | 0.00018 / 0.00023 |

Still-image scores are bit-identical across 0.5.4, 0.5.7-replicate and
0.5.7-symmetric, so the padding default does not affect this port, which is
still-image only. Video scoring is not ported and not checked here.

### 4c. Full 0.5.4 → 0.5.7 source review, and what we did about it

`diff -r` of the two installed packages:

| file | change | affects our still-image port? |
|---|---|---|
| `csf.py`, `lpyr_dec.py`, `interp.py`, `color_spaces.json`, every `csf_lut_*.json` | **unchanged** | — |
| `cvvdp_parameters.json` | `version` string only | no |
| `cvvdp_metric.py` | video path refactored into `read_block_of_frames`; `temp_padding="symmetric"` added; pre-filtered (SPEM) video bypass; MPS device default; image buffer `torch.empty` → `torch.zeros` (all 6 channels are written anyway); dead masking branch (`smooth_clamp_cont` / `fvvdp_ch_gain`, unused under `masking_model: mult-mutual`) removed; distogram export gains a batch axis | no |
| `display_model.py`, `utils.py`, `vq_metric.py` | exceptions → `vq_exception`; `get_best_device`; config-file extension check | no |
| `video_source.py` | a debug-only mean-luminance log skipped for `RGB2020pq`; batch-singleton shape check | no |
| `video_source_file.py` | ffmpeg/YUV video decode: BT.601 → BT.709 YUV matrix, 4:2:2 support, frame counting | no; we take RGB, not YUV video |
| `run_cvvdp.py` | CLI: `--device auto`, `--count-frames`, `--temp-padding {replicate,symmetric,valid}`, `--temp-resample <fps>` | no |
| `display_models.json` | **adds `65inch_hdr_pq_{1Knit,2Knit,4knit}` and `lg_oled_2026_hdr_pq`** | our vendored copies (`cvvdp`, `cvvdp-gpu`) are already identical to 0.5.7's |

So nothing needs porting for still images. The one thing 0.5.7 unlocks is
**conformance coverage**: those four HDR presets used to be imazen-only and
excluded from the matrix, because pycvvdp could not produce a golden for them.

Done (see `crates/cvvdp/docs/CVVDP_CONFORMANCE.md`, "Results — conformance-v2"):

- The four presets are added to the matrix (13 displays × 31 situations = 403
  cells). Goldens are built with pycvvdp 0.5.7.
- **CPU 403/403 within 1e-3** (max 0.000877). **GPU 400/403** (max 0.00139); the
  3 misses are the already-documented Finding-B cells. The new HDR presets pass
  every cell on both impls (max Δ 0.00061 CPU, 0.00084 GPU).
- On the 279 cells shared with v1, the v1 (0.5.4) and v2 (0.5.7) goldens agree to
  **7.6e-6 JOD**.
- `build_conformance_goldens.py` now records the installed pycvvdp version (it
  used to copy the Rust pin, so 0.5.7 goldens would have been labelled v0.5.4).
  A display it cannot construct is now fatal; it used to become a null golden
  that the harness silently skipped.
- The matrix TSV is named by reference version
  (`cvvdp_conformance_matrix_pycvvdp_v0.5.7.tsv`). A hard-coded `2026-05-26`
  made every run overwrite the May record.

Not done: the port's `PYCVVDP_REFERENCE_VERSION` stays **v0.5.4**. It is
lockstep-pinned to the per-stage R2 goldens (`cvvdp-gpu/tests/it/parity.rs`,
`version_lockstep.rs`), so bumping it means regenerating and uploading those. That
is justified by the numbers above, but it needs an upload. The same applies to the
`pycvvdp-scorer` image and its `cvvdp_pycvvdp_v054` column.

## 5. What changed

- `zenmetrics batch` / `score-pairs --metric cvvdp --display-model <name>`: the
  CPU port now honours the named preset's photometry **and** geometry
  (`crates/zenmetrics-cli/src/metrics/cvvdp_cpu.rs`). Before this, the flag
  applied only to `cvvdp-gpu`, and CPU `cvvdp` was hard-wired to `standard_4k`.
- **The default is unchanged.** With no flag, the output is bit-identical to the
  board table. A non-default display writes to its own column,
  `cvvdp_cpu_imazen_v0_1_0_<display>` (e.g. `..._standard_fhd`), so the two can
  never be joined by accident. `--display-model standard_4k` keeps the plain
  column and is bit-identical to the default (unit test). `--hdr` together with
  `--display-model` is refused for CPU `cvvdp`.
- The AIC configuration is the upstream preset name `standard_fhd`, documented
  in `--help` as the JPEG AIC display.

## 6. Which display every existing board row used

Every CVVDP value on the zensim board was computed at **`standard_4k`**. Before
this change the CPU path could not do anything else, and
`/mnt/v/output/zensim/reports/refmetrics/run_gpu_metrics.sh` passes no
`--display-model` to the GPU runs.

| board data | CVVDP source | display | re-score for AIC comparability? |
|---|---|---|---|
| `peer_cvvdp` · aic4 | `aic4_cvvdp.tsv` (`cvvdp_cpu_imazen_v0_1_0`) | standard_4k | **done**: `aic4_cvvdp_standard_fhd.tsv` → new row `peer_cvvdp_aicfhd` |
| `peer_cvvdp` · aic3 | `aic3_cvvdp_heldout.tsv` (600 pairs) | standard_4k | **yes, not done**: AIC-3 is a held-out set outside this work order |
| `peer_cvvdp` · sdr25 | `sdr25_cvvdp.tsv` (50 pairs, JPEG-AI SDR25, AIC-3 method) | standard_4k | **yes, not done**: held-out, same reason |
| `peer_cvvdp` · cid22 / konjnd / csiq / live | CPU `cvvdp_cpu_imazen_v0_1_0` | standard_4k | no: not AIC data; the 4k default is a legitimate, consistent choice |
| `peer_cvvdp` · kadid / tid | GPU `cvvdp_imazen_v0_0_1` | standard_4k | no (as above) |
| `peer_cvvdp` · imazen26 / nonphoto / hfnlproxy | fill4 sidecars, `cvvdp_cpu_imazen_v0_1_0` | standard_4k | no |
| 270 candidate rows, `per_pair.kadis.cvvdp` | KADIS-700k canonical `score_cvvdp_cpu_imazen_v0_1_0` | standard_4k | no: reference column, not an AIC axis |

## 7. Reproduce

```bash
# ours (build: cargo build --release -p zenmetrics-cli [--features sweep])
zenmetrics batch --metric cvvdp --display-model standard_fhd \
    --pairs /mnt/v/output/zensim/v2-backfill-2026-07-20/aic4_pairs.tsv --output ours_fhd.tsv
# reference (pairs TSV needs image_path/codec/q/knob_tuple_json/ref_path/dist_path)
python scripts/sweep/pycvvdp_worker.py score-pairs --pairs-tsv pairs.tsv \
    --out-parquet py_fhd.parquet --display-name standard_fhd
```

Scratch outputs (parquets, TSVs, the analysis script `analyze.py`):
`~/tmp/devin/cvvdpfix/`. The AIC-display AIC-4 table the board row reads:
`/mnt/v/output/zensim/reports/refmetrics/aic4_cvvdp_standard_fhd.tsv` (+ `.meta`).
