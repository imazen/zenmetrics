# Clean picker program — train/val/test split, corpus, and the 4 codec pickers

**Read this first for ANY picker work.** It encodes the split rule, the canonical
corpus, and the per-codec deliverables so a blind/forgetful session does the right
thing by default. Set 2026-06-26.

## THE SPLIT RULE (one source of truth: `scripts/picker/origin_split.py`)

Split is **by ORIGIN image, by the last digit of the origin's numeric id**, and
**every sizing/crop/encode derivative inherits the origin's bucket** (no derivative
leaks across the split). Deterministic — no seed, no manifest lookup needed:

| last digit of origin id | bucket |
|---|---|
| 0, 2, 4, 6, 8 | **train** |
| 1, 3, 5 | **validation** |
| 7, 9 | **test** |

- **Train only ever sees even-origin content.** Never train, tune, or pick a
  threshold on a 1/3/5 (val) or 7/9 (test) origin or any of its derivatives.
- Always call `origin_split.split_of(name)` — do NOT re-implement parity or use a
  random/seeded shuffle. The old `train_hybrid` seeded per-rendition 20% split was
  WRONG (per-rendition → scale leakage; random → not reproducible). Fixed to use
  this helper.

## CANONICAL CORPUS = imazen-26 (provenanced), NOT dense-r6

- **Use `imazen-26`** — the sha256-provenanced origin set in
  `/mnt/v/output/imazen-26-features/imazen26_manifest.tsv` (2157 origins;
  `PROVENANCE.md` per source folder under `/mnt/v/imazen-26/`). Under the split
  rule: **1082 train / 657 val / 418 test** origins, balanced across all 12 content
  classes (~50/30/20 each — verified).
- **Materialized split:** `/mnt/v/output/imazen-26-features/imazen26_split_evenodd.tsv`
  (`stem  split  manifest_split  content_class  source  original_path`). Regenerate
  any time from the manifest + `origin_split.py` (deterministic) — see
  `scripts/picker/segment_imazen26.py`.
- **dense-r6 (`/mnt/v/output/dense-corpus-r6-2026-06-26/`) is SUPERSEDED for clean
  training:** it was built from the `K500_even` representatives so it is
  **train-biased** (560/672 origins even → only 64 val + 48 test origins — too thin
  for a held-out), and its `o_`/`v2_src` renditions are NOT in the canonical
  rendition index. `o_<id>` = imazen-26, `v2_src<NNNN>` = imazen-26-png-v2 (same
  lineage). Clean pickers re-sweep from the segmented imazen-26 corpus.

## THE 4 PICKER DELIVERABLES (status — update every session)

Each: sweep the codec's knob grid over the segmented corpus → join content features
→ train_hybrid (train=even, val=1/3/5, **report test=7/9**) → bake ZNPR `.bin` →
**commit the `.bin` into the codec crate** (`<codec>/benchmarks/` or `src/`).

| codec/task | sweep | picker trained | clean split | `.bin` committed |
|---|---|---|---|---|
| jxl lossy   | dense-r6 (train-biased, superseded); clean re-sweep PENDING | provisional `zenjxl_lossy_dense_*` (leaky split) | ❌ → fixing | ❌ |
| jxl lossless | PENDING | ❌ | — | ❌ |
| zenjpeg     | PENDING (older `zenjpeg.*` exists) | older only | ❌ | ❌ |
| zenavif     | PENDING (older `zenavif.*` exists) | older only | ❌ | ❌ |

## WORKFLOW (fleet → picker)

1. **Renditions:** `gen_dense_corpus.py` over the segmented imazen-26 (all 3 splits;
   keep odd origins so val/test exist). Name derivatives so `origin_split` recovers
   the origin id.
2. **Sweep** (fleet): `zenmetrics sweep --plan <plan>` per codec → omni TSV +
   persisted variants to R2 (`zentrain` bucket). Hetzner CPU fleet (`launch_fleet.sh`
   / `hetzner_cpu_sweep.sh`) for CPU metrics; vast for GPU metrics (cvvdp etc. — CPU
   path now works too, C1/C1b). $ cap: small; kill idle boxes.
3. **Pareto:** `scripts/picker/omni_to_pareto.py` (joins features; target metric).
4. **Train:** `train_hybrid.py --codec-config <cfg>` — split via `origin_split`.
5. **Bake:** `tools/bake_picker.py` → `.bin`; **commit the `.bin`**.
6. Commit constantly; `jj git fetch` often (a repo cleanup merge may be landing).

## Provenance discipline

Any new corpus/rendition set MUST be indexed (rendition→origin→original sha256) per
`scripts/provenance/index_corpus.py` so the split + dedup are auditable. Do not ship
an unprovenanced corpus into training.

## RUNBOOK + status (updated 2026-06-26)

**Done (committed):**
- Canonical split helper `scripts/picker/origin_split.py` + segmentation
  `scripts/picker/segment_imazen26.py` (zenmetrics 9fca2a10).
- `train_hybrid` wired to the 3-way origin split + held-out TEST report (zenanalyze
  2989bffa). Validated on dense-r6: val top-3-verify 0.52% / **TEST 0.42%**
  (val→test +0.08pp). Needs `scripts/picker` on PYTHONPATH (process_remaining.sh +
  loo_ablation.sh fixed, 15e20c06).
- **jxl lossy interim bin** `zenjxl/benchmarks/zenjxl_lossy_picker_v0.1_dense-r6-evenodd_2026-06-26.bin`
  (zenjxl 54646bcc) — clean split, but train-biased dense-r6 corpus; supersede with v0.2.

**Clean re-sweep runbook (per codec; the remaining deliverable):**
1. **Pick a balanced REPRESENTATIVE set** that spans last-digits 0–9 (so val/test exist):
   `imazen26_representatives_K500_2026-06-14.tsv` — NOT the `_even` one — per the dense-sampling
   discipline (k-means reps + dense ladder). NO stem-mapping needed: `origin_split` now extracts
   the LEADING stem, so it splits raw descriptive imazen-26 names (`1003_general_…_4000x3000.sdr.png`
   → 1003 → val) and crops (`…_c25_tl`) correctly — feed originals straight to gen_dense_corpus.
2. **Renditions:** `gen_dense_corpus.py --src <originals-dir> --out <corpus>` (the manifest
   `original_path`s of the representative set). Output renditions keep the leading stem → splittable.
3. **Sweep** (fleet): `zenmetrics sweep --plan <plan> ...` per codec → omni TSV + variants→R2.
   Plans: jxl lossy = `lossy_dense`; jxl lossless = the modular plan; zenjpeg/zenavif = their
   scalar-axis plans (see docs/PLAN_SWEEPS.md). Hetzner CPU fleet (now cvvdp-capable) +/- vast.
4. **Pareto:** `omni_to_pareto.py --metric-col score_<m>` (per metric).
5. **Train:** `PYTHONPATH=scripts/picker:scripts/picker/configs:<za>/zentrain/{tools,examples} \
   PICKER_TARGET=<m> python3 train_hybrid.py --codec-config <cfg>` → reports val + TEST.
6. **Bake:** `bake_picker.py` → `.bin`; **commit into `<codec>/benchmarks/`**. TODO before v1:
   have train_hybrid emit `output_bounds` (per-output p01/p99 on val) so the bake's OOD-on-output
   check isn't a no-op (current bins warn "no output_bounds").

**Status table:** jxl lossy = interim bin done (v0.2 imazen-26 pending) · jxl lossless / zenjpeg /
zenavif = clean re-sweep PENDING. Commit constantly; `jj git fetch` often (cleanup merge may land).
