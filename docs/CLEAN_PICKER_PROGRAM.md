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
