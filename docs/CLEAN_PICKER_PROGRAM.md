# Clean picker program — train/val/test split, corpus, and the 4 codec pickers

## PROGRAM STATUS 2026-06-26 (post parallel-audit — all 4 priorities greenlit, running simultaneously)

User greenlit ALL of: fix plans + re-sweep complete · wire top-K + effort masking · zenpicker + PNG
size model · per-codec loop system. Permissions: **branches+PRs for foundational zenpredict work**;
a **`zencodec-helpers` crate** in the zencodec repo for loop/picker infra; **`modes_full` ("rdfull")
on zenjpeg**; **cap PNG effort** (no pathologically-slow); **ablation runs for jxl-modular on a
non-photo-sufficient corpus** (2nd corpus OK).

**Parallel-audit findings (4 agents, 2026-06-26):**
1. **Knob coverage:** `scalar_dense` PINS content categoricals (subsampling/aq/scan) — it's the scalar-head
   plan, NOT the picker plan. jpeg + avif pickers trained on it are PARTIAL (no subsampling pick). Fix:
   re-sweep jpeg/avif with **`modes_full`** (budgeted); modular picker must use `modes_full` (not effort-only
   `scalar_dense`); **zenpng needs a `max_colors`×dither quantize axis** (effort-capped). jxl-lossy already
   correct (`lossy_dense`).
2. **Masking/top-X:** engine has `argmin_masked_top_k` but it's `advanced`-gated + UNUSED → the proven ≤1%
   top-3-verify path is unreachable at runtime; effort masking is dead at every call site; no compute-tier
   metadata. Fix in-flight: zenpredict PR (stable top-K + `cell_compute_tier` metadata).
3. **zenpicker:** `MetaPicker` exists but instantiated NOWHERE; existing meta-bake is 5-family (fails the
   6-family `validate_family_order`) + used the old banned split. Needs re-bake (6 families / drop GIF,
   origin_split, qualified-name features) + a host call site (zencodecs/imageflow level).
4. **lossy-vs-lossless = a FORMULA, not a picker:** pick lossless iff `target_q ≥ T(content)` (T₀≈84, →~70
   for flat/low-edge); 88% w/ content class; compare PREDICTED BYTES (mistakes cost ~79% median). Caveat:
   measured on 100 tiny photo imgs — needs the real modular sweep on the multi-size NON-PHOTO corpus to firm T.

**In-flight parallel tracks (2026-06-26):**
- zenpredict branch+PR: stable top-K + `cell_compute_tier` metadata (agent).
- `zencodec-helpers` crate (zencodec repo): runtime top-K-verify picker helper + factored loop glue (agent).
- 2nd NON-PHOTO corpus for jxl-modular/lossless + the lossy/lossless formula (agent).
- Fleet: avif finishing (scalar_dense = partial); to RE-SWEEP jpeg+avif with `modes_full` (rdfull); complete
  jxl-lossy chunk-2 (box died, lost 30tr/24val/15test origins) — it's on the correct `lossy_dense` plan.
- Then: modular ablation on the non-photo corpus; PNG quantize-axis + size model; zenpicker re-bake + host;
  per-codec loop system (built on zencodec-helpers).

Per-codec deliverable status: **jpeg picker SHIPPED but PARTIAL** (scalar_dense, no subsampling) — supersede
with a `modes_full` rdfull re-sweep. jxl-lossy: correct plan, completing. avif: re-sweep modes_full. jxl-lossless:
modular ablation pending (non-photo corpus). png: needs quantize axis + size model.



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
| jxl lossy   | dense-r6 (interim, best-from-spent-data) | `zenjxl_lossy_picker_v0.1_dense-r6-evenodd` | ✓ even/odd | ✓ (zenjxl 54646bcc) |
| jxl lossless | clean-corpus sweep PENDING (no held-out data exists) | ❌ | — | ❌ |
| zenjpeg     | clean-corpus sweep PENDING (no held-out data exists) | older only | ❌ | ❌ |
| zenavif     | clean-corpus sweep PENDING (no held-out data exists) | older only | ❌ | ❌ |

### Coverage audit 2026-06-26 — held-out (odd origins) was NEVER swept (read before any re-sweep)

A full audit of every existing jxl-lossy sweep (jxl-all / combined / p0 / dense-r6 /
picker-pp) established the decisive fact: **every sweep ever run used even-only
`K500_even` representatives, so the held-out (odd-origin) split was essentially never
swept.** Union of ALL existing jxl-lossy rows = **666 train(even) origins, 112 odd
(val 64 + test 48, all from dense-r6 alone)**. Same for the other codecs: zenjpeg 143
even / **0 odd**, zenavif 41 even / **0 odd**, zenwebp 59 even / **0 odd**, jxl-lossless
0 usable. So **"just filter existing rows into a held-out" is impossible** — there is
nothing odd to filter. Training data (even) is abundant; held-out (odd) is the universal gap.

- **jxl lossy is DONE via the interim picker.** Reassembling spent data can't beat it:
  the richest single even source (jxl-all, 179 origins) is only ~202 *renditions* (≈1/origin,
  thin), whereas the interim trained on dense-r6's 1523 train renditions + 112 odd held-out
  and hit **val 0.52% / TEST 0.42%** (≤1% MET, val→test +0.08pp = generalizes). Don't re-sweep
  jxl-lossy. (Optional v0.2 on the clean corpus for a beefier held-out — not required.)
- **Reuse-even + sweep-odd is NOT cleanly mixable.** `gen_dense_corpus` (PIL Lanczos from
  the largest rendition) ≠ the existing even data's pipeline (Rust Lanczos3 from originals);
  config vocab also drifts (today's `lossy_dense` = 35 cells ⊂ jxl-all's 37, `prog1/prog2`
  pruned). Mixing pipelines would shift train vs held-out. So the clean path for the 3 needy
  codecs is **ONE self-consistent corpus end-to-end.**
- **Clean corpus + features READY:** `/mnt/v/output/clean-picker-corpus-2026-06-26/`
  (414 origins, 4497 renditions: train 2307 / val 1382 / test 808; consistent `o_<stem>`
  naming) + `clean_features.tsv` (extracted `--sizes 0` from the rendition PNGs, so
  features ↔ swept pixels are identical regardless of resize kernel) + `_features_manifest.tsv`.
- **Sized fleet job (the genuinely-needed spend):** zenjpeg `scalar_dense` 627 cells/img,
  zenavif `scalar_dense` 475, jxl-lossless `modes_full --plan-budget 400` → 315 modular/img.
  × 4497 renditions ≈ **6.4M cells total**, ~$10–12 on a Hetzner cpx51 job-system fleet (~1.5 h).
  jxl-lossy excluded (interim suffices).

## GATE CALIBRATION + RD-OVERHEAD FINDINGS (2026-06-28)

**Tiny-feature blocker cleared (zenanalyze #49, `origin/main`).** `analyze_features`
now mirror-tiles too-small inputs internally (`mirror_tile_packed`) and recovers the
percentile features, so `DATA_STARVED` no longer fires from NaN tiny features — every
caller gets content-aware features at any size with zero external handling. Interim
`tile_fill_tiny_features.py` retired. (memory: `data-starved-tiny-feature-nan`.)

**`LOW_ARGMIN` was the wrong gate — demoted 0.30 → 0.10** (train_hybrid `ebf15fa6`; now
a degenerate-picker sanity floor, not the quality gate). `argmin_acc` (exact byte-min
match) penalizes RD-equivalent NEAR-TIES: webp's K=1 RD overhead **3.37%** is LOWER than
jpeg's **6.41%** — yet jpeg PASSED at 36% argmin and webp FAILED at 23.3%. argmin_acc
anti-correlates with RD quality. Real quality gates: `max_mean_overhead_pct` (RD overhead
— the deployed quantity) + per-zq/size p99 tails + `WORST_ROW`.

**The real overhead problem is the TAIL, not the mean.** After demoting LOW_ARGMIN the
clean bakes still refuse on: `WORST_ROW` (one image at 259% webp / 334% avif) + avif
`PER_ZQ_TAIL` (zq=50 p99 80.4%). Mean is already fine (webp 3.5% / avif 6.8%).

**The optimizer is NOT the limiter — features/cells/landscape are.** GBDT teacher
(per-cell `HistGradientBoosting`) ≈ MLP student on overhead (webp 3.42% vs 3.48%; avif
7.08% vs 6.84% — MLP wins there). So lowering overhead is NOT a better-model problem.
Measured levers:
- **top-K-verify deploy** → mean <1% by K=5 (webp 0.92%, avif 0.84%), but does NOT fix the
  tail (worst rows are the ~7–24% where the oracle isn't in the top-K).
- **"predict more knobs" is mostly already exploited** — `modes_full` sweeps 48–54
  categorical cells + scalar heads (trellis λ, aq_coupling, quality); the 2026-06-09
  ML_FRAMEWORK "6 of 16 knobs" gap is STALE. The lever is features / finer scalar ladders
  (`scalar_dense`) / verify, not more knobs.
- **Feature gap OR irreducible** — the GBDT-all-features floor (3.42%) is either a feature
  gap or content-dependence beyond what we extract; resolve via permutation importance
  (§4.5 input-ablation, `picker_tree_ab`) BEFORE adding features.

**Dual-model (decision-tree code-heuristics + MLP) is viable** — GBDT teacher ≈ MLP
student, so a tree picker is a peer, not a downgrade (and may help the tail: trees catch
axis-aligned interactions a small MLP misses). Open work: tree → Rust code-heuristic /
compact bake per codec API + the per-codec tree-vs-MLP choice. Instrument:
`zenanalyze/zenpicker-train/src/bin/picker_tree_ab.rs` (extended 2026-06-28 with
`--split-map`/`--eval-split` to honor the origin split). User directive 2026-06-28: ship
BOTH code-heuristics (all APIs) AND MLPs (for the predictive features).

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

## Concurrency model — MEASURED 2026-06-26 (no thrash; the real lever is the outer loop)

The sweep uses ONE global rayon pool sized to `--jobs` (cells `par_iter`'d; encoders' internal
rayon shares it — so `--jobs` is the total-thread cap, no oversubscription). Measured cells/s vs
`--jobs` on a 28-core box (4-img sample), and live cpx41 CPU% (`/proc/stat`):
- **avif** `scalar_dense`: 4→5.2, 8→9.9, 12→12.9, 16→15.3, 24→17.9, 28→19.7 — ~0.5 core/cell
  (rav1e has serial phases), scales with jobs toward cores. `--jobs=nproc` OK (slow cells fill cores).
- **jxl** `lossy_dense`: plateaus ~100 cells/s from `--jobs≈8` (multi-threaded encode fills the pool;
  more jobs don't help). `--jobs=nproc` optimal.
- **jpeg** `scalar_dense`: fast cells → only ~50% box util on a many-image chunk (measured: live cpx41
  jpeg box = 49% busy / 51% idle / **0% iowait**). NOT `--jobs` and NOT the orchestrator Mutex
  (`--use-legacy-scheduler` measured identical: 387 vs 394 cells/s). It's the **serial outer source
  loop** (run.rs:434 decodes each source serially, then parallel-encodes its cells) — for a fast codec
  with many sources the per-source decode/setup between cell-bursts is ~half the wall time. A 4-image
  local run at `--jobs=8` hit 394 cells/s = full 8-core saturation, confirming the gap scales with
  source-count for fast codecs.

**Verdict:** no thrash; `--jobs=nproc` is fine-to-optimal for the EXPENSIVE codecs (avif/jxl) that
dominate cost. **Efficiency follow-up (deferred, not blocking — user chose "let it finish" 2026-06-26):**
pipeline the serial source loop (decode-next overlapped with encode-current, or flat par_iter over
(source×cell) with a decoded-source cache) → lifts fast-codec util to ~100%. Mainly helps jpeg-class
sweeps; avif/jxl already well-utilized. The user's other ideas (real-time CPU-util adaptation,
estimate_encode_resources-driven scheduling) are alternatives to this same outer-loop fix.

## LIVE PRODUCTION RUN 2026-06-26 (resume/collect steps for a blind session)

Smart chunk fleet LAUNCHED (decode-once `zenmetrics sweep` per box, orchestrator
cached-reference, `--encoded-out-dir` persists variants, `--jobs=nproc`). Runs (R2
under `s3://zentrain/jxl-lossy/runs/<RUN>/`, omni at `…/omni/box-*.omni.tsv`):
- `clean-jpeg-213753`  (zenjpeg scalar_dense, 3 boxes)
- `clean-jxllossy-213753` (zenjxl lossy_dense, 6 boxes)
- `clean-avif-214356`  (zenavif scalar_dense, 12 boxes)
q-grid `5,15,30,50,70,85,95`; metrics `ssim2`+`zensim`; full clean corpus (4497 renditions).
Monitor: `/tmp/chunk_fleet_monitor.sh` (bg) destroys each box on its `done/box-<idx>.done`
marker + logs `/tmp/chunk_fleet_monitor.log`. **A blind session: check boxes via
`hcloud server list | grep clean-`, destroy any idle leftover, then collect.**

**PIPELINE VALIDATED 2026-06-26 — zenjpeg clean picker SHIPPED.** Held-out TEST (7/9 origins):
argmin 0.47% / top-2 0.235% / top-3 0.165%, val→test +0.01pp (generalizes). Committed to
zenjpeg `main` 50c61658 (`benchmarks/zenjpeg_picker_v0.4_clean-imazen26-evenodd_2026-06-26.bin`).
TWO gotchas hit + fixed (do these for avif/jxl too):
1. `clean_features.tsv` (from `extract_features_for_picker` built `--features api`) has `name@hex8`
   columns + an `image_path` key — but `omni_to_pareto` needs **`feat_<name>` cols + a `variant_name`
   key**. Fix: post-process → `clean_features_vn.tsv` (add `variant_name` = basename w/o `.png`;
   rename `<name>@<hex>` → `feat_<name>`). 50 of the picker `_WANTED` features survive.
2. Each codec's `<codec>_picker.py::parse_config_name` was written for an OLD grammar and crashes on
   the current `scalar_dense`/`lossy_dense` names (zenjpeg choked on `jp3_tr14.75cpl+1cl1_small_420`).
   Fixed zenjpeg to extract the trellis float robustly (tolerate cpl/blur/bracket sub-knobs).
   **avif/jxl configs likely need the same robustness fix** before their train_hybrid runs.
Run timing: chunk fleet is ~2.7× slower than the optimistic estimate (half-util + sequential fetch —
the deferred outer-loop fix), ~5h wall for avif; still correct + within budget.

**Collect → train → bake → commit (per codec, once omni lands):**
1. Merge box omnis: `s5cmd cp 's3://zentrain/jxl-lossy/runs/<RUN>/omni/box-*.omni.tsv' .`
   then concat (one header). That IS the picker omni (image_path/codec/q/knob_tuple_json/
   encoded_bytes/score_ssim2/score_zensim).
2. `omni_to_pareto.py --omni <merged> --features-tsv
   /mnt/v/output/clean-picker-corpus-2026-06-26/clean_features.tsv --metric-col score_zensim
   --out-pareto … --out-features …` (variant_name join is exact: omni `/data/o_<stem>.scaleWxH.png`
   → `o_<stem>.scaleWxH` == clean_features variant_name).
3. `train_hybrid.py --codec-config <codec>_picker` (PYTHONPATH incl. scripts/picker) — origin
   split auto (even=train / 1,3,5=val / 7,9=test), reports val + TEST top-3-verify.
4. `bake_picker.py` → `.bin`; **commit the `.bin` into the codec crate** (`<codec>/benchmarks/`).

**Remaining after the lossy 3:**
- **jxl-lossless** — chunk-mode OOMs on modular (315 cells/image ramps to 13–24 GB in one
  process). Run it on a big-RAM box (ccx/cpx with ≥32 GB) at low `SWEEP_JOBS`, OR via the job
  system (fresh process per cell bounds memory). Plan: `modes_full --plan-budget 400` → ~315 modular cells/img.
- **jxl-lossy v0.2** — its omni is landing from `clean-jxllossy-213753`; supersede the interim bin.

**Efficiency follow-ups (the cell/chunk system, per user 2026-06-26):**
- Chunk worker fetches its chunk SEQUENTIALLY (one `s5cmd cp` per file) → slow startup on big
  chunks. Fix: batch via `s5cmd run` (parallel). Mitigation used now: more boxes = smaller chunks.
- `hetzner_cpu_sweep.sh` doesn't clean `/tmp/hz_chunk_*` before `split` → stale chunk files
  accumulate and get uploaded (harmless: boxes only claim `chunk-<their-idx>`, which are the real
  full-coverage chunks; extras are unclaimed waste). Fix: `rm -f /tmp/hz_chunk_*` before split.
- **Per-cell job system (the "cell system"): if kept (for lossless), GROUP jobs by source image**
  so the source decodes ONCE per group instead of per cell (the re-decode waste that made the
  jxl per-cell smoke ~0.3 enc/s vs chunk-mode's 72). Sort the manifest by `image_path` + cache the
  decoded source across consecutive same-source jobs in the worker/executor.

## Fleet execution notes — VALIDATED 2026-06-26 (read before launching the scaled run)

The full job-system path is **proven end-to-end** on Hetzner (real `zenmetrics jobexec`
executor → real JXL bitstreams `ff0a…` persisted as content-addressed blobs + Parquet
ledger; `clean-picker-corpus-2026-06-26` is uploaded to `codec-corpus/clean-picker-corpus-2026-06-26/`).
Two `launch_fleet.sh` bugs were found+fixed (commit fe0d0ec0):
- `ZEN_MANIFEST_FILE=<declared manifest>` — launch a REAL sweep (was hardcoded synthetic spec).
- real-manifest now defaults `ZEN_EXEC` to the baked `zenfleet-exec` shim (envblock's
  `-e ZEN_EXEC` overrode the image default → a real launch silently ran `/bin/cat` = fake blobs).
- `N_JOBS=0` skips the local tier (fleet-only; keep the shared workstation responsive).
Declare path: `zenmetrics sweep --plan … --dry-run --emit-cells cells.jsonl` (rewrite
`image_path` → basename so the worker resolves via `ZEN_CORPUS_PREFIX`) → `zenfleet-ctl
declare-encodes --cells … --out manifest.json` → `ZEN_MANIFEST_FILE=manifest.json
ZEN_WORKER_IMAGE=$ZEN_FLEET_IMAGE_CPU ZEN_CORPUS_BUCKET=codec-corpus
ZEN_CORPUS_PREFIX=clean-picker-corpus-2026-06-26 launch_fleet.sh 0 <N_HZ> 0 0`.

⚠ **EFFICIENCY (settled before scaling): per-cell ENCODE jobs re-decode the source PNG for
EVERY cell** (jobexec `run_one_job` decodes per call), so a dense per-image grid (560
jxl-lossy cells/image) decodes each source ~560×. Measured ~0.3–0.5 enc/s on a cpx22 — at
8.9M cells that is many box-hours and over the $ budget. For a DENSE per-image sweep the
efficient tool is **chunk mode** (`zenmetrics sweep` decodes each source ONCE, encodes all
its cells in-process; `--encoded-out-dir` persists variants) — i.e. the Hetzner split tool
`scripts/sweep/hetzner_cpu_sweep.sh`. The per-cell job system stays the right tool for
sparse/heterogeneous/long-tail-resumable work and for jxl-LOSSLESS (modular memory needs a
fresh process per cell). Decision for the scaled run: chunk mode for the lossy codecs +
job system for jxl-lossless. (Box type: use cpx41/cpx51, not the cpx22 launcher default.)

## HDR + gain-map track (user-greenlit 2026-06-26, parallel to the SDR pickers)

Goal: alongside SDR renditions, emit **SDR / HDR / gain-map triples** so an HDR /
Ultra-HDR picker has clean inputs. Status: inputs identified + manifested; tool is
the next bounded build.

- **Inputs (manifested):** `/mnt/v/output/imazen-26-features/imazen26_hdr_gainmap_pairs.tsv`
  (`origin split sdr_path hdr_path`) — **76 SDR+HDR pairs, 38 train / 20 val / 18 test**
  under the even/odd split. SDR = 8-bit `.sdr.png`; HDR = **16-bit PQ** `.hdr.png`.
- **Derive (NOT extract — the corpus is PQ-HDR-PNG, not Ultra HDR containers; originals
  sampled are plain JPEG).** Use `ultrahdr-rs` (READ-ONLY dep — a concurrent agent has a
  worktree there; never edit it): `Encoder.set_sdr_image(sdr).set_hdr_image(hdr).encode()`
  → `Decoder::new(jpeg).decode_gainmap()` → write the `GainMap` as standalone
  `o_<stem>.gainmap.png` (+ the `GainMapMetadata`: min/max log2, gamma, offsets, hdr capacity).
  ⚠ FORMAT: `set_hdr_image` expects ultrahdr-rs's HDR `PixelBuffer` format — match the
  16-bit PQ `.hdr.png` decode to it exactly (PQ EOTF → the expected linear/encoded layout),
  or the derived map is garbage. Verify one pair's round-trip (decode_hdr ≈ input) before batch.
- **Output:** `o_<stem>.sdr.png` / `.hdr.png` / `.gainmap.png` triples + a provenance index
  (rendition→origin→split→metadata). Then they can sweep/score like the SDR set.
- **HDR scoring is unblocked now:** cvvdp runs on CPU (C1/C1b), so the HDR picker is no longer
  blocked on a GPU-only cvvdp build (was the `hdr-picker-blocked-encode-infra` blocker).
