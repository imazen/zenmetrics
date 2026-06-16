# zenmetrics recovery register — 2026-05-08

Compiled from a read-only sweep of zenmetrics branches with last-commit ≥ 2026-04-25. Cross-ref: `~/work/zen/RECOVERY_PLAN_2026-05-08.md`.

## Verdict table

| branch | commit | date | item | what it adds | verdict | files |
|---|---|---|---|---|---|---|
| (HEAD detached) | `d1560b8` | 2026-05-07 | per-fleet janitor | reaps idle workers (<100 cells/min OR <5% CPU after 8-min grace); destroys fleet when TSVs reach target | **kept (live in v15)** | `scripts/sweep/sweep_janitor.py` |
| master | `edb98eb` | 2026-05-06 | Dockerfile.sweep v13 | bundles `zenmetrics` binary for vast.ai (GHCR `imazen/zenmetrics-sweep:0.6.3`) | kept | `Dockerfile.sweep`, `Dockerfile.sweep.v13` |
| feat/butteraugli-multi-column | `4da7b8e` | 2026-05-04 | expert JXL knob decouple + butteraugli multi-column emit | separates effort macro into fine-grained expert overrides; emits `score_butteraugli_max` and `score_butteraugli_pnorm3` columns | kept — required for zentrain multi-target loss | `crates/zenmetrics-cli/src/sweep/` |
| feat/sweep-v12-balanced | (recent) | 2026-05-06 | balanced cross-codec v12 sweep design | per-codec cell budgets aligned for fair training | partial — informs v16 | sweep config |
| feat/dockerfile-sweep | (recent) | 2026-05-06 | Dockerfile.sweep refresh | aligns with 0.6.8 binary; includes runtime deps for libwebp/libaom/libjxl | kept | `Dockerfile.sweep` |
| feat/migrate-sweep-scripts | (recent) | 2026-05-05 | move sweep scripts under `scripts/sweep/` | reorg; no behavior change | kept | `scripts/sweep/v15/`, `scripts/sweep/sweep_diag.py` |
| feat/features-backfill | (recent) | 2026-05-04 | backfill 300-feature parquets into existing TSVs | adds missing zensim features to parquet sidecars | kept | feature writer |
| fix/sweep-binary-path | (recent) | 2026-05-06 | rayon-binary path fix | `0.6.8` binary in expected location | kept | onstart script |
| fix/patch-jxl-encoder-dos-fix | (recent) | 2026-05-06 | upstream jxl encoder DoS patch | security fix | kept | encoder patch |
| security-fixes-h1-h4 | (recent) | 2026-05-06 | security audit fixes (4 issues) | various | kept | various |
| (live, not yet a sweep) | — | 2026-05-07 | **v16 cross-codec sweep — empty/blocked** | `scripts/sweep/v16/` exists but no `chunks_gpu.jsonl`, no `launch.sh`. Was intended cross-codec follow-up to v15. The 2026-05-07 attempt produced empty score columns (workers reported "[done]" but TSVs missing GPU metric values) — root cause likely binary-path mismatch or GPU unavailability on offered boxes | **blocked** — needs single-box smoke before re-launch | `scripts/sweep/v16/` (mostly empty) |

## Cross-cutting status: v15 sweep COMPLETE

- 30 vast.ai boxes, ~50 chunks distributed (2026-05-06 launch)
- GPU TSV columns populated (ssim2_gpu, butteraugli_gpu)
- Tracking artifact: `/tmp/v15-prep/v15_instances.txt`, `scripts/sweep/v15/chunks_gpu.jsonl`
- **Action**: mirror v15 TSVs to R2 + backfill CPU dssim before training pickers/zensim
- Janitor + atomic chunk-claim infra is robust: cgroup-aware parallelism (3-5× throughput on multi-core boxes); atomic chunk claim via S3 If-None-Match; orphaned-claim recovery scripted

## v16 cross-codec sweep — needed for Phase 3 corpus expansion

Per the user's direction, zensim's primary training authority is the **synthetic compression of CID22 _training_ set sources scored against ssim2** — and the canonical `training_safe_synthetic_extended.csv` already has that. v16 was intended to add cross-codec coverage (jxl/avif/webp on the same source corpus) as **augmentation**, not replacement.

Open design questions for v16:
1. Source corpus: keep CID22 training tiles + CLIC + kodak + gb82-sc (canonical)? Or use the v15r 981-image corpus (OpenAI-tagged web content)?
2. Codec grid: zenjpeg-420-{e0,e1,e2,e4} + zenwebp + zenavif + zenjxl + zenpng. ~150 q-points each = 750 q-points × ~3000 sources × 5 codecs = ~11M cells. Likely too large.
3. Subsampling: 20-50 centroid sources per content class × 6 sizes × 5 codecs × 21 q-points = ~25k cells/codec, ~125k total. Feasible.

Single-box smoke needs to confirm: (a) `zenmetrics --version` works inside the docker image; (b) GPU is accessible (`nvidia-smi` returns); (c) one cell completes with all metric columns populated.

## Cherry-picks for main (Phase 2)

1. **feat/butteraugli-multi-column** → `master` if not already there. Multi-target emission is a hard requirement for any zentrain multi-task zensim trainer. Verify it's on master and confirm column names match what zentrain consumes.
2. **fix/patch-jxl-encoder-dos-fix** + **security-fixes-h1-h4** → master (security; should be high-priority).
3. **fix/sweep-binary-path** → master if not already there.
4. **Document v15 completion** + janitor usage in `docs/SWEEP_OPERATIONS.md` so a future agent doesn't reinvent the wheel.

## Drop / archive

- Older sweep configs that v15 supersedes — archive in `recovered-archive/sweep-pre-v15`.
- Any spike branches with no measured outcome.

## Notable docs to preserve

- `scripts/sweep/v15/launch_gpu.sh` — reference launcher (the v16 launcher should adapt this).
- `scripts/sweep/sweep_janitor.py` — reaper logic.
- `scripts/sweep/sweep_diag.py` — per-worker waste analyzer.
- The cgroup-aware parallelism fix (~2026-05-04) — root cause that 3-5×'d throughput.
