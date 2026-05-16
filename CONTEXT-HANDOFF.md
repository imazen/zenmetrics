# CONTEXT-HANDOFF — 2026-05-16 ~01:00 MDT

Session context for resumption. Active goal: **cvvdp-gpu production-ready
AND vast.ai backfill cvvdp metrics**.

---

## TL;DR — state at handoff

- **cvvdp-gpu**: production-ready. 414+ ticks of test/doc pins shipped
  on `feat/cvvdp-gpu-scaffold`. Parity-locked to pycvvdp v0.5.4 within
  0.005 JOD. Constants pinned across pool/csf/pyramid/display/color/
  masking/lib_constants/goldens_metadata/params_placeholder/error_traits.
  GPU memory predictor + `recommend_parallel` API shipped.

- **vast.ai backfill**: **in progress**, ~1018 / 6000 sidecars
  (17%, half-corpus target). 53 boxes running. Auto-PARALLEL just
  shipped (`127684c`) — new boxes will auto-detect optimal
  concurrency based on cgroup CPU + RAM + free GPU memory.
  ETA at current 1188/hr rate: ~4.2 hours to half-corpus DONE.

- **Goal status**: cvvdp-gpu DONE; backfill IN PROGRESS (running).

---

## Repo layout

- **Primary worktree (cvvdp-gpu loop)**: `~/work/zen/zenmetrics--cvvdp/`
- **Secondary worktree (sweep edits)**: `~/work/zen/zenmetrics--cvvdp-new/`
- **Git working dir (commit ops)**: `~/work/zen/zenmetrics/.git`
- **Branch**: `feat/cvvdp-gpu-scaffold` (pushed to origin)
- **Both worktrees use jj (Jujutsu)**: `jj log`, `jj new -m "..."`, etc.

---

## Active R2 / vast.ai infrastructure

### R2 paths (account `${R2_ACCOUNT_ID}` from `~/.config/cloudflare/r2-credentials`)

- **Working binary** (latest production):
  `s3://coefficient/binaries/zen-metrics-0.6.4-cvvdp-cuda12.6-v4-batched-linux-x86_64-gpu`
  - Built from local CUDA 12.6 toolchain via `cargo build -p zen-metrics-cli --features sweep,gpu,gpu-cuda,gpu-cpu`
  - Contains CvvdpBatchScorer (OOM fix, tick 384)
  - Verified: `strings | grep -i cuCoredumpDeregisterComplete` returns empty
  - 132 MB stripped

- **Sidecars (in-progress)**: `s3://zentrain/cvvdp-backfill-2026-05-15-half/cvvdp_imazen/`
  - 1018 parquets at handoff time
  - Each contains 100 rows: `image_path / codec / q / knob_tuple_json / cvvdp_imazen_v0_0_1`
  - All real JOD scores, 0 NaN

- **Chunks manifest**: `s3://coefficient/jobs/cvvdp-backfill-imazen-2026-05-15/chunks.jsonl`
  (12000 chunks total; half-corpus = 6000)

- **Onstart script** (workers fetch this at boot):
  `s3://coefficient/jobs/cvvdp-backfill-imazen-2026-05-15/onstart_cvvdp_backfill_imazen.sh`

- **Chunk worker** (per-chunk processing):
  `s3://coefficient/jobs/cvvdp-backfill-imazen-2026-05-15/cvvdp_backfill_chunk_worker.sh`

- **Failure logs**: `s3://coefficient/logs/cvvdp-backfill-imazen-2026-05-15/`
  (64 failures total at handoff; auto-cull holding)

- **Claims**: `s3://coefficient/claims/cvvdp-backfill-imazen-2026-05-15/`

- **Heartbeats**: `s3://coefficient/heartbeats/cvvdp-backfill-imazen-2026-05-15/`

### Vast.ai fleet (run ID `cvvdp-backfill-imazen-2026-05-15`)

- **Active**: ~53 boxes mixing PARALLEL=1, PARALLEL=2, and (new)
  PARALLEL=auto. The newest 5 boxes use the auto-PARALLEL code
  from commit `127684c`.
- **Instance manifest**: `/tmp/cvvdp-backfill-imazen-cvvdp-backfill-imazen-2026-05-15/instances.txt`
  (LAST batch only — earlier batches' IDs lost when launch_imazen.sh
  overwrote the file; use `vastai show instances` to enumerate)
- **Cost**: $0.04-0.07/hr per box; total burn so far ~$5-8

### Recent commits (origin/feat/cvvdp-gpu-scaffold tip)

```
127684c feat(sweep): port v15 cgroup-aware PARALLEL detection + add GPU memory cap
00c5875 docs(cvvdp-gpu): README Build section — CUDA 12.6 SDK works for non-Blackwell GPUs
effd0ae feat(sweep): auto-detect PARALLEL from free GPU memory + CPU cores at onstart
f861d02 test(cvvdp-gpu): pin band_frequencies invariants — positive, decreasing, min-dim boundary
ebe21f8 test(cvvdp-gpu): pin Cvvdp::score determinism across repeated calls on same instance
57fc522 test(cvvdp-gpu): direct unit tests for kernels::masking::safe_pow
8177979 test(cvvdp-gpu): pin Error Clone + std::error::Error trait contract
7c4d475 docs(cvvdp-gpu): doc N_RHO + re-document N_L_BKG (separate the joint comment)
14582e8 test(cvvdp-gpu): pin Cvvdp::score = f64::from(compute_dkl_jod) bit-equality
... (~400+ ticks earlier)
```

---

## Active background tasks

- **Monitors** (persistent):
  - `bgxa4altm` — Auto-cull wedged workers (≥3 failures in 5min)
  - `b9hvrtwy8`, `bc9oo3qbr`, `b9rrssewv`, `bhw9c238r`, `b9rrssewv`,
    `bega8kiq6` (some stale) — heartbeat/sidecar event streams. Many
    are old from v19-v24 fleets; can TaskStop the stale ones safely.

- **Wakeup**: A `/loop` wakeup is scheduled for ~01:13:00 MDT that
  will re-fire the cvvdp-gpu loop prompt and check fleet progress.

---

## How to resume

1. **Check fleet progress**:
   ```bash
   source ~/.config/cloudflare/r2-credentials
   EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
   SC=$(s5cmd --endpoint-url "$EP" --profile r2 ls s3://zentrain/cvvdp-backfill-2026-05-15-half/cvvdp_imazen/ | wc -l)
   FAIL=$(s5cmd --endpoint-url "$EP" --profile r2 ls s3://coefficient/logs/cvvdp-backfill-imazen-2026-05-15/ | wc -l)
   echo "Sidecars: $SC / 6000 ($(python3 -c "print(round($SC/6000*100,1))")%), failures: $FAIL"
   ```

2. **If sidecars >= 5500**, run finalize:
   ```bash
   cd ~/work/zen/zenmetrics--cvvdp-new
   SWEEP_RUN_ID=cvvdp-backfill-2026-05-15-half WORK=/tmp/cvvdp-finalize-final \
     UPLOAD_CONSOLIDATED=0 bash scripts/sweep/cvvdp_backfill/finalize.sh
   ```
   Then commit the manifest:
   ```bash
   cp /tmp/cvvdp-finalize-final/manifest.json benchmarks/cvvdp_backfill_half_2026-05-16.json
   cd ~/work/zen/zenmetrics--cvvdp-new
   jj new -m "feat(sweep): commit cvvdp-backfill half-corpus DONE manifest"
   jj bookmark set feat/cvvdp-gpu-scaffold -r @
   jj git push --bookmark feat/cvvdp-gpu-scaffold
   ```

3. **If still running**, scale up if rate < 20/min via:
   ```bash
   cd ~/work/zen/zenmetrics--cvvdp-new
   SWEEP_RUN_ID=cvvdp-backfill-imazen-2026-05-15 N_BOXES=10 MAX_DPH=0.30 MIN_RAM_GB=16 \
     SWEEP_BIN_OVERRIDE=s3://coefficient/binaries/zen-metrics-0.6.4-cvvdp-cuda12.6-v4-batched-linux-x86_64-gpu \
     bash scripts/sweep/cvvdp_backfill/launch_imazen.sh
   ```
   (new launcher defaults PARALLEL=0 = auto-detect)

4. **Cull wedged workers** (if any single worker has many recent
   failures):
   ```bash
   # Find the wedged worker
   for f in $(s5cmd --endpoint-url "$EP" --profile r2 ls s3://coefficient/logs/cvvdp-backfill-imazen-2026-05-15/ | sort -k1,2 -r | head -10 | awk '{print $NF}'); do
     CID="${f%.fail.log}"
     W=$(s5cmd --endpoint-url "$EP" --profile r2 cat "s3://coefficient/claims/cvvdp-backfill-imazen-2026-05-15/${CID}.claim" 2>/dev/null | awk -F'\t' '{print $3}')
     echo "$W"
   done | sort | uniq -c | sort -rn | head -3
   # Destroy by label (find ID via vastai show instances --raw)
   ```

5. **Tear down** when done:
   ```bash
   bash scripts/sweep/destroy_all.sh
   ```

---

## Key engineering history (v22→v25 cascade)

The OOM cascade that consumed multiple ticks is captured in commits:

1. **Tick 382 (`db7b80a`)** — CUDA 12.6 binary build (avoids broken
   `cuCoredumpDeregisterCompleteCallback` symbol from cuda-13020).
2. **Tick 383 (`0f9ccf7` then `91...`)** — SWEEP_BIN_OVERRIDE +
   cuda_path stub.
3. **Tick 384 (`0422c96`)** — **The OOM fix**: `CvvdpBatchScorer`
   caches `Cvvdp::new` across pairs in score-pairs. Per-pair
   re-allocation was fragmenting GPU memory + ballooning host-pinned
   NVRTC PTX cache.
4. **Tick 386 (`946f5c2`)** — `cuda-cudart-dev-12-6` install. cubecl
   emits kernel source with `#include <cuda_runtime.h>` and NVRTC
   needs the headers at runtime; without them, every kernel compile
   fails → 100 NaN rows per chunk (caught by SSH'ing into a v24
   worker and getting the actual NVRTC error behind the masking
   `InvalidImageSize` error variant).
5. **Tick 413 (`effd0ae`)** — Auto-detect PARALLEL via nvidia-smi +
   nproc.
6. **Tick 414 / today's last** (`127684c`) — Port v15's
   cgroup-aware logic (handles vast.ai-style host-vs-container
   nproc mismatch + RAM cap) + GPU memory cap.

---

## What NOT to do

- **Don't rebuild the binary unless tests fail**. Current v4 binary
  is verified-working end-to-end (sample sidecar
  `v15r_zenjpeg-3586.parquet` has 100 rows, 0 NaN, JOD in [9.66,
  9.91]).
- **Don't force-push `feat/cvvdp-gpu-scaffold`**. CLAUDE.md global
  rule — destructive ops require explicit user OK.
- **Don't change the algorithm**. Recent ticks are pure
  test/doc/sweep-infra additions; the scoring code hasn't moved
  since `0422c96`. The fleet IS running the correct algorithm.
- **Don't kill all monitors**. Some are stale (v19-v22 fleets) but
  harmless. `bgxa4altm` (auto-cull) is the only one that matters
  for fleet hygiene.
- **Don't `cargo clean`** between ticks — it forces NVRTC + cudarc
  recompile (~6 min cold).

---

## Outstanding work (besides backfill completion)

- Validate auto-PARALLEL on a real box (SSH in, check log line
  `auto-detect PARALLEL=N (cgroup_cpu=… cpu_cap=… ram_cap=… gpu_cap=…)`).
- Decide whether to do full-corpus (12000 chunks) after half-corpus
  completes. Half = 6000 was the user-authorized scope.
- The 64 historical failures haven't been retried — the chunks they
  represent might be unscored. `finalize.sh` skips missing chunks
  silently; user should decide whether to re-run them.
- pycvvdp parity column (`cvvdp_pycvvdp_v054`) is not being
  produced — the imazen-only fleet path skips it. To get parity
  validation, would need to also run pycvvdp_worker.py on the same
  chunks (separate fleet or merged image).
