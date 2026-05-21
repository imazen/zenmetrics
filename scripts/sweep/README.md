# vast.ai metric backfill — operator guide

**This directory contains everything needed to score a corpus of
(source, distorted) image pairs against one or more GPU metrics on
rented vast.ai boxes.** If you've never run this before, READ THE
WHOLE FILE before typing any `vastai` command. It can spend real
money fast.

## Architecture (post 2026-05-19)

The sweep infra is now a **unified Rust worker** living in
`crates/vastai-fleet`. One binary, two operational modes:

| Mode | Purpose | When to use |
|---|---|---|
| `omni` (default) | Encode each cell + score 6 GPU metrics + save encoded variant + write sidecar | Fresh sweeps; first pass against new chunks.jsonl |
| `feature-backfill` | Read existing omni sidecar from R2, download cached encoded variants, compute CPU zensim's 300-feature vector per cell **without re-encoding** | Get features for an already-encoded corpus |

The dispatch loop, claim mechanism, R2 IO, parquet IO, and CubeCL
session all live in this one binary. The previous bash chain
(`onstart_omni_backfill.sh` + `omni_backfill_chunk_worker.sh`) is
kept as a fallback path but **production uses the Rust worker**.

**Why this matters:** Phase B's in-process pipeline ships one CubeCL
init per worker process (was: one per group, ~30× per chunk).
Measured 2.7× throughput vs the bash predecessor (442 vs 165
chunks/hr on the same 6-box fleet) with GPU util 47-65% / CPU util
90-95% (was: 0-6% / 42-78%). See task #73 commit notes for
details.

## Container images

| Tag | What it ships | Pin for |
|---|---|---|
| `v26` | Single-file collapsed image: Inline-sweep Rust worker (omni / feature-backfill / source-features modes) + 6 GPU metrics CUDA-12.0-bound + all-local codec deps + cuda_dlsym_stub LD_PRELOAD | All new sweep work |

Always pin the tag — `:latest` doesn't exist. Build from
`Dockerfile.sweep.v26` (single source of truth — no chain).

**History (2026-05-21):** the v14→v25 incremental chain was
collapsed into single-file `Dockerfile.sweep.v26`. All earlier
vNN Dockerfiles were deleted; their deltas are inlined in v26 in
proper layer order. See the git log of those deleted files for
incremental change history (cuda_dlsym_stub evolution, jxl-encoder
bumps, cudarc binding pin chase, vastai-fleet worker rollouts).

## The proven end-to-end pipeline (2026-05-19)

This is the path that landed 2933 omni sidecars + 2933 zensim
feature parquets across two runs (`cvvdp-v15rc-2026-05-18` and
`omni-multi-codec-2026-05-19`):

1. **Generate chunks.jsonl** — `generate_cvvdp_backfill_chunks.py
   --filter-codec v15rc_zenjpeg` (or per codec). Upload to
   `s3://coefficient/jobs/<run-id>/chunks.jsonl`.
2. **Single-instance smoke (omni mode)** — `launch_single_instance.sh
   --docker ghcr.io/imazen/zen-metrics-sweep:v26 --onstart
   onstart_unified.sh`. Verify the first sidecar lands at
   `s3://zentrain/<run-id>/omni/<chunk>.parquet`. Schema check
   should show all 6 metric columns + `encoded_filename` non-empty.
3. **Fleet fanout (omni)** — `launch_backfill.sh --n-boxes 6
   --docker :v26 --onstart onstart_unified.sh`. PC=2 default; AIMD
   tunes between 1-4 based on `nvidia-smi` util.
4. **Watch omni sidecars** populate. ~50 chunks/hr/box with v23.
   `vastai-fleet watch --target-sidecars <N>` auto-destroys at end.
5. **Single-instance smoke (feature-backfill mode)** —
   `launch_single_instance.sh --docker :v26 --onstart
   onstart_feature_backfill.sh`. Verifies the feature parquet
   lands at `s3://zentrain/<run-id>/zensim_features/<chunk>.parquet`.
6. **Fleet fanout (feature-backfill)** — same launcher with v24
   image. Per-chunk runtime is dominated by encoded-variant
   download from R2 + CPU zensim compute (~5 sec/chunk on a
   modern CPU).
7. **Auto-destroy + verify** — `vastai-fleet watch` + a sidecar
   count check.

## Known constraint: chunks with `encoded_filename: null`

The v22 omni runs that pre-date the `--distorted-out-dir` worker
fix produced omni sidecars where every row's `encoded_filename` is
the arrow Null dtype — no encoded variants were ever uploaded to
R2 for those chunks. Feature-backfill mode cannot process them
(nothing to score). The fix is a **re-encode pass**:

```bash
# 1. Find affected chunks
omni=$(s5cmd ls s3://zentrain/<run-id>/omni/ | awk '{print $NF}' | sed 's/.parquet//')
feat=$(s5cmd ls s3://zentrain/<run-id>/zensim_features/ | awk '{print $NF}' | sed 's/.parquet//')
missing=$(comm -23 <(echo "$omni" | sort) <(echo "$feat" | sort))

# 2. Build a chunks.jsonl with just those, upload to a fresh run prefix
# 3. Launch omni-mode fleet against that file:
launch_backfill.sh --docker :v26 --onstart onstart_unified.sh \
    --run-id v15rc-reencode-<DATE> --chunks <fresh-prefix>/chunks.jsonl
# This overwrites the omni sidecars + uploads encoded variants to
# the original run's encoded/ prefix (because each chunk record's
# run_id field is preserved). The freshly-updated omni sidecars
# now have string-typed encoded_filename.
# 4. Then run feature-backfill against those chunks.
```

Cost for a 346-chunk reencode pass: ~$2 (5 boxes × 90 min).

---

---

## Cost-safety checklist (read first)

Before you touch anything:

1. **vast.ai balance.** Check `vastai show user --raw | jq .credit`.
   Anything under $5 means you can't even create instances (per-account
   minimum). $12 budget covers a full v15rc-class backfill (~500 k cells)
   with sharded fanout; less if you're imprudent.
2. **Bandwidth is the silent killer.** Vast.ai HOSTS charge $0.026-0.03
   per GB transferred. A 3 GB docker image pull × 30 boxes = $2.34 in
   bandwidth alone before any compute. Plus source-image syncs. **Always
   shard sources across boxes (one box owns one slice of basenames) or
   you'll re-download the same data 5-10× across the fleet.**
3. **The trap wrapper is your friend.** Every box boots through
   `run_with_error_trap.sh` which (a) refuses to start without a visible
   GPU, (b) self-destroys on rc!=0 with stderr uploaded to R2. Don't
   bypass it.
4. **Check `vastai show instances-v1` regularly.** Orphan instances are
   the #1 source of overruns. If a `vastai destroy` was interrupted
   mid-prompt, the box keeps running. The `vast_cost_watch.sh` script
   in this dir polls and alerts.
5. **NEVER run `launch_backfill.sh` without first running `launch_single_instance.sh`** on the same `--onstart` + `--docker` combo. The
   single-instance smoke catches every common bug before you fan out
   to 30 boxes.

---

## The canonical happy path

```bash
# 1. Sanity-check creds + balance
vastai show user --raw | python3 -c 'import json,sys;d=json.load(sys.stdin);print(f"credit ${d[\"credit\"]:.2f}")'
# Must print > $5; refill at console.vast.ai if not.

# 2. Generate chunks for your sweep + corpus
python3 scripts/sweep/generate_cvvdp_backfill_chunks.py \
    --unified-dir /mnt/v/zen/zensim-training/2026-05-07/unified \
    --run-id <YYYY-MM-DD-NICK> \
    --source-r2-prefix s3://zentrain/sweep-v15r-2026-05-06/sources \
    --input-r2-prefix s3://zentrain/unified-2026-05-07 \
    --output-r2-prefix s3://zentrain/<YYYY-MM-DD-NICK> \
    --chunk-size 200 \
    --out /tmp/<NICK>/chunks.jsonl
# 3. Upload chunks + worker to R2
s5cmd --profile r2 --endpoint-url $R2_ENDPOINT cp \
    /tmp/<NICK>/chunks.jsonl s3://coefficient/jobs/<YYYY-MM-DD-NICK>/chunks.jsonl
s5cmd --profile r2 --endpoint-url $R2_ENDPOINT cp \
    scripts/sweep/omni_backfill_chunk_worker.sh \
    s3://coefficient/jobs/<YYYY-MM-DD-NICK>/omni_backfill_chunk_worker.sh

# 4. SMOKE: ONE box, SKIP_CLAIMS=1, watch it produce 1 sidecar
SKIP_CLAIMS=1 ./scripts/sweep/launch_single_instance.sh \
    --metric cvvdp \
    --run-id <YYYY-MM-DD-NICK> \
    --chunks s3://coefficient/jobs/<YYYY-MM-DD-NICK>/chunks.jsonl \
    --docker ghcr.io/imazen/zen-metrics-sweep:v17 \
    --onstart scripts/sweep/onstart_omni_backfill.sh \
    --max-dph 0.10 --min-gpu-ram-mb 8000

# 5. Watch it. Sidecar at s3://zentrain/<RUN-ID>/omni/<chunk>.parquet
#    means the pipeline works.
watch -n 60 's5cmd --profile r2 --endpoint-url $R2_ENDPOINT ls s3://zentrain/<RUN-ID>/omni/ | wc -l'

# 6. When the smoke produces sidecars at a healthy rate, FANOUT
./scripts/sweep/launch_backfill.sh \
    --metric cvvdp \
    --run-id <YYYY-MM-DD-NICK> \
    --chunks s3://coefficient/jobs/<YYYY-MM-DD-NICK>/chunks.jsonl \
    --docker ghcr.io/imazen/zen-metrics-sweep:v17 \
    --onstart scripts/sweep/onstart_omni_backfill.sh \
    --n-boxes 10 --max-dph 0.10

# 7. Auto-destroy when target sidecar count is reached
vastai-fleet watch \
    --label-prefix <YYYY-MM-DD-NICK> \
    --target-sidecars <N_CHUNKS_MINUS_GRACE> \
    --r2-prefix s3://zentrain/<YYYY-MM-DD-NICK>/

# 8. Always verify the fleet is gone
vastai-fleet status --label-prefix <YYYY-MM-DD-NICK>
# Should print "instances: 0".
```

---

## File map — what does what

### Images (Dockerfiles → published tags on `ghcr.io/imazen/zen-metrics-sweep:*`)

| File | Tag | Status | Notes |
|---|---|---|---|
| `Dockerfile.sweep.v26` | **`v26` (recommended)** | ✅ shipping | Single-file collapsed image (replaces the v14→v25 chain). FROM ubuntu:24.04 directly. Bakes apt deps + CUDA NVRTC+dev 12-6 + pyarrow + s5cmd + jq + cuda_dlsym_stub.so + zen-metrics (CUDARC_CUDA_VERSION=12000) + vastai-fleet (inline-sweep) + all onstart/worker scripts. Supports omni, feature-backfill, source-features modes. |
| `scripts/sweep/Dockerfile.pycvvdp` | `pycvvdp` | active (rare) | Only used by the dual-impl cvvdp parity flow. Separate from the main sweep image because pycvvdp pulls in ~3 GB of pytorch. |

**Historical (deleted 2026-05-21):** the v14→v25 chain (Dockerfile.sweep
+ Dockerfile.sweep.v13 + Dockerfile.sweep.v14 + .v15 + .v18 + .v19 +
.v21 + .v22 + .v23 + .v24 + .v25) was collapsed into single-file v26.
Each prior file FROMed the previous tag on ghcr.io — fine for shipping
deltas as small layers, but bad for new contributors trying to
understand what the image is. v26 inlines every delta in proper layer
order. See `git log -- Dockerfile.sweep.v*` for the incremental
history (cuda_dlsym_stub evolution, jxl-encoder bumps, cudarc binding
pin chase, vastai-fleet worker rollouts).

### Onstart scripts (entrypoint for each container)

| File | Used by | Status |
|---|---|---|
| `onstart_unified.sh` | **omni mode via the Rust `vastai-fleet worker` binary** | ✅ recommended (v26) |
| `onstart_feature_backfill.sh` | **feature-backfill mode via the Rust worker (sets WORKER_MODE=feature-backfill)** | ✅ recommended (v26) |
| `onstart_omni_backfill.sh` | Legacy bash dispatcher for the omni pipeline | active (fallback) |
| `onstart_cvvdp_backfill_imazen.sh` | cvvdp single-impl backfill | active |
| `onstart_cvvdp_backfill.sh` | cvvdp dual-impl (cvvdp-gpu + pycvvdp) | active (rare) |
| `onstart_iwssim_backfill_v14.sh` | iwssim backfill (v14 image baseline) | active |
| `onstart_iwssim_backfill.sh` | iwssim backfill (legacy v3 image) | deprecated |
| `onstart_v2.sh`, `onstart_v3.sh` | pre-v14 legacy generic worker | deprecated |

### Chunk workers (process one chunk = 100-200 rows)

The Rust worker (`crates/vastai-fleet/src/worker/`) replaces these
bash scripts when the container runs `onstart_unified.sh` or
`onstart_feature_backfill.sh`. The bash workers stay in the image
as safety-net fallbacks (`vastai-fleet worker` falls through to
the bash `omni_backfill_chunk_worker.sh` if the inline path
fails — defence in depth).

| File | Used by | Status |
|---|---|---|
| `omni_backfill_chunk_worker.sh` | Legacy bash worker (now: Rust fallback path only) | active (fallback) |
| `metric_backfill_chunk_worker.sh` | single-metric backfills (iwssim/ssim2/cvvdp-imazen) | active |
| `cvvdp_backfill_chunk_worker.sh` | cvvdp dual-impl onstart | active |
| `iwssim_backfill_chunk_worker.sh` | legacy iwssim-only onstart | deprecated (superseded by metric_backfill) |

### Launchers

| File | Use for | Status |
|---|---|---|
| `launch_single_instance.sh` | 1 box smoke test, iterating on a fix | ✅ current |
| `launch_backfill.sh` | N-box fleet fanout (requires n ≥ 3) | ✅ current |
| `deploy_fast.sh` | legacy fast-deploy, pre-vastai-fleet | deprecated |
| `dispatch.sh` | legacy cron-driven dispatcher | deprecated |
| `vastai_zen_metrics_sweep.sh` | legacy v3-era launcher | deprecated |

### Chunk generators

| File | Output |
|---|---|
| `generate_cvvdp_backfill_chunks.py` | chunks.jsonl from unified-V_X parquets (any metric — name predates the omni mode) |
| `generate_jobspecs.py` | legacy jobspec format (pre-chunks.jsonl) |
| `generate_jobspecs_v06.py` | legacy v06 sweep jobspec |

### Helpers

| File | Purpose |
|---|---|
| `run_with_error_trap.sh` | EXIT-trap wrapper. nvidia-smi pre-flight + self-destroy on rc!=0 + stderr upload. **Every onstart should be invoked through this.** |
| `cuda_dlsym_stub.c` | LD_PRELOAD shim. Fixes cudarc 0.19.4 vs CUDA 13.x driver symbol mismatch. Baked into v17 image. |
| `fleet_util_snapshot.sh` | Per-box GPU/CPU/RAM/uptime dump. Auto-detects fleet boxes by label prefix. Use to verify util after launch. |
| `sweep_janitor.py` | Sidecar consolidation + dedup |
| `fleet_status.sh` | One-shot dashboard wrapping `vastai-fleet status` |
| `finalize.sh` | Post-sweep R2-sidecar consolidation into per-codec parquets |
| `vast_cost_watch.sh` | Continuous burn-rate monitor; alerts if total cost exceeds threshold |

---

## Failure modes (read before launching)

| Symptom | Diagnosis | Fix |
|---|---|---|
| Instance stays in `cur_state=stopped` forever | vast.ai now requires explicit `vastai start instance <ID>` after create | Already fixed in `launch_single_instance.sh`. If you wrote your own launcher, add the start call. |
| Onstart bash dies with `ldconfig: command not found` | Image PATH dropped /sbin | Use v15+ image (has `ENV PATH=/usr/local/sbin:/usr/sbin:...`). |
| `xargs: invalid number "auto" for -P` | Onstart got PARALLEL=auto, expected numeric | Use v15+ onstart (treats auto as 0 = rayon auto-detect). |
| Every cell panics with `cuCoredumpDeregisterCompleteCallback` undefined symbol | cudarc 0.19.4 vs CUDA 13.x driver | Use v17 image (has cuda_dlsym_stub.so LD_PRELOAD shim). |
| Sidecars don't appear; box idles at 0% GPU | Claim markers from a previous run blocked all chunks | Set `SKIP_CLAIMS=1` for smoke runs, OR clear `s3://coefficient/claims/<run-id>/` first. |
| Bandwidth charges crush the budget | Each box re-downloads source images redundantly | Use a sharded chunk file (one source per shard) OR launch with `WORKER_INDEX`/`WORKER_COUNT` so each box owns a slice. (Sharding pending — see task #72.) |
| GHCR pull fails with 401 unauthorized | Image is private + the `--login` flag's GHCR token is stale | Make image public OR refresh `gh auth token` and re-launch. |
| feature-backfill worker panics `as_string::<i32>()` / `"string array"` | omni sidecar's `encoded_filename` column inferred as Null type (no encoded variants ever saved for this chunk). | The Rust worker now skips these gracefully (`fix(feature-backfill)` 2026-05-19). To populate features for those chunks, **re-encode them** — see "Known constraint" section above. |
| feature-backfill SIGSEGV on older Xeon CPUs | Initially blamed on archmage SIMD dispatch; actually traced to the panic above leaking through tokio's task abort. Fixed 2026-05-19. | Use v26 image. |

---

## Emergency cleanup

```bash
# Show every instance currently on your account
vastai show instances-v1 --raw | python3 -c "
import json,sys
d=json.load(sys.stdin)
for i in d if isinstance(d,list) else d.get('instances',[]):
    print(f\"  id={i.get('id')} label={i.get('label','?')[:40]:40s} status={i.get('actual_status')} dph=\${i.get('dph_total',0):.4f}\")"

# Destroy them all (DANGEROUS — type the run-id label prefix explicitly)
vastai-fleet destroy --label-prefix <YOUR-RUN-ID>

# Or one by one
yes y | vastai destroy instance <ID>
```

If the dashboard says nothing's running but your credit keeps dropping,
check `vastai show invoices --raw` for bandwidth charges still being
calculated from earlier-today instance work — those land at day boundary.

---

## Money-saving knobs

- **`SKIP_CLAIMS=1`** — bypass claim check (single-instance smoke only;
  unsafe for fleet fanout).
- **`PARALLEL=0`** — let the onstart auto-detect cores. Better than
  capping unless you've measured.
- **`--max-dph 0.07`** — hard cap on the `--max-dph` flag prevents
  picking an expensive box if cheap ones are scarce.
- **`--min-gpu-ram-mb 8000`** — RTX 3060 / 4060 territory; cvvdp + ssim2
  + butteraugli all fit at 1024². Don't pay for 24 GB unless you've
  measured a real OOM.
- **`reliability>0.99`** — already in the query; lower at your peril.

## Don't

- Don't launch from `launch_backfill.sh` without a single-instance smoke
  first.
- Don't fan out to N>5 boxes without watching the first one produce
  sidecars at the rate you expected.
- Don't put credentials in shell history. The launcher reads them from
  `~/.config/cloudflare/r2-credentials` + `gh auth token`.
- Don't `git push` `vastai-fleet` or `zen-metrics` binaries; they're in
  `.gitignore` (each ~3-280 MB).
- Don't trust the vast.ai web UI's "destroyed" status as final — some
  destroys take 30-60 seconds to register. Re-check.
