# vast.ai metric backfill — operator guide

**This directory contains everything needed to score a corpus of (source,
distorted) image pairs against one or more GPU metrics on rented vast.ai
boxes.** If you've never run this before, READ THE WHOLE FILE before
typing any `vastai` command. It can spend real money fast.

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
| `Dockerfile.sweep.v15` (extends v14) | **`v17` (current)** | ✅ shipping | v14 base + vastai-fleet + run_with_error_trap + cuda_dlsym_stub LD_PRELOAD + omni worker + cvvdp worker. **Use this.** |
| `Dockerfile.sweep.v14` | `v14-omni` | active | Base image for v15-v17. Bakes zen-metrics + cuda-cudart-dev-12-6. Rebuilt when zen-metrics binary changes. |
| `Dockerfile.sweep.v13` | `v13` | deprecated | Single-stage from-source build. Slow rebuild; superseded by v14's precompiled-binary path. |
| `Dockerfile.sweep` (root) | — | deprecated | Vestigial pre-v13 prototype. Slated for deletion (P5d). |
| `scripts/sweep/Dockerfile.sweep` | — | deprecated | Same. |
| `scripts/sweep/Dockerfile.pycvvdp` | `pycvvdp` | active (rare) | Only used by the dual-impl cvvdp parity flow. |

### Onstart scripts (entrypoint for each container)

| File | Used by | Status |
|---|---|---|
| `onstart_omni_backfill.sh` | **omni (all 6 GPU metrics + encoded variants)** | ✅ current |
| `onstart_cvvdp_backfill_imazen.sh` | cvvdp single-impl backfill | active |
| `onstart_cvvdp_backfill.sh` | cvvdp dual-impl (cvvdp-gpu + pycvvdp) | active (rare) |
| `onstart_iwssim_backfill_v14.sh` | iwssim backfill (v14 image baseline) | active |
| `onstart_iwssim_backfill.sh` | iwssim backfill (legacy v3 image) | deprecated |
| `onstart_v2.sh`, `onstart_v3.sh` | pre-v14 legacy generic worker | deprecated |

### Chunk workers (process one chunk = 100-200 rows)

| File | Used by | Status |
|---|---|---|
| `omni_backfill_chunk_worker.sh` | omni onstart | ✅ current |
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
| `sweep_janitor.py` | Sidecar consolidation + dedup |
| `fleet_status.sh` | One-shot dashboard wrapping `vastai-fleet status` |
| `finalize.sh` | Post-sweep R2-sidecar consolidation into per-codec parquets |
| `vast_cost_watch.sh` (new — this PR) | Continuous burn-rate monitor; alerts if total cost exceeds threshold |

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
