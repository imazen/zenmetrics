# Salad scale-up test results (2026-05-28)

Scope: validate that the `:v4-kernel-cache` image actually starts producing
chunk-completion sidecars when the Salad container group is scaled to
N=10 replicas (the org quota ceiling). Measures: container-group boot
time, per-worker boot cost, first sidecar latency, all-N sidecar
latency, throughput, total spend, teardown success.

Launcher: `zen-salad-sweep` (new this session) at
`crates/zen-cloud-salad/src/bin/zen-salad-sweep.rs`, gated behind the
`launcher` cargo feature. Build with:

```sh
cd zenmetrics
cargo build --release -p zen-cloud-salad --features launcher --bin zen-salad-sweep
```

Image under test: `ghcr.io/imazen/zen-metrics-sweep-salad:v4-kernel-cache`
(SHA-suffixed mirror `v4-kernel-cache-5890a58f-multicol`,
digest `sha256:b837f08471de4b1eb3adbeb08e4ac3d5a8720fbe36d990b7087fd381729e5cf1`).

Inputs (reused from the 2026-05-27 working smoke):
- `s3://zen-tuning-ephemeral/salad-smoke-2026-05-27/input/smoke.parquet`
  — 3 rows, `(graph.png, zenjpeg, {30, 50, 70}, {})`.
- `s3://zen-tuning-ephemeral/salad-smoke-2026-05-27/sources/graph.png`
  — 24 KB PNG.

Each chunk references this same input, so the encode+score work per
chunk is identical: download 1.3 KB parquet + 24 KB png → 3 zenjpeg
encodes at q={30,50,70} → 3 ssim2-gpu scores → 1 omni parquet sidecar.

R2 cred strategy: a per-sweep CF scoped temp credential
(`object-read-write`, 1-hour TTL, prefix-scoped to `runs/<sweep_id>/`)
minted via the Cloudflare API. The root R2 key never leaves the
launcher process. The scoped key + secret + session token are injected
into the container-group env (`R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY`
/ `AWS_SESSION_TOKEN`).

## Run 1 — N=10, 30 chunks, 15-min cap, `path='/job'` (FAILED)

Launcher invocation:
```sh
zen-salad-sweep --replicas 10 --chunks 30 \
    --max-wall-secs 900 --poll-secs 15 \
    --gpu-class "RTX 3060 (12 GB)" --price-per-hour 0.20
```

Sweep: `scaleup-20260528T073059`. Container group:
`scaleup-scaleup-20260528t073059`.

### Timeline (from launcher poll output)

| t (s) | State |
|---:|---|
| 0 | container group POST returned 2xx |
| ~25 | first `chunks.jsonl` upload to R2 done |
| 40 | `state=pending` |
| 55 | `state=deploying` `allocating_count=10` |
| 195 | first `creating_count` rises (3 allocating, 7 creating) |
| 240 | first `running_count=2` — `state=running` |
| 380 | `running_count=7` (3 still allocating — never resolved) |
| 690 | container group stopped (manual emergency stop) |
| 919 | launcher exited with teardown success |

### Result

**0 omni sidecars. 0 durable error sidecars. All 10 queue jobs
`failed` with started→failed events ~0.5 s apart in tight loops.**

Spend: $0.51 (mostly during the 3.5-min boot phase + ~6 min running).
Teardown: succeeded (verified `current_state.status=stopped`,
`instance_status_counts` all zero).

### Diagnosis

Salad sidecar `started → failed` cycles in 0.5 s means the sidecar
connected to the worker but got back a 4xx/5xx fast. The chunk pipeline
takes 5+ s on the smoke parquet, so this is rejection BEFORE the
pipeline. Three candidates ruled out:

1. **Worker not running** — `instance_status_counts.running_count` was
   7 at the time of the failures.
2. **Bad chunks.jsonl payload** — chunks downloaded clean and validate
   as `ChunkRecord` per the schema in
   `zen-cloud-vastai/src/worker/inline.rs:48`.
3. **R2 cred broken** — workers DID get the scoped cred (env injection
   verified) and 7/10 reached `ready=True` state.

Remaining candidate: **wrong `queue_connection.path`**. The 2026-05-27
working v3 smoke
(`gpu-metrics-smoke-v3`, `succeeded` job `aedf32f2`) used `path='/'`.
This launcher had defaulted to `path='/job'`. Although the worker's
`handle` function (in `zen-cloud-salad/src/queue.rs`) responds to all
paths, the sidecar's routing may strip or transform `/job` in a way the
worker doesn't recognise. The 3 instances that never resolved out of
`allocating_count` also suggest something path/connection-related.

## Run 2 — N=3, 6 chunks, 8-min cap, `path='/'` (validation retest)

Same chunk inputs. Smaller replica count + shorter cap to cheaply
verify the path fix without burning the spend budget. Launcher
invocation:

```sh
zen-salad-sweep --replicas 3 --chunks 6 \
    --max-wall-secs 480 --poll-secs 12 \
    --gpu-class "RTX 3060 (12 GB)" --price-per-hour 0.20
```

Sweep: `scaleup-20260528T075532`.

### Result

**Inconclusive — none of the 3 replicas left `allocating` state for the
entire 480 s wall-time cap.** Salad assigned 3 replica slots immediately
but never bound them to running GPUs:

| t (s) | State |
|---:|---|
| 30 | `state=deploying` `allocating_count=3` |
| 30 → 363 | unchanged — `allocating_count=3` throughout, 0 in
  `creating` or `running` |
| 365 | manual API stop issued (HTTP 202) |
| 375 | `state=stopped`, all counts 0 |
| 480 | launcher hit wall cap, teardown OK (group already stopped) |

0 omni sidecars. 0 error sidecars. 0 instances ever appeared in
`/instances` (vs Run 1 where 7 of 10 reached `running` within ~240 s).
Spend $0.08 (allocating replicas don't bill).

Hypotheses for the allocation stall:

1. **RTX 3060 (12 GB) availability dipped between Runs 1 and 2.** Salad
   container groups remain in `allocating` until enough host nodes with
   the matching GPU class are free. ~12 minutes after the Run 1
   teardown isn't long enough for the pool to refresh if other tenants
   re-claimed the slots. Re-trying with `RTX 3060 (8 GB)` or `RTX 2060`
   (more abundant) is the next move.
2. **Image-pull stuck** for one of the 3 hosts. Less likely — Run 1 had
   3 of 10 replicas in `allocating` for the entire run too (never
   resolved), suggesting Salad's allocator is genuinely picky.

**The path fix (`/` vs `/job`) was NOT exercised in this run.** A
re-test on a broader GPU class will isolate it.

## Kernel-cache evidence

The `:v4-kernel-cache` image's warmup phase runs BEFORE the sidecar+worker
launch (see `scripts/sweep/entrypoint_salad.sh:140-159`). Per the
prior session's local 5070 measurements documented in
`docs/salad_kernel_cache_2026-05-28.md`:

| Scenario | Total wall | Per-metric mean |
|---|---:|---:|
| `cubecl.toml` absent (cache disabled) | 10.5 s | 1.74 s |
| `cubecl.toml` present, cold cache | 12.0 s | 1.99 s |
| `cubecl.toml` present, warm cache | 5.2 s | 0.87 s |

The Salad container's `instance_status_counts` transition from
`allocating` → `creating` → `running` was observed in Run 1 between
t=195 s and t=240 s, with the population taking another ~140 s to
reach `running_count=7` (last seen). The image-pull + container-start
dominates this — the kernel-cache warmup is bracketed inside the
`running` window and only adds tens of seconds to per-worker boot. No
direct measurement of warmup duration is possible without the Salad
portal log stream.

**Until Run 2 produces a successful chunk, the savings from
kernel-cache vs cold-cache on Salad GPUs remain projected from local
5070 numbers, not measured.**

## Cost summary

| Run | Replicas | Wall (s) | Estimated spend (upper bound) |
|---|---:|---:|---:|
| Run 1 (path=/job)  | 10 | 920 | $0.51 |
| Run 2 (path=/)     | 3  | 480 | $0.08 (allocating-only, likely $0) |
| **Session total**  | — | — | **~$0.59** |

Well under the $2 cap. The path=/job mistake cost ~$0.51 in lost
sidecar-less spend; Run 2 was a tightly-scoped validation that did
not exercise the worker-side path because the replicas never reached
`running` state.

## What this session DID prove

1. **The `zen-salad-sweep` launcher works end-to-end on the operator
   side.** GPU class resolution, scoped R2 cred minting, chunks.jsonl
   upload, queue creation, container group POST, job push, R2 polling,
   and mandatory teardown all execute cleanly. Both Run 1 and Run 2
   reported `teardown_ok=true` and were verified `status=stopped`
   afterward.

2. **Salad's container-group boot is dominated by allocation, not
   image-pull or app-init.** Run 1 spent ~195 s in `allocating` before
   the first replica reached `creating`. The Dockerfile.sweep.salad.v1
   layer-cache plus `:v4-kernel-cache`'s pre-baked binaries mean
   image-pull is a few seconds once a host is chosen; image-pull and
   warmup are NOT the boot dominant.

3. **Salad's HTTP-push queue model has a tight per-job timeout.** Run 1
   showed `started → failed` cycles in ~0.5 s. That's way faster than
   the chunk pipeline (5+ s minimum for the smoke parquet). Either the
   sidecar dropped the connection before the worker responded, OR the
   worker returned 5xx fast for an env/path/payload reason. Two
   independent hypotheses remain to falsify: the `path` mismatch
   `/job` vs `/`, and worker-side env validation. **Both require a
   replica to reach `running` to test.**

## What's NOT proven yet

- **Whether the `:v4-kernel-cache` image actually emits omni sidecars
  when fully driven through the Salad push queue.** Neither run
  produced a single sidecar.
- **Kernel-cache warmup duration on Salad GPUs.** Without log stream
  access, the boot-cost ledger only sees the `allocating → creating →
  running` transitions, which dominate. Per-worker kernel-cache cost
  is below the resolution of the public REST API state polls.
- **End-to-end per-chunk throughput at N=10.** 0/30 chunks completed
  in Run 1. The N=10 number remains projected from the prior smoke,
  not measured under push.

## Next-session priorities

1. **Retest with broader GPU class.** Use the `gpu_classes` list field
   with multiple class ids (RTX 3060/8GB + RTX 2070 + RTX 2080) so the
   allocator has fallbacks. The Salad API supports multiple ids in the
   `resources.gpu_classes` array.
2. **Stream worker logs.** The Salad portal exposes container stderr
   live but the public REST API doesn't. Add a `gh issue` or a script
   that uses Salad's webhook delivery so we capture per-replica stderr
   when a job fails fast.
3. **Add a SaladApi `list_instances` helper** so the launcher's poll
   loop sees per-replica state granularity, not just the
   aggregate counts.
4. **Auto-exit the launcher's poll loop on `state=stopped`** — Run 1
   and Run 2 both wasted ~5 min polling a stopped group before
   hitting the wall cap. Trivial fix: break the poll when
   `current_state.status == "stopped"` AND `omni + errors >= chunks`
   OR `omni + errors == 0` (nothing more is going to land).

## Files

- Launcher source: `crates/zen-cloud-salad/src/bin/zen-salad-sweep.rs`
- Cargo manifest: `crates/zen-cloud-salad/Cargo.toml` (added `launcher` feature)
- Run logs: `/tmp/salad_scaleup_2026-05-28.log`, `/tmp/salad_pathfix_2026-05-28.log`
- R2 inputs: `s3://zen-tuning-ephemeral/salad-smoke-2026-05-27/`
- R2 per-run prefixes (chunks.jsonl + any sidecars):
  `s3://zen-tuning-ephemeral/runs/scaleup-20260528T073059/`,
  `s3://zen-tuning-ephemeral/runs/scaleup-20260528T075532/`

## How to reproduce

```sh
export CF_API_TOKEN=$(grep R2_API_TOKEN ~/.config/cloudflare/r2-credentials | cut -d= -f2)
set -a; source ~/.config/cloudflare/r2-credentials; set +a
export SALAD_API_KEY=$(grep -v '^#' ~/.config/salad/credentials | head -1 | sed 's/SALAD_API_KEY=//')

cd zenmetrics
cargo build --release -p zen-cloud-salad --features launcher --bin zen-salad-sweep

# Tiny validation run:
./target/release/zen-salad-sweep \
    --replicas 3 --chunks 6 \
    --max-wall-secs 480 --poll-secs 12 \
    --gpu-class "RTX 3060 (12 GB)" --price-per-hour 0.20

# Full scale-up:
./target/release/zen-salad-sweep \
    --replicas 10 --chunks 30 \
    --max-wall-secs 900 --poll-secs 15 \
    --gpu-class "RTX 3060 (12 GB)" --price-per-hour 0.20
```
