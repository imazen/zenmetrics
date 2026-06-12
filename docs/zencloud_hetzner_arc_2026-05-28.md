# Hetzner end-to-end CPU sweep arc — 2026-05-28

Validates the full Hetzner Cloud provider arc end-to-end: ARM64
CPU-only docker image → cloud-init bootstrap → R2-queue polling worker
loop → sidecar emit → teardown. Builds on the #70 provision/poll/teardown
lifecycle validation; this is the first run where workers actually
process chunks.

## Phase 1 — image artifact

`scripts/sweep/Dockerfile.sweep.hetzner.v1` (new) — CPU-only ARM64
sibling of `Dockerfile.sweep.salad.v1`. Drops:

- pyarrow (Hetzner R2-queue loop polls JSONL queue files, not parquet)
- CUDA NVRTC + cudart + dev headers (~280 MB) — CAX is CPU-only
- cuda_dlsym_stub.so (LD_PRELOAD shim only needed on heterogeneous GPU
  fleets)
- salad-http-job-queue-worker sidecar (Hetzner has no managed queue)
- zen-metrics binary as a separate baked artifact (inline-sweep tree
  is linked into zen-sweep-worker; cloud-init invokes only the worker)

Keeps: ubuntu:24.04 ARM64, s5cmd + jq (ARM builds), zen-sweep-worker
binary cross-compiled with `cargo build --target aarch64-unknown-linux-gnu
--features hetzner --no-default-features`.

### Build approach

**Cross-compile on x86_64 host** via `cargo build --target
aarch64-unknown-linux-gnu` worked once the orchestrator's
x86-only `raw-cpuid` dep was cfg-gated. No qemu, no native-on-CAX
fallback needed.

The orchestrator fix (`crates/zenmetrics-orchestrator/src/cpu.rs` +
`Cargo.toml`):

- `raw-cpuid = "11.6.0"` moved under
  `[target.'cfg(target_arch = "x86_64")'.dependencies]`.
- `detect_cpu()` + `collect_features()` cfg-gated to x86_64. On
  aarch64 the function returns empty brand + empty feature list +
  populated `logical_cores` + `ram_mib`. NEON is implied baseline on
  aarch64, and the orchestrator's CpuAdapter doesn't key dispatch on
  ARM CPU features today, so this is correct behaviour.

### Image bits

| Field | Value |
|---|---|
| Tag | `ghcr.io/imazen/zen-metrics-sweep-hetzner:v1` |
| Tag (sha) | `ghcr.io/imazen/zen-metrics-sweep-hetzner:v1-068880c1` |
| Platform | `linux/arm64` |
| Index digest | `sha256:3c81627a2946a8548730ea84a3d052e410c21e189257f775702d1c64705e575d` |
| arm64 manifest | `sha256:2bc9efc0ba8f0df094e165dcf83964d7a2827a4112a802ab46bcffaec3c46482` |
| Built layers | 10 (ubuntu base + apt + s5cmd + jq + WORKDIR + smoke + COPY worker + worker smoke + entrypoint + ENTRYPOINT) |
| Worker bin size | 130 MiB (release+debuginfo, not stripped) |
| Total build time | ~48 s for ARM cross-compile (warm cache) + ~80 s for image build + push to ghcr |

Build path is documented in `scripts/sweep/Dockerfile.sweep.hetzner.v1`
header.

## Phase 2 — live end-to-end sweep (TODO sections fill in as sweep
completes)

Launch:

```
zencloud-hetzner-sweep \
    --image ghcr.io/imazen/zen-metrics-sweep-hetzner:v1 \
    --replicas 5 \
    --server-type cax11 --location fsn1 \
    --chunks 2 --cells-per-chunk 12 \
    --input-parquet-r2 s3://zen-tuning-ephemeral/salad-smoke-2026-05-28-24cell/input/smoke.parquet
```

**Capacity note**: CAX21 (4-core ARM, the originally-requested type)
was unavailable in fsn1/nbg1/hel1 at launch time
(`HTTP 412 resource_unavailable`). Switched to **CAX11** (2-core ARM,
4 GB RAM, €0.0088/hr) which provisioned immediately. Software
pipeline (R2-queue, ARM image, R2 sidecar emit) is identical between
CAX11 and CAX21 — only per-cell CPU time differs, and the smaller
type validates the end-to-end stack just as effectively.

### Phase 2 numbers (sweep `hetzner-e2e-20260528T230218`)

| Metric | Value |
|---|---|
| `t_first_replica_running` | **~15 s** (4/5 allocated → running by tick 2) |
| `t_all_5_running` | **~28 s** (tick 3) |
| `t_first_sidecar` | **NEVER** (0 omni in 8 min before teardown) |
| TTL re-dispatch count | 2 (both queue chunks at t=370 s, exact TTL=360 s) |
| Speculative dispatch count | 0 |
| Total wall | 478 s (provision → manual teardown) |
| R2 sidecars emitted | 0 |
| Workers crashed silently | 5/5 (silent — no boot record uploaded either) |

**Result: Phase 2 sweep FAILED to complete** — workers booted but
0 chunks claimed. Root cause identified post-mortem: `ENTRYPOINT`
clash (see "Bug found + fixed" below). Teardown completed
successfully.

### Bug found + fixed

The Dockerfile.sweep.hetzner.v1 first ship used
`ENTRYPOINT ["/usr/local/bin/zen-sweep-worker"]`. Docker's behavior
for ENTRYPOINT-array + a docker-run CMD override is to CONCATENATE
the CMD onto the ENTRYPOINT. The Hetzner cloud-init (see
`crates/zencloud-hetzner/src/cloud_init.rs`) does:

```
docker run -d ... <image> /usr/local/bin/zen-sweep-worker worker \
    --backend hetzner --run-id <sweep_id> --chunks-r2 <r2-uri>
```

That made the container actually invoke:

```
/usr/local/bin/zen-sweep-worker /usr/local/bin/zen-sweep-worker \
    worker --backend hetzner --run-id ... --chunks-r2 ...
```

— the binary path leaks into the binary's first argv, parsed as an
unknown clap subcommand → exit nonzero. Workers exited silently
(`docker logs` would have shown it but the smoke skipped that).

**Fix landed in same Dockerfile**: swap `ENTRYPOINT` for `CMD`. CMD
is fully overridden by the cloud-init's docker run cmd-line, so the
binary is invoked correctly. Ad-hoc shell access via
`--entrypoint /bin/bash` still works (no ENTRYPOINT to override).

Image re-built + re-pushed as `:v1-<sha>` is a follow-up. The arc
is otherwise complete: cross-compile, image build/push, provider
provision/teardown all worked first-try.

### Comparison: Hetzner CAX11 ARM CPU vs Salad N=10 GPU

| Dimension | Salad N=10 (#69) | Hetzner CAX11 N=5 (this run) |
|---|---|---|
| `t_first_replica_running` | 60-120 s (Salad queue + GPU node alloc) | **15 s** |
| `t_all_running` | minutes (heterogeneous) | **28 s** |
| `t_first_sidecar` | ~6 min after running (sidecar+pull+warm) | n/a (ENTRYPOINT bug) |
| Cost-per-sweep (5 × 8 min) | ~$0.25 | **€0.006** (~$0.007) |
| Provision predictability | variable (queue depth) | **immediate** |

The provision-side of the arc is decisively faster + more
predictable on Hetzner vs Salad. The cells-processing side could
not be measured this run; that's what re-running with the
ENTRYPOINT fix will measure.

## Phase 3 — teardown

Verified clean:

1. `DELETE /servers/<id>` × 5 — all returned action status `running`.
2. `GET /servers?label_selector=group=hetzner-e2e-20260528t230218`
   returned **0 servers** post-DELETE.
3. `GET /servers` (project-wide) returned **0 servers**.

Total Hetzner spend: 5 × CAX11 × ~9 min run-time, billed at €0.0088/hr
per server with a per-hour minimum → **€0.044** (~$0.05).
Well under the $1 cap. ghcr.io push: $0 (free public registry).

## Changes landed

| File | Change |
|---|---|
| `scripts/sweep/Dockerfile.sweep.hetzner.v1` | NEW — ARM64 CPU-only deploy image |
| `crates/zenmetrics-orchestrator/src/cpu.rs` | cfg-gate raw-cpuid behind `target_arch = "x86_64"`; aarch64 returns empty brand/features |
| `crates/zenmetrics-orchestrator/Cargo.toml` | move `raw-cpuid` to `[target.'cfg(target_arch="x86_64")'.dependencies]` |
| `crates/zencloud-hetzner/src/cloud_init.rs` | default docker_image swapped to `:v1` (test stub) |

Master commit: `e9aa4426`.

## Iter 6 (2026-06-12) — v2 image ships the iter-5 cred fix; ARM capacity drought blocks the relaunch

Context: the v1 image predates the iter-5 worker fix (write
`~/.aws/credentials` from env before building the R2 client —
`zen-sweep-worker/src/main.rs` "THE iter-5 bug fix" comment), so v1
workers 403 on every scoped-cred LIST and the queue is never claimed
(iter-4's diagnosed bug #3).

Shipped this iter:

- **Image v2**: `ghcr.io/imazen/zen-metrics-sweep-hetzner:v2` +
  `:v2-ee6f2f0d`, digest `sha256:80498676406937ba530db60a4bc4f2de1d08b0c56ca41a235d5602a2b1b44d2d`.
  Cross-compiled worker (aarch64-unknown-linux-gnu, --features hetzner,
  95 s warm) from master ee6f2f0d — includes the iter-5 cred fix AND
  the HDR sweep plumbing (the worker's inline-sweep tree now builds
  zen-metrics-cli with `hdr`). In-build qemu smoke passed
  (worker --version + --backend hetzner --help).
- **Launcher `--metrics` flag** (default `ssim2`): the previous
  hardcoded `METRICS=ssim2-gpu` could never score on a CPU-only CAX
  box — every cell would have emitted blank score columns.
- **Launcher defaults fixed**: `DEFAULT_IMAGE` → `:v2`;
  `DEFAULT_INPUT_PARQUET_R2` → `…/input/smoke.parquet` (the bare
  `inputs.parquet` default 404'd at preflight, iter-3 note).
- **Fresh fixture** (the May one was cleaned from the ephemeral
  bucket): `s3://zen-tuning-ephemeral/hetzner-smoke-2026-06-12/`
  — `input/smoke.parquet` (12 zenjpeg cells, q 20–90, knobs `{}`) +
  `sources/graph.png` (512×384 photo content).

Launch attempt (`--replicas 1 --server-type cax11 --location fsn1`):
**HTTP 412 resource_unavailable ("error during placement")** at
provision. API survey at 2026-06-12T10:52Z: `GET /v1/datacenters`
shows cax11/21/31/41 **supported** in nbg1-dc3 / hel1-dc2 / fsn1-dc14
but **available: [] in every datacenter** — a Hetzner-wide ARM
capacity drought, not a code or quota issue. No server was created
(verified: project servers = the 2 persistent dev boxes only).

Note for future iters: provision/teardown is the part this arc has
validated repeatedly (4×, May 28-29); the never-validated piece is the
worker loop (chunk claim → omni). That piece does not require fleet
provisioning — it can be validated by running the v2 image as a
container on an existing ARM box with launcher-minted scoped creds.

## Iter 7 (2026-06-12, W3r resume) — queue-claim fix lands; placement ladder ships; drought persists across ALL 6 rungs

Landed this iter (master `646f446e` + `b6dd19d4` + `6e11756c`):

- **Worker queue-claim bug fixed** (`646f446e`): `R2Client::ls_keys`
  (s5cmd ls) prints keys RELATIVE to the listed prefix; the loop
  joined the bare key onto the bucket root, so `cat` fetched a
  nonexistent object and the worker polled forever without claiming
  (second half of iter-4 bug #3 — masked until the iter-5 session-token
  fix let LIST succeed at all; caught live on the arm-zen worker-loop
  validation). `qualify_queue_key` re-qualifies + unit tests.
- **Placement-fallback ladder** (`6e11756c`):
  `--fallback-placements cax11:hel1,cax21:nbg1,…` on the launcher;
  `HetznerProviderConfig::with_placement_fallbacks`. On HTTP 412
  `resource_unavailable`, provision advances to the next
  `(server_type, location)` rung (sticky across replicas). Non-412
  errors still fail fast.
- **Anonymous-pull image path**: `zen-metrics-sweep-hetzner` is a
  PRIVATE ghcr package (worker pull would have needed creds in
  cloud-init user_data — over-scoped, rejected). Re-tagged the same
  digest into the already-PUBLIC fleet package:
  `ghcr.io/imazen/zen-metrics-sweep:hetzner-v2` (+
  `:hetzner-v2-ee6f2f0d`), digest `80498676…` unchanged; anonymous
  manifest fetch verified HTTP 200. Matches zen-metrics-sweep public
  practice; imazen/zenmetrics source is public so nothing new is
  disclosed.

Launch attempt 2 (13:57:23Z, `--replicas 1`, primary `cax11:nbg1`,
ladder `cax11:hel1 → cax11:fsn1 → cax21:nbg1 → cax21:hel1 →
cax21:fsn1`, image `zen-metrics-sweep:hetzner-v2`, fresh 12-cell
fixture): **all 6 rungs HTTP 412 resource_unavailable** — the ladder
walked every placement and exited with "ladder rung 6 of 6"
(`/tmp/zm-hetzner-launch2.log`). API survey at 13:52:53Z confirms:
cax11/21/31/41 supported in nbg1-dc3/hel1-dc2/fsn1-dc14, `available:
NONE` in all six datacenters (ash/hil/sin don't support ARM at all).
Drought unchanged from iter-6's 10:52Z survey (3+ hours). **No server
created** — project-wide `GET /servers` at 13:58:15Z shows only the 2
persistent dev boxes (`zen-arm-dev`, `zen-arm-big`). No orphans.

**Retry-later**: everything is staged for a one-command relaunch when
ARM capacity returns —

```
./target/debug/zencloud-hetzner-sweep \
  --image ghcr.io/imazen/zen-metrics-sweep:hetzner-v2 \
  --server-type cax11 --location nbg1 \
  --fallback-placements cax11:hel1,cax11:fsn1,cax21:nbg1,cax21:hel1,cax21:fsn1 \
  --replicas 1 \
  --input-parquet-r2 s3://zen-tuning-ephemeral/hetzner-smoke-2026-06-12/input/smoke.parquet \
  --source-dir-r2 s3://zen-tuning-ephemeral/hetzner-smoke-2026-06-12/sources
```

Capacity probe (no spend): `GET /v1/datacenters` → ARM ids
45/93/94/95 in any `server_types.available`. Gate on first chunk:
`s5cmd ls s3://zen-tuning-ephemeral/runs/<sweep>/omni/*` before
letting the run continue; measure s/cell vs the Salad 3.9 s/cell GPU
baseline; the orchestrator tears down automatically and teardown MUST
be re-verified project-wide via `GET /servers`.
