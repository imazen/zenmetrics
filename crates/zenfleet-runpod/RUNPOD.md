# zenfleet-runpod — RunPod provider

The RunPod backend for the `zenfleet-cloud` worker trait layer
(spec §1.10, Phase F). RunPod rents GPU pods — commodity COMMUNITY-cloud
GPUs or datacenter SECURE-cloud GPUs — another vast.ai / SaladCloud
alternative for the same sweep workload class.

RunPod offers two product modes. This crate implements the **Pods
(pull)** path, which is the shippable Phase F; **Serverless (push)** is a
documented follow-on (see the last section), not implemented here.

## Architecture: Pods are PULL, identical to vast.ai

A RunPod **pod** is a rented GPU box that boots a generic container with
your image. Credentials + sweep wiring arrive as ordinary container
environment variables (from the pod-create request's `env` map), and the
worker *pulls* its work — exactly like vast.ai. There is no managed
queue and no push sidecar (that is Salad's model, and RunPod
serverless's). So the RunPod `JobQueue` is the **same R2-ETag atomic
token-race claim** the vast.ai backend uses.

Rather than copy-paste a third claim implementation, `zenfleet-runpod`
reuses vast.ai's proven claim **algorithm** verbatim: `queue.rs` depends
on `zenfleet_vastai::worker::claim::try_claim` (+ `ClaimConfig` /
`ClaimOutcome`) and `zenfleet_vastai::worker::r2::R2Client`. The claim
algorithm lives in exactly one place (vast.ai's `claim.rs`); this crate
only owns the RunPod-specific URI conventions and the
`zenfleet_cloud::JobQueue` mapping. (The `worker` feature of
`zenfleet-vastai` is GPU-free — it pulls only the claim + R2 plumbing,
not the codec tree.)

| | vast.ai | RunPod (Pods) |
|---|---|---|
| product mode | rented GPU box | rented GPU pod |
| queue ownership | BYO (we use R2) | BYO (we use R2) |
| distribution model | pull (worker claims) | pull (worker claims) |
| `next_chunk()` impl | atomic R2 ETag claim | **same** atomic R2 ETag claim |
| object storage | Cloudflare R2 | BYO R2/S3 (RunPod has none) |
| worker identity | hostname / `/proc/1/environ` | `RUNPOD_POD_ID` (plain env) |
| credential source | vast env injection (pid-1) | plain pod env |

## Trait mapping

| Core trait | RunPod impl | How |
|---|---|---|
| `JobQueue` | `queue::RunpodChunkQueue` | Pull over an R2 `chunks.jsonl` manifest with the shared atomic claim (reuses `try_claim`). `next_chunk` pops + claims, skipping done/peer-held/lost/errored; `ack_chunk` is a no-op (the claim + sidecar are the durable state). `skip_claims` short-circuits the race for single-instance smoke runs. |
| `CredentialSource` | `host::RunpodEnvCredentials` | Plain `std::env::var` over the pod env: BYO `R2_*`/`AWS_*`/`S5CMD_*` + sweep wiring (`SWEEP_RUN_ID`, `CHUNKS_R2`, …) + reserved `RUNPOD_POD_ID` / `RUNPOD_POD_HOSTNAME` / `RUNPOD_GPU_COUNT`. No IMDS, no pid-1 trick — RunPod injects everything into the app process. |
| `WorkerHost` | `host::RunpodWorkerHost` | `RUNPOD_POD_ID` → `$WORKER_ID` → hostname for id; `$WORKDIR` → `/workspace`; `RUNPOD_GPU_COUNT` then `nvidia-smi` for GPU count. |
| `BlobStorage` | `storage::RunpodBlobStorage` | Re-export of the shared `zenfleet-s3` impl. RunPod has no native object store; workers BYO R2/S3. **No second S3 client** (spec §1.10). |
| `Heartbeat` | `heartbeat::R2Heartbeat` / `heartbeat::NoopHeartbeat` | RunPod tracks pod liveness via its own dashboard + `GET /pods/{podId}`, so the worker defaults to `NoopHeartbeat`. `R2Heartbeat` (reusing the shared `zenfleet-s3` `BlobStorage`) is available for cross-fleet monitoring parity with vast.ai. |

## Launcher provisioning (`launch` module)

Operator-workstation side — provisions a pod and pushes work (the worker
then PULLs chunks from R2). NOT baked into the deploy image, NOT on the
hot path. `launch::RunpodApi` (hand-rolled `reqwest` + `serde`; there is
no official Rust SDK — RunPod ships Python/Go/JS/GraphQL).

### Current API: REST (verified 2026-05-27 against the live OpenAPI)

RunPod migrated from GraphQL to a REST API; **REST is the go-forward
path** (the GraphQL API still exists, but the docs steer new integrations
to REST — confirmed via the RunPod blog "Streamline GPU Cloud Management
with RunPod's New REST API" and `docs.runpod.io/api-reference`).

- **Base URL:** `https://rest.runpod.io/v1`
- **Auth header:** `Authorization: Bearer <api-key>` (RunPod console →
  Settings → API Keys). Key resolved from `$RUNPOD_API_KEY` or
  `~/.config/runpod/credentials` (mirrors the R2 + Salad conventions).
- **Endpoints** (extracted from `GET https://rest.runpod.io/v1/openapi.json`):
  - `POST   /pods`              — create (rent) a pod.
  - `GET    /pods/{podId}`      — inspect a pod (status, cost, gpu).
  - `POST   /pods/{podId}/stop` — stop (pause) a pod.
  - `DELETE /pods/{podId}`      — terminate (delete) a pod.

`PodCreateInput` fields (verified, camelCase, RunPod defaults shown):
`imageName`, `gpuTypeIds` (array of GPU **display-name** ids, e.g.
`"NVIDIA GeForce RTX 4090"`), `gpuCount` (1), `cloudType`
(`"SECURE"`/`"COMMUNITY"` — `COMMUNITY` is the cheap commodity tier, the
vast.ai-equivalent), `containerDiskInGb` (50), `volumeInGb` (20),
`volumeMountPath` (`/workspace`), `env` (object), `dockerEntrypoint`,
`dockerStartCmd`, `containerRegistryAuthId`. The `Pod` response carries
`id`, `desiredStatus`, `costPerHr`, `gpu`, `machineId`, …

### GPU-type discovery

The REST v1 surface does **not** expose a `/gpu-types` list endpoint
(unlike Salad, whose REST API lists GPU classes). `gpuTypeIds` takes GPU
display-name id strings directly. Discover the exact available ids via
the RunPod console, or the legacy GraphQL `gpuTypes { id displayName
memoryInGb }` query at `https://api.runpod.io/graphql`. The launcher
therefore takes the GPU id string as-is — there is no `resolve_gpu_class`
round-trip the way Salad needs.

## Deploy image

Per the BAKE-EVERYTHING rule, a RunPod Pods deploy image is **the same
shape as the vast.ai image** — there is no extra sidecar to bake (unlike
Salad). It bakes:

1. `zenfleet-sweep` (built `--features runpod-sweep` — glue + the
   shared inline encode+score compute), plus `s5cmd`, the CUDA runtime,
   and the stable apt packages (layered cheap→expensive per the
   BAKE-EVERYTHING layering rule).
2. An entrypoint that runs `zenfleet-sweep worker --backend runpod`.
   The pull worker reads `SWEEP_RUN_ID` / `CHUNKS_R2` / the BYO `R2_*`
   creds from the pod env (set via the pod-create `env` map) and pulls
   chunks from R2.

No boot-time `apt`/`pip`/`cargo` — first boot is production. Because the
pull entrypoint reuses the vast.ai image shape, building that image is a
follow-on that can reuse the vast image's Dockerfile + onstart with only
the `--backend runpod` flag changed; it is **not built in this pass**.

## Testing

The `compute` closure is backend-agnostic and is covered by the worker's
own tests. The real atomic R2 token-race (`try_claim`) shells out to
`s5cmd` against a real bucket, so it is exercised by the operator's real
smoke run (see below), not off-node.

`tests/queue_roundtrip.rs` exercises the `JobQueue` contract the generic
`run_worker` loop depends on, off any RunPod node, via the real
`skip_claims` code path: a multi-chunk manifest drains in order through
`next_chunk`, each chunk surfaces its raw manifest line as the payload,
each is `ack`'d (no-op), then the queue yields `None`. Plus the
embedded unit tests cover URI substitution, manifest parsing,
claim-outcome→chunk mapping (with a fake claim fn for every
`ClaimOutcome` variant), credential collection, GPU-count probing, and
the launcher's request/response (de)serialization.

**A real RunPod smoke run is the operator's gate** — it needs a RunPod
account + GPU spend: rent a pod via `RunpodApi::create_pod`, point it at
a `chunks.jsonl` manifest in R2, confirm the worker claims + processes a
chunk and uploads the sidecar, then `terminate_pod`.

## Serverless (push) — documented follow-on, NOT implemented

RunPod's **Serverless** product is the push analogue (closer to Salad's
model): you deploy a *serverless endpoint* with a container image, RunPod
auto-scales worker replicas, and routes each request's `input` to your
worker. It is investigated here for a future `zenfleet-runpod`
serverless `JobQueue`, but deliberately **not implemented in Phase F**
(the Pods/pull path is the clean, zero-protocol-risk shippable).

Findings (docs.runpod.io/serverless, runpod/runpod-python,
runpod-workers/worker-template, verified 2026-05-27):

- **The worker→queue contract is undocumented.** RunPod publishes the
  *handler* contract (`handler(event)` where `event["input"]` is the job
  payload) and the Python SDK entrypoint `runpod.serverless.start({
  "handler": handler })`, but **not** the internal protocol the SDK
  speaks to fetch jobs from RunPod's queue and return results. The SDK
  encapsulates the poll/ack loop; the worker API it talks to is not a
  published, stable contract.
- **A Rust binary cannot cleanly speak the serverless contract today.**
  Because the internal worker API is undocumented and SDK-private, a
  native Rust serverless `JobQueue` would have to either (a) reverse-
  engineer the SDK's poll/ack protocol (brittle — RunPod can change it
  under the SDK at will), or (b) ship a thin **handler shim**: a tiny
  Python `rp_handler.py` that calls `runpod.serverless.start` and, per
  job, forwards `event["input"]` to the co-located `zenfleet-sweep`
  over local IPC (HTTP/stdin) — structurally the same bridge pattern
  Salad uses, but with the Python SDK as the queue↔app bridge instead of
  Salad's Go HTTP sidecar.
- **Recommended future shape (shim, not reverse-engineer).** Mirror the
  Salad design: a `serverless::RunpodServerlessJobQueue` that runs a
  tiny local HTTP receiver; bake a `rp_handler.py` into the image that
  `runpod.serverless.start`s and POSTs each `event["input"]` to the
  receiver, returning the HTTP response body as the handler result. The
  `compute` closure stays identical. This reuses Salad's proven
  receiver/`ack`→HTTP-status mapping (`queue.rs` there) and avoids
  betting on an undocumented internal protocol. Build it only if RunPod
  serverless's autoscaling-per-request economics beat Pods for a given
  sweep — Pods (steady rented GPUs for a multi-hour batch sweep) is the
  natural fit for our workload, so serverless is genuinely optional.
- **Local testing** for the serverless path would use the SDK's
  `--rp_serve_api` flag (a local FastAPI shim emulating the endpoint),
  but the shim-vs-native decision above is the load-bearing one.
