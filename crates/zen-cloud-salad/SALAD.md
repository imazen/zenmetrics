# zen-cloud-salad — SaladCloud provider

The SaladCloud backend for the `zen-cloud-core` worker trait layer
(spec §1.9). SaladCloud runs containers on distributed consumer GPUs at
low $/hr — a commodity-GPU alternative to vast.ai for the same workload
class.

## Architecture: why the app speaks HTTP, not gRPC

Spec §1.9 hedged the worker↔sidecar transport as "a thin Rust gRPC
client with `tonic`, OR run the sidecar as the gRPC *server* per the
sample's pattern." Reading the upstream sidecar
(`github.com/SaladTechnologies/salad-cloud-job-queue-worker`) settles
it: the **only shipped sidecar binary is `salad-http-job-queue-worker`**,
and it forwards each queue job to the app over **plain HTTP**, not gRPC.

The gRPC service in the sidecar's `service_v1alpha.proto`
(`JobQueueWorkerService { AcceptJobs, CompleteJob, RejectJob }`) is the
contract between the **sidecar and Salad's managed queue backend** — it
is internal to the sidecar (the sidecar is the gRPC client to Salad's
queue). The sidecar then forwards each received `Job { job_id, port,
path, input }` to the app as:

```
POST http://localhost:<job.port><job.path>
body = <job.input>
```

and reads the **HTTP response body** back as the job *output*, returning
it to the queue via `CompleteJob` (a non-2xx / connection error →
`RejectJob` → Salad re-queues). Source of truth: the sidecar's
`internal/workers/workers.go` (`http://localhost:%d%s` request build +
`Output: responseBody`) and the `with-shell-script` /
`with-s6-overlay` mandelbrot samples, which run a plain HTTP app
alongside the sidecar.

So `zen-cloud-salad` serves a **tiny local HTTP endpoint** rather than
embedding a tonic gRPC server. This matches the only sidecar Salad ships
and avoids a redundant gRPC layer the platform does not use. Spec §1.9
Phase 2 ("reimplement the sidecar protocol natively in Rust, dropping
the Go sidecar") remains future work — the bake-the-Go-binary path
ships faster and reuses Salad's supported bridge.

## Trait mapping

| Core trait | Salad impl | How |
|---|---|---|
| `JobQueue` | `queue::SaladJobQueue` | Local HTTP receiver. `next_chunk` blocks on the next sidecar POST; `ack_chunk` returns the HTTP response the sidecar reads back as the job output. `Done`/`Skipped` → 200; `Failed`/`Retryable` → 5xx (sidecar rejects → Salad re-queues). One job at a time, matching the single-flight `run_worker` loop. |
| `CredentialSource` | `host::SaladEnvCredentials` | Reads container-group env (BYO `R2_*`/`AWS_*`/`S5CMD_*` + reserved `SALAD_MACHINE_ID` / `SALAD_CONTAINER_GROUP_ID`). |
| `WorkerHost` | `host::SaladWorkerHost` | `SALAD_MACHINE_ID` → `$WORKER_ID` → hostname for id; `$WORKDIR` → `/workspace`; `nvidia-smi` for GPU count. |
| `BlobStorage` | `storage::SaladBlobStorage` | Re-export of the shared `zen-cloud-s3` impl. Salad has no native object store; workers BYO R2/S3. **No second S3 client** (spec §1.9 item 4). |
| `Heartbeat` | `heartbeat::SaladHeartbeat` | Log-only no-op. Salad manages instance liveness; the sidecar handles per-job acks (spec §1.9 item 5). |

`host::SaladImds` is a minimal IMDS client (`/v1/status`, `/v1/token`,
`Metadata: true` header, link-local `http://169.254.169.254:80`,
proxies disabled) for readiness + the workload identity token.

## Launcher provisioning (`launch` module)

Operator-workstation side — provisions Salad infra and pushes work. NOT
baked into the deploy image, NOT on the hot path. `launch::SaladApi`
(hand-rolled `reqwest` + `serde`; there is no Rust SDK) covers:

- `resolve_gpu_class(name)` — `GET .../gpu-classes`, resolve id by name.
- `create_queue(..)` — `POST .../queues`.
- `create_container_group(..)` — `POST .../containers` with
  `queue_connection { path, port, queue_name }` (the `port` MUST match
  the worker's `SaladQueueConfig` bind port), `gpu_classes`, replicas,
  image, env, optional registry auth, `restart_policy: always`,
  `autostart_policy: true`.
- `push_job(s)` — `POST .../queues/{name}/jobs` (`{ input, metadata }`);
  `input` is forwarded to the worker as the chunk payload.
- `get_container_group` / `stop_container_group` — monitor / scale.

Auth: `Salad-Api-Key: <key>` header. Key from `$SALAD_API_KEY` or
`~/.config/salad/credentials`. Request/response shapes track the Salad
OpenAPI as exposed by the official `salad-cloud-sdk-python` (the
field-name source of truth).

## Deploy image

Per the BAKE-EVERYTHING rule, the deploy image bakes:

1. `zen-sweep-worker` (built `--features salad-sweep` — glue + the
   shared inline encode+score compute).
2. The multi-arch `salad-http-job-queue-worker` Go sidecar (from the
   upstream GitHub release; baked, not boot-fetched).
3. An entrypoint that runs both (the upstream `with-shell-script` /
   `with-s6-overlay` pattern): the sidecar + `zen-sweep-worker worker
   --backend salad`. Set `SALAD_JOB_PORT` to the container group's
   `queue_connection.port` so the worker's HTTP receiver binds the port
   the sidecar POSTs to.

## Testing

The `compute` closure is backend-agnostic and is covered by the worker's
own tests. Only the thin Salad `JobQueue`/`CredentialSource`/`WorkerHost`
glue is Salad-node-only (the sidecar + IMDS need a real Salad node).
`tests/queue_roundtrip.rs` exercises the HTTP receiver end-to-end off a
Salad node by simulating the sidecar's blocking POST. **A real 1-replica
Salad container-group smoke sweep is the operator's gate** — it needs a
Salad node, the sidecar, IMDS, BYO R2 creds, and spend.
