# zenfleet-salad — SaladCloud provider

The SaladCloud backend for the `zenfleet-cloud` worker trait layer
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

So `zenfleet-salad` serves a **tiny local HTTP endpoint** rather than
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
| `BlobStorage` | `storage::SaladBlobStorage` | Re-export of the shared `zenfleet-s3` impl. Salad has no native object store; workers BYO R2/S3. **No second S3 client** (spec §1.9 item 4). |
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

### Per-sweep scoped R2 credential (mint + inject)

Salad runs containers on hardware the operator does NOT own (distributed
*consumer* GPUs). Injecting the **root** R2 key into a container exposes
the whole R2 account to a hostile node operator who can read the
container env. So the launcher mints a credential scoped to ONE bucket
(object-read-write, short TTL) and injects only that. Blast radius of a
compromised node is then one bucket's objects, not the account.

The minter is the shared, provider-agnostic
`zenfleet_cloud::r2creds::mint_scoped_r2_cred` (runpod/vastai reuse it).
The Salad-shaped flow:

1. `R2ParentCreds::from_env()` reads the operator-box parent keys:
   `CF_API_TOKEN` || `R2_API_TOKEN` (Cloudflare REST bearer), plus
   `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` (root R2
   S3 key+secret). These mirror `~/.config/cloudflare/r2-credentials` and
   are NEVER injected into a worker.
2. Build a `ScopedCredSpec::new("<working-bucket>")` (6h default TTL,
   `object-read-write`); `.with_prefixes([...])` / `.with_ttl_seconds(n)`
   to tighten (TTL clamped to `[900, 604800]`).
3. `SaladApi::create_container_group_with_scoped_cred(req, Some((&parent,
   &spec)))` mints the cred and injects `R2_ACCESS_KEY_ID` /
   `R2_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` into the container-group
   env (existing env-injection mechanism; these three are added/overwritten).
   Pass `None` to use caller-supplied creds verbatim (minting is opt-in,
   not forced). Returns the group + the minted `ScopedR2Cred` (with
   `expires_at`) for re-mint bookkeeping.

The injected `AWS_SESSION_TOKEN` is REQUIRED — the entrypoint writes it
to `~/.aws/credentials` as `aws_session_token`; a temp key+secret without
it 403s. The verified Cloudflare endpoint is **account-level**
(`.../r2/temp-access-credentials`, bucket in the body); the bucket-scoped
path returns CF error 10015. See `~/work/claudehints/topics/r2-credentials.md`.

**Follow-ons (NOT built yet):**

- **Single-bucket scope.** R2 temp creds are scoped to one bucket. A
  sweep that READS source images from bucket A and WRITES artifacts to
  bucket B needs either presigned source URLs in the job input or a
  SECOND read-scoped cred (e.g. a separate `R2_SOURCE_*` env block). The
  current flow mints one object-read-write cred for the WORKING bucket
  only.
- **Re-mint before TTL expiry.** A sweep that outlasts the cred TTL needs
  the launcher to re-mint and update the container-group env (the max TTL
  is 7d; the 6h default covers most sweeps, so this matters only for very
  long runs). `ScopedR2Cred::expires_at` is the bookkeeping hook; the
  re-mint loop is not implemented.

## Deploy image

Per the BAKE-EVERYTHING rule, the deploy image bakes:

1. `zenfleet-sweep` (built `--features salad-sweep` — glue + the
   shared inline encode+score compute).
2. The multi-arch `salad-http-job-queue-worker` Go sidecar (from the
   upstream GitHub release; baked, not boot-fetched).
3. An entrypoint that runs both (the upstream `with-shell-script` /
   `with-s6-overlay` pattern): the sidecar + `zenfleet-sweep worker
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
