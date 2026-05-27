# Zen workspace consolidation spec — cloud-agnostic worker + centralized stats / training / picker infrastructure

**Status:** APPROVED 2026-05-26 (user: "every single detail you suggested
and all of your recommendations are perfect — commit it all to a spec
and dispatch"). This is the authoritative design doc for the
cross-repo consolidation work it describes. Phase A of §1 is dispatched
on approval; later phases sequence behind it.

**Scope:** five refactors that centralize maintenance of (a) cloud
orchestration, (b) IQA statistics, (c) model training, (d) per/cross
codec picker training, and (e) workspace documentation. The unifying
thread: one canonical home per concern, a cloud-agnostic deployed
worker binary, and TOML recipes as the universal reproduce-this-exactly
interface.

**Provenance of decisions:** built on the 2026-05-26 dedup audit
(`imazen/zensim` `benchmarks/dedup_VERIFIED_synthesis_2026-05-26.md`)
and the zenstats consolidation that landed the same day
(`imazen/zenmetrics@36d71ca3` + `imazen/zensim@032b0119`).

---

## 1. Cloud orchestration — cloud-agnostic worker binary

### 1.1 The core principle

**The binary deployed on a compute node must NOT know which cloud it
runs on.** Provider-specific code lives only in pluggable backend
crates and in the launcher (which runs on the operator's workstation).
The worker is a generic "claim job → fetch inputs → compute → upload
artifacts → heartbeat" loop parameterized over traits.

This is the failure the current `vastai-fleet` (4957 LOC, tested Rust,
the live workhorse) and `coefficient`'s multi-provider stack solve
imperfectly: vast.ai assumptions (`/proc/1/environ` credential read,
R2-only storage, vast-specific onstart) are entangled with the generic
job loop. The 2026-05-17 "BAKE EVERYTHING, NEVER APT-INSTALL AT BOOT"
discipline (zensim `CLAUDE.md`) is the deployment half; this spec is
the code half.

### 1.2 Crate layering

```
zen-sweep-worker  (the binary baked into the docker image)
   │   generic; selects backend at runtime via config/feature
   │
   uses traits from:
zen-cloud-core    (pure traits + types; NO gpu / cloud / parquet deps)
   │   JobQueue · BlobStorage · Heartbeat · CredentialSource · WorkerHost
   │   generic job-loop runner: claim → fetch → compute → upload → refresh
   │
   ▼   provider impls (each its own crate, each implements the core traits):
zen-cloud-vastai  · vast.ai API + /proc/1/environ + Cloudflare R2   (pull / BYO-queue)
zen-cloud-salad   · SaladCloud API + IMDS + managed Job Queue + R2/S3 (push / managed-queue)
zen-cloud-runpod  · RunPod API + Serverless-queue OR Pods + R2/S3   (push OR pull)
zen-cloud-akash   · Akash SDL deploy + bid/lease + R2/S3             (pull / BYO-queue)
zen-cloud-gcp     · GCP Batch + GCS                (from coefficient/gcp.rs)
zen-cloud-do      · DigitalOcean + Spaces          (from coefficient/digitalocean.rs)
zen-cloud-local   · localhost + filesystem + sqlite queue (dev / Tower NAS)
zen-cloud-k8s     · k8s Job + S3-compatible        (future)

zen-fleet-launch  (the controller binary, runs on the operator workstation)
   │   provisions workers, uploads job chunks, monitors, scales, costs
   │   THIS is where provider-specific provisioning RPCs live — never
   │   on the worker.
```

### 1.3 Worker vs launcher — the load-bearing separation

| | Worker (`zen-sweep-worker`) | Launcher (`zen-fleet-launch`) |
|---|---|---|
| Runs on | the compute node (GPU box) | the operator workstation |
| Knows the cloud? | **NO** — generic over traits | YES — provisions per provider |
| Hot path? | yes (burns $/hr) | no |
| In the docker image? | yes — the only compute binary | no |
| Provider coupling | zero | unavoidable (provisioning differs) |

The launcher's vast.ai "find a $0.30/hr GPU" workflow is structurally
different from GCP Batch's "submit job → get assigned a node" — there is
no free unification there, and that's fine, because the launcher is not
the hot path. The **worker** being generic is what buys the value.

### 1.4 The five value-props (why this is worth ~1 extra week vs a flat extraction)

1. **One docker image** runs on vast.ai, GCP, DO, k8s, laptop — no
   per-provider rebuilds.
2. **Local dev / CI** — run the worker against `zen-cloud-local`
   (filesystem `BlobStorage` + sqlite `JobQueue`). Same binary, no
   cloud, no spend. Catches bugs before burning vast.ai dollars.
3. **Tower NAS overnight runs** — `zen-cloud-local` pulls jobs from a
   queue on Tower; same worker, no new code path.
4. **Multi-cloud sweep** — the launcher splits one sweep across vast.ai
   (cheap commodity GPUs) + GCP Batch (preemptible high-end) + the local
   7950X with zero worker-code awareness. Best price/perf workflow.
5. **coefficient's GCP/DO code stops rotting** — it becomes
   `zen-cloud-gcp` / `zen-cloud-do`, actively maintained alongside vastai.

### 1.5 Core trait surface (`zen-cloud-core`)

```rust
/// Pull-or-push job source. vast.ai is pull (atomic R2 ETag claim);
/// GCP Batch / k8s are push (controller assigns). Both expressible:
pub trait JobQueue {
    /// Return the next chunk to work, or None when the queue is drained.
    /// Pull impls do the atomic claim here; push impls return the
    /// pre-assigned chunk.
    fn next_chunk(&mut self) -> Result<Option<Chunk>, CloudError>;
    /// Acknowledge a chunk's terminal outcome (done / failed / retryable).
    fn ack_chunk(&mut self, id: &ChunkId, outcome: ChunkOutcome) -> Result<(), CloudError>;
}

/// Object storage. R2 / GCS / S3 / Spaces / local FS all fit.
pub trait BlobStorage {
    fn put(&self, key: &ArtifactKey, bytes: &[u8]) -> Result<(), CloudError>;
    fn get(&self, key: &ArtifactKey) -> Result<Vec<u8>, CloudError>;
    fn head(&self, key: &ArtifactKey) -> Result<Option<BlobMeta>, CloudError>;
    fn list(&self, prefix: &str) -> Result<Vec<ArtifactKey>, CloudError>;
    fn delete(&self, key: &ArtifactKey) -> Result<(), CloudError>;
}

/// Liveness signal. vast: heartbeat file in R2; k8s: liveness probe.
pub trait Heartbeat {
    fn beat(&self, worker: &WorkerId, status: WorkerStatus) -> Result<(), CloudError>;
}

/// Credential resolution. vast: /proc/1/environ; gcp: metadata server;
/// do: env; local: dotenv.
pub trait CredentialSource {
    fn resolve(&self) -> Result<std::collections::HashMap<String, String>, CloudError>;
}

/// Host-environment introspection (GPU count, scratch dir, node id).
pub trait WorkerHost {
    fn worker_id(&self) -> WorkerId;
    fn scratch_dir(&self) -> std::path::PathBuf;
    fn gpu_count(&self) -> usize;
}

/// The generic job loop — the heart of the worker. Backend-agnostic.
pub fn run_worker<Q, S, H, C, W, F>(
    queue: &mut Q, storage: &S, heartbeat: &H, host: &W, mut compute: F,
) -> Result<WorkerSummary, CloudError>
where
    Q: JobQueue, S: BlobStorage, H: Heartbeat, W: WorkerHost,
    F: FnMut(&Chunk, &S, &W) -> Result<ChunkOutcome, CloudError>,
{ /* claim → fetch → compute → upload → ack → beat, looped */ }
```

The `compute` closure is what differs per sweep (encode+score, picker
train, IQA panel batch) — everything around it is shared.

### 1.6 The five design decisions (all RESOLVED)

1. **Job-queue semantics** — `next_chunk()` + `ack_chunk()` covers both
   pull (vast atomic claim inside `next_chunk`) and push (controller
   pre-assigns, `next_chunk` returns it). RESOLVED: this shape.
2. **Storage abstraction** — `put/get/head/list/delete` on
   `ArtifactKey`. Every cloud SDK is this shape. RESOLVED: trivial.
3. **Credential injection** — `CredentialSource::resolve() ->
   HashMap`. Each impl knows its provider's convention. RESOLVED.
4. **Compile-time vs runtime backend selection** — RESOLVED: **cargo
   features + trait objects**, NOT dlopen. `zen-sweep-worker
   --features vastai,gcp,local`; runtime `--backend <name>` picks among
   the compiled-in set. dlopen is overkill at our scale.
5. **Crate home** — RESOLVED: **zenmetrics workspace**, same reasoning
   as zenstats. `zen-cloud-*` are self-contained members with ZERO
   `zenmetrics-root` deps, so any consumer (coefficient, jxl-encoder,
   zensim picker training) depends on the sub-crate without dragging GPU
   infrastructure. coefficient → depends on `zen-cloud-core` +
   `zen-cloud-vastai` + adds `zen-cloud-gcp`/`-do` on top.

### 1.7 Migration phases

**Phase A — carve, no behaviour change (DISPATCHED on approval).**
- `vastai-fleet` → renamed `zen-cloud-vastai` (impl of the core traits).
- New `zen-cloud-core` crate: the traits + types + generic `run_worker`
  loop, extracted from `vastai-fleet`'s current public API.
- New `zen-sweep-worker` binary: depends on `zen-cloud-core` +
  `zen-cloud-vastai` (default features). Same outputs as today's worker.
- Existing docker images build with `--features vastai` (default) and
  produce byte-identical sweep results. The 13-test JSON parser + tokio
  worker tests in vastai-fleet MUST stay green throughout.
- Acceptance gate: a 1-box smoke sweep on vast.ai produces the same
  artifacts as a pre-carve baseline; all vastai-fleet tests pass; the
  docker image builds and the worker entrypoint's tool-verification
  (`command -v zen-sweep-worker; …`) passes.

**Phase B — validate the abstraction with `zen-cloud-local`.**
- Add `zen-cloud-local` (filesystem `BlobStorage` + sqlite `JobQueue` +
  dotenv `CredentialSource`). Run `zen-sweep-worker --backend local`
  against a local job queue end-to-end. If the trait shapes are wrong,
  fix them HERE — cheaply, before more providers. Add a local-backend
  integration test to CI (no cloud spend).

**Phase C — SaladCloud provider `zen-cloud-salad` (USER PRIORITY — the
vast.ai alternative; full design in § 1.9).**
- New `zen-cloud-salad` crate implementing the core traits for
  SaladCloud. This is the user's primary second provider, prioritised
  AHEAD of the coefficient gcp/do extraction (which is legacy cleanup).
- Worker side: bake the multi-arch Go `salad-cloud-job-queue-worker`
  sidecar into the image; `zen-cloud-salad`'s `JobQueue` impl speaks the
  sidecar's local gRPC contract (blocking receive in `next_chunk()`),
  `CredentialSource`/`WorkerHost` read the Salad IMDS, `BlobStorage`
  reuses the R2/S3 impl shared with vast.
- Launcher side: `zen-fleet-launch` gains a Salad provider — create
  container group via the public API, attach a managed queue, set GPU
  class + replica count, push jobs to the Salad queue.
- Acceptance gate: a 1-replica Salad container-group smoke sweep
  produces the same artifacts as the vast.ai path on the same chunks.

**Phase D — extract coefficient's providers.**
- `zen-cloud-gcp` (from `coefficient/gcp.rs` + GCS) and `zen-cloud-do`
  (from `coefficient/digitalocean.rs` + Spaces). coefficient becomes a
  thin `zen-fleet-launch` wrapper that adds provisioning for those
  providers on top of the shared worker. Cross-repo GPU-parity check
  (CubeCL vs cudarse) lands here as the one missing conformance test.

**Phase E — adopt everywhere, delete the forks.**
- `jxl-encoder/zenjxl-tuning-sweep`, per-codec picker-training sweeps,
  and V_X bake sweeps all switch to `zen-sweep-worker` + a per-sweep
  `compute` closure instead of forking the bash.
- Delete the 11 deprecated `onstart_v2/v3` bash forks across repos.
- Any shell that remains is boot-time-only hydrate/verify (`zen-fleet.sh`),
  per the BAKE-EVERYTHING rule.

### 1.8 The trade flagged (and accepted)

The launcher does NOT get unified — provisioning a vast.ai box vs a GCP
Batch node vs a k8s pod is structurally different and stays per-provider
in `zen-fleet-launch`. Accepted: the launcher is the operator's
workstation tool, not the hot path. The worker being generic is the win.

### 1.9 SaladCloud provider (`zen-cloud-salad`) — full design

Added 2026-05-26 on user request ("add salad.com as a vast.ai
alternative"). SaladCloud runs containers on distributed consumer GPUs
at low $/hr — a strong commodity-GPU alternative to vast.ai. It is the
first real exercise of the "add a provider" path, and it validates the
push-vs-pull `JobQueue` abstraction.

**The architectural wrinkle: Salad's queue is MANAGED + PUSH.** vast.ai
is bring-your-own-queue + pull (the worker claims chunks from R2 via
atomic If-Match ETag writes). SaladCloud provides a *managed* Job Queue
plus a multi-arch Go sidecar binary (`salad-cloud-job-queue-worker`,
Apache-2.0, github.com/SaladTechnologies/salad-cloud-job-queue-worker)
that runs INSIDE your container, receives jobs from the queue, and
forwards them to your application over local gRPC. This is exactly the
divergence `JobQueue::next_chunk()` was designed to hide:

| | vast.ai | SaladCloud |
|---|---|---|
| queue ownership | BYO (we use R2) | managed by Salad |
| distribution model | pull (worker claims) | push (sidecar → app gRPC) |
| `next_chunk()` impl | atomic R2 ETag claim | blocking recv on sidecar gRPC |
| object storage | Cloudflare R2 | BYO R2/S3 (Salad has none) |
| worker identity | `/proc/1/environ` | Salad IMDS |
| credential source | vast env injection | IMDS + container-group env |

**Worker-side integration (`zen-cloud-salad`, Phase 1 = pragmatic):**
1. Bake the Go `salad-cloud-job-queue-worker` sidecar into the
   `zen-sweep-worker` docker image (tiny multi-arch Go binary; per the
   BAKE-EVERYTHING rule, no boot-time fetch). It is the queue→app bridge.
2. `JobQueue` impl: `next_chunk()` blocks reading the next job off the
   sidecar's local gRPC stream; `ack_chunk()` returns the result to the
   sidecar, which returns it to the managed queue. (The sidecar speaks
   protobuf/gRPC — its `.proto` is in the upstream repo; we generate a
   thin Rust gRPC client with `tonic`, OR run the sidecar as the gRPC
   *server* and zen-sweep-worker as the client per the sample's pattern.)
3. `CredentialSource` + `WorkerHost`: read the Salad IMDS (instance
   metadata service auto-discovered on a Salad node — `SALAD_MACHINE_ID`,
   container-group id, GPU info) plus any env vars set in the container
   group definition.
4. `BlobStorage`: reuse the shared R2/S3 impl (factor it out of
   `zen-cloud-vastai` into a `zen-cloud-s3` helper both depend on, OR
   keep an S3 impl in `zen-cloud-core` behind a feature — decide during
   Phase C; do not duplicate the R2 client).
5. `Heartbeat`: Salad manages instance liveness natively and the sidecar
   handles per-job acks, so the Salad `Heartbeat` impl is largely a
   no-op / thin status-report. Don't reimplement vast's R2-heartbeat.

**Phase 2 (future optimization, NOT Phase C):** reimplement the sidecar
protocol natively in Rust (talk to Salad's queue + IMDS directly,
dropping the Go sidecar). Only worth it if the sidecar proves a
reliability or packaging burden — the bake-the-Go-binary path ships
faster and reuses Salad's supported, tested bridge.

**Launcher-side (`zen-fleet-launch` Salad provider):**
- Auth: `Salad-Api-Key: <key>` header. Key from the portal
  (portal.salad.com → API Keys); store in `~/.config/salad/credentials`
  mirroring the R2 creds convention.
- Provision: `POST https://api.salad.com/api/public/organizations/
  {org}/projects/{project}/containers` — body sets the image, replica
  count, GPU class id (GET the class list first to resolve the id),
  vCPU/RAM, container-registry auth, env vars, and the attached job
  queue. There is NO Rust SDK (Salad ships Python/Go/Java/JS/.NET only),
  so hand-roll the REST calls with `reqwest` + serde, or generate a
  thin client from Salad's OpenAPI spec (in salad-cloud-docs repo).
- Job submission: push chunk descriptors into the Salad managed queue
  via the queue API; Salad fans them out to the sidecars.
- Monitor / scale: poll container-group + instance status; adjust
  replica count.

**Local-testing caveat:** the Salad sidecar "only runs on a SaladCloud
node" (IMDS dependency) — so the Salad path canNOT be exercised by
`zen-cloud-local` the way other backends can. Phase C's smoke test must
run on a real 1-replica Salad container group. Salad has stated a
local-test tool is planned; until then, the `compute` closure itself is
backend-agnostic and IS covered by the `zen-cloud-local` integration
test (Phase B), so only the thin Salad `JobQueue`/`CredentialSource`
glue is Salad-node-only.

**Why prioritised ahead of gcp/do:** the user actively wants Salad as a
running vast.ai alternative (cheap commodity GPUs, same workload class).
gcp/do are coefficient-legacy cleanup with no active demand. Salad is
Phase C; gcp/do drop to Phase D.

### 1.10 RunPod + Akash provider candidates (user request 2026-05-26)

The user wants RunPod and Akash as further vast.ai/Salad alternatives.
Both slot into the same `zen-cloud-*` provider pattern — **no
architecture change**, just two more crates + launcher provisioning.
They validate the trait layer a second and third time after Salad.

**`zen-cloud-runpod` (Phase F — closest to Salad/vast, lowest effort).**
RunPod offers two modes; the trait layer covers both:
- **Serverless (push):** RunPod's queue invokes a handler per job. The
  `JobQueue::next_chunk()` impl blocks on the handler invocation and
  `ack_chunk()` returns the result — same shape as Salad's HTTP receiver.
  Best fit for the sweep workload (autoscale-to-zero, pay-per-job).
- **Pods (pull):** rent a persistent GPU pod; the worker claims chunks
  from R2 exactly like vast.ai. Reuses `zen-cloud-vastai`'s claim logic
  + `zen-cloud-s3` storage verbatim.
- Launcher: RunPod has a **GraphQL API** (pod create/terminate, GPU type
  selection, serverless endpoint deploy). Hand-roll `reqwest` + a GraphQL
  query string (no official Rust SDK). Auth: API key header.
- Storage: BYO R2/S3 (RunPod has network volumes but S3 is the portable
  choice — reuse `zen-cloud-s3`).
- Identity/creds: RunPod injects env vars into the container; no special
  IMDS. `CredentialSource` reads env; `WorkerHost` reads
  `RUNPOD_POD_ID` / `RUNPOD_*`.
- Deploy image: same BAKE-EVERYTHING pattern; serverless mode needs a
  RunPod handler shim (their python/go handler SDK) OR the worker speaks
  their serverless HTTP contract directly — decide during Phase F by
  reading the current RunPod serverless worker protocol.

**`zen-cloud-akash` (Phase G — decentralized, most launcher-divergent).**
Akash is a Cosmos-based decentralized compute marketplace. The **worker**
side is identical to vast/runpod-pods (BYO S3 + R2-pull queue); the
**launcher** is where Akash diverges hard:
- Deploy via an **SDL manifest** (YAML describing the container, GPU
  resources, pricing) → submit a deployment tx → providers **bid** →
  accept a bid → **lease** → the workload runs. This is a multi-step
  on-chain flow with a wallet + escrow (AKT tokens), not a single REST
  call.
- Launcher needs: Akash CLI (`provider-services` / `akash`) or the
  Cosmos SDK gRPC, a funded wallet, bid evaluation, lease management.
  This is genuinely more work than any other provider's launcher — keep
  it LAST. The `zen-fleet-launch` Akash module wraps the CLI rather than
  reimplementing the Cosmos tx flow in Rust (the CLI is the supported
  path).
- Worker: zero Akash-specific code — it's a generic S3+pull worker. So
  `zen-cloud-akash` may be launcher-only (no worker crate), with the
  deployed image being the same generic `zen-sweep-worker` the local /
  vast pull-path uses.
- Storage/identity: BYO R2/S3; env-var creds (Akash injects via the SDL
  `env`). No IMDS.

**Sequencing rationale:** RunPod (Phase F) is a near-clone of the
Salad/vast effort — high value, low marginal cost, do it right after
Salad ships. Akash (Phase G) has the same trivial worker but the most
divergent launcher (on-chain bid/lease), so it lands last and wraps the
Akash CLI rather than reimplementing the chain flow. Both depend only on
the Phase A trait layer (done) + the shared `zen-cloud-s3` (done in
Phase C) — neither needs Phase B/D/E.

---

## 2. Stats — finish migration, then publish `zenstats`

**Done this session:** `zenstats` crate (paper-correct Mohammadi 2025
panel: SROCC + PLCC + KROCC + OR per ITU-T P.1401 + PWRC per Mohammadi
§ VII SA-ST AUC + Z-RMSE + MRR + bootstrap CI + decisive) at
`imazen/zenmetrics@36d71ca3`. zensim's `panel.rs` is a re-export shim.
bake_verdict's silently-stale inline OR/PWRC fixed (V39 numbers
corrected: CID22 OR 0.04→0.00, PWRC 0.93→0.98). ensemble_mix +
eval_bake_per_band + mlp_train/utils routed through panel. zenanalyze
(0 real inline reimpls) + jxl-encoder (1 migrated) done.

**Remaining:**
1. Migrate the ~14 zensim-local py reimpls that carry DEPRECATED banners
   but still have active callers (e.g. `mohammadi_eval.py` ← called by
   `fit_output_spline.py`, `run_experiment.sh` [now deleted],
   `verify_panel_parity.py`). Migrate the callers to `scripts/lib/zen_stats.py`,
   then delete the reimpl.
2. Re-dispatch the coefficient py-stats migration (was blocked by a
   sibling agent's gpu-strip work; marker now clear as of 2026-05-27T01:04Z).
3. **Optional** `pwrc_wu_auc_ca` — add the full Wu 2017 Eq 19 AUC_ca
   (A·D·M product integrated over `[min{2σ̂}, max{2σ̂}]`) ALONGSIDE the
   current SA-ST AUC, for callers wanting Mohammadi-Table-3-comparable
   ABSOLUTE numbers (CVVDP=5.92, IW-SSIM=5.76). The Wu formulas are
   verified (pdf-oxide extraction of arXiv:1705.05126): Eq 9 activation
   `A(x,T)=1/(1+exp(-C₁(|xᵢ-xⱼ|-T)))`, Eq 13 concordance `D=sgn·sgn`,
   Eq 14 importance `M=(e^dᵢⱼ+e^lᵢⱼ-2)/Σ`, Eq 15 `S=ΣADM`, Eq 19
   `AUC_ca=∫S dT`. Rank order agrees with SA-ST AUC; this only matters
   for absolute-scale paper reproduction. Skip unless reproducing Table 3.
4. **Publish `zenstats 0.1.0` to crates.io** once external migration is
   clean — so external IQA research builds on the same panel. Gate:
   `cargo semver-checks`, README verified, all tests green on
   windows-11-arm + macos-intel + i686 per CLAUDE.md release rules.

---

## 3. Training — Rust canonical, TOML manifests as input

**Today:** Python `train_v_next_mlp.py` and Rust `zensim_mlp_train`
both exist; recent work (V39, V46) is Rust-side. Bake manifests
(`zensim/weights/manifests/*.toml`, landed `8a95a2a16d`) capture the
recipe per shipped bake as OUTPUT (provenance) — trainers take CLI
flags, not TOML.

**Proposal:**
1. **Flip TOML from output to input.** `zensim_mlp_train --manifest
   manifests/foo.toml` reproduces a bake exactly: the manifest carries
   the full training command + every flag + every input file (path +
   sha256 + row count + R2/Tower mirror URL) + post-training steps
   (spline injection etc.). New bake = new manifest = exact regenerate.
   This solves the V32 recipe-archaeology incident the manifest schema
   was built for, end-to-end.
2. **Retire the Python trainer in zensim** (`train_v_next_mlp.py`).
   zenanalyze/zentrain's Python feature-generation pipeline stays — that
   is pre-training data-prep, a different concern. Only the train LOOP
   migrates; the Rust trainer can shell out to the Python data-prep CLI
   as a step rather than re-implementing pandas joins.
3. Same TOML-recipe pattern extends to picker training (§4).

**Trade:** the Python pipeline's pandas/plotly data-prep niceties aren't
trivial to port. Don't — call them as a step. Only the train loop is
Rust-canonical.

---

## 4. Per/cross codec picker training — unified Rust pipeline

**Today:** zenmetrics sweep produces `unified_*_cvvdp.parquet` per-codec
cuts. Each codec has its own picker bake (only `zenwebp_picker_v0.1.bin`
shipped). zenpicker's `MetaPicker` consumes them via
`zenpredict::Predictor`, but training is per-codec via Python
`zentrain/examples/*.py` config modules. Neither zensim (IQA) nor
zenmetrics (GPU scoring) owns picker training.

**Proposal: new `zenpicker-train` binary in the zenanalyze workspace**
(where zenpicker lives). Inputs:
- input parquet path (from `zen-metrics-cli assemble` output);
- codec family (`Jpeg|Webp|Jxl|Avif|Png|Gif`, or `meta` for cross-codec);
- target metric column (zenstats nomenclature — the same names panel
  uses);
- TOML manifest (the §3 recipe interface).

It:
- uses **zenstats** for eval-time SROCC / PLCC / per-band gate (one stat
  path, the same the metric work uses);
- uses **zenmetrics CubeCL kernels** to GPU-accelerate the inner loop;
- uses **cmaes** (already a zensim-validate dep) for the
  hyperparameter search the Python scikit-learn path currently does;
- outputs a ZNPR v3 picker bake + a sibling TOML manifest;
- **regenerates the cross-codec `MetaPicker` automatically** when any
  per-codec bake updates — it is just a `Predictor` over the same
  feature space, so a sweep-data update cascades to a fresh meta-bake.

**Data-flow it formalizes:**
```
sweep (zen-sweep-worker) → unified parquet (zen-metrics-cli assemble)
  → zenpicker-train (per-codec) → per-codec bakes + manifests
  → zenpicker-train (meta)      → MetaPicker bake + manifest
  → zenstats panel gate         → ship/no-ship verdict
```

**Trade:** picker training is Python-first because zentrain's
hyperparameter search is mature in scikit-learn. Re-implementing in Rust
means porting the search (cmaes leverage helps). One-time ~1 week.
Recurring benefit: pickers regenerate automatically when sweep data
changes; no manual Python re-run.

---

## 5. Docs — single workspace architecture map

**Today:** each repo has its own `CLAUDE.md` (zensim's is 2000+ lines),
plus `zenanalyze/everything.md`, `DATA_PROVENANCE.md`, and N
`benchmarks/dedup_*.md` audits. No single entry pointing at all of them.

**Proposal:** a standalone `imazen/zen-workspace` repo at `~/work/zen/`
holding one `ARCHITECTURE.md`: the data-flow diagram (sources → features
→ bakes → pickers → codecs), the per-repo responsibility map, current
canonical paths, and where each big decision lives. Per-repo `CLAUDE.md`
stays for agent context; `ARCHITECTURE.md` is the human-readable map.
This spec is the first artifact of that consolidation; it can seed the
new repo or stay in zenmetrics/docs until the repo is created.

**Trade:** evergreen docs drift. Mitigation: keep the immutable
timestamped dedup-ledger discipline for audits; `ARCHITECTURE.md` is a
thin, frequently-touched index, not a deep doc.

---

## 6. Crate-home summary (all RESOLVED)

| Crate / binary | Home | Rationale |
|---|---|---|
| `zenstats` | zenmetrics workspace | landed 36d71ca3; self-contained member |
| `zen-cloud-core` | zenmetrics workspace | self-contained, zero gpu deps |
| `zen-cloud-vastai` | zenmetrics workspace | renamed from vastai-fleet |
| `zen-cloud-salad` | zenmetrics workspace | SaladCloud (managed queue + IMDS), user-priority alt |
| `zen-cloud-local` | zenmetrics workspace | dev/Tower backend |
| `zen-cloud-runpod` | zenmetrics workspace | RunPod serverless-queue OR pods; user-requested alt |
| `zen-cloud-akash` | zenmetrics workspace (launcher-only likely) | Akash SDL/bid/lease; worker is generic S3+pull |
| `zen-cloud-gcp` / `-do` | zenmetrics workspace | from coefficient |
| `zen-sweep-worker` | zenmetrics workspace | the deployed binary |
| `zen-fleet-launch` | zenmetrics workspace | operator controller |
| `zenpicker-train` | zenanalyze workspace | next to zenpicker |
| `ARCHITECTURE.md` | new `imazen/zen-workspace` | workspace-wide map |

coefficient depends on `zen-cloud-{core,vastai,salad,gcp,do}`;
jxl-encoder + zensim picker training depend on `zen-sweep-worker`. No
dependency cycle: the `zen-cloud-*` crates have zero `zenmetrics-root`
deps.

---

## 7. Dispatch / sequencing

| Phase | Repo | Independent? | Status |
|---|---|---|---|
| §1 Phase A — cloud carve | zenmetrics | foundational | **IN FLIGHT** (zen-cloud-core landed de66b1b0) |
| §1 Phase B — local backend | zenmetrics | after A | queued |
| §1 Phase C — **SaladCloud** `zen-cloud-salad` | zenmetrics | after A (user priority) | queued |
| §1 Phase D — gcp/do extract | zenmetrics + coefficient | after B | queued |
| §1 Phase E — adopt + delete forks | all | after C/D | queued |
| §1 Phase F — **RunPod** `zen-cloud-runpod` | zenmetrics | after A (user-requested alt) | queued |
| §1 Phase G — **Akash** `zen-cloud-akash` | zenmetrics + launcher | after A (launcher-divergent, do last) | queued |
| §2 stats finish + publish | zensim + coefficient | independent | queued |
| §3 TOML-driven trainer | zensim | independent of cloud | queued |
| §4 zenpicker-train | zenanalyze | needs §2 (zenstats) + §3 pattern | queued |
| §5 ARCHITECTURE.md | new repo | independent | queued |

Phase A lands first because the rest of §1 depends on the trait
abstraction it introduces, and §4 reuses the same `compute`-closure
worker shape. Salad (Phase C) needs only the trait layer (Phase A), NOT
the local backend (Phase B) — so it can start the moment A lands, in
parallel with B. §2,
§3, §5 are independent and can run in parallel as capacity allows.
