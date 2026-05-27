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
zen-cloud-vastai  · vast.ai API + /proc/1/environ + Cloudflare R2
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

**Phase C — extract coefficient's providers.**
- `zen-cloud-gcp` (from `coefficient/gcp.rs` + GCS) and `zen-cloud-do`
  (from `coefficient/digitalocean.rs` + Spaces). coefficient becomes a
  thin `zen-fleet-launch` wrapper that adds provisioning for those
  providers on top of the shared worker. Cross-repo GPU-parity check
  (CubeCL vs cudarse) lands here as the one missing conformance test.

**Phase D — adopt everywhere, delete the forks.**
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
| `zen-cloud-local` | zenmetrics workspace | dev/Tower backend |
| `zen-cloud-gcp` / `-do` | zenmetrics workspace | from coefficient |
| `zen-sweep-worker` | zenmetrics workspace | the deployed binary |
| `zen-fleet-launch` | zenmetrics workspace | operator controller |
| `zenpicker-train` | zenanalyze workspace | next to zenpicker |
| `ARCHITECTURE.md` | new `imazen/zen-workspace` | workspace-wide map |

coefficient depends on `zen-cloud-{core,vastai,gcp,do}`; jxl-encoder +
zensim picker training depend on `zen-sweep-worker`. No dependency cycle:
the `zen-cloud-*` crates have zero `zenmetrics-root` deps.

---

## 7. Dispatch / sequencing

| Phase | Repo | Independent? | Status |
|---|---|---|---|
| §1 Phase A — cloud carve | zenmetrics | foundational | **DISPATCHED** |
| §1 Phase B — local backend | zenmetrics | after A | queued |
| §1 Phase C — gcp/do extract | zenmetrics + coefficient | after B | queued |
| §1 Phase D — adopt + delete forks | all | after C | queued |
| §2 stats finish + publish | zensim + coefficient | independent | queued |
| §3 TOML-driven trainer | zensim | independent of cloud | queued |
| §4 zenpicker-train | zenanalyze | needs §2 (zenstats) + §3 pattern | queued |
| §5 ARCHITECTURE.md | new repo | independent | queued |

Phase A lands first because §1 B/C/D depend on the trait abstraction it
introduces, and §4 reuses the same `compute`-closure worker shape. §2,
§3, §5 are independent and can run in parallel as capacity allows.
