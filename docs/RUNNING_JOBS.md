# Running jobs on the zen job system

A practical, end-to-end guide to declaring work, bringing up a heterogeneous fleet (including your
**Unraid basement box**), running it, reading results, and tearing down. It is honest about what is
production-proven versus what you supply (the **executor** for real encode/score work).

---

## 1. The mental model — one queue, many interchangeable workers

```
  you ──declare──▶  manifest (DesiredJob[])  ──▶  R2 lease-queue  ◀── workers claim & execute
   (zenfleet-ctl)         ▲                          (one bucket)         (local / basement / burst)
                        │                               │                       │
                   coverage ◀── Parquet ledger ◀────────┴── content-addressed ──┘
                   (catalog)     (the truth)                blobs/<sha256>
```

- **Jobs are content-addressed.** A `JobId` is `sha256(kind + sorted inputs)`. Declaring the same work
  twice is a structural no-op — there is no "did I already run this?" bookkeeping.
- **The ledger is the truth, not the queue.** Every finished job writes a row to a columnar Parquet
  ledger in R2 (latest-wins on `(job_id, ts)`). Coverage, the dashboard, and the reconciler all read
  the ledger. A worker/box/queue/dashboard can die at any point and the run converges on restart.
- **The queue is an R2 conditional-write lease.** A worker claims a job by `PutObject` with
  `If-None-Match: *` on `claims/<job_id>` — exactly one worker wins, so no double execution.
- **Workers are interchangeable and provider-agnostic.** Adding or removing a tier never touches job
  logic. Workers are **pull-based**: outbound HTTPS to R2 only, so a NAT'd basement box is a
  first-class tier with no inbound ports.

### Job kinds (`zenfleet_core::JobKind`)
`Encode {codec,q,knobs}` · `Metric {metric}` · `ScoreFile {metrics, hdr, hdr_transfer}` ·
`Feature {regime}` · `Diffmap {metric}` · `Resample {kernel,w,h}` · `Bake {view}`. Each has a
`profile()` giving its **resource class** (`CpuLight/CpuHeavy/CpuArm/Gpu/HighRam` — for capability
routing) and **GC regenerability** (expensive encodes are kept; cheap re-scores are an LRU cache).

`ScoreFile` is the no-re-encode whole-file scorer: one job = one source file + N pre-encoded
variant content-shas (`inputs`) + M metrics; the executor decodes the reference ONCE and scores
every metric for every variant (the `SourceSha` grouping made concrete). Manifests are built by
`scripts/jobsys/build_scorefile_manifest.py` / `build_scorefile_from_pairs.py`; write-back via
`scripts/jobsys/writeback_scores.py`. For HDR corpora see "HDR ScoreFile" below.

### HDR ScoreFile (persisted-pairs HDR corpora, e.g. kadis-hdr)

`ScoreFile { hdr: true, hdr_transfer }` scores an HDR corpus through the job system instead of an
ad-hoc `score-pairs --hdr` chunk fleet. The executor decodes the reference and every variant to
**absolute-luminance nits** (PQ-PNG / PQ-JXL / PQ-AVIF / EXR via `hdr::decode_to_nits`) and applies
the exact per-metric feeding `zenmetrics score-pairs --hdr` uses (the shared layer in
`zenmetrics-cli/src/hdr.rs` — one implementation, both callers):

| metric | HDR feeding |
|---|---|
| `cvvdp-gpu` | faithful linear planes, `DisplayTarget::hdr(1000)` |
| `butteraugli-gpu` | faithful umbrella `HdrScorer` (intensity_target = peak) |
| `cvvdp` (native CPU) | `to_cvvdp_rgb8` per-image-peak u8 |
| `zensim` / `zensim-gpu` | PU21 u8 shell + the **v1 u8-shell** feature row (372 `with-iw` on GPU) |
| `ssim2(-gpu)` / `iwssim(-gpu)` / CPU `butteraugli` | PU21/PQ u8 shell → `run_metric` |
| `dssim(-gpu)` | REFUSED loudly — no HDR path by design (external dssim-core transform) |

Declare a persisted-pairs HDR corpus with `scripts/jobsys/build_scorefile_hdr_pairs.py
<corpus_prefix_s3> <run_id>` (refs/ + dist/ + pairs.tsv layout → `jobs/<run>/{variants.tar,
variant_index.tsv, manifest.json[.gz], dist_sha_map.tsv}`; the sha map rejoins JSONL rows —
keyed `encode_sha` — to the corpus pairs rows and their per-cell q/knob identity). Launch
with `ZEN_CORPUS_PREFIX=<corpus prefix>` plus the usual `ZEN_VARIANTS_TAR_URI` /
`ZEN_VARIANT_INDEX_URI`. Verified end-to-end 2026-07-14 (EXR smoke corpus, CUDA box):
1 job × 2 variants × 5 metrics → 12 rows, all finite, 372-feature rows, zero errors.

Rules and gotchas:

- **Job identity**: `hdr`/`hdr_transfer` are `#[serde(default, skip_serializing_if)]` — SDR jobs
  serialize byte-identically to the pre-HDR schema (existing ledgers/ids untouched), while
  `hdr: true` correctly yields a DIFFERENT content-addressed id (different work).
- **`hdr_transfer`** is optional; absent = `pu-rescale` (the validated fleet setting). `"pq"` is
  the other accepted value.
- **Executor version gate** ⚠️: an executor image from before this feature (or built without the
  `hdr` cargo feature) doesn't know the flag. The new executor REFUSES `hdr: true` without its
  `hdr` feature; an *old* binary would silently score SDR — pin the executor image ≥ the version
  shipping the HDR arm before declaring HDR manifests.
- **Variant extensions matter in HDR** (`decode_to_nits` dispatches on extension): use a 4-column
  `variant_index.tsv` (sha, offset, size, **name**) so the executor can name range-GET temps with
  the member's real extension; persisted-pairs cells carry a distortion label (not a codec) in
  `cell.codec`.
- **Output rows** carry `"hdr": true` for provenance; zensim emits its metric row AND a `feature`
  row from the same u8-shell call, matching the SDR arm's shape (write-back unchanged).
- **Build**: while the codec siblings are mid-refactor, the executor builds codec-free via the
  lean `jobexec` feature: `cargo build --release -p zenmetrics-cli --no-default-features
  --features jobexec,hdr,cpu-metrics,gpu,gpu-cuda,gpu-cpu` (PQ-PNG decode additionally needs the
  `png` feature once zenpng settles; EXR works in every `hdr` build). `sweep` implies `jobexec`.
- **Parity**: `tests/hdr_pair_parity.rs` + `jobexec::hdr_tests` lock the executor's values to the
  `score-pairs --hdr` feeding bit-exactly (CPU metrics; the GPU faithful paths reuse the same
  score-pairs primitives verbatim; live GPU parity measured 2026-07-14 — deltas 0 everywhere,
  cvvdp-gpu 4.8e-7 from atomic-f32 pooling order).
- **Decode-ahead**: the HDR variant loop prefetches + decodes variant *i+1* on a thread while
  scoring variant *i* (bounded to one nits buffer of readahead). Verified to emit byte-identical
  rows to the serial loop.

---

## 2. The executor contract  ⚠️ read this before running real work

The orchestration (declare → queue → claim → content-address → ledger → coverage → fleet → GC) is
**production-proven**. The thing that does the *actual* encode/score is a separate program you point
the worker at with `ZEN_EXEC`. The worker speaks one tiny contract per job
(`crates/zenfleet-worker/src/lib.rs::exec_command`):

| direction | payload |
|-----------|---------|
| **stdin** → executor | the `DesiredJob` as one JSON object, then EOF |
| executor → **stdout** | the job's **output bytes** (these are content-addressed to `blobs/<sha256>`) |
| exit code | `0` = success · non-zero = deterministic FAILED row (with `error_class`) · spawn failure = transient (retried) |

A `DesiredJob` on stdin looks like:
```json
{"kind": {"kind": "metric", "metric": "cvvdp"},
 "inputs": ["aa43…sha256…", "bb91…sha256…"],
 "cell": {"image_path": "corpus/img-00001.png", "codec": "zenjpeg", "q": 80, "knob_tuple_json": "{}"}}
```

**What ships today:** the synthetic executor `ZEN_EXEC=/bin/cat` honors the contract trivially (echoes
the job JSON back as the "output"). It proves the whole pipeline end-to-end — declare, claim,
content-address, ledger, coverage, multi-provider concurrency — **without doing real work**. Every
live proof in this repo (including the 3-provider run) used it.

**The real executor now ships: `zenmetrics jobexec`.** It honors the contract: reads the `DesiredJob`
on stdin, resolves the source (local / `s3://` / `$ZEN_CORPUS_PREFIX` via s5cmd), and for an `encode`
job emits the encoded bytes, for a `metric` job re-encodes the cell + scores it (`run_metric`) and
emits a JSON score row. Codecs: zenpng/zenjpeg/zenwebp/zenavif/zenjxl. Metrics: CPU ssim2/butteraugli/
zensim today (GPU metrics return a clear "needs a GPU build" error — they want a GPU build + tier).
Proven end-to-end through the real worker (encode + score → content-addressed blob, blob sha == output_sha).

**It is baked into a ready image: `ghcr.io/imazen/zenfleet-worker:exec`** (the worker base +
`zenmetrics` + the `zenfleet-exec` shim; `ZEN_EXEC` defaults to the real executor). Set
`ZEN_WORKER_IMAGE=ghcr.io/imazen/zenfleet-worker:exec` on any tier to run real jobs — or better, source
`scripts/jobsys/fleet.env` and use `$ZEN_FLEET_IMAGE_CPU` (the single source of truth). ⚠️ The
deprecated splinter name `zenfleet-worker-exec` is BANNED by the ghcr guard — variants are TAGS on the
canonical `zenfleet-worker` package (`:exec`, `:exec-gpu`), never new package names (see
`docs/GHCR_PACKAGES.md`). Ensure `zenfleet-worker` is public (GitHub → imazen packages → visibility →
Public) so fleet boxes pull credential-less. Rebuild with `scripts/jobsys/build_executor_image.sh` after
changing the executor. `scripts/jobsys/example_executor.py` remains a template if you'd rather write your own.

---

## 3. Prerequisites (one-time)

- **Built CLIs on the workstation:** `cargo build --release -p zenfleet-worker -p zenfleet-ctl`
  (gives `target/release/zenfleet-ctl`, `zenfleet-worker`, `zenfleet-gc`).
- **R2 root credentials** at `~/.config/cloudflare/r2-credentials` (used only on the workstation to
  mint scoped creds + upload the manifest — never shipped to a worker box).
- **The public worker image** `ghcr.io/imazen/zenfleet-worker:latest` (multi-arch amd64+arm64, built by
  `.github/workflows/jobworker-image.yml`). Pulls credential-less.
- For burst tiers: `hcloud` (Hetzner), `vastai`, or the Salad key at `~/.config/salad/credentials`.
- `aws` CLI v2 + `s5cmd` on the workstation (for manifest upload + ledger reads).

---

## 4. Declare the work

A spec is items × metrics. Each item is one (source image, codec, quality, knobs) cell plus the
`encode_sha` (content id of the encoded variant). `zenfleet-ctl declare` expands it into a manifest.

```bash
cat > /tmp/spec.json <<'JSON'
{ "items": [
    {"image_path":"corpus/img-00001.png","codec":"zenjpeg","q":80,"knob_tuple_json":"{}",
     "encode_sha":"<sha256 of the encoded variant>"},
    {"image_path":"corpus/img-00002.png","codec":"zenjpeg","q":80,"knob_tuple_json":"{}",
     "encode_sha":"<sha256>"}
  ],
  "metrics": ["cvvdp"] }
JSON

target/release/zenfleet-ctl declare --spec /tmp/spec.json --out /tmp/manifest.json
```

- Declaring is **idempotent** — re-declaring already-done work expands to the same `JobId`s, which the
  ledger reports as `done` (zero new work).
- `q5–q60` matter as much as `q60–q100` for web-focused codecs — sweep the low-q range with equal
  density (repo rule).

### 4b. Plan-driven sweeps: declare from the codec's planner

All five zen codecs own their sweep spaces (zenjpeg `zenjpeg::encode::sweep`, zenavif
`zenavif::sweep`, zenjxl `zenjxl::sweep`, zenwebp `zenwebp::sweep`, zenpng `zenpng::sweep` —
curated provenance-stamped axes, resolved-state fingerprint dedup, validity filtering,
main-effects-first ordering, budget ladder; zenpng is all-lossless and rides the q=0 sentinel).
Cross-codec contract + per-codec axis/scalar inventory: **`docs/PLAN_SWEEPS.md`**. There are
TWO ways to execute a plan (plus the vast.ai chunk-fleet path, which carries the same per-cell
identity in plan-mode input parquets — PLAN_SWEEPS.md §3); choosing the wrong one re-creates
the "100k AVIF encodes never finish" problem:

| | chunk mode | job-system mode |
|---|---|---|
| entry | `zenmetrics sweep --codec zenjpeg --plan rd_core\|modes_full` | `--plan … --dry-run --emit-cells cells.jsonl` → `zenfleet_ctl::declare_encodes` |
| unit of retry | (image × whole plan) | one cell (content-addressed `JobId`) |
| completion | chunk bookkeeping; a dead box redoes its chunk | declare → gap → run → re-reconcile; converges across any number of partial passes |
| right for | GPU-metric fleet runs that finish in one pass | big/expensive sweeps (AVIF-class) that will not |

Both carry the SAME per-cell identity — `{"cell":"<stratum-id>","fp":"<fingerprint>","plan":"<name>"}`
in the TSV/parquet `knob_tuple_json` column and in `JobKind::Encode.knobs` (hashed into the `JobId`).

```bash
# Emit the declare manifest (no encodes run; q must be integer-valued — CellId.q is i64):
zenmetrics sweep --codec zenjpeg --sources corpus/ --q-grid 5,10,...,95 \
  --plan modes_full --plan-budget 1824 --dry-run \
  --emit-cells /tmp/cells.jsonl --output /tmp/plan.tsv
# -> /tmp/plan.plan.json   (audit manifest: alias merges, invalid strata, budget drops)
# -> /tmp/cells.jsonl      (one EncodeDeclareItem per source × cell)
# Then declare: zenfleet_ctl::parse_emit_cells + declare_encodes -> DesiredJob[] -> gap -> manifest.
```

**The executor contract for a plan cell** (`zenmetrics jobexec` AND the sweep runner's
plan-identity tuple path, knobs JSON containing `"plan"`):

1. Parse `{cell, fp, plan}` from the cell's `knob_tuple_json`.
2. Reconstruct the config from the **self-describing** stratum id (per codec:
   `config_from_cell_id` for zenjpeg/zenavif, `variant_from_cell_id` for
   zenjxl/zenwebp/zenpng — lossless grammar, documented at each function); no plan spec,
   budget, or q-grid is needed to re-execute a stored job years later.
3. Verify `fingerprint(&config) == fp` (`sweep::plan::resolve_verified`). Mismatch = deterministic
   FAILED row — the id-grammar drift tripwire; never a silently wrong encode.
4. Encode; stdout = the encoded bytes (content-addressed to `blobs/<sha256>` by the worker).
   `metric` jobs on plan cells work the same way (they re-encode via the same path, then score).

The grammar is additive-only (see `config_from_cell_id` docs in zenjpeg); its roundtrip test
enforces that every id the planner emits parses back fingerprint-exact. The two documented
non-self-describing cases (`custom` table bytes, content-hashed boundary-RD knobs) are rejected at
declare time, not discovered at execute time.

---

## 5. Bring up the fleet

`scripts/jobsys/launch_fleet.sh` brings up any mix of tiers on one queue. It mints scoped temp creds,
uploads the manifest, and (this is the trick) starts the run **paused**, brings every tier up, then
**resumes** so they race from a standing start — and gives each worker a **per-worker shuffled
manifest** so the lowest-latency node doesn't monopolize the lease-queue.

```bash
# args: N_JOBS  HETZNER_X86  VAST  HETZNER_ARM  SALAD          (SPENDS MONEY on the paid tiers)
bash scripts/jobsys/launch_fleet.sh 120 1 0 0 1   # local + Hetzner-x86 + Salad = 3 providers
```

This is exactly the configuration proven on 2026-05-30: ledger DONE rows
`{local: 69, hetzner: 27, salad: 24}` = 120, exactly-once, fast node pulling more. Drop any tier to
`0`; the local tier always runs.

For **real work** set `ZEN_EXEC` to your baked executor before launching (default is `/bin/cat`).

---

## 6. The Unraid basement tier  🏠

Your Unraid box (basement, behind NAT) is the **persistent, outbound-only** tier. Because workers are
pull-based it needs **no inbound ports / port-forward / tunnel** — only outbound HTTPS to R2.

**Step 1 — on the workstation**, mint a scoped credential and print the container command for a given
run (never puts the root key on the Unraid box):

```bash
bash scripts/jobsys/unraid_worker.sh <RUN> 7 cpu_heavy,cpu_light
#                                      run  ttl_days  capability(optional)
```

It mints a 7-day, prefix-scoped, object-read-write R2 credential, uploads a shuffled
`manifest-unraid.json`, and prints a ready-to-paste `docker run` plus the Unraid "Add Container"
fields.

**Step 2 — on the Unraid box**, either paste the `docker run` at a terminal, or in the Unraid GUI:
**Docker → Add Container** →
- **Repository:** `ghcr.io/imazen/zenfleet-worker:latest`
- **Network:** `bridge` (no published ports)
- **Restart policy:** `No` — the worker drains its share of the run, then exits cleanly. (It is
  *run-scoped*: it works one run's manifest and exits. To run another job, start it again with the new
  run's variables. A persistent always-on daemon that auto-discovers new runs is a future enhancement.)
- **Variables:** add each `-e KEY=VALUE` the helper printed (`AWS_*`, `ZEN_R2_ENDPOINT`, `ZEN_BUCKET`,
  `ZEN_RUN`, `ZEN_MANIFEST_URI`, `ZEN_PROVIDER=basement`, `ZEN_WORKER=unraid`, `ZEN_EXEC`,
  `ZEN_CONTROL_KEY`, optional `ZEN_CAPABILITY`).

**Step 3 — verify from the workstation:** `bash scripts/jobsys/fleet watch <RUN>` — you'll see
`provider=basement` rows appear alongside the other tiers.

**Persistent credential (optional):** to avoid re-minting weekly, create a long-lived R2 API token in
the Cloudflare dashboard (R2 → Manage API Tokens → *Object Read & Write*, scoped to the bucket) and
set `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` from it (drop `AWS_SESSION_TOKEN`). Still never the
root key.

**Real executor on Unraid:** the stock public image only carries the synthetic `/bin/cat` path. For
real encode/score work, bake your executor into a worker image (FROM `ghcr.io/imazen/zenfleet-worker`,
`COPY` your executor in), push it, and point `ZEN_WORKER_IMAGE` / the container's repository at it with
`ZEN_EXEC=/path/to/executor`.

---

## 7. Monitor

- **Dashboard (no SSH):** the Railway control plane at
  `https://zenfleet-dash-production.up.railway.app` (Basic-Auth). Live fleet per worker, progress per
  kind, cost, failure drill-down, result peek, GC dry-run, pause/drain/resume, Kill.
- **CLI:** `bash scripts/jobsys/fleet watch <RUN>` — claims taken + ledger DONE rows grouped by
  provider (proves which tiers are concurrently working the queue).
- **Coverage:** `target/release/zenfleet-ctl catalog --manifest /tmp/manifest.json --ledger <ledger.parquet>`
  — done/poison/gap per codec×metric.

---

## 8. Get results

- **Scores / outputs** are the **content-addressed blobs** at `s3://$BUCKET/$RUN/blobs/<sha256>` and
  the **ledger** Parquet rows (`s3://$BUCKET/$RUN/ledger/*.parquet`) carrying `(job_id, provider,
  status, output_sha, …)`. Read the ledger with pyarrow, or peek a blob in the dashboard.
- **Coverage in one query** (`zenfleet-ctl catalog`) tells you exactly which cells are done; `gap` emits
  the not-yet-done subset so you only ever enqueue the remainder (idempotent by construction).

---

## 9. Tear down & garbage-collect

```bash
bash scripts/jobsys/teardown_fleet.sh <RUN>   # deletes every paid box for this run, by label; Salad group; local container
```
- Or use the dashboard **Kill** (same label selector) / **Stop-spend** (hard budget cap auto-tears-down
  paid tiers, free keeps draining).
- **GC** with `zenfleet-gc` (dry-run by default; `--execute` to delete). Reachability GC refuses to delete
  referenced blobs and refuses to auto-delete unreferenced *irreplaceable* blobs (it surfaces them);
  cheap-regenerable blobs are an LRU-capped cache. Always preview the dry-run first.

**Teardown verification (do every time):** `hcloud server list -l group=<RUN>` empty, Salad group gone,
no stray containers — and your own `zen-arm-dev` / unrelated boxes untouched.

---

## 10. Complete worked example (synthetic — runnable today)

This runs the whole pipeline end-to-end against real R2 with the synthetic executor, costing a handful
of tiny R2 objects. It is the honest "it works" demo; swap `ZEN_EXEC` for a real executor to do real
work.

```bash
cargo build --release -p zenfleet-worker -p zenfleet-ctl
bash scripts/jobsys/demo_e2e_r2.sh          # declare→gap 4→0, converge, coverage, blobs+ledger+lease
bash scripts/jobsys/launch_fleet.sh 120 1 0 0 1   # 3 real providers concurrent (paid: Hetzner+Salad)
bash scripts/jobsys/unraid_worker.sh <RUN>        # add the basement tier to that run
bash scripts/jobsys/fleet watch <RUN>          # watch DONE rows by provider
bash scripts/jobsys/teardown_fleet.sh <RUN>       # tear it all down
```

---

## 11. Checklist to run YOUR real job

1. **Corpus in R2 (read-only)** — corpus lives in `codec-corpus` (read-only); set `ZEN_CORPUS_BUCKET`
   + `ZEN_CORPUS_PREFIX` so `jobexec` resolves each `cell.image_path` to
   `s3://$ZEN_CORPUS_BUCKET/$ZEN_CORPUS_PREFIX/<image_path>` (falls back to `$ZEN_BUCKET` if
   `ZEN_CORPUS_BUCKET` unset; or use `s3://…`/local paths). Run output writes to the run bucket
   (`zentrain`), NOT the corpus — see the bucket roles in CLAUDE.md. `jobexec` fetches with s5cmd
   (read-only corpus cred `ZEN_CORPUS_AWS_*` when corpus ≠ run bucket).
2. **Executor + image — already built.** `zenmetrics jobexec` does real encode+score (§2), baked into
   the canonical `ghcr.io/imazen/zenfleet-worker:exec` (source `scripts/jobsys/fleet.env` →
   `$ZEN_FLEET_IMAGE_CPU`). Ensure that package is public so fleet boxes pull it. (Only write your own
   via `example_executor.py` + `build_executor_image.sh` if you need a codec/metric `jobexec` lacks.)
3. **Declare** the real spec (§4) — `items` of `(image_path, codec, q, knob_tuple_json, encode_sha)` ×
   `metrics` — and check coverage (`catalog`); enqueue only the gap.
4. **Launch** with the real image: `ZEN_WORKER_IMAGE=ghcr.io/imazen/zenfleet-worker:exec` (or just let
   `launch_fleet.sh` / `unraid_worker.sh` use `$ZEN_FLEET_IMAGE_CPU` from `fleet.env`) +
   `ZEN_CORPUS_BUCKET=codec-corpus ZEN_CORPUS_PREFIX=<your corpus prefix>` on `launch_fleet.sh` (§5)
   and `unraid_worker.sh` for the basement tier (§6). `ZEN_EXEC` defaults to the real executor.
5. **Monitor** (§7), **collect** scores/encodes from the ledger/blobs (§8), **tear down + GC** (§9).

## Planned: ~5-min chunking + async IO + resource bounds (user 2026-06-27)

Goal: combine cells into ~5-minute work-stealing claim units; make worker IO
async; bound peak memory to ≤75% of machine RAM and parallelism to ≤cores (no
oversubscription → no cache thrash). Land incrementally; each chunk CI-green;
smoke on a freed box before flipping any default.

- **Chunk 1 — DONE (`zenfleet-core` `00d1b39e`).** `schedule::BoxBudget::pack_chunks(&[JobCost], target_wall_sec)`
  groups cells into claim units sized to the box's envelope: chunk wall ≈
  `Σcost_sec / concurrency` (heaviest cell binds via `max_concurrent`, so heavy
  chunks auto-size small and never OOM). Pure + 5 unit tests. `JobCost {cost_sec,
  peak_mem_bytes, threads}`.

- **Chunk 2 — worker concurrency-under-budget + ~5-min chunk-claim — DONE
  (`zenfleet-worker` `e17962ef`, opt-in / default-OFF).** `ZEN_CHUNK_WALL_SEC > 0`
  switches `run()` (early-return → `run_chunked`) to `execute_gap_chunked`: the
  reconciler's gap is packed by `pack_chunks` into ~`ZEN_CHUNK_WALL_SEC`-second
  units, **one R2 lease per chunk** (`claim_or_steal_r2_key`, the new string-keyed
  core of `claim_or_steal_r2`; chunk-id = `chunk-`+sha256 of member job-ids, so the
  claim is exclusive). A won chunk's cells run as **fresh processes** (one-shot
  `exec_command` → keeps the modes_full per-cell memory bound — see Known Bugs)
  under `BoxBudget::can_admit` (Σpeak_mem ≤ 0.75×`/proc/meminfo` MemTotal, Σthreads
  ≤ `available_parallelism`) via `std::thread::scope` + a shared-cursor/`InFlight`
  condvar admission loop (no tokio — that's chunk 3). Per-cell `cost_sec` =
  `JobKind::estimate_cost_sec` (rough resource-class × peak-mem proxy, floor 1s;
  refine from measured omni `encode_ms`); a missing `ResourceHint` falls back to
  512 MB / 1 thread. **Default 0.0 ⇒ byte-identical per-cell path when unset.**
  Idempotence/crash-recovery preserved: chunks are formed FROM the gap (the per-cell
  done-check still gates every cell), and a **durable per-chunk ledger sidecar** is
  written the moment a chunk finishes (`chunk_ledger_uri` inserts `chunk-<id8>`
  before `.parquet`), so a crash re-runs only the still-missing cells; the next pass
  folds the sidecars in and skips the now-Done cells.
  **Deliberately worker-side ONLY** — `declare_encodes` is UNCHANGED (the ledger
  stays per-cell content-addressed; chunking is purely a claim-level optimization,
  NOT a declare-side manifest change). Tests: zenfleet-core +2 (`estimate_cost_sec`),
  zenfleet-worker +6 (chunk-id stability, meminfo parse, sidecar-uri, host budget,
  in-process pack+run+idempotence, re-claim-skips-done, run() end-to-end chunked
  convergence over per-chunk sidecars). Follow-up: spot-fast-release of a chunk claim
  on SIGTERM (TTL stale-reclaim covers preemption today).

- **Chunk 3 — async IO.** Worker IO is all blocking `Command` (`aws s3api`/`s5cmd`):
  claim `crates/zenfleet-worker/src/lib.rs:181-199`, blob put/get `:128-150`,
  jobexec fetch `crates/zenmetrics-cli/src/.../jobexec.rs:124-187`, exec
  `:560-590`. Wrap in a tokio runtime + `JoinSet`; overlap a chunk's source
  fetches with the prior cell's encode/score/upload (prefetch the next cell while
  the current scores). No tokio in the worker today — introduce it scoped to the
  claim/exec loop.

Map provenance: Explore agent trace 2026-06-27 (file:line above). `BoxBudget`
admission logic (`can_admit`/`max_concurrent`/`recommend_concurrency`) already
exists + is tested but is only called by `provision.rs` (fleet sizing), never the
worker — chunk 2 wires it into execution.
