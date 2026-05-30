# Running jobs on the zen job system

A practical, end-to-end guide to declaring work, bringing up a heterogeneous fleet (including your
**Unraid basement box**), running it, reading results, and tearing down. It is honest about what is
production-proven versus what you supply (the **executor** for real encode/score work).

---

## 1. The mental model — one queue, many interchangeable workers

```
  you ──declare──▶  manifest (DesiredJob[])  ──▶  R2 lease-queue  ◀── workers claim & execute
   (zen-jobctl)         ▲                          (one bucket)         (local / basement / burst)
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

### Job kinds (`zen_job_core::JobKind`)
`Encode {codec,q,knobs}` · `Metric {metric}` · `Feature {regime}` · `Diffmap {metric}` ·
`Resample {kernel,w,h}` · `Bake {view}`. Each has a `profile()` giving its **resource class**
(`CpuLight/CpuHeavy/CpuArm/Gpu/HighRam` — for capability routing) and **GC regenerability**
(expensive encodes are kept; cheap re-scores are an LRU cache).

---

## 2. The executor contract  ⚠️ read this before running real work

The orchestration (declare → queue → claim → content-address → ledger → coverage → fleet → GC) is
**production-proven**. The thing that does the *actual* encode/score is a separate program you point
the worker at with `ZEN_EXEC`. The worker speaks one tiny contract per job
(`crates/zen-jobworker/src/lib.rs::exec_command`):

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

**What you supply for real work:** any program honoring the stdin-JSON → stdout-bytes contract, **baked
into a worker image**. `scripts/jobsys/example_executor.py` is a working template — branch on
`kind`, fetch inputs from R2 by sha, shell out to your codec/scorer, write the result bytes. A
reference real executor (`zen-metrics jobexec`) is specified by this contract but **not yet
implemented** — don't assume real codec/metric execution runs out of the box; you wire it.

> If you tell me your real job (codec/metric + where your corpus lives in R2), I'll implement the
> matching executor and bake a real-executor image so steps 4–8 below run your actual workload.

---

## 3. Prerequisites (one-time)

- **Built CLIs on the workstation:** `cargo build --release -p zen-jobworker -p zen-jobctl`
  (gives `target/release/zen-jobctl`, `zen-jobworker`, `zen-jobgc`).
- **R2 root credentials** at `~/.config/cloudflare/r2-credentials` (used only on the workstation to
  mint scoped creds + upload the manifest — never shipped to a worker box).
- **The public worker image** `ghcr.io/imazen/zen-jobworker:latest` (multi-arch amd64+arm64, built by
  `.github/workflows/jobworker-image.yml`). Pulls credential-less.
- For burst tiers: `hcloud` (Hetzner), `vastai`, or the Salad key at `~/.config/salad/credentials`.
- `aws` CLI v2 + `s5cmd` on the workstation (for manifest upload + ledger reads).

---

## 4. Declare the work

A spec is items × metrics. Each item is one (source image, codec, quality, knobs) cell plus the
`encode_sha` (content id of the encoded variant). `zen-jobctl declare` expands it into a manifest.

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

target/release/zen-jobctl declare --spec /tmp/spec.json --out /tmp/manifest.json
```

- Declaring is **idempotent** — re-declaring already-done work expands to the same `JobId`s, which the
  ledger reports as `done` (zero new work).
- `q5–q60` matter as much as `q60–q100` for web-focused codecs — sweep the low-q range with equal
  density (repo rule).

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
- **Repository:** `ghcr.io/imazen/zen-jobworker:latest`
- **Network:** `bridge` (no published ports)
- **Restart policy:** `No` — the worker drains its share of the run, then exits cleanly. (It is
  *run-scoped*: it works one run's manifest and exits. To run another job, start it again with the new
  run's variables. A persistent always-on daemon that auto-discovers new runs is a future enhancement.)
- **Variables:** add each `-e KEY=VALUE` the helper printed (`AWS_*`, `ZEN_R2_ENDPOINT`, `ZEN_BUCKET`,
  `ZEN_RUN`, `ZEN_MANIFEST_URI`, `ZEN_PROVIDER=basement`, `ZEN_WORKER=unraid`, `ZEN_EXEC`,
  `ZEN_CONTROL_KEY`, optional `ZEN_CAPABILITY`).

**Step 3 — verify from the workstation:** `bash scripts/jobsys/watch_fleet.sh <RUN>` — you'll see
`provider=basement` rows appear alongside the other tiers.

**Persistent credential (optional):** to avoid re-minting weekly, create a long-lived R2 API token in
the Cloudflare dashboard (R2 → Manage API Tokens → *Object Read & Write*, scoped to the bucket) and
set `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` from it (drop `AWS_SESSION_TOKEN`). Still never the
root key.

**Real executor on Unraid:** the stock public image only carries the synthetic `/bin/cat` path. For
real encode/score work, bake your executor into a worker image (FROM `ghcr.io/imazen/zen-jobworker`,
`COPY` your executor in), push it, and point `ZEN_WORKER_IMAGE` / the container's repository at it with
`ZEN_EXEC=/path/to/executor`.

---

## 7. Monitor

- **Dashboard (no SSH):** the Railway control plane at
  `https://zen-jobdash-production.up.railway.app` (Basic-Auth). Live fleet per worker, progress per
  kind, cost, failure drill-down, result peek, GC dry-run, pause/drain/resume, Kill.
- **CLI:** `bash scripts/jobsys/watch_fleet.sh <RUN>` — claims taken + ledger DONE rows grouped by
  provider (proves which tiers are concurrently working the queue).
- **Coverage:** `target/release/zen-jobctl catalog --manifest /tmp/manifest.json --ledger <ledger.parquet>`
  — done/poison/gap per codec×metric.

---

## 8. Get results

- **Scores / outputs** are the **content-addressed blobs** at `s3://$BUCKET/$RUN/blobs/<sha256>` and
  the **ledger** Parquet rows (`s3://$BUCKET/$RUN/ledger/*.parquet`) carrying `(job_id, provider,
  status, output_sha, …)`. Read the ledger with pyarrow, or peek a blob in the dashboard.
- **Coverage in one query** (`zen-jobctl catalog`) tells you exactly which cells are done; `gap` emits
  the not-yet-done subset so you only ever enqueue the remainder (idempotent by construction).

---

## 9. Tear down & garbage-collect

```bash
bash scripts/jobsys/teardown_fleet.sh <RUN>   # deletes every paid box for this run, by label; Salad group; local container
```
- Or use the dashboard **Kill** (same label selector) / **Stop-spend** (hard budget cap auto-tears-down
  paid tiers, free keeps draining).
- **GC** with `zen-jobgc` (dry-run by default; `--execute` to delete). Reachability GC refuses to delete
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
cargo build --release -p zen-jobworker -p zen-jobctl
bash scripts/jobsys/demo_e2e_r2.sh          # declare→gap 4→0, converge, coverage, blobs+ledger+lease
bash scripts/jobsys/launch_fleet.sh 120 1 0 0 1   # 3 real providers concurrent (paid: Hetzner+Salad)
bash scripts/jobsys/unraid_worker.sh <RUN>        # add the basement tier to that run
bash scripts/jobsys/watch_fleet.sh <RUN>          # watch DONE rows by provider
bash scripts/jobsys/teardown_fleet.sh <RUN>       # tear it all down
```

---

## 11. Checklist to run YOUR real job

1. **Corpus in R2** — your source images uploaded under the bucket; know their paths/shas.
2. **Executor** — implement the contract (§2) for your codec/metric (start from
   `scripts/jobsys/example_executor.py`); test it locally:
   `echo '<DesiredJob json>' | ./your_executor` → bytes on stdout, exit 0.
3. **Real-executor image** — `FROM ghcr.io/imazen/zen-jobworker`, `COPY` your executor + any codec/
   scorer binaries in, push to a registry; set `ZEN_WORKER_IMAGE` to it and `ZEN_EXEC` to your executor.
4. **Declare** the real spec (§4) and check coverage (`catalog`) — enqueue only the gap.
5. **Launch** the fleet (§5) + add the Unraid basement tier (§6).
6. **Monitor** (§7), **collect** results from the ledger/blobs (§8), **tear down + GC** (§9).
