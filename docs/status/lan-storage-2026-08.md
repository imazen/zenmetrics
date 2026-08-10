# LAN object storage migration — status (2026-08)

**Goal (user directive 2026-08-08):** migrate the job system's object-store operations off
cloud storage onto a LAN store, with the NAS as the storage host. The cost driver is
OPERATIONS (~$200/mo — claim leases, per-cell sidecars, ledger snapshots, heartbeats:
millions of small PUT/GET/LIST per sweep), not stored bytes; egress is free, so archiving
out is ~free and the win is killing the ongoing op stream.

Store specifics (endpoint value, host, data paths, credentials) live in the private
homefleet repo (`zenmetrics/tower/LANSTORE.md`) and `~/.config/zen/lanstore.env`. Public
repos use only `ZEN_S3_ENDPOINT` and neutral terms — see `docs/RUNNING_JOBS.md` §2b.

## State: LIVE — store up, plumbing landed, job system proven end-to-end on LAN

| milestone | status |
|---|---|
| Store container up on the NAS host (pinned MinIO, capped CPU/mem, NVMe-backed) | DONE 2026-08-08 |
| 3 buckets mirroring cloud names 1:1 (`zentrain`/`codec-corpus`/`zenfuzz`) | DONE |
| Conditional-write gate (`PUT If-None-Match: *` race = exactly one winner) | **PASS** (sequential 200→412; concurrent race: 1 winner, 1 `PreconditionFailed`) |
| `ZEN_S3_ENDPOINT` plumbing (resolver lib + 45 operator scripts + docs) | DONE — `84ff5158`, `b5318411`, `8c2fc367`, `4185b098` |
| Worker in-proc client http-endpoint fix + demo repair | DONE — `55f8a339` |
| Full job-system E2E on the LAN store (declare→lease→exec→blobs→ledger→coverage→idempotent restart) | **PASS** (`demo_e2e_r2.sh` rc=0 on both stores) |
| Worker claim from a LAN worker box → writeback → operator ledger read | **PASS** (6/6 done, chunked default path) |

## Measured: LAN vs cloud store, same operations, same tools

Operator box → store, s5cmd (256-way) bursts of 1000 × 120 B objects, plus 10 sequential
single-object PUTs (the claim/heartbeat op shape). 2026-08-08:

| op | LAN store | R2 | ratio |
|---|---|---|---|
| 1000-object PUT burst | **0.62 s** | 1.63 s | 2.6× |
| 1000-object GET burst | **0.37 s** | 1.82 s | 4.9× |
| sequential single PUT | **~41 ms** | ~370 ms | 9.0× |
| full E2E demo (4 jobs, 2 worker passes, coverage) | **2.95 s** | 6.86 s | 2.3× |
| worker box: 200-object PUT burst | 0.16 s | — | — |
| worker box: 6-job claim+exec+writeback (chunked path) | 0.35 s | — | — |

Marginal op cost on the LAN store: $0. Latency-bound paths (claims, heartbeats,
snapshot cadences) gain ~9×; concurrency-heavy bursts gain 2.6-4.9×.

## What changed in this repo

- `scripts/lib/s3env.sh` + `scripts/lib/zen_s3env.py` — THE endpoint+creds resolver.
  `ZEN_S3_ENDPOINT` unset ⇒ R2 exactly as before (byte-identical behavior); set ⇒ that
  endpoint + `ZEN_S3_ACCESS_KEY_ID`/`ZEN_S3_SECRET_ACCESS_KEY` (env or `ZEN_S3_ENV` file).
- Standing entry points source the lib: `refresh_snapshots.sh`+`compact_ledgers.py`,
  `launch_fleet.sh` (LAN mode: static store cred, no CF mint, **cloud tiers refused
  loud**), `demo_e2e_r2.sh`, `writeback_scores.py`.
- 39 more operator-side scripts honor the override mechanically. Deliberately pinned to
  R2: cloud launchers (`pool_launch.sh` fails loud under `ZEN_S3_ENDPOINT`; `gpu_*`/
  `hetzner_*`/`loo_fleet`/`launch_backfill`/`hqfill_A_remote`/`backfill_overnight_manager`
  ignore it) — a LAN store is unreachable from cloud boxes, and a stray operator env must
  never strand a paid fleet. Box-side worker scripts are untouched (env by injection).
- `zenfleet-worker` s3io: `allow_http` for plain-http endpoints (https unaffected). Blob
  puts previously failed `upload_fail` against any http endpoint while the aws-CLI claim
  path worked.
- `demo_e2e_r2.sh` pins `ZEN_CHUNK_WALL_SEC=0`: its coverage/gap assertions read the exact
  `--ledger-out` object, which only the serial path writes (pre-existing breakage since
  chunk-default claiming landed; repaired for both stores).

## Cutover rules (unchanged from RUNNING_JOBS §2b)

- **In-flight cloud waves finish on R2 untouched** (their creds/endpoints are injected per
  run; default-unset env cannot affect them). Cutover applies to NEW declares only.
- **LAN-resident fleets** (household worker boxes + local) declare with `ZEN_S3_ENDPOINT`
  set. **Cloud-burst fleets stay on R2** — no inbound path to the LAN, by design.
- Mixed store on ONE run is unsupported: one queue/ledger endpoint per run.

## Measured op bill (API-measured 2026-08-10) — what the migration is actually chasing

Method: mint an Analytics-Read token with `CF_TOKEN_MINT_TOKEN`, query GraphQL
`r2OperationsAdaptiveGroups` (dimensions `date|datetimeHour|datetimeMinute` × `actionType`
× `bucketName`). Per-minute granularity distinguishes a cron (bursts at :00/:30) from a
daemon (smooth drip) — that alone identified every writer. Analytics lag ~10-15 min.

14 days to 2026-08-10 — Class A was **15.6M ops, 98.2% of it PutObject**:

| bucket → op | 14d | share of Class A |
|---|---:|---|
| **zenfuzz → PutObject** | 10,839,370 | **69%** |
| zentrain → PutObject | 4,481,300 | 29% |
| zenfuzz → ListObjects | 201,230 | 1% |
| zentrain → ListObjects | 82,190 | <1% |

Monthly at that rate: Class A ~$151, Class B ~$55 (71M Get/Head), storage ~$33 (2.2 TB) —
**~$239/mo**, consistent with the ~$200/mo estimate this migration was scoped against.

**The headline finding inverts part of the migration premise.** The single biggest op
stream was NOT the job system — it was the fuzz farm re-uploading its entire corpus every
target visit, because `s5cmd sync`'s default compares size AND mtime and the preceding
`cp` pull stamps every local file `mtime=now`. Fixed in place with `--size-only` on both
directions (`~/work/zenfuzz-farm/fuzz-rotate.sh`, deployed 2026-08-10); verified ~99%
reduction on a scratch prefix before deploy. Expected post-fix: **Class A ~$29/mo, total
~$104/mo** — i.e. ~$135/mo of the ~$239/mo bill was one bug, recovered without moving a
byte. Full writeup: `~/work/zenfuzz-farm/OPERATIONS-NOTES.md` (2026-08-10 section).

**Consequences for this migration:**
- The remaining R2 op bill is ~$29/mo Class A, of which the job system (`zentrain`) is the
  bulk. LAN cutover still wins on **latency** (~9× on claim/heartbeat paths, measured
  above) and marginal-op cost, but the *urgency* was the fuzz bug, and that is gone.
- **Storage, not ops, is now the binding constraint on moving fleet data to the NAS.**
  `zentrain` is **1,850 GB and growing +252 GB/14d (~18 GB/day)**. On 2026-08-10 the NAS
  had **501 GiB free on the NVMe tier** (`zenstore` itself is only 5 MB so far) and
  **1.8 TiB free on the array at 95% full**. A wholesale `zentrain` move does not fit
  today, and the ≥1 TiB-free floor makes it user-gated. The op-generating traffic (claims,
  ledgers, heartbeats, sidecars) is tiny in bytes and moves freely — keep bulk artifacts
  and the byte-heavy prefixes off the NAS until the array is relieved.
- `zenfuzz` is 0.4 GB across **556k objects** — the extreme object-count-to-bytes ratio is
  exactly why it dominated ops while being invisible in storage. Archive it as
  **tar-per-target**, never object-per-file, wherever it lands.

## Standing op generators — audit + retarget state

| generator | cadence / volume | store today | state |
|---|---|---|---|
| zenfleet claims/sidecars/heartbeats | per-wave, the dominant op stream | R2 | new LAN-resident declares go LAN (proven above) |
| `refresh_snapshots.sh` cron (:*/30) | live — compacting bf944 pool runs (verified 12:03Z today) | R2 | honors `ZEN_S3_ENDPOINT`; flips WITH its pools' next generation (a LAN snapshot for an R2 pool would be unreadable to its workers) |
| fuzz farm corpus sync (`fuzz-rotate.sh`, 2 cloud boxes) | ~288 pull/push cycles/day vs `zenfuzz` | R2 | **RESOLVED 2026-08-10 — it was a BUG, not a placement problem. Do not migrate it.** It was **69% of all Class A ops** (~900k PutObject/day, ~$1.6k/yr); a one-line `--size-only` fix cut it ~99% *in place*. See "Measured op bill" below. Stays on R2 (cloud-resident boxes, no inbound LAN path) |
| workstation `triage-crashes.sh` cron (:*/30) | crash-prefix LISTs vs `zenfuzz` | R2 | stays on R2. `zenfuzz` ListObjects is only ~15k/day (~1% of Class A) — not worth moving |

## Archive inventory + R2 rundown (REGISTERED; execution user-gated / deferred)

Bucket totals no longer need a full LIST (that op class is what we're paying for; a 2-min
attempt timed out previously) **and no longer need the dashboard**. The stored R2 tokens
lack analytics scope, but you can **mint one** with `CF_TOKEN_MINT_TOKEN` (permission group
`b89a480218d04ceb98b4fe57ca29dc1f` "Account Analytics Read") and read exact per-bucket
size/object-count/op-counts from the GraphQL API — see "Measured op bill" below. Totals in
that section are API-measured 2026-08-10; the (a)/(b)/(c) inventory below still uses
documented sizes + the NAS mirror state.

**(a) Already mirrored to the NAS array (no copy needed — verify sha spots, then eligible
for rundown):** `kadis700k` (+GPU variant canonicals), `kadis-720-2026-07-24`,
`kadis-924-2026-07-27`, `kadis-hdr-2026-07-13`, `fill4-6codec-2026-07-01`,
`synthetic-v2` tables+bitstream images (~38 GiB, 729,703 objects, R2-verified
2026-06-22), `canonical-2026-05-21`, `imazen-26-*`, `jxl-*` (hqfill/lossy/all),
`clean-picker-corpus`, `canonical-picker-2026-07-01-zensimA` — per
`/mnt/tower/output/` listing + `~/work/zen/DATA_PROVENANCE.md` triple-mirror records.

**(b) Needs one archive pass to the NAS** (rate-limited, nice'd, per-prefix `du` FIRST,
array must keep ≥1 TiB free; sha256 spot-verify ≥3 files per prefix): the ext-canonical
family (`ext720-canonical-2026-07-22` 871 MiB documented; `ext924-`/`ext944-` similar),
`canonical/2026-06-27` picker datasets (locally mirrored at `/mnt/v` — copy from local,
not from R2), `eval-grids`, `tbig-720`/`tbig-944`, `v2-eval-*`, `zensim-compare-site`,
historical sweeps (`sweep-v1*`, `unified-2026-05-07`, `cvvdp-*`, `omni-*`,
`multi-codec-*`) and `zenfuzz/corpus+crashes` (small files; archive as tar-per-target,
not object-per-file). Sizing rule is mandatory before any copy: NVMe hot tier had 500 GiB
free and the array 2.0 TiB at 95% on 2026-08-08 — large prefixes go to the array tier
only, and anything that would leave <1 TiB free waits for the user.

**(c) Live-wave data — migrates only after the current waves drain:** everything under
`zentrain/jobs/` (213 run prefixes today, incl. the avifgen GPU-rescue queue and the HDR
score/diffmap waves), `hdr-corpora`, `hdrgrid-*`, plus the fuzz-farm live prefixes
(`zenfuzz/corpus`, `crashes`) while the farm still syncs against R2.

**R2 rundown proposal (NO deletions performed — user decision):**
1. After (a) sha-spot-verification: delete the (a) prefixes from R2. Est. saving: the
   bulk of stored bytes (tens of GiB each for kadis/synthetic-v2 families) — but note
   storage is NOT the main bill; this is hygiene + closing the account eventually.
2. After (b) archive passes complete + verify: delete those prefixes.
3. When the in-flight waves drain and their writebacks/views land (their outputs are
   consolidated to `/mnt/v` + NAS as part of each wave's terminal sequence): delete the
   drained `jobs/` runs. Claims/ledger sidecars of DRAINED runs are pure op residue —
   zero reuse value once the consolidated parquet exists.
4. The ops bill then decays to: fuzz-farm sync + any cloud-burst waves. Decide the fuzz
   hub question (above) on measured numbers; keep R2 only as the cloud-burst + offsite
   tier, or close it fully and provision per-campaign buckets when bursting.
5. Scratch from this migration: `zentrain/_ops-bench-20260808/` (1010 tiny objects) and
   `codec-corpus/jobsys-demo-20260808-1215{36,40}/` — delete with step 1.

## Follow-ups (registered)

- Rebuild + push `ghcr.io/imazen/zenfleet-worker` images so the baked worker carries the
  `allow_http` fix (until then, LAN fleets run the freshly-built binary or a locally
  built image; cloud fleets are unaffected).
- Bare-metal (non-docker) LAN workers need the aws CLI for the chunked claim path (the
  docker image bakes it; one worker box was missing it and silently `skipped` its whole
  gap — installed there now). Long-term: port the remaining aws-CLI claim/read calls to
  the in-proc client so the binary is dependency-free.
- One worker box's env file is pending (SSH host key rotated by a live wave's
  reprovision; write `~/.config/zen/lanstore.env` there when that wave drains).
- Wire `ZEN_SNAP_RUNLIST`-driven snapshot refresh to the LAN store together with the
  first LAN-resident pool generation.
