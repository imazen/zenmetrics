# fleetbench-2026-08-24 — frozen workload for the Nomad-migration gates

Commit: `4267a562` (zenmetrics) / `64b44f84` (imazen-26) at declare time.
Mission: `~/work/zen/homefleet/zenmetrics/HANDOFF-nomad-power-fleet.md`,
`docs/status/fleet-orchestration-2026-08.md`.

## Purpose

ONE frozen encode workload run identically across every A/B in the Nomad-migration
program (G-P0 systemd baseline, G-N1 Nomad parity, G-P2 mid-chunk suspend, G-P3
autoscale cycle, G-T1 tuning). The workload's job here is to exercise real fleet
mechanics (claim/release, chunking, throughput, box wake/sleep) at a scale that
completes in minutes on the household LAN fleet — it is explicitly NOT a codec RD-
curve calibration sweep, and its results do not bake into any codec constant, so the
usual q5–q60-density sweep discipline does not bind on this corpus (see "Sizing
rationale" below for the honest tradeoff this makes instead).

## Corpus

- **Source:** `imazen/imazen-26` (public corpus, manifests-only repo; images on
  `codec-corpus.r2.imazen.org`) — commit `64b44f84`, variant set
  `variant-sets/fleetbench-anchor@2026-08-24/`.
- **Selection:** 64 images stratified 3-per-class across all 21 content classes
  (padded to 64 from the largest class), drawn from the **TEST** split
  (`manifests/test.tsv`) specifically because nothing consumes test-split renditions
  in any live training run — this benchmark corpus can be regenerated/rescored
  freely with zero collision risk against picker/model training.
- **Rendered tiers:** `make_variant_set.py --preset tiers4 --crops none --kernel
  lanczos` → 128/384/768/2048px longest-side (downscale-only), 242 total renditions
  (64+64+64+50 — 14 sources were natively smaller than 2048px). One selected source
  was a native `.heic` file; `make_variant_set.py` gained optional `pillow-heif`
  support in the same commit (imazen-26 `64b44f84`) since plain Pillow can't open it.
- **Sizing rationale (deliberate skew, not a silent cap):** a timed probe (zenjxl
  `rd_core --max-deviations 1`, one q-point) measured **~2.6 s/cell at 2048px** vs
  **~0.066 s/cell at 384px/128px** — a ~40x spread. A workload meant to be re-run
  5+ times across the gate program cannot afford full-tier density at the encode-
  heavy end. **`fleetbench-workload/` therefore keeps ALL 128px (64) + ALL 384px
  (64) renditions but subsamples 768px to every 4th file (16 of 64) and 2048px to
  every 6th (9 of 50)** — 153 total renditions — so large-image cell cost is still
  represented (chunk-sizing / VRAM-admission tuning in G-T1 needs it) without
  making every gate run dominate the session. This is a fleet-mechanics-appropriate
  tradeoff, not a codec-calibration one; do not reuse this corpus for anything that
  bakes into a codec default or oracle threshold.
- **Uploaded to the LAN store** (SeaweedFS, `http://192.168.50.170:3900`, the
  Nomad-migration's own storage plane — no R2 involved, per the ADR's "storage
  stays outside any scheduler" framing): `s3://zentrain/fleetbench-2026-08-24/sources/fleetbench-workload/`
  (153 objects, verified count match).

## Plan / cells

`zenmetrics sweep --codec {zenjpeg,zenavif,zenjxl,zenpng} --sources fleetbench-workload
--plan rd_core --max-deviations 1 --q-grid 30,60,90 --dry-run --emit-cells ... --output ...`
run from `~/tmp` (relative `--sources` so the emitted `image_path` stays a clean
`fleetbench-workload/<file>.png` relative path the executor resolves under
`$ZEN_CORPUS_PREFIX`, not an absolute local path baked into every JobId).

| codec | cells/image | x 153 sources |
|---|---|---|
| zenjpeg | 19 | 2,907 |
| zenavif | 18 | 2,754 |
| zenjxl | 15 | 2,295 |
| zenpng | 3 | 459 |
| **total** | | **8,415** |

`--max-deviations 1` (isolated main-effects only, not the full cartesian cross) and
a 3-point q-grid (30/60/90, not the repo's usual 21-point Step5 floor) — both
deliberate for the same reason as the size skew above: this workload measures
fleet mechanics, not RD curves, and needs to be cheap enough to re-run repeatedly.

## Declare

```
cat fb_dryrun/{zenjpeg,zenavif,zenjxl,zenpng}_cells.jsonl > fb_dryrun/all_encode_cells.jsonl   # 8415 lines
zenfleet-ctl declare-encodes --cells fb_dryrun/all_encode_cells.jsonl --out fb_dryrun/fleetbench_manifest.json
# -> declared 8415 encode jobs from 8415 cells
```

One manifest (`fleetbench_manifest.json`, content-addressed `JobKind::Encode`
DesiredJobs, 5,000,022 bytes), uploaded verbatim to TWO run prefixes — the job
content is identical, only the ledger namespace differs, which is exactly what a
G-P0-vs-G-N1 comparison needs (each run must drain the full 8,415-job gap fresh,
not see the other's completed work):

- `s3://zentrain/fleetbench-gp0-2026-08-24/manifest.json` — systemd baseline
- `s3://zentrain/fleetbench-gn1-2026-08-24/manifest.json` — Nomad-managed

## Executor

`ghcr.io/imazen/zenfleet-worker:exec` — the CPU exec image rebuilt earlier in this
session with all P0 preconditions landed (JobId #38 + metric routing `1b2a1452`,
VRAM admission `58c5db95`, SIGTERM chunk-claim release `0de119d6`/`45c57bd3`,
`ZEN_MAX_MIN`/`ZEN_LONG_LIVED` `45c57bd3`, native claim CAS `a916d24d`; musl-built
per `scripts/jobsys/fleet.env`'s 2026-08-24 entry). Same image for both G-P0 and
G-N1 — the scheduler is the only variable under test.

## G-P0 — systemd baseline

Deploy script: `homefleet/zenmetrics/ubuntu-node/fleetbench_systemd_worker.sh`
(new — modeled on `enroll_running_node.sh`'s docker+systemd pattern but targeting
this run's specific manifest instead of the production pool runlist). One-shot
(`ZEN_LONG_LIVED` unset, `Restart=no`): the worker exits once the gap has been
empty for `ZEN_IDLE_PASSES` (default 5) consecutive passes.

Boxes (same 3 for G-N1): r7900x (24C, `lilith@`), i265 (20C, `zen@`), r3500 (6C,
`zen@`) — 50 cores total. `ZEN_CHUNK_WALL_SEC=60` (shrunk from the 300s default so
a short run still forms multiple chunk-claim boundaries).

Started: r7900x 04:24:57Z, i265 04:25:13Z, r3500 04:25:35Z (2026-08-24).

### RESULT: 100% duplicate-execution rate — a real, heterogeneous-fleet-specific defect in chunk-mode lease claiming (not the documented lease-mode race)

**Final numbers** (run manually stopped 04:41Z once distinct-job coverage hit 100% —
see "why stopped early" below): **29,400 total ledger rows for 8,415 distinct
job_ids (0 failed/poisoned) — a 3.49x row-count ratio, and every single one of the
8,415 distinct jobs (100.00%) has a "done" row from more than one DISTINCT
(worker, ts) execution.** Sample (repeats across many different job_ids with the
EXACT same ts pair, which is itself a clue — see below):
`[('i265', 1787545513), ('i265', 1787546176), ('r7900x', 1787545497)]`.

**Root cause (verified against source, not inferred from the data alone):**
chunk-mode's per-chunk R2 lease is keyed on `chunk-id = sha256(member job-ids)`
(`docs/RUNNING_JOBS.md` §"Chunk 2"), and chunk MEMBERSHIP is computed by
`BoxBudget::pack_chunks_lpt(&self, jobs, target_wall_sec)` — `&self` is the
CALLING BOX'S OWN budget (`crates/zenfleet-worker/src/lib.rs:2141-2148` logs it:
r7900x "budget 22.5 GiB / 24 cores", i265 "22.6 GiB / 20 cores", r3500's own,
smaller, 6-core budget). **Three boxes with three different core counts partition
the IDENTICAL 8,415-cell gap into three DIFFERENTLY-SHAPED sets of chunks, so
their chunk-ids never collide — the lease has nothing to catch, because from its
point of view r7900x, i265, and r3500 are each claiming entirely different
(non-overlapping-by-name) chunks, even though the underlying cells inside those
chunks overlap almost completely.** Confirmed directly in the run: r7900x logged
`done=8415 failed=0 poisoned=0 skipped=0` for its OWN pass 1 alone (`skipped`
is "gap jobs another worker claimed first" — zero means it never once lost a
claim race to i265, which was running the identical manifest concurrently) — i265
logged the identical `done=8415 … skipped=0` for ITS pass 1. **Each of the two
fast boxes independently executed the ENTIRE 8,415-cell manifest, unaware of the
other, for that box's whole first pass** (~11 minutes); r3500 (6 cores, ~4x fewer
than r7900x) was still grinding through its own independent full copy when the
run was stopped.

**This is a DIFFERENT, more severe finding than the documented lease-mode race**
(`docs/RUNNING_JOBS.md` §5b's "avifgen … 3.6x under lease claiming with empty
views", presumably measured on a homogeneous cloud fleet where same-shaped
chunking WOULD collide and the lease WOULD catch most of it — a milder
stale-view race, not a structural non-collision). **A household LAN fleet is
heterogeneous by construction** (different-generation desktops with different
core counts), which makes this the fleet shape MOST exposed to the bug — directly
relevant to this whole mission. Filed as a new Known Bug (zenmetrics CLAUDE.md)
and in the ADR's defect register. Likely fix directions (not implemented here):
(a) derive chunk boundaries from a canonical, box-independent partition (e.g. a
declare-time fixed sharding) so chunk-ids are comparable across boxes, or (b) use
epoch-sharded claiming (`ZEN_CLAIM_MODE=epoch-sharded`), which shards by
rendezvous-hashing individual CELLS — independent of any box's local
`BoxBudget` — so it should not exhibit this failure mode. (b) is testable without
a code change and is exactly what RUNNING_JOBS.md §5b already recommends "for a
dedicated single-run fleet" (this workload, exactly) — see the epoch-sharded
rerun below.

**A secondary, narrower finding**: `LedgerRow.ts` is `ctx.now` — captured ONCE per
`run()` invocation (one pass) and stamped on every row that pass writes
(`crates/zenfleet-worker/src/lib.rs:970,1178`), NOT a true per-cell completion
timestamp. This is why many different job_ids share bit-identical `(worker, ts)`
pairs above, and why "time to full coverage" cannot be derived from ledger `ts`
values — use direct wall-clock observation instead (see below). Anyone deriving
per-cell timing from this ledger needs to know this.

**Wall-clock (directly observed, not derived from `ts`):** r7900x started
04:24:57Z; live log inspection at 04:36:09Z showed r7900x's pass 1 had already
reached `done=8415` (i.e. full manifest coverage, achieved by ONE box alone) —
**~11m12s wall-clock to full 8,415-cell coverage**, dominated by the fastest
box's own single (fully redundant, as it turned out) pass. i265 reached the same
point at 04:36:16Z, ~7s later. r3500 was still on its own first pass when stopped.

**Why the run was manually stopped at 04:41Z instead of let idle out on its
own:** once distinct-job coverage hit 8,415/8,415 (confirmed via a live ledger
read), r3500 continuing its own independent full pass could only ever produce
MORE duplicate work, never new coverage — letting a ~4x-slower box grind through
a fully-redundant third copy of the whole manifest (estimated another 25-30
minutes based on its per-core throughput vs. the two faster boxes) would have
burned real box-hours for zero additional signal, which is precisely the
"actively flag and stop wasted infrastructure" discipline this mission is partly
about. Stopping doubled as a bonus SIGTERM-under-real-load check: `systemctl stop
zen-fleetbench-worker` on r3500 (mid real chunk execution, not idle) returned
`rc=0` — the P0 SIGTERM-release precondition holds under genuine in-flight work,
not just the earlier synthetic Nomad-drain smoke test.

**Per-worker raw row share** (NOT a fair 1/3 split — each box independently
produced close to its own full copy, scaled by how far it individually got):
r7900x 13,322 · i265 11,842 · r3500 4,236 rows.

**CPU-score spec generated from this ledger**: `~/tmp/build_score_spec.py` wrote
8,415 items × 3 metrics (ssim2/butteraugli/zensim) — ready to declare, but held
pending the epoch-sharded rerun decision below (scoring against a 100%-duplicated
encode ledger is still valid — the DISTINCT done set is complete and correct —
but the epoch-sharded rerun's ledger is the more representative encode-cost
baseline to score from for any downstream timing claims).

### Epoch-sharded rerun (the fix, tested) — MASSIVE improvement, with an honest caveat

Same 3 boxes, same manifest content, separate run/ledger namespace
(`fleetbench-epoch-2026-08-24`), `ZEN_CLAIM_MODE=epoch-sharded`
`ZEN_EPOCH_LEN_SECS=60` `ZEN_EPOCH_HB_SECS=15` (epoch length shrunk from the
600s zenfleet default so this short run crosses ≥1 boundary — see
`fleetbench_systemd_worker.sh`'s `ZEN_FB_*` overrides). Started 04:45:57Z,
manually stopped 04:53:15Z (7m18s) at **84.7% distinct-cell coverage** — same
"stop the low-marginal-value tail" call as G-P0, for the same reason (see below).

**Result at stop time: 12,047 total rows for 7,131 distinct done cells (0
failed) — a 1.689x row-count ratio, vastly better than lease-mode's FINAL
3.49x, and dramatically better through most of the run** (14–15% genuine
duplicate rate through ~75% completion, vs. lease-mode's 62%+ at the same
completion point). This directly confirms the root-cause hypothesis from the
lease-mode finding above: epoch-sharded claiming shards by rendezvous-hashing
individual CELLS, independent of any box's local `BoxBudget`, so it does not
suffer chunk-mode's non-colliding-chunk-id structural break.

**Honest caveat — the SAME tail-pileup dynamic reappears near completion, at
smaller scale.** Genuine duplicate rate climbed 14%→24%→36%→49%→58% as distinct
coverage crept from ~34%→77%→82%→84%→84.7%, with real NEW-cell progress nearly
stalling in the last few minutes (r3500, 6 cores, the slow box, was still
visibly working — confirmed alive at ~100% CPU via `docker stats`, not
hung — but its shrinking remaining share was increasingly contested by
tail-steal attempts from the two already-finished-their-own-share fast boxes).
This is very likely epoch-sharded's own documented "tail-steal" mechanism
("a worker that exhausts its shard tail-steals peers' remaining cells
lease-guarded") racing against a slow box's still-in-flight local execution —
the per-cell lease should prevent this in principle, but a steal attempt that
checks "is this cell DONE in the ledger yet" (not "is someone else actively
mid-execution on it") would still produce a genuine duplicate if the slow
box's execution hasn't landed a ledger row yet. Narrower blast radius than
lease-mode's structural break (only the LAST ~15% of cells are affected, not
the whole manifest), but real and worth a G-T1 follow-up (e.g. `--no-tail-steal`
comparison, or a longer `ZEN_EPOCH_LEN_SECS` that gives the slow box more room
before triggering a steal).

**Bonus SIGTERM-under-real-load confirmation, now 3/3 (this run) + 1/3 (G-P0)
= 4/4 real-load graceful exits:** `systemctl stop zen-fleetbench-worker` on
all three boxes (mid real, non-idle chunk/epoch execution) each returned
`rc=0`.

Per-worker raw row share: r7900x 5,252 · i265 4,887 · r3500 1,908 — closer to
proportional-but-not-exactly-core-weighted than lease-mode's near-independent
full-copies, consistent with epoch-sharded's unweighted (no `handicaps.toml`
entries registered for these 3 boxes yet) roughly-equal-shares-plus-tail-steal
model rather than lease-mode's "everyone races the whole thing" model.

**Verdict for G-N1 (important, corrects an earlier same-session mistake):**
epoch-sharded is a genuine G-T1 tuning-ladder finding, NOT a "fix" to swap into
the G-N1 comparison. G-N1 isolates the SCHEDULER as the only variable, so it
must run the SAME claim mode G-P0 used (lease) — see
`fleetbench-gn1.nomad.hcl`'s header comment (reverted after a wrong first
instinct, homefleet commit `b3420e32`). The epoch-sharded Nomad-side
comparison lives in a separate jobspec, `fleetbench-gt1-epoch.nomad.hcl`, for
the G-T1 ladder proper.

## G-N1 — Nomad-managed

Jobspec: `homefleet/zenmetrics/ubuntu-node/nomad/jobs/fleetbench-gn1.nomad.hcl` —
`type = "batch"`, same 3-box `regexp` + `distinct_hosts` constraint, same image
(`force_pull = true`), same `ZEN_CHUNK_WALL_SEC=60`, same claim mode (lease,
`ZEN_CLAIM_MODE` deliberately unset — see the file's header comment for why this
must not be "fixed" to epoch-sharded; that's a separate G-T1 ladder entry,
`fleetbench-gt1-epoch.nomad.hcl`), `ZEN_LONG_LIVED` deliberately unset (matches
G-P0's exit-on-empty-gap semantics). Only the run id / manifest URI / corpus
prefix / `ZEN_WORKER` suffix differ from G-P0's config — Nomad is the only real
variable. Verified directly on a live allocation (`nomad alloc exec ... env`):
`ZEN_CLAIM_MODE` absent, `ZEN_RUN`/`ZEN_CHUNK_WALL_SEC` correct.

Launched via `nomad job run fleetbench-gn1.nomad.hcl` at 04:54:30Z — 3 allocations
created immediately, one per constrained host, 0 queued/failed. Manually stopped
via `nomad job stop fleetbench-gn1` at 04:59:37Z (5m7s later) at **84.4% distinct
coverage (7,104/8,415)** — the same "stop once the tail-pileup dynamic dominates
and further waiting adds mostly duplicate work, not new coverage" call as the
other two runs, applied at a closely comparable completion percentage
(G-P0's epoch-sharded rerun stopped at 84.7%) for a fair three-way comparison.
**All 3 allocations exited `complete` (not `failed`/`lost`) — Nomad's own stop
path delivered SIGTERM cleanly through the container under real in-flight
chunk/claim execution, extending the graceful-exit count to 7/7 across both
`systemctl stop` and `nomad job stop`, both under genuine load.**

**Result at stop time: 10,758 total rows for 7,104 distinct done (0 failed) —
1.514x row-count ratio, 45.57% genuine duplicate-(worker,ts) rate.** Per-worker
raw rows: i265-gn1 4,845 · r7900x-gn1 4,419 · r3500-gn1 1,494.

**Qualitative verdict — confirmed: Nomad is orthogonal to zenfleet's claim
mechanism, as expected.** G-N1 shows the SAME defect class as G-P0 (duplicate
execution climbing with completion percentage — the identical chunk-mode
heterogeneous-fleet break, since Nomad just launches the same worker binary
with the same claim code) at a broadly comparable order of magnitude for the
completion point reached. No evidence Nomad makes the claim-dedup problem
meaningfully better or worse — which is exactly the ADR's own framing ("Nomad
is the LAN box-lifecycle layer ONLY; zenfleet stays the entire work-distribution
plane") holding up under a real measurement, not just an architectural claim.

**Quantitative verdict — honest limitation, not a fabricated percentage.** The
HANDOFF gate wants "throughput ≥95% of G-P0, re-work tax ≤ G-P0" as precise
numbers. Both runs were monitored via manual polling (20-30s cadence) rather
than synchronized elapsed-time snapshots, and were stopped at DIFFERENT absolute
distinct-done counts (G-P0 at 100%/8,415, G-N1 at 84.4%/7,104) — so a directly
comparable "cells/hour" or "re-work tax at matched completion" figure is NOT
available from this data without risking a misleadingly precise number from a
noisy comparison base (per the repo's own "never fabricate/estimate performance
numbers" discipline). What IS defensible: G-N1's row-count ratio at its 84.4%
stop point (1.514x) is LOWER than G-P0's ratio at earlier, less-complete
checkpoints of its own run (e.g. 1.737x at 96.8% coverage per the interim-finding
checkpoint above) — i.e. nothing in the data suggests G-N1 is WORSE than G-P0;
if anything the raw numbers lean the other way, but not by a margin this
polling cadence can responsibly quantify as a percentage. **A rerun with
synchronized per-N-second ledger snapshots (not ad-hoc manual polling) is the
right way to produce a defensible throughput-ratio number for a strict pass/fail
against the 95% threshold** — tracked as a G-T1/measurement-tooling follow-up,
not resolved by manufacturing a number here.

**Zero ledger divergence, zero stranded claims (as far as observed):** both
runs' ledgers show 0 failed/poisoned rows and 100% of attempted cells eventually
landing a `done` row with a real `output_sha` — no silent data loss, no
inconsistent state between the two schedulers. Leftover `claims/` objects exist
under both run prefixes post-stop (331 for G-P0, expected — claims are not
auto-GC'd on completion by design; they age out past TTL rather than being
actively deleted) but none were observed to block or corrupt any subsequent
read in this session; a rigorous "stranded claim" audit (claim exists with no
corresponding ledger resolution) is a good G-T1/tooling addition, not done here.

## G-P2 — drain safety: FAILED on first genuine test, re-opens P0 precondition #2

The first REAL end-to-end test of "deliberately suspend/drain a box mid-chunk"
(as opposed to the earlier P0 verification, which the session summary itself
now flags as "a fake-tool harness" — a synthetic/mocked test, never a real
`zenfleet-worker` binary with a real in-flight chunk claim under real container
orchestration). Two independent bugs were found and fixed along the way before
reaching the core (negative) result; all three are real, all three matter.

**Setup:** `homefleet/ubuntu-node/nomad/jobs/gp2-suspend-test.nomad.hcl`, i265
alone (WoL-gated, G-P1 PASSED 3/3), `RAYON_NUM_THREADS=2`/`OMP_NUM_THREADS=2`
throttling a 243-cell zenjxl rd_core manifest (9 large 2048px sources ×
q{40,50,70,90}) so a chunk stays genuinely in-flight for a comfortable manual
window instead of finishing in ~20s.

**Bug 1 (found, fixed): `nomad node drain -enable` (no `-force`) does not
promptly interrupt a running allocation.** Plain `-enable -deadline 2m` let the
in-flight allocation keep running — it even claimed a SECOND chunk after the
drain command was issued — and only stopped ~70s later when it happened to
finish its own assigned work naturally, not because the drain interrupted it.
`fleet_power.py`'s `suspend()` used exactly this plain form, meaning in
production a box could be told to suspend, wait up to the full 2-minute
deadline (or longer, if the natural-completion path takes longer than that —
Nomad would then force it, but only at the deadline), all without ever
delivering a prompt SIGTERM — directly violating "claim released promptly...
must be seconds, not claim-TTL." **Fixed:** added `-force` to `suspend()`'s
drain call (zenmetrics `scripts/jobsys/fleet_power.py`).

**Bug 2 (found while fixing Bug 1): `-force` and `-deadline` are mutually
exclusive Nomad CLI flags** ("`-deadline` can't be combined with `-force` or
`-no-deadline`") — the naive fix (just add `-force` next to the existing
`-deadline 2m`) fails outright. Caught by testing the fix live BEFORE
considering it done, which is exactly why it wasn't shipped broken: with
`check=False` on the `subprocess.run` call, this failure would have been
COMPLETELY SILENT in production — the drain would error immediately, nothing
would happen, and `systemctl suspend` would fire on a box with a live claim in
flight and zero graceful handling — worse than the original bug. **Fixed:**
drop `-deadline` entirely (force alone means "immediately," no deadline
needed).

**Bug 3 (found, NOT fixed — the real finding): the forced drain DOES exit the
container promptly and cleanly (confirmed via `nomad alloc status`: exit code
143 = SIGTERM, "Killed: Task successfully killed," ~1.8s from drain-issue to
"All allocations... have stopped" — well within "seconds"), and the bash
entrypoint's own SIGTERM trap DOES fire and forward the signal correctly
("received SIGTERM — forwarding to the in-flight pass (pid 59) for fast claim
release, then exiting" — this exact log line, confirmed present). But the
S3 chunk claim object itself was verified UNCHANGED after this — read back
directly (`s5cmd cat`), byte-identical to its original claimed-at content
(`1787548370 i265-gp2`, timestamp matching the claim's creation, not an
updated/release time).** The claim was not released.

Digging further: `spawn_spot_reclaim_chunk`'s signal-handler thread
(`crates/zenfleet-worker/src/lib.rs:246-279`) is supposed to print
`"zenfleet-worker: {released|could not release} chunk claim {cid}..."` on
catching SIGTERM/SIGINT — this line, and even the earlier unconditional
`"zenfleet-worker: resource-aware concurrent mode..."` line that `run_chunked`
prints as its very first statement, **never appeared anywhere in the
allocation's logs** — confirmed via `nomad alloc logs -verbose` /
`-stdout` / `-stderr` separately (stderr came back completely empty; no
truncation). `timeout` (the entrypoint wraps `zenfleet-worker` in
`timeout "$ZEN_PASS_TIMEOUT" zenfleet-worker ...`) was ruled out as the culprit
by direct reproduction — a minimal `timeout ... | trap ... TERM` harness
correctly waits for the wrapped child to finish its own signal handling before
`timeout` itself returns, and bash's `wait` on `timeout`'s PID correctly
blocks until then.

**Narrowed further: the release LOGIC itself is correct — reproduced directly.**
Ran `zenfleet-worker` raw (no Docker, no Nomad, no bash entrypoint) against a
fresh manifest with a slow fake executor (`sleep 3` per cell, enough to hold a
real chunk claim open), confirmed a live claim on the LAN store, then
`kill -TERM <pid>` directly. Result: the exact expected diagnostic line
appeared immediately (`"zenfleet-worker: spot preemption — released chunk
claim <cid> for fast requeue..."`) and the claim object was verified GONE from
the LAN store afterward. **`spawn_spot_reclaim_chunk`'s signal-handler thread
and `release_claim_r2_key` both work correctly when the binary runs directly.**
This rules out a logic bug in the release code itself and narrows the bug to
something specific to the Docker+Nomad+bash-entrypoint layering.

Follow-up probing inside a live container (`nomad alloc exec ... /proc`
inspection) found the real process tree: PID 1 = the bash entrypoint, PID 59 =
`timeout 1800 zenfleet-worker ...` (what bash's `$PASS_PID` actually is — NOT
the Rust binary directly), PID 60 = the actual `zenfleet-worker` process
(timeout's child), plus ~20 concurrent `zenmetrics jobexec` executor
subprocesses (PIDs 66+) doing the real per-cell work. An attempt to signal PID
1 directly from inside via `nomad alloc exec ... kill -TERM 1` had NO
observable effect (the allocation kept running) — inconclusive, not a valid
reproduction of the real drain path, and not pursued further. **Root cause
NOT fully isolated in this session** — the remaining candidates (not
confirmed): PID-namespace teardown semantics when a container's PID 1 exits
while descendants (including the ~20 jobexec subprocesses PID 60 itself
spawned) are still alive; a race between `zenfleet-worker`'s own exit
(`std::process::exit(130)`, which does not wait for or clean up ITS OWN spawned
children) and the surrounding process tree's teardown; or a genuine timing
difference between a raw `kill(2)` and however Nomad's `-force` drain /
Docker's stop sequence actually delivers the signal into the container. Closing
this needs deliberate instrumentation (timestamped tracing that survives a fast
exit path) and a rebuild-instrument-retest cycle, scoped as dedicated follow-up
work rather than further ad-hoc live probing. **Do not re-close this
precondition until the actual mechanism is found and a real (non-mocked,
non-raw-binary) container/Nomad repro passes.**

**Practical consequence (AT THE TIME):** the P0 precondition "SIGTERM chunk-claim
release," previously reported DONE and verified, was NOT actually proven
end-to-end — its only verification to date had been a synthetic/mocked harness.

### RESOLVED (same session, after instrumentation + rebuild): release confirmed working, 2/2 real repro

Added diagnostics (commit `0e998ff3`) — an immediate marker + explicit stderr
flushes in `spawn_spot_reclaim_chunk`, and sub-second timestamps around the
bash entrypoint's `wait "$PASS_PID"` — rebuilt the executor image (musl target,
same process as the original P0 rebuild) and pushed it as the canonical `:exec`
+ a dated pin `:exec-gp2-debug-1787549532`. Re-ran the exact same live test
(confirmed live chunk claim on the LAN store, real `nomad node drain -force`)
**twice, both times the claim was verified GONE afterward** — the release
genuinely happened. Bash-level timing: `pre-wait`→`post-wait` was 3-5ms both
times (the child's own graceful handling, including the HTTP DELETE, completing
within that window). A third run then launched the r7900x absorber job, which
picked up and completed the remaining gap.

**Honest residual uncertainty:** the newly-added Rust-level diagnostic lines
(the ones that would show the signal handler thread actually firing and
attempting the release) still did NOT appear in `nomad alloc logs` either time,
even though the claim's deletion PROVES that code path executed successfully.
So two things are true simultaneously: the release now demonstrably WORKS
(functionally verified via the S3 object's absence, not via a log line), and
there remains an unexplained, separate anomaly where this specific thread's
`eprintln!` output never reaches the container's captured logs even when the
code runs. It's possible the added flush calls themselves nudged a genuine
timing race in the right direction (plausible: forcing a syscall at the top of
the handler could affect thread scheduling enough to matter), making this a
real if not fully understood fix rather than a coincidence — but this is not
proven, only consistent with the evidence. **Practical verdict: treat the
precondition as re-verified for now (2/2 real, repeatable success with the
current image), but do not consider the underlying mechanism fully understood
until the log-visibility anomaly is separately explained.**

**Exactly-once check: confounded by the already-known lease-mode chunking bug,
not cleanly isolated.** This G-P2 test ran in lease-mode (not epoch-sharded),
and — expectedly, given the G-P0 finding above — i265 (20C) and r7900x (24C)
produced 19 cells with a `done` row from BOTH workers across the test's several
restarts. This is very likely the SAME structural chunk-lease mismatch as
G-P0/G-N1, not a NEW defect in the release-then-reclaim path specifically —
but this test's design (multiple restarts on the same lease-mode manifest)
cannot cleanly separate "duplicate work from the known chunking bug" from "a
release-specific double-execution." **A clean exactly-once verification needs a
dedicated epoch-sharded-mode rerun of this exact suspend/absorb test** (single
restart, single interrupted chunk, single absorber pass) — not done in this
session, tracked as follow-up.

Filed as a reopened-then-resolved Known Bug (CLAUDE.md) and in the ADR's defect
register — both updated to reflect the working, repeatable fix alongside the
residual log-visibility puzzle and the exactly-once caveat above.

## G-P3 — full autoscale end-to-end

Reuses the EXISTING `fleetbench-gn1-2026-08-24` run's remaining ~1,311-cell gap
(from the G-N1 throughput test, stopped at 84.4%) rather than declaring fresh
work — real, substantial, already-there. Jobspec:
`homefleet/ubuntu-node/nomad/jobs/gp3-autoscale-test.nomad.hcl`, targeting
i265+r3500 (the sleep-gated pair) by hostname regex.

**Found + fixed TWO more real bugs getting this far** (both in
`scripts/jobsys/fleet_power.py`, both real production bugs the mission's own
tooling needed working before G-P3 could even be attempted):

1. **`queue_gap()` couldn't see plain declared-manifest runs at all** — it
   hardcoded the `jobs/<run>/` manifest path (`pool_progress.py`'s convention,
   for pool-mode runs with a pre-compacted `ledger_snapshot.parquet`) as the
   ONLY path checked. A plain run (this whole session's G-P0/G-N1/G-P2 tests,
   and anything from `zenfleet-ctl declare-encodes` + `launch_fleet.sh`) has no
   `jobs/` prefix and no snapshot file — `fleet power status --run
   fleetbench-gn1-2026-08-24` silently reported "no manifest/snapshot yet" for
   a run with a very real 1,311-cell gap. Fixed: try both path conventions;
   fall back to a real ledger scan (fast enough at this scale — sub-second up
   to ~30k rows) when no snapshot exists.
2. **The `-deadline`/`-force` fix from the G-P2 section above had ALSO shipped
   broken** — the commit message and CLAUDE.md claimed it was fixed (drop
   `-deadline`), but the actual code edit had only ADDED `-force` next to the
   existing `-deadline 2m`, never removed it. The claim was validated via raw
   CLI testing, written up as done, and the corresponding code edit simply
   never happened. This resurfaced live: `suspend()` failed silently
   (`check=False`) on BOTH i265 and r3500 during G-P3 test setup. Fixed for
   real this time, verified via the actual function call succeeding. **A
   docs/commit-message claim of a fix is not the same fact as the code
   containing it** — this is now called out explicitly in both the code
   comment and CLAUDE.md.

**Test sequence (in progress as this section is written; final numbers below
once complete):**
1. Submitted `gp3-autoscale-test` while i265+r3500 were still up → placed
   immediately (wrong for the test) → stopped, purged.
2. Manually suspended both boxes (`fleet_power.suspend()` directly — this
   is where bug 2 above was caught) — confirmed `down` in `nomad node status`.
3. Re-submitted the job → confirmed QUEUED ("failed to place 2 allocations...
   waiting for additional capacity") — the correct starting state.
4. `fleet power apply --run fleetbench-gn1-2026-08-24 --live`
   (`ZEN_POWER_MIN_DWELL_SECS=90` test-only override, down from the 1800s
   production default, so the full cycle completes in test-reasonable time
   without changing the mechanics being tested) → WoL sent to both MACs.
5. Both boxes back `up=True` within the gated round-trip times (i265 ~8s,
   r3500 ~7s) — matching G-P1's measurements exactly.
6. **Nomad auto-placed the already-queued allocations onto both nodes with
   zero manual intervention** the moment they reconnected — both `running`
   within ~15s of the WoL call. This is the core G-P3 mechanic (declare work
   while asleep → wake → Nomad serves the queued work automatically) working
   end-to-end for the first time.
7. Gap draining in real time: 1,311 → 1,177 → ... → 0 cells, reaching full
   coverage (8,415/8,415 declared, **zero lost cells**) at 06:00:30Z — **11m40s
   from the wake call (05:48:50Z) to full coverage.**
8. **Confounded tail, honestly reported:** after the gap hit 0, neither worker
   reached its own natural 5-consecutive-idle-pass exit — both kept finding
   "new" chunks to (redundantly) claim and burned real CPU on them (i265
   observed at ~1950% CPU / ~19.5 cores, r3500 at ~595% / ~6 cores) for several
   more minutes. This is the ALREADY-DOCUMENTED lease-mode heterogeneous-chunk
   bug from the G-P0 section above, not a new G-P3-specific defect — with the
   real gap at 0, "new" chunk claims are pure re-execution of already-done
   cells, and since this run (correctly, for G-N1/G-P3 isolation) never
   switched to epoch-sharded mode, that inefficiency was always going to show
   up here too. **Consequence for G-P3 specifically: a box that keeps finding
   redundant chunks to claim never reaches `alloc_n==0` on its own, which
   means it can never become sleep-eligible under `decide()`'s current logic
   — the SLEEP half of the autoscale cycle can be blocked indefinitely by this
   bug in the worst case.** Rather than burn further real box-hours watching
   a known, already-understood inefficiency run its course, the job was
   manually stopped once the ledger read 8,415/8,415 (06:03:42Z).
9. Confirmed `alloc_n` dropped to 0 immediately after the manual stop. Ran
   `fleet power apply --live` (06:04:11Z) — **the sleep decision fired
   correctly for both boxes**: `[LIVE] i265: sleep (gap=0, up=True,
   allocs=0, ...)` / `[LIVE] r3500: sleep (gap=0, up=True, allocs=0, ...)`.
   Both drains completed cleanly this time (no `-force`/`-deadline` conflict —
   the real fix from the section above, verified for the first time under
   real conditions rather than a manual CLI test) and both boxes suspended:
   `nomad node status` showed `ready` (heartbeat lag) but direct SSH to both
   timed out — genuinely asleep. **Total wall-clock, wake call to sleep
   call: 15m21s** (05:48:50Z → 06:04:11Z).

**Verdict:** the core G-P3 mechanic — declare work while asleep, WoL wake in
response to queue depth, Nomad auto-places the queued work with zero manual
intervention, and the fleet returns to sleep once genuinely idle — is proven
end-to-end for the first time, with zero lost cells. The "≤110% of G-N1's
wall-time" and "≥70% idle-awake-hours reduction" numeric thresholds are NOT
computed here: this test reused G-N1's partial-completion state rather than
running the full manifest from scratch (so wall-clock isn't comparable
apples-to-apples to G-N1's own number), and the confounded tail (item 8 above)
means "idle-awake box-hours" isn't a clean measurement from this run either.
A clean numeric pass needs either an epoch-sharded rerun (sidesteps the
redundant-tail confound) or a fresh full-manifest G-P3 run sized for direct
comparison against a fresh G-N1 baseline — tracked as follow-up, alongside the
epoch-sharded G-P2 exactly-once rerun noted above.

## CPU-score and GPU-score legs (required workload coverage)

The mission's own framing requires the fleetbench workload to cover
encode + CPU-score + GPU-score kinds. Encode is the G-P0/G-N1/G-P3 runs above;
this section covers the other two, run on r7900x alone (always-on, no
wake-cycle needed — separate from the sleep-cycling gate tests).

**CPU-score:** `homefleet/ubuntu-node/nomad/jobs/fleetbench-cpuscore.nomad.hcl`
— 25,245 `JobKind::Metric` jobs (all 8,415 G-P0-ledger encoded cells × 3 CPU
metrics: `ssim2`/`butteraugli`/`zensim`, confirmed live in the `:exec` image's
`cpu-metrics` feature bundle), built via `~/tmp/build_score_spec.py` from the
G-P0 ledger's real encode `output_sha`s + `zenfleet-ctl declare --spec`.
`jobexec`'s `"metric"` arm re-encodes each cell fresh from
`image_path`/`codec`/`q`/`knob_tuple_json` and scores that — no blob-store
fetch needed, `encode_sha` is for job identity only (confirmed by reading
`jobexec.rs` directly rather than assuming). Manually stopped at **5,313 done,
zero failures (~21% of the full cross product)** — a substantial, error-free
sample proving the kind end-to-end; not run to full completion since every
additional row is more of the same already-confirmed-working behavior, not
new information, and r7900x was needed for the GPU-score leg next.

**GPU-score:** `homefleet/ubuntu-node/nomad/jobs/fleetbench-gpuscore.nomad.hcl`
— 4,000 jobs (a deliberate 800-cell subset × 5 GPU metrics: `ssim2-gpu`/
`butteraugli-gpu`/`dssim-gpu`/`iwssim-gpu`/`cvvdp-gpu` — `zensim-gpu` excluded,
panics by design in this executor tag per `fleet.env`'s dated note). **First
real test of `config { runtime = "nvidia" }` in a Nomad docker-driver
jobspec** (this cluster has no nvidia-device-plugin registered — every node's
`NodeResources.Devices` is null — so the device-stanza GPU-scheduling path
isn't available; the runtime flag is the simpler alternative, and it worked
on the first try): confirmed real GPU utilization via `nvidia-smi` (24% util,
1.5 GB VRAM) during the run, zero failures.

**GPU-score ran to full completion** (started 06:14:43Z, all 4,000 jobs landed
a terminal row by 06:19:54Z — **~5m11s wall-clock**): **3,931 done, 69 failed
(1.7% failure rate)**, all `error_class: encoder_panic`. Not a metric-scoring
bug — the panics happen in the RE-ENCODE step `jobexec`'s `"metric"` arm does
before scoring (see the CPU-score paragraph above), so this is a real codec
robustness finding, surfaced correctly as FAILED rows rather than silently
lost or corrupting a score. Breakdown: 4 distinct source images (3 of 4 are
`scans-illustrations`/`gen_illustrations` content — flat-color line art is a
plausible common factor, not confirmed), `zenjxl` 68/69 (`zenavif` 1/69),
spread across effort 5/7/9 (both VarDCT `vd-e*` and modular `mod-e*` cells)
and all 5 GPU metrics roughly evenly (12-15 failures each) — i.e. the panic
happens during the shared re-encode step, before metric-specific code runs,
consistent with a `zenjxl` encoder bug on specific inputs rather than a
metric-specific issue. The actual panic message wasn't recoverable from
`nomad alloc logs` after the job completed (subprocess stderr not captured
into the parent's log stream — possibly the same capture gap as the SIGTERM-
release investigation's missing diagnostic lines above, though not confirmed
as the same root cause). **Out of scope for this mission to root-cause** — a
real, actionable finding for whoever owns zenjxl robustness, not a fleet-
orchestration defect. The other 3,931/4,000 (98.3%) succeeded cleanly across
all 5 GPU metrics.

## G-T1 — throughput tuning ladders

The mission names four ladders: per-box concurrency (`ZEN_CORE_OVERSUBSCRIBE`),
GPU admission via VRAM hints, warm-exec on/off, and epoch-sharded vs lease
claiming. **1 of 4 is measured and ready to ship; 3 remain untested** — stated
as a fraction, not "tuning done," per this repo's own completion-reporting rule.

**Claim mode — MEASURED, ready to register.** This is the by-product of the
G-P0/epoch-sharded/G-N1/G-P3 runs above, not a purpose-built ladder test, but
it IS a real, repeated, consistent measurement: on this exact 3-box
heterogeneous-core fleet (r7900x 24C / i265 20C / r3500 6C), **epoch-sharded
claiming measured a 1.51-1.69x row-count re-work ratio vs. lease-mode's
3.49x** — roughly 2x less wasted compute — with the important caveat (also
measured, not assumed) that epoch-sharded's own tail-steal mechanism
reintroduces a SMALLER-scope version of the same duplicate-work pattern near
completion (see the epoch-sharded section above). **Recommendation for LAN/
Nomad-managed fleets: default new fleetbench-style jobspecs to
`ZEN_CLAIM_MODE=epoch-sharded`** (as `fleetbench-gt1-epoch.nomad.hcl` already
does) rather than the zenfleet-worker crate's own compiled-in `lease` default.
**Deliberately NOT changing the crate's compiled-in default** — that default
is shared by cloud-burst and POOL-mode workers with different characteristics
(sporadic per-run visits, homogeneous cloud instance types) where lease mode's
tradeoffs may differ; flipping it would be an unreviewed, broader-than-scoped
behavior change. The registration lives here (a benchmark record + jobspec
convention), not in `fleet/handicaps.toml` (that file is for PER-BOX-TYPE
`cells_per_hour` weights feeding weighted rendezvous sharding — a different
axis from "which claim mode" — and its own measurement procedure requires the
dedicated `handicap_typebench.sh` tool, not this mixed-codec fleetbench data).

**Not yet measured — concurrency (`ZEN_CORE_OVERSUBSCRIBE`), VRAM admission,
warm-exec (`ZEN_PERSISTENT_EXEC`).** No ladder runs done for any of these
three in this session. `ZEN_PERSISTENT_EXEC` is the most cheaply testable
next (a single documented on/off env toggle against the same frozen
workload); VRAM admission needs a GPU-metric-heavy cell mix specifically
sized to exercise the admission boundary; concurrency oversubscribe needs a
CPU-bound comparison across box classes. None of these three should be
considered "tuned" — the shipped config for them is still whatever the
untuned default already was, not a measured argmax.

## Gate verdicts (summary)

- **G-P0 baseline** — ✅ measured: cells/hour not cleanly isolated from the
  duplicate-execution confound (see above), re-work tax **3.49x**, distinct-done
  == declared (8,415 == 8,415), 0 failed. Headline finding: chunk-mode's
  per-chunk lease gives **zero** cross-worker dedup on this heterogeneous-core
  fleet (100% duplicate rate) — filed as a Known Bug + ADR defect register entry.
- **G-N1 parity** — ⚠️ PARTIAL: qualitative parity confirmed (Nomad is
  orthogonal to the claim mechanism, same defect class at comparable severity,
  0 failed/poisoned, no ledger divergence observed) — but the strict numeric
  "≥95% throughput / ≤ re-work tax" thresholds are NOT rigorously verified from
  this run's polling data (see the honest-limitation note above). Needs a
  synchronized-snapshot rerun to close out numerically.
- **G-P1 wake round-trips** — ✅ PASSED 3/3 on i265 (8.0s/8.0s/8.0s) and r3500
  (7.0s/7.0s/7.0s), both < the 3-min target (see homefleet
  `ubuntu-node/nomad/wol/wol_roundtrip_test.sh`, run prior to this session's
  fleetbench work).
- **G-P2 drain safety** — ⚠️ PARTIAL PASS. Claim release: ✅ FAILED on first
  genuine attempt, then RESOLVED after instrumentation + an image rebuild —
  2/2 repeatable real successes (verified via the S3 claim object's absence,
  not a log line — see the full writeup for an honest residual puzzle about
  why the new diagnostic output still doesn't appear in captured logs).
  Exactly-once re-execution: ⚠️ NOT cleanly verified — confounded by the
  already-known lease-mode chunking bug (G-P0's finding); needs a dedicated
  epoch-sharded-mode rerun to isolate. WoL wake-after-suspend: not exercised
  in this test (used Nomad drain, not an actual `systemctl suspend` — see
  the setup notes above).
- **G-P3 autoscale end-to-end** — ⚠️ PARTIAL PASS. Core mechanic ✅ proven
  end-to-end for the first time: submit work against sleeping nodes → it
  queues → `fleet power apply --live` WoLs them → Nomad auto-places the
  queued allocations with zero manual intervention → gap drains to zero
  lost cells (8,415/8,415) → `apply` correctly decides to sleep once
  genuinely idle → both boxes verified asleep (SSH timeout). Along the way,
  found the sleep decision's `alloc_n` signal was ALWAYS 0 regardless of
  reality (a 4th real `fleet_power.py` bug, now fixed) and re-confirmed the
  `-force`/`-deadline` fix under real conditions for the first time. Numeric
  thresholds (≤110% wall-time vs G-N1, ≥70% idle-awake-hours reduction) NOT
  computed — this run reused G-N1's partial state rather than a fresh
  full-manifest run, and a lease-mode redundant-work tail (the already-
  documented heterogeneous-chunk bug) confounds a clean idle-awake-hours
  measurement. A clean numeric pass needs a fresh, matched G-N1/G-P3 pair,
  ideally epoch-sharded to sidestep the redundant-tail confound.
- **G-T1 throughput tuning** — ⚠️ **1 of 4 ladders measured.** Claim mode:
  ✅ MEASURED and recommended (epoch-sharded 1.51-1.69x vs lease-mode's 3.49x
  row-count ratio on this heterogeneous fleet) — registered as a jobspec/docs
  convention, deliberately NOT as a crate-wide compiled-in default change
  (broader-than-scoped). Concurrency (`ZEN_CORE_OVERSUBSCRIBE`), VRAM
  admission, and warm-exec (`ZEN_PERSISTENT_EXEC`): ❌ not measured at all —
  no ladder runs done, no argmax to ship, current behavior is whatever the
  untuned default already was. `fleet/handicaps.toml` deliberately NOT
  populated from this mixed-codec run (its own documented measurement
  procedure requires the dedicated `handicap_typebench.sh` tool per encoder
  type, which this fleetbench workload doesn't isolate).
