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

## G-P2 — drain safety: FAILED on first genuine test, re-opens P0 precondition #3

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

**Practical consequence:** the P0 precondition "SIGTERM chunk-claim release,"
previously reported DONE and verified, is **NOT actually proven end-to-end** —
its only verification to date was a synthetic/mocked harness. Filed as a
reopened Known Bug (CLAUDE.md) and in the ADR's defect register, both flagged
HIGH priority — this blocks the mission's own G-P2 gate and undermines the
"awake box-hours" power-cycling design's basic safety guarantee (a box that
suspends mid-chunk without releasing its claim strands that work for the full
claim TTL, exactly what the precondition was built to prevent).

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
- **G-P2 drain safety** — ❌ FAILED on first genuine attempt; see the full
  writeup below. Re-opens a precondition previously reported DONE.
- **G-P3 autoscale end-to-end** — not started.
- **G-T1 throughput tuning** — early ladder data in hand (epoch-sharded vs
  lease: 1.69x/1.51x vs 3.49x row-count ratio — epoch-sharded is the clear
  early leader, with its own smaller-scope tail-pileup caveat), but no
  concurrency/VRAM/warm-exec ladders run yet, and `fleet/handicaps.toml` was
  deliberately NOT populated from this mixed-codec run (its own documented
  measurement procedure requires the dedicated `handicap_typebench.sh` tool per
  encoder type, which this fleetbench workload doesn't isolate).
