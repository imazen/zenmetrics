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

### Epoch-sharded rerun (the fix, tested)

<!-- FILL IN: ZEN_CLAIM_MODE=epoch-sharded rerun on the same 3 boxes / same
     manifest content — final row count, distinct job_ids, duplicate rate,
     wall-clock to full coverage. This is the fair, correctly-configured
     baseline G-N1 should be measured against, per RUNNING_JOBS.md §5b's own
     guidance for "a dedicated single-run fleet" (this workload, exactly). -->

## G-N1 — Nomad-managed

Jobspec: `homefleet/zenmetrics/ubuntu-node/nomad/jobs/fleetbench-gn1.nomad.hcl` —
`type = "batch"`, same 3-box `regexp` + `distinct_hosts` constraint, same image
(`force_pull = true`), same `ZEN_CHUNK_WALL_SEC=60`, `ZEN_LONG_LIVED` deliberately
unset (matches G-P0's exit-on-empty-gap semantics so wall-clock-to-exit is
comparable). Only the run id / manifest URI / corpus prefix change.

<!-- FILL IN once launched: launch command, per-alloc start times, finish
     timestamps, total wall-clock, final ledger row count, comparison ratio vs
     G-P0 (target >=95% throughput), re-work tax vs G-P0 (target <=), ledger
     divergence check (target zero). -->

## Gate verdicts

<!-- FILL IN: G-P0 status, G-N1 status (pass/fail against the HANDOFF thresholds),
     and pointers to G-P2/G-P3/G-T1 sections once those runs use this same
     workload. -->
