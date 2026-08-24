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

**Interim finding (mid-run, ~7483/8415 distinct jobs done, will be finalized below):
lease-mode claiming has a LARGE genuine duplicate-execution rate on this 3-box LAN
fleet — 61.85% of distinct job_ids had a "done" row from more than one DISTINCT
(worker, ts) pair** (e.g. r7900x finished a cell at ts=1787545497, i265 independently
finished the SAME job_id 14s later at ts=1787545513 — both wrote successful ledger
rows). This is the documented lease-mode race (`docs/RUNNING_JOBS.md` §5b: "the
avifgen encode run measured 3.6x under lease claiming with empty views"), sharply
amplified here vs. the geo-distributed cloud fleets that number was measured on:
near-zero LAN latency between 3 fast local boxes means claim-view staleness windows
overlap constantly. **This is expected lease-mode behavior being measured
correctly, not a bug in the systemd deployment** — `zenfleet-worker`'s claim
algorithm is identical regardless of what launches the process, so G-N1 (same
image, same default lease-mode claiming, Nomad-launched instead of systemd-
launched) should show a SIMILAR duplicate rate if Nomad is genuinely orthogonal to
work-distribution per the ADR's framing; a large delta between G-P0 and G-N1 here
would itself be the anomaly worth chasing. Neither run set `ZEN_CLAIM_MODE=epoch-
sharded` — epoch-sharded claiming (the documented fix for exactly this failure
mode) is deferred to the G-T1 claim-mode tuning ladder, run separately against
this same frozen workload once G-P0/G-N1 parity is established under the shared
(lease-mode) baseline.

Raw row-count ratio (rows / distinct job_ids) at the same checkpoint: 1.7369 —
noted separately from the 61.85% figure above because it also counts normal
claimed-then-done bookkeeping (not itself waste); the duplicate-(worker,ts) metric
is the one that actually measures wasted compute.

<!-- FILL IN once the run reaches all-inactive: finish timestamps, total wall-clock
     (from earliest start to latest finish), final ledger row count, final distinct
     job_id count (expect 8415, zero poison/failed), final duplicate-execution rate,
     per-box row share. -->

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
