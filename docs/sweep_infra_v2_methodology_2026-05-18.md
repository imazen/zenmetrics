# Sweep Infrastructure v2 — Methodology

**Date:** 2026-05-18
**Branch:** `feat/sweep-infra-unified`
**Scope:** vast.ai metric backfill workflow (cvvdp / iwssim / ssim2 / dssim / zensim / butteraugli)

## What this fixes

The 2026-05-17 and 2026-05-18 backfill sessions accumulated three
load-bearing bugs in the operational pipeline:

1. **Silent bogus-data failure.** The iwssim docker image had a
   NaN-on-identical bug; workers happily completed chunks and uploaded
   525 sidecars where every iwssim value was 0 or NaN. The failure was
   not detected until V_24 training catastrophically failed three hours
   later, wasting compute and chasing a phantom training bug.
2. **Destroyer fragility.** The bash + python heredoc destroyers at
   `/tmp/cvvdp-resume/run_destroy_{iwssim,ssim2}_754.sh` parsed `vastai
   show instances --raw` with `json.loads()`. The ssim2 destroyer hit
   `json.decoder.JSONDecodeError: Expecting value: line 1 column 1
   (char 0)` once the destroy condition was met and the script crashed
   without destroying anything, leaving 15 boxes running orphaned for
   90 minutes — a real-money cost.
3. **Per-metric duplication.** Three near-identical worker scripts
   (`iwssim_backfill_chunk_worker.sh`, `ssim2_backfill_chunk_worker.sh`,
   `cvvdp_backfill_chunk_worker.sh`) and three near-identical launchers
   added up to ~770 lines of mostly-similar bash per metric, with the
   `--fail-on-bogus` logic absent from all three.

## Components delivered

| File / crate | Role | Commit |
|---|---|---|
| `crates/zenmetrics-cli/src/main.rs` | `score-pairs --fail-on-bogus` gate (10 unit tests) | `242d4b4a` |
| `scripts/sweep/metric_backfill_chunk_worker.sh` | Unified single-metric worker | `5b98e50c` |
| `crates/zenfleet-vastai/` | Rust binary: `status` / `destroy` / `watch` (22 tests) | `3a849a69` |
| `scripts/sweep/launch_backfill.sh` | Unified launcher | `50982c33` |
| `scripts/sweep/fleet_status.sh` | One-shot dashboard | `5de4c676` |

## End-to-end pipeline

```
+--------------------------+    +--------------------------+
| chunks.jsonl on R2       |    | docker image with        |
| (per generator)          |    | zenmetrics-cli baked in |
+-----------+--------------+    +-------------+------------+
            \                                /
             \                              /
              v                            v
        +-----------------------------------+
        | launch_backfill.sh --metric ...   |    operator
        |  -> vastai create instance ...    | <----------+
        |  -> prints zenfleet-vastai watch cmd |            |
        +------+--------------------+-------+            |
               |                    |                    |
               v                    v                    |
       +---------------+    +----------------+           |
       | N vast.ai     |    | (background)   |           |
       | workers, each |    | zenfleet-vastai   |           |
       | running       |    | watch          |           |
       | metric_       |    |                |           |
       | backfill_     |    | polls every    |           |
       | chunk_worker  |    | 90s; destroys  |           |
       | --fail-on-    |    | when target    |           |
       | bogus         |    | sidecars or    |           |
       +---+-------+---+    | wall cap hit   |           |
           |       |        +-------+--------+           |
           v       v                |                    |
   rc=0:           rc=2:            |                    |
   upload          upload           v                    |
   sidecar.parquet failure.log      destroy each         |
   to R2           to R2/failures/  instance via         |
                                    `vastai destroy`     |
                                                         |
                                                         v
                                                +----------------+
                                                | fleet_status.sh|
                                                | <run-id>       |
                                                | (anytime: snap |
                                                | of state +     |
                                                | sample valid)  |
                                                +----------------+
```

## Migration matrix

For new backfills, replace the per-metric scripts as follows:

| Old script | New invocation |
|---|---|
| `scripts/sweep/iwssim_backfill_chunk_worker.sh` | `scripts/sweep/metric_backfill_chunk_worker.sh --metric iwssim` |
| `scripts/sweep/ssim2_backfill_chunk_worker.sh` | `scripts/sweep/metric_backfill_chunk_worker.sh --metric ssim2` |
| `scripts/sweep/cvvdp_backfill_chunk_worker.sh` (single-impl mode) | `scripts/sweep/metric_backfill_chunk_worker.sh --metric cvvdp` |
| `scripts/sweep/cvvdp_backfill_chunk_worker.sh` (dual-impl with pycvvdp) | **unchanged** (two-scorer flow, not in v2 scope) |
| `scripts/sweep/iwssim_backfill/launch.sh` | `scripts/sweep/launch_backfill.sh --metric iwssim ...` |
| `scripts/sweep/cvvdp_backfill/launch_imazen.sh` | `scripts/sweep/launch_backfill.sh --metric cvvdp ...` |
| `scripts/sweep/cvvdp_backfill/launch.sh` (dual-impl) | **unchanged** (dual-impl flow, not in v2 scope) |
| `/tmp/cvvdp-resume/run_destroy_*.sh` | `zenfleet-vastai destroy --label-prefix <run-id>` |
| Auto-destroy heredoc | `zenfleet-vastai watch --label-prefix <run-id> --target-sidecars N --r2-prefix s3://...` |
| Manual "how many workers?" check | `zenfleet-vastai status --label-prefix <run-id>` or `fleet_status.sh <run-id>` |

## `--fail-on-bogus` checks

The `score-pairs --fail-on-bogus` gate runs **after the parquet is
written** and re-reads the in-memory score vector. Exit codes:

- **rc=0**: scores pass every check — proceed.
- **rc=1**: hard error before the parquet was written.
- **rc=2** (new): parquet was written, but the score distribution
  failed at least one sanity check. The worker uploads a failure log
  instead of the sidecar; the orchestrator treats the chunk as
  poisoned.

The checks (`fn bogus_check` in `crates/zenmetrics-cli/src/main.rs`):

1. `n_nan == 0` — score-pairs writes NaN on per-pair decode/score
   failures, but a chunk shouldn't have any silent NaN rows. If it
   does, the kernel failed silently somewhere.
2. `n_at_identity / n_total < 0.5` — for each metric we know the
   "identical inputs" value (iwssim → 1.0, cvvdp → 10.0, ssim2 →
   100.0, dssim → 0.0, butteraugli → 0.0). If ≥ 50% of rows are
   exactly at that value, the kernel almost certainly hit a default-
   fail short-circuit on at least half the chunk.
3. `max - min > 0.01` over ≥ 4 finite rows — a real metric on a
   real quality sweep produces variance. Constant output across a
   chunk means the kernel never ran (or ran identically every time —
   which itself is a bug worth flagging).
4. Mean is within `metric_range_bounds(metric)` — per-metric
   documented range (e.g. iwssim mean must be in `[-0.001, 1.001]`).

Metrics without a known range (future additions) return `None` from
`metric_range_bounds` and skip checks #2 and #4.

## Validation

- **`--fail-on-bogus` unit tests** (10/10 passing):
  - `bogus_check_passes_clean_iwssim_distribution` — real 0.70..0.98 spread passes.
  - `bogus_check_passes_clean_cvvdp_distribution` — real 6..9.8 JOD spread passes.
  - `bogus_check_passes_clean_ssim2_distribution` — real 30..95 spread passes.
  - `bogus_check_rejects_any_nan` — one NaN row trips the gate.
  - `bogus_check_rejects_all_zero_iwssim` — the 525-sidecar failure mode (all 0.0).
  - `bogus_check_rejects_majority_at_identity` — 60% at JOD 10.0 trips.
  - `bogus_check_rejects_constant_output` — 100 rows all 0.8 trips.
  - `bogus_check_rejects_mean_out_of_range` — mean -0.5 for iwssim trips.
  - `bogus_check_handles_few_rows` — 3-row chunks aren't false-positives.
  - `bogus_check_rejects_empty_column` — 0-row column rejected.
- **`zenfleet-vastai` parse + cli tests** (22/22 passing):
  - Empty stdout `""` returns no instances cleanly (the exact 2026-05-18
    destroyer-crash input).
  - Deprecation banner glued onto JSON body — preamble-strip works.
  - Mixed fleet with a `null` row in the array — warns + skips, keeps
    rest.
  - Truncated JSON — error, not panic.
  - v0 bare-array vs v1 envelope shape — both accepted.
  - `destroy --dry-run` against a 5-row fixture — confirms 5 destroy
    ops without calling real vastai.
- **`zenfleet-vastai status` real-CLI smoke**: ran against current vast.ai
  state (0 instances) — reports cleanly, no crash. The exact input
  that crashed the bash destroyer.

## Operator workflow (new sweep)

```bash
# 1. Generate chunks (per metric's existing generator).
python3 scripts/sweep/generate_<metric>_backfill_chunks.py --run-id <run-id> ... > /tmp/chunks.jsonl
s5cmd cp /tmp/chunks.jsonl s3://coefficient/jobs/<run-id>/chunks.jsonl

# 2. Build + install zenfleet-vastai once.
cargo build --release -p zenfleet-vastai
cp target/release/zenfleet-vastai ~/.local/bin/

# 3. Launch fleet.
scripts/sweep/launch_backfill.sh \
    --metric <metric> \
    --run-id <run-id> \
    --chunks s3://coefficient/jobs/<run-id>/chunks.jsonl \
    --docker ghcr.io/imazen/zenmetrics-sweep:<tag> \
    --max-dph 0.30 --n-boxes 30 \
    --watch &      # or omit --watch and run zenfleet-vastai watch separately

# 4. Monitor (any time).
scripts/sweep/fleet_status.sh <run-id>

# 5. If something's wrong, destroy.
zenfleet-vastai destroy --label-prefix <run-id>
```

## Constraints / not-yet-covered

- The dual-implementation cvvdp flow (cvvdp-gpu + pycvvdp side-by-side
  with parity logging) is **not** folded into the unified worker. That
  shape — two scorers, two sidecars, an extra parity TSV — has its
  own concerns; the unified worker stays single-metric.
- `zenfleet-vastai` is intentionally small (~600 LOC across `main.rs` +
  `parse.rs`). It is not a vast.ai SDK; it shells out to `vastai` for
  every action. The defensive parser is the load-bearing part.
- The `--fail-on-bogus` gate is a distribution-level check on per-chunk
  output, not a parity gate against an authority. It catches gross
  failure modes (NaN, constant, default-fail short-circuit) but
  wouldn't catch e.g. a kernel that produces *plausible-looking but
  systematically wrong* scores. For that, the existing parity tests
  against pycvvdp (cvvdp_goldens) remain the gold standard.

## Future work

- Port the dual-impl cvvdp flow into a `--dual-impl` mode on the
  unified worker once we're confident enough in the single-impl path.
- Add a `zenfleet-vastai rotate` subcommand for replacing a fraction of
  the fleet without destroying all — useful for slow-rolling new
  docker images.
- Persist `--fail-on-bogus` failure stats to the run's R2 prefix as a
  rolling-window summary; right now each failure is its own log file
  and operators have to `ls failures/` to count them.
