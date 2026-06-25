# zenmetrics sweep operations

Captures operational lessons for running zenmetrics sweeps on vast.ai
fleets. See parent README.md for codebase orientation.

**History note (2026-05-06):** these scripts originated in
`turbo-metrics/scripts/sweep/`. They migrated here when the
`zenmetrics-cli` and `*-gpu` crates were extracted from
turbo-metrics into the standalone zenmetrics workspace
(see zenmetrics commit `4a729b6`). The turbo-metrics fork retains
only the CUDA-specific crates and upstream-tracking pieces. All
new sweep work happens in this repo.

## Key files (current — 2026-05-19)

The **unified Rust worker** in `crates/zenfleet-vastai` is the
production entrypoint. Sweep operations are:

- `onstart_unified.sh` — execs `zenfleet-sweep worker --backend
  vastai --mode omni` (the default mode). Replaces the bash
  `onstart_omni_backfill.sh` chain. Used with `Dockerfile.sweep.v26`.
- `onstart_feature_backfill.sh` — execs `zenfleet-sweep worker
  --backend vastai --mode feature-backfill`. Reads existing omni sidecars + cached
  encoded variants from R2, writes zensim 300-feature parquets
  without re-encoding. Used with `Dockerfile.sweep.v26`.
- `generate_cvvdp_backfill_chunks.py` — chunk generator (slices
  the unified-V_X parquet into 200-row chunks, emits
  `chunks.jsonl` for upload to `s3://coefficient/jobs/<run>/`).
- `launch_backfill.sh` — vast.ai fleet launcher: `--n-boxes 1` to
  smoke-test, `--n-boxes N` to fan out. Passes `PARALLEL_CHUNKS`,
  `SKIP_CLAIMS`, `METRICS`, `CHUNKS_R2` env vars through to the worker.
  (`launch_single_instance.sh` was deleted 2026-06-25 — `--n-boxes 1` is
  the smoke now; the v26 image's `run_with_error_trap.sh` does the
  panic-log-upload + self-destroy it used to do by hand.)
- `scripts/jobsys/fleet watch <run>` — the ONE monitor: per-box GPU/CPU
  util + $-burn + idle / failed-to-start flags (replaced the deleted
  `fleet_util_snapshot.sh` + `vast_cost_watch.sh`).
- `hetzner_cpu_sweep.sh` — **heterogeneous SPLIT, Hetzner-CPU half** (2026-06-23):
  each cheap box fetches its chunk + runs `zenmetrics sweep` (CPU ssim2+zensim) +
  `--encoded-out-dir` → tars the variants to R2 (the master record). Scoped 3h
  creds; `--image docker-ce`; biggest-first type×location fallback. Hard limits:
  Hetzner account caps at **5 servers**; **cpx41 phased out** → cpx51 (16c) is the
  per-box ceiling (ccx53/63 capacity-out).
- `split_score_worker.sh` — **SPLIT vast.ai-GPU half**: entrypoint of the thin
  `ghcr.io/imazen/zenmetrics-sweep:v29-split` image (FROM `:v29-2026-06-23`, all 6
  GPU metrics). Pulls persisted variants + `pairs.tsv`, runs `zenmetrics score-pairs`
  per GPU metric (butter/cvvdp/ssim2-gpu/zensim-gpu/dssim) → sidecars. **Launch via
  `--onstart-cmd bash /usr/local/bin/split_score_worker.sh`** — vast ignores the
  image ENTRYPOINT and runs its own ssh init + the onstart-cmd. (see memory
  `heterogeneous-fleet-split`)

The per-metric/legacy bash workers (`omni_backfill_chunk_worker.sh`,
`cvvdp_backfill_chunk_worker.sh`, `iwssim_backfill_chunk_worker.sh`) and
onstarts (`onstart_omni_backfill.sh`, `onstart_v3.sh`, `onstart_cvvdp_*`)
were **deleted 2026-06-25** — production scoring is in-process in the Rust
worker (`inline-sweep`). The one surviving bash worker is
`metric_backfill_chunk_worker.sh` (single-metric, driven by
`launch_backfill.sh`). cvvdp now scores through the unified worker, not a
separate dual-impl bash flow.

The v14→v25 chain was collapsed into single-file
`Dockerfile.sweep.v26` on 2026-05-21; all earlier vNN Dockerfiles
were deleted. See `README.md` for the v26 image's layer plan and
the proven end-to-end pipeline that landed 2933 omni sidecars +
2933 zensim feature parquets across two production runs on
2026-05-19 (those runs used v24, which v26 supersedes with the
same runtime contract).

## CRITICAL: every onstart MUST self-destroy on failure

Every new sweep onstart script MUST wrap its main loop with an
EXIT trap that, on any non-zero exit, (a) uploads the tail of its
captured log to R2 under
`s3://coefficient/jobs/${SWEEP_RUN_ID}/worker-logs/${WORKER_ID}-failure.log`
and (b) issues a `vastai destroy instance ${CONTAINER_ID}` (or the
equivalent REST DELETE) so the box stops billing the moment its
work fails. Without this, a worker that exits in 6-80 s leaves the
vast.ai instance running at \$/hr until an external `zenfleet-vastai
destroy` cleans it up — which is exactly the failure mode the
2026-05-18 EXP-LARGER-LARGE cascade hit four times.

Two equivalent ways to satisfy the contract:

1. **Image-level wrapper (preferred for v26):** the v26 image's
   `ENTRYPOINT` already chains through
   `/usr/local/bin/run_with_error_trap.sh`, which installs the EXIT
   trap, captures stderr, and shells out to the baked
   `/usr/local/bin/zenfleet-vastai self-destroy` on rc≠0. New onstarts
   running in v26 inherit this automatically — no changes needed in
   the script itself.

2. **Inline trap (required for anything that runs
   without `run_with_error_trap.sh`):** install an `on_exit` trap
   directly in the onstart script. See
   `onstart_iwssim_backfill_v14.sh` lines 60-140 as the canonical
   pattern: tee stdout+stderr to `$ONSTART_LOG`, hydrate
   `CONTAINER_*` from `/proc/1/environ`, then `trap on_exit EXIT`
   where `on_exit` composes a context header + last 200 lines and
   calls `curl -X DELETE` against `console.vast.ai/api/v0/instances/${CONTAINER_ID}/`.

Whichever path you pick, **xargs return code MUST propagate to the
script's exit**. The default `xargs ... < chunks` at end-of-script
discards rc on its own line; capture into `xargs_rc=$?` and `exit
"$xargs_rc"` if non-zero. Without this, a chunk loop that fails every
chunk in 6 s still ends the onstart with rc=0 and the trap does nothing.
(The unified `onstart_unified.sh` sidesteps this — the Rust worker
propagates its own exit code; the respawn loop aborts on any non-zero rc.)

## Worker mechanics

1. vast.ai pulls the baked v26 image (everything is baked — no apt /
   pip / binary download at boot; see global CLAUDE.md "BAKE EVERYTHING").
2. `--onstart-cmd` runs `onstart_unified.sh`, which execs
   `zenfleet-sweep worker --backend vastai --mode omni`. That one Rust
   process:
   - Hydrates env from `/proc/1/environ` (R2_*, SWEEP_*, WORKER_*, …).
   - Reads `chunks.jsonl` from `s3://coefficient/jobs/<run>/`.
   - Loops: atomic-claim a chunk (token race + sidecar idempotency),
     encode + score all metrics **in-process** (one CubeCL init), write
     the parquet sidecar — no per-chunk subprocess, no `zenmetrics --batch`.

   (The legacy bash flow — `onstart_v3.sh` downloading s5cmd/jq + a release
   tarball + looping `zenmetrics --batch` — was removed 2026-06-25.)

## CRITICAL: cgroup-aware parallelism (2026-05-04 fix)

`nproc` inside a vast.ai container reports the **host** CPU count
(often 56) — NOT the container's effective cgroup allocation (usually
8–16). Setting `xargs -P $(nproc)` oversubscribes and thrashes.

The Rust worker reads the cgroup allocation (not `nproc`) in
`crates/zenfleet-vastai/src/worker/adapt.rs`; both cgroup v1 and v2 are
handled. (The original bash fix lived in `onstart_v3.sh::cores_from_cgroup()`
/ `ram_gb_from_cgroup()`, now removed.) Final formula:

```
PARALLEL = min(cgroup_cores, ram_gb * 2/3) - 2
```

Subtract 2 cores for the heartbeat loop + system overhead. RAM cap
keeps each parallel slot ≤1.5 GB to avoid OOM on lossy encodes (jxl
encoder spikes to ~1 GB on 1024×1024 images).

**Result:** 3-5× throughput improvement on multi-core boxes (2026-05-04
empirical, on v06 sweep). Without this fix, a 56-core/24GB-RAM box
was running 56-way parallel and OOM-thrashing; now runs ~14-way and
finishes 4× faster.

## CRITICAL: vast.ai filter quirks

**`cpu_ram` is in GB, not MB** despite API responses giving MB.
`cpu_ram>16` means >16 GB RAM. The `ram` field doesn't filter at all
on this CLI version (1.0.8); use `cpu_ram`.

**`verified=true` excludes most cheap offers** — drop it for sweep work.
We don't need datacenter-class boxes; cheap consumer GPUs are fine.

**`--raw -o dph_total` conflicts.** `-o` is for the columnar (non-raw)
output; combining with `--raw` produces non-JSON. Use:
```
vastai search offers '...' --order dph_total --raw
```

**Required RAM has trended up** with more knobs sweeping. v06 worked at
12 GB; v07/v08 with patches/gaborish need 24+ GB; v09 with
force_strategy needs 32 GB on multi-MB images. Set `MIN_RAM_GB=24` in
launchers as the default; bump to 32 for v09+.

## Atomic chunk claim mechanism

Each worker, before processing a chunk, attempts:
```
mc cp /tmp/claim-<chunkid>.txt s3://coefficient/claims/<sweep>/<chunkid>.claim
```

If the claim file already exists, `mc cp` fails (S3 If-None-Match
emulation via mc) and the worker moves on. If it succeeds, the worker
processes the chunk.

**FAILURE MODE: orphaned claims.** If a worker crashes mid-chunk, its
claim file remains. No other worker will pick up that chunk. Symptoms:
chunks stop landing on R2 even though instances are "running".

**Recovery:**
```bash
# Cleanest: just delete all claims for the sweep. The done-chunk skip
# logic checks .tsv existence, NOT claim existence — so workers will
# re-iterate the chunk list, skip done chunks (via .tsv check), and
# pick up only the unfinished ones.
aws --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    s3 rm s3://coefficient/claims/<sweep>/ --recursive
```

We did this 2026-05-06 across v08/v09/v11 (140 stale claims) after
killing 192 idle workers — the surviving 7 then resumed picking up
the last few chunks.

## Per-chunk runtime is HUGE

Each chunk processes 5 images × ~500 knob combos × encode + 4 metrics =
roughly **3-5 hours per chunk**. v08 worker logs showed individual
chunks taking 10,000-18,000 seconds.

Implications:
- A worker crash mid-chunk loses ~3 hours of work.
- "Stalled" sweeps may not be stalled — just slow.
- Don't kill workers preemptively. Check the heartbeat (1-min interval)
  before assuming death.

## Per-sweep instance tracking — IMPORTANT

The original launcher wrote `/tmp/v06_binary_instances.txt` and
overwrote it on every relaunch, regardless of run_id. **Multiple
agents launched in series clobbered each other's tracking and led to
killing the wrong workers.**

Current launcher pattern uses per-sweep file:
```bash
INSTANCE_FILE="/tmp/${SWEEP_RUN_ID}_instances.txt"
> "$INSTANCE_FILE"  # truncate per run
echo "$ID $offer_id $WORKER_ID" >> "$INSTANCE_FILE"
```

Always check `/tmp/<sweep>_instances.txt` before bulk-killing. If
unsure, query vast.ai labels and filter by sweep prefix.

## Cost control

Idle "running" instances burn ~$0.04-0.20/hr each. For 200 instances
that's ~$20/hr = $480/day even if no chunks are processing.

Burn-rate audit recipe:
```bash
vastai show instances --raw | python3 -c "
import json,sys
d=json.loads(sys.stdin.read())
insts = d if isinstance(d, list) else d.get('instances', [])
total = sum(i.get('dph_total') or 0 for i in insts)
print(f'instances: {len(insts)}, burn: \${total:.2f}/hr')"
```

After clearing claims, you only need ~1 worker per outstanding chunk
(plus a small buffer). 8 workers for 7 chunks is plenty; the extras
are pure waste once the chunk count drops.

Mass-destroy pattern (parallel via xargs):
```bash
echo "$KILL_IDS" | xargs -P 20 -I {} sh -c 'echo y | vastai destroy instance {} >/dev/null 2>&1; echo killed {}'
```

`vastai destroy instance <id>` requires `y` confirmation; pipe `echo y |`.

## Decoder/encoder bug hunting from sweep data

Sweep workers occasionally fail with encoder hangs or decoder rejects.
Examples surfaced in this fleet:

- **jxl-encoder e9 checker pattern hang** (issue #27, PR #28 in jxl-encoder repo).
- **zenjxl-decoder rejecting files libjxl C accepts** — confirmed via
  `djxl 0.10.3` as the authority. **libjxl is the authority, not
  jxl-oxide.** zenjxl-decoder is a fork of jxl-rs; both inherit the
  same VarDCT-strict-validation lineage and reject some files libjxl C
  accepts. When triaging, always test with `djxl` directly.

## Sweep status, May 2026

| Sweep | Date | Knobs | Chunks (R2) | Status |
|---|---|---|---|---|
| v06 | 2026-05-04 | distance, effort, biters, ziters | ~200 | DONE — 165k cells |
| v07 | 2026-05-05 | + lz77, force_strategy, gaborish, patches, lf_frame, pdl, progressive | 34 | DONE — 32k cells |
| v08 | 2026-05-05 | v06 grid + v07 winners (patches, gaborish, pdl) | 98/100 | 98% complete (2 stragglers) |
| v09 | 2026-05-05 | force_strategy + max_strategy_size + mode | 17/21 | 81% complete |
| v10 | 2026-05-05 | multi-codec (zenjxl/avif/webp) | 60/60 | DONE |
| v11 | 2026-05-05 | per-distance-band specialist | 19/20 | 95% complete |

All sweeps mirrored to R2 at `s3://zentrain/sweep-vXX-YYYY-MM-DD/`.

## Where reports live

- Per-sweep picker reports: `~/work/zen/zenanalyze/benchmarks/picker_v0X_*.md`
- Encoder default decisions: `~/work/zen/zenjxl/benchmarks/INDEX.md`
- Local sweep TSVs + analysis scripts: `~/sweep-data/`
