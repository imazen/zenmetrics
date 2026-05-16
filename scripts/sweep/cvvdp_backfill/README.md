# cvvdp-backfill — operator runbook

The PINNED TASK pipeline (per `~/work/zen/zenmetrics--cvvdp-new/CLAUDE.md`):
backfill CVVDP JOD scores into the existing unified-V_X parquet store
at `/mnt/v/zen/zensim-training/2026-05-07/unified/`, with both the
`cvvdp_imazen_v*` (this crate) and `cvvdp_pycvvdp_v054` (canonical
pycvvdp v0.5.4) implementations landing as separate parquet sidecars
that join back on the identity tuple `(image_path, codec, q,
knob_tuple_json)`.

Schema spec: [`crates/cvvdp-gpu/docs/CVVDP_SIDECAR_SCHEMA.md`](../../../crates/cvvdp-gpu/docs/CVVDP_SIDECAR_SCHEMA.md).

## Pipeline shape

```text
generate_cvvdp_backfill_chunks.py     →  chunks.jsonl
       |                                       |
       |  (uploaded to R2)                     |
       v                                       |
s3://coefficient/jobs/<run>/chunks.jsonl       |
       |                                       |
       v                                       |
cvvdp_backfill/launch.sh  →  vast.ai fleet     |
       |                          |            |
       |                          v            |
       |              onstart_cvvdp_backfill.sh
       |                          |
       |                          v
       |              cvvdp_backfill_chunk_worker.sh (per chunk)
       |                          |
       |                          v
       |              s3://zentrain/<run>/cvvdp_imazen/<chunk>.parquet
       |              s3://zentrain/<run>/cvvdp_pycvvdp_v054/<chunk>.parquet
       |                          |
       |                          v
       |              cvvdp_backfill/finalize.sh
       |                          |
       v                          v
                  /mnt/v/zen/zensim-training/<date>/cvvdp_sidecars/*.parquet
                  s3://zentrain/<run>/consolidated/*.parquet
```

Six scripts, three live in `scripts/sweep/` proper (chunk-gen +
worker + onstart — paired with the existing sweep tooling) and
three live in `scripts/sweep/cvvdp_backfill/` (this directory —
the cvvdp-backfill-specific launch + finalize entry points).

## Quick-start

Assumptions:

- vast.ai cli authenticated (`vastai login`).
- gh cli authenticated and has `write:packages` (`gh auth status`).
- `~/.config/cloudflare/r2-credentials` exports `R2_ACCOUNT_ID`,
  `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`.
- Docker images pushed to ghcr.io (see "Docker images" below).
  Default tags baked into `launch.sh`:
  - `ghcr.io/imazen/zen-metrics-sweep:0.6.4-cvvdp-<short>`
  - `ghcr.io/imazen/pycvvdp-scorer:0.5.4`

### 1. Generate the chunk manifest

```bash
python3 scripts/sweep/generate_cvvdp_backfill_chunks.py \
    --unified-dir /mnt/v/zen/zensim-training/2026-05-07/unified \
    --run-id cvvdp-backfill-2026-05-15 \
    --source-r2-prefix s3://zentrain/sweep-v15-2026-05-06/sources \
    --input-r2-prefix s3://zentrain/unified-2026-05-07 \
    --output-r2-prefix s3://zentrain/cvvdp-backfill-2026-05-15 \
    --out /tmp/chunks.jsonl
```

At default `--chunk-size 100`, the 7-parquet store splits into
~23,747 chunks. For a smoke pass over zenwebp only:

```bash
python3 scripts/sweep/generate_cvvdp_backfill_chunks.py \
    --filter-codec zenwebp --max-chunks 5 \
    --chunk-size 20 \
    --unified-dir /mnt/v/zen/zensim-training/2026-05-07/unified \
    --run-id cvvdp-backfill-smoke \
    --source-r2-prefix s3://zentrain/sweep-v15-2026-05-06/sources \
    --input-r2-prefix s3://zentrain/unified-2026-05-07 \
    --output-r2-prefix s3://zentrain/cvvdp-backfill-smoke \
    --out /tmp/chunks-smoke.jsonl
```

### 2. Upload manifest + worker script to R2

```bash
SWEEP_RUN_ID=cvvdp-backfill-2026-05-15
PREFIX="s3://coefficient/jobs/${SWEEP_RUN_ID}"

s5cmd --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    --profile r2 cp /tmp/chunks.jsonl "${PREFIX}/chunks.jsonl"

s5cmd --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    --profile r2 cp scripts/sweep/cvvdp_backfill_chunk_worker.sh \
    "${PREFIX}/cvvdp_backfill_chunk_worker.sh"
```

(`launch.sh` will upload `onstart_cvvdp_backfill.sh` automatically.)

### 3. Launch the fleet

```bash
N_BOXES=6 DRY_RUN=1 bash scripts/sweep/cvvdp_backfill/launch.sh
# verify the offer picks look sane, then:
N_BOXES=6 bash scripts/sweep/cvvdp_backfill/launch.sh
```

Defaults: 6 instances, `MAX_DPH=0.30`, `MIN_RAM_GB=16`,
`MIN_DISK_GB=40`, `PARALLEL=2` chunk-workers per box.
See `cvvdp_backfill/launch.sh` env-var section for overrides.

### 4. Monitor

Recommended: `cvvdp_backfill/status.sh` aggregates manifest size,
heartbeats (boot/run/done counts + newest timestamp), per-impl
sidecar counts with completion percentages, and any failure logs
— one screen of output:

```bash
SWEEP_RUN_ID=cvvdp-backfill-2026-05-15 bash scripts/sweep/cvvdp_backfill/status.sh

# Periodic poll (Ctrl-C to stop):
SWEEP_RUN_ID=... watch -n 60 'bash scripts/sweep/cvvdp_backfill/status.sh'
```

Raw R2 paths if you want to drive `s5cmd` yourself:

```bash
# Heartbeats (one .boot / .run / .done per worker):
s5cmd --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    --profile r2 ls "s3://coefficient/heartbeats/${SWEEP_RUN_ID}/"

# Per-chunk failure logs (the worker only emits these on failure):
s5cmd --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    --profile r2 ls "s3://coefficient/logs/${SWEEP_RUN_ID}/" | tail

# Live tail of a specific worker:
vastai logs <instance-id>

# Progress: count sidecars vs total chunks:
s5cmd --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    --profile r2 ls "s3://zentrain/${SWEEP_RUN_ID}/cvvdp_imazen/" | wc -l
```

### 5. Finalize when fleet has finished

```bash
SWEEP_RUN_ID=cvvdp-backfill-2026-05-15 \
LOCAL_UNIFIED_DIR=/mnt/v/zen/zensim-training/2026-05-15/unified \
    bash scripts/sweep/cvvdp_backfill/finalize.sh
```

Outputs (under `$LOCAL_UNIFIED_DIR/cvvdp_sidecars/` if set):

```text
cvvdp_imazen_v12_zenwebp.parquet         consolidated for one source-parquet
cvvdp_pycvvdp_v054_v12_zenwebp.parquet
parity_v12_zenwebp.tsv                   joined side-by-side with diff column
manifest.json                            chunk counts, row counts, parity stats
```

Also pushed to `s3://zentrain/<run>/consolidated/` unless
`UPLOAD_CONSOLIDATED=0`.

### 6. Tear down

```bash
bash scripts/sweep/destroy_all.sh   # or per-instance: vastai destroy instance <id>
```

### 6b. (Optional) Gate automation on parity

`finalize.sh` always writes `manifest.json` regardless of how blown out
parity is. For an automation step that needs a pass/fail signal — a
nightly fleet run, a release gate, a PR job — wrap the manifest with
`assert_parity.py`:

```bash
# Defaults: mean & median |diff| <= 0.10 JOD, max <= 0.50 JOD,
# null parity tolerated.
python3 scripts/sweep/cvvdp_backfill/assert_parity.py \
    "$LOCAL_UNIFIED_DIR/cvvdp_sidecars/manifest.json"

# Strict (every source must have both impls present):
python3 scripts/sweep/cvvdp_backfill/assert_parity.py \
    --require-parity-on-all \
    "$LOCAL_UNIFIED_DIR/cvvdp_sidecars/manifest.json"
```

Exit 0 = pass, exit 2 = at least one threshold blown, exit 3 = a
source had `parity: null` and `--require-parity-on-all` was set. See
the script's header docstring for the full flag list.

## Docker images

The fleet uses two images. Build + smoke + push lives in the
build-agent runbook (see `~/work/zen/zenmetrics--cvvdp-new/CLAUDE.md`
PINNED TASK status). The expected tags:

- `ghcr.io/imazen/zen-metrics-sweep:0.6.4-cvvdp-<short>` — built from
  `feat/cvvdp-gpu-scaffold` (NOT canonical master, which lacks
  `score-pairs` + `sweep --pairs-tsv`). Use the `Dockerfile.sweep.v13`
  in the scaffold worktree; build context is the parent dir of
  `zenmetrics--cvvdp-new/` + `zenjpeg/` + `zenanalyze/`.
- `ghcr.io/imazen/pycvvdp-scorer:0.5.4` — self-contained PyTorch
  2.5.1 + CUDA 12.4 + pycvvdp 0.5.4 build. Stable; rebuild only if
  the pycvvdp pin moves.

The launcher refuses to start if `chunks.jsonl` is missing from R2.
Image tags are passed via `--env`; workers `docker pull` them at
boot to keep cold-pull jitter out of the chunk loop.

## Troubleshooting

**"chunks.jsonl missing"** — generator didn't run or upload didn't
land. Re-run step 1 + step 2.

**Many `[lost-claim]` entries in heartbeats** — workers racing on the
same `chunks.jsonl` order. The onstart shuffles chunks but with high
worker count + small chunk count, races happen. Wait it out; the
atomic-claim layer makes them safe (just wasteful).

**Per-chunk failures** — check `s3://coefficient/logs/<run>/<chunk>.fail.log`.
The chunk-worker captures stderr from the sweep + score-pairs
docker runs there. The claim is NOT released on failure (avoids
thundering-herd retries); other workers see the stale-claim and skip
after 600s if you want to retry.

**Parity diff blown out (max > 0.5 JOD)** — the
`cvvdp_imazen` implementation has regressed against pycvvdp.
Investigation entry-point: `crates/cvvdp-gpu/docs/CHROMA_DRIFT_INVESTIGATION.md`.
Diff per (codec, q) in the parity TSV to find which cells broke.

**Worker can't pull ghcr image** — `GHCR_TOKEN` env var missing.
The launcher sets it automatically from `gh auth token`; verify
`gh auth status` shows `write:packages` scope.

## When NOT to use this

This pipeline is the **backfill** path — given an existing unified
parquet, score it with both cvvdp implementations and produce
sidecars. If you're running a fresh sweep (new codec + knob grid),
use the v15 dispatcher (`scripts/sweep/v15/launch_gpu.sh`) which
runs sweep + scoring in one pass. The two pipelines share the
atomic-claim layer + R2 conventions; they don't share chunk shape
or per-chunk command.
