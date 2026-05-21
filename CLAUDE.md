# zenmetrics CLAUDE.md

See global ~/.claude/CLAUDE.md for general instructions.

## Data provenance — READ BEFORE TRAINING

**[`~/work/zen/DATA_PROVENANCE.md`](../DATA_PROVENANCE.md)** is the
canonical record of which R2 sidecars came from which codec commits.
Consult before training any picker / metric / regression on the
backfilled data — codecs like `jxl-encoder` shift RD curves between
commits, so mixing v22-produced and v23-produced JXL rows poisons the
fit. The doc records:

- R2 paths (input parquets, sidecars, encoded variants)
- Codec HEAD commit SHAs per backfill image (v22 / v23)
- Sidecar schema (column types + meanings)
- Reading recipes (pyarrow + s3fs)

Append a new section to that doc when you start a new backfill.

## PINNED TASK — CVVDP scoring on zensim training datasets

**Status: queued, multi-tick. Survives context compaction.** Do not
drop until shipped. User ask (verbatim, 2026-05-14, three messages):

> "we need cvvdp values for zensim development and all data sets it
> uses, we can use vastai dockerimages - but we should distonguish
> between different cvvdp implementations in col name"
> "parquet sidecars or however we do it"
> "enqueue this task so that a conpaction will not skip it, loop it"

### Requirements

1. **Compute CVVDP JOD scores** for every (ref, dist) pair zensim
   currently trains on. Inventory in
   `~/work/zen/zenanalyze/everything.md` — the "unified V_X store"
   at `/mnt/v/zen/zensim-training/2026-05-07/unified/` has
   2.37M rows × 7 codec/sweep parquets. Plus anchor sets:
   CID22, KADID-10k, TID2013, KonJND-1k.
2. **Distinguish implementations via column name** so multiple
   cvvdp variants land side-by-side without collision:
   - `cvvdp_pycvvdp_v054`  — canonical pycvvdp v0.5.4 reference
   - `cvvdp_gpu_imazen_<short_commit>` — our zenmetrics cvvdp-gpu
   - `cvvdp_burn_<short_commit>` — namespace reserved for a
     potential Burn-based port. Tick 324's spike (4.32× regression
     vs. the hand-written separable kernel at 4000×3000 on RTX 5070)
     ruled out the original Burn plan; the namespace stays free in
     case a future re-attempt wants to claim it. See
     `crates/cvvdp-gpu/docs/BURN_PORT_PLAN.md`'s "Status: ABANDONED"
     banner and `crates/burn-conv-spike/README.md`.
3. **Parquet sidecars**, matching zenmetrics' existing sweep
   convention (`image_path / codec / q / knob_tuple_json /
   <metric>_score / feat_*`).
4. **Compute infra: vast.ai docker images.** Reuse
   `Dockerfile.sweep.v26` (collapsed single-file image; replaced
   the v14→v25 chain on 2026-05-21) + `scripts/sweep/v15/launch_gpu.sh`
   + `scripts/sweep/onstart_v3.sh` per "Sweep runner discipline".
   pycvvdp installs ~3 GB of pytorch — its image must be separate
   from the cvvdp-gpu image to keep cold-start fetch under control.

### Progress markers (update each tick)

- [x] Inventory zensim parquet sidecars (paths, schemas, row counts):
      `/mnt/v/zen/zensim-training/2026-05-07/unified/` has 7 parquets,
      351 cols each, identity `(image_path, codec, q, knob_tuple_json)`,
      no cvvdp columns yet. 2.37M rows total.
- [x] cvvdp metric registered in `zen-metrics-cli` (it dispatches
      via `cvvdp_gpu::score` behind the `gpu-cvvdp` feature)
- [x] Versioned column name implemented (`cvvdp_gpu::CVVDP_COLUMN_NAME`,
      default `cvvdp_imazen_v<VER>`, `CVVDP_IMPL_TAG` env override).
      See commit b2b7f135.
- [x] Spec column schema (dtype, nullability, sidecar layout) —
      see `crates/cvvdp-gpu/docs/CVVDP_SIDECAR_SCHEMA.md`
- [x] pycvvdp scoring worker: `scripts/sweep/pycvvdp_worker.py`
      (consumes pairs TSV, writes parquet sidecar) +
      `scripts/sweep/Dockerfile.pycvvdp` (pytorch 2.5.1 + CUDA 12.4
      + pycvvdp 0.5.4). End-to-end verified locally on a synth pair
      (JOD 10.0 / 9.63 for identical vs chroma-shifted 64×64 inputs).
- [x] zen-metrics-cli `score-pairs` subcommand consumes the pairs
      TSV and writes parquet sidecars directly with the metric's
      versioned column name (cvvdp → `cvvdp_imazen_v<VER>`). The
      `Dockerfile.sweep.v26` image bakes `zen-metrics`, so the
      cvvdp-gpu scorer ships in that image with no new Dockerfile.
      Verified n=4 against pycvvdp on the same pairs: implementations
      agree within 0.03 JOD (q50–90, 64×64 noise images).
- [x] Encoder driver that re-encodes from
      `(image_path, codec, q, knob_tuple_json)` and emits the pairs
      TSV that the pycvvdp worker consumes:
      `zen-metrics-cli sweep --distorted-out-dir <DIR> --pairs-tsv <TSV>`
      (PNG fastest-effort dist images; deterministic filenames hashed
      on `(src_path, knob_json)`). End-to-end smoke-tested on a 2-image
      × 2-q grid: zen-metrics sweep → pycvvdp_worker score-pairs →
      4-row parquet sidecar with `cvvdp_pycvvdp_v054` column.
- [x] Per-chunk dual-implementation runner:
      `scripts/sweep/dual_impl_chunk.sh`. Drives one sweep + both
      scorers (cvvdp-gpu + pycvvdp) + a parity TSV side-by-side.
      Smoke-tested: 4/4 cells joinable, mean |diff| 0.0245 JOD,
      max 0.0300 JOD on the synth zenjpeg q50/q90 corpus.
- [/] Multi-instance dispatch (vast.ai fan-out wrapping
      `dual_impl_chunk.sh`) — chunk-claim from R2, run, upload
      sidecars back. Extends the existing v15 launcher rather than
      rebuilding from scratch. Chunk generator + worker + onstart +
      launcher all shipped (commits d2eb0f7c, 87deac34, 32a3b64a,
      c572c192). Push of corrected `zen-metrics-sweep:0.6.4-cvvdp-*`
      image still gated on a real-GPU smoke run (see 2026-05-15
      retry note below — local WSL2 can't satisfy the GATE).
- [/] Verification pass — local n=18 measured 2026-05-15 retry
      (decision D, q={30,70,90} × 6 zenwebp sources): cvvdp-gpu was
      forced through cubecl-cpu runtime (CUDA-in-snap-docker fails
      on this WSL2 host) where `atomic<f32>` panics short-circuit
      the kernel to default JOD=10.0, so the comparison reduces to
      "pycvvdp − 10.0" — mean |diff| 0.18, median 0.16, max 0.58.
      The parity number is unrepresentative of the gpu-cuda path
      and the GATE remains unverified.
- [ ] Production run + parquet write-back to
      `/mnt/v/zen/zensim-training/<date>/unified/`

### Local docker smoke 2026-05-15 — BUILT but NOT PUSHED

Built two images locally (canonical master HEAD aba984c context):

- `ghcr.io/imazen/pycvvdp-scorer:0.5.4` — sha256:e86bfb22aa82… (6.54 GB)
  - `pycvvdp-worker --help` works
  - End-to-end score-pairs CPU run on n=3 pairs: identical pair → JOD 10.000,
    cross pairs → 1.86 / 1.70. Image is functional.
- `ghcr.io/imazen/zen-metrics-sweep:0.6.4-aba984c` — sha256:30c2572f6891… (230 MB)
  - `zen-metrics --version` → `zen-metrics-cli 0.6.0`
  - `zen-metrics sweep ...` on n=6 sources × q={30,90}: 12/12 cells, no failures
  - `sweep --help` contains zenjpeg (Dockerfile RUN check passed at build)

**NOT PUSHED.** The `scripts/sweep/dual_impl_chunk_docker.sh` integration smoke
that the task's GATE was supposed to verify cannot run against the canonical-HEAD
v13 binary: `score-pairs` subcommand and `sweep --pairs-tsv` / `--distorted-out-dir`
flags live on `feat/cvvdp-gpu-scaffold` only (commit 14689621), NOT on canonical
master HEAD (aba984c). So the wrapper script's step 1 (`sweep --pairs-tsv …`)
exits with an unknown-flag error against the freshly-built 0.6.4-aba984c image,
and the parity GATE (mean |diff| < 0.10 JOD) is unverifiable.

Two paths to unblock:
1. Build the v13 image from the cvvdp branch instead (would require user override
   of decision (C) above), OR
2. Land the score-pairs / pairs-tsv subcommand on canonical master via a small
   targeted merge from feat/cvvdp-gpu-scaffold before rebuilding.

Secondary issue surfaced (local-env only, not a sweep design defect): snap docker
on this WSL2 box cannot bind-mount `/usr/lib/wsl/lib/libnvidia-ml.so.1` so
`--gpus all` fails with NVML ERROR_LIBRARY_NOT_FOUND. The pycvvdp CPU path still
works end-to-end; production vast.ai workers do not hit this.

### Local docker smoke 2026-05-15 (retry, decision D) — STILL NOT PUSHED

Picked up the unblock path 1 from the previous note: built v13 from the
`feat/cvvdp-gpu-scaffold` tree directly. New tag is
`ghcr.io/imazen/zen-metrics-sweep:0.6.4-cvvdp-76854e8` (the `-cvvdp-`
infix flags branch-origin to future readers).

Build-context approach: (b) materialised at
`/home/lilith/work/zen/_build-ctx-cvvdp-76854e8/` — rsynced scaffold
zenmetrics tree (excluding `target/`, `.jj/`, and the 6.7 GB
`scripts/cvvdp_goldens/`) plus trimmed `zenjpeg/` and `zenanalyze/`
siblings. Two builds were needed:

1. First build (canonical-features `--features sweep`, log
   `/tmp/build-rebuild.log`): produced an image with the scaffold's
   `score-pairs` subcommand and `sweep --pairs-tsv` flag (✓) but
   `cvvdp` disabled at runtime — `score-pairs --metric cvvdp` emits
   `metric 'cvvdp' is disabled (rebuild with --features gpu-cvvdp)`.
   This is the actual blocker the prior note missed: the Dockerfile
   never enabled GPU features, so even with the right subcommand
   shape, `score-pairs --metric cvvdp` cannot work against any
   image built straight off canonical's Dockerfile.sweep.v13.
2. Fixed by updating `Dockerfile.sweep.v13` to build with
   `--features sweep,gpu,gpu-cuda,gpu-cpu`. Second build (~5 min
   including 298 s LTO link) produced
   `sha256:d8786ca29428…` (576 MB, was 230 MB without GPU/cubecl).

`score-pairs --metric cvvdp` now reaches the cvvdp-gpu code, but on
this WSL2 box neither runtime path completes correctly:

- `--gpu-runtime cuda` panics at `cubecl-cuda runtime.rs:53` with
  `DriverError(CUDA_ERROR_OPERATING_SYSTEM)` even with libcuda.so.1
  + libnvidia-ml.so.1 bind-mounted at `/wsl-cuda/` and
  `LD_LIBRARY_PATH` set. Snap docker on WSL2 needs
  nvidia-container-toolkit, which isn't installed; task forbids
  fixing that.
- `--gpu-runtime cpu` (cubecl-cpu via tracel-mlir) hits
  `not yet implemented: This type is not implemented yet.
  atomic<f32>` in `cubecl-cpu compiler/visitor/elem.rs:38` once
  per pair. The score-pairs loop swallows the panic, writes
  `cvvdp = 10.0` as the default-fail value, and continues. Result
  is a parquet with all 18 rows at JOD 10.0 — kernel never
  executed.

Joinable n=18 (all 6 sources × q={30,70,90} mapped):
- `mean |diff| = 0.1805 JOD`
- `median |diff| = 0.1611 JOD`
- `max |diff|   = 0.5790 JOD`

GATE thresholds were `mean < 0.10` and `max < 0.30` — both miss.
But these numbers measure "pycvvdp − 10.0" because the imazen
column is the fall-through default, not the real cvvdp-gpu output.
The GATE remains unverifiable on this host.

Wrapper fixes shipped this tick (commit `f7e321b6`):

- `Dockerfile.sweep.v13`: `--features sweep,gpu,gpu-cuda,gpu-cpu`
  plus inline `score-pairs --help` / `list-metrics` sanity checks
  in the same RUN.
- `scripts/sweep/dual_impl_chunk_docker.sh`:
  - `--entrypoint /usr/local/bin/zen-metrics` for the zen-metrics
    image (prior wrapper assumed CMD — production entrypoint is
    `zen-metrics-worker`, which then expected R2 env vars).
  - `ZEN_GPU_RUNTIME` env var forwarded as `--gpu-runtime` to
    `score-pairs` for runtime override (defaults to letting the
    binary auto-detect).

What's still needed:

1. A real-GPU smoke run on a CUDA-capable host (vast.ai box or any
   non-snap, non-WSL2 box with nvidia-container-toolkit). With
   --gpus all and `--gpu-runtime cuda`, the n=18 parity should
   land in the < 0.10 JOD band (consistent with the n=4 sentinel
   that already passed at 0.03 JOD).
2. Push of both images is still gated on that real-GPU smoke.

### Blocker note

`crates/cvvdp-gpu/docs/CHROMA_DRIFT_INVESTIGATION.md` (0.117 JOD on
chroma_shift fixture, narrowed through tick 200) gates only the
`cvvdp_gpu_imazen_*` column. The `cvvdp_pycvvdp_v054` column can
ship now via `scripts/cvvdp_goldens/.venv`'s already-working pycvvdp
installation. Don't block the pycvvdp column on cvvdp-gpu parity.

### Loop integration

Every /loop tick should re-read this section and consider whether
the next concrete improvement should advance the PINNED TASK rather
than the next cvvdp-gpu kernel parity tick. If the pinned task has
forward progress available (a missing inventory, an unwritten spec,
an unbuilt Docker image) prefer it over yet another parity test.

## Local CUDA toolkit (for building/running GPU metrics)

The water-cooled 7950X workstation has CUDA 13.2.1 SDK installed at the
default location, but **nvcc is not on PATH by default**. CUDA layout:

    /usr/local/cuda            → /usr/local/cuda-13.2  (current symlink)
    /usr/local/cuda-13.2/bin/nvcc
    /usr/local/cuda-13.2/lib64/  (libcudart.so etc.)

Other versions also installed: 12.6, 13. Use `/usr/local/cuda` (the
symlink) unless you have a reason for a specific version.

To compile a `cargo` invocation that needs nvcc, prepend:

    PATH=/usr/local/cuda/bin:$PATH
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

But note: **cubecl-cuda dynamically loads CUDA at runtime** via dlopen,
so building `--features sweep,gpu,gpu-cuda` succeeds even with nvcc off
PATH. The runtime fallback is sufficient for `zen-metrics` builds. Set
PATH explicitly only when shelling out to nvcc directly.

GPU info: `nvidia-smi` driver 596.21 / CUDA capability runtime 13.2.

## Sweep build cheat sheet

- **Default CPU+GPU build (development)**:
  `cargo build --release -p zen-metrics-cli`
  → includes both `cpu-metrics` (default) and `sweep` codecs. ~2 min cold,
  seconds incremental.

- **Forced GPU-only sweep build (production worker)**:
  `cargo build --release -p zen-metrics-cli --no-default-features --features sweep,png,gpu,gpu-cuda`
  → drops cpu-metrics so CPU butteraugli/zensim/ssim2 are *unavailable*;
  any chunk specifying a CPU metric will fail loudly. Use this for vast.ai
  workers so they can't silently fall back to slow CPU scoring. ~4 min cold.

- **WGPU variant (broader GPU compatibility, no CUDA SDK required)**:
  `cargo build --release -p zen-metrics-cli --no-default-features --features sweep,png,gpu,gpu-wgpu`
  → uses Vulkan/Metal/DX12 via wgpu. Use when targeting AMD/Intel GPUs
  on vast.ai. CUDA NVIDIA GPUs work but CUDA backend is faster.

## Sweep runner discipline

- **GPU metrics only on production workers.** Mixing CPU/GPU scores
  across a sweep produces inconsistent training data — pickers/trainers
  expect a single metric backend. The forced-GPU build above prevents
  accidental fall-back.
- **Pre-uploaded binary lives at**
  `s3://coefficient/binaries/zen-metrics-<version>-linux-x86_64`
  (R2 endpoint: `${R2_ACCOUNT_ID}.r2.cloudflarestorage.com`). Workers
  fetch via `SWEEP_BIN_OVERRIDE` env var.
- **Onstart script**: `scripts/sweep/onstart_v3.sh`. Fans out N parallel
  zen-metrics processes per box (one per CPU core) sharing the GPU for
  scoring; each claims its own chunk from `chunks.jsonl` on R2.
- **Every onstart MUST self-destroy on failure** — upload tail log to
  R2 + issue `vastai destroy instance ${CONTAINER_ID}`. See
  `scripts/sweep/CLAUDE.md#critical-every-onstart-must-self-destroy-on-failure`
  for the two acceptable patterns (image-level
  `run_with_error_trap.sh` wrapper on v15+, or inline `on_exit` trap
  as in `onstart_iwssim_backfill_v14.sh`). Workers that exit without
  destroying burn \$/hr until externally cleaned up — that's the
  cost-leak the 2026-05-18 EXP-LARGER-LARGE incident chased.

## CHANGELOG.md

Maintained in repo root.
