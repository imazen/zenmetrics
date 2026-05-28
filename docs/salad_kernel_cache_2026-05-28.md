# Salad sweep image: kernel-cache warmup + multi-column score-pairs (2026-05-28)

Tag: **`ghcr.io/imazen/zen-metrics-sweep-salad:v4-kernel-cache`**
(SHA-suffixed mirror: `v4-kernel-cache-5890a58f-multicol`)
Image digest: `sha256:b837f08471de4b1eb3adbeb08e4ac3d5a8720fbe36d990b7087fd381729e5cf1`
Image size: **997 MB** (same as `:v3`; new layers contribute <2 MB net)
Built from: zenmetrics master `5890a58f` + this work landed at commit
`5355493d`.

## Goal

Make Salad workers start processing jobs *fast*. Earlier `:v1`–`:v3`
images had no kernel-cache discipline: every fresh container paid the
full cubecl NVRTC compile cost on its first kernel invocation — measured
in prior work at ~18 s for `ssim2-gpu` alone on an RTX 3060, and the
6-metric default set would compound that to ~60–90 s of pure compile
time poisoning the first chunk's wall.

This work brings the compile cost into a **bracketed boot phase**
(visible in the entrypoint log) so the first sidecar-POST'd job
runs against pre-compiled kernels.

## cubecl-cuda's on-disk PTX cache (research)

Found at `zenforks-cubecl-cuda 0.10.1` (crates.io) /
`crates/cubecl-cuda/src/compute/context.rs`. Mechanism:

- `CudaContext::new` reads `CubeClRuntimeConfig::get()` (a global
  singleton built once from `cubecl.toml` walking up from cwd, or
  defaults).
- When `compilation.cache: Some(CacheConfig::*)` is set, a
  `CompilationCache<StableHash, PtxCacheEntry>` is constructed rooted
  at the resolved path. Entries serialise the `Vec<c_char>` PTX
  bytes + `entrypoint_name` + `shared_mem_bytes`.
- `compile_kernel` first hashes the `KernelId::stable_hash()`
  (type_name + address_type + cube_dim + mode + info). Cache hit →
  `load_ptx` straight to driver. Cache miss → NVRTC compile → insert
  into cache → load.

**Cache path layout** (post the per-arch patch shipped in `0.10.1`):
```
<root>/cuda/<cubecl-common version>/<cubecl_sha>/sm_<arch>/<driver>/ptx.json.log
```
This makes the cache key safe across:
- cubecl crate-graph rev advances (the `cubecl_sha` segment, from
  `option_env!("CUBECL_GIT_SHA")` in `build.rs`).
- Heterogeneous GPU fleets (the `sm_<arch>` segment) — critical for
  Salad's RTX 3060 (sm_86) ↔ RTX 4090 (sm_89) mixed pool.
- CUDA driver bumps (the driver-version segment via
  `cudarc::driver::result::version::driver()`).

The driver's own PTX→SASS cache at `~/.nv/ComputeCache/` sits
downstream of this; cubecl's layer is upstream and feeds compiled
PTX to it.

**Critical finding:** `CompilationConfig.cache = None` is the
default. Without an explicit cubecl.toml or programmatic
`CubeClRuntimeConfig::set` enabling it, **the cache code path
compiles but is never used**. This is the root cause of the
`:v1`–`:v3` cold-start cost — the runtime had no PTX cache because
none of those images shipped a `cubecl.toml`.

### Why pre-warming at docker-build is impractical for Salad

The cache is per-`sm_<arch>`. Salad assigns containers across
3060/3080/3090/4060/4070/4080/4090 etc. (sm_86 through sm_89). A
build-time pre-warm on the operator's GPU (here: RTX 5070 = sm_120)
would write `sm_120/ptx.json.log`; nodes with sm_86 hit a sibling
miss and recompile anyway. Worse, the build host lacks GPU
passthrough during typical docker-build, so NVRTC can't even run.
**Runtime warmup is the only universal answer** for a
heterogeneous fleet.

## Implementation

### 1. Bake `cubecl.toml` into the image WORKDIR

```toml
# scripts/sweep/cubecl.toml (baked at /workspace/salad-sweep/cubecl.toml)
[compilation]
cache = { file = "/var/cache/cubecl" }
```

(`CacheConfig`'s `File(PathBuf)` variant deserialises as
`{ file = "..." }` — confirmed empirically against the cubecl-runtime
serde derive in `/tmp/cachetest`.)

The cache dir is mkdir'd at image build time. cubecl walks up from
the worker's cwd looking for `cubecl.toml`, finds the one we baked,
and enables the cache.

### 2. Bake warmup fixtures

- `/opt/zen/warmup/ref_64.png` + `dist_noisy_64.png` (existing
  test fixtures, ~9 KB each).
- `/opt/zen/warmup/ref_256.png` + `dist_noisy_256.png` (newly
  generated, deterministic seed `0xDEADBEEF`) — needed because
  `iwssim-gpu` rejects images smaller than 176×176.

### 3. `scripts/sweep/warmup_kernels.sh`

Invokes `zen-metrics score-pairs --metric <X>` once per default GPU
metric on the appropriate-size fixture. Reports per-metric wall time
+ total wall + cache-dir size. Fail-soft: a warmup error does NOT
abort the boot; the worker still launches and a genuine GPU break
surfaces via the durable error-sidecar on the first real job.

```
[warmup] GPU: NVIDIA GeForce RTX 5070, 12.0, 596.21
[warmup]   zensim-gpu: OK in 1.12s
[warmup]   ssim2-gpu: OK in 0.61s
[warmup]   butteraugli-gpu: OK in 1.16s
[warmup]   cvvdp: OK in 0.70s
[warmup]   dssim-gpu: OK in 0.67s
[warmup]   iwssim-gpu: OK in 0.92s
[warmup] cubecl cache: 1.7M in 2 file(s) at /tmp/cubecl_cache_test_dir
[warmup] warmup total: 5.22s
```

### 4. Entrypoint integration

`scripts/sweep/entrypoint_salad.sh` runs the warmup pass between
env-hydration and sidecar+worker launch:

```bash
cd "${WORKDIR:-/workspace/salad-sweep}"     # so cubecl finds cubecl.toml
warmup_kernels.sh || log "warmup script returned non-zero (continuing per fail-soft policy)"
log "warmup phase: ${warmup_elapsed}s total wall"
salad-http-job-queue-worker &
zen-sweep-worker worker --backend salad ... &
wait -n
```

## `zen-metrics score-pairs` multi-column fix

User feedback during this work: "fix whatever limits metrics to a
single score, they often produce multiple named values."

The warmup script first had to special-case `butteraugli-gpu` to
route through `batch` instead of `score-pairs`, because the latter
errored on metrics with `column_names().len() != 1`. The 2-column
case was rejected with:

> `score-pairs supports single-column metrics only; butteraugli-gpu emits 2 columns (...). Use `batch` for now.`

Fixed in `crates/zen-metrics-cli/src/main.rs`:

- `cmd_score_pairs` now enumerates `MetricKind::column_names()` and
  writes one parquet Float64 column per metric column alongside the
  identity-tuple columns (image_path / codec / q / knob_tuple_json).
- `score_one_pair` returns `Vec<f64>` aligned with `column_names()`;
  cvvdp's cached-scorer fast path emits `vec![v]`.
- Loud per-row diagnostic when a metric returns a different column
  count from what `column_names()` declares; pad/truncate to keep
  schema width stable.

Verified:
```
$ zen-metrics score-pairs --metric butteraugli-gpu ... --out-parquet x.parquet
[score-pairs] wrote 1 rows (0 NaN-failures) to x.parquet with score column(s) ["butteraugli_max_gpu", "butteraugli_pnorm3_gpu"]

$ python -c 'import pyarrow.parquet as pq; t = pq.read_table("x.parquet"); print(t.column_names)'
['image_path', 'codec', 'q', 'knob_tuple_json', 'butteraugli_max_gpu', 'butteraugli_pnorm3_gpu']
```

Single-column metrics (ssim2-gpu, cvvdp, dssim-gpu, iwssim-gpu,
zensim-gpu) round-trip unchanged. Warmup script simplified to a
single `score-pairs` call per metric (no `batch` special case).

## Local measurements (RTX 5070, sm_120, driver 596.21)

The 5070 has a populated `~/.nv/ComputeCache/` (1.1 GB) so the
*driver* PTX→SASS cache is already warm in every measurement. Only
the *cubecl-level* in-process compile + JIT-fetch cost varies.

Warmup script over all 6 default GPU metrics (zensim-gpu, ssim2-gpu,
butteraugli-gpu, cvvdp, dssim-gpu, iwssim-gpu), one pair each:

| Scenario | Total wall | Per-metric mean | PTX written |
|---|---:|---:|---:|
| `cubecl.toml` absent (cache disabled) | 10.45 s | 1.74 s | 0 bytes |
| `cubecl.toml` present, cold cache | 11.96 s | 1.99 s | 1.7 MB |
| `cubecl.toml` present, warm cache | **5.22 s** | **0.87 s** | 0 new bytes |

On the 5070, the warm-cache hit is **~5.2 s faster than the
no-cache baseline** — about half the warmup cost. The savings are
amplified on Salad's commodity consumer GPUs because (a) the NVRTC
compile cost is higher (3060 measured ~18 s for ssim2-gpu in
prior work; mid-tier x86 host CPUs are slower than the dev box's
7950X), and (b) the driver-cache hit ratio on a fresh Salad node is
0%.

**Expected per-Salad-worker boot cost with v4-kernel-cache:**
- First container ever on a node: ~30–90 s warmup wall (NVRTC
  compile + cubecl disk-write), then PTX cached on local disk.
- Container restart on the same node: ~5–15 s warmup wall (PTX
  read + load_ptx). The disk path is in the container's writable
  layer; Salad's container lifecycle determines whether it survives
  across instance reboots.
- Without v4: indistinguishable from "container hangs" for the
  first ~60–90 s of real job processing, because the first sidecar-
  POST'd job has to wait for kernels to compile inline.

## Scale-up test (N=10 replicas) — not run this session

Phase 4 of the original brief (a scale-up test on a real Salad
container group with chunks pushed through the managed queue) was
not run. Note: Salad's default org quota is **10 container replicas
total** (verified via `GET /organizations/imazen/quotas`:
`container_replicas_quota: 10`), so the brief's "N=20" target needs
to be reduced to N=10 unless a quota bump is requested via Salad
support. The remaining work is:

1. A launcher tool. `crates/zen-cloud-salad/src/launch.rs` has the
   API client (`SaladApi`) and scoped-cred minting (`create_
   container_group_with_scoped_cred`). What's missing is a CLI
   binary or example that calls the launcher end-to-end:
   `resolve_gpu_class → mint_sweep_r2_cred → create_queue →
   create_container_group → push_jobs → poll group / R2 sidecars
   → stop_container_group`.
   - Suggested location: `crates/zen-cloud-salad/examples/
     run_scaleup.rs` or a new `crates/zen-cloud-salad/src/bin/
     salad-launcher.rs`.
2. A throwaway chunks generator. ~40–80 small `ChunkRecord` JSON
   lines, each pointing at maybe 1–2 image basenames in a public
   R2 source dir + a tiny output prefix. Pushed to R2 at
   `s3://coefficient/jobs/<run>/chunks.jsonl` so the worker's
   `--chunks-r2` arg resolves them.
3. Timing harvest from:
   - Salad API `current_state` polls (allocations / state
     transitions: `pending → starting → running`).
   - R2 sidecars (per-chunk `omni/*.parquet` upload mtime ≈
     chunk-done time).
   - Worker entrypoint logs visible via Salad's portal-side stream
     (or our durable error sidecars on failure).
4. Spend budget. Each replica costs roughly $0.10–$0.30/hour
   depending on GPU class. 10 × $0.20/h × 20 min hard cap ≈ $0.67
   worst case, well under $5. Tear-down: `SaladApi::stop_
   container_group(name)` + delete the queue + (optional) clean
   up the container group resource itself.

The mechanism to *measure* the warmup savings on real Salad GPUs
requires the above launcher to land first. Without it, the savings
are projected from the local 5070 numbers + the prior 18-s-per-
metric 3060 figure.

## Files touched

- `Dockerfile.sweep.salad.v1` — added L9c (cubecl.toml + warmup
  fixtures + warmup script bake) right before the entrypoint COPY.
- `scripts/sweep/cubecl.toml` — new; enables cubecl PTX cache.
- `scripts/sweep/warmup_kernels.sh` — new; runs `score-pairs` per
  default GPU metric on baked fixtures.
- `scripts/sweep/entrypoint_salad.sh` — calls warmup before
  sidecar+worker launch.
- `crates/zen-metrics-cli/src/main.rs` — `cmd_score_pairs` accepts
  multi-column metrics; `score_one_pair` returns `Vec<f64>`.
- `crates/zen-metrics-cli/tests/fixtures/{ref,dist_noisy}_256.png` —
  256×256 fixtures for iwssim-gpu's 176-px minimum.

## How to reproduce locally

```bash
# 1. Build zen-metrics + zen-sweep-worker:
cd zenmetrics
CUDARC_CUDA_VERSION=12000 cargo build --release -p zen-metrics-cli \
    --no-default-features --features 'sweep,png,gpu,gpu-cuda,gpu-cpu'
CUDARC_CUDA_VERSION=12000 cargo build --release -p zen-sweep-worker \
    --features salad-sweep

# 2. Stage binaries for docker COPY:
cp target/release/zen-metrics ./zen-metrics
cp target/release/zen-sweep-worker ./zen-sweep-worker

# 3. Build + push the image:
docker build -f Dockerfile.sweep.salad.v1 \
    -t ghcr.io/imazen/zen-metrics-sweep-salad:v4-kernel-cache .
docker push ghcr.io/imazen/zen-metrics-sweep-salad:v4-kernel-cache

# 4. Local warmup smoke (no Salad needed):
mkdir -p /tmp/wcwd && cp scripts/sweep/cubecl.toml /tmp/wcwd/
cd /tmp/wcwd
PATH=$(pwd)/../zenmetrics/target/release:$PATH \
WARMUP_DIR=crates/zen-metrics-cli/tests/fixtures \
CUBECL_CACHE_DIR=/var/cache/cubecl \
bash scripts/sweep/warmup_kernels.sh
```
