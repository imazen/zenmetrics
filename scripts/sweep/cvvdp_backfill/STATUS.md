# cvvdp-backfill — vast.ai run status (2026-05-16)

## TL;DR

BLOCKED on cudarc-binds-ahead-of-shipped-driver. Every fleet
attempt (14 so far) panics at score-pairs --metric cvvdp with:

```
thread 'DSD-0-0' panicked at cudarc/src/driver/sys/mod.rs:22536:
Expected symbol in library: DlSym {
    source: "/usr/local/cuda/compat/libcuda.so: undefined symbol:
             cuCoredumpDeregisterCompleteCallback"
}
```

The symbol is gated in cudarc behind
`#[cfg(any(feature = "cuda-13020"))]`. cubecl-cuda's Cargo.toml
declares cudarc with `cuda-version-from-build-system` +
`fallback-latest` features, so the build host's CUDA SDK version
determines what symbols cudarc emits. Our local build host has
CUDA 13.2 SDK installed, baking in cuda-13020 symbols.

**The symbol does not exist in any released NVIDIA driver yet** —
verified by inspecting libcuda.so.580.65.06 (driver 580, CUDA 13.0)
and libcuda.so.590.48.01 (driver 590, CUDA 13.1.2). Both ship only
4 Coredump-related entry points: Get/Set Attribute / AttributeGlobal.
DeregisterComplete and RegisterComplete are absent from every
released driver.

## Fleet history (all destroyed pre-completion)

| Fleet | Filter | Failure | Notes |
|---|---|---|---|
| v1-v3 | (early variants) | docker-in-docker / chunk_worker entrypoint / docker missing | Ticks 360-363 |
| v4 (36851086+) | imazen-only switch | pip3 missing on PATH | Tick 365 part 1 |
| v5 (36851199+) | + R2 wrapper | s5cmd [default] AWS profile | Tick 365 part 2 |
| v6 (36851303+) | + missing source uploads | sweep stderr was masked | Tick 366 |
| v7 (36851382+) | + sweep stderr capture | `GROUPS` shadowed bash built-in | Tick 367 — **the root cause finally surfaced** |
| v8-v9 (36851796+) | + GROUP_LINES rename | step 4 succeeded for the first time; step 5a cudarc panic | Tick 368 |
| v10 (36852251+) | `cuda_max_good>=13` filter | same | |
| v11 (36852292+) | `cuda_vers>=12.5` | same | |
| v12 (36852404+) | `cuda_vers>=13.0` | same | |
| v13 (36852552+) | `compute_cap>=860 cuda_vers>=13.0` Ampere+ | same | Tick 369 |
| v14 (36853111+) | + new image FROM nvidia/cuda:13.0.0-base | same — now at `/usr/local/cuda/compat/libcuda.so` | Tick 370-372 |

## Confirmed working components

| Stage | Status |
|---|---|
| docker push (ghcr.io) | ✓ done |
| R2 parquet upload (7 unified parquets) | ✓ done |
| R2 source images (817 needed basenames) | ✓ done |
| Boot bootstrap (python3+pip3+pyarrow) | ✓ working |
| chunk claim + R2 atomic-token | ✓ working |
| Step 1 (parquet download) | ✓ working |
| Step 2 (sources sync) | ✓ working |
| Step 3 (slice + group) | ✓ working |
| Step 4 (sweep — encode + ssim2-score) | ✓ working — 100/100 cells succeed on every chunk |
| Step 5a (score-pairs cvvdp) | ✗ BLOCKED — cudarc DlSym panic |

## True root cause

`cudarc` 0.19.4's `cuCoredumpDeregisterCompleteCallback` is gated
behind `cuda-13020`. cubecl-cuda's deps:
```
[dependencies.cudarc]
features = [
    "std", "driver", "nvrtc", "nccl",
    "fallback-dynamic-loading",
    "cuda-version-from-build-system",
    "fallback-latest",
]
```

`cuda-version-from-build-system` resolves to whatever CUDA SDK
is detected at compile time. Without an SDK, `fallback-latest`
picks cuda-13020 (highest defined). Either path lands on the
unsupported symbol.

## Fix paths (in order of invasiveness)

1. **Install CUDA 13.0 SDK in Dockerfile builder stage.** The
   current builder is `rust:1.93-slim-bookworm` with NO CUDA SDK,
   so `fallback-latest` kicks in. Adding the NVIDIA CUDA 13.0
   apt repo + `cuda-toolkit-13-0` package + setting `CUDA_HOME`
   pins the build to 13000 features. ~10 min build addition,
   ~500 MB image-size increase, one-time.

2. **Vendor cubecl-cuda with patched Cargo.toml.** Replace
   `cuda-version-from-build-system` + `fallback-latest` with an
   explicit `cuda-13000` feature. Cleanest but adds a vendored
   crate to maintain.

3. **Wait for cudarc upstream.** Their feature gating ahead of
   shipped drivers is a real bug — file an issue and wait.

## What's pushed but blocked

- `ghcr.io/imazen/zen-metrics-sweep:0.6.4-cvvdp-libcuda` (913MB).
  The image's libcuda is correct; the binary inside it isn't.
  Superseded by the next rebuild.

## Cost so far

- ~14 fleet attempts, all destroyed pre-data.
- Total $ burn: probably $3-5.
- User authorized ~$50; well under budget.

## Recommendation

Path **(1)** — install CUDA 13.0 SDK in the Dockerfile builder
stage and rebuild. Most pragmatic. Estimated cost: 10 min build
addition + 30 min push + 10 min relaunch+verify. Should produce
the first working chunks once cudarc is compiled against the
correct CUDA SDK target.
