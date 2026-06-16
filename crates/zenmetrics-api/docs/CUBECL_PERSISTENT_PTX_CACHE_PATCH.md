# Persistent PTX cache — patch + cache-key design

**Status: design doc + ready-to-apply patch.** This document is the
deliverable for `PHASE8_PLAN.md` Phase 8e.2 + 8e.3. The patch is staged
here as a `git format-patch`-style diff against
`lilith/cubecl@de2f9857` (the rev our workspace currently pins); it is
**not** applied to the cubecl checkout. Per CLAUDE.md, this repo does
not modify other repos in place. Execution (clone fork, apply diff,
test, push branch, bump our workspace pin) is user-driven follow-on.

The headline of the investigation is that **cubecl-cuda already has a
persistent PTX cache** (`crates/cubecl-cuda/src/compute/context.rs:70`,
backed by `CompilationCache<StableHash, PtxCacheEntry>` from
`cubecl-common::cache::Cache`). The "fresh-process cold start =
~18 s of NVRTC" symptom we observed is **the cache key being too narrow
for our usage**, not a missing cache layer. Two structural cases miss:

1. **Bumping the cubecl fork rev for codegen-only changes** — the cache
   `version` parameter defaults to `cubecl-common`'s `CARGO_PKG_VERSION`,
   not the cubecl crate-graph HEAD SHA. Codegen lives in `cubecl-cuda` /
   `cubecl-core` / per-compiler crates. Bumping our fork by N commits
   without changing `cubecl-common`'s `Cargo.toml` version field leaves
   the cache reading stale PTX into the new codegen pipeline.
2. **Multi-GPU dev boxes** — the cache directory is per-cubecl-version
   but not per-compute-capability. A workstation that swaps a 5070
   (sm_120) for a 5090 (sm_120) on the same project gets stale-key
   safety only because the compute_cap matches. Swap to a workstation
   with sm_89 or sm_70, and the cache happily hands sm_120-targeted
   PTX to a different architecture. The PTX would either fail to load
   or — worse — succeed with subtle codegen mismatches.

Cubecl-cuda invokes NVRTC with `--gpu-architecture=sm_<arch>`
(`context.rs:147-151`), so the PTX is architecture-specific. The cache
key MUST include the architecture or it's a correctness bug.

The patch below makes the cache key path layout:

```
<root>/cuda/<cubecl-common version>/<cubecl_sha>/<compute_cap>/<cuda_runtime>/ptx.json.log
```

…where each new axis is appended as a path segment so a cubecl-version /
compute-cap / runtime-version mismatch reads a sibling file, not the
same one.

---

## Findings — current cache behaviour (de2f9857)

### Where it lives

`crates/cubecl-cuda/src/compute/context.rs:60-88` constructs the cache
inside `CudaContext::new`:

```rust
ptx_cache: {
    use cubecl_runtime::config::RuntimeConfig;
    let config = cubecl_runtime::config::CubeClRuntimeConfig::get();
    if let Some(cache) = &config.compilation.cache {
        let root = cache.root();
        Some(CompilationCache::new(
            "ptx",
            CacheOption::default().name("cuda").root(root),
        ))
    } else {
        None
    }
},
```

`cache.root()` resolves to one of (per `cubecl-runtime/src/config/cache.rs`):

- `CacheConfig::Target` (default) → walks up from `cwd` looking for
  `Cargo.toml`; on hit, `<workspace>/target/`. On miss, `dirs::cache_dir().join("cubecl")`.
- `CacheConfig::Local` → `cwd`.
- `CacheConfig::Global` → `dirs::config_local_dir()`.
- `CacheConfig::File(p)` → caller-specified.

The final file path under cubecl-common's `get_persistent_cache_file_path`
is `<root>/<name>/<version>/<path>.json.log` →
`<root>/cuda/<cubecl-common-CARGO_PKG_VERSION>/ptx.json.log`.

### Where compile happens

`crates/cubecl-cuda/src/compute/context.rs:97-237` (`compile_kernel`).
The flow:

1. If cache is enabled: hash the `KernelId` (`stable_hash`), look up in
   the cache. Hit → call `load_ptx(entry.ptx, …)` and return.
2. Miss → call NVRTC: `nvrtc::create_program` →
   `nvrtc::compile_program` → `nvrtc::get_ptx`.
3. Insert the resulting PTX into the cache (`cache.insert(hash, PtxCacheEntry { ptx, … })`).
4. `load_ptx(ptx, …)` to upload PTX → driver module.

### What's in `KernelId::stable_hash`

`crates/cubecl-runtime/src/id.rs:129-141`:

```rust
pub fn stable_hash(&self) -> StableHash {
    let mut hasher = StableHasher::new();
    self.type_name.hash(&mut hasher);
    self.address_type.hash(&mut hasher);
    self.cube_dim.hash(&mut hasher);
    self.mode.hash(&mut hasher);
    self.info.hash(&mut hasher);
    hasher.finalize()
}
```

The hash captures the **kernel's identity** (type, dims, mode, info).
It does NOT capture:
- The kernel's source code bytes (we hash the kernel struct's type info,
  not the codegen output — different cubecl versions can produce
  different PTX from the same `KernelId`).
- The CUDA architecture targeted (`sm_<arch>`).
- The CUDA driver version.
- The cubecl crate-graph HEAD SHA.

The cache **disk layout** mitigates the cubecl-version case via the
`<cubecl-common version>` path segment, but only when:

- `cubecl-common`'s `Cargo.toml` version bumps. Our fork rev advance
  doesn't bump it.
- The cache files are per-host (not shared across machines via NAS /
  blob storage). If they're shared, the architecture mismatch becomes
  a real bug.

---

## Patch — what it changes

Three additive changes, all in `cubecl-cuda/src/compute/context.rs`:

### Change 1 — append cubecl_sha + compute_cap + cuda_runtime to the cache path

The current cache opens as:

```rust
CompilationCache::new(
    "ptx",
    CacheOption::default().name("cuda").root(root),
)
```

After patch:

```rust
let cubecl_sha = option_env!("CUBECL_GIT_SHA").unwrap_or("unknown");
let compute_cap = format!("sm_{}", arch);  // arch in scope from CudaContext::new
let cuda_runtime = cuda_runtime_version_string();  // see helper below

CompilationCache::new(
    "ptx",
    CacheOption::default()
        .name("cuda")
        .root(root)
        // version stays default (CARGO_PKG_VERSION) — leaves backward-
        // compatible disk layout for callers not setting CUBECL_GIT_SHA.
        // The three new axes are appended via the path argument so they
        // contribute to the directory layout without colliding with the
        // existing version segment.
)
```

…then the `"ptx"` path argument to `CompilationCache::new(...)` is
replaced with:

```rust
let cache_path = format!("{cubecl_sha}/{compute_cap}/{cuda_runtime}/ptx");
CompilationCache::new(cache_path, …)
```

Resulting on-disk layout:

```
<root>/cuda/<cubecl-common-version>/<cubecl_sha>/<compute_cap>/<cuda_runtime>/ptx.json.log
```

Any of the three trailing axes changing reads a sibling file. Stale
PTX safety becomes structural rather than version-string-dependent.

### Change 2 — `build.rs` capturing the cubecl fork's git SHA

Add a `cubecl-cuda/build.rs`:

```rust
fn main() {
    // Try to detect the cubecl crate's git SHA (when built from a git
    // checkout, e.g. our lilith/cubecl fork pinned via Cargo workspace).
    // Falls back to the upstream crate's `CARGO_PKG_VERSION` when built
    // from crates.io (no .git directory present).
    let sha = std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .current_dir(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .output()
        .ok()
        .and_then(|o| if o.status.success() {
            Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
        } else {
            None
        })
        .unwrap_or_else(|| std::env!("CARGO_PKG_VERSION").to_string());

    println!("cargo:rustc-env=CUBECL_GIT_SHA={}", sha);

    // Rerun build.rs when the working tree changes (best effort — on
    // released crate this never re-runs because there's no .git).
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let git_head = std::path::Path::new(&manifest).join("../../.git/HEAD");
    if git_head.exists() {
        println!("cargo:rerun-if-changed={}", git_head.display());
    }
}
```

The `option_env!("CUBECL_GIT_SHA")` call in `context.rs` reads this at
compile time. If the env var is absent (older builds, alternate
checkout layouts), `unwrap_or("unknown")` keeps the cache functional
with a degraded key.

### Change 3 — `cuda_runtime_version_string()` helper

`cubecl-cuda` already depends on `cudarc`. We can query the CUDA
driver version through it. New helper at the bottom of
`context.rs`:

```rust
fn cuda_runtime_version_string() -> String {
    // CUDA's driver-version API returns an integer like 12080 for 12.8.0.
    // We render as "12.8.0" so the directory name is human-readable.
    unsafe {
        match cudarc::driver::result::version::driver() {
            Ok(v) => {
                let major = v / 1000;
                let minor = (v % 1000) / 10;
                let patch = v % 10;
                format!("{major}.{minor}.{patch}")
            }
            Err(_) => "unknown".to_string(),
        }
    }
}
```

`cudarc::driver::result::version::driver()` calls `cuDriverGetVersion`
(per cudarc 0.16's driver/result module). The exact import path may
need adjusting against the cudarc version cubecl pins; if cudarc
0.16+ moved this function, the helper picks up the new path. Worst
case the helper returns `"unknown"` and the cache becomes invalidate-
when-cubecl-sha-changes-only.

---

## Patch — full diff

This is staged but **NOT applied** in this repo. Apply on
`lilith/cubecl@de2f9857` (or equivalent fork branch) as part of the
`feat/persistent-cache` branch per `CUBECL_FORK_STRATEGY.md`.

```diff
diff --git a/crates/cubecl-cuda/build.rs b/crates/cubecl-cuda/build.rs
new file mode 100644
index 00000000..0a1b2c3d
--- /dev/null
+++ b/crates/cubecl-cuda/build.rs
@@ -0,0 +1,29 @@
+//! Capture the cubecl crate-graph HEAD SHA so the persistent PTX cache
+//! key includes it. Without this, bumping our fork rev for codegen-only
+//! changes (where `cubecl-common`'s `Cargo.toml` version is unchanged)
+//! leaves the cache reading stale PTX into the new codegen pipeline.
+//!
+//! Falls back gracefully when built from crates.io (no .git directory)
+//! or in CI sandboxes that strip git history — the cache key just
+//! degrades to the package-version-only path layout.
+
+fn main() {
+    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
+    let sha = std::process::Command::new("git")
+        .args(["rev-parse", "--short=12", "HEAD"])
+        .current_dir(&manifest_dir)
+        .output()
+        .ok()
+        .and_then(|o| {
+            if o.status.success() {
+                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
+            } else {
+                None
+            }
+        })
+        .unwrap_or_else(|| std::env!("CARGO_PKG_VERSION").to_string());
+
+    println!("cargo:rustc-env=CUBECL_GIT_SHA={}", sha);
+
+    // Best-effort: re-run when the cubecl repo HEAD moves. Walks up
+    // from the manifest dir to find a .git directory (handles
+    // checkout-as-workspace-member layouts).
+    let mut p = std::path::PathBuf::from(&manifest_dir);
+    loop {
+        let head = p.join(".git").join("HEAD");
+        if head.exists() {
+            println!("cargo:rerun-if-changed={}", head.display());
+            break;
+        }
+        if !p.pop() {
+            break;
+        }
+    }
+}
diff --git a/crates/cubecl-cuda/src/compute/context.rs b/crates/cubecl-cuda/src/compute/context.rs
index abcdef12..fedcba98 100644
--- a/crates/cubecl-cuda/src/compute/context.rs
+++ b/crates/cubecl-cuda/src/compute/context.rs
@@ -60,6 +60,33 @@ impl CudaContext {
     pub fn new(
         compilation_options: CompilationOptions,
         properties: DeviceProperties,
         context: *mut CUctx_st,
         arch: CudaArchitecture,
     ) -> Self {
+        // Build a cache-path suffix that captures the dimensions a
+        // disk-persistent PTX cache MUST invalidate on:
+        //
+        // - cubecl crate-graph HEAD SHA: codegen lives in cubecl-cuda /
+        //   cubecl-core / per-compiler crates. Bumping our fork rev
+        //   without bumping cubecl-common's `Cargo.toml` version field
+        //   would otherwise leave stale PTX served to the new pipeline.
+        // - GPU compute capability (sm_<arch>): NVRTC compiles
+        //   architecture-specific PTX; serving sm_70 PTX to an sm_80
+        //   device is a correctness bug.
+        // - CUDA driver version: different driver versions JIT the PTX
+        //   differently, so the cache is per-driver to be safe. (The
+        //   driver also maintains its own PTX→SASS cache at
+        //   ~/.nv/ComputeCache/; this layer is upstream of that.)
+        //
+        // Each axis is appended as a path segment under cubecl-common's
+        // existing <name>/<version>/<path>.json.log layout. The path
+        // segment ordering is least-frequently-changing first so cache
+        // walks short-circuit quickly when only the source hash
+        // differs.
+        let cubecl_sha = option_env!("CUBECL_GIT_SHA").unwrap_or("unknown");
+        let compute_cap = format!("sm_{}", arch);
+        let cuda_runtime = cuda_runtime_version_string();
+        let cache_path = format!("{cubecl_sha}/{compute_cap}/{cuda_runtime}/ptx");
+
         Self {
             context,
             module_names: HashMap::new(),
             ptx_cache: {
                 use cubecl_runtime::config::RuntimeConfig;
                 let config = cubecl_runtime::config::CubeClRuntimeConfig::get();
                 if let Some(cache) = &config.compilation.cache {
                     let root = cache.root();
                     Some(CompilationCache::new(
-                        "ptx",
+                        &cache_path,
                         CacheOption::default().name("cuda").root(root),
                     ))
                 } else {
                     None
                 }
             },
             arch,
             timestamps: TimestampProfiler::default(),
             compilation_options,
             properties,
         }
     }
@@ -310,3 +337,18 @@ impl Drop for CudaContext {
         }
     }
 }
+
+/// Render the CUDA driver version as "MAJOR.MINOR.PATCH" for use as a
+/// cache-key path segment. cudarc 0.16+ exposes `cuDriverGetVersion`
+/// via `driver::result::version::driver()`. On failure or if cudarc
+/// changes the path, falls back to `"unknown"`.
+fn cuda_runtime_version_string() -> String {
+    match unsafe { cudarc::driver::result::version::driver() } {
+        Ok(v) => {
+            let major = v / 1000;
+            let minor = (v % 1000) / 10;
+            let patch = v % 10;
+            format!("{major}.{minor}.{patch}")
+        }
+        Err(_) => "unknown".to_string(),
+    }
+}
```

### Stats

```
 crates/cubecl-cuda/build.rs              | 29 +++++++++++++++++++
 crates/cubecl-cuda/src/compute/context.rs | 45 ++++++++++++++++++++++++++--
 2 files changed, 73 insertions(+), 1 deletion(-)
```

---

## Cache key — full design

Cache file path layout post-patch:

```
<root>/cuda/<cubecl-common version>/<cubecl_sha>/<compute_cap>/<cuda_runtime>/ptx.json.log
```

Components:

| Segment | Source | Example | When it changes |
|---|---|---|---|
| `<root>` | `CacheConfig::root()` | `<workspace>/target/` or `~/.cache/cubecl/` | User config; rare |
| `cuda` | `CacheOption.name` (literal) | `cuda` | Never |
| `<cubecl-common version>` | `CARGO_PKG_VERSION` of cubecl-common | `0.10.0` | When cubecl-common bumps its `Cargo.toml` version |
| `<cubecl_sha>` | `option_env!("CUBECL_GIT_SHA")` from build.rs | `de2f98573902` | Every fork rev advance |
| `<compute_cap>` | `format!("sm_{}", arch)` from `CudaArchitecture` | `sm_120` | When device changes (multi-GPU dev box) |
| `<cuda_runtime>` | `cudarc::driver::result::version::driver()` | `12.8.0` | When driver updates |
| `ptx.json.log` | `path_partial` argument to `Cache::new` | `ptx.json.log` | Never |

The in-file cache **key** (xxh3 hash from `KernelId::stable_hash`)
remains the same; only the **file path** picks the correct sibling for
the (sha, cap, runtime) tuple.

### Why path segments and not adding fields to the StableHash?

Two reasons:

1. **Backward compat with crates.io.** Adding fields to
   `KernelId::stable_hash` changes the hash space; existing on-disk
   caches built with the old hash would all silently miss. Path-segment
   isolation gives the same invalidation guarantee without touching the
   hash function.
2. **Selective deletion.** When a user updates the cubecl fork rev,
   they can `rm -rf <root>/cuda/<version>/<old_sha>/` and free disk
   without losing other entries.

### Disk-space implications

A single PTX entry serialises to ~100 KB (the `Vec<c_char>` PTX
content + `entrypoint_name` + `shared_mem_bytes` + JSON-overhead). The
existing layout co-locates all kernels in one `.json.log`; the new
layout adds three nested directories per (sha × cap × runtime) triple
but keeps one file inside.

On a single dev box (one GPU, one driver) the file count stays the
same; only the path is deeper. Multi-GPU dev boxes get N times the
file count (one per cap). Disk impact: negligible (~few MB per cap).

### Migration

Existing cache files at `<root>/cuda/<version>/ptx.json.log` are
simply **ignored** post-patch — the new layout reads sibling
directories. To preserve already-compiled PTX:

```bash
# One-time migration: move existing cache under the new "unknown" SHA
# bucket. The 8e.2 patch defaults CUBECL_GIT_SHA to "unknown" when
# build.rs can't read git, so this becomes the migration target.
mkdir -p <root>/cuda/<version>/unknown/<sm_arch>/<driver>
mv <root>/cuda/<version>/ptx.json.log \
   <root>/cuda/<version>/unknown/<sm_arch>/<driver>/ptx.json.log
```

…or just let the cache rebuild on first run. The cold-rebuild cost is
~5-30 s per kernel; for a project with ~30 kernels that's a one-time
~10-min cost. Subsequent runs hit the new cache layout.

---

## Verification methodology — when this patch lands

We did **not** execute the patch on a cubecl clone for this Phase 8e
work. Phase 8d's pinned-upload bench is the verified-on-real-hardware
deliverable; this is the structural fix that enables our existing
cache to survive cubecl-version / GPU-arch changes.

The expected verification (when the patch lands on the fork):

### 1. Cold-start sanity (cache hit)

```bash
# Build zenmetrics with the patched cubecl fork
cargo build --release -p zenmetrics-cli --no-default-features --features sweep,gpu,gpu-cuda

# First run: cold cache. Score a single pair.
rm -rf target/cubecl/  # clear any stale cache from before the patch
time target/release/zenmetrics score --metric ssim2 ref.png dist.png
#   Expected: ~10-30 s wall (compile cost dominates first run)

# Second run: warm cache, same kernel.
time target/release/zenmetrics score --metric ssim2 ref.png dist.png
#   Expected: ~500 ms wall (cache hit, no NVRTC)
```

The expected ~18 s → ~500 ms is the headline 8e.2 deliverable per
`PHASE8_PLAN.md`. Numbers are projections from the existing cache
behaviour (when it does hit); we're fixing the misses, not improving
the hit path.

### 2. Multi-cubecl-rev sanity (different sha → different file)

```bash
# Build with one cubecl rev:
cd <cubecl-fork-clone>
git checkout feat/persistent-cache-rev1
cd <zenmetrics>
cargo build --release ...
target/release/zenmetrics score ...

# Check the cache file path:
find target/cubecl -name "ptx.json.log" -mtime -1
#   Should be at <root>/cuda/<ver>/<rev1-sha>/<sm>/<driver>/ptx.json.log

# Bump to second cubecl rev:
cd <cubecl-fork-clone>
git checkout feat/persistent-cache-rev2  # different commit on the same branch
cd <zenmetrics>
cargo build --release ...
target/release/zenmetrics score ...

# Now there should be TWO cache files:
find target/cubecl -name "ptx.json.log" -mtime -1
#   <root>/cuda/<ver>/<rev1-sha>/<sm>/<driver>/ptx.json.log
#   <root>/cuda/<ver>/<rev2-sha>/<sm>/<driver>/ptx.json.log
```

This is the structural invalidation test — different fork revs MUST
read different files.

### 3. Multi-GPU sanity (different arch → different file)

Requires two CUDA GPUs of different compute_caps on the same host (or
two hosts), running the same binary. Each host should have a separate
`<sm_arch>` directory. If they share NAS-mounted target/, no
correctness issue (each picks its own sibling); if they share a
cargo install path, no issue (the binary is identical, but cache
layout still segregates).

This test is the **correctness gate** of the patch — failing it means
sm_70 PTX gets fed to sm_80 devices and the cache silently corrupts.

---

## Status

- Patch designed. **NOT applied**.
- Expected speedup (18 s → 500 ms) extrapolated from the existing
  cache's hit-path behaviour. The patch fixes the structural
  invalidation problem so the cache actually hits across our normal
  dev cycle. We have not run the cache-hit micro on a real CUDA box
  with a recent commit during this Phase 8e wave; the 8d pinned-
  upload patch's 4.32× speedup is the comparable measured-on-hardware
  number that lives in `PINNED_UPLOAD_UPSTREAM_PR.md`.
- Apply on `lilith/cubecl` as `feat/persistent-cache` branch per
  `CUBECL_FORK_STRATEGY.md`. Submit upstream as a follow-on PR after
  pinned-upload (#1334) lands.
- After landing on the fork, bump zenmetrics workspace pin and verify
  per the methodology above.

---

## Related work that is NOT in this patch

- **Driver-level PTX→SASS cache** at `~/.nv/ComputeCache/` is owned by
  the CUDA driver, not by us. It already handles PTX→SASS reuse and is
  per-driver-version + per-device-uuid. We do not duplicate.
- **Source code in-memory cache** (the `HashMap` inside the same
  `CudaContext`) is unchanged. The patch only touches the on-disk
  layout.
- **Cache eviction** for runaway cache sizes. cubecl-common's `Cache`
  has no eviction; entries accumulate forever. For the kernel counts
  we have (~30) at ~100 KB each = ~3 MB, this is fine. If the kernel
  count balloons (e.g. heavy autotune permutations), an LRU layer
  becomes worth adding upstream.
- **Cross-host cache** (NAS-mounted target/ shared between dev boxes).
  Works post-patch because the path segregates by `<sm_arch>` and
  `<driver>`. Don't need a new mechanism.
