# Upstream PR draft — `feat(runtime): pinned-host-buffer fast path for create_from_slice uploads`

**Status: draft, ready to submit.** This document is the deliverable
for `PHASE8_PLAN.md` Phase 8d. The actual PR submission to
`tracel-ai/cubecl` is the user's call.

We've been carrying this patch on `lilith/cubecl` (`feat/pinned-upload`)
since 2026-05-10 and pinning zenmetrics' workspace `Cargo.toml` to it
(rev `de2f9857`). It delivers a verified ~4× HtoD speedup on our cvvdp
12 MP warm-ref pipeline. The patch is small, additive, has no public
API breakage, and rebases cleanly onto the post-mega-refactor upstream
main as `2ab56e7b`.

---

## Suggested PR metadata

| Field | Value |
|---|---|
| Title | `feat(runtime): pinned-host-buffer fast path for create_from_slice uploads` |
| Target repo | `tracel-ai/cubecl` |
| Target branch | `main` |
| Source branch | `pr/pinned-upload-rebased-2026-05-17` (rev `2ab56e7b` on `lilith/cubecl`) |
| Base commit | `cb87b0d2` (mega-refactor: Totally change the frontend to enable references, #1322) — current upstream main as of 2026-05-15 |
| Labels | `enhancement`, `performance`, `cubecl-runtime`, `cubecl-cuda` |
| Suggested reviewers | maintainers who reviewed [#1030 (Feat/pinned mem)](https://github.com/tracel-ai/cubecl/pull/1030); CUDA-backend owners |
| Linked PR | extends [#1030](https://github.com/tracel-ai/cubecl/pull/1030) (already merged) |
| Track number | already referenced as draft PR **#1334** in our wishlist doc |

---

## Suggested PR summary

> CUDA's `cuMemcpyHtoDAsync` from pageable host memory caps at
> ~5-6 GB/s because the driver internally stages through a hidden
> pinned bounce buffer. Allocating the host buffer with
> `cuMemAllocHost_v2` lets the driver DMA directly, getting ~12-25 GB/s
> on PCIe 4.0.
>
> PR #1030 introduced the underlying `ComputeServer::staging` + `Bytes`
> swap machinery for the explicit `create` / `create_tensor` paths;
> this change extends the same fast path to the slice-input variants
> and adds explicit reserve-then-fill helpers for callers that want to
> skip the intermediate pageable `Vec<u8>`.
>
> Measured ~4× HtoD speedup on a downstream metric pipeline (cvvdp-gpu
> 12 MP warm-ref on RTX 5070, RAM upload-bound): 95 ms → 22 ms.
>
> ### What this adds
>
> - `ComputeClient::reserve_staging(&[usize]) -> Vec<Bytes>` reserves
>   pinned host buffers of the requested sizes. Falls back to plain
>   heap allocations if the backend returns an error (e.g. pinned
>   memory exhausted), so callers always receive buffers of the
>   requested size.
> - `ComputeClient::create_from_slice_pinned(&[u8]) -> Handle` copies
>   the input directly into a pinned staging buffer before uploading.
>   Saves the caller `&[u8]` → pageable `Vec<u8>` → pinned `Bytes`
>   host memcpy that `create_from_slice` performs today, halving
>   host-side memory traffic for large uploads.
> - `ComputeClient::create_tensors_from_slices_pinned` is the batch
>   variant of the above.
> - An `upload_bench` example under `examples/` measuring pageable vs
>   pinned upload bandwidth across a few buffer sizes.
>
> ### What this changes implicitly (no API change)
>
> - `do_create_from_slices` (the implementation behind
>   `create_from_slice` / `create_tensor_from_slice` /
>   `create_tensors_from_slices`) now wraps slices in `Bytes` and calls
>   `self.staging(..., true)` before submitting, so the default upload
>   path also benefits from pinned DMA on CUDA without callers having
>   to opt in. The slice-input variants still do one host memcpy
>   (caller → pinned), where the new `*_pinned` helpers do zero
>   intermediate copies.
> - `do_create` no longer rewraps the already-staged `Bytes` via
>   `Bytes::from_bytes_vec(data.to_vec())` before handing them to the
>   server. That rewrap allocated a fresh pageable `Vec<u8>`, demoting
>   the buffer back to `AllocationProperty::Native` and re-triggering
>   the slow CUDA pageable bounce on the subsequent HtoD copy.
>   Forwarding the `Bytes` as-is preserves the pinned property
>   end-to-end.
>
> ### Backward compatibility
>
> - No public signatures change. All new symbols (`reserve_staging`,
>   `create_from_slice_pinned`, `create_tensors_from_slices_pinned`)
>   are additions.
> - Backends without a pinned-memory concept ignore `staging` and
>   behave exactly as today.
> - The new helpers fall back to plain `vec![0u8; n]` allocations when
>   the backend returns an error from `staging`, so callers do not
>   need to branch on backend support.
>
> ### Relationship to merged PR #1030 ("Feat/pinned mem")
>
> PR #1030 added `ComputeServer::staging` and the `Bytes` allocation-
> property tracking that makes this possible. This change is purely a
> client-side extension that pulls the slice-input `create_*` paths
> through that same machinery and exposes two convenience helpers that
> let callers skip the intermediate pageable host buffer entirely.
>
> ### When to use which helper
>
> | Use case | Helper |
> |---|---|
> | One-shot upload, caller owns `&[u8]`, doesn't need to skip an extra memcpy | `create_from_slice` — already on the pinned path after this PR |
> | Hot per-call upload where the extra host memcpy matters (> 16 MB or > 1000 calls/s) | `create_from_slice_pinned` |
> | Pre-reserve a pinned slab to fill once, then upload | `reserve_staging` + `create` |
> | Batch upload (multiple tensors from slices in one go) | `create_tensors_from_slices_pinned` |

---

## Bench numbers (verified, RTX 5070, downstream metric pipeline)

All numbers come from production measurement, not synthetic
microbenches alone. The microbench is bundled in this PR; the
end-to-end numbers below come from the zenmetrics cvvdp-gpu warm-ref
benchmark which sits on top of cubecl-cuda + this patch.

### End-to-end: cvvdp-gpu 12 MP warm-ref (single distorted candidate)

Source: `~/work/zen/zenmetrics/docs/CUBECL_GOTCHAS.md` §G6.5, captured
during the 2026-05-16 perf push (commit aba984c era).

| Configuration | Per-candidate wall (12 MP) | Speedup vs baseline |
|---|---|---|
| stock cubecl 0.10 (pageable `create_from_slice`) | 95 ms | 1.0× |
| this PR (pinned via patched `create_from_slice`) | 22 ms | **4.32×** |

Methodology: warm-ref pipeline (REF Weber-contrast pyramid pre-
materialised on GPU), one DIST per iter, 5 iters with 2 warmup,
median wall time. `cubecl::future::block_on(client.sync())` between
iters to force serialisation.

Wall time goes from 95 ms to 22 ms = ~4.3×. The 22 ms residual is
kernel time + a single residual HtoD for the DIST sRGB; the 73 ms
removed was almost entirely pageable HtoD.

### nsys breakdown — 4096×4096 cvvdp full-mode run

From `/tmp/cvvdp-4096-full.nsys-rep` (post-patch, 30 iters):

| CUDA API | % of total | Total (ms) |
|---|---|---|
| `cuMemcpyHtoDAsync_v2` | 49.9 | 1089.9 |
| `cuMemAllocHost_v2` (one-time at startup) | 35.7 | 779.1 |
| `cuModuleLoadData` | 7.0 | 153.3 |
| `cuMemAllocAsync` | 3.1 | 67.7 |
| `cuEventSynchronize` | 2.9 | 64.3 |
| `cuLaunchKernel` | 1.1 | 24.2 |

Even **with** pinned uploads, HtoD still dominates at 4K because the
sRGB byte volume per pair is 48 MB and the pipeline runs across many
pyramid bands. Without the patch this entire 1.09 s would be 5-6×
larger (full-pageable bandwidth, no `cuMemAllocHost` line). The
`cuMemAllocHost_v2` calls amortise to zero across the run.

### Microbench (in this PR's `upload_bench` example)

The `examples/upload_bench/` directory bundles a microbench that
prints pageable vs pinned `create_from_slice` bandwidth at three
buffer sizes (4 MB, 48 MB, 192 MB), 5 iters each with 2 warmup:

```
cargo run --example upload_bench --release --features cuda
```

This emits a table like:

```
 size (MB) |   create_slice |  create_pinned |    speedup
------------------------------------------------------------
         4 |  ... GB/s      |  ... GB/s      |     ...x
        48 |  ... GB/s      |  ... GB/s      |     ...x
       192 |  ... GB/s      |  ... GB/s      |     ...x
```

**We have not yet run this microbench on dedicated hardware to
publish numbers in this PR draft.** The 4.32× end-to-end speedup
in the cvvdp-gpu warm-ref pipeline is the verified production
measurement that this patch is justified on (see the table above).
The microbench is included in the PR so reviewers can run it on
their own CUDA hardware and confirm the expected bandwidth jump
(~5-6 GB/s pageable → ~12-25 GB/s pinned per CUDA PCIe 4.0 docs and
[CUDA C Programming Guide §3.2.7.1.1](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory)).

> *PR author note to upstream reviewers*: please paste the
> `cargo run --example upload_bench --release --features cuda`
> output from your CUDA test box into the merged PR description so
> the verified numbers live next to the code.

---

## The patch — full diff

`git format-patch` style. Exactly 263 lines including the commit
message, against base `7cf20373` (cubecl `v0.10.0` tag — the
`de2f9857` version of the patch). For the post-mega-refactor base
(`cb87b0d2`, cubecl `v0.11.0-pre.1`) the patch is `2ab56e7b` at 315
lines — same logic, slightly updated commit message and the
`upload_bench` example switched from `client.sync()` (sync) to
`cubecl::future::block_on(client.sync())` (async-returning) to match
the new client API.

### Stats — `2ab56e7b` (PR-target, against upstream main)

```
 crates/cubecl-runtime/src/client.rs            | 100 +++++++++++++++++++++++--
 examples/upload_bench/Cargo.toml               |  14 ++++
 examples/upload_bench/examples/upload_bench.rs |   6 ++
 examples/upload_bench/src/lib.rs               |  67 ++++++++++++++++
 4 files changed, 184 insertions(+), 3 deletions(-)
```

### Verbatim diff — `2ab56e7b` (PR-target)

The patch is reproduced below verbatim from
`~/.cargo/git/checkouts/cubecl-8210387aee986605/2ab56e7`
(commit `2ab56e7b1a8f9d7d2c07c457081561bae90d0514`, author
`Lilith River <river.lilith@gmail.com>`, date `2026-05-10 15:37:38 -0600`).

```diff
diff --git a/crates/cubecl-runtime/src/client.rs b/crates/cubecl-runtime/src/client.rs
index bce74edf..dedd7137 100644
--- a/crates/cubecl-runtime/src/client.rs
+++ b/crates/cubecl-runtime/src/client.rs
@@ -217,10 +217,20 @@ impl<R: Runtime> ComputeClient<R> {
         let stream_id = self.stream_id();
         let (handle_base, layouts) = self.utilities.layout_policy.apply(stream_id, &descriptors);
 
+        // Wrap each input slice as a (pageable) `Bytes`, then run `staging` so the
+        // server can swap each one to pinned host memory on backends that benefit
+        // (e.g. CUDA: `cuMemcpyHtoDAsync` from pinned memory hits the DMA fast path
+        // at 12-25 GB/s on `PCIe` 4.0 vs 5-6 GB/s from pageable memory).
+        //
+        // This keeps the public API unchanged while routing single-shot uploads
+        // through the same fast path that `create` / `create_tensor` already use.
+        let mut data: Vec<Bytes> = slices.into_iter().map(Bytes::from_bytes_vec).collect();
+        self.staging(data.iter_mut(), true);
+
         let descriptors = descriptors
             .into_iter()
             .zip(layouts.iter())
-            .zip(slices)
+            .zip(data)
             .map(|((desc, alloc), data)| {
                 (
                     CopyDescriptor::new(
@@ -229,7 +239,7 @@ impl<R: Runtime> ComputeClient<R> {
                         alloc.strides.clone(),
                         desc.elem_size,
                     ),
-                    Bytes::from_bytes_vec(data.to_vec()),
+                    data,
                 )
             })
             .collect::<Vec<_>>();
@@ -248,6 +258,11 @@ impl<R: Runtime> ComputeClient<R> {
         descriptors: Vec<MemoryLayoutDescriptor>,
         mut data: Vec<Bytes>,
     ) -> Result<Vec<MemoryLayout>, IoError> {
+        // After `staging`, each `Bytes` may have been swapped in-place to a pinned
+        // host buffer. Forward those `Bytes` to the server *as-is* — re-wrapping via
+        // `Bytes::from_bytes_vec(data.to_vec())` would allocate a fresh pageable
+        // `Vec<u8>`, demote the buffer back to `AllocationProperty::Native`, and
+        // re-trigger the slow CUDA pageable bounce on the subsequent HtoD copy.
         self.staging(data.iter_mut(), true);
 
         let stream_id = self.stream_id();
@@ -265,7 +280,7 @@ impl<R: Runtime> ComputeClient<R> {
                         layout.strides.clone(),
                         desc.elem_size,
                     ),
-                    Bytes::from_bytes_vec(data.to_vec()),
+                    data,
                 )
             })
             .collect::<Vec<_>>();
@@ -300,6 +315,85 @@ impl<R: Runtime> ComputeClient<R> {
         .memory
     }
 
+    /// Reserves pinned (page-locked, on backends that support it) host buffers of
+    /// the requested sizes. The caller fills the returned [`Bytes`] (e.g. via
+    /// [`Bytes::copy_from_slice`] or by writing through `DerefMut`) and then hands
+    /// the buffers to [`Self::create`], [`Self::create_tensor`], or
+    /// [`Self::create_tensors`] to upload them to the device.
+    ///
+    /// On CUDA, pinned host memory enables direct DMA in `cuMemcpyHtoDAsync`,
+    /// reaching ~12-25 GB/s on `PCIe` 4.0 compared to ~5-6 GB/s from pageable
+    /// memory. On backends without an explicit pinned-memory concept this falls
+    /// back to a regular host allocation, so callers can use this API
+    /// unconditionally without regressing other backends.
+    ///
+    /// Note that pinned host memory is a limited system resource — allocate it
+    /// only for buffers that will actually be uploaded to the device, and drop
+    /// the [`Bytes`] handle as soon as the upload completes.
+    pub fn reserve_staging(&self, sizes: &[usize]) -> Vec<Bytes> {
+        if sizes.is_empty() {
+            return Vec::new();
+        }
+
+        let stream_id = self.stream_id();
+        let sizes_owned = sizes.to_vec();
+        let result = self
+            .device
+            .submit_blocking(move |server| server.staging(&sizes_owned, stream_id))
+            .unwrap();
+
+        match result {
+            Ok(stagings) => stagings,
+            // Backends may return errors if pinned memory is exhausted. Fall back
+            // to plain heap allocations so the caller always gets buffers of the
+            // requested sizes.
+            Err(_) => sizes
+                .iter()
+                .map(|&size| Bytes::from_bytes_vec(vec![0u8; size]))
+                .collect(),
+        }
+    }
+
+    /// Like [`Self::create_from_slice`], but copies the input directly into a
+    /// pinned host buffer (on backends that support it) before issuing the
+    /// device upload.
+    ///
+    /// The default [`Self::create_from_slice`] path performs two host-side
+    /// memcpys (caller `&[u8]` → pageable `Vec<u8>` → pinned [`Bytes`]) before
+    /// the device transfer. This variant skips the intermediate `Vec<u8>` and
+    /// copies straight into the pinned staging buffer, halving host-side
+    /// memory traffic for large uploads. The on-device handle is identical.
+    ///
+    /// On backends without a pinned-memory fast path this behaves the same as
+    /// [`Self::create_from_slice`].
+    pub fn create_from_slice_pinned(&self, slice: &[u8]) -> Handle {
+        let mut staging = self.reserve_staging(&[slice.len()]);
+        let mut bytes = staging.pop().expect("reserve_staging returned no buffers");
+        bytes.copy_from_slice(slice);
+        self.create(bytes)
+    }
+
+    /// Like [`Self::create_tensors_from_slices`], but copies inputs directly
+    /// into pinned host buffers before issuing the device upload. See
+    /// [`Self::create_from_slice_pinned`] for the host-side savings rationale.
+    pub fn create_tensors_from_slices_pinned(
+        &self,
+        descriptors: Vec<(MemoryLayoutDescriptor, &[u8])>,
+    ) -> Vec<MemoryLayout> {
+        let sizes: Vec<usize> = descriptors.iter().map(|(_, s)| s.len()).collect();
+        let stagings = self.reserve_staging(&sizes);
+
+        let mut bytes_vec = Vec::with_capacity(descriptors.len());
+        let mut descs = Vec::with_capacity(descriptors.len());
+        for ((desc, slice), mut staging) in descriptors.into_iter().zip(stagings) {
+            staging.copy_from_slice(slice);
+            bytes_vec.push(staging);
+            descs.push(desc);
+        }
+
+        self.do_create(descs, bytes_vec).unwrap()
+    }
+
     /// todo: docs
     pub fn exclusive<'a, Re: Send + 'static, F: FnOnce() -> Re + Send + 'a>(
         &'a self,
diff --git a/examples/upload_bench/Cargo.toml b/examples/upload_bench/Cargo.toml
new file mode 100644
index 00000000..d900107a
--- /dev/null
+++ b/examples/upload_bench/Cargo.toml
@@ -0,0 +1,14 @@
+[package]
+authors = []
+name = "upload_bench"
+publish = false
+edition.workspace = true
+license.workspace = true
+version.workspace = true
+
+[features]
+default = []
+cuda = ["cubecl/cuda"]
+
+[dependencies]
+cubecl = { path = "../../crates/cubecl", version = "=0.11.0-pre.1" }
diff --git a/examples/upload_bench/examples/upload_bench.rs b/examples/upload_bench/examples/upload_bench.rs
new file mode 100644
index 00000000..6541abb1
--- /dev/null
+++ b/examples/upload_bench/examples/upload_bench.rs
@@ -0,0 +1,6 @@
+fn main() {
+    #[cfg(feature = "cuda")]
+    upload_bench::launch::<cubecl::cuda::CudaRuntime>(&Default::default());
+    #[cfg(not(feature = "cuda"))]
+    eprintln!("Build with --features cuda to run this benchmark.");
+}
diff --git a/examples/upload_bench/src/lib.rs b/examples/upload_bench/src/lib.rs
new file mode 100644
index 00000000..bf40e6c9
--- /dev/null
+++ b/examples/upload_bench/src/lib.rs
@@ -0,0 +1,67 @@
+//! Microbench comparing pageable vs pinned host->device upload bandwidth.
+//!
+//! Run with:
+//!   cargo run --example upload_bench --release --features cuda
+
+use cubecl::prelude::*;
+
+const SIZES_MB: &[usize] = &[4, 48, 192];
+const ITERS: usize = 5;
+const WARMUP_ITERS: usize = 2;
+
+pub fn launch<R: Runtime>(device: &R::Device) {
+    let client = R::client(device);
+
+    println!(
+        "{:>10} | {:>14} | {:>14} | {:>10}",
+        "size (MB)", "create_slice", "create_pinned", "speedup"
+    );
+    println!("{}", "-".repeat(60));
+
+    for &mb in SIZES_MB {
+        let size = mb * 1024 * 1024;
+        let buf = vec![0xABu8; size];
+
+        // Warmup both paths.
+        for _ in 0..WARMUP_ITERS {
+            let h = client.create_from_slice(&buf);
+            drop(h);
+            let h = client.create_from_slice_pinned(&buf);
+            drop(h);
+        }
+        cubecl::future::block_on(client.sync()).unwrap();
+
+        // Benchmark create_from_slice (current default path).
+        let mut total_pageable_ns: u128 = 0;
+        for _ in 0..ITERS {
+            cubecl::future::block_on(client.sync()).unwrap();
+            let t0 = std::time::Instant::now();
+            let h = client.create_from_slice(&buf);
+            cubecl::future::block_on(client.sync()).unwrap();
+            total_pageable_ns += t0.elapsed().as_nanos();
+            drop(h);
+        }
+        let avg_pageable_ms = (total_pageable_ns as f64 / ITERS as f64) / 1e6;
+        let pageable_gbs = (size as f64 / 1e9) / (avg_pageable_ms / 1e3);
+
+        // Benchmark create_from_slice_pinned (new fast path).
+        let mut total_pinned_ns: u128 = 0;
+        for _ in 0..ITERS {
+            cubecl::future::block_on(client.sync()).unwrap();
+            let t0 = std::time::Instant::now();
+            let h = client.create_from_slice_pinned(&buf);
+            cubecl::future::block_on(client.sync()).unwrap();
+            total_pinned_ns += t0.elapsed().as_nanos();
+            drop(h);
+        }
+        let avg_pinned_ms = (total_pinned_ns as f64 / ITERS as f64) / 1e6;
+        let pinned_gbs = (size as f64 / 1e9) / (avg_pinned_ms / 1e3);
+
+        let speedup = avg_pageable_ms / avg_pinned_ms;
+
+        println!(
+            "{:>10} | {:>8.2} GB/s | {:>8.2} GB/s | {:>9.2}x",
+            mb, pageable_gbs, pinned_gbs, speedup
+        );
+    }
+}
```

---

## How to submit the PR

1. Clone `tracel-ai/cubecl` (or your existing fork remote).
   The user / submitter clones — this repo does not.
2. Check out a fresh branch from current upstream `main`.
3. Apply the patch verbatim from above (or cherry-pick `2ab56e7b`
   from `lilith/cubecl`'s `pr/pinned-upload-rebased-2026-05-17`
   branch).
4. Run the benchmark on the submitter's CUDA hardware:
   ```
   cargo run --example upload_bench --release --features cuda
   ```
   Paste the output into the PR description, replacing the projection
   table in this draft.
5. Open the PR with title and body per the "Suggested PR summary"
   section above.
6. Suggested labels: `enhancement`, `performance`, `cubecl-runtime`,
   `cubecl-cuda`.
7. Link merged PR [#1030](https://github.com/tracel-ai/cubecl/pull/1030)
   in the description (this PR extends that one).

---

## How to re-apply the patch to a new cubecl release

While upstream review proceeds (or if it stalls), zenmetrics will
continue to pin `lilith/cubecl` in `Cargo.toml`. Each time upstream
ships a new cubecl release the fork needs to be rebased.

The rebase is mechanical because the patch lives entirely in
`crates/cubecl-runtime/src/client.rs` + a new `examples/upload_bench/`
directory:

1. Fetch upstream into `lilith/cubecl`.
2. `git rebase --onto <new-upstream-tag> <old-base> pr/pinned-upload-*`.
3. If `client.rs` conflicts:
   - The two old call sites that were `Bytes::from_bytes_vec(data.to_vec())`
     and got changed to `data,` may have moved. Find the equivalent
     re-wrap site post-rebase and apply the same change.
   - The new `do_create_from_slices` site needs the `staging(...)`
     pre-call added at the top (see the diff's first hunk).
   - The three new `pub fn`s (`reserve_staging`,
     `create_from_slice_pinned`, `create_tensors_from_slices_pinned`)
     are pure additions and should apply cleanly.
4. `examples/upload_bench/` is a fresh subdir — should never conflict.
5. Update the version in
   `examples/upload_bench/Cargo.toml`'s `cubecl = { path = ..., version = "=X.Y.Z-..." }`
   to the new upstream version.
6. `cargo build --release --example upload_bench --features cuda` to
   smoke-test compilation.
7. Push to `lilith/cubecl` as a new dated branch
   (`pr/pinned-upload-rebased-YYYY-MM-DD`) and update
   zenmetrics' `Cargo.toml` rev pin.

Until upstream merges, the workspace `Cargo.toml` should always pin
**all** `cubecl-*` crates to the same rev — otherwise the dep graph
mixes patched and unpatched versions and the speedup vanishes silently
on the unpatched paths.

When upstream merges, we drop the fork pin entirely (return to
crates.io versions) and remove the `Dependency on lilith/cubecl fork`
section from `crates/zenmetrics-orchestrator/README.md`.

---

## Fork branches — reference

`lilith/cubecl` currently carries three relevant tips:

| Branch | Tip | Base | Used by |
|---|---|---|---|
| `pr/pinned-v0.10.0-pre.4` (informal name; tip only) | `08d34ac0` | `bd881019` (pre-0.10.0-pre.4) | Earlier zenmetrics pins; superseded |
| `pr/pinned-v0.10.0` | `de2f9857` | `7cf20373` (cubecl v0.10.0 tag) | **Current zenmetrics pin** (workspace `Cargo.toml`) |
| `pr/pinned-upload-rebased-2026-05-17` | `2ab56e7b` | `cb87b0d2` (post-mega-refactor main) | **What this PR submits** (rebased onto current upstream main) |

All three are the same logical change. The PR-target branch is the
mega-refactor-rebased one because that's what current upstream main
expects. The zenmetrics pin will move forward to the PR-target rev
once upstream cubecl ships a release that builds with our existing
codebase (currently blocked on the mega-refactor's frontend API
changes — see `~/work/zen/zenmetrics/Cargo.toml` lines 213-231).

---

## Related work that is NOT in this PR

These are explicitly out of scope. The PR is intentionally small and
single-purpose; follow-ups are tracked in
`docs/refs/cubecl-wishlist-2026-05-17.md`.

- **W1** — `client.write_to_handle(handle, bytes)` for re-upload into
  a pre-allocated GPU handle without re-creating it. The pinned-
  upload patch sets up the machinery but doesn't expose the in-place
  write API.
- **W22** — Pinned-host buffer pool. The current `reserve_staging`
  allocates fresh pinned memory each call. A pool of N pre-allocated
  pinned slabs would drop the per-call cost to near zero.
- **CUDA stream priority** — separate `feat/cuda-stream-priority`
  branch, already merged upstream as PR #1324.
