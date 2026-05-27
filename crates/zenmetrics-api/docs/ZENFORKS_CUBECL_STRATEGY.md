# `zenforks-cubecl-*` strategy — Imazen's published cubecl fork

**Status: live since 2026-05-27 (Phase 8f).** Supersedes
[`CUBECL_FORK_STRATEGY.md`](CUBECL_FORK_STRATEGY.md), which described
an unmaintained-fork plan that we pivoted away from to a crates.io
publication.

This doc is the operational playbook for maintaining the
`zenforks-cubecl-*` family of crates that the zenmetrics workspace
(and downstream consumers) depend on for GPU compute.

---

## Where things live

| Component | Location |
|---|---|
| Source tree | <https://github.com/imazen/zenforks-cubecl> (main branch) |
| Local checkout | `/home/lilith/work/zenforks-cubecl-work/` |
| Published crates | <https://crates.io/search?q=zenforks-cubecl> |
| Upstream | <https://github.com/tracel-ai/cubecl> (added as `upstream` remote) |
| Patch source docs | This repo's `crates/zenmetrics-api/docs/`: `PINNED_UPLOAD_UPSTREAM_PR.md`, `CUBECL_PERSISTENT_PTX_CACHE_PATCH.md`, `CUBECL_METAL_ATOMIC_FIX.md` |

---

## Why the rename approach

The original Phase 8e plan was to maintain a long-lived fork at
`imazen/cubecl` and pin downstream consumers to git revs on it. That
plan had two problems:

1. **Git pins break downstream builds at every fork-rev bump.** Every
   consumer's `Cargo.toml` needs updating, lockfiles re-resolve, CI
   caches invalidate. The friction compounds over time.
2. **Future consumers outside imazen** (anyone wanting our pinned-upload
   or PTX cache patches) need to either fork-and-pin themselves or take
   a private git-dep dependency, neither of which is ergonomic.

The rename approach solves both:

- Consumers pin against crates.io versions like any other crate. No
  git pin, no rev tracking, no consumer-side cache invalidation on
  publish.
- The `[lib] name = "cubecl_*"` shim means consumer source code reads
  `use cubecl_runtime::*` unchanged. Only Cargo.toml needs the
  `{ package = "zenforks-cubecl-*", version = "..." }` alias.

---

## What got renamed and what didn't

The 11 renamed crates are the ones that need patches plus their
transitive workspace consumers (any crate that depends on a patched
crate). Anything that is a leaf in the dep graph (no dep on a patched
crate) stays as-is on upstream's crates.io publication.

### Renamed → `zenforks-cubecl-*`

| Renamed | Patched in this fork? | Why renamed |
|---|---|---|
| `zenforks-cubecl-runtime` | YES (pinned-upload, 0.10.0+) | Patch target |
| `zenforks-cubecl-cuda` | YES (PTX cache widening, 0.10.1+) | Patch target |
| `zenforks-cubecl-wgpu` | YES (Metal atomic capability honesty, 0.10.1+) | Patch target |
| `zenforks-cubecl-core` | no | deps on cubecl-runtime |
| `zenforks-cubecl-opt` | no | deps on cubecl-core |
| `zenforks-cubecl-cpp` | no | deps on cubecl-runtime + core + opt |
| `zenforks-cubecl-cpu` | no | deps on cubecl-runtime + core + opt + std |
| `zenforks-cubecl-hip` | no | deps on cubecl-runtime + cpp + core |
| `zenforks-cubecl-spirv` | no | deps on cubecl-runtime + core + opt |
| `zenforks-cubecl-std` | no | deps on cubecl-runtime + core |
| `zenforks-cubecl` | no | umbrella — deps on all above |

### Stay on upstream `tracel-ai/cubecl` crates.io

| Crate | Why no rename |
|---|---|
| `cubecl-common` | Leaf (no cubecl-* deps) |
| `cubecl-ir` | Only deps on cubecl-common + cubecl-macros-internal (both leaves) |
| `cubecl-macros` | Only deps on cubecl-common |
| `cubecl-macros-internal` | Leaf |
| `cubecl-zspace` | Leaf |

---

## Version conventions

`zenforks-cubecl-*` versions track upstream cubecl's release. The fork
keeps the SAME numeric version as upstream when it ships purely a
rename; bumps the patch number when adding our own patches.

| zenforks ver | Upstream base | What's in it |
|---|---|---|
| 0.10.0 | cubecl v0.10.0 (`7cf20373`) | vanilla rename + pinned-upload patch (production-proven from `lilith/cubecl@de2f9857`) |
| 0.10.1 | cubecl v0.10.0 | adds PTX cache widening (cubecl-cuda) + Metal Atomic<f32> capability honesty (cubecl-wgpu) |
| 0.11.0 (future) | cubecl v0.11.x | rebase all patches onto the upstream mega-refactor |

When upstream's `0.11.0` ships, we'll release `zenforks-cubecl-* 0.11.0`
(matching upstream's number again, semver-major to track upstream's
breaking changes). Patch-level bumps within a series carry our own
incremental work.

---

## Consumer pin convention

In a downstream `Cargo.toml`:

```toml
[workspace.dependencies]
cubecl         = { package = "zenforks-cubecl",         version = "0.10.1" }
cubecl-runtime = { package = "zenforks-cubecl-runtime", version = "0.10.1" }
cubecl-core    = { package = "zenforks-cubecl-core",    version = "0.10.1" }
cubecl-cuda    = { package = "zenforks-cubecl-cuda",    version = "0.10.1" }
cubecl-wgpu    = { package = "zenforks-cubecl-wgpu",    version = "0.10.1" }
cubecl-cpu     = { package = "zenforks-cubecl-cpu",     version = "0.10.1" }
cubecl-cpp     = { package = "zenforks-cubecl-cpp",     version = "0.10.1" }
cubecl-hip     = { package = "zenforks-cubecl-hip",     version = "0.10.1" }
# Non-renamed crates stay on upstream:
cubecl-common  = "0.10.0"
cubecl-ir      = "0.10.0"
```

Source code reads `use cubecl_runtime::*;` without modification.

zenmetrics' workspace `Cargo.toml` has the above shape in
`[workspace.dependencies]`; member crates pull via
`cubecl = { workspace = true }` as before.

---

## Patches we carry (as of 0.10.1)

### 1. Pinned-upload (cubecl-runtime/src/client.rs)

Routes `create_from_slice` / `create_tensor_from_slice` /
`create_tensors_from_slices` through `ComputeServer::staging` so CUDA
HtoD transfers DMA directly from pinned host memory at 12-25 GB/s on
PCIe 4.0 instead of staging via pageable bounce buffer at 5-6 GB/s.

Production-proven: ~4.3x speedup on the zenmetrics cvvdp-gpu 12 MP
warm-ref pipeline (95 ms → 22 ms). Source patch:
[`PINNED_UPLOAD_UPSTREAM_PR.md`](PINNED_UPLOAD_UPSTREAM_PR.md).

Upstream PR: [tracel-ai/cubecl#1334](https://github.com/tracel-ai/cubecl/pull/1334)
(drafted, submission user-gated).

### 2. PTX cache widening (cubecl-cuda/src/compute/context.rs + build.rs)

Adds three axes to cubecl-cuda's existing persistent PTX cache key:

- `CUBECL_GIT_SHA` (captured at build time): invalidates on any
  zenforks-cubecl source change, not just `cubecl-common`'s
  `CARGO_PKG_VERSION` bump.
- `sm_arch` (CudaArchitecture): NVRTC compiles arch-specific PTX;
  appending the arch makes safety structural instead of relying on
  unique cubecl versions per host.
- `driver_version` (cuDriverGetVersion via cudarc): different driver
  versions JIT the same PTX into different SASS.

Resulting on-disk layout:
```
<root>/cuda/<cubecl-common ver>/<git-sha>/<sm_arch>/<driver_ver>/ptx.json.log
```

Eliminates the "fresh-process cold start = ~18 s of NVRTC re-compile
because the cache key was too narrow" failure mode. Source patch:
[`CUBECL_PERSISTENT_PTX_CACHE_PATCH.md`](CUBECL_PERSISTENT_PTX_CACHE_PATCH.md).

### 3. Metal Atomic<f32> capability honesty (cubecl-wgpu/src/backend/metal.rs)

Drops `AtomicUsage::Add` from the Metal backend's f32 atomic
declaration. The previous declaration caused
`Atomic<f32>::fetch_add` callers to silently no-op because naga's
MSL backend doesn't emit `atomic_fetch_add_explicit` for f32 even
when Metal 3 supports it.

After the patch, callers requesting `Atomic<f32> + Add` on Metal
fail at construct-time with an actionable error message instead of
silently returning zero scores at execute-time.

Source patch (Part A only):
[`CUBECL_METAL_ATOMIC_FIX.md`](CUBECL_METAL_ATOMIC_FIX.md).

**Not in this patch:** Part B (CAS-loop lowering in WGSL codegen
to emit a correct u32-bitcast atomic for f32-atomic-add on Metal).
That requires a wider change to cubecl-wgpu's WGSL Type system and
binding layer; deferred to a follow-on release. The downstream
zenmetrics workarounds (default-off `fast-reduction` on
butteraugli-gpu / dssim-gpu, Metal-reject on cvvdp-gpu) remain the
production correctness fix.

---

## Release workflow — adding a new patch

When we need to ship a new patch (or rebase onto a new upstream
release):

1. **In the zenforks-cubecl working tree** (`/home/lilith/work/zenforks-cubecl-work/`):
   ```bash
   git fetch upstream
   git checkout main
   # Apply the patch (e.g. via `git apply patches/whatever.patch` or
   # manual editing per the per-patch doc in zenmetrics).
   ```

2. **Bump versions.** Run `python3 scripts/bump_to_0_10_X.py` (or
   create a new one). The renamed crates bump; non-renamed stay
   pinned at their upstream version.

3. **Smoke build + test** (per locked policy: smoke-build ALL
   patches together before publishing):
   ```bash
   cargo build --workspace --no-default-features --lib
   cargo test -p zenforks-cubecl-runtime --lib
   # cuda/wgpu lib tests still hit the upstream-v0.10.0 enumset bug —
   # see PHASE8F_STATE.md for the exact failure mode.
   ```

4. **Publish in dep order** with rate-limit awareness (1 new crate
   per 10 min after the first 5-burst). Use
   `scripts/publish_remaining.sh` as a template:
   ```bash
   # Dep order: runtime -> core -> opt -> std -> cpp ->
   #            cuda -> hip -> spirv -> wgpu -> cpu -> cubecl umbrella
   ```

5. **Tag + GitHub release.** Every published version MUST have a
   tag + release per ~/.claude/CLAUDE.md hard rule:
   ```bash
   git tag v0.10.X
   git push origin v0.10.X
   gh release create v0.10.X --title "v0.10.X" --generate-notes
   ```

6. **Bump downstream pins.** In zenmetrics `Cargo.toml`
   `[workspace.dependencies]`, update each `cubecl-* = { package =
   "zenforks-cubecl-*", version = "0.10.X" }` to the new version.
   Commit + push.

---

## Rebasing onto a new upstream cubecl

When `tracel-ai/cubecl` ships a new release (say v0.11.0):

1. Fetch upstream tags: `git fetch upstream --tags`
2. Decide: rebase or recreate?
   - **Rebase** if the patches still apply cleanly:
     `git rebase v0.11.0`
   - **Recreate** if upstream had a major refactor that moved code
     significantly (the v0.11.0 mega-refactor case):
     - `git checkout -b zenforks-v0.11.0-prep v0.11.0`
     - Re-run `python3 scripts/rename_to_zenforks.py`
     - Re-apply each patch by hand against the new code shape
     - Smoke build + test
3. Bump workspace.package.version to 0.11.0; re-update
   `bump_to_0_10_1.py` to `bump_to_0_11_0.py` or similar.
4. Publish dep-order; tag + release.
5. Bump downstream pins.

**Why the recreate-vs-rebase decision matters:** patches that get
deeply edited during rebase (because upstream moved code) need extra
testing — the test surface for our patches is small (cubecl-runtime
client.rs tests, the upload_bench example, our downstream metric
crates). Always run the upload_bench example on a CUDA host before
shipping a new 0.X.0 — that's the verification that the pinned-upload
patch still does what it says.

---

## CI considerations

There is no CI on imazen/zenforks-cubecl yet. We rely on:

1. **Local smoke build + test** before publishing (per locked policy).
2. **Downstream zenmetrics CI** — the parity sweep and the
   `*-gpu` crates' test suites exercise the fork in production.
3. **Manual hardware smoke** — a CUDA box for pinned-upload + PTX
   cache verification, a macOS host for Metal atomic verification.

Eventually we should add a tiny `ci.yml` that builds the workspace
across `{cuda, wgpu, cpu, hip}` × `{ubuntu, macos}` to catch
regressions. That's lower priority than getting downstream
consumption working first.

---

## When a patch lands upstream

When upstream merges one of our patches:

1. Note the upstream PR # and merged commit in
   `CUBECL_FORK_STRATEGY.md`'s "Per-patch feature branches" table
   (mark MERGED).
2. On the next upstream release that contains the merge, our next
   `zenforks-cubecl-*` rebase **drops the carry of that patch** —
   it's now in upstream's source.
3. The renamed crates stay renamed (no value in un-renaming once
   downstream pins are stable).
4. Eventually, if ALL patches land upstream and our fork carries
   nothing custom, the renamed crates can become a pure rename of
   upstream and we publish thinner releases that just track upstream
   version bumps. Or we deprecate the rename and consumers migrate
   back to upstream. That's a future decision.

---

## Things that explicitly do NOT work yet

- **Metal `Atomic<f32>::fetch_add` correctness.** Part B (CAS-loop
  lowering) is not implemented. Callers that want
  `Atomic<f32>::fetch_add` on Metal get a hard error at construct
  time. Downstream workarounds in zenmetrics' `*-gpu` crates are the
  production path on Metal.
- **CI on imazen/zenforks-cubecl.** No workflows configured. The 0.10.x
  releases pass downstream zenmetrics CI but not a per-fork matrix.
- **Auto-rebase on upstream release.** Manual rebase per release.
- **Auto-publish on tag push.** Manual `cargo publish` invocation per
  release (with rate-limit awareness).

---

## Historical context

The pre-Phase-8f strategy lived in
[`CUBECL_FORK_STRATEGY.md`](CUBECL_FORK_STRATEGY.md), which described
a long-lived `imazen/cubecl` git fork with feature branches and
consumer `git rev =` pinning. That doc is retained as historical
methodology; the actual maintenance approach is what's in this doc.

The pivot happened mid-Phase-8e (2026-05-27) when the user reviewed
the fork strategy and noted that the friction of git-pin maintenance
outweighed the benefits of fork autonomy. Published crates with a
rename solve all the consumer-friction problems at the cost of one
extra `package = "..."` alias in `Cargo.toml` — a strictly better
trade.

The execution lives in:
- `PHASE8F_STATE.md` (in the zenforks-cubecl repo) — chronological
  log of stages 1–9
- The `imazen/zenforks-cubecl` repo's commit history
- crates.io as the durable artifact
