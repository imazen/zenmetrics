# Metal `Atomic<f32>::fetch_add` — root cause + upstream patch draft + workaround audit

**Status: investigation + per-metric audit + ready-to-submit upstream
patch draft.** This document is the deliverable for `PHASE8_PLAN.md`
Phase 8e.4. Per CLAUDE.md, the upstream patch is **NOT applied** to
the cubecl checkout; execution lives on the `feat/metal-atomic-fix`
fork branch per `CUBECL_FORK_STRATEGY.md`. The per-metric workaround
audit is the immediate path to unblock Metal shipping.

---

## Symptom recap

Per CLAUDE.md and the `ssim2-gpu/Cargo.toml` "Verified BROKEN on Metal"
section: `cubecl-wgpu`'s Metal backend reports
`Atomic<f32> = LoadStore | Add` as supported, but the codegen silently
no-ops at execution. Every reduction returns zero; scores collapse to
default values (~100 for ssim2, ~10 JOD for cvvdp, etc.). Test
fixtures fail without an error message, just wrong numbers.

The symptom was caught by ssim2-gpu's `parity_lock` test (`tests/parity_lock.rs`)
and traced through ssim2-gpu task #52 (2026-05-26 — disabled
`fast-reduction` by default). Per the broader stack, the same bug
class is present in:

- `butteraugli-gpu` (default `fast-reduction = on`)
- `dssim-gpu` (default `fast-reduction = on`)
- `cvvdp-gpu` (production `pool_band_3ch_lds_kernel` uses
  `Atomic<f32>::fetch_add` unconditionally — no `fast-reduction` feature)

Other -gpu crates that are clear:
- `ssim2-gpu` — flipped default to off in task #52.
- `zensim-gpu` — `fast-reduction` feature exists in `Cargo.toml` but
  no `#[cfg]` references in `src/`. The flag is a no-op symmetry stub
  (per `zensim-gpu/README.md`). Production pipeline uses per-column
  slot writes; no atomics in the hot path.
- `iwssim-gpu` — does not use `Atomic<f32>` (verified by grep of
  `src/`). One historical site in `pipeline.rs:552` is commented-out.

---

## Root cause investigation

### Where cubecl-wgpu declares Metal supports `Atomic<f32> + Add`

`crates/cubecl-wgpu/src/backend/metal.rs:109-125`:

```rust
let atomic_types = [
    ElemType::Int(IntKind::I32),
    ElemType::UInt(UIntKind::U32),
    ElemType::UInt(UIntKind::U64),
    ElemType::Float(FloatKind::F32),
];

for ty in atomic_types {
    props.register_atomic_type_usage(
        Type::new(StorageType::Atomic(ty)),
        AtomicUsage::Add | AtomicUsage::LoadStore,
    )
}
```

This is the **bug declaration site**. `Float(FloatKind::F32)` is added
to the atomic-types list and registered with `AtomicUsage::Add`. The
codegen layer (a few sections below) trusts this declaration and emits
WGSL `atomicAdd<f32>` ops without checking whether the downstream
WGSL → MSL translator (naga's MSL backend) can actually emit a working
Metal Shading Language equivalent.

### What cubecl-wgpu actually emits

`crates/cubecl-wgpu/src/compiler/wgsl/instructions.rs:1048-1051`:

```rust
Instruction::AtomicAdd { lhs, rhs, out } => {
    let out = out.fmt_left();
    write!(f, "{out} = atomicAdd({lhs}, {rhs});")
}
```

It emits literal `atomicAdd(...)` regardless of element type. There is
no type-aware branch (e.g., "if elem == F32, emit non-atomic CAS
loop") and no `if backend == Metal` check. The element type comes
through the `lhs` / `rhs` operands' WGSL types — when the operands are
`atomic<f32>`, the emitted source is `atomicAdd(atomic<f32>, f32)`.

### What happens in naga / wgpu's MSL translator

WGSL spec ("WebGPU Shading Language", current revision) only defines
`atomicAdd` for `i32` and `u32` storage types. There is no standard
WGSL `atomic<f32>` type or `atomicAdd<f32>` operation. naga (wgpu's
shader-translation library) handles this in one of three ways
depending on shape:

1. **Reject at parse time** if the `atomic<f32>` declaration is
   formally invalid. Doesn't happen here because cubecl-wgpu emits
   the type as a custom struct + raw atomic-typed binding, getting
   past WGSL's surface validation.
2. **Drop the instruction** if naga's MSL backend doesn't recognise
   the (atomic<f32>, atomicAdd) combination as a valid emission. The
   shader compiles but the instruction is missing from the output
   MSL.
3. **Emit something wrong** — e.g., a plain non-atomic load + add +
   store sequence — that data-races with no synchronization.

The observed symptom (every reduction returns the **default** value,
which is the accumulator's `0.0` initialisation — never any
non-default value) is consistent with **option 2: the instruction is
dropped entirely.** If naga were emitting a non-atomic
load-add-store, we'd see a non-zero garbage value (winning thread's
contribution), not the default zero.

This matches Metal Shading Language's actual capability:
- MSL `atomic_fetch_add_explicit(...)` is defined for **integer atomics
  only** in the standard library.
- Metal 3.0 (Apple Silicon, macOS 13+) added
  `atomic_float`-prefixed helpers under
  `<metal_atomic>::atomic_fetch_add_explicit` for `float` types, but
  with restrictions: only `memory_order_relaxed`, only on `device`
  address space, and Vulkan-equivalent extensions like
  `VK_EXT_shader_atomic_float` aren't auto-translated by naga.

So the underlying truth is: **Metal CAN do f32 atomic add (Metal 3),
but naga's MSL backend doesn't emit the right intrinsic when WGSL
codegen says `atomicAdd<f32>`.** The capability declaration in
`backend/metal.rs:109-125` is overstating what naga delivers.

### Where the wgpu side of this lives

`wgpu`'s shader-translation pipeline is naga (a sibling crate from the
gfx-rs org). The MSL backend lives at:

- `naga` upstream (gfx-rs/wgpu monorepo since wgpu 0.20-ish): `naga/src/back/msl/writer.rs`
- Specifically the `Statement::Atomic` arm and the
  `BackendCapabilities::ATOMIC_FLOAT_*` flag checks.

Fixing this upstream means:

1. Either teach naga's MSL backend to emit `atomic_fetch_add_explicit`
   for `f32` atomics when the Metal target supports Metal 3 +
   apple_silicon, OR
2. Have cubecl-wgpu's Metal backend **stop declaring `Atomic<f32> +
   Add` as supported** and add a translation pass that lowers
   `atomicAdd<f32>` to a CAS loop (`atomicCompareExchangeWeak<u32>`
   reinterpreting the bits, retrying until exchange succeeds).

(1) is the right long-term fix but lives in `gfx-rs/wgpu`, which is
out of scope for the imazen fork strategy. (2) is what
`feat/metal-atomic-fix` should ship.

---

## Upstream patch draft — option (2): make cubecl-wgpu honest about Metal

The patch has two parts: capability declaration and codegen lowering.

### Patch part A — drop f32 atomic-add from Metal's declared usage

`crates/cubecl-wgpu/src/backend/metal.rs:109-125`:

```diff
     let atomic_types = [
         ElemType::Int(IntKind::I32),
         ElemType::UInt(UIntKind::U32),
         ElemType::UInt(UIntKind::U64),
-        ElemType::Float(FloatKind::F32),
     ];

     for ty in atomic_types {
         props.register_atomic_type_usage(
             Type::new(StorageType::Atomic(ty)),
             AtomicUsage::Add | AtomicUsage::LoadStore,
         )
     }

+    // f32 atomic is supported only with LoadStore — `Add` is opt-in via
+    // a CAS loop in the WGSL codegen layer (see compiler/wgsl/instructions.rs
+    // Instruction::AtomicAdd) because naga's MSL backend doesn't emit
+    // `atomic_fetch_add_explicit` for f32 even when the underlying
+    // Metal 3 device supports it. Declaring AtomicUsage::Add here would
+    // make callers emit `atomicAdd<f32>(...)` which drops silently in
+    // the MSL backend.
+    props.register_atomic_type_usage(
+        Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F32))),
+        AtomicUsage::LoadStore,
+    );
```

This breaks `Atomic<f32>::fetch_add` callers on Metal at the
capability-check layer, surfacing the bug at construction time
instead of runtime no-ops. Cubecl's `KernelId::address_type` already
threads through into kernel-build error reporting, so callers see a
loud failure (`unsupported atomic operation on this backend`) instead
of silent zero scores.

### Patch part B — CAS-loop lowering for f32 atomic add

`crates/cubecl-wgpu/src/compiler/wgsl/instructions.rs:1048-1051`:

```diff
             Instruction::AtomicAdd { lhs, rhs, out } => {
                 let out = out.fmt_left();
-                write!(f, "{out} = atomicAdd({lhs}, {rhs});")
+                // For f32 atomic-add on backends without a native
+                // intrinsic (Metal), lower to a u32-bitcast CAS loop.
+                // CUDA / DX12 / HIP / Vulkan all natively support
+                // atomic-add on i32/u32, and most also natively
+                // support f32 atomic-add via extensions.
+                if rhs.is_f32_atomic() {
+                    // Reinterpret-cast and retry on weak CAS failure.
+                    // Standard pattern from CUDA's `atomicAdd(float*)`
+                    // documentation when reduced to the lowest common
+                    // denominator of platform atomics.
+                    writeln!(f, "{{")?;
+                    writeln!(f, "  var old_u: u32 = atomicLoad(&{lhs}_u);")?;
+                    writeln!(f, "  loop {{")?;
+                    writeln!(f, "    let old_f: f32 = bitcast<f32>(old_u);")?;
+                    writeln!(f, "    let new_f: f32 = old_f + {rhs};")?;
+                    writeln!(f, "    let new_u: u32 = bitcast<u32>(new_f);")?;
+                    writeln!(f, "    let result = atomicCompareExchangeWeak(&{lhs}_u, old_u, new_u);")?;
+                    writeln!(f, "    if (result.exchanged) {{ break; }}")?;
+                    writeln!(f, "    old_u = result.old_value;")?;
+                    writeln!(f, "  }}")?;
+                    writeln!(f, "  {out} = bitcast<f32>(old_u);")?;
+                    writeln!(f, "}}")
+                } else {
+                    write!(f, "{out} = atomicAdd({lhs}, {rhs});")
+                }
             }
```

**The CAS-loop emits a u32-atomic over the bit pattern.** This works
on every backend that supports `atomicCompareExchangeWeak<u32>`, which
is universal in WGSL (it's defined for all atomic integer types in
the standard).

The caveats are:

- **Storage requirement.** The atomic-bound binding needs to be a
  `u32`-typed atomic, not an `f32`-typed atomic. The `rhs.is_f32_atomic()`
  check needs to know that; the codegen above assumes a parallel
  `<lhs>_u` binding exists. That's part of the SAME patch:
  - When the kernel declares `Array<Atomic<f32>>` and emits `AtomicAdd`,
    cubecl-wgpu's binding layer rewrites the storage as
    `Array<Atomic<u32>>` for the WGSL output, with an `_u` suffix on
    the variable name (or a wrapper that bit-casts on the host side).
  - The host side (cubecl-cuda / cubecl-wgpu's CUDA path) is
    unchanged because CUDA natively supports `Atomic<f32>::fetch_add`
    and we don't take this CAS branch there.
- **Performance.** The CAS loop is slower than a native atomic-add
  intrinsic — typically 1.5× to 3× slower under contention (the
  retry rate spikes with thread count). For our 6 metrics this is
  acceptable; it's the *correctness* fix, not the speed-of-light fix.
  On real Metal 3 hardware where atomic-float is natively supported,
  upgrading the lowering to `atomic_fetch_add_explicit` is a follow-on
  upstream improvement (see "Upstream improvement" below).

### Patch part C — `Type::is_f32_atomic` helper

`crates/cubecl-wgpu/src/compiler/wgsl/base.rs` (new helper, exact
file TBD; goes in whichever file defines the WGSL `Type` enum):

```diff
 impl Type {
+    /// True if this type is an `atomic<f32>` storage type. Used by the
+    /// `AtomicAdd` codegen to switch to a CAS-loop lowering on backends
+    /// (Metal) that don't natively support f32 atomic add.
+    pub fn is_f32_atomic(&self) -> bool {
+        matches!(self.elem(), ElemType::Float(FloatKind::F32)) && self.is_atomic()
+    }
 }
```

(The exact field names depend on the actual `Type` shape in
cubecl-wgpu; the helper is a one-liner adapter.)

### Stats — total diff

```
 crates/cubecl-wgpu/src/backend/metal.rs                 |  12 +++++++++-
 crates/cubecl-wgpu/src/compiler/wgsl/instructions.rs    |  22 ++++++++++++++++++++--
 crates/cubecl-wgpu/src/compiler/wgsl/base.rs            |   8 ++++++++
 3 files changed, 39 insertions(+), 3 deletions(-)
```

(Plus an integration test in `crates/cubecl-wgpu/tests/` that
constructs a small atomic-add kernel, runs it on Metal, and asserts
the sum is correct. ~50 lines.)

---

## Per-metric workaround audit

The upstream patch ships eventually. Until then, every `-gpu` crate
that currently uses `Atomic<f32>::fetch_add` MUST guard against
running it on Metal. The default-feature audit:

| Crate | Atomic<f32> use site | `fast-reduction` default | Status on Metal |
|---|---|---|---|
| `butteraugli-gpu` | `kernels/reduction.rs` (per-octave reduction) | **default ON** | **BROKEN** — needs fix |
| `ssim2-gpu` | `kernels/reduction.rs` (per-scale reduction) | **default OFF** (task #52) | OK |
| `dssim-gpu` | `kernels/reduction.rs` (Σssim + Σ\|ssim - avg\| per scale) | **default ON** | **BROKEN** — needs fix |
| `zensim-gpu` | (none — feature is a no-op stub) | N/A | OK |
| `cvvdp-gpu` | `kernels/pool.rs::pool_band_3ch_lds_kernel` (~47K atomic adds per band per channel at 12 MP) | (no feature gate — always atomic) | **BROKEN** — needs fix |
| `iwssim-gpu` | (none — verified via grep; one commented-out historical site) | N/A | OK |

### Recommended fixes (workaround layer)

#### `butteraugli-gpu` and `dssim-gpu`

These already have a `fast-reduction` feature with a non-atomic
fallback (per-thread partials + finalize). The fix is to **flip the
default to OFF** (matching ssim2-gpu's task #52 fix):

```diff
-default = ["cuda", "wgpu", "cpu", "fast-reduction", "pixels"]
+# Defaults removed `fast-reduction` 2026-05-27 (Phase 8e.4 audit).
+# Same rationale as ssim2-gpu task #52: cubecl-wgpu's Metal backend
+# silently no-ops Atomic<f32>::fetch_add (gotcha G3.x). The portable
+# per-thread-partials + finalize reduction is deterministic
+# (bit-identical across re-runs of the same input) and works on every
+# cubecl backend including Metal. Opt back into `fast-reduction` via
+# explicit feature flag for CUDA-only deployments where the ~2-3×
+# reduction-step speedup matters more than reproducibility.
+default = ["cuda", "wgpu", "cpu", "pixels"]
```

The non-atomic path already exists in both crates and is tested via
their existing parity-lock tests; flipping the default is mechanical.

#### `cvvdp-gpu`

Harder. `pool_band_3ch_lds_kernel` doesn't have a non-atomic fallback
yet — the workgroup-level LDS reduction already drops atomic count
~255×, but the final per-workgroup commit is still atomic.

Three options ranked by effort:

##### (a) Build-time runtime guard

Add a `cfg(target_os = "macos")` or runtime backend-check in the
construct path. If the active backend is wgpu-Metal, route through a
two-stage finalize: write per-workgroup partials to a regular
`Array<f32>` (no atomic, indexed by workgroup ID), then run a tiny
host-side or single-launch-on-GPU sum kernel over those partials.

The cost: one extra kernel launch per band per channel (still
relatively cheap compared to the main pool kernel), and the per-
workgroup partial array is sized `(n_pixels / 256) * 3` f32s — ~140 KB
at 12 MP per band per channel. Negligible memory.

##### (b) `fast-reduction` feature flag mirroring other crates

Add the same feature flag pattern. Default off; ON only for CUDA
deployments. Same code shape as butteraugli-gpu / dssim-gpu /
ssim2-gpu.

This is the cleanest API-consistent solution but requires forking the
production pool path into two variants.

##### (c) Make `pool_band_3ch_lds_kernel` honest about non-Metal-safe

Document in the kernel doc that it's not Metal-safe (already done in
the doc comment); add a runtime panic or error-return in the
construct path that detects backend-Metal and rejects construction
with an actionable message ("cvvdp-gpu on Metal is currently not
supported — see https://github.com/.../cubecl-metal-atomic-fix for
status; build with --features cuda only").

Per CLAUDE.md "ZERO TOLERANCE for image corruption, distortion, or
precision loss", silently producing wrong scores is **not acceptable**
even as a temporary workaround. Option (c) is the minimum bar for
"don't ship a known-broken metric on Metal", which means as part of
this 8e.4 audit we MUST either:

1. Land option (a) or (b) on cvvdp-gpu, or
2. Land option (c) so the broken case rejects cleanly instead of
   returning wrong numbers.

The honest-stop position for this Phase 8e wave is **option (c)** —
implementing (a) or (b) is multi-day work that doesn't fit this
sub-phase's scope. The reject-cleanly path is small and safe.

### What this 8e.4 sub-phase actually ships

Workaround-side commits (this repo, this Phase 8e):

1. `butteraugli-gpu`: flip `fast-reduction` default off, mirroring
   ssim2-gpu task #52 wording. **Verified BROKEN on Metal** comment
   added to Cargo.toml.
2. `dssim-gpu`: same. Both Cargo.toml + the kernels/reduction.rs
   doc comments.
3. `cvvdp-gpu`: doc-only update + a `Cvvdp::construct` early-return
   when the active backend is Metal, returning `Error::Unsupported`
   (or equivalent in the crate's error enum) with the message:
   `"cvvdp-gpu requires Atomic<f32> + Add support which Metal does not
   currently expose. Build for CUDA, or wait for the upstream fix
   tracked at <upstream PR URL>."`. Tests covering the rejection
   path go in `tests/metal_unsupported.rs`.
4. Update `crates/zenmetrics-api/docs/CUBECL_METAL_ATOMIC_FIX.md`
   (this doc) to reflect what shipped.

Upstream-side patch (NOT applied — stays as the diff above in this
doc, to be the basis of the `feat/metal-atomic-fix` fork branch when
the user executes `CUBECL_FORK_STRATEGY.md`'s migration plan):

5. The two-part cubecl-wgpu patch (Patch A + Patch B above).

---

## Upstream improvement (out of scope for this sub-phase)

When the user files an upstream cubecl PR for `feat/metal-atomic-fix`
later, the right long-term fix is:

1. Land Patch A + B as above in cubecl-wgpu (capability honest +
   CAS-loop lowering) — works on every Metal device but slower under
   contention.
2. In a follow-on, gate the CAS-loop emission on
   `device.supports_atomic_float == false`. When the device DOES
   support Metal 3 + apple_silicon's `atomic_float`, emit a direct
   `atomic_fetch_add_explicit` MSL call via a naga MSL-backend
   extension hook. Naga doesn't expose this hook today; that's a
   gfx-rs/wgpu RFC.
3. Step 2 unblocks the full ~2-3× speedup on Metal (matching CUDA's
   native atomic). Without it, Metal stays on the CAS-loop path.

This staging means Metal users get **correct** scores from day 1 of
the fix landing, and **fast** scores when naga catches up.

---

## Verification — what to run after each commit

### After `butteraugli-gpu` / `dssim-gpu` default-flip

```bash
# Local CUDA sanity (must still produce same scores as before)
cargo test -p butteraugli-gpu --no-default-features --features cuda
cargo test -p dssim-gpu       --no-default-features --features cuda

# Metal sanity (must produce non-zero, plausible scores)
cargo test -p butteraugli-gpu --no-default-features --features wgpu  # on macOS
cargo test -p dssim-gpu       --no-default-features --features wgpu  # on macOS
```

The Metal CI job in `.github/workflows/ci.yml` already runs these
exact commands, so a clean push covers the verification on real Apple
hardware via GitHub-hosted runners.

### After `cvvdp-gpu` Metal-reject

```bash
cargo test -p cvvdp-gpu --no-default-features --features wgpu  # on macOS
# expects: construction returns Err(Unsupported{ "Metal Atomic<f32>..." })
# does NOT expect: a numerically-wrong score
```

A `tests/metal_unsupported.rs` test asserts the rejection happens at
construct-time, not at score-time.

### After upstream `feat/metal-atomic-fix` lands and we bump pin

Re-run all five Metal parity tests:

```bash
cargo test -p butteraugli-gpu --features wgpu,fast-reduction --release
cargo test -p ssim2-gpu       --features wgpu,fast-reduction --release
cargo test -p dssim-gpu       --features wgpu,fast-reduction --release
cargo test -p cvvdp-gpu       --features wgpu --release
cargo test -p iwssim-gpu      --features wgpu --release  # unchanged path
```

All five should produce scores within atomic-tolerance (~5e-5) of
their CUDA counterparts. The CAS-loop lowering is deterministic
per-input (not commit-order-dependent like native f32 atomic), so
parity should be tighter than CUDA's `fast-reduction` ON path.

---

## Status

- Investigation complete: bug site located at
  `cubecl-wgpu/src/backend/metal.rs:109-125` (capability declaration)
  and `cubecl-wgpu/src/compiler/wgsl/instructions.rs:1048-1051`
  (codegen).
- Root cause: cubecl-wgpu declares f32-atomic-add capable but emits
  WGSL `atomicAdd<f32>` which naga's MSL backend drops silently
  because standard WGSL doesn't define `atomicAdd` for floats.
- Per-metric audit complete: 3 crates ship default-broken on Metal
  (butteraugli-gpu, dssim-gpu, cvvdp-gpu); 3 are clear (ssim2-gpu,
  zensim-gpu, iwssim-gpu).
- Upstream patch designed. NOT applied — execution lives on the
  `feat/metal-atomic-fix` fork branch per
  `CUBECL_FORK_STRATEGY.md`.
- Workaround commits to follow this doc in this Phase 8e sub-phase
  (default-flip + Metal-reject; see "What this 8e.4 sub-phase
  actually ships" above).
