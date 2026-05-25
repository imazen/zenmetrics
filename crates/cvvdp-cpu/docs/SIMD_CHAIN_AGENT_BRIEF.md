# SIMD Chain Agent — cvvdp-cpu Performance Brief

**Paste-and-go brief for the SIMD chain agent.** Read top to
bottom, claim a sibling workspace, then land Chunks 1-5 in order
(or in parallel where the plan permits).

## Mission

Bring cvvdp-cpu's `Cvvdp::score` wall time at 1024² (8t) from
**222 ms** to **≤ 50 ms** by SIMD-vectorizing the dominant hot
loops. Per the 2026-05-25 user directive: "optimize cvvdp-cpu to
the level of zensim". Zensim measures **14.66 ms / 1024² / 8t**
(228-feature standard config, 2026-05-25 bench).

Phase-1 target: 50 ms (3.4× zensim — headroom for cvvdp's
structural complexity).
Stretch goal: 25 ms (1.7× zensim — requires structural changes
into Phase 2).
Hard floor: 222 ms (must NEVER regress on any size).

## Inputs to read FIRST

Read these in order. Skipping ahead is forbidden — every
file has load-bearing context.

1. `/home/lilith/.claude/CLAUDE.md` — global standards.
   Especially **"Performance Optimization"** + **"SIMD Target
   Feature Boundaries"** + **"Diagnosing Slow Rust Builds"**.
   The archmage `#[arcane]` / `#[rite]` / `#[magetypes]` patterns
   here are mandatory.

2. `~/work/zen/zenmetrics/CLAUDE.md` — repo conventions (mid-late
   2026 zenmetrics state).

3. `~/work/claudehints/topics/multi-agent-excellence.md` — sibling-
   workspace mandate, honest-stop discipline, per-chunk file
   ownership.

4. `~/work/claudehints/topics/porting.md` — floating-point
   determinism + SIMD parity testing strategy.

5. `~/work/zen/zenmetrics/crates/cvvdp-cpu/docs/SIMD_OPTIMIZATION_PLAN.md`
   — **THE PLAN**. 5 ranked chunks with file scopes, expected
   speedups, risks, dependencies, reference implementations.

6. `~/work/zen/zenmetrics/crates/cvvdp-cpu/benchmarks/zensim_perf_target_2026-05-25.{tsv,meta}`
   — the perf target derived from zensim.

7. `~/work/zen/zenmetrics/crates/cvvdp-cpu/benchmarks/cvvdp_cpu_perf_baseline_2026-05-25.{tsv,meta}`
   — the current baseline + 222 ms anchor + 113-ms recovery budget.

8. `~/work/zen/zenmetrics/crates/cvvdp-cpu/benchmarks/cvvdp_cpu_flamegraph_2026-05-25.svg`
   — visual attribution. Spend 5 minutes scrolling the flamegraph
   so you know what 32% looks like in the wide bars.

9. `/home/lilith/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/cvvdp_cpu_port_shipped_2026-05-24.md`
   — Agent A's port memo. Note Agent A's prediction that powf
   was the #1 cost; the flamegraph CORRECTS this (powf is #5;
   pyramid Gaussian is #1).

10. `~/work/zen/zensim/zensim/src/simd_ops.rs` — zensim's SIMD
    implementation. The closest in-tree reference for archmage
    `#[arcane]` + `#[rite]` patterns on convolution + reduction.

11. `~/work/archmage/docs/site/content/magetypes/examples/` —
    archmage canonical patterns. `convolution_5tap.rs` and
    `blur.rs` are the closest shape to our pyramid kernels.

12. `~/work/butteraugli/butteraugli/src/precompute.rs` — TLS pool
    pattern for Chunk 4. The honest-stop memo at
    `/home/lilith/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/w44_phase3_b7c_tls_pool_2026-05-23.md`
    explains why the TLS pool was a +3.6% wall REGRESSION for
    butteraugli (which has only ~408 pool ops / encode); cvvdp-cpu
    has a much larger allocation footprint per encode (10 Vecs
    × 7 bands × 3 channels = 210 allocs / encode), so re-validate
    with paired A/B during Chunk 4.

13. Each chunk's reference implementation listed in the SIMD plan.

## Workspace claim

```bash
# Choose a sibling workspace per chunk to allow parallel landing.
# Chunks 1, 2, 3, 4 have no inter-dependencies and can land in
# parallel agents.

cd ~/work/zen
jj workspace add ~/work/zen/zenmetrics--cvvdp-cpu-simd-ch<N>
cd ~/work/zen/zenmetrics--cvvdp-cpu-simd-ch<N>
date -u +%Y-%m-%dT%H:%M:%SZ > /tmp/ts && \
  printf '%s agent-cvvdp-cpu-simd-ch<N> chunk-<N>-<topic>\n' \
    "$(cat /tmp/ts)" > .workongoing
```

**Refresh `.workongoing` every 2 minutes of active work**:

```bash
date -u +%Y-%m-%dT%H:%M:%SZ > /tmp/ts && \
  printf '%s agent-cvvdp-cpu-simd-ch<N> <current-activity>\n' \
    "$(cat /tmp/ts)" > .workongoing
```

## File ownership

**YOURS** (the SIMD chain agent owns all of these):
- `crates/cvvdp-cpu/src/*` — algorithm changes for SIMD kernels.
- `crates/cvvdp-cpu/tests/*` — parity tests + extensions.
- `crates/cvvdp-cpu/Cargo.toml` — feature flags + version bumps
  (parity agent finishes their work BEFORE you start; Cargo.toml
  is yours after that).
- `crates/cvvdp-cpu/CHANGELOG.md` — perf-chunk entries.
- `crates/cvvdp-cpu/benchmarks/chunk<N>_<topic>_<date>.{tsv,meta}`
  — per-chunk paired A/B benches.
- `crates/cvvdp-cpu/benchmarks/cvvdp_cpu_perf_<date>.{tsv,meta}`
  — refreshed perf baselines (DO refresh these after EACH chunk
  to catch ratchet wins/losses on smaller sizes).
- `crates/cvvdp-cpu/docs/SIMD_OPTIMIZATION_PLAN.md` — update with
  ACTUAL measured speedup vs EXPECTED after each chunk.

**DO NOT TOUCH**:
- `crates/cvvdp-gpu/*` — separate agent's territory. If a SIMD
  kernel needs to land there (e.g., shared `gaussian_blur_sigma3_simd`
  for cvvdp-gpu callers), file a separate chunk + spawn a separate
  agent.
- `crates/cvvdp-cpu/docs/SIMD_CHAIN_AGENT_BRIEF.md` — this brief.
  Pristine after you finish.
- The scoping agent's bench artifacts:
  - `benchmarks/zensim_perf_target_2026-05-25.{tsv,meta}`
  - `benchmarks/cvvdp_cpu_perf_baseline_2026-05-25.{tsv,meta}`
  - `benchmarks/cvvdp_cpu_flamegraph_2026-05-25.svg`
  These are the anchor point — never overwrite them. Create new
  dated bench files alongside.

## Acceptance gates (every chunk)

Before pushing each chunk to master:

1. **Build**: `cargo build -p cvvdp-cpu --release` PASS on ALL
   feature combinations:
   ```bash
   cargo build -p cvvdp-cpu --release
   cargo build -p cvvdp-cpu --release --no-default-features --features "alloc"
   cargo build -p cvvdp-cpu --release --no-default-features --features "alloc parallel"
   cargo build -p cvvdp-cpu --release --no-default-features --features "alloc parallel pixels"
   ```

2. **Tests**: `cargo test -p cvvdp-cpu --release` PASS:
   - All 22 existing tests (parity vs host_scalar 1e-4, parity
     corpus 1e-4, diffmap invariants 6, color/pyramid/masking/
     diffmap units 9).
   - Any new SIMD-parity tests you added per the chunk's test
     strategy.

3. **Linting**:
   ```bash
   cargo clippy -p cvvdp-cpu --no-deps -- -D warnings
   cargo fmt --check
   ```
   Both PASS clean.

4. **Perf**: paired A/B `cargo run --release -p cvvdp-cpu --example
   time_size_sweep`:
   ```bash
   # Pre-chunk baseline
   git checkout master
   ./target/release/examples/time_size_sweep --output /tmp/pre.tsv
   # Post-chunk
   jj edit <chunk-commit>
   ./target/release/examples/time_size_sweep --output /tmp/post.tsv
   diff -u /tmp/pre.tsv /tmp/post.tsv
   ```
   - **Required**: ≥ 10% wall reduction at 1024² (8t) WITH ZERO
     regression on any size (within ±2% measurement noise).
   - **Diagnostic**: also profile via `cargo flamegraph` to verify
     the expected hot symbol is no longer dominant.

5. **Multi-arch CI**: post-push, watch CI run for:
   - `x86_64-unknown-linux-gnu` (default linux runner)
   - `i686-unknown-linux-gnu` (cross-i686 job — verify SIMD tier
     falls back to scalar correctly)
   - `aarch64-apple-darwin` (macos-latest)
   - `aarch64-pc-windows-msvc` (windows-11-arm)
   - `x86_64-pc-windows-msvc` (windows-latest)
   - `x86_64-apple-darwin` (macos-15-intel or macos-26-intel)

   If ANY platform fails, STOP and fix forward; do not land the
   next chunk until all 6 are green.

## DO NOT (binding for every chunk)

- **DO NOT use `-C target-cpu=native`** in `Cargo.toml` profile,
  CI, or `RUSTFLAGS`. Runtime SIMD dispatch via archmage is the
  right path.
- **DO NOT use the `wide` crate.** archmage `#[arcane]` /
  `#[rite]` / `#[magetypes]` is the canonical pattern.
- **DO NOT use `unsafe`.** `#![forbid(unsafe_code)]` is set in
  cvvdp-cpu's `lib.rs`. archmage provides safe SIMD via tokens.
- **DO NOT widen the JOD tolerance above 1e-3** without explicit
  user confirmation. The current `tests/parity_against_host_scalar`
  gate is 1e-4; that's where it should stay unless a measured
  SIMD chunk needs a controlled relaxation.
- **DO NOT change the public API.** `Cvvdp::score`,
  `score_with_diffmap`, `warm_reference`, etc. stay byte-stable.
  Field additions to `Cvvdp` struct are OK (e.g.,
  `thread_pool: Option<Arc<rayon::ThreadPool>>` per Chunk 5).
- **DO NOT touch `cvvdp-gpu` source.** If `gaussian_blur_sigma3`
  needs a SIMD variant in cvvdp-gpu (for cvvdp-gpu callers),
  spawn a separate agent for that — your scope is cvvdp-cpu only.
- **DO NOT cite "FMA precision"** for any wall-time delta. The
  wins here are SIMD-vectorization wins (per W44-66 user
  correction).
- **DO NOT default-on a chunk that requires reading a new
  envrionment variable.** All SIMD chunks should be
  unconditionally enabled (or behind a Cargo feature flag that's
  default-on).
- **DO NOT enable any new dependency on `cubecl` or other GPU
  runtime.** cvvdp-cpu is CPU-only. archmage is a compile-time
  SIMD dispatch crate, not a GPU one.
- **DO NOT inline `gaussian_blur_sigma3` from cvvdp-gpu into
  cvvdp-cpu** without going through the SIMD port path. The
  scalar version IS in cvvdp-gpu; copy-pasting it to cvvdp-cpu
  is duplication without speedup. Port to SIMD or skip.

## Honest-stop protocol

If during a chunk you discover:

- **Measured wall-time reduction is < 50% of expected** (e.g.,
  Chunk 1 expected 62-93 ms, measured < 31 ms): STOP, write a
  HONEST-STOP memo at `~/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/
  cvvdp_cpu_simd_ch<N>_honest_stop_<date>.md` documenting the
  measurement, the likely cause (LLVM autoversion already at
  ceiling? boundary handling dominates? memory bandwidth bound?),
  and a proposed next chunk that attacks the real bottleneck.
  Do NOT commit half-finished SIMD code; revert the chunk and
  ship the memo + bench TSVs.

- **JOD parity tolerance is violated** (e.g., 5 cells go above
  1e-3): STOP, report which cells, what the new max delta is,
  what change caused it. Do NOT widen the tolerance without
  user confirmation.

- **A multi-arch CI platform fails** and you can't reproduce
  locally: STOP, document the platform's failure mode, ask the
  user how to proceed (NEON token type might need a tweak, or
  the platform's SIMD width differs and the inner loop needs
  reshaping).

- **A reference implementation is wrong** (e.g., the archmage
  example you wanted to copy doesn't compile against current
  archmage): STOP, file an issue in archmage repo, propose
  a fix in this chunk's HONEST-STOP memo, ask the user.

## Order of operations

Recommended order (each chunk is ~1-3 days of work):

1. **Chunk 4 first** (TLS pool + pre-clear): ZERO parity risk,
   1d work. Establishes the scratch infrastructure other chunks
   will use. **Pros**: derisked, enables structural cleanup
   that helps Chunks 1+2 land cleanly.

2. **Chunk 1** (SIMD gaussian_blur_sigma3): HIGHEST impact, lowest
   complexity (well-known 13-tap separable pattern). 2-3d work.
   Targets the #1 hotspot at 32% self-time. Land this and the
   222ms drops to ~130-160ms in one stroke.

3. **Chunk 2** (SIMD pyramid reduce/expand): Same shape as
   Chunk 1, smaller kernel. 2d work. Targets #2 + #4 hotspots
   at combined 24% self-time. Land this and we hit ~80-110ms.

4. **Chunk 3** (SIMD safe_pow): Reuses vexp/vln helpers, ~1d
   work. Targets #5 + part of #8 at combined 10% self-time.
   Brings us to ~70-95ms.

5. **Chunk 5** (SIMD CSF apply + persistent rayon pool): Depends
   on Chunk 3's vexp. 2d work. Targets #6/9 + rayon plumbing
   at combined 13% self-time. Final push to ≤ 50ms target.

**Parallel landing**: Chunks 1, 2, 3, 4 are independent in source
scope (different files; non-overlapping). 4 agents can land them
in parallel SIBLING WORKSPACES. Chunk 5 stacks on Chunk 3's
vexp helper.

**Total wall**: 8-12d of agent work to clear all 5 chunks.

## After landing each chunk

1. Update `SIMD_OPTIMIZATION_PLAN.md` with the ACTUAL measured
   speedup column (add it next to "Expected ms saved").

2. Update `CHANGELOG.md` with a one-line summary referencing
   the commit SHA.

3. Push to `master` via `jj git push --change @-` per chunk.

4. Write a memo at
   `/home/lilith/.claude/projects/-home-lilith-work-zen-jxl-encoder/memory/
   cvvdp_cpu_simd_ch<N>_shipped_<date>.md` with:
   - One-line summary of what shipped.
   - Pre/post 1024² wall (median + min + max of 5 iters).
   - Top 3 surprises.
   - Acceptance gates: all 6 PASS/FAIL.
   - What's queued next.

5. Refresh `MEMORY.md` index entry (per CLAUDE.md "memory" rules).

6. Clean up sibling workspace per "Cleanup-on-merge MANDATORY"
   (CLAUDE.md):
   ```bash
   cd ~/work/zen/zenmetrics
   jj workspace forget cvvdp-cpu-simd-ch<N>
   rm -rf ~/work/zen/zenmetrics--cvvdp-cpu-simd-ch<N>
   ```

## After landing ALL 5 chunks

Final regression report (separate small chunk after the chain):
- Re-run `time_size_sweep` on all 6 sizes × 3 content classes.
- Update `crates/cvvdp-cpu/benchmarks/cvvdp_cpu_perf_post_simd_<date>.{tsv,meta}`.
- Verify zensim parity (cvvdp-cpu ≤ 50 ms at 1024²; ≤ 3.4× zensim
  per-pixel slope).
- Stretch: if Chunks 1+2+3 land cleanly, file a Phase-2 RFC for
  band fusion / strip pipeline (the only remaining lever to hit
  zensim parity at 14.66 ms).

## Closing notes

- The 5-chunk plan is a *floor* on EV, not a ceiling. If during
  Chunk 1 you discover a bigger lever (e.g., the rayon plumbing
  is actually 25% wall — measurement noise above could shift
  it), pivot. The flamegraph is your guide, not the plan.

- cvvdp-cpu has zero downstream callers using the public API
  right now (JPEG XL buttloop integration is still being scoped).
  This means the SIMD chunks can move fast on internal refactors
  without worrying about API breakage. Use this freedom — internal
  helpers may need new signatures, scratch-aware kernels may
  need new types, etc.

- The `unsafe`-free constraint is non-negotiable per the JXL
  encoder repo's `#![forbid(unsafe_code)]` rule. archmage's
  token-based SIMD dispatch IS the safe path. If you find
  yourself wanting `unsafe`, you're holding the wrong tool —
  re-read CLAUDE.md "SIMD Target Feature Boundaries" section.

- The scoping agent (me) deliberately did NOT modify source code.
  All measurements + planning happened in isolation from
  parity-agent's work. When you start, both my plan AND the
  parity agent's algorithm parity will be on master; you stack
  on top of that.

Good luck. Ship Chunks 1-5; the user will know they landed when
1024² wall drops from 222 ms to ≤ 50 ms.
