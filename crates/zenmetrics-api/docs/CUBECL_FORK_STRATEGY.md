# Cubecl fork strategy — `imazen/cubecl` maintenance plan

**RESOLVED 2026-05-27 via Phase 8f.** The user directed a pivot mid-
Phase-8e: instead of an org-transfer + git-fork dep + upstream-PR
strategy, the actual plan is to publish renamed crates
(`zenforks-cubecl-*`) to crates.io. Phase 8f executed this: 11 renamed
crates ship at `zenforks-cubecl-*@0.10.x` on crates.io from
[imazen/zenforks-cubecl](https://github.com/imazen/zenforks-cubecl).
The active maintenance playbook is now
[`ZENFORKS_CUBECL_STRATEGY.md`](ZENFORKS_CUBECL_STRATEGY.md) — read
that for the current workflow.

This document is retained as **internal methodology reference** — the
branch structure, per-patch rebase notes, and CI matrix below all
apply to whatever fork repo we use (lilith/cubecl, imazen/cubecl, or
the soon-to-exist published `zenforks-cubecl-*` repos), so the
underlying maintenance content stays valuable even though the
distribution mechanism changed.

The 8e.2 / 8e.3 / 8e.4 deliverables (PTX cache patch, Metal atomic
fix) are **distribution-independent** — they describe code changes
that ship in whatever crate-rename or org-rename we end up with.
Re-reading them after the pivot is unnecessary; they apply as-is.

---

**ORIGINAL Status (pre-pivot): design doc, ready for user-driven execution.** This document is the
deliverable for `PHASE8_PLAN.md` Phase 8e.1 as originally framed. The actual GitHub fork moves
(transfer `lilith/cubecl` → `imazen/cubecl`, push branches, configure CI on
the fork) were user-driven follow-on; this repo cannot perform them per
CLAUDE.md "NEVER touch other repos".

The goal is to lock in a maintenance protocol now, so that:

1. Patches we carry on `cubecl` survive the lilith → imazen GitHub identity
   transition without losing ownership.
2. Future rebases onto upstream releases are mechanical, not archeological.
3. Downstream consumers of `zenmetrics-orchestrator` (and the six `*-gpu`
   metric crates) have a single, stable repo URL to pin against.
4. Multiple in-flight patches stack cleanly without interleaving each
   other's rebase pain.

---

## Decision: move from `lilith/cubecl` to `imazen/cubecl`

### Why move

`lilith/cubecl` is tied to an individual GitHub account. Two failure modes:

- If the `lilith` account is renamed, deactivated, or transferred, every
  downstream `Cargo.toml` pinning `git = "https://github.com/lilith/cubecl.git"`
  silently 404s on the next `cargo update` or fresh clone.
- The fork's permissions / collaborators / CI are managed by one person.
  Team handoffs require account-level access transfer, which is slow and
  error-prone on GitHub.

`imazen/cubecl` (org-owned) survives both: contributors can be added /
removed via org admin, the URL stays stable across personnel changes, and
CI billing / settings live at the org level where they belong.

### Why not just keep `lilith/cubecl`

Considered. The migration cost is real:

- Every consumer's `Cargo.toml` rev pin needs updating to the new URL.
- `cargo` lockfiles need refresh (`cargo update -p cubecl` etc.).
- Any out-of-tree readers of the fork (docs, blog posts) become stale.

But the migration cost is **one-time**; the maintenance-amortised cost of
keeping a personal fork forever compounds. We do this once.

### What "move" means in practice

GitHub supports two transfer mechanisms:

- **Repo transfer** (`Settings → Danger Zone → Transfer ownership`): hands
  the entire repo, including issues / PRs / wikis / stars / forks, to
  another account or org. URL changes from `lilith/cubecl` to
  `imazen/cubecl`; GitHub auto-redirects the old URL **indefinitely** unless
  a new repo named `lilith/cubecl` is created. (`git` and `gh` follow the
  redirect; web UIs follow it; `Cargo.toml` pins follow it via libgit2.)
- **Fork-and-mirror**: create a fresh fork at `imazen/cubecl` from
  `tracel-ai/cubecl`, then push the patch branches to it. The old
  `lilith/cubecl` remains as-is until deprecated.

**Recommended: repo transfer.** It preserves issue history (the
`#1334` draft-PR reference, our fork's `feat/pinned-upload` branch with its
full commit history, etc.) and gets free URL redirection. Fork-and-mirror
loses issues unless we manually re-create them.

---

## Branch structure on `imazen/cubecl`

The fork carries multiple in-flight patches. To keep them composable
(stackable, reorderable, separately submittable upstream), each patch lives
on its own branch off a shared base.

### Trunk-like branch: `imazen-main`

`imazen-main` is the **only branch that consumers pin against** for steady-
state use. It tracks upstream `main` (or whichever release tag the
zenmetrics workspace `Cargo.toml` currently expects) with all our active
patches merged on top.

```
upstream main  : A ─── B ─── C ─── D                  ← tracel-ai/cubecl
                              ↘
imazen-main    :               C ─── D ─── P1' ─── P2' ─── P3'
                                          (pinned)(persist) (metal)
```

- `imazen-main` is rebuilt from `upstream main` on each upstream release.
- Each patch (P1 = pinned-upload, P2 = persistent-cache, P3 = metal-atomic-fix)
  rebases onto the new upstream tip; conflicts get resolved in the patch
  branch, then squashed onto `imazen-main`.
- Tags on `imazen-main` follow `vUPSTREAM+imazen.N` — see versioning below.

### Per-patch feature branches

Each in-flight or upstreamable patch keeps its own dedicated branch:

| Branch | Purpose | Upstream PR | Current status (2026-05-27) |
|---|---|---|---|
| `feat/pinned-upload` | Pinned-host-buffer fast path for `create_from_slice` uploads (4× HtoD speedup) | Draft #1334 to `tracel-ai/cubecl` | Drafted; submission user-gated. See `PINNED_UPLOAD_UPSTREAM_PR.md`. |
| `feat/persistent-cache` | Disk-persistent PTX cache to skip cold compile (~18s → ~500ms) | Planned, see `CUBECL_PERSISTENT_PTX_CACHE_PATCH.md` | Patch designed; not yet executed on a clone. |
| `feat/metal-atomic-fix` | Fix cubecl-wgpu Metal backend silently no-opping `Atomic<f32>::fetch_add` | Planned, see `CUBECL_METAL_ATOMIC_FIX.md` | Investigation pending. |

### Why this layout

- **Stackability.** `imazen-main` carries all three patches stacked, so
  downstream consumers get every fix without juggling rev pins.
- **Independent upstream PRs.** Each `feat/*` branch is rebased onto the
  current upstream tip, not `imazen-main`. That keeps the upstream PR
  diff minimal and dependency-free; reviewers see a self-contained change
  per branch.
- **Easy rebase.** When upstream ships a new release, each `feat/*` branch
  is rebased independently. If one conflicts heavily and the others don't,
  we can publish a new `imazen-main` with only the clean ones — partial
  forward-progress on bad rebase days.

### Anti-pattern to avoid

Do **not** layer patches on top of each other (`feat/persistent-cache`
based on `feat/pinned-upload` based on `imazen-main`). Each patch must
be rebasable in isolation onto stock upstream so it can be submitted
upstream individually. Stacking inside `imazen-main` is fine — that's the
trunk's job — but per-patch branches stay independent.

---

## Versioning: `vUPSTREAM+imazen.N`

The fork's tags follow upstream cubecl's version, with an `+imazen.N`
suffix denoting the Nth iteration of our patches on that upstream base.

Examples (illustrative — not all of these exist yet):

| Tag on `imazen/cubecl` | Upstream base | Notes |
|---|---|---|
| `v0.10.0+imazen.1` | cubecl `v0.10.0` (tracel-ai tag `7cf20373`) | Initial pinned-upload-only release. Equivalent to current `lilith/cubecl` rev `de2f9857`. |
| `v0.10.0+imazen.2` | cubecl `v0.10.0` | Adds persistent PTX cache. |
| `v0.10.0+imazen.3` | cubecl `v0.10.0` | Adds Metal `Atomic<f32>` workaround / fix. |
| `v0.11.0+imazen.1` | cubecl `v0.11.0` (post-mega-refactor) | Rebase of all three patches onto upstream `v0.11.0`. |

### Why this scheme

- **Sortable.** SemVer-style `+build-metadata` (PEP 440 / SemVer 2.0).
  `cargo` doesn't parse it semantically (we pin via git rev anyway), but
  humans reading the tag list see the upstream base + our patch count.
- **Unambiguous to upstream.** A bug report referencing `v0.10.0+imazen.2`
  tells upstream the bug is on top of their `v0.10.0`, not on stock.
- **Avoids the "we shipped 0.10.1" trap.** Using `+` not `-` keeps the tag
  out of upstream's release-version namespace.

### Consumer pin convention

zenmetrics `Cargo.toml` pins via git `rev` (commit SHA), not git `tag`. The
tag is informational — a human-readable label that maps SHAs back to the
upstream base and patch count. When the workspace bumps the cubecl pin, the
commit message should reference the tag:

```
chore(workspace): bump cubecl pin to v0.11.0+imazen.1 (rev abcdef12)

Rebased our three patches (pinned-upload / persistent-cache /
metal-atomic-fix) onto upstream v0.11.0. The pinned-upload patch
applied cleanly; persistent-cache needed minor conflict resolution
in cubecl-cuda/src/compute/server.rs around the mega-refactor's
new compute-server shape.
```

---

## Maintenance protocol

### Rebasing onto a new upstream release

When `tracel-ai/cubecl` ships a new release (any minor or patch version):

1. **Fetch upstream.**
   ```
   cd <imazen/cubecl clone>
   git remote add upstream https://github.com/tracel-ai/cubecl.git  # one-time
   git fetch upstream
   ```

2. **Rebase each `feat/*` branch onto the new upstream tag.**
   ```
   git checkout feat/pinned-upload
   git rebase --onto upstream/v0.11.0-tag-or-rev <old-base>
   # resolve conflicts (see per-patch rebase notes below)
   git push --force-with-lease origin feat/pinned-upload
   ```
   `--force-with-lease` (NEVER `--force`) protects against clobbering
   collaborator pushes.

3. **Rebuild `imazen-main`.**
   ```
   git checkout -B imazen-main upstream/v0.11.0-tag
   git merge --ff-only feat/pinned-upload
   git merge --ff-only feat/persistent-cache
   git merge --ff-only feat/metal-atomic-fix
   ```
   If any `merge --ff-only` fails because patches conflict pairwise, fall
   back to `git cherry-pick` and resolve in `imazen-main` directly. Then
   re-base the affected `feat/*` branches off the resolution.

4. **Tag.**
   ```
   git tag v0.11.0+imazen.1
   git push origin v0.11.0+imazen.1 imazen-main
   ```

5. **Run CI on the fork** (see CI section below). All backends should
   build; tests for our three patches should pass.

6. **Bump zenmetrics' workspace pin.**
   ```
   # in zenmetrics' root Cargo.toml, update all 10 cubecl-* entries to
   # the new rev. Commit message references the tag.
   cargo update -p cubecl
   cargo build --workspace --no-default-features --features cuda --release
   # smoke-test on a real GPU before pushing
   ```

### Per-patch rebase notes

#### `feat/pinned-upload`

Mechanical. The patch lives entirely in:

- `crates/cubecl-runtime/src/client.rs` (one file, ~100 lines added)
- `examples/upload_bench/` (new directory — never conflicts)

The two old call sites that were `Bytes::from_bytes_vec(data.to_vec())` and
got changed to `data,` may have moved across mega-refactor-style upstream
changes. Find the equivalent re-wrap sites post-rebase and apply the same
change. See `PINNED_UPLOAD_UPSTREAM_PR.md` "How to re-apply" for full
steps.

#### `feat/persistent-cache`

Lives in `cubecl-cuda/src/compute/server.rs` (or wherever upstream moved
PTX compile after a refactor). The patch wraps `compile_source(...)` in a
disk-cache lookup. If upstream restructures the PTX compile path, the
patch needs to find the new compile site — `git log -p --follow
cubecl-cuda/src/compute/server.rs` from before the refactor usually shows
where the function moved.

See `CUBECL_PERSISTENT_PTX_CACHE_PATCH.md` for the full patch and rebase
landmarks.

#### `feat/metal-atomic-fix`

Lives in `cubecl-wgpu/src/compiler/wgsl/` (or wherever upstream's WGSL
codegen lives — exact path TBD by 8e.4 investigation). The patch either
emits the correct Metal Shading Language `atomic_fetch_add_explicit`
intrinsic, or routes Metal-detected backends through a non-atomic codegen
path.

If 8e.4 lands as the "workaround in our metrics" path (audit + flip the
default of `fast-reduction` off everywhere) instead of an upstream codegen
fix, `feat/metal-atomic-fix` may end up empty — and that's fine. The
branch reservation in this doc stays, in case we revisit the upstream fix
later.

### Submitting a `feat/*` branch upstream

When a patch is ready to upstream (matches upstream review style, has
benches, doesn't break upstream tests):

1. Open the PR from the `feat/*` branch directly to
   `tracel-ai/cubecl:main`.
2. Once merged upstream, mark the corresponding row in this doc's
   "Per-patch feature branches" table as `MERGED` with the upstream PR
   number.
3. On the next upstream release containing the merge, drop the patch from
   `imazen-main`'s rebase recipe. The `feat/*` branch can be kept for
   historical reference (don't delete) but stops being included in the
   `imazen-main` merge sequence.
4. zenmetrics' workspace `Cargo.toml` keeps pinning `imazen-main` until
   all three patches have merged upstream, at which point it returns to
   crates.io versions (per `PINNED_UPLOAD_UPSTREAM_PR.md` sunset plan).

---

## CI expectations on `imazen/cubecl`

The fork's CI replicates upstream's matrix (all backends) plus our
patches' tests.

### Minimal CI matrix

```yaml
# .github/workflows/ci.yml on imazen/cubecl

jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        backend: [cuda, wgpu, cpu, hip]
        # cuda on ubuntu-latest only (NVIDIA runners)
        # wgpu on all three (vulkan / metal / dx12)
        # cpu on all three
        # hip on ubuntu-latest only (AMD ROCm)
        exclude:
          - { os: macos-latest, backend: cuda }
          - { os: macos-latest, backend: hip }
          - { os: windows-latest, backend: cuda }   # actually works, but skip until we own a Windows-NVIDIA runner
          - { os: windows-latest, backend: hip }
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - name: Build
        run: cargo build --release --features ${{ matrix.backend }}
      - name: Test
        run: cargo test --release --features ${{ matrix.backend }}

  upload-bench:
    # Validates the pinned-upload patch on CUDA.
    runs-on: ubuntu-latest-nvidia   # self-hosted or paid runner with NVIDIA
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo run --release --example upload_bench --features cuda
      # Expect: pinned path ~3-4× faster than pageable at 192 MB on PCIe 4.0
```

### Runner notes

- **`ubuntu-latest-nvidia`** is hypothetical here. GitHub-hosted runners
  do not have NVIDIA GPUs. Options: self-hosted runner on a workstation
  with NVIDIA, or paid runners (e.g. `gpu-cn-standard-runner` on actuated
  / namespace.so / `whichever-gpu-runner-service`). The pinned-upload
  bench is the **one** CI step that requires a real CUDA GPU; everything
  else builds without one.
- **`macos-latest`** is Apple Silicon by default (per CLAUDE.md
  "ci-releases.md" note). Exercise the wgpu Metal backend there.
- **HIP** requires ROCm — typically self-hosted on an AMD GPU box.

### What we monitor on fork CI

- All four backends build on every push to `imazen-main` or any `feat/*`
  branch.
- `upload_bench` example runs and reports a sane speedup (failure if
  pinned bandwidth drops below 2× pageable at 192 MB — that means
  pinned-memory routing broke).
- Per-patch tests added in `feat/persistent-cache` and
  `feat/metal-atomic-fix` pass on their respective backends.

### What upstream cubecl CI does (don't duplicate)

- Existing matrix tests across upstream-supported backends.
- Cargo doctests for the public client API.
- We don't need to re-run those; instead, when rebasing onto a new
  upstream release, run `cargo test --workspace --all-features` locally
  before tagging `vUPSTREAM+imazen.N` to confirm upstream's own tests
  still pass on top of our patches.

---

## `MAINTAINERS.md` to land on `imazen/cubecl`

Copy this content into `imazen/cubecl/MAINTAINERS.md` after the org
transfer. Keep it short — it's a pointer to this doc, not a replacement.

```markdown
# Maintainers

This is the imazen fork of [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl).

## What this fork is for

We carry a small number of patches against upstream cubecl that downstream
imazen projects (zenmetrics, the six `*-gpu` metric crates, imageflow's
GPU stack) depend on while waiting for upstream PRs to merge or
proof-of-concept work to be cleaned up for submission.

## Branches

- `imazen-main` — trunk. Tracks upstream + our active patches.
  **Consumers pin to a rev on this branch.**
- `feat/pinned-upload` — pinned-host-buffer fast path. Draft upstream PR #1334.
- `feat/persistent-cache` — disk-persistent PTX cache. Upstream-PR-eligible.
- `feat/metal-atomic-fix` — fix cubecl-wgpu Metal Atomic<f32> no-op codegen.

## Versioning

Tags follow `vUPSTREAM+imazen.N` (e.g. `v0.10.0+imazen.1`). See the
maintenance protocol in zenmetrics' `CUBECL_FORK_STRATEGY.md`.

## Contributing

This fork is **not** a fork-of-convenience for general cubecl development.
General cubecl changes go upstream (`tracel-ai/cubecl`). Only contribute
here when:

1. Your change is being prepared as an upstream PR but isn't ready yet, OR
2. Your change is something upstream is unlikely to accept (e.g. an
   imazen-specific workaround).

## Contacts

- Primary maintainer: see git history / `.github/CODEOWNERS`
- Issue tracker: this repo's GitHub Issues; for general cubecl bugs use
  [tracel-ai/cubecl/issues](https://github.com/tracel-ai/cubecl/issues).

## Full maintenance protocol

[`CUBECL_FORK_STRATEGY.md` in the zenmetrics repo](https://github.com/imazen/zenmetrics/blob/master/crates/zenmetrics-api/docs/CUBECL_FORK_STRATEGY.md)
is the source of truth. Update there, not here.
```

---

## Migration plan (user-driven)

The following steps require GitHub-account-level access. They are user
actions; this repo cannot perform them.

### Phase 1: prep (this repo)

1. Land this doc (`CUBECL_FORK_STRATEGY.md`).
2. Land per-patch docs (`CUBECL_PERSISTENT_PTX_CACHE_PATCH.md`,
   `CUBECL_METAL_ATOMIC_FIX.md`).
3. zenmetrics workspace `Cargo.toml` continues pinning `lilith/cubecl` for
   now — no changes needed.

### Phase 2: transfer (user action)

1. On GitHub, `lilith/cubecl Settings → Danger Zone → Transfer ownership`
   to `imazen` org.
2. GitHub auto-redirects `lilith/cubecl` → `imazen/cubecl` indefinitely
   (caveat: this redirect breaks if someone creates a new `lilith/cubecl`
   repo, so don't).
3. Verify `git clone https://github.com/lilith/cubecl.git` follows the
   redirect (it should — libgit2 honours GitHub's `git-redirect` headers).

### Phase 3: rename branches

1. On `imazen/cubecl`, rename the current default branch
   (probably `master` or `main`) to `imazen-main`.
2. Push `feat/pinned-upload`, `feat/persistent-cache` (if executed),
   `feat/metal-atomic-fix` (if executed) as named branches.
3. Tag the current `imazen-main` tip as `v0.10.0+imazen.1` (matching
   the current `lilith/cubecl` rev `de2f9857`).

### Phase 4: update consumers

1. In zenmetrics workspace `Cargo.toml`, change all 10 cubecl-* entries:
   ```
   - rev = "de2f98573902efe60717cbfc7f8e4f9d630d723e", git = "https://github.com/lilith/cubecl.git"
   + rev = "<same rev on imazen/cubecl>", git = "https://github.com/imazen/cubecl.git"
   ```
   (GitHub's auto-redirect makes the URL change non-mandatory for
   functionality, but pinning the canonical URL keeps tooling/IDEs from
   showing redirect warnings.)
2. Update `crates/zenmetrics-orchestrator/README.md`'s "Dependency on
   `lilith/cubecl` fork" section: rename to "Dependency on `imazen/cubecl`
   fork", swap URLs.
3. Update `CHANGELOG.md` — note the fork URL change as a non-breaking
   workspace-level change.
4. Commit + push.

### Phase 5: CI on the fork

1. Enable GitHub Actions on `imazen/cubecl`.
2. Land the CI workflow described in the "CI expectations" section above.
3. Verify the matrix builds clean on a push to `imazen-main`.

### Phase 6: archive `lilith/cubecl`

The auto-redirect keeps URLs working, so this is optional. If desired,
add a deprecation note to `lilith/cubecl/README.md` pointing at
`imazen/cubecl`. Do **not** delete `lilith/cubecl` — that would break the
redirect.

---

## What this doc explicitly does NOT do

- Does not move any commits on GitHub. The user runs the transfer.
- Does not change `lilith/cubecl` branch state. The fork stays untouched
  until the user transfers it.
- Does not bump the zenmetrics workspace pin. That waits until Phase 4
  above lands.
- Does not write `imazen/cubecl/MAINTAINERS.md`. That gets pasted in by
  the user after Phase 2.

The deliverable is the plan; execution is user-driven follow-on.
