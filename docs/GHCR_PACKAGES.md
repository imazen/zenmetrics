# GHCR package names — one package per artifact

We publish container images to `ghcr.io/imazen`. The standing problem: every
iteration the worker/sweep image got **renamed** instead of **re-tagged**, so a
handful of artifacts splintered into a dozen near-duplicate ghcr packages. As of
this writing the org has **11 container packages that should be 4**:

| Live ghcr packages (11) | Should be (4) |
|---|---|
| `zenmetrics-sweep`, `zen-metrics-sweep`, `zen-metrics-sweep-hetzner`, `zen-metrics-sweep-salad` | **`zenmetrics-sweep`** (+ `:hetzner`, `:salad`, `:split`, `:vNN` tags) |
| `zenfleet-worker`, `zenfleet-worker-exec`, `zenfleet-worker-exec-gpu`, `zen-jobworker`, `zen-jobworker-exec` | **`zenfleet-worker`** (+ `:exec`, `:exec-gpu` tags) |
| `pycvvdp-scorer` | **`pycvvdp-scorer`** |
| `zen-train` | **`zen-train`** |

**Tags are free; package names are forever.** A package name, once created, sits
in the org package list until someone deletes it, and the user has to look at all
of them. So:

> **The rule: ONE ghcr package per artifact. Variants are TAGS, never new package names.**

A GPU build, a provider variant, a generation bump, a commit pin — those are all
**tags on the canonical package** (`zenmetrics-sweep:exec-gpu`,
`zenmetrics-sweep:hetzner`, `zenmetrics-sweep:v27`, `zenfleet-worker:<sha>`), not
`zenmetrics-sweep-gpu` / `zen-metrics-sweep-hetzner` / a sixth worker name.

## Source of truth

[`ghcr-packages.json`](../ghcr-packages.json) at the repo root is the **allowlist**.
It lists the canonical `packages[]` (with each one's purpose, what builds it, and
its tag convention) and a `deprecated` map from each known splinter to the
canonical `name[:tag]` it should become. Nothing else is authoritative — not a
comment, not a Dockerfile, not this doc.

## The canonical packages

| Package | Role | Tags |
|---|---|---|
| `zenmetrics-sweep` | Codec-sweep + metric-score worker (the omni image) | `latest`, `vNN`, `hetzner`, `salad`, `split`, `<sha>` |
| `zenfleet-worker` | Job-system fleet worker **+ the shared base** (`base-*` tags) every Rust leaf FROMs | `latest`, `exec`, `exec-gpu`, `base-x86`, `base-arm`, `base-x86-cuda`, `<sha>` |
| `pycvvdp-scorer` | pycvvdp (PyTorch) CVVDP reference scorer — kept separate (3 GB torch) | `latest`, `<pycvvdp-version>` |
| `zen-train` | Standalone burn trainer (parquet → ZNPR) | `latest`, `<sha>` |

## Base images: `FROM zenfleet-worker:base-*`

The shared base is **not a separate package** — it's `base-*` tags on
`zenfleet-worker` (one worker-family package holding both the runnable worker tags
and the base tags). Built by
[`docker/base/Dockerfile`](../docker/base/Dockerfile) +
[`.github/workflows/base-image.yml`](../.github/workflows/base-image.yml), it carries
the stable, heavy deps every Rust image needs to **train or run** — Ubuntu 24.04,
aws-cli v2 + s5cmd (R2), `python3` + `pyarrow` + `numpy`, `jq`, a C toolchain, and
(the cuda tag only) the CUDA 12.6 runtime + `cuda_dlsym_stub`. Three tags from one
multi-stage Dockerfile (the only correct way to express "with and without CUDA,
plus ARM" — variants are TAGS, never `*-cuda` / `*-arm` packages):

| Tag | Stage / platform | Use |
|---|---|---|
| `zenfleet-worker:base-x86` | `nocuda`, linux/amd64 | CPU encode/score, training-data prep, anything GPU-less |
| `zenfleet-worker:base-arm` | `nocuda`, linux/arm64 | Hetzner CAX / Oracle ARM tiers |
| `zenfleet-worker:base-x86-cuda` | `cuda`, linux/amd64 | GPU metrics + cubecl-cuda training |

It bakes **no** pytorch (that bloat stays in `pycvvdp-scorer`) and **no** Rust
toolchain — leaf binaries are built in CI and `COPY`d in. A leaf image becomes
tiny — just the binary + entrypoint on top of the shared, already-pulled base:

```dockerfile
FROM ghcr.io/imazen/zenfleet-worker:base-x86-cuda
ARG ZEN_METRICS_BINARY=./zenmetrics      # built in CI with CUDARC_CUDA_VERSION=12000
COPY --chmod=0755 ${ZEN_METRICS_BINARY} /usr/local/bin/zenmetrics
COPY --chmod=0755 scripts/sweep/onstart_unified.sh /usr/local/bin/onstart_unified.sh
ENTRYPOINT ["/usr/local/bin/onstart_unified.sh"]
```

Migrating the live leaf Dockerfiles (`Dockerfile.sweep.v27`,
`crates/zenfleet-worker/Dockerfile`, …) onto `zenfleet-worker:base-*` is a
deliberate follow-up — do it once the `base-*` tags are on ghcr and the fleet is
quiet, one leaf at a time, so a running campaign's relaunches never pull a
half-built base.

## How it's enforced

- **`scripts/ci/check_ghcr_packages.py`** (CI: `.github/workflows/ghcr-guard.yml`,
  local: `just ghcr-check`) scans the repo for `ghcr.io/imazen/<name>` references
  and classifies each name:
  - **canonical** → OK.
  - **deprecated** (a known splinter) → **WARN** in active infra (so the live
    fleet keeps working), tolerated in historical docs.
  - **unknown** (a *new* name in neither list) → **ERROR** in active infra (fails
    CI), warn in docs.

  So the only way to introduce a new `ghcr.io/imazen/<name>` is to **add it to
  `ghcr-packages.json` in the same change** — a one-line, reviewed decision. That
  review is the chokepoint that stops splintering.

- **`scripts/ci/audit_ghcr_org.py`** (`just ghcr-audit`) lists the *live* org
  packages and diffs them against the manifest, printing which canonical packages
  are present, which deprecated splinters still exist, and the commented-out
  recipe to migrate + delete them. It never deletes anything.

"Active infra" = files that actually create or pull images: `.yml`/`.yaml`,
`.sh`, `.rs`, `.py`, `.toml`, `Dockerfile*`, `justfile`. "Docs" = `.md`/`.txt`.

## Adding a genuinely new artifact

If you really need a new package (not a variant of an existing one):

1. Add it to `packages[]` in `ghcr-packages.json` with `purpose`, `builds_from`,
   and a `tag_convention`. Justify in the PR why it isn't a tag on an existing
   package.
2. Reference it from your workflow/Dockerfile/script. `just ghcr-check` now passes.

If it's a *variant* of an existing artifact (GPU build, provider flavor,
generation), **do not add a package** — push a tag on the canonical package.

## Migrating the existing splinters (teed up, do when the fleet is quiet)

This is a coordinated, externally-visible change (an image push + a code edit +
a package delete). Do it per artifact, **not** mid-campaign, and re-tag before
touching code so nothing 404s:

1. **Re-tag, don't rebuild.** Point the canonical name at the existing image
   digest so both names resolve during the transition:
   ```bash
   docker buildx imagetools create -t ghcr.io/imazen/zenmetrics-sweep:salad \
       ghcr.io/imazen/zenmetrics-sweep-salad:v6
   ```
2. **Flip the references** to the canonical `name:tag`. The current ones (from
   `just ghcr-check` warnings) are the Rust `DEFAULT_IMAGE` constants in
   `zenfleet-salad`/`zenfleet-hetzner`/`zenfleet-runpod`, the
   `scripts/jobsys/*` launch/build scripts, and the provider Dockerfiles.
3. **Verify** `just ghcr-check` is clean (and eventually `just ghcr-check
   --strict` — once a splinter has zero infra references, drop it from
   `deprecated` and the strict gate enforces it stays gone).
4. **Delete the orphan** only after nothing references it:
   ```bash
   gh api -X DELETE /orgs/imazen/packages/container/zenmetrics-sweep-salad
   ```
5. When `deprecated` is empty, make CI run `--strict` so any reintroduction fails.

Note the spelling drift the audit surfaced: the **code** references
`zenmetrics-sweep-salad`/`-hetzner` while the **live ghcr package** is the
`zen-metrics-sweep-salad`/`-hetzner` spelling — two names for one image. Both are
in `deprecated` and collapse to `zenmetrics-sweep:salad` / `:hetzner`.

## Quick reference

```bash
just ghcr-check          # fail if any infra ghcr name isn't canonical (CI gate)
just ghcr-check-strict   # also fail on grandfathered splinters (post-migration)
just ghcr-audit          # diff live org packages vs the manifest (+ cleanup recipe)
python3 scripts/ci/check_ghcr_packages.py --list   # show canonical + deprecated
```
