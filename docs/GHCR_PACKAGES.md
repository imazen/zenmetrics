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
