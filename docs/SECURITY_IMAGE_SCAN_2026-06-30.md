# Public image credential audit — ghcr.io/imazen/* (2026-06-30)

Every container package under `ghcr.io/imazen` is **public** (world-pullable), so any
credential baked into one is a public leak. This is the audit of the current images
plus the ongoing-monitoring that was added to catch future leaks.

Raw scan output: [`SECURITY_IMAGE_SCAN_2026-06-30_rawscan.txt`](SECURITY_IMAGE_SCAN_2026-06-30_rawscan.txt).

## Verdict

**No leaked credentials. Nothing to rotate.**

Every scanned image is clean: `0` verified secrets, and every "unverified" hit is a
placeholder/example URI inside a third-party dependency — none imazen-authored. The
fleet "bake-everything" design holds: images carry tools + code, and read every
credential from the runtime environment (the launcher injects scoped temp R2 creds).

## Scope

12 public container packages, scanned at their most-recent tag + `:latest` + `:kadis`
+ every tag pushed in the last week (29 tags for ENV/history; 18 distinct image
families for full filesystem scans, covering every package and base/leaf variant):

`zenmetrics-sweep` (incl. `:kadis`), `zenfleet-worker` (incl. `:exec`, `base-x86`,
`base-x86-cuda`, `base-arm`), `zenfleet-worker-exec`, `zenfleet-worker-exec-gpu`,
`zen-base`, `pycvvdp-scorer`, `zen-train`, and the deprecated splinters
`zen-metrics-sweep`, `zen-metrics-sweep-salad`, `zen-metrics-sweep-hetzner`,
`zen-jobworker`, `zen-jobworker-exec`.

Tools: trufflehog 3.95.7 (verified + unverified) and crane 0.21.7. Method, in order
of leak-likelihood:

1. **Image config: ENV + build history** (crane config) across 29 tags — the #1 leak
   vector (baked `ENV SECRET=...`, `--build-arg` creds in `RUN` history).
2. **Filesystem secret scan** (trufflehog docker mode) of 18 image families — every
   layer + config, for known credential formats, verified against providers.
3. **Targeted high-signal grep** over the imazen-authored layers (export rootfs,
   exclude third-party dep trees) for custom-format secrets trufflehog's named
   detectors can miss: `AKIA…`, PEM private keys, `R2_*`/`AWS_*`/… literal
   assignments, baked `~/.aws/credentials` / `.env`.
4. **Script + binary inspection**: read the onstart/entrypoint/worker scripts and
   `strings`-grepped the rust binaries for credential VALUES.

## Findings (all clean)

| Check | Result |
|---|---|
| Secret-shaped **ENV values** (29 tags) | **None.** Only `RUST_LOG=…vastai…` and `WORKDIR=…salad-sweep` matched the *name* regex; both benign. |
| **Build-history** lines (231 deduped) with secret content | **None.** No `--build-arg` creds, no `echo <base64>`, no `.aws`/`.env` writes. |
| trufflehog **verified secrets** (every image) | **0.** |
| trufflehog **unverified** hits | 0–11/image, all the same handful of **third-party placeholder URIs** (see below). |
| Baked **credential files** (`.aws/credentials`, `.env`, `.config/cloudflare`, `*.pem`/`*.key`) | **None.** Only public CA bundles (`cacert.pem`, `sks-keyservers.netCA.pem`). |
| Imazen **scripts** — hardcoded secret values | **None.** Creds referenced only as env-var NAMES. |
| Imazen **binaries** — embedded credential values | **None.** Only env-var NAMES compiled in (read at runtime). |

### The unverified hits are all third-party placeholders

Every unverified finding is an example/placeholder URI shipped inside a dependency,
using obvious placeholder hosts (`example.com`, `localhost`, `host.com`, `bogus.net`,
`minio.siteb.example.com`) and usernames (`username`, `user`, `joe`, `jschmoe`,
`foobar`, `userid`):

- `pyarrow` `_s3fs` docs/headers/`.so`: `http://username:password@host` / `@localhost:8020`
- `urllib3` / `pip` `url.py`: `https://username:password@host.com:80/path`
- `pandas` / `boltons` test fixtures; Tcl `http` (`jschmoe@bogus.net`); `future` (`joe@proxy.example.com`)
- `mc` (MinIO client) help text: `foobar:foo12345@minio.siteb.example.com`

None are in imazen code, scripts, or config. trufflehog correctly verified none.

### Credential handling is env-only (by design)

- `fleet-entrypoint.sh` requires `${ZEN_R2_ENDPOINT:?}` etc. and documents
  `AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN — SCOPED temp R2
  creds (never root)`, injected by the launcher.
- `onstart_*` / `*_chunk_worker.sh` consume `R2_ACCOUNT_ID / R2_ACCESS_KEY_ID /
  R2_SECRET_ACCESS_KEY` from the environment.
- `onstart_cvvdp_imazen.sh` *writes* `~/.aws/credentials` at **runtime** from
  `${R2_ACCESS_KEY_ID}` / `${R2_SECRET_ACCESS_KEY}` env — that file is **not baked**
  (the credential-file check found none in any image).
- The rust binaries (`zenmetrics`, `zenfleet-sweep`, `zenfleet-vastai`) contain only
  the env-var NAMES they read; `strings`-grep for credential VALUE formats was empty.

## Ongoing monitoring (added)

Two layers, single source of truth in [`scripts/ci/scan_image_secrets.sh`](../scripts/ci/scan_image_secrets.sh):

- **Layer 1 — trufflehog** `docker --image … --results=verified --fail`: every layer
  + config, fails on a VERIFIED live secret.
- **Layer 2 — curated grep**, scoped to OUR config/script/cred files (onstart
  scripts, `.env`, `.aws/credentials`, `*.pem`/`*.key`, configs), excluding
  third-party dep trees / docs / system libs. Catches the custom-format gaps
  trufflehog's named detectors miss (R2 non-AKIA keys, baked private-key FILES,
  baked creds files) with the allowlist applied to matched *content only*.

Validated: flags `R2_SECRET_ACCESS_KEY=<literal>`, `HCLOUD_TOKEN=<literal>`, baked
`aws_secret_access_key`, and PEM private keys (even in an `example`-pathed file);
does NOT flag env-references (`${VAR}`, `$VAR`); zero false positives across all 12
packages (the aws-cli `examples/*.rst` AKIA/PEM samples and `libssh`/`libgnutls`
format markers are correctly excluded).

### What runs where

| Mechanism | File | Trigger | Action on finding |
|---|---|---|---|
| Org-wide scanner | [`.github/workflows/image-secret-scan.yml`](../.github/workflows/image-secret-scan.yml) + [`scripts/ci/scan_all_ghcr_images.py`](../scripts/ci/scan_all_ghcr_images.py) | daily cron · `workflow_run` after both image builds · `workflow_dispatch` | **fails the run** + opens/updates a `security` issue assigned to `lilith` |
| Build gate (bases) | [`.github/workflows/base-image.yml`](../.github/workflows/base-image.yml) via [`.github/actions/scan-image-secrets`](../.github/actions/scan-image-secrets/action.yml) | every `base-*` build | **fails the build** |
| Build gate (worker) | [`.github/workflows/jobworker-image.yml`](../.github/workflows/jobworker-image.yml) (scans by-digest, before merge tags) | every worker build | **fails build → merge never tags** |
| kadis local gate | `kadis-distort:docker/build_sweep_kadis.sh` | local build, pre-push | **exits non-zero → do not push** |

The org-wide scanner enumerates every package, dedups by digest, and scans one tag
per digest. Because `zenmetrics-sweep:kadis` is a *tag* of the canonical package, the
manually-built kadis image is covered automatically (kadis-distort has no CI of its
own; its build script adds a local pre-push trufflehog gate).

### Run it manually

```bash
# one or more image refs (tag or @digest)
scripts/ci/scan_image_secrets.sh ghcr.io/imazen/zenmetrics-sweep:kadis
# whole org (recent tags) or everything
gh auth token | crane auth login ghcr.io -u "$USER" --password-stdin
SCAN_SCOPE=recent python3 scripts/ci/scan_all_ghcr_images.py
```

If a future benign string ever trips Layer 2, extend `ALLOW_RE` in
`scan_image_secrets.sh` (content-only), or set `SCAN_NO_GREP=1` to fall back to
trufflehog-only. A real finding means: **investigate, ROTATE the exposed credential,
then delete/retag the leaking image** — deletion alone is not enough.

## Action required

**None.** No credential is exposed; no rotation needed.
