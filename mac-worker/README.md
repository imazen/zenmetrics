# zen macOS fleet worker

Turn a spare Mac (Mac mini/Studio, or an idle desktop) into a fleet worker for the zensim-720 feature
backfill. It runs the **same** `zenfleet-worker` the cloud/tower boxes run, pulling jobs from the
**same** R2 pool — it just becomes another worker on the ledger. These are **CPU** feature-extraction
nodes: no GPU, no CUDA.

Two ways to run it. Pick by machine type:

| | **Native + launchd** (recommended) | **colima + the exec image** (zero-build) |
|---|---|---|
| What runs | native `darwin` `zenfleet-worker`/`zenmetrics` binaries under a launchd LaunchDaemon | the `ghcr.io/imazen/zenfleet-worker:exec` container under headless Docker (`colima`) |
| Apple Silicon | **native arm64** (full speed) | amd64 image **emulated** (works, slower) |
| Intel Mac | native x86_64 | amd64 image, native-in-VM |
| Survives logout/reboot | yes (root LaunchDaemon, no GUI) | yes (colima is headless; not Docker Desktop) |
| Build step | build the 2 binaries once (macOS CI or on the Mac) | none — pull the image |

The native path mirrors the Linux systemd `zen-worker` service and the Windows SYSTEM task
(`../windows-worker`). Lead with it for a dedicated node; use colima if you want zero build.

---

## Credentials (both paths)

Mint a **scoped, 7-day** R2 temp cred on the **dev box** (which holds the CF token — never put the
account root key on a node). This is the same minter the Linux nodes use:

```bash
# on the dev box:
bash ubuntu-node/mint_cred.sh          # prints AWS_ACCESS_KEY_ID / _SECRET / _SESSION_TOKEN
```

Paste those three lines into `worker.env` (copy from `worker.env.example`). The session token is
**mandatory** — temp creds 403 without it. Re-mint weekly (7-day cap). `worker.env` is gitignored;
never commit it.

---

## Path A — native binaries + launchd (recommended)

### 1. Build the two binaries (once) — `build_mac.sh`

**Workspace prerequisite (same as the Windows/Linux workstation build).** The `zenmetrics` executor
decodes every codec + runs the CPU metrics, so it depends on ~12 **sibling repos** via `path = "../…"`
(`../zenpng`, `../zenwebp`, `../zenavif`, `../zenjxl`, `../jxl-encoder`, `../zensim`,
`../../butteraugli`, `../fast-ssim2`, …) at the revs the zenmetrics workspace `Cargo.toml` pins. There
is **no clean-checkout CI build** for it (the fleet images bake a *precompiled workstation* binary) —
so build it on a Mac that has the **zen workspace cloned in the expected layout** (`~/work/zen/…`),
exactly like `../windows-worker/build_win.bat` builds against the WSL workspace. The `zenfleet-worker`
binary, by contrast, has only in-repo deps and builds from the zenmetrics checkout alone.

```bash
# on a Mac WITH the zen workspace cloned, from the zenmetrics repo root:
bash mac-worker/build_mac.sh            # builds both binaries + assembles mac-worker/payload/
```

`build_mac.sh` runs the verified feature set (`sweep,png,jpeg,webp,avif,jxl,cpu-metrics`, mirroring
`build_win.bat`), builds for the **host arch** (Apple Silicon → `aarch64-apple-darwin`, Intel →
`x86_64-apple-darwin`), downloads the matching `s5cmd`, and drops a ready `payload/` folder. If the
sibling repos aren't present it fails loud telling you which are missing — it never silently produces a
half-worker.

**Prefer zero build?** Skip to **Path B (colima)** below — it needs no workspace and works today.

### 2. Assemble the payload

One folder holds everything you copy to the Mac:

```
zenmetrics  zenfleet-worker  s5cmd
run.sh  install.sh  uninstall.sh  com.imazen.zenworker.plist  worker.env
```

- `s5cmd`: grab the macOS build (arm64 or amd64 to match) from https://github.com/peak/s5cmd/releases
- `gtimeout` (optional but recommended, for per-pass wall-clock caps): `brew install coreutils`
- `worker.env`: copy `worker.env.example`, fill in the scoped creds + `ZEN_R2_ENDPOINT`.

### 3. Install (registers the LaunchDaemon)

```bash
cd /path/to/payload
sudo bash install.sh
```

It templates the payload path into `com.imazen.zenworker.plist`, installs it to
`/Library/LaunchDaemons/`, and `launchctl bootstrap`s it. The worker starts immediately, runs at every
boot **before/without login**, and `KeepAlive` relaunches it (a drained pool re-checks for new work
after `ThrottleInterval`).

```bash
sudo launchctl print system/com.imazen.zenworker | grep -E 'state|pid'   # is it running
tail -f /path/to/payload/worker.log                                       # what it's doing
sudo bash uninstall.sh                                                     # stop + remove
```

### Idle-only (shared desktop, not a dedicated node)

The LaunchDaemon above runs whenever the Mac is on. To only work when the Mac is **idle** (like the
Windows kids-PC worker), gate `run.sh` on `ioreg`'s HID idle time — add near the top of the pool loop:

```bash
idle=$(ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print int($NF/1000000000); exit}')
[ "${idle:-0}" -lt 600 ] && { sleep 60; continue; }   # someone used it in the last 10 min -> wait
```

`Nice 10` + `LowPriorityIO` (already in the plist) keep it neighborly even without the idle gate.

---

## Path B — colima + the exec image (zero-build) — `dev_colima.sh`

No binary build, no sibling repos; reuses the exact image + `fleet-entrypoint.sh` the tower runs.
`colima` is headless Docker (no Docker Desktop GUI/login dependency). **This is the fastest way to a
running worker on a Mac for development** — one command:

```bash
brew install colima docker
cp worker.env.example worker.env      # fill in scoped creds (ubuntu-node/mint_cred.sh) + ZEN_R2_ENDPOINT
bash dev_colima.sh                     # starts colima if needed, pulls :exec, runs the worker
docker logs -f zen720-mac              # watch it claim + score
```

`dev_colima.sh` uses the **same** env-shape the tower's `zen720-basement` uses (see
`scripts/jobsys/tower_worker_refresh.sh`) and passes `--platform linux/amd64` on Apple Silicon.
To survive reboot without a login, run `colima` itself under a per-user LaunchAgent (`colima start` on
load) or keep the Mac auto-logged-in. For a **native arm64** container instead of emulation, an arm64
`:exec` tag must be built first — `scripts/jobsys/build_executor_image.sh` is currently **amd64-only**
(needs an arm64 `zenmetrics` binary built where the sibling crates live); until then Apple Silicon runs
the amd64 image emulated (works, slower — fine for CPU feature extraction).

---

## Verify it's working (either path)

The worker writes blobs + a ledger shard to R2 as it scores. From the dev box:

```bash
# ledger shards this worker wrote (its name appears in the filename):
aws s3 ls s3://zentrain/jobs/<some-run>/ledger/ --endpoint-url "$R2_ENDPOINT" | grep mac
```

and `worker.log` (native) or `docker logs zen720-mac` (colima) shows `cyc=N run=... done=M` lines.

## Validation note

Written to mirror the verified Linux (`../ubuntu-node`) and Windows (`../windows-worker`) workers and
the tower's working `:exec` container. The macOS-native binary build + launchd load have **not yet been
run on a physical Mac** — validate the first node end-to-end (build → payload → `install.sh` → a
`done=` line in `worker.log` → a `mac-…parquet` ledger shard in R2) and note any macOS-specific fixes
back here.
