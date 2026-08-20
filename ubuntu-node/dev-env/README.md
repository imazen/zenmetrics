# dev-env — reproducible tool fleet for ANY zen node

Brings any Ubuntu node — the orchestrator, a PXE-installed worker, an ARM box — up to the
capabilities of the outgoing x86_64 dev box. Arch-neutral: mise resolves per-platform, and
nothing here pins a CPU target.

Immediate driver: the outgoing dev box (WSL2 on a 7950X) is being replaced by a bare-metal
9950X3D that takes over orchestration. But the same files provision every other node, and
several will run Claude concurrently — see "Concurrent Claude" below.

| file | what it is |
|---|---|
| [`inventory-software.sh`](inventory-software.sh) | the collector. Run it on any box; diff two outputs to see what is missing. Reproducible, not hand-maintained. |
| [`inventory-x86_64.md`](inventory-x86_64.md) | generated output for the outgoing x86_64 box — the measured baseline to bring other nodes up to |
| [`mise.toml`](mise.toml) | user-space toolchains and CLIs. **Validated: 79/79 entries resolve against mise 2026.8.10, zero failures.** |
| [`apt-packages.txt`](apt-packages.txt) | the system layer — libraries and dev headers only. **All 73 verified to resolve on Ubuntu 26.04.** |
| [`mise.wsl.toml`](mise.wsl.toml) | WSL-only overlay — the per-box divergence mechanism |
| [`mise.aarch64.toml`](mise.aarch64.toml) | ARM variant (secondary; carries the Neoverse-N1 RUSTFLAG) |
| [`setup-arm-box.sh`](setup-arm-box.sh), [`install_mise.sh`](install_mise.sh) | ARM bootstrap, moved here out of untracked scripts/ |

Division of labour: **mise owns user-space toolchains and CLIs; apt owns system libraries
and dev headers.** A tool appears in exactly one of the two. CLIs mise provides are
deliberately absent from the apt list even though the old box had them apt-installed.

## Where this sits, and what it replaces

For a PXE-installed node the stack is:

1. `render_install.py` — base OS, docker, ssh, a minimal apt set
2. `apt-packages.txt` — system libraries and dev headers *(here)*
3. `mise.toml` — user-space toolchains and CLIs *(here)*
4. `setup-node-dev.sh` (homefleet) — the older combined apt+cargo installer. It overlaps
   steps 2–3 and predates mise. **Prefer 2+3**; keep `setup-node-dev.sh` only until nodes
   are migrated.

`mise.toml` also **supersedes `~/work/zen/scripts/.mise.toml`**, which was ARM-only, pinned
`RUSTFLAGS = "-C target-cpu=neoverse-n1"`, covered ~25 of this box's 76 cargo crates, and is
tracked in no repository despite calling itself the canonical source of truth. Three configs
for one job is the thing to avoid — this is the one.

## Concurrent Claude across boxes

Several nodes will run Claude Code at once. Two things that matter:

- **Claude Code is installed by `render_install.py`** already (`curl https://claude.ai/install.sh | bash`
  as the `zen` user), so PXE nodes get it without extra work. On a hand-built box, install it
  the same way.
- **The `.workongoing` marker protocol is what stops concurrent sessions colliding.** It is a
  per-repo-root lock file, and it only works if every session honours it: claim before
  reading or building, refresh while working, delete when done. Concurrency across boxes does
  not change the rule — but it makes ignoring it much more expensive, because two agents on
  two machines editing one NFS-mounted or freshly-cloned checkout have no other interlock.

Arch note: the x86_64 resolution is verified (79/79 against mise 2026.8.10). **The aarch64
resolution is not yet verified** — run `mise ls --current` on an ARM box before trusting it
there.

## What the inventory actually found

**Most of the user-space toolchain is already on the new box.** `/home` was moved wholesale
onto the 4 TB drive, which carried `~/.cargo` (all **76** crates), `~/.local/bin` (**177**
entries), `~/.deno`, `~/.bun` and `~/bin` with it. Native binaries work: `uv 0.10.0`,
`ruff 0.14.14`, `s5cmd` and `restic` all run.

**But the Python entry points are broken.** They were installed against Python **3.10** and
the new box ships **3.14.4**, with no 3.10 present. Verified by running them:

```
aws              #!/usr/bin/python3   -> Traceback
poetry           #!/usr/bin/python3   -> Traceback
pre-commit       #!/usr/bin/python3   -> Traceback
huggingface-cli  #!/usr/bin/python3   -> Traceback
```

Copying `~/.local` across a Python-minor-version boundary does not work, and this is
exactly what mise pinning `python = "3.12"` plus reinstalling those tools fixes. Do not
"fix" it by symlinking a python — reinstall the tools under a pinned interpreter.

**Missing on the new box entirely:** `node`, `npm`, `go`, `docker`, `mise`, `pipx`, and all
of `/usr/local/bin` (9 binaries — see below). apt manual selections: 183 vs 54.

**GPU: none at inventory time, an RTX 2080 incoming.** `lspci` on the new box reported only
`AMD Granite Ridge [Radeon Graphics]` — the 9950X3D's integrated adapter. An RTX 2080 is
being moved into it, so CUDA tooling *is* required; `apt-packages.txt` documents the
NVIDIA-repo step, and `nsys` / `nsys-ui` from `/usr/local/bin` become worth placing again.
Install once `lspci | grep -i nvidia` shows the card.

Two consequences worth holding in view:

- **It is a downgrade from the outgoing box, not a like-for-like move.** That box has an
  RTX 5070 (Blackwell); the 2080 is Turing, compute capability 7.5, 8 GB. Anything that
  depended on newer CUDA features or on more than 8 GB of VRAM needs re-checking, not
  assuming.
- **The 2080 currently lives in `lianli`**, which the fleet roster lists as one of its two
  GPU metric-scoring nodes. Moving it makes the orchestrator GPU-capable at the cost of
  that scorer, leaving `node-2` (RTX 3070) plus the orchestrator. Worth a deliberate look
  at fleet GPU topology rather than treating it as a pure gain.

**Two stale runtimes are deliberately NOT reproduced.** The old box carried `node v12.22.9`
(EOL since 2022) and `deno 1.32.1` only because they came from Ubuntu 22.04 apt and a stale
manual install. `mise.toml` specifies `node = "lts"` and `deno = "latest"` instead —
replicating those versions would be replicating a defect.

## Not mise-manageable — handle these separately

**Local-path cargo installs** (built from working copies, so no crates.io version exists):

```
cargo-copter      ~/work/cargo-copter
cargo-read        ~/work/cargo-read
cargo-superwork   ~/work/cargo-superwork
jxl-inspect       ~/work/jxl-inspect
```

Install with `cargo install --path <dir>` after the repos are present.

**Git-sourced:** `summarize` from `https://github.com/rust-lang/measureme#569c4c3b`
(the measureme profiling summariser, pinned to a rev).

**`/usr/local/bin` on the old box** — nine binaries placed by hand, none package-managed:
`benchmark_xl`, `cjpegli`, `djpegli` (libjxl build outputs), `nsys`, `nsys-ui` (NVIDIA
Nsight Systems — the workspace's mandated GPU profiler), `kubectl`, `lazygit`, `cagent`,
`wsl-open` (WSL-only, drop it).

**`~/bin`** — 12 local scripts that came across with `/home`, including `run-heavy`'s
siblings. Note `shrink-wsl-disk.md` is WSL-only.

## Keeping several boxes in sync — the templating answer

**One shared `mise.toml`, plus a per-box `mise.<env>.toml` selected by `MISE_ENV`.**

```
mise install                 # dev / any node: the shared base
MISE_ENV=wsl mise install    # the WSL box: base + mise.wsl.toml on top
```

**Verified against mise 2026.8.10**, not assumed: environment files layer additively and
override by key. On the real config, base resolves 79 entries; `MISE_ENV=wsl` resolves 80,
with `npm:wsl-open` present only under the overlay. A tool named in both takes the overlay's
version; tools only in the base are inherited.

Keep the overlays tiny — every entry in one is a divergence between boxes, and the point is
to have as few as possible. Right now the only genuine WSL divergence is `wsl-open`.

`mise trust <dir>` is required before mise will read a config it has not seen. Expect to run
it once per box per checkout.

## `~/work/zen/scripts/` — resolved

That directory was not a git repo and its contents were tracked nowhere. As of 2026-08-20
the mise-related files were **moved into this directory and committed**, and the copies
under `scripts/` removed:

- `.mise.toml` → [`mise.aarch64.toml`](mise.aarch64.toml)
- `install_mise.sh`, `setup-arm-box.sh` → here, with their path references updated

**Correction:** an earlier draft of this file claimed `run-heavy` was untracked. It is not —
`scripts/run-heavy` is a symlink to `~/work/claudehints/scripts/run-heavy`, which is tracked
(claudehints `c482e5e`). Still untracked and still only on the outgoing box:
`arm`, `cross-platform-test.sh`, `fuzz-sweep.sh`.

## Verified by actually running it (2026-08-20, Ubuntu 26.04, 9950X3D)

- **apt: 73 requested -> 241 installed with dependencies, exit 0, no errors.**
- **mise: 79/79 installed, 0 missing.** Verified running in a login shell, and several are
  large jumps over the outgoing box: node v12.22.9 -> **26.7.0**, deno 1.32.1 -> **2.9.5**,
  rustc 1.97.1 -> **1.98.0**, python 3.10.12 -> **3.14.7**.
- `cargo:tract` is **pinned to 0.22.0, not "latest"** — tract went library-only at 0.23 and
  ships no binary, so `cargo install` fails outright. There is no separate tract-cli crate.
  A case where "latest" is actively wrong.
- **148 of the 177 copied `~/.local/bin` entries were broken Python scripts** (3.10 shebangs,
  libs in `~/.local/lib/python3.10`) against 25 working native binaries. They were not inert:
  `~/.local/bin` precedes `/usr/bin` in a login shell, so the broken pip `cmake` shadowed the
  working apt cmake 4.2.3 and threw a Traceback, which would break any build shelling out to
  it. Quarantined to `~/.local/bin.python310-broken/` (preserved, with a README).
- Two defects the dry-run caught that `apt-cache show` did not:
  - `qemu-user-static` is a *pure virtual* package on Ubuntu 26 with several providers,
    so `apt-get install` refuses it even though `apt-cache show` succeeds. Replaced with
    `qemu-user-binfmt`. **Validate this list with `--dry-run`, never `apt-cache show`.**
  - `xfsprogs` was missing entirely — `/home` is XFS on these boxes, so `xfs_growfs` and
    `xfs_repair` were unavailable. Added.
- The box has **60 GiB RAM**, not the 30 GiB an older probe recorded. That clears the
  earlier concern about the measured >39.9 GiB jxl-at-108MP peak not fitting.

## Python CLIs — use `uv tool install`, never `pip --user`

This is the fix for the 148-broken-scripts problem, not a style preference. `pip --user`
puts every CLI's libraries in one shared `~/.local/lib/python3.X/site-packages`, so a single
interpreter upgrade breaks all of them at once — which is exactly what happened crossing
3.10 -> 3.14. `uv tool install` gives each CLI its own isolated environment.

Installed as the core set: `meson`, `pre-commit`, `poetry`, `awscli`, `vastai`,
`huggingface_hub`.

**NOT reinstalled, deliberately — decide before pulling it in.** The old
`~/.local/lib/python3.10` is **11 GB**, including **torch at 1.7 GB**. The ML/imaging stack
that sits on it — `cvvdp` (the pycvvdp metric, workspace-relevant), `pyiqa`, `rembg`,
`super-image`, `tensorboard`, `transformers-cli`, `accelerate`, `torchrun`, `playwright` —
is a separate and much larger decision than a CLI toolchain.

Two gotchas hit while doing this:

- `uv tool install awscli` fails with "Executables already exist" if stale non-Python files
  from a previous install are still in `~/.local/bin`. On this box those were `aws.cmd` (a
  *Windows* batch file carried over from WSL) and two shell completers — not Python scripts,
  so the quarantine correctly left them. Use `--force` after clearing them.
- `summarize` from the measureme profiler needs `RUSTFLAGS="--cap-lints=warn"`: the pinned
  rev sets `#![deny(warnings)]`, and rustc 1.98 now reports `unsafe { __cpuid() }` as an
  unnecessary unsafe block, turning a warning into a hard error.
  **Do NOT `cargo install summarize` from crates.io** — that name belongs to an unrelated
  audio/video CLI. It must come from the git rev.

## Order of operations on a fresh box

1. `sudo apt install -y $(grep -vE '^\s*(#|$)' apt-packages.txt)`
2. Install mise, then `mise install` against `mise.toml`
3. `cargo install --path` the four local crates; `cargo install --git` for `summarize`
4. Reinstall the Python CLIs under the pinned interpreter (they do not survive the copy)
5. Place the `/usr/local/bin` binaries that still apply
