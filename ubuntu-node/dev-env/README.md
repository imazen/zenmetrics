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

**The new box has NO NVIDIA GPU.** `lspci` reports only `AMD Granite Ridge [Radeon
Graphics]` — the 9950X3D's integrated adapter. The outgoing WSL box had an RTX 5070, so
local CUDA work does **not** carry over: no `cuda-toolkit`, no `nvidia-container-toolkit`,
and the `nsys` / `nsys-ui` profilers from `/usr/local/bin` have nothing to profile. GPU
metric scoring has to run on a fleet node with a discrete card. This is a real capability
regression for an "orchestrator" that previously also did local GPU work — worth deciding
deliberately rather than discovering later.

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

## Order of operations on a fresh box

1. `sudo apt install -y $(grep -vE '^\s*(#|$)' apt-packages.txt)`
2. Install mise, then `mise install` against `mise.toml`
3. `cargo install --path` the four local crates; `cargo install --git` for `summarize`
4. Reinstall the Python CLIs under the pinned interpreter (they do not survive the copy)
5. Place the `/usr/local/bin` binaries that still apply
