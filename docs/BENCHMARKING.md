# Reproduce IQA CPU benchmarks

Replace the example `i265` host label with your public CPU-derived machine label.
The portable CPU command uses Rust stable, just and the checked-out workspace's
pinned sibling dependencies. It enables CPU metrics and threading, with no ORT,
CUDA, neural models or Python. Use a C/C++ toolchain and CMake for native build
prerequisites. Git is needed to clone this workspace and its pinned siblings.

```sh
bash scripts/ci/clone-siblings.sh
just bench-iqa smoke ssim2,gmsd --host i265 --fixture synthetic-v1 --skip-native
```

`synthetic-v1` is a deterministic integer RGB fixture, not a photograph quality
or performance claim. `smoke` is 64×64; production size names are `cid22`,
`aic-small`, `aic-medium`, `aic-large`, `2k`, `4k`, `8k`. Use
`--corpus /path/to/AIC2026/img` instead of `--fixture` for the manifest-verified
S67 pair. Encoded and derived RGB SHA-256s must match the checked-in manifest.
One pair is measured at 1/all available threads in score and score+map modes;
unsupported maps remain explicit. Sample counts are 5 below 2K, 3 at AIC-large,
2K and 4K, and 1 at 8K. Every Rust timing comes from zenbench.

The portable path currently requires explicit `--skip-native`. It refuses to
silently omit native competitors. Full pinned native comparator orchestration
without Python/Meson is **not implemented**; use the explicitly advanced route
below for that existing panel. This is a remaining portability gap.

## Linux

Tested on i265 (Linux) 2026-10-17, commit `f95192aaa39986cc5438fda68716a1d95cee706c`: two CPU metrics at 64×64, 1/all threads, both modes; 8 rows, then `added=0 skipped=8`. Existing tools were used; the package-install commands below were not rerun on a clean machine.

```sh
sudo apt-get update
sudo apt-get install -y build-essential git curl cmake pkg-config
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"
cargo install just --locked
bash scripts/ci/clone-siblings.sh
just bench-iqa smoke ssim2,gmsd --host i265 --fixture synthetic-v1 --skip-native
```

The `ubuntu-latest` job in [run 39029766690](https://github.com/imazen/zenmetrics/actions/runs/39029766690), source `f95192aaa39986cc5438fda68716a1d95cee706c`, also passed on 2026-10-17. Its downloaded artifact contains one content key and 8 timed n>=5 rows; the workflow's repeat check passed. This validates the runner setup, not a fresh apt/rustup manual install.

## macOS ARM64

Tested on mac (Apple M4 Pro, ARM64) 2026-10-17, commit `f95192aaa39986cc5438fda68716a1d95cee706c`: the documented two-metric smoke and repeat passed with 8 rows, then `added=0 skipped=8`. Rust 1.99.0, just 1.57.0 and CMake 4.4.3 were present. The 12 GiB wrapper recorded 0.95 GiB peak RSS over the complete clone/build/two-run command. This was not a clean Homebrew installation test.
Install Apple's command-line tools and Homebrew first, then:

```sh
xcode-select --install
brew install rustup just cmake pkg-config
export PATH="$(brew --prefix rustup)/bin:$PATH"
rustup toolchain install stable --profile minimal
rustup default stable
bash scripts/ci/clone-siblings.sh
just bench-iqa smoke ssim2,gmsd --host mac --fixture synthetic-v1 --skip-native
```

When a configured resource wrapper is present, cells use its 12 GiB cap. Plain
execution without a wrapper records that it has no memory-cap qualification.
macOS Intel was not tested and is outside this smoke matrix.

## Windows

Tested on `windows-latest` 2026-10-17, commit `f95192aaa39986cc5438fda68716a1d95cee706c`: CPU build and execution passed twice, each with 8 successful n>=5 rows. **Resume failed**: the two invocations produced different worker-binary hashes and separate content keys. The recipe now uses one Cargo build configuration and executes the coordinator directly; its Windows rerun is pending. Do not treat the failed resume check as passed.
The following clean-install commands are untested; the workflow uses its
preprovisioned C++ toolchain and explicit Rust/just setup actions. Install
Visual Studio 2022 Build Tools with **Desktop development with C++**,
Git for Windows (including Git Bash), Rust stable and CMake. In PowerShell:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact
winget install --id Git.Git --exact
winget install --id Rustlang.Rustup --exact
winget install --id Kitware.CMake --exact
```

Open a new PowerShell so Rustup's PATH changes are visible, then run
`cargo install just --locked`. Select the C++ workload in Visual Studio Installer; then open Git Bash in the
checkout and run the two commands at the top. Windows smoke uses native MSVC;
WSL is a separate Linux testbed. The portable worker has an 1100-second child
kill/wait bound. The plain Windows path has no Job-object memory ceiling or
CPU-affinity qualification; the smoke does not claim either.

## Results and opt-ins

Results are content-addressed under `target/bench-iqa/<key>/`, or `--output DIR`.
`rows.jsonl`, per-cell logs, input hashes and `run.meta` retain the commit,
binary SHA-256, OS, explicit host label, thread budgets and toolchain versions.
A repeated command skips completed modes within the same content identity.
Dashboard projections are written to the registered ignored directory
`benchmarks/iqa_compete_2026-10-16/repro-results/<key>/`. The dashboard lane's
existing build ingests those shards; retain or submit the generated shards
with their `run.meta` when publishing results. CI uploads both raw and dashboard
outputs as artifacts, rather than committing images or large generated data.

For the advanced existing harness, explicitly install Python 3.12 and uv; native
competitors also need Meson, Ninja, Highway, Brotli and LittleCMS2. See the
[study protocol](../benchmarks/iqa_compete_2026-10-16/README.md). Run
`just bench-iqa-advanced ...` to opt into that path; `--gpu-runtime cuda|wgpu`,
caller-provided licensed `--model-root`, and Python competitors are explicit
choices there. CUDA also requires a compatible NVIDIA driver/runtime. None is
needed by the portable CPU smoke.

The manual `Portable IQA benchmark smoke` workflow runs Linux, macOS ARM64 and
Windows with two metrics at one small size, checks resume and uploads artifacts.
No full benchmark matrix is started by that workflow.

Smoke run and artifacts: [three-OS workflow, f95192aa](https://github.com/imazen/zenmetrics/actions/runs/39029766690). Runner outcomes are recorded below only after completion. CI artifacts use the `ci` label and are smoke diagnostics, not published CPU-host performance rows.

## Optional external DVIFM panel

`--with-dvifmish` probes an optional checkout at `DVIFMISH_DIR`, otherwise
`../dvifmish` relative to this checkout, and C++ builds at `DVIFM_CPP_DIR`.
Build its `bench_pair` release example and C++ benchmark workers using that
checkout's own recipes first. The external driver owns presets and agreement
labels. Missing implementations each produce an explicit `unavailable` line
and entry in the raw `run.meta`, without failing the ordinary CPU panel.
Synthetic smoke inputs cannot enter this canonical-input-only external panel.
An available worker failure remains an error.

The panel requires Python 3 and runs under the configured resource wrapper when
available. It writes a separate `private-dvifm` directory within the raw output;
its results and optional metadata never enter dashboard projections. Keep that
raw output private. The default command does not invoke the optional panel.

## Dashboard build versions

Competition ingestion preserves `impl_version` as `implementation_version`.
Speed filters and table exports expose it; ranking labels include it. Cost fits,
thread-scaling baselines and rank-change identities separate implementation
versions, including different compiler/build binaries of the same method.
Unversioned historical rows retain an unspecified version. External measurement
registration remains separate from version support.
