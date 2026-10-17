# Reproduce IQA CPU benchmarks

The portable CPU command uses Rust stable, just and the checked-out workspace's
pinned sibling dependencies. It enables CPU metrics and threading, with no ORT,
CUDA, neural models or Python. Use a C/C++ toolchain and CMake for native build
prerequisites. Git is needed to clone this workspace and its pinned siblings.

```sh
bash scripts/ci/clone-siblings.sh
just bench-iqa smoke ssim2,gmsd --host mycpu --fixture synthetic-v1 --skip-native
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

Runtime test status: pending Round 18 smoke; not yet claimed tested.

```sh
sudo apt-get update
sudo apt-get install -y build-essential git curl cmake pkg-config
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"
cargo install just --locked
bash scripts/ci/clone-siblings.sh
just bench-iqa smoke ssim2,gmsd --host linuxcpu --fixture synthetic-v1 --skip-native
```

## macOS ARM64

Runtime test status: pending Round 18 smoke; not yet claimed tested.
Install Apple's command-line tools and Homebrew first, then:

```sh
xcode-select --install
brew install rustup-init just cmake pkg-config
rustup-init -y --profile minimal
. "$HOME/.cargo/env"
bash scripts/ci/clone-siblings.sh
just bench-iqa smoke ssim2,gmsd --host mac --fixture synthetic-v1 --skip-native
```

When a configured resource wrapper is present, cells use its 12 GiB cap. Plain
execution without a wrapper records that it has no memory-cap qualification.
macOS Intel was not tested and is outside this smoke matrix.

## Windows

Runtime test status: pending GitHub Actions windows-latest smoke; untested.
Install Visual Studio 2022 Build Tools with **Desktop development with C++**,
Git for Windows (including Git Bash), Rust stable and CMake. In PowerShell:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact
winget install --id Git.Git --exact
winget install --id Rustlang.Rustup --exact
winget install --id Kitware.CMake --exact
cargo install just --locked
```

Select the C++ workload in Visual Studio Installer; then open Git Bash in the
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
