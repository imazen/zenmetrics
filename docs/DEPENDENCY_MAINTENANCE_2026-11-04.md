# Consolidated dependency maintenance, 2026-11-04

The dependency updates proposed in PRs 97–103 and 105–107 are applied together,
without merging their overlapping lockfile branches. PR 101's lucide-react
1.61.0 is superseded by PR 107's 1.65.0.

| Component | Selected update |
| --- | --- |
| Fleet dashboard | lucide-react 1.65.0, Vite 8.3.5, Node types 26.6.6 |
| Burn RankNet spike | burn 0.22.1, forust-ml 0.8.0, rustls 0.23.46 |
| Burn convolution spike | rustls 0.23.46 |
| Benchmark Python environment | filelock 4.2.1, setuptools 84.1.0 |
| Standalone benchmark runner | wasmtime and wasmtime-wasi 50.0.1 together |

The Wasmtime proposal updated only wasmtime. Updating its WASI companion in
the same graph avoids mixing the engine/store types of two major releases.
The package manager selected the available 50.0.1 patch. The source package's
base MSRV is unchanged; the optional Wasmtime dependency reports Rust 1.97.

Release notes: [Wasmtime 50](https://github.com/bytecodealliance/wasmtime/releases/tag/v50.0.0),
[burn 0.22.1](https://github.com/tracel-ai/burn/releases/tag/v0.22.1).

## Validation

The following checks pass on the final dependency source tree:

- Dashboard clean install and TypeScript/Vite production build.
- `uv lock --check` for the benchmark Python environment (59 resolved packages).
- All four existing runner WASM smoke tests: scalar/SIMD export calls,
  independent instances, WASI Preview 1 imports, and refusal without WASI.
- Both RankNet spike targets compile with burn/forust updates.
- The convolution spike compiles with the updated rustls lock.

No GPU kernel or training run was performed. The convolution spike retains
nine pre-existing float-literal fallback warnings. Historical benchmark
results and their frozen bundles are not regenerated or requalified.

Reproduce with `just maintenance-deps-check`, under the normal resource wrapper
and a disk-backed TMPDIR. An isolated `CARGO_TARGET_DIR` is supported. The full
stdout/stderr of the review run is retained at
`$HOME/tmp/zenmetrics-pr-review-20261104/dependency-tests.log` on the review host.
The parent workspace Cargo.lock and sibling pins are unchanged.
