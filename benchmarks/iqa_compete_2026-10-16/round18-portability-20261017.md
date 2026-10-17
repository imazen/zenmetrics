# Round 18 portability receipts — 2026-10-17

MISSING: the Rust-only entry requires explicit `--skip-native`; a CMake-only native competitor build is not implemented. Manual clean-machine install commands are labelled untested. Windows Actions is still running and macOS Actions is queued at this receipt update.

Implementation tested: `f95192aaa39986cc5438fda68716a1d95cee706c`.

| Test | Result | Resource receipt |
| --- | --- | --- |
| i265 Linux, 2 metrics × 64×64, 1/20 threads, both modes | 8 timed rows, n≥5; repeat added=0 skipped=8 | run-heavy rc=0, 5 s, peak RSS 0.13 GiB, min available 21802 MiB, peak load 1.58 |
| mac Apple M4 Pro ARM64, same panel, 1/12 threads | 8 timed rows, n≥5; repeat added=0 skipped=8 | 12 GiB wrapper; rc=0, 147 s clone/build/two runs, peak RSS 0.95 GiB, min available proxy 13621 MiB, peak load 7.91 |
| ubuntu-latest Actions | passed smoke, resume and artifact upload; downloaded artifact has 8 successful n≥5 rows | dedicated runner; no host run-heavy cap claimed |

Linux/macOS raw fixture SHA-256s match. Local CPU dependency-tree audit contains no `ort`, `ort-sys`, `cudarc`, `cubecl` or `zenmetrics-nn` normal dependency.

Local gates passed: fmt; CLI clippy; portable coordinator/worker clippy; 14/14 CLI feature combinations; 108 harness tests; hygiene (5346 tracked files); actionlint 1.7.12. Gate wrapper rc=0, 48 s, peak RSS 0.41 GiB, min available 21214 MiB, peak load 1.32.

[Manual three-OS smoke and artifacts](https://github.com/imazen/zenmetrics/actions/runs/39029766690). This run uses only two metrics at one small size; no full matrix.

Private raw retention: `~/tmp/iqa-compete/round18-linux-final/`, `~/tmp/iqa-compete/round18-mac/`, `~/tmp/iqa-compete/round18-ci-linux/`.

## Log identities

- `round18-gates-v2.log`: SHA-256 `e06d199cd1183ad9e41cecc48a55edcdd601d5b06ca40e51ed28c2e15e8581f1`.
- `round18-linux-final.log`: SHA-256 `dd2bc171f14c675b5118e9d5afbca1e83c37532f66e3e57c76596c679c70f44e`.
- `round18-feature-tree.log`: SHA-256 `a82bba2da718319eb32acd079224491e88eb8eda54ceda4d9fb5cd1529b6a252`.
- `round18-actionlint-final.log`: SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- `round18-mac/smoke.log`: SHA-256 `b7495192638bd80b4d9ca12f16ca60046b6cd2f5fd38046cdcc72199b1279dc2`.
