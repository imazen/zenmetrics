# Round 18 portability receipts — 2026-10-17

MISSING: the Rust-only entry requires explicit `--skip-native`; a CMake-only native competitor build is not implemented. Manual clean-machine install commands remain labelled untested. The Windows resume fix is verified on all three Actions OSes below.

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

## Windows result and correction

Downloaded Windows artifact: two cohorts, each with 8 successful n>=5 rows, source f95192aa. Worker hashes: `2eb19ac32026cc46f7cddccfc02e91754ea3804a5c8e894b50a055e676feadbd` and `257b7c1247d3bd0122bb5075cc86ed325906c1f146c52b11306d5f218f9368c6`. The resume gate failed and remains recorded as failure. The recipe correction invokes the built executable directly after one Cargo build configuration, preserving the strict binary-hash content key and unchanged zero-new-cells assertion. That rerun also failed; the measured metadata fix and successful final run are recorded below.

## Windows resume investigation

The single-build recipe correction did not fix Windows resume: run
[39030620635](https://github.com/imazen/zenmetrics/actions/runs/39030620635)
also produced two executable hashes. The original completed log shows
`zenmetrics-cli` recompilation and worker relinking during the repeat.
Identity fields other than the executable hash were identical.

Run [39032428725](https://github.com/imazen/zenmetrics/actions/runs/39032428725)
retained both Windows executables. Each is 17,592,320 bytes. Exactly 20 bytes
differ: one byte in each of four timestamp fields and the 16-byte RSDS PDB
GUID. The COFF timestamp starts at file offset 240; debug timestamps start at
15003396, 15003424 and 15003452; the GUID occupies 15003756–15003771.
No code or other data differs. Normalizing the four full timestamp fields and
GUID makes the complete files identical, verified with the Rust normalizer.

| Identity | First executable | Rebuilt executable |
| --- | --- | --- |
| Full SHA-256 | `ff69e6b4de4385aed88a9cb407aa441a9444be84e285b2142a1193166299cb81` | `21b4f1f01139fbfc2de047b9bcc7fe7929064b74b3b2e20e284f1cf4e93609fc` |
| Normalized SHA-256 | `e3d9cf66dea537617c34128d2fb529cc5fd1cc3032a875f864aa3bf505538798` | `e3d9cf66dea537617c34128d2fb529cc5fd1cc3032a875f864aa3bf505538798` |

The parser follows the [Microsoft PE layout](https://learn.microsoft.com/windows/win32/debug/pe-format),
supports PE32/PE32+, and retains all other bytes, including code, machine,
section layout, PDB path/age and unknown debug data. Unit controls require
those changes to invalidate the identity and reject truncated directories.
Non-PE files remain byte exact. Raw timing rows retain exact executable hashes;
`run.meta` labels the normalized identity and records the exact current file
hash separately. The zero-new-timed-cells assertion remains unchanged.

The timestamp-only candidate failed the Windows repeat; that failure led to
the complete byte comparison above. The first diagnostic capture also failed
artifact upload because it mixed Windows drives; subsequent captures keep all
files beneath the checkout.


## Final resume verification

[Run 39033906478](https://github.com/imazen/zenmetrics/actions/runs/39033906478),
source `20df8b7d8155a29765e902538e410fd4a7331b39`, completed **success** on
2026-10-17. All three jobs passed normalization tests, the first benchmark,
the unchanged zero-new-timed-cells assertion and artifact upload.

| Runner | Threads | Timed rows | Minimum n | Repeat |
| --- | --- | --- | --- | --- |
| ubuntu-latest | 1/4 | 8 | 5 | added=0 skipped=8 |
| macos-latest ARM64 | 1/3 | 8 | 5 | added=0 skipped=8 |
| windows-latest | 1/4 | 8 | 5 | added=0 skipped=8 |

Downloaded raw artifacts each contain one cohort, the expected source SHA,
and `pe-relink-metadata-v1`. The final captured Windows executables differ
bytewise but normalize identically with the Rust implementation. Exact file
hashes remain in the receipts. Linux/macOS hashes are not normalized.

The landing tree was rebased onto master and passed fmt, CLI/coordinator
clippy, all 14 CLI feature cells, 108 Python harness tests, all 4 Rust
coordinator tests and hygiene (5429 tracked files). Resource wrapper: rc=0,
52 s, peak RSS 0.41 GiB, min available 13005 MiB, peak load 2.15.
Actionlint 1.7.12 passed. These are correctness/resume controls, not speed claims.
