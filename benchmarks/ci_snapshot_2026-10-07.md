# CI lock snapshot export measurement

Implementation: `8cbf9cbb9673ad300043fa130daac6a3dd2865b0`.
Date: 2026-10-07. Host: `dev`. Command: `du --apparent-size -s -B1 <directory>`.
Measurements run through `run-heavy --mem 16G --jobs 8` on the same host.

| Snapshot | zenmetrics source subtree (bytes) | Entire snapshot (bytes) |
|---|---:|---:|
| Prior workerfix | 17,607,365,824 | 18,120,559,570 |
| Prior workerfix2 | 17,607,366,801 | 18,120,560,557 |
| Tracked-tree export at `8cbf9cbb` | 51,234,005 | 514,303,898 |

The new exporter copies tracked files from a single jj snapshot, including
working source changes. Marked Cargo output directories, repository metadata,
ignored local overrides and generated data are excluded. All 23 sibling
revisions were checked against `ci/sibling-pins.tsv`; cloning is unchanged.
The source inventory matched all 3,159 tracked paths exactly, with zero marked
Cargo output directories. Cargo.lock and sibling-pins.tsv retained their hashes.
No old snapshots or build outputs were deleted for this measurement.

Exporter setup failures return exit 2; lock mismatches return exit 1 without
rewriting the lock. Both `--check` and `--check --rev` pass.
Four synthetic regressions cover working source edits/additions/deletions,
executable bits, relative links, ignored output, accidentally tracked Cargo
output, immutable revisions, unchanged regen, lock mismatch refusal and setup
failure status. With unchanged assertions, the prior exporter fails the two
cache-copy cases; all four pass with the fix.

The new snapshot also passes all 57 worker unit tests and 8 integration tests,
worker clippy with `-D warnings`, and scoped worker formatting. The combined
run measured: `run-heavy: done rc=0 8s | peak-RSS 0.90GiB |
min-avail 48570MiB | peak-load 2.54`. Size measurement measured:
`run-heavy: done rc=0 0s | peak-RSS 0.01GiB | min-avail 49552MiB |
peak-load 1.70`. Neither is a performance comparison under controlled load.

Full logs, snapshot inventories and final export measurements are retained in
`~/tmp/snapfix/`; completion evidence is in the requested `SNAPFIX_DONE.md`.
Run-heavy resource lines are in those logs; these are storage measurements,
not a build-time or memory-performance comparison.
