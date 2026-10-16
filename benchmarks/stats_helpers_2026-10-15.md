# Student and directional-bootstrap helper validation

The public helpers reuse the crate's existing Student owner. Independent scipy
1.18.1 `t.ppf` and `t.sf` fixtures cover df 1, 2, 9 and 30 at all nine Holm tail
levels, with additional tail and invalid-input cases. The fixture generator is
`crates/zenmetrics-stats/scripts/student_reference.py`.

`Interval::p_nonpositive` uses finite draws only and includes zero ties. Tests
cover positive and negative constants, signed zero, no finite draws, and a
mixture of negative, zero, positive and nonfinite outcomes. The two-sided
`p_zero` calculation and finite-draw count are unchanged. Downstream struct
literals must initialize the added field; no dependency or lockfile changed.

Implementation commits: `d61c43fc` (Student), `737dc977` (directional probability).
All runs used the repository's pinned snapshot (23 sibling pins):

```sh
TMPDIR="$HOME/tmp/e34prep" CARGO_PROFILE_DEV_DEBUG=0 \
LOCK_SNAP_DIR="$HOME/tmp/e34stats-lock-snap" \
"$HOME/work/claudehints/scripts/run-heavy" --mem 24G --jobs 8 -- \
just stats-helpers-check

TMPDIR="$HOME/tmp/e34prep" CARGO_PROFILE_DEV_DEBUG=0 \
LOCK_SNAP_DIR="$HOME/tmp/e34stats-lock-snap" \
"$HOME/work/claudehints/scripts/run-heavy" --mem 24G --jobs 8 -- \
just stats-helpers-serial-check
```

Default-feature and no-default-feature suites both passed: 98 unit tests,
1 coverage test, 9 independent-reference tests and 2 Student tests. Doc tests
passed (none present). All-target Clippy passed with `-D warnings`.

```text
default tests + Clippy: rc=0 10s | peak-RSS 0.25GiB | min-avail 39085MiB | peak-load 27.02
no-default-features: rc=0 26s | peak-RSS 0.23GiB | min-avail 37584MiB | peak-load 30.10
```

These resource lines include snapshot setup and are not a statistics benchmark.
Full logs: `~/tmp/e34prep/stats-directional-check-fixed.log` and
`~/tmp/e34prep/stats-serial-check.log`. Existing Cargo manifest warnings report
ignored non-root profiles and an unused workspace patch; no source lint remains.
