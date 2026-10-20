# Latest synchronized IQA results

The five-minute result synchronization service refreshes this directory only
from finalized saved observations on mac, i265 and r7900x. Git history retains
each published snapshot. The timestamped initial capture remains under
`benchmarks/iqa_saved_2026-10-20T0056Z/`.

Select `preferred=True` qualified rows. Preserve exact binary, input, variant,
geometry, worker, engine and cold/warm identities; never pool revisions or
protocols. Pending, refused and gate-flagged attempts remain explicit.
Only canonical scalar, one/eight-worker, non-8K observations are published.
Private metrics use the existing aggregate allowlist; no scores, private code,
model assets or raw capture paths are projected.

`sync.json` records the latest capture window and evidence identity. Hashes,
coverage and protocol records accompany the three host CSVs. A cycle with no
new measured evidence makes no commit. Source checkout ownership or uncommitted
work defers publication without changing that work. Network/validation failure
retains the previous verified export and writes a local error receipt.

See `docs/IQA_RESULT_SYNC.md` for service operation and logs.

## Individual measurements for scatter plots

Use `mac-rounds.csv`, `i265-rounds.csv` and `r7900x-rounds.csv` for individual
points. Filter `preferred=True` and `qualified_timing=True`. Plot `time_ns`
(or divide by 1000000 for milliseconds) against pixels (`width * height`),
metric, host or thread count. Separate cold and warm phases and retain binary,
input, engine and variant identities. `round_index` is zero-based;
`cold_repeat` identifies the three independent cold processes.

A cold point is one fresh-process constructor-plus-first-pair measurement.
A warm point is one zenbench round's mean time per call. Its `iterations`,
`elapsed_total_ns` and `compensated_total_ns` are retained when available;
iterations within a round were not independently timed. These are timing
samples, not image-quality scores or AIC agreement observations.

Older saved captures retain round means but may lack per-round gate detail.
A blank `gate_clean` means unavailable, never clean by assumption. All-clean
attempt receipts imply clean rounds; otherwise individual gate flags remain
unknown until detailed saved samples are available. The attempt-level
qualification is always retained. Private CW-SSIM's existing allowlisted
aggregate lacks warm round arrays, so those warm points are absent rather than
imputed from its median. Its individual cold repeats remain available.
