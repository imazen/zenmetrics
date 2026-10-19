# Execution coverage, 2026-10-19

Missing: scalar execution evidence for 90 of the 124 included CLI names,
map comparisons for applicable routes, and qualified timings for every name.
The coverage table indexes published artifacts only; active local studies
are added after their results have been checked and published.

[Per-name coverage](execution-coverage.tsv) has scalar observations for
34/124 included names: 13 names have complete scalar grids within existing
gates, GMSD has 18 even-size scalar/map passes and six instrumented canonical
odd-size failures, and twenty SSIM-family names have observations without
assigned cross-implementation bounds. None of these totals implies complete
validation of the metric: timing and applicable maps are separate columns.
The retained timing table is a retired partial cohort, not the active run.

The table also retains the three excluded source-census rows with zero
required cases. `wd` is assigned to the learned phase as requested, correcting
the scheduling classification without rewriting the historical source census.
Historical source claims remain in [the source inventory](inventory-source.md).

[The coverage index](execution-coverage.json) hashes each selected source
index and timing prefix. It selects the newer resize dependency for VSI;
original image-2.12 failures remain in their separately retained cohort.
The generator verifies every input shard hash and deduplicates output
columns into (pair, size, threads) cases. It assigns no new bounds.

Regenerate with `python3 -B benchmarks/port_spotcheck_2026-10-18/make_execution_coverage.py benchmarks/port_spotcheck_2026-10-18`.
This reads committed scalar tables and writes a small index; it executes no
metric and makes no performance claim.
