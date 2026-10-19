# Execution coverage, 2026-10-19

Missing: scalar execution evidence for 87 of the 124 included CLI names,
map comparisons for applicable routes, and qualified timings for every name.
The coverage table indexes published artifacts only; active local studies
are added after their results have been checked and published.

[Per-name coverage](execution-coverage.tsv) has scalar observations for
37/124 included names: 25 names have complete scalar grids within their
existing or authorized external gates, nine SSIM-family names have scalar
passes plus separately counted matching size refusals, GMSD has eighteen
even-size scalar/map passes and six instrumented canonical odd-size failures,
VIFp has 22 numerical misses under its unchanged bound, and SSIMULACRA2
has observations without an assigned external bound. None of these totals implies complete
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
columns into (pair, size, threads) cases. It reads the separately authorized PyIQA SSIM policy-applied cohort and keeps
matching size refusals separate from scalar passes.

Regenerate with `python3 -B benchmarks/port_spotcheck_2026-10-18/make_execution_coverage.py benchmarks/port_spotcheck_2026-10-18`.
This reads committed scalar tables and writes a small index; it executes no
metric and makes no performance claim.
