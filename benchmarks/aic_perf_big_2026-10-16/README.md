# CAMBI and Butteraugli-original CPU work

Baseline production source: `849ea8da`. Build with runtime dispatch and no
`target-cpu=native`. The CAMBI driver enables `parallel,simd,avx512`; the
Butteraugli driver imports the production module directly and compiles with
`rustc -O -g`. The latter module uses f32 image planes and selected f64
expressions; preserving their types is part of exact parity.

`just aic-perf-big-baseline OUTPUT` builds and preserves both binaries, records
single-thread instructions/cycles and wall time, records sixteen-thread wall
time, and captures native perf samples. The orchestrator runs outside the shared lock and gives each build, wall
measurement and profile a separate bounded lock/run-heavy resource scope.
Set a dedicated `LOCK_SNAP_DIR`; hardware counters use `sudo -n perf` because
the host restricts unprivileged perf. The immutable shell body is loaded before
execution. Full output is retained in the caller's log; perf data stays outside
git and its textual reports will accompany the results here.

`just aic-perf-big-parity <baseline-dir> <candidate-dir> <output.tsv> <scratch>`
compares score bits and hashes of complete Butteraugli error maps on the AIC
panel. CAMBI uses the CLI's even crop and ordered studio-601 luma expression.
An identical refusal is retained as an explicit result, rather than silently
skipping a pair. Both drivers also provide procedural wall inputs; generation
and CAMBI luma construction are outside the timed scoring loop.

Measurements and optimization qualification are pending.

The initial noise scout returns exactly zero CAMBI at 1024 square. Its
receipts and binary remain a separate control. Before changing metric code,
the CAMBI wall fixture is extended from the original-author pinned 10-bit
block-eight, one-code-value staircase. The banded baseline uses the updated
driver and a separate preserved binary; comparisons use that same driver
on both sides. The 61-pair path and its studio-601 ingress are unchanged.
