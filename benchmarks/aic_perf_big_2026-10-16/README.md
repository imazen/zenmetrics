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

The first auto-PMU scouts retain their separate core/atom counter lines.
Subsequent measurements pin perf itself to CPU 2 and explicitly select
`cpu_core/instructions/` and `cpu_core/cycles/`; this avoids scaling an
incidental pre-affinity atom-core sample. CPU 2 belongs to the observed
`/sys/devices/cpu_core/cpus` set `0-7`.

The nonzero banded CAMBI profile (`baseline-banded-cambi.profile.txt`) assigns
62.36% of samples to histogram row updates and 16.31% to c-value evaluation.
The first optimization therefore targets integer histogram updates, while
preserving all floating-point c-value calculations and pooling order.

The first histogram-run candidate passes the pinned CAMBI scores, stride and
map/score tests and a direct ordered-pixel-window comparison across clipped
runs, pads and wrapping increments/decrements. `cambi-candidate1.log` retains
the initial candidate scout; its score bits match both banded baselines.
These initial observations are not the final interleaved A/B result.

After convolution tiling, the preserved candidate profile puts 55.32% of
samples in Malta neighborhoods and 17.39% in convolution. The candidate
passes all standalone Butteraugli tests and preserves both scout score bits.
The next experiment processes eight independent interior Malta pixels per
window, retaining each pixel's neighbor/path order and the original border
path. It introduces no reduction across pixels.

The integer-run scout is superseded and is not the production change: it
proved bit identity but regrouped histogram updates, contrary to the required
ordering contract. Its helper/assertions remain test-only research. Production
keeps the original serial histogram function text-identical and schedules
independent column regions only when the existing parallel feature/pool and
geometry permit it. Region widths preserve original eight-lane groups and
scalar tails; each histogram cell receives the original ordered +/-1 updates.
The original floating-point point evaluator and final pooling are unchanged.
