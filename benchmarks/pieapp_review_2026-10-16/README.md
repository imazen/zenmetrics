# PieAPP review measurements

Host: i265. Release build with runtime dispatch, no target-cpu=native.
Production source: `995d6ba3` (review fixes on merge `f62324c4`).

The cap driver measures model loading and two cold evaluations of the same
procedural RGB pair at 3000×3000, checks repeat bits, and reports each call.
`/usr/bin/time -v` surrounds the complete process, including input creation.
The enclosing review workflow runs under the shared heavy lock and
`run-heavy --mem 16G --jobs 16`. RTen uses `RTEN_NUM_THREADS=16`.

CUDA cap start load average: `6.84 101.42 174.97` (1/5/15 minutes).
RTen cap start load average: `3.56 79.40 161.57`.
These are cap/resource observations, not quiet-machine throughput claims.

Full review run and remaining measurements are in progress.
