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

CUDA completed both evaluations in 72.60 seconds process wall time, with
maximum RSS 1,293,308 KiB and identical score bits.

RTen completed its first evaluation in 845.578715 seconds. The coordinator
terminated the process during the second evaluation at 1607.74 seconds
process wall, with maximum observed RSS 744,300 KiB. The first score is
retained in `cap-rten-interrupted.log`; complete two-evaluation time and repeat
identity are unmeasured. This is a shared i265 observation under load. It is
not a lower bound of 1607 seconds for one evaluation: the log contains a
completed first call. The cap will not be rerun.

The combined initial workflow exited 143 after the coordinator's stop:
`run-heavy: done rc=143 1686s | peak-RSS 1.23GiB | min-avail 18567MiB | peak-load 24.44`.
Remaining checks run with separate lock acquisitions, bounded below twenty
minutes each. Full review validation is still in progress.
