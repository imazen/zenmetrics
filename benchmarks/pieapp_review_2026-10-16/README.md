# PieAPP review measurements

Host: i265. Release build with runtime dispatch, no target-cpu=native.
Production source: `995d6ba3` (review fixes on merge `f62324c4`).

The cap driver measures model loading and two cold evaluations of the same
procedural RGB pair at 3000×3000, checks repeat bits, and reports each call.
`/usr/bin/time -v` surrounds the complete process, including input creation.
The initial cap workflow ran under the shared heavy lock and
`run-heavy --mem 16G --jobs 16`. RTen uses `RTEN_NUM_THREADS=16`.
The remaining review steps acquire and release that lock independently.

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

Reproduce the remaining gates with `just nn-pieapp-review MODEL_DIR LADDER
NEW_OUTPUT`, outside any enclosing flock. Set `LOCK_SNAP_DIR` explicitly.
Create the unchanged golden groups first with `just nn-pieapp-review-shards
MODEL_DIR NEW_OUTPUT/shards`. Each step has a 1150-second timeout followed
by at most twenty seconds for termination. The runner never repeats the cap.
The head and weight-sign probes have their own bounded lock acquisitions.

Revision 3 corrects the independently reconstructed weight head to signed
`weight + 1e-6`. `weight-identify.json` records fourteen opaque public-output
controls; `reconstruction-v3.json` records exact equality against each control.
`export-v3.json` identifies the fresh graph and its separate export conversion
checks. End-to-end native qualification is pending. `konjnd-shards.json` binds
four disjoint 252-pair groups to the unchanged original 1008-pair input table.
The original cap files above measure revision 2; the corrected CUDA cap will
have a distinct receipt. The interrupted RTen cap is never repeated.

Fresh export/control step resource receipt:
`run-heavy: done rc=0 7s | peak-RSS 1.51GiB | min-avail 24791MiB | peak-load 11.37`.
