# Completed score references after timing timeout

A process can return its initial CPU score and then exceed the outer deadline
before collecting any zenbench rounds. That returned score can establish a
same-input algorithm comparison. It cannot establish CPU compute latency.

The reference reader accepts exactly one completed, finite first-score diagnostic
from a CPU score row whose retained status is `skipped:timeout_*`, with its
original log path and SHA-256. It creates an in-memory comparison reference;
the source row, timeout status, zero samples and missing timing statistics stay
unchanged. A qualified CPU timing takes precedence. Matching still requires
the exact host, OS, metric, raw-pixel hash, size and thread budget.

Comparison rows disclose `reference_kind=completed_untimed_warmup`, the source
timeout status and log SHA-256. Charts label this reference explicitly. The
ten-round CPU timing minimum and every existing score-parity bound remain
unchanged. Missing, invalid, nonfinite or incomplete diagnostics cannot supply
a reference, nor can diagnostics from unrelated process failures.

The [actual PieAPP control](warmup-reference-control.json) records the 2K/1T
CPU timeout with no timed rounds and the completed score used to compare the
already measured CUDA score. Its log and immutable source collection are
retained outside the repository. No new inference was run for that comparison.
