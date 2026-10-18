# IQA chart snapshot, 2026-10-18

Observed durable benchmark attempts from i265 and ARM64 mac, captured at the
UTC times in each `snapshot.json`. This is a fixed historical snapshot, not live
coverage. The user explicitly requested committing structured measurement data.

`observations.jsonl` contains one row per recorded attempt. `sources.json` joins
its `source` SHA256 to the source report, frozen manifest identity and execution
policy. `snapshot.json` binds both files by SHA256. Times are nanoseconds;
`round_means_ns` are real warm round batch means, not individual-call percentiles.
Cold timing includes scorer construction and first scoring, excluding RGB loading.

Use only `qualified_timing=true` for timing charts. Warm qualification requires
at least ten real rounds with every original gate clean. Failures, unsupported
routes and flagged timing rows remain present. Null timing fields are missing
observations. Do not turn them into zero. Scores across different metrics are
not comparable perceptual-quality measurements.

The snapshot includes production, controls, interrupted/retried cells and immutable
successor outputs. It deliberately does not collapse these into one headline row.
Select source reports explicitly and retain `impl_binary_sha`, input hashes,
geometry, mode, phase, worker count and cold repeat when grouping. Multiple
qualified attempts or reused lineages do not represent additional distinct coverage.
Existing 8K observations are retained even though future 8K work is user-disabled.
The source reports are captured atomically one by one; capture is not simultaneous
across hosts. Raw logs, resource receipts and frozen bundles remain on their hosts.

Reproduce the projection with
`just iqa-chart-export ROOT NEW_OUTPUT HOST` on each host. The exporter reads
existing reports only and never retries, recovers or modifies an attempt.
