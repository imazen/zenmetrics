# Five-minute IQA result synchronization

Owner direction: capture, commit, push and synchronize incoming saved results
every five minutes. This schedule never stops or launches a benchmark.

`just iqa-results-sync CONFIG` runs one cycle. CONFIG is private, outside git:
it names the three hosts' saved campaigns, remote scratch directories and the
coordinator repository. Read-only captures run concurrently with low priority;
active unfinished attempts do not become observations. Capture/export helpers
are copied from the committed source revision into each cycle's private scratch.

The public projection reuses the allowlisted saved-result exporter and validates
checksums, scalar mode, worker counts, geometry and protocol before publication.
Only `benchmarks/iqa_saved_latest/` is refreshed. No raw scores, model assets or
private implementation details enter that projection. Existing timestamped
captures and all source benchmark attempts remain unchanged.

The coordinator takes `.workongoing` atomically before repository mutation and
requires an empty working copy. An occupied or dirty checkout defers publication.
It fetches master, commits the new evidence, pushes through `scripts/safe_push.sh`
and verifies remote ancestry. Evidence-identical captures create no extra commit.
A source push racing the result push is retried after a normal forward rebase;
there is no force push or approval-guard bypass.

The user service `iqa-results-sync.service` is driven by
`iqa-results-sync.timer` at five-minute UTC boundaries. Inspect with
`systemctl --user status iqa-results-sync.timer iqa-results-sync.service` and
`journalctl --user -u iqa-results-sync.service`. The private config's scratch
root contains `status.json`, `last-published.json`, immutable raw capture cycles
and full push logs. A failed cycle preserves the previous remote export.

The coordinator runs on efficiency cores at low CPU/I/O priority, with a 2 GiB
service memory limit. This is result capture/publication, not a scoring process;
benchmark workers retain their own shared heavy locks and placement gates.
