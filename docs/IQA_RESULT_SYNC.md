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

After fixing a guard rejection and completing the normal guarded push, run
`just iqa-results-sync-recover CONFIG FULL_COMMIT`. It verifies that commit is
on remote master, checks the public projection, and requires every saved-data
file to match the committed bytes before updating private sync bookkeeping.
It neither pushes nor changes repository files. The cycle lock prevents a
concurrent capture from changing bookkeeping during recovery.

The coordinator runs on efficiency cores at low CPU/I/O priority, with a 2 GiB
service memory limit. This is result capture/publication, not a scoring process;
benchmark workers retain their own shared heavy locks and placement gates.

### Scatter measurements

Each cycle also projects `*-rounds.csv`: one saved cold repeat or warm zenbench
round per row, with `time_ns`, exact metric/host/binary/input/thread identities,
phase, round index and qualification. A warm point is the measured per-call
mean within a round, not an invented timing for each iteration. The sync's
evidence fingerprint covers both attempt and round CSVs. No raw image-quality
scores enter either table. Missing individual round arrays stay missing.

Private NN campaigns can enter the same timing-only projection after explicit
selection in the private sync config. Their campaign names start with
`private-nn-`; provenance exposes content hashes without model file names or
local paths. Saved controls and remaining-only production cells share exact
identity keys, so preferred selection counts each cell once. Timing controls
do not establish native numerical parity: DMM retains `failed_native_gate` in
the export even when its timing gate passes. Unrelated public NN jobs remain
stopped.
