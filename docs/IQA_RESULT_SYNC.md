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
requires an empty working copy. An occupied or dirty checkout blocks publication
and fails the cycle (below).
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

### A blocked sync fails loudly

A cycle with new evidence that cannot publish it exits 2, so
`systemctl --user status iqa-results-sync.service` shows failed and the journal
carries an `IQA RESULT SYNC BLOCKED since … (N cycles): <reason>` line with
the detail. This covers a lock held by another lane, a dirty checkout (including
the commit a refused push leaves behind), a checkout parent not on master, a
refused or failed guarded push (`push refused by dashboard guard`, `… hygiene
guard`, `… lock guard`, not a fast-forward, transport) and a failed capture.
`status.json` records `blocked`, a fixed `reason` class, `stalled_since`,
`blocked_cycles`, the first reason, and private `detail` (diff summary, or the
traceback plus the last 25 lines of `push.log`). A cycle that publishes, or a
recovery, clears it. "No new measured evidence" is not a failure.

The dashboard watcher reads this status over ssh each cycle (`BRIEF_SYNC_HOST`,
`BRIEF_SYNC_PATH`) and the page shows a `results sync blocked` check, warn at
first and error after 15 minutes; `results sync not running` (error) when the
status is over 20 minutes old; and a warning when it cannot be read. Only the
reason class and times reach the page: guard output can quote the lines it
matched, so the detail stays private. Tests: `just iqa-results-sync-check` and
`just brief-sync-check`.

Concurrent forward updates are retried at both rejection stages: the initial
ancestry check and the final push. A final push failure triggers rebase only
when a fresh fetch proves that remote master advanced forward. An unchanged or
sideways remote, transport failure, or hygiene/lock/dashboard refusal remains a
failure. At most three guarded pushes run, without bypassing any guard.

After fixing a guard rejection and completing the normal guarded push, run
`just iqa-results-sync-recover CONFIG FULL_COMMIT`. It verifies that commit is
on remote master, checks the public projection, and requires every saved-data
file to match the committed bytes before updating private sync bookkeeping.
It neither pushes nor changes repository files. The cycle lock prevents a
concurrent capture from changing bookkeeping during recovery.

The coordinator runs on efficiency cores at low CPU/I/O priority, with a 2 GiB
service memory limit. This is result capture/publication, not a scoring process;
benchmark workers retain their own shared heavy locks and placement gates.

### Retired hosts and coverage

When a host leaves for good, give its config entry
`"retired": {"since": "<date>", "capture": "<dir>"}`, where `<dir>` is a copy of
its last complete capture (the `captures/<host>` folder of a cycle, with its
`capture-receipt.json`). The sync then never contacts that host: every cycle
links the frozen capture in, refuses one whose receipt names a different host,
and `sync.json` lists the host under `retired_hosts`. The mac was retired this
way on 2026-10-20; its last capture is from 22:20 UTC that day.

`coverage.json` gives each host a `cells` block: every in-scope planned cell
counted once across a parent campaign and its reruns, at its best state
(ok, refused by its input contract, failed, pending). The per-campaign counts
beside it overlap and must not be summed; the dashboard did until 2026-10-21,
which left a rerun parent's cells "pending" forever. The page calls the data
final once no host that is still running has a pending cell; a retired host's
pending cells read as "not run", and the "no new timings" warning stands down because a finished campaign produces no new results.

Each cycle keeps a full copy of the captures (about 190 MB). Only the newest
12 cycle directories, plus the one last published, are kept; on 2026-10-20 the
unpruned copies (about 55 GB a day) filled i265's disk.

### Scatter measurements

Each cycle also projects `*-rounds.csv`: one saved cold repeat or warm zenbench
round per row, with `time_ns`, exact metric/host/binary/input/thread identities,
phase, round index and qualification. A warm point is the measured per-call
mean within a round, not an invented timing for each iteration. The sync's
evidence fingerprint covers both attempt and round CSVs. No raw image-quality
scores enter either table. Missing individual round arrays stay missing.

An explicit monitored-memory stop in a hash-bound worker log is exported as
`resource_guard_stop=monitored_memory_guard` and a resource failure class. Exit
137 alone does not establish that classification. A killed macOS scope may
have no finalized BSD peak-RSS value; that value stays unavailable rather than
being replaced with the configured guard limit. These failed attempts remain
unqualified and retain their original retry lineage.

Private NN campaigns can enter the same timing-only projection after explicit
selection in the private sync config. Their campaign names start with
`private-nn-`; provenance exposes content hashes without model file names or
local paths. Saved controls and remaining-only production cells share exact
identity keys, so preferred selection counts each cell once. Timing controls
do not establish native numerical parity: DMM retains `failed_native_gate` in
the export even when its timing gate passes. Unrelated public NN jobs remain
stopped.

Mac capture scratch now uses a fresh directory with the `.noindex` suffix,
matching the production-output convention. The private configuration was backed
up before switching its remote scratch field; previous capture cycles remain
at their original paths. This redirects future captures only. It does not alter
frozen workers, results, attempt caps, resource gates or measurement clocks,
and does not establish a cause for any earlier gate flag.

A sealed `insufficient warm samples` rejection has the public failure class
`insufficient_warm_samples`. It remains unqualified and does not establish an
unsupported implementation or numerical failure. Missing timing arrays and
sample counts are not imputed from partial worker output; the raw failed
attempts and any partial diagnostic measurements remain preserved privately.

### Private NN qualification at 2026-10-20 08:00 UTC

Saved snapshot `11005b1deca4fea824a61a358e07e989dbe218c3` contains every
terminal private NN retry. Select campaign names beginning `private-nn-`,
`preferred=True` and `qualified_timing=True` in the host observation CSVs.
The seven-route canonical grid is 336 logical cells per host:

| Host | Timing-qualified | Exhausted gate-flagged | Other exhausted failures | Unmeasured, blocked |
| --- | ---: | ---: | ---: | ---: |
| mac | 268 | 3 | 65 | 0 |
| r7900x | 335 | 1 | 0 | 0 |
| i265 | 289 | 1 | 2 | 44 |

Mac's 65 failures comprise 64 TOPIQ large-input monitored-memory stops and one
DMM warm cell with insufficient samples. i265's two failures are AHIQ warm
controls: one times out and one produces insufficient samples; their exhausted
caps block admission of the 44 remaining AHIQ cells. These are measured campaign
limits, not unsupported implementations. No partial warm sample is qualified.
All twelve Mac production retries reproduce their parent score bits; nine
recover clean timing. Three remain gate-flagged after their final original
attempt. The protected retry service exited zero and is not running.

The table counts retained controls and qualified replacements once. It does
not combine engine or worker revisions, and it does not count failed attempts
as timings. DMM retains its failed native numerical gate; neither repeatable
scores nor timing qualification establish native-oracle parity or agreement
with an AIC published column. Raw attempts and exhausted budgets are preserved.
