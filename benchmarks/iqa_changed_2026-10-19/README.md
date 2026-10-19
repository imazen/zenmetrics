# Changed CPU metric remeasurement, 2026-10-19

Source: `519a6ecb63cfa596904f240ae083c1685b0e96ab`, based on the
27-name changed-since-f7668c7 selection in `fa8194c917c9`.
The scalar/thread coordinator options use the existing worker's `only_mode=score`;
metric arithmetic and numerical/resource gates are unchanged.

Mac and i265 use the manifest-verified S67 AIC PNG pair, six geometries
(cid22, aic-small, aic-medium, aic-large, 2k, 4k), and one/eight workers.
No maps or 8K measurements are scheduled. i265 is restricted to CPUs 0–7,
its eight P-cores. The portable recipe measures warm scoring with five samples
below AIC-large and three at AIC-large, 2K and 4K. These are separate observations
from the earlier constructor-inclusive cold / ten-round durable campaign;
do not relabel them as that protocol or as native-oracle parity.

The new snapshots and binaries are isolated from existing production bundles.
Raw roots on each host are `~/tmp/iqa-changed-20261019T1620Z/`, including
`backup/`, `snapshot/`, `control/`, `results/` and coordinator logs.
Backups include source bundles and exact pre-run report copies (28 on i265,
60 on mac). Prior frozen inputs, executables, results and attempt caps remain
unchanged. No repository tags are created.

The i265 scalar control passes four rows (PSNR and GMSD at one/eight workers,
five samples each), with no maps. Its run-heavy receipt records rc0, 4 seconds,
0.13 GiB peak RSS. Source build: rc0, 97 seconds, 0.87 GiB peak RSS under
12 GiB / eight jobs. Mac controls pass the same four scalar rows; full-panel completion is pending;
launching a service is not completion.

NN CPU admission was stopped by user direction. The i265 boundary receipt
records 2026-10-19T16:35:49.307472Z with no owned watchdogs and the coordinator
stopped. Mac's coordinator was disabled and removed at 2026-10-19T16:53:31.881301Z,
with no owned watchdogs; its idle durable controller exited. Its last running
scope finished with verified cleanup before the regular build acquired the lock. Exact
service stop receipts remain under `nn-stop/boundary.json` in each raw root.

## r7900x full CPU inventory

The owner additionally requests all non-NN CPU routes on r7900x, which has no
baseline timings. Its first phase uses the 27 changed names, then a separate
successor selects the remaining compiled CPU names and the six pinned private
dvifmish routes, excluding the already measured 27. The existing checkout's
uncommitted NIQE/PIQE work is untouched; a separate source clone and pinned
snapshot are used. Both phases retain scalar-only, one/eight-worker and
non-8K selection. Full inventory and selected names are saved with the results.

`just iqa-portable-collect NEW_OUTPUT ROOT...` exports the scalar rows with
exact config/log hashes and resource summaries. Peak RSS retains the wrapper's
0.01 GiB reporting precision; it is not a metric-only heap measurement.
`qualified_timing` requires successful rows and all required zenbench samples
clean under this portable protocol. Failed gates remain unqualified. The
collector refuses identity mismatches, unbound logs and existing export outputs.

## Mac input preparation refusal

The source build passes under the Mac 12 GiB / eight-job guard: rc0,
743 seconds, 0.99 GiB peak RSS. A Bash 3 empty-array launcher error preceded
all worker outputs and was preserved separately. The corrected launcher passes
the four scalar controls, then refuses newly prepared AIC derivatives whose
hashes differ at cid22, aic-small, aic-medium and 2k. The eight differing files
and all original bytes are preserved under `results/`; no timing row was produced.
No cause is claimed from this observation.

A fresh `results-canonical/` output copies and verifies every RGB file from the
original frozen Mac bundle before running. `canonical-input-copy.json` binds
source/copy hashes and rejected generated hashes. Existing bundles and the
manifest checksum gate are unchanged. Exact BSD-time maximum RSS bytes are
retained where the per-worker log provides them; rounded wrapper RSS remains
explicitly labeled on all hosts.

## Collected i265 panel

[i265/timings.csv](i265/timings.csv) is the flat chart table;
[i265/observations.jsonl](i265/observations.jsonl) retains complete rows and
bound resource evidence. [i265/snapshot.json](i265/snapshot.json) binds hashes.
The complete matrix has 324 rows: 27 names × six geometries × two worker budgets.
312 timing rows pass every required sample gate. Twelve MS-SSIM size-contract
refusals remain explicit, with no timing qualification. No maps or 8K rows occur.
The production scope completes rc0 in 1326 seconds, 8.94 GiB peak RSS,
minimum available memory 18352 MiB, peak load 4.21. The Linux recipe gives workers separate 16 GiB systemd scopes;
the outer launcher uses 12 GiB and is not their ancestor scope. The build alone
uses its 12 GiB scope. [Actual worker policies](i265/resource-policy.json) retain
the resource header counts and maximum-RSS row. Timing qualification is unchanged.

## Collected r7900x initial panel

[r7900x-initial27/timings.csv](r7900x-initial27/timings.csv) contains the first
27-name panel, not the full inventory. All 324 requested scopes finish:
312 successful timing rows, of which 301 are clean; eleven retain heavy-process
flags and remain unqualified. Twelve deterministic MS-SSIM size refusals are
retained separately. The initial clone/build/control/panel job takes 1670 seconds, peak RSS 9.47 GiB, minimum available
memory 14300 MiB, peak load 4.62. As on i265, native recipe workers use 16 GiB
scopes separately from the 12 GiB outer launcher.

The first remainder setup lacked outside-workspace sibling links and failed
before measurement. That snapshot and log are preserved. A fresh snapshot
includes those links, with an independent target directory so the initial
worker binary and its source snapshot are preserved. Full CPU remainder
measurement is pending; this initial table does not establish that coverage.

## Collected Mac panel and bounded retries

[mac/timings.csv](mac/timings.csv) and [mac/observations.jsonl](mac/observations.jsonl)
contain 327 observed rows covering 324 distinct requested scopes. There are
312 distinct clean timings and twelve MS-SSIM size refusals. Three initial
CPU-load-flagged rows remain unqualified. Each received one additional scope
with the identical config, same worker/input hashes, sample counts and gates,
after five seconds outside the clocks. All three retries are clean and their
score bits exactly match their parent observations. No successful clean cells
were repeated. The primary panel completes rc0 in 1517 seconds, peak RSS
10.49 GiB, minimum available-page proxy 10953 MiB, peak load 6.38. The Mac
12 GiB guard monitors process-tree RSS; it is not a cgroup hard limit.

The retry manifest records a one-attempt cap per flagged cell. A repeat
invocation refuses the existing output, with all prior artifact hashes
unchanged. Proof is retained in [retry-no-overwrite.json](mac/retry-no-overwrite.json).
The same read-only scalar collector binds native BSD-time RSS bytes and rounded
wrapper RSS to every row. Under the shared lock, `just iqa-portable-retry-flagged`
can execute this bounded continuation; it never modifies the parent attempts.
