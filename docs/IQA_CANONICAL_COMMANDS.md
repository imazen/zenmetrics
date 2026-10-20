# Canonical IQA benchmark commands

`just bench-iqa` and `just bench-iqa-changed` freeze new scalar CPU campaigns.
They require explicit worker, build receipt, source revision, verified RGB pair
manifest, new bundle/output paths, resource wrapper and shared host lock:

```sh
just bench-iqa-changed --binary /absolute/iqa-compete \
  --pairs /absolute/pairs.json --build-receipt /absolute/build.json \
  --source-commit REVISION --bundle /absolute/new-bundle \
  --output /absolute/new-results --lock "$HOME/tmp/HOST-heavy.lock" \
  --run-heavy "$HOME/work/claudehints/scripts/run-heavy"
```

Build and qualify the native worker beforehand under the same shared lock.
On hosts requiring checked performance-core placement, also supply the qualified
`--zenbench` executable and explicit `--cpu-set`. Preparation verifies committed
RGB hashes and build receipt identity while holding admission. Each measurement
then acquires that same shared lock; preparation releases it before the durable
runner starts, preventing nested-lock deadlock.

The protocol is three fresh-process cold repeats, including construction and
first evaluation after RGB loading, plus ten clean warm zenbench rounds with a
100 ms target. Warm scoring evaluates the full pair repeatedly after initialization.
The six geometries are 512×512, 840×944, 1680×1888, 3355×2516, 2048×1080 and
3840×2160, with scalar scores at threads 1 and 8. The wrapper uses 12G/8 jobs;
its actual platform resource receipt, including peak memory, remains in worker
logs. Native score and resource gates are unchanged. A controller `ok` alone
does not establish that every warm round passed its gate.

Existing bundle/output paths are refused. Resume retained campaigns through
`just iqa-durable-run MANIFEST OUTPUT WRAPPER LOCK`; preserve their original
manifest, worker, attempt caps and lineage. `iqa-durable-prepare` now explicitly
selects scalar scores, six non-8K sizes, threads 1/8 and three cold repeats.
The low-level Python freezer retains explicit map/size options for historical
work; those options do not change already frozen campaigns.

`bench-iqa-supplemental` retains the historical portable 5/3/1 panel.
`bench-iqa-advanced-supplemental` retains historical Python/GPU/NN experiments.
Their samples must remain separate from canonical observations. Kernel benchmarks,
profilers and parity controls answer different questions and do not replace this
comparison protocol. NN campaigns require separately qualified adapters and run
last in the current three-host program.

Check command construction and admission with `just iqa-canonical-command-check`.

Historical `iqa-compete-*` panel executors also carry a `-supplemental` suffix.
Recipes named smoke, probe, control, tests or check are diagnostics; their
intentionally short budgets are not comparison measurements. Historical recipes
that explicitly request 8K remain supplemental and are not part of the active
non-8K campaign. Build recipes do not themselves qualify timing observations.

New durable runners also acquire shared admission for initial full asset hashing.
This prevents coordinator preflight CPU work from overlapping another admitted
measurement. Hashing remains outside cold and warm clocks; frozen older runners
retain their original identities.

After a campaign is terminal, `just iqa-canonical-gate-retry BUNDLE RESULTS
NEW_DIRECTORY LOCK` validates its retained attempt seals and freezes only warm
cells with resource flags or missing rounds and an original attempt remaining.
`retry-plan.json` records the exact maximum attempts for each successor and
lists exhausted unqualified cells. It does not install a service or grant new
attempts. Replacements retain the parent report hash; final selection counts
each logical cell once. Clean cells and input-contract refusals are not repeated.

A create-once reservation beside the parent report prevents preparing another
successor against the same remaining caps. Preserve that reservation if
preparation fails; inspect its recorded successor before taking further action.
Existing manually prepared successor lineages must be audited before using this
helper on their original parent.

Exports retain the executed worker engine when the worker reports it, including
weightless private CW-SSIM’s native Rust route. The NN engine environment remains
a fallback for older workers; absent engine evidence remains null.

The [captured input-contract audit](../benchmarks/iqa_input_contracts_2026-10-20/README.md)
distinguishes HDR luminance/viewing refusals and fixed-factor MS-SSIM window
refusals from unavailable implementations or worker stalls. Preserve these raw
attempts and limits; larger valid geometries remain in the frozen queue.

### Changed-only successors after a broad campaign

`prepare_changed_successor.py BUNDLE RESULTS NEW_DIRECTORY METRICS_JSON --lock LOCK`
requires the old coordinator to have stopped and acquires its coordinator lock.
It rebuilds the report from sealed attempts, retains completed changed cells,
excludes unchanged cells and partitions unfinished cells by remaining original
attempt budget. A create-once parent reservation prevents duplicate successors.
Run each frozen successor with exactly the recorded `max_attempts`; never rerun
completed cells to replace a campaign identifier. The approved changed list has
27 routes. Original results and source/version identities remain separate.

On 2026-10-20 the Mac/i265 operational scope was restricted to the approved
27 changed/added routes. Broad parent campaigns remain retained evidence, not
instructions to finish unchanged methods again. The i265 boundary snapshot
retained 896 completed changed cells and froze 400 unfinished cells: 48 with
one remaining attempt and 352 with two. Successful parent cells are never
selected into these successors. r7900x retains its full non-NN scope because
it has no earlier timing panel. Progress denominators must follow the active
scope (1296 changed cells versus 4608 full-inventory cells), not silently mark
excluded unchanged cells complete.

### Saved-result publication during an active campaign

`bench-iqa-capture-saved` reads atomic reports and finalized sealed attempts,
without adopting unfinished workers or stopping measurements. Capture windows
and file hashes identify what existed when each host was read. Raw reports
stay outside public git. `bench-iqa-publish-saved` emits only allowlisted timing,
resource and provenance fields, with explicit preferred-row identity and
without per-image scores. Different binaries remain separate series; portable
panels cannot substitute for canonical cells. The
[2026-10-20 saved snapshot](../benchmarks/iqa_saved_2026-10-20T0056Z/README.md)
retains partial observations, refusals and gates separately.

### Rust versus original implementations

`just bench-iqa-canonical PLAN NEW_RESULTS` runs the explicit original/reference
commands in a frozen comparison plan. Select a subset with `--metrics name,name`;
`--blocks 3 --samples 5` controls the alternating acquisition blocks. Every arm
retains its command, stdout, stderr, process memory receipt and numerical result.
The runner serializes measurements under the local host resource locks.

Create the plan with `just bench-iqa-canonical-plan` and explicit `--binary`,
`--pairs`, `--source`, `--python`, `--extras`, `--brisque-weights`, `--ssim1`,
`--ssim2`, `--output`, `--metrics`, `--sizes` and `--threads`. These paths name
installed reference assets; the command does not download or silently substitute
an implementation. Additional original implementations use `--registry FILE`,
which supplies their commands, numerical bounds, scope and artifact hashes.
Both RGB inputs and executable/reference assets are verified before and after
each block. Existing plan and result paths are refused.

Ratios mean **Rust runtime / canonical runtime**: below 1 is faster. They are
descriptive medians of alternating block ratios. A missing bound remains an
observed numerical difference, rather than an asserted parity pass. A mismatched
input or timed boundary withholds its ratio. This comparative protocol is
separate from the frozen cold3/warm10 CPU campaign above.

`just bench-iqa-canonical-publish NEW_DIRECTORY RESULTS...` validates retained
blocks and emits compact source-bound tables for the README generator.
`just bench-iqa-canonical-check` checks parsing and result projection. Builder
recipes for original DSSIM, SSIMULACRA1, Butteraugli and libvmaf accept explicit
source paths; run builds under the same resource wrapper and host locks.

For an explicitly stopped broad parent, `prepare_gate_retry.py --stopped-parent
--metrics EXACT_NAMES_JSON` can select only named, unqualified warm cells with
remaining original attempts. The exclusive coordinator lock still proves the
old controller is absent; every prior attempt/config/log seal is verified.
The default still requires a terminal campaign. Completed clean cells are never
selected, and create-once reservations prevent reusing the retry cap.

The current comparative entry point checks every canonical warm score against
the plan's existing bound. Small nondeterministic reductions remain recorded;
they are not rejected solely for differing between warm calls. Retained older
plans keep their original runner. Optional plan fields `arm_timeout_seconds`
and `rust_chunk_budget_ms` extend acquisition budgets for slow neural models;
the native worker's 850000 ms cap and ten warm rounds remain unchanged.

`just bench-iqa-canonical-resume RESULTS --blocks N --samples N` finishes
missing blocks after the original coordinator has stopped. It preserves both
completed and interrupted attempts. `just bench-iqa-canonical-interval RESULTS
NEW_JSON` bootstraps complete alternating-block ratios, without pooling inner
samples, and records every native resource flag. An interval is descriptive;
flagged observations never become an accepted regression through resampling.

Original neural comparisons can use an already frozen graph bundle.
`bench-iqa-canonical-model-verify` checks explicitly selected copies against its
existing hash manifest. `bench-iqa-canonical-original-nn-plan` freezes original
LPIPS, ST-LPIPS, DISTS and DeepDC callables, graph identities, source, reference
weights and the existing JND/1000 bounds. Build and prepare under shared
admission; run neural measurements after the classical comparisons.

For a comparative run restricted to the 27 changed routes since `f7668c7`, use:

```sh
just bench-iqa-canonical-changed /absolute/reference-manifest.json /absolute/new-results
```

The manifest supplies `binary`, `rust_source` and `metric_plans` (one explicit
reference-plan path for each changed name). The command verifies those bindings,
freezes `new-results.plan.json`, then compares threads 1 and 8 in ascending
retained historical duration. Each case keeps its valid geometry, input hashes,
reference convention and existing numerical bound. This order comes from the
committed changed-route timing list; it is not a prediction for a different host.
Both output paths must be new. Use `--metrics NAME,...` to restrict execution of
the frozen plan, or pass `--blocks N --samples N` for acquisition counts.

`bench-iqa-canonical-author-plan MANIFEST NEW_DIRECTORY` freezes separately
specified original author-network callers. The manifest names the reference
Python, helper, configuration, source/weight assets and already verified native
graph directory for each metric. The existing JND/1000 column bound is read from
`jnd_scale.tsv`. It retains the requested inference engine and explicit weight
consent. Constructor controls do not establish scoring parity.

`bench-iqa-canonical-queue MANIFEST` executes explicit `{plan, output, options}`
entries sequentially. Optional `{results, blocks}` prerequisites require their
complete frozen case list; `--wait` keeps the visible controller waiting for
those cases. Each acquired block still obtains both host locks and its memory
cap. Process failures remain in their cohort and do not discard later queued
checks. Existing output directories are refused.

`bench-iqa-canonical-interval RESULTS NEW_JSON --additional SUCCESSOR_RESULTS`
combines a completed fixed follow-up only when its recorded parent plan, source,
worker and case contract agree. All blocks remain in the primary estimate. A
separate diagnostic counts whole blocks with at least ten clean native resource
observations; this subset does not turn the result into an accepted regression.

Both PieAPP routes are excluded from the owner-directed local canonical recheck.
AIMOS is excluded: proprietary, with no released implementation identified.

`bench-iqa-canonical-warm-profile WORKER CONFIG NEW_DIRECTORY` uses Linux perf
control FIFOs to enable sampling after the worker's existing completed-warmup
record. Model construction and initialization are excluded; benchmark calibration
and measured rounds remain included. It adds no metric timers or counters.
Run under the same host locks, thread budget and resource wrapper as the compared
case. The command retains perf's report, raw recording hashes, full logs and
configuration identity. Its default sampling frequency is 199 Hz.
