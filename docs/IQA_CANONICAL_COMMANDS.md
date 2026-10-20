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
