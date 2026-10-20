# Resumable AIC score regeneration

Use the supplied AIC CSV and the actual AIC PNG files. The rescorer reads source/distorted filenames, binds file SHA256s into a portable plan, runs all pending selected CLI metrics for each pair through one `compare --output json` invocation, checkpoints successful scalar columns, and writes a separate calculated CSV. The published input CSV is never edited or used as our computed scores.

All new score exports retain full precision. JSON is the execution protocol; CSV uses 17 significant digits for binary64 round trips. The CLI's score, compare, folder and batch text exports and the `aic_vmaf_features` example now use round-trip formatting too. Existing six-decimal historical tables are preserved and are not promoted into this checkpoint format.

## Build on a new computer

Check out the desired source revision. Follow the repository's resource-wrapper/lock instructions for builds and long runs. The CI-pinned snapshot build avoids local sibling drift:

```sh
just snap-cargo @ build --release -p zenmetrics-cli \
  --bin zenmetrics --example aic_vmaf_features --features nn,pyiqa-ports
```

The default snapshot binaries are under `$HOME/tmp/zenmetrics-lock-snap/target/release/`; a caller's `LOCK_SNAP_DIR` overrides that root. Record the actual source revision/build receipt with `--source-commit`. The run also hashes the actual executable, optional feature executable, runner source, selected settings and model files. A binary with fewer features can still score its available routes; native refusals remain explicit, not completed cells. `list-metrics` is a name catalog, not proof that every route is compiled.

For learned metrics, provide exported model files with `--models DIR`, the intended `--engine`, and any explicit named weight acceptances with `--accept-weights`. The CLI's existing consent parser remains authoritative. This command does not download or export weights. IDFIQA/MSSWD are private dependencies and AIMOS is unavailable; those input columns remain explicitly blocked. DVIFM historical columns have no qualified public CLI mapping. A-FINE and site-specific variant routes calculate our identified implementation; this does not establish equality to published values.

## Prepare one immutable plan

```sh
just aic-rescore prepare --aic-csv /data/metrics_cropped.csv \
  --images /data/AIC-cropped --output /results/aic/plan.json
```

`--images` accepts either the PNG directory or a standard extracted AIC root with `PTC-crops-all/`, `sources/` and `distorted/` children. Every name must resolve uniquely; missing files, duplicate identities and ambiguous paths refuse preparation. No basename or score similarity is used to guess a pairing.

For the [July/drop combined CSV](../benchmarks/aic_csv_reconcile_2026-10-20/README.md), add `--view cropped` or `--view fullres`. Mixing both views in one plan is refused. Image hashes bind the selected view, including when copied to a new computer. `--images` on `run` relocates identical input files.

By default, preparation selects all input score columns. WS-HV is selected automatically when present in the input; for a July CSV, add `--extra-metric ws-hv`. `--columns HDR_VDP_2,HDR_VDP_3` creates a narrower plan if that is the intended entire campaign. JND display columns remain blocked unless an explicit `--jnd-config` supplies their fit; supported fits are `a*abs(sign*(x-b))**c`. Published JND cells are not copied as our results. Missing fits and unsupported transforms refuse rather than being guessed.

## Fill HDR-VDP first, then the rest

```sh
just aic-rescore run --plan /results/aic/plan.json \
  --images /data/AIC-cropped \
  --binary /build/zenmetrics \
  --vmaf-features /build/examples/aic_vmaf_features \
  --source-commit SOURCE_REVISION --threads 8 \
  --output /results/aic/scores --only HDR_VDP_2,HDR_VDP_3
```

Run the identical command again to resume. Remove `--only` to fill the other selected columns in the same campaign. The selector only schedules work and does not change result identity. HDR-VDP routes are ordered first within each pair. `HDR_VDP_2` uses `hdrvdp-aic2026` Q and `HDR_VDP_3` uses `hdrvdp3-aic2026` Q; native defaults or Q_JOD are never silently substituted.

All regular metrics for a pair share its decoded reference and distorted RGB8 bitmaps. ADM2, VIF_vmaf and VIF scale columns use the optional existing VMAF-feature executable, in a separate per-pair adapter call with its own decode. Its successful results are checkpointed independently. Providing that executable at the beginning avoids changing campaign identity later. The rescorer does not claim these two adapter calls share buffers.

Pairs run serially with the requested per-process thread environment. This local tool does not create a distributed sweep or replace zenfleet. `--limit-pairs N` gives a bounded control/resume check. `--timeout SECONDS` is optional; no time cutoff is imposed by default. The default is two attempts per unfinished scalar column across invocations; `--max-attempts` changes that explicit budget. Completed cells are never rescored. Changing a binary/model/engine/thread budget requires a new campaign directory, preserving old results.

## Checkpoints and outputs

- `cells/*.json`: authoritative per-pair scalar checkpoints, bound to the run identity, pair and payload digest. Successful metrics survive failures of other metrics. Cell writes use same-directory atomic replacement and fsync.
- `attempts/*/`: exact argv, child PID, stdout, stderr and exit information, including failed/interrupted attempts. A started attempt consumes its budget even if interrupted. SIGTERM/Ctrl-C kills and waits for its owned child group on Unix; a recorded unfinished live child blocks overlapping work. Windows uses its direct child process; Unix process-group cleanup is the exercised containment path.
- `.writer.lock`: OS-backed single-writer lock, released on process exit. A live writer cannot be displaced by a guessed stale timeout.
- `scores.csv`: input header/order plus explicitly requested extra columns; metadata copied, only our calculated scores filled. Numeric failures/unavailable/blocked cells stay empty.
- `status.csv`: one status/reason per selected image/column. `summary.json` distinguishes incomplete work from success. Exit 0 means every selected column is scored/derived and post-run hash checks pass; exit 2 means work remains, is blocked or failed. Integrity/usage failures exit 1; interruption exits 130.

CSV projections refresh every 100 attempted pairs and on a clean exit; `--export-every N` changes that interval. The per-pair checkpoints are durable immediately. Starting again rebuilds projections from checkpoints before new work. Unselected columns remain blank rather than borrowing published values. Successful cells are not imported from legacy rounded TSVs.

The run verifies image bytes before work and again at the end, and hashes model assets and executables before/after. A changed identity refuses reuse. A mid-run asset/input mutation marks the campaign invalid. `summary.complete` requires successful post-run verification; progress before that is not completion. Nonfinite/null/ambiguous outputs remain failed observations with retained raw logs. External ORT/CUDA runtime installation and device reproducibility are separate from the executable/model identities; retain those runtime receipts when using non-RTEN engines.

## Controls

```sh
just aic-rescore-check
just aic-precision-check /build/zenmetrics ref.png dist.png NEW_CONTROL_DIRECTORY
```

The controls include a real interrupted process, live-writer/orphan refusal, relocation with identical bytes, failed-column retry without repeating completed metrics, binary/model/input drift, corrupt checkpoints, explicit JND derivation, view selection and conflicting CSV union refusal. The native precision control compares exact JSON scores to score/compare/batch text exports. The [native 61-pair qualification](../benchmarks/aic_rescore_2026-10-20/README.md) records 610 finite HDR/WS-HV/PSNR/VMAF-feature scores, exact checkpoint-to-CSV identity and a no-op completed resume.
