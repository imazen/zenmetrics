# Benchmark dashboard

**Retired from publication (2026-10-21).** https://imazen.github.io/zenmetrics/ now
serves only a redirect (`web/retired-dashboard/`) to the metric brief,
https://metric-brief.pages.dev/ (`scripts/complexity/brief.sh`), which publishes speed,
reproduction status and model cards. Its agreement and quality panels are not
published: that data is outdated. `dashboard.yml` still builds and checks this
dashboard on every relevant push, because it guards benchmark registration; the
built pages are no longer deployed. The rest of this file describes that build.

## Coverage and limits

The catalog follows the current CLI source; pending metric ports appear when
those changes land. Timing includes historical CPU, GPU, backend and NN panels.
Agreement includes both reference-oracle and published-score residuals. Coverage
is an inventory of observations, not a claim that every metric, platform or size
has been qualified.

Full-resolution AIC2026 scored sources are unavailable on the dashboard build
host. Reachable historical scored snapshots support CID22, KADID, TID and AIC
sample panels; an AIC sample is never labelled the full AIC2026 corpus. Original
metric-scoring commits are not recoverable from those parquets. Their hashes,
recorded columns and bootstrap method qualify the historical observations only.

[Dataset citations](../README.md#datasets) apply to the statistics below.

The quality view also includes all Margarine tables with SROCC/PLCC/KROCC,
retaining experiment, normalization, arm, scope and subset. These are distinct
conditions. Heatmap cells with conflicting values say `multiple scopes`; filter
before ranking. Per-image scatter points retain the recorded target field,
including JND or DMOS where applicable; the axis title does not relabel them MOS.

## Add or classify an upload

1. Add measured files and metadata under `benchmarks/`. Register an exact schema
   in [datasets.toml](../benchmarks/dashboard/datasets.toml), with family, adapter
   and glob. `allow_absent` declares an expected incoming family; its match count
   appears in Coverage, including zero. Versioned competitor progress directories
   match explicit `-progress-v*` globs.
2. Every otherwise unregistered data/metadata/pointer file must have an explicit
   `ignored` entry and reason in
   [classification.tsv](../benchmarks/dashboard/classification.tsv). This is a
   reviewed configuration, not a blanket glob. New unclassified files fail CI.
   The site publishes the excluded count only, not excluded filenames.
3. Run `just dashboard-data-check`, `just dashboard-plot-check`,
   `just dashboard-generated-check` and `just dashboard-build`. Unknown columns,
   malformed numeric observations or missing required files fail the build.
4. Install [browser dependencies](../scripts/dashboard/requirements.txt), then
   run `just dashboard-smoke` and `just dashboard-clean-check`. The smoke port is
   configurable with `DASHBOARD_PORT` or `--port`, restricted to 3000–3999.
5. Land reviewed changes to `master`. The workflow builds, tests and deploys the
   artifact. Concurrency is per ref and never cancels an in-progress deployment.

Public per-image TSVs may use the `quality-points` adapter with exactly `metric`,
`dataset`, `pair`, `mos`, `score`. Do not infer correlations or intervals from a
small scatter preview.

The native-reference comparison publisher and interval command add explicit
classification rows for the files they successfully generate inside this
repository's `benchmarks/` tree. Those comparison cohorts, summaries, source
indexes and descriptive intervals remain separate from the canonical multi-host
timing series. Scratch projections do not modify the registry. Other files in
the output directory are never classified by these commands; unrelated uploads
still fail the unknown-file guard.

## Statistical and display contracts

The first view ranks all positive measured timings at one exact width × height,
fastest first, on a logarithmic millisecond axis (or ms/MP). The default chooses
the single source and exact workload cell with most distinct zenmetrics metrics;
newest source date breaks ties. Host, device, reported thread count, mode, OS,
backend, phase, engine, content and precision flags remain separate. Historical
workload names stay visible; `score` is labelled `score-only`. Unknown threads
are explicitly `missing`, never assumed single-threaded or all-core.

Source numbers accompany bars; selecting multiple sources shows a warning.
No source averages or size interpolation are used. Competitors have diamond
marks and dashed outlines. Whiskers show reported p10–p90, when available.
The libjxl v0.12 panel retains all supplied samples, including first calls;
median and linearly interpolated percentiles are computed from those samples.
It is a timing panel, not a parity qualification.
The paired Rust 2K/4K extract shares the supplied libjxl input pairs and a
registry-declared RGB8 / 80-nit native-CPU comparison group. This group binds
those two named sources; it does not merge other historical workloads. Host,
device, threads, mode and exact size still filter separately. The source warning
remains: Rust excludes initialization/warmup and reports ten zenbench round
means, whereas libjxl includes first-call samples. All reported rounds remain,
including gate-clean false observations (one of ten at 2K, two of ten at 4K).
Input and binary hashes identify the Rust measurements; their repository commit
was not recorded. No current commit is substituted for it.


All-size columns retain identical workload filters; changing layout clears the
single-file source restriction and displays the mixed-source warning. The two-size slope chart ranks the
fastest selected observation per metric / implementation, and omits missing
endpoints. URL state records every selector; CSV exports the displayed rows.
Missing metrics include catalog entries with no timing. Generate the separate
coverage inventory with `just dashboard-speed-coverage`: counts are unions over
modes and implementations, useful for finding absent cells, not speed rankings.


Timing dots retain source, host, OS, implementation, backend, engine, threads,
mode and phase. Pixel axes are logarithmic. Every group labels its own y scale;
color identifies thread count and square marks identify warm calls. Competition
adapters retain sample count, p10/p90, peak RSS, CPU/GPU/SIMD identity and parity
status. Status is an enum plus details; failed or unsupported observations remain
rows with missing timings. Historical NaN/refusal tokens are explicit missing
measurements, not zero-cost points. Two 2026-10-12 timing files contain an
extra leading metric label on one row: their explicit adapter removes it only
when it repeats the row metric. Other malformed rows fail. The rounded analysis
file has a separate mixed-table/diagnostic adapter; raw inputs remain unchanged.

Cost fits group by metric, implementation, host, OS, backend, engine, threads,
mode, phase and content across sources. Repeated observations at each measured
size reduce to one median before OLS. α is the intercept from sizes ≤512²; β is
the slope from sizes ≥1024². Both ranges show distinct-size n and R², require at
least two distinct sizes and flag negative α. These describe the selected
historical observations; no interpolated or extrapolated points are drawn.
Thread speedup uses the median one-thread baseline within the same metric,
size and workload, divided by that thread count's median. Different sizes are
never pooled into a scaling ratio.

Agreement uses real pair IDs on the x axis, color by metric and distinct
oracle/published series. Exact zeros occupy a separate labelled lane, including
all-zero selections. Missing observations are counted separately. A small error
against a seven-digit printed oracle does not establish algorithmic agreement.
CAMBI, SSIMULACRA1 and WS-HV published reproduction remains visibly unconfirmed.
The `aic_review` SSIMULACRA1 panel is canonical here because it is the later
Round 5 reviewed scoring run; the differing `aic_more` Round 4 file is retained
and explicitly classified as historical. Neither establishes published parity.

Quality regeneration uses raw Spearman, Pearson and Kendall tau-b without fitting
an output remap. Magnitudes appear in the leaderboard and signed coefficients
remain available. Fixed-seed reference-image cluster resampling yields percentile
95% intervals; this is not observer-level uncertainty. Each metric has explicit
finite-row/exclusion counts. A deterministic preview contains at most 24 actual
rows per metric; its source identity is hashed and no local image paths are
published. Install `requirements-quality.txt` and run `just dashboard-correlations`
with the hash-matched read-only source to regenerate the aggregate shards.

## Catalog, extracts and provenance

`catalog.py` generates committed TSV shards from CLI names, backend/GPU/display/
luma flags, output columns, NN/NR membership, model cards joined by their output
columns, crate licenses and JND scales. Static fallback column names are labelled
as such because enabled features can version them; the exact Rust output-column
expression is retained alongside those aliases. Model directions and weight
terms stay separate from code licenses. Parity cells distinguish a model card’s declared gate, a ledger’s reported
gate and observed error, and no evidence; they link to the actual evidence.
Naming a reference alone does not establish parity.
Reference prose renders as text with Markdown decoration removed. Ditto entries
resolve to the preceding ledger row.

`extract_markdown.py` produces machine-readable, source-linked extracts of the
provenance tables, DIVERGENCES ledger, AIC fitted-JND and residual tables, coverage
audit and NN AIC crosscheck. CI checks both sets of committed extracts against
current inputs. Shards stay below 30 KB. The build scans its output with the same
shared hygiene patterns as the repository check, including private local patterns
when installed. Only canonical CPU-derived host labels are accepted.

Source SHA-256, available commit/date/host/command and metadata links accompany
every selection. Missing provenance fields remain missing. Oracle hashes are not
substituted for build commits. Raw metadata, paths and excluded inventories are
not copied into the published artifact.

## Implementation and verification

Python 3.13 standard-library adapters emit per-family JSON. Native browser modules
render SVG, filters and tables; no CDN or browser package dependency is required.
Families load on demand. URL hashes preserve view and filters; CSV contains the
full sorted selection, while tables render up to 2,000 rows. Theme persists.

Rejection controls cover unknown schemas, nonfinite numbers, new unclassified
uploads, shared hygiene, card-column joins, hex residuals and distinct published
comparisons. Numeric plot controls check across-source median fits and per-size
speedup. Browser checks load every view, require data and charts, reject console
and HTTP errors, and verify source counts against the manifest, CSV, URL restore,
empty selections, theme persistence and phone width. The clean check builds and
serves a tracked-source archive independently of ignored local files.

## Deployment regeneration

The Pages workflow regenerates catalog shards and markdown extracts before
building the site. Its build log and GitHub step summary report whether the
committed copies differed, with added/changed/removed file counts. Staleness
alone does not fail deployment. Rendering/schema errors and unclassified
benchmark uploads still fail; the build and publication hygiene checks remain.
Generated shard registries accept changing shard counts within their existing
schemas. Regeneration removes obsolete owned shards and preserves unrelated
files for the inventory gate to inspect.

Use `just dashboard-regenerate` to follow the deploy preparation locally.
`just dashboard-generated-check` and each generator's `--check` remain strict
local currentness checks. The clean-snapshot smoke follows deploy regeneration.

## Chart implementation and data updates

The browser renders reusable SVG plotting functions in `plots.js` and
`ranking.js`, without a third-party chart library. These are descriptive views
of registered observations; source conditions and sampling caveats limit their
comparability. A chart does not qualify a metric or a backend by itself.

Pushes to master touching registered benchmark locations, metric/catalog inputs,
source documents or dashboard code trigger assembly and deployment. Newly added
benchmark files require an explicit schema registration or an explicit reviewed
classification; unknown files fail assembly. Existing registered files are read
afresh on every build. No benchmark is rerun automatically by this workflow.

The ORT WebGPU receipt includes RTEN and CUDA timing controls and an exported-
golden panel. Explicit backend labels distinguish GPU WebGPU from CPU RTEN;
reported default thread counts stay unknown. LOADFAIL and OOM outcomes remain
failed rows with missing numeric results, separate from successful goldens.

## Priority list, stars and reproduction commands

`scripts/dashboard/favorites.json` is the owner's priority list (2026-10-17),
ordered by first implementation in zenmetrics: the earliest of the crate's
`Cargo.toml` being added, the `MetricKind` variant appearing, or the CLI name
appearing on master. It needs git history, so it is committed; regenerate it
with `python3 scripts/dashboard/favorites.py` and verify with `--check`, which
recomputes it at its recorded revision. The Models view filters and orders by
it. Stars are per-viewer and live only in that browser's storage. Model
records carry CLI reproduction commands derived by `scripts/dashboard/repro.py`
from the CLI's feature gates. `benchmarks/dashboard/unimplemented.toml` lists
planned or declined methods with the source line that records each status.
