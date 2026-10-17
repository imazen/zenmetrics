# Benchmark dashboard

The [GitHub Pages dashboard](https://imazen.github.io/zenmetrics/) is a static
observation browser. The Cloudflare scoring viewer and its workflow are separate.
The dashboard deploys after reviewed changes reach `master`.

## Add observations

1. Commit the measured table and its metadata under `benchmarks/`. Keep generated
   source files within repository size limits; large external pointer targets are
   inventoried but are not fetched automatically.
2. Add a `[[dataset]]` entry in
   [the registry](../benchmarks/dashboard/datasets.toml): `glob`, `family`,
   `adapter`, and the exact permitted `columns`. An empty glob fails unless
   `allow_absent = true` explicitly declares an incoming dataset. Never use that
   flag to conceal a missing required input.
3. Run `just dashboard-data-check` and `just dashboard-build`. Install the pinned
   browser dependency with `python -m pip install -r scripts/dashboard/requirements.txt`
   and `python -m playwright install chromium`, then `just dashboard-smoke`.
4. Push the reviewed change to `master`. The dashboard workflow builds and checks
   the artifact before deploying it. Branch dispatch builds do not deploy.

Unknown columns, malformed rows, invalid numbers, missing required registered
inputs, and unknown adapters fail the build. New schemas require an adapter and
rejection controls. Competitor progress schemas are registered in anticipation of
that lane landing; no other lane's unreviewed data is copied into this branch.

## Stack and publication boundary

Python 3.13 standard library adapters emit compact per-family JSON. HTML, CSS and
native ES modules render SVG and tables without a package install, CDN or third
party browser runtime. Faceted filtering and grouping suffice for the current
size; each family loads only when opened. This keeps builds reproducible and
avoids loading the entire benchmark archive into a browser. No parquet is loaded
because the selected source families are small TSV/JSON inputs; add a pinned
columnar adapter if a future registered family requires it.

The build copies only dashboard assets and normalized public fields. Raw metadata
and image paths are not copied. Source hashes and links bind observations to
tracked files. Missing commit, host, date or command fields stay `missing` rather
than being inferred from an unrelated hash. In particular an oracle commit in a
README is not a build commit. Metadata links preserve access to original evidence.

Every selection lists its source, SHA-256, available build identity and metadata
links. Competition rows retain host, OS, implementation, engine, device, threads,
mode, status, comparison status and reproduction requirement. Unsupported rows
remain in the table with missing timings. They do not become zero-cost points.
Repeated runs and distinct sources are never averaged together.

## Reading the views

- Speed plots individual timing observations in ms/MP against pixels, grouped by a
  selected dimension. Hover reveals each observation's conditions. The second
  plot exposes thread count. Use host/OS/backend/mode filters for comparisons.
  Missing crop sizes, 2K, 4K, 8K or Windows/Linux pairs stay absent until measured.
  The encoder-loop data uses planar linear input and is labelled separately from
  competitor score and score+map cells.
- The cost model reports **α** from an OLS fit on measured sizes ≤512², and **β**
  from a separate OLS fit on sizes ≥1024². Each needs at least two distinct sizes.
  Fits are grouped by source and every workload dimension, and are descriptive;
  the interface never plots fabricated or extrapolated samples.
- Agreement displays log absolute and relative delta, reports exact zeros
  separately and preserves missing size and JND fields. The initial AIC panels
  are historical observations, not claims of current parity qualification.
- Quality displays the reported SROCC/PLCC/KROCC and sample count, retaining
  dataset, scope, subset, arm and normalization. Heatmap cells with several
  differing observations say `multiple scopes`; filter before ranking. The
  initial panels cover historical Margarine experiments on AIC-4 samples, CID22,
  KADID and CSIQ. They do not compare all metrics. No confidence intervals or
  per-image MOS rows are available in the selected inputs; the scatter explicitly
  reports missing data. Broader human-quality tables need registration.
- Catalog extracts the current CLI names (113 at the initial snapshot; pending
  AHIQ and DMM are not invented as available). Reference evidence joins exact
  ledger entries; missing entries and license fields stay explicit. CPU/GPU
  labels reflect CLI routing, not an exhaustive NN runtime support matrix.

URL hashes retain the selected view and filters. Tables sort by any displayed
column, CSV downloads contain the entire filtered selection, and theme choice
persists locally. Only the first 2,000 rows render in the table; charts and CSV
use the full selection. Missing inventory is visible in the coverage panel.

## Gates

The workflow runs strict adapter controls, builds from the checked-out sources,
and serves that exact artifact for headless browser tests. Tests load all four
views, require populated tables and numeric charts where applicable, reject
console/HTTP errors, and exercise filters, sorting, CSV download, URL restore,
theme persistence, an empty selection and phone width. Screenshots can be saved
with `python scripts/dashboard/check.py target/dashboard --screenshots <directory>`.
No Cargo or sibling checkout is needed.
