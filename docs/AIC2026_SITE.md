# AIC2026 results pages

**Retired (2026-10-21).** https://metrics.imazen.org/ now hosts only the browser
workbench. The site build no longer merges `/aic2026/`; `web/site/_redirects`
sends those URLs to the matching tabs of the metric brief,
https://metric-brief.pages.dev/ (reproduction, model cards, cost), a separate
site. The human-score agreement, Pareto and statistics pages are not published:
their data is outdated. The deploy check now requires `/aic2026/` to answer 301.
The rest of this file is the historical record of how the pages were built.

The existing comparison viewer at https://metrics.imazen.org/ is hosted by
Cloudflare Pages (`.github/workflows/site.yml`, project `zenmetrics-site`).
The separate benchmark dashboard deploys to https://imazen.github.io/zenmetrics/
through `dashboard.yml`. These were verified from the workflows and GitHub
Pages API on 2026-10-18; resolve them again before changing deployment.
The AIC2026 extension is built by `scripts/dashboard/aic2026.py` in both
pipelines and merged into the viewer without replacing its workers or assets.

## Generation and checks

`python scripts/dashboard/build.py --output <output>` generates the dashboard
and its `aic2026/` section from committed data. `GITHUB_SHA` pins production
page footers. A local jj checkout uses its working commit; source-only archives
need `AIC_SOURCE_COMMIT` to identify a commit, otherwise they explicitly say
`unversioned-archive`. Such an archive is useful for the safe-push build check,
not a version-qualified deployment. No model is loaded and no network is
contacted during generation.

The existing `just dashboard-data-check`, `just dashboard-plot-check`,
`just dashboard-build` and `just dashboard-smoke` recipes exercise the shared
pipeline. Use a caller-owned disk-backed `TMPDIR` and the shared heavy-job
wrapper for builds/tests. The browser check visits every AIC page at desktop
light and phone dark widths, verifies source identity and rejects overflow or
third-party resource requests. Static origin checks run during build and smoke;
citation links permit optional outbound navigation, never resource loading.

`publish_extension.py` verifies identical viewer/AIC commit identities before
copying the extension and refreshing the viewer's file hashes.
Production runs serialize without cancelling an active build. Repeated
evidence/lint pushes on 2026-10-18 cancelled the cold WASM build before any
extension could deploy; finishing the active run lets its caches populate.
The newest queued follow-up then deploys, with its own strict source checks.
`verify_live.py --url https://metrics.imazen.org/aic2026/ --source <commit>`
requires HTTP 200 and that source identity on every generated page.
The HTTP verifier identifies itself as `zenmetrics-site-verification/1.0`.
On 2026-10-18 the edge returned 403 to Python's default user agent and 200
to this explicit agent and curl; source/JSON/content checks remain strict.

## Data and qualification

- The configuration inventory follows every CLI name, public pyiqa cards,
  counted-source identities, explicit ingress and the declared native gates.
  Matching an AIC convention does not qualify numerical parity: `dmm-pyiqa`
  still fails its 0.001 JND gate (worst 0.071 JND).
- Runtime tables retain source, shape, thread budget and repeat count. CPU task
  time and elapsed wall time remain separate. Missing one-thread PieAPP timing
  is **skipped by owner**; no new PieAPP timing is scheduled here.
- Screenshot coverage lists all 14 Pedzisz draft entries and all 31 names and
  visible median ranks in the authors' rank figure. Haar identification,
  released-build DVIFM and luma-only FSIM fixed-factor work remain explicitly
  pending aicfix until their committed evidence arrives.
- The Pareto page uses the committed preliminary public-split aggregate:
  5,768 images, 42 sources, mean within-source Spearman, 2,000 source-bootstrap
  draws (seed 20261017). It combines published-column agreement with counted
  costs of our mapped implementations, which can have disclosed pipeline gaps.
- Private IDFIQA/MSSWD publication is restricted to the explicit aggregate
  allowlist in `private-aggregates.json`. No code, graphs, checkpoints, storage
  URLs, preprocessing internals or per-image scores are exported.
- The worked CID22-A example uses all 2,192 admitted pairs from 25 sources in
  the frozen zensim human gauntlet. `scatter_example.py` exports its canonical
  scatter-v2 aggregates and a 64×64 density grid, with the original file hash.
  It does not fit a second diagnostic implementation or subsample points.
  Historical peer scoring provenance remains incomplete. The fitted panel is
  descriptive; JND cross-validation and observer trials are not measured here.

## Data terms checked 2026-10-18

The [DaRUS dataset API](https://darus.uni-stuttgart.de/api/datasets/:persistentId/?persistentId=doi:10.18419/DARUS-6156)
reported version 2.0, CC BY-SA 4.0. The
[dataset README](https://darus.uni-stuttgart.de/api/access/datafile/639751)
states the same licence, subject to source-specific metadata exceptions.
Credit: Mohsen Jenadeleh, Jon Sneyers, João Ascenso et al., *JPEG AIC2026: A
Large-Scale Dataset for Fine-Grained Assessment of Image Coding* (2026),
[dataset DOI](https://doi.org/10.18419/DARUS-6156) and
[paper](https://arxiv.org/abs/2607.22783).

This extension publishes AIC aggregates only. Current subjective scores,
uncertainties, data version and derived-chart terms are requested in the owner's
email draft. The draft is not sent by the site workflow.

## Mainland reachability

The owner confirmed mainland reachability on 2026-10-18 and directed that
China testing stop and hosting comparisons be skipped. The existing host
remains Cloudflare Pages; no DNS or hosting change is part of this extension.
The original HTTP observations remain in the private execution receipt.

The viewer's header links the explorer, AIC results, reproduction evidence,
benchmarks, pinned source and the AIC download section. Downloaded JSON
contains public configuration/card records and aggregate scatter diagnostics.
Its scoped entrance theme uses existing light/dark colours, system/serif
fonts and CSS geometry; the explorer's controls and assets remain intact.
The 390-pixel entrance check exposed an existing unwrapped view-button group
extending the document to 462 pixels. The merged site's phone rule wraps that
group; canvas panning stays inside its existing scroll container.

## Known bugs: configuration extraction (fixed 2026-10-18)

The catalog's original `scoring_compiled_in` parser consumed everything after
`let nr`, including the newer pyiqa group. The generated recipes therefore
mislabelled those FR ports as NR and omitted their required `pyiqa-ports`
feature. Parse each bounded `matches!` assignment instead. Model-card paths
now resolve to their actual package; CW-SSIM stays classical and has no model
files. Restricted weight recipes explicitly state the required per-column
consent without assuming consent. Regression controls require these contracts
for every pyiqa CLI name. Metric arithmetic and parity limits are unchanged.

### Exact ingress declarations

The settled fixed-factor definition replaces only FSIM/FSIMc/VSI
`kernel_size = max(1, round(min(H,W)/256))` with `kernel_size=k`. Pyiqa
SSIM-family `-kN` variants set `downsample=True` and `kernel_size=k`; other
steps are unchanged. Branch and implementation verification belongs to
portcheck2 and is separate from this docs/site update.

Recorded SSIM colour, multiscale, rounding and factor flags come from the
CLI dispatch; they do not qualify the settled definition. Its `f32` suffix means
unrounded input: the native kernel actually accumulates in f64. Complete box
blocks are averaged from the top-left; incomplete bottom/right blocks are
dropped. Historical tally assumptions are labelled separately from current
configuration metadata.

VMAF's AIC column mappings select `vmaf-studio601` and
`vmaf-neg-studio601`. Default `vmaf`/`vmaf-neg` use BT.709. The identified
Butteraugli proposal uses the 3-norm output and an explicit piecewise map,
not the default maximum output. The four explicit AIC routes landed in
`64391e26`. The 61-crop receipt qualifies HDR3's stated gate and CAMBI's median
gate. HDR2 retains both its published-column and independent RGB-oracle gaps;
native proposal-Butteraugli fails its stricter median gate. The separate
9,618-row published-column map does not establish 9,618 native scores.
Those explicit routes have separate names/configurations and no cost tally
yet. Do not transfer an old route's tally to them silently. Screenshot DVIFM,
luma-only FSIM and Haar work remains pending until committed evidence arrives.
The owner also requested the landing link to https://gigapixels.imazen.org/.

### Haar chart qualification

The completed aggregate method search identifies 0/11 Haar variants. Forty
methods across 91 twelve-source pools (792 five-source subsets each) remain
outside the chart's 0.5-rank reading uncertainty. Only 12/31 chart labels match
the available 64-column roster, so its ranks cannot establish exact chart
reproduction. No candidate fitting is qualified. Public `haar-search.json`
contains aggregates only; private inputs and implementation stay private.
The chart does not label its center marker's statistic: “median” is a
provisional transcription convention from the brief, not a verified label.

### Current qualification and benchmark sources (2026-10-18)

The first AIC generator incorrectly reused implementation/status labels from the
explicitly superseded October 15 coverage audit. Butteraugli-original consequently
retained the old libjxl mismatch despite its distinct original-reference port.
Current reproduction now uses the naming/qualification ledger and reproduction
status, the pyiqa native receipts, private aggregate cards, and explicit native
AIC route receipts. `columns.tsv` is only a name/availability inventory.
`retained-comparisons.tsv` retains numeric observations for unchanged qualified
panels; it contains no historical implementation-status labels. Newer native
receipts override those values. Every row identifies its comparison scope and
evidence; reference-versus-column and native-oracle errors remain distinct.

The i265 benchmark sessions confirmed the public chart snapshot at
`benchmarks/iqa_chart_snapshot_2026-10-18/`. It contains qualified timing attempts
from i265 and ARM64 Mac, including cold/warm and scalar/native-map modes. The AIC
page verifies snapshot hashes and admits only qualified timings; grouping retains
host, report lineage, binary/input identities, shape, workers and phase. Counts
of attempts or names are not counts of reproduced algorithms. A frozen executable's
uncompiled aliases do not mean current source lacks those implementations.

New `aic_refusals_2026-10-18/*-qualified-perf.tsv` panels are separate r5900xt
measurements. DVIFM core counts remain partial and its aggregate timings do not
identify the historical published columns. Fixed-luma FSIM's three tallies and
36 timing observations describe unmerged PR 92, not master CLI routes. These
measurements are shown separately from historical i265 weight-table fits.

Generic `butteraugli` benchmark rows report the headline max output of a shared
call. They are not independently measured pnorm3 timings. The screenshot coverage
label retains this distinction; the explicit proposal route has its own scoped
r5900xt supplement.

### AIMOS exclusion (owner clarification, 2026-10-19)

AIMOS is excluded (proprietary, unreleased). The proprietary Cloudinary metric
was never publicly described or released. It is absent from implementation,
reproduction and author-question queues. Public projections contain only this
exclusion status, with no private probe measurements.
