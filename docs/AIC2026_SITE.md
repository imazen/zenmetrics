# AIC2026 results pages

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
`verify_live.py --url https://metrics.imazen.org/aic2026/ --source <commit>`
requires HTTP 200 and that source identity on every generated page.

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

The fixed-factor SSIM family reads colour, multiscale, rounding and factor
flags directly from the current CLI dispatch. Its `f32` suffix means
unrounded input: the native kernel actually accumulates in f64. Complete box
blocks are averaged from the top-left; incomplete bottom/right blocks are
dropped. Historical tally assumptions are labelled separately from current
configuration metadata.

VMAF's AIC column mappings select `vmaf-studio601` and
`vmaf-neg-studio601`. Default `vmaf`/`vmaf-neg` use BT.709. The identified
Butteraugli proposal uses the 3-norm output and an explicit piecewise map,
not the default maximum output. CAMBI and HDR native configuration gaps stay
attached to their nominal method mappings until their AIC routes qualify.
