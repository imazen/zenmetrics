# Explicit AIC2026 scoring conventions

The new names preserve the existing metric routes. These are measurements on
r5900xt, not i265. The frozen crop panel contains 61 pairs.

| route | median site-JND | maximum site-JND | Spearman | qualification |
|---|---:|---:|---:|---|
| hdrvdp3-aic2026 | 0.00133569 | 0.0128073 | 1 | passes the stated crop gate |
| hdrvdp-aic2026 | 0.00365059 | 0.0514863 | 0.999841 | reproduces configuration-search residual; not exact column parity |
| cambi-aic2026 | 0.0000000817689 | 0.00212503 | 1 | passes stated median gate |
| proposal-butteraugli | 0.000000306732 | 0.0000504274 | 1 | median gate fails; 61 native pairs only |

Full-precision native/published comparisons are in the four named TSVs.
CAMBI's nonlinear site mapping amplifies rounding near zero; its maximum raw
difference is 4.80485e-7. No maximum CAMBI gate was specified.
The proposal's 9,618-row published-column mapping is distinct from independently
running the native scorer on all those images; this panel establishes only 61.

HDR-VDP 3 uses 200-nit sRGB, 10-bit code quantization before the EOTF,
D65 XYZ, OLED emission, quality task, age 24, no surround and 37.842505866 ppd.
It exports both Q and Q_JOD. HDR-VDP 2 uses unrounded codes, BT.709 and 37.84 ppd.
The CLI options `--hdrvdp-sdr-peak`, `--hdrvdp-sdr-ppd` and
`--hdrvdp-sdr-code-bits` override only these explicit routes.

CAMBI uses the integer swscale-compatible luma expression before the existing
8-bit CAMBI kernel. Proposal-Butteraugli applies the documented two-line maximum
to the existing native pnorm3. Existing CLI names keep their numerical paths and
six-decimal batch serialization; the four new names serialize full f64 precision.

Commands: `zenmetrics batch --metric NAME --pairs PAIRS --output OUT --jobs 1`
with RAYON_NUM_THREADS=8, each name above. Source input/configuration provenance
is the [configuration search](../aic_confsearch_2026-10-17/README.md).
Resource receipt: run-heavy done rc=0 68s | peak-RSS 1.16GiB |
min-avail 50968MiB | peak-load 4.99. This is whole-panel process resource use,
not per-call timing or a comparison with i265.

The independent Octave runner accepts an external HDR-VDP-2.2.2 installation,
pairs TSV and output path; no reference implementation is bundled.

## HDR-VDP 2 independent reference

The Octave 2.2.2 oracle completed all 61 RGB crops. Against the published
column, its median raw residual is 0.02651 Q and median site-JND 0.003953;
the residual remains in the independent reference. Native versus Octave has
median 0.0086901909133843 Q and maximum 0.014212375528167343 Q.
Thus the older luminance-only golden bound does not cover this RGB panel.
The two differences must be investigated separately; neither establishes that
changing one display setting will eliminate the published-column gap.

Oracle receipt: run-heavy done rc=0 264s | peak-RSS 0.51GiB |
min-avail 51480MiB | peak-load 3.13. The external reference tree differs from
the official file in diagnostic result fields only; the original score is
unchanged. The pure-MATLAB pyramid fallback ran under Octave, without MEX.

## Screenshot inventory supplements

The four DVIFM draft families map to private port-lap-v0.3 (RGB),
port-lap-v0.2 (Y), port-2d-v0.2 and port-min-v0.2. The benchmark adapter is
private commit e27b88ab6181. The public `dvifm-counts.tsv` contains only
aggregate operation counts at 840x944 and 1024x1024. It models core logical
padded-block work; strip overlap, lane padding, allocation/setup, traffic and
span remain unqualified. It is not a complete machine-instruction tally.
Draft and contributed-release constants are not asserted identical.

The private Haar search completed all 40 method combinations and 91 pools,
with 792 five-source subsets per pool. `haar-method-aggregates.tsv` omits
source identities and all per-image subjective values. Best available-roster
mean rank error is 9.666666666666666, maximum 43, outside the 0.5-rank chart
reading uncertainty. Only 12 chart labels match the available competitor
roster. Zero of eleven unknown Haar variants is identified; the method gate
fails, and no candidate fitting was attempted.

Pointwise targets are incomplete: 1372 train/678 test rows are finite,
with 3718 joined rows excluded explicitly. JND_score uses all 3920/1848.
The original script's empty reduction failed before this admission correction.
Search resource receipt: run-heavy done rc=0 63s | peak-RSS 0.22GiB |
min-avail 52054MiB | peak-load 1.23. Private aggregate/search commit: 51b3fae310b3.

## HDR-VDP 2 strict arithmetic control

The same 61 crops with `--hdrvdp-sdr-strict` give median 0.00365021 and
maximum 0.0514859 site-JND against the published column. Native strict versus
relaxed differs by median 1.6615683477994025e-6 Q, maximum
6.280069452202497e-5 Q. Strict versus independent Octave remains median
0.008689808892256679 Q, maximum 0.014212052720857571 Q. Strict arithmetic
therefore does not explain either residual. No default or reference setting
was changed. Full rows and the published-column summary are retained beside
this document. Combined build, strict-panel, clippy and refusal-control receipt:
run-heavy done rc=0 59s | peak-RSS 0.84GiB | min-avail 50806MiB | peak-load 5.22.

## Runtime supplements and unmerged fixes

The qualified HDR/AHIQ and fast-route tables cover 1024x1024 and 3840x2160,
one/eight worker budgets and three repeats per cell on r5900xt. All eight
routes complete. The original HDR names require absolute luminance and
correctly refuse this RGB8 fixture; their explicit AIC routes supply that
contract. NR routes require the NR build feature. AHIQ uses explicit weight
consent. These build/input refusals do not establish a metric-kernel panic.
See each `*-perf.meta` for source, commands, resource receipts and timing
qualifications. The original short-delta panel is retained as diagnostic data.

The fixed-luma FSIM panel has 36 observations and the private DVIFM aggregate
panel has 48; all complete with identical repeated score strings. FSIM uses
[unmerged PR 92](https://github.com/imazen/zenmetrics/pull/92), not a master CLI
route. Its three tally files retain the inherited analytical FFT/count-model
qualifications. DVIFM exports only aggregate observations and partial counts.
Neither panel identifies an unavailable authors' per-image score column.

`fsim-branch-comparison.tsv` retains before/after timing and OS peak RSS for
the existing luma route. Scores match exactly at both sizes and budgets.
Eight-thread deltas are short and noisy; no speedup is claimed. The branch
changes the luma call path and remains unmerged for owner review.

`fix/zensim-gpu-refusal` (08a7be724327) changes the disabled route's panic to
an explicit error, without enabling scoring. The real 64x64 process control
changes exit 101 to exit 2. Its regression passes; the branch remains unmerged.
The master inventory truthfully retains the panic. Exact resource receipts,
RSS observations and private-adapter mappings are in `screenshot-timings.meta`.

## Landing-page publication

The live root now carries the seven-link navigation and responsive theme,
including [Gigapixels](https://gigapixels.imazen.org/). The served HTML matches
its manifest hash. All 42 other asset hashes remain unchanged. Viewer,
AIC evidence and landing HTML retain separate truthful source identities;
the landing-only update does not qualify a new metric/WASM build.
The source is f84dd4b24f78; live and desktop-light/phone-dark layout receipts
are retained here. Both layouts pass with no horizontal overflow.
Layout resource receipt on r5900xt: rc0 2s, peak-RSS0.18GiB,
min-avail51707MiB, peak-load1.04. Deployment resource receipt on dev:
rc0 5s, peak-RSS0.31GiB, min-avail37016MiB, peak-load1.56.
