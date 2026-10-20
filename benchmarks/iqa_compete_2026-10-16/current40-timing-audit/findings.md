# Current40 timing-boundary source findings

Detailed route matrix is in progress. Selection is verified for 40/40 routes and
80/80 derived cases; selection alone does not qualify timing or numerical parity.
No active helper, plan or result has been edited; no scoring/build/CI ran.

## HDR preparation placement

The selected AIC plan SHA256 is
`0f053bca0de3fb093e73b8b3348755ea473f605096ec38ef410aca5b7dc0bf14`.
Its `hdrvdp-aic2026-pair-cid22-t1` and
`hdrvdp3-aic2026-pair-cid22-t1` rows become both one/eight-thread comparative
cases and claim resident immutable parameters. The selected
`canonical_hdrvdp_cell.py` SHA256 is
`4033ded44b11fcab8f9cad0dedde5d2d74103dfc43701965eab074877c3b1cb1`.
Lines 44–49 time the official call without replacing its spectral loader;
original v2 `hdrvdp.m:247,249` and v3 `hdrvdp_visual_pathway.m:58,62` reach
`load_spectral_resp.m:10` (`dlmread`) inside every warm clock. Native spectra
come from embedded CSV. File-I/O placement differs; preserve the AIC convention
and qualify this scope rather than borrowing the base resident cohort.

The base promoted plan hashes are
`50129e991b88e3215ae69475c0fc917eaf5b7d6fd580ebddb60f0e535151f0a7`
(v2) and `fe237685c47ccf9972678f442591a03d21332c8802ab724bfc4f176d09c145c0`
(v3). The base fixture preloads **interpolated** original-loader outputs before
clocks (`first_boundary_hdrvdp_resident.py:105–116`, SHA256
`858f2ae0ad84dfb306de0173af381332c972462b1975f56797d8c0d3a46c091f`);
its shim/cache (`:46–60`) returns those arrays during evaluation.
Exact production EA v2 `metric.rs:241,282` and v3
`v3/metric.rs:214`, `v3/pathway.rs:227` still parse and interpolate embedded
tables per timed warm call. MTF/JND caching does not cover spectral loaders.
Reference interpolation is outside and native interpolation inside the clock.
Passing array/ABBA/protocol/native score controls establish numerical behavior,
not equal parameter-preparation placement.

Owner disposition: retain numerical controls and raw base-HDR timings as
explicitly asymmetric descriptive evidence, excluded from strict acceptance.
The [future raw-table-resident design](hdr-parameter-design.md) excludes file
loading while keeping original interpolation timed, and names residual CSV-
parsing placement. It changes no existing source or observation.

## DSSIM constructor and harness provenance limit

Selected author plan SHA256
`e543e3f753575578504b228ed63655b81e82c6c042a24e12f6139d9b21476af4`
selects `dssim-author-512x512-t1` then templates both budgets. Current
`dssim_reference.c:28,41` stages context creation/destruction outside clocks
31–37; native `metrics/mod.rs:2212–2220` constructs the umbrella metric on every
call, and `cpu_dispatch.rs:231–236` constructs its boxed Dssim state. This is
an explicit configuration-lifetime difference, with no measured size/effect.
The selected plan binds the author binary and `build.sha256`, but that receipt
is emitted by `build_dssim_reference.sh:15–16` without hashing the C harness.
No C-harness source path appears in the selected plan's bindings. Exact harness
source-to-binary correspondence therefore needs supporting retained build
provenance before treating this C source as independently proven binary input.
This is a source-proof limit, not evidence of a different observed binary.
