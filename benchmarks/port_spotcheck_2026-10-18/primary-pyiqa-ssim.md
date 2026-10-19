# Primary PyIQA SSIM grid, 2026-10-19

Missing: documented cross-implementation bounds for these twenty CLI names,
map comparisons where applicable, and qualified cold/warm timing ratios.
Every scalar observation remains **UNBOUND / UNQUALIFIED**.

The [480 observations](primary-pyiqa-ssim.json) cover all twenty registered
base and fixed-factor names, three raw content pairs, four geometries and
1/28 threads. There are 390 scored comparisons and 90 primary size errors
with matching candidate refusals. The index binds all six small TSV shards,
original timeout attempts, isolated supplements and artifact manifests.
The original three 300-second timeouts remain recorded; seven missing
conventions were subsequently collected in separate unchanged-timeout calls.

The fixed-factor MS-SSIM correction keeps the twofold image pyramid separate
from each level's fixed-factor scoring view. At factors 2–6 its respective
largest primary differences become 1.0223587643132248e-8,
5.184784268585929e-9, 2.973977730746924e-9, 2.0381685228443303e-9 and
1.4596902575547688e-9. Factor 6 at 1 MP now refuses the same too-small
window as primary. Before/after base scores and all scored single-scale
fixed variants remain bit-identical. The unmerged source bookmark remains
available; owner integration 59f81e48 independently brought the correction
to master. These observations retain their original executable identities.

The largest unchanged-default differences are 1.727458853739705e-7 for
unrounded RGB SSIM, 3.803296377835608e-8 for unrounded base MS-SSIM,
and 1.5453514410235236e-5 for the original rounded SSIM class. They are
measured differences, not newly selected acceptance bounds.

## Precision controls

[Three diagnostic cases](primary-pyiqa-ssim-precision.json) reproduce their
unchanged primary values exactly. Passing a float64 Gaussian to the original
function reduces the selected RGB difference from 1.727458853739705e-7 to
3.0309088572266774e-14 and the selected MS-SSIM difference from
3.8032963667333775e-8 to 6.40884012526044e-11. Primary constructs its
Gaussian in float32 under the captured default dtype before promoting it.

For the selected rounded SSIM class, primary's promoted float32 luma
coefficients put two reference pixels on a different rounding result.
Float64 luma alone reduces the difference from 1.5453514410235236e-5 to
8.521540695305418e-8; a float64 Gaussian as well reduces it to
6.994405055138486e-15. These controls explain those selected cases; they
do not replace the primary outputs or qualify the rest of the grid.

[Commands, source identities and resource receipt](primary-pyiqa-ssim.meta)
retain the full scope. No metric, threshold, expected test value or installed
primary source changed for these diagnostics.
