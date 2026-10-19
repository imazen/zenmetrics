# Primary VIFp RGB luma precision finding

Missing: verified candidate correction, regression test, before/after speed
and memory comparison, and qualified canonical cold/warm timing ratios.
The frozen Rust baseline passes two of the [24 scalar cases](primary-vif.json)
and misses the unchanged 1e-9 absolute bound in 22. This finding concerns
`vif` (pixel-domain VIFp); `vifvec` is a separate algorithm.

On the 1024 squared photo pair, at both thread budgets, Rust returns
0.5992444156065622 versus 0.5992444196997502 from unchanged author
[`vifp_mscale.m`](https://github.com/sattarab/image-quality-tools/tree/163fff128a86260efcc962de52196c71a3d3d06e/metrix_mux/metrix/vif/vifp_release).
The absolute difference is 4.093188010756421e-9. Original repeat calls agree
exactly. All three raw content pairs, four geometries and 1/28 threads return
scores, so these are numerical findings rather than process failures.

Source inspection identifies a candidate explanation: the private RGB luma
coefficient array is f32, then promoted to f64 for the f64 pipeline. The
reference applies the decimal constants in double precision. A private f64
coefficient correction is being evaluated on `fix/vif-rgb-luma-precision`;
it has not been independently qualified or merged. The candidate comparison
also requires `vifvec` scores and size refusals to remain unchanged. Neither
the numerical bound nor existing test expectations have changed.

The primary top-level function returns a scalar, without a native map API.
The panel receipt is rc=1 (recorded numerical findings), 69s, peak RSS 1.00 GiB,
minimum available 12532 MiB and peak load 2.21. [Metadata](primary-vif.meta)
binds original source, frozen executable, input hashes, commands and both
process RSS values. It supplies no qualified speed ratio.
