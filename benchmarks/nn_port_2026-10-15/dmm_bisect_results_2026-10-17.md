# DMM individual-change acceptance, 2026-10-17

**Missing:** org2r_order, lartg, orm2r_order, right_sger_order, f64_norm, reference_norm.

Each candidate starts from the frozen solver, with one numerical change. Acceptance requires all 48 frozen row limits, no increase in any row residual against the measured baseline, and identical complete CUDA repeat scores.

| Change | Source | Within frozen limits | Improved / equal / worse residuals | Median / max netlib error | CUDA repeats | Decision |
|---|---|---:|---:|---:|---:|---|
| baseline | `66bb98605c83` | 48/48 | 0 / 48 / 0 | 7.954546606e-04 / 5.544746884e-02 | 24/24 identical | baseline |
| epsilon | `1ae5f65d90cb` | 48/48 | 30 / 0 / 18 | 7.894833450e-04 / 5.544770590e-02 | 24/24 identical | reject |
| small_n_scaling | `83ab12724001` | 40/48 | 28 / 0 / 20 | 7.679859217e-04 / 5.539626087e-02 | 24/24 identical | reject |
| org2r_order | `2af0ff21a3d3` | unmeasured | unmeasured | unmeasured | unmeasured | pending |
| lartg | `c105b75b215b` | unmeasured | unmeasured | unmeasured | unmeasured | pending |
| orm2r_order | `2de0604521bf` | unmeasured | unmeasured | unmeasured | unmeasured | pending |
| right_sger_order | `9d9132a22d77` | unmeasured | unmeasured | unmeasured | unmeasured | pending |
| f64_norm | `6f825d9096d1` | unmeasured | unmeasured | unmeasured | unmeasured | pending |
| reference_norm | `5ea8c65b0582` | unmeasured | unmeasured | unmeasured | unmeasured | pending |

small_n_scaling: native panel controls did not all pass; this candidate cannot be retained.

Resource receipts for the complete three-panel runs (parity, not speed benchmarks):

```text
baseline: run-heavy: done rc=0 840s | peak-RSS 8.31GiB | min-avail 13909MiB | peak-load 11.04
epsilon: run-heavy: done rc=0 855s | peak-RSS 8.32GiB | min-avail 13851MiB | peak-load 8.70
small_n_scaling: run-heavy: done rc=1 894s | peak-RSS 8.29GiB | min-avail 13746MiB | peak-load 8.59
```
