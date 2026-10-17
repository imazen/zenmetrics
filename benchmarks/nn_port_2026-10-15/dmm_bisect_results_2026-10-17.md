# DMM individual-change acceptance, 2026-10-17

Each candidate starts from the frozen solver, with one numerical change. Acceptance requires all 48 frozen row limits, no increase in any row residual against the measured baseline, and identical complete CUDA repeat scores.

| Change | Source | Within frozen limits | Improved / equal / worse residuals | Median / max netlib error | CUDA repeats | Decision |
|---|---|---:|---:|---:|---:|---|
| baseline | `66bb98605c83` | 48/48 | 0 / 48 / 0 | 7.954546606e-04 / 5.544746884e-02 | 24/24 identical | baseline |
| epsilon | `1ae5f65d90cb` | 48/48 | 30 / 0 / 18 | 7.894833450e-04 / 5.544770590e-02 | 24/24 identical | reject |
| small_n_scaling | `83ab12724001` | 40/48 | 28 / 0 / 20 | 7.679859217e-04 / 5.539626087e-02 | 24/24 identical | reject |
| org2r_order | `2af0ff21a3d3` | 48/48 | 24 / 0 / 24 | 7.953349912e-04 / 5.544749321e-02 | 24/24 identical | reject |
| lartg | `c105b75b215b` | 48/48 | 22 / 0 / 26 | 7.954556295e-04 / 5.544746776e-02 | 24/24 identical | reject |
| orm2r_order | `2de0604521bf` | 40/48 | 23 / 0 / 25 | 7.614877929e-04 / 5.529337334e-02 | 24/24 identical | reject |
| right_sger_order | `9d9132a22d77` | 28/48 | 22 / 0 / 26 | 7.327735206e-04 / 1.291220382e-01 | 24/24 identical | reject |
| f64_norm | `6f825d9096d1` | 29/48 | 26 / 0 / 22 | 5.087865108e-04 / 4.494167999e-02 | 24/24 identical | reject |
| reference_norm | `5ea8c65b0582` | 31/48 | 24 / 0 / 24 | 5.141195319e-04 / 8.781208543e-02 | 24/24 identical | reject |

small_n_scaling: native panel controls did not all pass; this candidate cannot be retained.

orm2r_order: native panel controls did not all pass; this candidate cannot be retained.

right_sger_order: native panel controls did not all pass; this candidate cannot be retained.

f64_norm: native panel controls did not all pass; this candidate cannot be retained.

reference_norm: native panel controls did not all pass; this candidate cannot be retained.

Resource receipts for the complete three-panel runs (parity, not speed benchmarks):

```text
baseline: run-heavy: done rc=0 840s | peak-RSS 8.31GiB | min-avail 13909MiB | peak-load 11.04
epsilon: run-heavy: done rc=0 855s | peak-RSS 8.32GiB | min-avail 13851MiB | peak-load 8.70
small_n_scaling: run-heavy: done rc=1 894s | peak-RSS 8.29GiB | min-avail 13746MiB | peak-load 8.59
org2r_order: run-heavy: done rc=0 830s | peak-RSS 8.32GiB | min-avail 13577MiB | peak-load 6.73
lartg: run-heavy: done rc=0 844s | peak-RSS 8.27GiB | min-avail 13874MiB | peak-load 5.23
orm2r_order: run-heavy: done rc=1 757s | peak-RSS 8.30GiB | min-avail 13434MiB | peak-load 14.31
right_sger_order: run-heavy: done rc=1 1036s | peak-RSS 8.31GiB | min-avail 13527MiB | peak-load 5.46
f64_norm: run-heavy: done rc=1 850s | peak-RSS 8.32GiB | min-avail 13851MiB | peak-load 5.94
reference_norm: run-heavy: done rc=1 874s | peak-RSS 8.29GiB | min-avail 13509MiB | peak-load 5.53
```
