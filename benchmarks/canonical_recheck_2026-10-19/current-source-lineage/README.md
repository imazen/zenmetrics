# Current-source coverage audit, 2026-10-20

The source review found one omitted revised route: `psnrhvs-daala` reaches the changed studio-601 RGB preprocessing helper. It is now included in the changed-only41 selection and a separate current-source24-cell grid. The two new base HDR routes require a separate48-cell numerical panel; their512photo controls do not cover that grid. These acquisitions remain separate from this source review.

The86 route classifications compare `efae6b8f` with normal worker source `ea767ba1`:24 reached material changes, one same-gate helper extraction, and61 without a reached material production change in that diff. This classifies source reachability; it does not prove unchanged compiled instructions, timing or numerical results. The complete rows are in `routes-00.json` through `routes-02.json`.

The coordinator independently rechecked every recorded published EFA/EA source hash and each asserted EA frozen-tree match. `source-hashes.json` retains the dependency and selected historical-source equivalence evidence; `cohort-evidence.json` binds the inspected plans. Older worker timings retain their original source identities. Revised VMAF full-call, MSSSIM, Daala and HDR panels do not retroactively qualify earlier results. No numerical bound or production metric code changed.
