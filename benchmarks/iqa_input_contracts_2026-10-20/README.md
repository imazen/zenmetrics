# Captured non-NN input-contract refusals

The 2026-10-20 00:07 UTC snapshots contain 224 failed attempts, all with explicit
input-contract errors. They are not implementation-unavailable or stalled-worker
classifications, and none supplies a qualified timing.

| Host | Fixed-pyramid window refusals | Absolute-luminance refusals | Total attempts |
|---|---:|---:|---:|
| i265 | 48 | 32 | 80 |
| mac | 32 | 16 | 48 |
| r7900x | 48 | 48 | 96 |

`hdrvdp` consumes absolute luminance; `hdrvdp3` also requires viewing conditions.
The frozen sRGB8 inputs do not satisfy those contracts. No luminance was invented
and no ingress conversion was introduced. Separately named AIC2026 routes retain
their own identities; this audit does not relabel them as these two routes.

In this worker revision, fixed-factor MS-SSIM `k3/k4/k5/k6` refuse 512×512.
For `k3`, scale 5 is 10×10; for `k4`, 8×8; for `k5`, 6×6; `k6` already reaches
10×10 at scale 4. The valid window requires at least 11×11. At 840×944, `k5/k6`
also refuse. Larger geometries remain part of the original frozen queue.
Downsampling and window limits are unchanged.

The JSON retains each exact error, original attempt status, cell/report identity
and worker-log hash. Counts are captured attempts, not exhausted logical cells;
remaining attempt caps are unchanged. Resource-flagged successful attempts are a
separate category and remain unqualified pending a permitted clean retry.
`provenance.json` binds the source observation snapshots and this classification.
