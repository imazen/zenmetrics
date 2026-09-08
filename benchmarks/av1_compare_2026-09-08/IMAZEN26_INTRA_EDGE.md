# Canonical imazen-26: corrected intra-edge ablation

2026-09-08, local native-API comparison. **Keep intra-edge filtering
opt-in.** This pilot does not justify an automatic enhancement bundle. At
matched quality the corrected experiment costs more bytes and time on both
selected targets. The tiny high-quality photo gains warrant a bounded broader
check, not a universal default or a screenshot detector rule.

## Measured QP 20 results

Payload bytes are raw AV1 OBU bytes, excluding AVIF container overhead. Time is
the median of three interleaved, single-thread fresh encoder lifecycles.

| Source | Encoder | Bytes | bpp | SSIMULACRA2 | Median ms |
|---|---|---:|---:|---:|---:|
| Photo 1000,512x384 | SVT native −1, pristine | 27,489 | 1.1185 | 71.084 | 3,187.2 |
| Photo 1000,512x384 | Same + intra-edge | 27,384 | 1.1143 | 70.655 | 3,351.3 |
| Photo 1000,512x384 | libaom0 | 30,162 | 1.2273 | 72.812 | 3,248.2 |
| Photo 1000,512x384 | zenav1-aom0 | 30,162 | 1.2273 | 72.812 | 6,774.3 |
| Screenshot 8100,512x320 | SVT native −1, pristine | 14,327 | 0.6996 | 84.057 | 6,181.0 |
| Screenshot 8100,512x320 | Same + intra-edge | 14,344 | 0.7004 | 83.677 | 6,256.0 |
| Screenshot 8100,512x320 | libaom0 | 15,765 | 0.7698 | 85.179 | 3,819.6 |
| Screenshot 8100,512x320 | zenav1-aom0 | 17,170 | 0.8384 | 84.267 | 3,148.3 |

The photo experiment is 0.38% smaller at this quantizer but loses 0.429 SSIM2
and takes 5.15% longer. The screenshot grows 0.12%, loses 0.380 SSIM2 and takes
1.21% longer. Across the five measured quantizers, photo QP 5/12 improve by
0.112/0.081 SSIM2 and shrink 0.40%/0.36%, with 3.70%/2.98% extra time. The
screenshot loses quality at every measured quantizer (0.088 to1.333 points).
The cell table includes all reversals/tradeoffs, timing ranges and hashes.

## Common-quality and time-budget comparisons

These are **estimates**, log-linear within measured QP 20..32 brackets; no
extrapolation or preset interpolation. There were no reversed RD segments.

| Target | Encoder | Estimated bytes | Estimated ms |
|---|---|---:|---:|
| Photo, SSIM2=70 | SVT native −1 | 26,333 | 3,236 |
| Photo, SSIM2=70 | SVT native −1 + intra-edge | 26,701 | 3,388 |
| Photo, SSIM2=70 | libaom0 | 26,808 | 3,099 |
| Photo, SSIM2=70 | zenav1-aom0 | 26,808 | 6,457 |
| Screenshot, SSIM2=80 | SVT native −1 | 11,434 | 6,490 |
| Screenshot, SSIM2=80 | SVT native −1 + intra-edge | 11,629 | 6,557 |
| Screenshot, SSIM2=80 | libaom0 | 11,259 | 3,565 |
| Screenshot, SSIM2=80 | zenav1-aom0 | 12,705 | 2,783 |

At the photo's 3.5s budget, native −1 has the smallest payload among these
measured arms. At the screenshot's 4s budget, neither SVT-1 arm fits; libaom0
and zenav1-aom0 do. At 7s, libaom0 remains smallest among these arms.
**This is not a complete backend frontier:** faster SVT presets and other AOM
presets are absent from this isolated ablation. These observations do not
justify routing all screenshots to AOM. The previous normal-preset sweep also
used different sources and reference choices; do not silently pool those curves.
The libaom/zenav1-aom screenshot RD difference remains a separate open audit.

## Execution and conformance evidence

- 120 completed encodes =2 training origins x5 quantizers x4 arms x3 rounds.
 Every output independently decoded; all40 cells were byte-deterministic
 across rounds. The request covers QP 5/12/20/32/48,8-bit420, default tune/SCM.
- 20 SVT cells additionally passed **untimed reconstruction replay**: identical
 prepared-input hash, exact measured OBU bytes, and every native reconstructed
 sample equal to libaom's decoder. Both native and enhanced settings are
 checked independently even when their outputs happen to match.
- Calibration input: canonical imazen-26 pin
 `187fbf338ce08e8e6654db7f04ddae58d5263da2`, registered SDR PNG variants of
 training origins 1000/8100, full-image Lanczos3 resize to max-edge512.
 These two origins are a pilot, not the minimum representative encoding set.
 Neither validation nor test outcomes were used for selection.
- BT709 limited420 from the same RGB conversion for every arm. Conversion
 ceilings are 80.217/photo and 87.927/screenshot; these are SDR measurements.
 Native10 conformance is tested separately, not measured RD here.
- Intel Core Ultra 7 265K, serial nice19 jobs 1, codec/RAYON/OMP threads 1,
 not affinity-pinned. Do not pool with different CPU cohorts or claim the
 estimated times as portable deadlines.
- SVT base reference is explicitly pristine mainline 4.2.0
 `9292ec8e32bce26f781f277ec8739b53426c4300`, for both arms. The linked hybrid
 C SVT arm is omitted. `aom-intra-edge-filter-v1` is explicit, off by default.
- Timed binary SHA256:
 `f8581baed256acc7bd30f638b063b5a47a651be4c0d620ceb126de61c2bfefb6`.
 Reconstruction verifier SHA256:
 `b97d592f558b6bff353500b58f75334ef269fb0e01a9a4d1ecbf7a5219313b05`.
 These two executables link the codec archives statically but depend on the
 system libc/libm/libgcc and loader. They are not fully static executables.
 The later `build-static.sh` result is a separate binary identity and does not
 replace or relabel these measured rows.
 The verifier is a later harness build; its byte-replay requirement ties its
 checks to the exact original measured outputs without retiming them.

The first sweep (`av1-imazen26-intra-edge-2026-09-08`, binary d6cd3261) is
**superseded for policy selection**. Expanded geometry tests exposed a real
chroma-neighbor defect after 4x4 luma splits. Reading the adjacent luma-only
block missed the chroma owner's smooth mode; those children also overwrote the
UV neighbor maps prematurely. The correction preserves chroma owners and
uses normative 8x8 group/tile availability. The apparent photo QP 20 gain in
the old run did not survive the correctness fix. No tool or gate was disabled.

Post-fix local correctness: 2626 workspace tests, 136 regression checks,
36 off/on geometry/depth/QP encodes, and 6 comparator tests passed.
Refreshed C identity gates pass: 1,100/1,100 hybrid normal8, 1,100/1,100
pristine normal8, and 320/320 pristine research cells (160 each at native8/10).
These counts do not close the old real-image parity gaps or the broader
policy/routing goal.

## Artifacts

- [All 40 measured cells](imazen26_intra_edge_cells.tsv)
- [Bracketed quality estimates](imazen26_intra_edge_matched.tsv)
- [Budget-constrained points, same hardware cohort](imazen26_intra_edge_time_budgets.tsv)
- Local complete artifact directory:
 `~/tmp/av1-imazen26-intra-edge-corrected-2026-09-08/` contains the raw rows,
 content-addressed OBUs, resized references, source/CPU/request provenance,
 measured binary and 20 reconstruction-verification records.
- Reproduce measurement with
 `~/tmp/svt-tracking/imazen26-intra-edge-corrected-request.json` and comparator
 `measure`; use comparator `verify-measurement` with `measurement_dir` and a
 new `output` path to check saved SVT rows outside timing.
- Logs: `~/tmp/svt-tracking/imazen26-intra-edge-corrected.log`,
 `zen-edge-measurement-recon.log`, `zen-edge-replay-{tests,build}.log`.

Durable archive on the configured LAN object store (`zentrain`), downloaded
and SHA256-verified after upload:

- `s3://zentrain/benchmarks/av1-compare/2026-09-08/intra-edge-corrected/evidence-3a644bc6f4c30fc3da0d3342632cc7a1325ea7695537965159d5ea75dd9197e4.tar.gz`
- SHA256 `3a644bc6f4c30fc3da0d3342632cc7a1325ea7695537965159d5ea75dd9197e4`; 709384601 bytes.
- Includes raw parity runs, both measured and verifier binaries, original
  canonical sources, exact measured-source snapshots checked against all
  recorded hashes, pinned C source archives, and the superseded pre-fix run.
