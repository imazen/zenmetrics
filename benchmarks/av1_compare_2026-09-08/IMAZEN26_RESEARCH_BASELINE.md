# Canonical imazen-26 research baseline — 2026-09-08

Completed **150 encodes / 50 cells / three interleaved rounds**, independently
decoded and byte-deterministic. This is a two-training-origin baseline, not a
representative subset or a calibrated routing policy. Native SVT -1 is now
reachable through the signed Rust pipeline, public AVIF wrapper and static
comparison executable. No Zen enhancement is enabled in these measurements.

## Measured size, quality and time

The table estimates matched RGB SSIMULACRA2 by log-linear interpolation between
measured QP endpoints. No quality extrapolation or reversed segments. These
are sparse anchor curves, not the full matched-time frontier. Payload is AV1
OBU bytes, excluding AVIF container overhead; times are per-cell medians.

| Origin | Target SSIM2 | Backend | Preset | Bytes | bpp | Time ms | QP bracket |
|---|---:|---|---:|---:|---:|---:|---|
| 1000 | 70 | c-svt-av1 | -1 | 25,755 | 1.048 | 2,238 | 20–32 |
| 1000 | 70 | libaom | 0 | 26,808 | 1.091 | 3,082 | 20–32 |
| 1000 | 70 | zenav1-aom | 0 | 26,808 | 1.091 | 6,429 | 20–32 |
| 1000 | 70 | zenav1-svt | -1 | 25,755 | 1.048 | 3,229 | 20–32 |
| 1000 | 70 | zenav1-svt | 0 | 26,149 | 1.064 | 1,766 | 20–32 |
| 8100 | 80 | c-svt-av1 | -1 | 11,494 | 0.561 | 2,673 | 20–32 |
| 8100 | 70 | c-svt-av1 | -1 | 7,844 | 0.383 | 3,052 | 32–48 |
| 8100 | 80 | libaom | 0 | 11,259 | 0.550 | 3,587 | 20–32 |
| 8100 | 70 | libaom | 0 | 7,866 | 0.384 | 3,430 | 32–48 |
| 8100 | 80 | zenav1-aom | 0 | 12,705 | 0.620 | 2,759 | 20–32 |
| 8100 | 70 | zenav1-aom | 0 | 8,708 | 0.425 | 2,455 | 32–48 |
| 8100 | 80 | zenav1-svt | -1 | 11,494 | 0.561 | 14,280 | 20–32 |
| 8100 | 70 | zenav1-svt | -1 | 7,844 | 0.383 | 16,117 | 32–48 |
| 8100 | 80 | zenav1-svt | 0 | 11,819 | 0.577 | 1,932 | 20–32 |
| 8100 | 70 | zenav1-svt | 0 | 7,966 | 0.389 | 2,302 | 32–48 |

The photo's Rust -1 estimate buys **1.5% fewer bytes for 83% more time** than
Rust preset0 at score70. The screenshot's score80 estimate buys **2.7% fewer
bytes for 7.4× the time**. These small estimated size gains need denser direct
verification; they do not justify making research mode a general default.
On the screenshot at measured QP20, C and Rust -1 produce exactly the same
**14,292B / 0.698bpp / 84.101 SSIM2**, but take **2,504ms vs13,528ms**.
This 5.4× gap is a useful performance-profiling witness; its cause is not yet
attributed. The photo's corresponding Rust/C ratio is1.44×.

Both sources match C SVT -1 at all five quantizers (**10/10 distinct pairs,
30/30 repeated pairs**). Libaom/Rust AOM0 match all five photo cells, but
**all five screenshot cells differ**. At screenshot QP20, libaom produces
**15,765B /0.770bpp /85.179 SSIM2 /3,811ms**, versus Rust AOM
**17,170B /0.838bpp /84.267 SSIM2 /3,132ms**.
These AOM differences remain separate correctness/configuration witnesses;
successful decoding is not bit parity.

## Inputs, protocol and limits

- Canonical metadata: imazen/imazen-26 at
  `187fbf338ce08e8e6654db7f04ddae58d5263da2`.
- Training origin1000 (photo), resized4032×3024→512×384; origin8100
  (web screenshot), resized1440×900→512×320. Lanczos3, full images rather than
  the earlier parity check's512×512 center crops. No held-out origin used.
- PNG-v3 URLs come from the pinned canonical split manifest; downloaded PNG
  hashes are recorded. The split manifest's raw-original SHA must not be
  mistaken for an expected PNG-v3 hash.
- Eight-bit SDR, BT.709 limited-range4:2:0, tune/SCM defaults, one requested
  thread (C SVT uses lp1). QP5/12/20/32/48 for every arm. RGB→YUV conversion
  and independent decode/scoring are outside the fresh-lifecycle API timer.
- Conversion-only SSIM2 ceilings: photo**80.217**, screenshot**87.927**.
  Chroma formats must be a separate scouting stratum; 4:2:0-only measurements
  cannot cover all high-quality routing regions. These are not HDR tests.
- Intel Core Ultra7 265K, serial `run-heavy`, nice19, RAYON/OMP1. CPU affinity
  was not pinned. Keep this cohort separate from fleet hardware and confirm
  routing crossover settings on an isolated pinned core.
- Static PIE, no dynamic NEEDED entries. Binary SHA256:
  `fb469f2c0fe497def07dea556aec08d33b7d253616a6114aa88211ddc6527966`.
  Backend revisions, source/input/output hashes, timing ranges and conversion
  ceilings are recorded in [all measured cells](imazen26_research_cells.tsv).
  [Matched brackets](imazen26_research_matched.tsv) preserve endpoints.
- Full local evidence: `~/tmp/av1-imazen26-research-baseline-2026-09-08/`:
  rows, OBUs, reference PNGs, request/source/CPU provenance and local patches.
  C SVT is the pinned hybrid source with HDR_MODE=OFF, not an unqualified
  claim of pristine mainline parity.

The next steps remain a canonical zenfleet scouting sweep, measured behavior
zones and minimum-set selection with explicit tolerances/optimality bounds,
followed by held-out validation. Immediately beyond native -1, the goal calls
for isolated AOM-technique adoption inside SVT; backend routing alone does not
satisfy that requirement. No production default, CI or main integration changed.

## Research screenshot profile

Separate from the timing rounds above, sampled two encodes of canonical8100,
512×320, QP20 at native -1 for each C/Rust SVT arm, same static executable.
`perf record -e cycles:u -F199 --call-graph dwarf,8192` recorded5,414 Rust
samples and1,010 C samples, no lost samples. Almost all samples are from the
CPU core PMU; the atom PMU has only source-hashing samples.

Rust `svtav1_dsp::me_sad::__arcane_block_sad_v3` accounts for **78.22% self
core cycles**. `intrabc::exhaustive_mesh_search` additionally accounts for
3.16% self cycles. C's hot SAD symbols are size-specialized four-candidate
AVX2 kernels (`svt_aom_sad4x8x4d_avx2`, `sad8x8x4d`, `sad8x16x4d`, etc.).
Source inspection confirms the Rust mesh's inner loop calls SAD separately for
each of four positions; C `av1me.c::exhaustive_mesh_search` calls `sdx4df`
on the four addresses together. Rust's generic SAD also uses scalar tails for
width4. This identifies a useful byte-preserving optimization target before
spending more time on broader Zen search; no improvement is claimed yet.

Keep the full research search, MV costs, position order, strict comparisons
and C's remainder behavior. Wire any batched kernel into the live search and
validate the kernel against C and whole-image byte identity before repeating
these RD/time measurements. Raw profiles and reports are retained under
`~/tmp/svt-tracking/research-screen-{rust,c}.perf*`.

Validation for the comparator change: all five library tests and scoped
Clippy with `-D warnings` pass. SVT's updated regression suite passes133/133.

## Batched mesh SAD follow-up

Local SVT change `d78559a2` wires four-candidate SAD into the IntraBC exhaustive
mesh, sharing source loads and using SIMD for width4. Candidate order, strict
comparisons, MV costs and C's remainder behavior are unchanged. Independent
pixel-sum tests exercise dispatch permutations, distinct strides and tight
last rows. All2,619 workspace tests (including C IntraBC search differentials)
and133/133 regression checks passed before this measurement.

Three-round QP20 repeat, same two images, static API protocol and CPU cohort:

| Origin | Backend/preset | Bytes | SSIM2 | Before ms | After ms |
|---|---|---:|---:|---:|---:|
|1000 photo|C SVT -1|27,421|71.675|2,189.8|2,193.9|
|1000 photo|Rust SVT -1|27,421|71.675|3,160.1|3,146.3|
|1000 photo|Rust SVT 0|26,684|70.571|1,753.3|1,752.5|
|8100 screenshot|C SVT -1|14,292|84.101|2,503.8|2,503.1|
|8100 screenshot|Rust SVT -1|14,292|84.101|13,528.4|6,105.9|
|8100 screenshot|Rust SVT 0|14,380|83.848|1,737.1|1,669.7|

All18 outputs match the corresponding earlier output hashes, including the
unchanged C controls. This witness's research runtime drops55% (2.22× speedup);
Rust still takes2.44× C time, so the remaining gap is not closed. The photo is
essentially unchanged. Broader size/preset/architecture performance remains
unmeasured for this optimization. Do not extrapolate the speedup to all images.
Data: `~/tmp/av1-imazen26-sad4-2026-09-08/rows.jsonl`, with OBUs/references.
This run precedes the user's subsequent migration from the Git-patched SIMD
snapshot to the published archmage/magetypes0.9.29; that dependency change has
its own validation gates.

Follow-up binary SHA256: `009386b5986d4337957d9ac93f209d78f2147473a03bc46d31eaee468a07c1a0`.

### Published archmage/magetypes0.9.29 validation

Removed the temporary Git patches from SVT, its six standalone performance
probes, zenavif and this comparator. All lockfiles resolve registry0.9.29 for
archmage, archmage-macros and magetypes. Published-version checks pass:
2,619 SVT workspace tests,133/133 byte-regression checks, five comparator tests,
27 zenavif backend/roundtrip tests, and all-target checks for all six probes.

The rebuilt static comparator repeats all18 QP20 encodes with unchanged output
hashes. Screenshot Rust -1 median is**6,113.6ms**, C**2,506.7ms**, Rust0
**1,671.8ms**. Photo Rust -1 is**3,152.5ms**, C**2,189.6ms**, Rust0**1,757.3ms**.
Byte counts and SSIM2 remain exactly as in the table above. The batched-SAD
speedup survives the published dependency migration; no RD regression observed
in these cells. This is still a two-source performance witness.

Data: `~/tmp/av1-imazen26-sad4-archmage029-2026-09-08/`.
Binary SHA256: `1a65ccad0407dda48f8643cc23e09ae992237a35c1fed4257b83f6632981d9f2`.
