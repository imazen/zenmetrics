# Canonical imazen-26 restoration-unit ablation

2026-09-08. **Keep restoration-unit search opt-in.** The experiment selected
128-pixel units on the photo at QP5/12, but neither case improved the measured
size/perceptual-quality tradeoff. The other photo quantizers retained 256.
Restoration search was bypassed for the screenshot at every tested quantizer;
its identical outputs do not establish that smaller units are unhelpful for
all screen content. No experiment or C coding tool was removed.

The completed fleet run contains 144 encodes: two canonical training origins,
six quantizers (5/12/20/32/48/60), four arms and three interleaved rounds.
All 48 cells were deterministic and independently decoded. Every one of the
24 unique Rust SVT cells passed automatic exact-OBU/reconstruction verification;
a later untimed replay also reproduced every input/output hash and recorded
the selected restoration size. Both Nomad workers drained; the canonical ledger
contains two completed jobs and zero gaps.

## Measured size, quality and time

QP20 rows below are direct measurements, not matched-quality comparisons.
The photo ran on a Ryzen 5 3500; the screenshot on a Ryzen 9 7900X. Compare
arms within an image only. Times from these CPUs are not pooled with the earlier
Intel measurements.

| Source | Arm | Bytes | bpp | SSIMULACRA2 | Median ms |
|---|---|---:|---:|---:|---:|
| 1000 | libaom 0 | 30,162 | 1.2273 | 72.812 | 5741.9 |
| 1000 | zenav1-aom 0 | 30,162 | 1.2273 | 72.812 | 13365.2 |
| 1000 | SVT -1 native | 27,489 | 1.1185 | 71.084 | 6235.6 |
| 1000 | SVT -1 + unit search | 27,489 | 1.1185 | 71.084 | 6393.7 |
| 8100 | libaom 0 | 15,765 | 0.7698 | 85.179 | 4005.1 |
| 8100 | zenav1-aom 0 | 17,170 | 0.8384 | 84.267 | 3466.7 |
| 8100 | SVT -1 native | 14,327 | 0.6996 | 84.057 | 7236.9 |
| 8100 | SVT -1 + unit search | 14,327 | 0.6996 | 84.057 | 7240.2 |

The only changed SVT payloads were on photo 1000:

| QP | Native bytes | Search bytes | Native SSIM2 | Search SSIM2 | Native ms | Search ms |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 60285 | 60299 | 78.5239 | 78.5229 | 3484.7 | 3633.7 |
| 12 | 38365 | 38370 | 75.7707 | 75.7055 | 4065.3 | 4211.6 |

Photo overhead ranged from 1.91% to 4.28% across the six quantizers. At
QP20 it was 2.53%, with exactly the same payload and score. Screenshot medians
differed by less than 0.05% while search was bypassed; these small differences
are not evidence of an optimization.

## Matched quality under common time budgets

These are log-linear estimates bracketed by measured QP20 and QP32, without
extrapolation. They do not establish crossovers against faster presets, which
are absent from this isolated ablation.

| Source / target SSIM2 | Arm | Estimated bytes | Estimated ms |
|---|---|---:|---:|
| 1000 / 70 | libaom 0 | 26,808 | 5,518 |
| 1000 / 70 | zenav1-aom 0 | 26,808 | 12,811 |
| 1000 / 70 | SVT -1 native | 26,333 | 6,335 |
| 1000 / 70 | SVT -1 + unit search | 26,333 | 6,493 |
| 8100 / 80 | libaom 0 | 11,259 | 3,803 |
| 8100 / 80 | zenav1-aom 0 | 12,705 | 3,139 |
| 8100 / 80 | SVT -1 native | 11,434 | 7,586 |
| 8100 / 80 | SVT -1 + unit search | 11,434 | 7,589 |

On the photo at SSIMULACRA2 70, libaom fits a 6-second budget while neither SVT -1
arm does. At 6.5 seconds, native SVT offers the smallest estimated payload
among these arms. On the screenshot at SSIMULACRA2 80, Rust AOM fits 3.5 seconds;
at 4 seconds libaom offers fewer bytes. At 8 seconds native SVT fits but still
uses more bytes than libaom. These are candidates for direct crossover checks,
not production routing decisions or portable time guarantees.

## Source, policy and verification

- Canonical imazen-26 commit `187fbf338ce08e8e6654db7f04ddae58d5263da2`.
  Origins 1000 and 8100 are both training images. No held-out outcomes were used.
- Photo 4032×3024 →512×384; screenshot 1440×900 →512×320, Lanczos3. Both
  use identical BT709 limited-range 8-bit 420 preparation. Conversion-only
  ceilings are 80.217 and 87.927. This is not native HDR coverage.
- Explicit SVT reference: mainline 4.2.0 `9292ec8e32bce26f781f277ec8739b53426c4300`.
  The linked hybrid C-SVT arm is intentionally absent from this comparison.
- Native preset -1 remains the anchor. `aom-restoration-unit-search-v1` searches
  legal 256/128/64 units with shared plane size, SVT filter/RD costs and frame
  signaling cost. Intra-edge is off in every arm. Default tune/SCM is retained.
- Each job holds all arms/repeats for one source on one worker, with one reserved
  core and codec/RAYON/OMP threads 1. Separate CPU/worker cohort hashes are retained.
- Timed fully static comparator: `e9538ef62986cf4605dedbbe059728ae1c3f6e9b123a3ec787e811a82e6b4d19`.
  Later untimed telemetry verifier: `b0199023c45d841a4945ebc7f4b40dc62ccd4c364db6428fdd7f691e51acdc55`.
  The verifier must reproduce the original bytes; it does not replace timings.
- Local gates for the encoder change: 2,627 workspace tests, 136 regression checks,
  1,100 hybrid normal8 C byte comparisons and 320 pristine research comparisons
  (160 each native 8/10-bit). Ten enabled reconstruction cases include odd/tile/SB128
  boundaries; a further public-wrapper equality/refusal check passed.
  These counts do not close the older real-image parity or policy/routing gaps.

Full tables: [cells](imazen26_restoration_cells.tsv),
[matched brackets](imazen26_restoration_matched.tsv),
[photo budgets](imazen26_restoration_photo_budgets.tsv),
[screenshot budgets](imazen26_restoration_screen_budgets.tsv).
Raw output blobs, exact source snapshots, binaries, manifests, ledgers and replay
evidence are referenced in [the provenance record](imazen26_restoration_provenance.json).
Thirty overlapping native SVT/libaom/Rust AOM control cells also reproduced
the earlier corrected intra-edge run's exact OBU hashes across the different
binaries and CPUs. This does not authorize pooling their timings.

This two-origin diagnostic is not the requested minimum encoding-behavior set.
Full-corpus RD/RD-speed scouting, held-out validation, native HDR/format coverage,
continuous effort/strict parity and actual zenavif routing remain required.
