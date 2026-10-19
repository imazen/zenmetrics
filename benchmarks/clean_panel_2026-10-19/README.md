# Explicit paired timing pilot

This replaces sample-index pairing with an explicitly identified acquisition block.
Each bounded block holds both host admission locks, then runs one Rust arm and one canonical
arm, releases the locks, and checks an output-directory STOP file before the next block. Order alternates by block. Each arm retains all ten internal observations;
the block ratio is the Rust median divided by the canonical median. The estimator
is the median of these independent block ratios. Inner samples are never paired.

The pilot command is deliberately limited to four blocks and never qualifies a
speed claim. Independent review must cover the first pilot before a production
panel or public landing. The fixed-look percentile-bootstrap implementation is
available for review; repeated optional stopping is not qualified by this pilot.

Run `just clean-panel-check` for the negative controls. `panel.py MANIFEST OUTPUT`
creates a new directory, verifies all bound files before and after each block,
retains failed/contaminated arms, and records the exact rejection reason. It calls
scorers directly beneath the existing contention supervisor and resource wrapper;
children do not acquire the experiment locks again. A failed sample rejects the
whole block. A source/build/input/scope mismatch or parity failure refuses a ratio.

## First cell: BRISQUE

The CLI computes both reference and distorted NR scores. The pilot canonical
adapter does the same, including RGB-to-tensor conversion within the timed call.
Input file loading and immutable model loading occur before timing on both sides.
No metric source or scalar arithmetic changes. The dated baseline build uses
current-master source, pinned siblings and existing benchmark-only preload/map
adapters; features and every adapter receipt belong in its provenance bundle.
This is a new baseline, not a relabeling of a historical executable.

The remaining eighteen cells are not admitted by the pilot. In particular,
libvmaf's preconverted-plane boundary and mDCTPSNR's process-per-sample boundary
need matching benchmark-only adapters before comparison. A missing adapter is a
blocker, never a reason to reuse a mismatched timing or waive ingress cost.

Private manifests may include machine-specific paths. Public evidence must use
content hashes and relative paths, with large raw artifacts stored outside git.

## First pilot receipt

New baseline source: `73c04e413fddc0e95e66b4147785910253772192`.
Executable SHA256: `6692a32d5d3d226574b5002068ca062881b1e16f7bc0c06040d1254808c682b0`.
One explicit A/B block completed; both arms exited zero and both outer contention
checks passed. Every native inner sample (10/10) failed the unchanged
`HeavyProcesses(4)` gate. The block was rejected, and the remaining three planned
pilot blocks were cancelled at the next block boundary. No timing ratio or CI is
qualified; the panel remains 0/19. The source of the gate's four-process count has
not been identified. The independent outer check does not override that gate.

Native BRISQUE was -12.43145751953125; canonical was -12.43414306640625.
Absolute difference 0.002685546875 passes the unchanged 0.015138428 scalar bound.
This is scalar parity for this pair, not timing qualification or bit identity.

Raw evidence is retained outside git at
`/mnt/v/aic2026-evidence-private/clean-panel-2026-10-19/`.
Its 33-file `SHA256SUMS` has SHA256
`0d7d33cd5019d0b49c5ef81105d2b3a07c774669d1e333e38387ca761c7e9833`.
The private manifest includes runtime/source/input/model identities; its SHA256 is
`b920522aa3d5b376f1b19716a9e3f9d546315f8c643bac6db84c7188099321bf`.
The completed block receipt SHA256 is
`8a67ea5fba3b7ca99bf2eb198c21a9f9a40da0197746ad214864f537f7e660f6`.

Build wrapper: rc=0, 115s, peak-RSS 0.87GiB, min-avail 12037MiB,
peak-load 6.75 (10GiB cap). Native pilot: rc=0, 2s, peak-RSS 0.10GiB,
min-avail 13661MiB, peak-load 2.48. Canonical pilot: rc=0, 4s,
peak-RSS 0.87GiB, min-avail 13174MiB, peak-load 2.53.
These are wrapper resource receipts, not qualified throughput claims.

Independent review is required before landing or expanding acquisition. Review
must address provenance coverage, equivalent timing scope, rejection behavior,
and the unresolved native gate. A production sequential inference procedure has
not been qualified by the pilot's descriptive fixed-look bootstrap.
