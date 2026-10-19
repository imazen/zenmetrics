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
