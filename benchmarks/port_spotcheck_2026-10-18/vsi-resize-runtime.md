# VSI reference resize control

Missing: full reference-runtime qualification and matched speed ratios.
The initial focused control below
covers **4 of 4 selected 64×64 pairs** at
factor 1 and one thread; it establishes no broader matrix result.

The photo finding changes when the reference's resize dependency changes.
The unchanged author `VSI.m` gives 0.94679144074028154 with system image
2.12.0, but 0.95679610895892431 with the unchanged upstream image-2.20.1
resize implementation. Frozen Rust gives 0.9567964257437082: the latter
absolute difference is **3.1678478384922215e-7**, below the unchanged 1e-4
bound. Rust source and inputs are identical across these comparisons.

Three integer-only procedural low-contrast patterns also fail against the
system 2.12 resize and pass with the explicit newer resize. Their largest
new difference is **1.7639143479186714e-8**. Exact values are in
[vsi-resize-runtime.tsv](vsi-resize-runtime.tsv); commands, hashes and process
memory receipts are bound by [vsi-resize-runtime.meta](vsi-resize-runtime.meta).
Neither score expectations nor a metric implementation changed.

This is a dependency-semantics control, not a full package installation.
The source archive is the hash-pinned official
[image 2.20.1 release](https://gnu-octave.github.io/packages/image/).
Only `imresize.m` and its private `interp_method.m` are copied, byte-for-byte,
into a fresh explicit overlay. Other image functions still resolve from system
image 2.12.0. The available executable is Octave 6.4.0; the complete image
2.20.1 package requires Octave 8.1.0. The selected functions execute on this
runtime, but this does not establish compatibility of the whole package.

The system 2.12 resize uses endpoint interpolation through `imremap`.
The selected newer implementation uses the antialiased, symmetric-padded,
normalized convolution described by this Rust port. Upstream's
[release notes](https://octave.sourceforge.io/image/NEWS.html) record added
antialiasing and interpolation compatibility changes in 2.14, followed by
weight normalization in 2.18. This dependency therefore needs an explicit
source binding; a shared `VSI.m` hash alone does not establish equal behavior.

Separate diagnostic copies round Lab outputs, use single-precision FFTs, or
round resize outputs. On the procedural inputs these stay near their
respective unmodified reference scores. Those counterfactuals motivated the
resize dependency check; they are modified diagnostic functions and never
replace the primary reference or its acceptance bound.

The initial 2.12 panel and its failures remain retained. Its measured
differences alone do not establish a Rust correctness defect against the
claimed resize convention.

The separate fixed-factor panel is now complete at factors 1–4, all three
retained content pairs, 64×64 / 255×257 / 1024×1024 / 3840×2160, and thread
budgets 1 and 28. Frozen Rust remains revision `e2d414c3`:

| Reference cohort | Cases | Numerical outcome | Maximum absolute difference | Unchanged gate |
|---|---:|---|---:|---:|
| FSIM/FSIMc author reference, retained system runner | 96 | 96 pass; both scalar columns checked | 1.329927450832713e-6 | 5e-5 |
| VSI author reference, explicit image-2.20.1 resize M-files | 96 | 96 pass | 3.1678478384922215e-7 | 1e-4 |
| VSI author reference, retained image-2.12 runner | 96 | 82 pass, 14 fail | 0.010004985003426614 | 1e-4 |

The [retained system-runner index](runner-image212.json) and
[new VSI index](runner-image2201.json) bind every output shard and case record.
Each scalar row includes complete-process GNU-time peak RSS and both arm's
resource-wrapper receipt. [primary-fixed-panel.meta](primary-fixed-panel.meta)
binds raw records, complete commands, and the publication controls. Every
Rust thread-budget pair is bit-identical in these two completed panels.
The primary functions expose scalar outputs, so no map comparison is claimed.
The separate [automatic-factor panel](primary-automatic.json) now passes
48/48 cases: 24 FSIM/FSIMc cases (both scalar columns) and 24 VSI cases over
the same input and thread grid. It executes the registered original author
source, including factor 8 at 4K, and selects the unsuffixed Rust names.
Maximum differences remain 1.329927450832713e-6 for FSIM/FSIMc and
3.1678478384922215e-7 for VSI, within the same unchanged gates.
These numerical panels acquire **zero qualified timing ratios**.

Repeatable preparation and execution recipes live in
[oracle.just](../iqa_compete_2026-10-16/oracle.just):
`vsi-low-contrast-prepare`, `vsi-precision-prepare`, `vsi-precision-run`,
`image-resize-source-check`, and `octave-image-source-panel`.
Every scorer still runs under both host locks and the memory wrapper.
