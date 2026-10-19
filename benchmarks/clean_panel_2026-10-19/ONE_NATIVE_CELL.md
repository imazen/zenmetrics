# New native Butteraugli max cohort — build preparation only

Owner selects `butteraugli_max`, 512x512, one thread, score only. The Rust
emitted column must be bound explicitly to its frozen dispatch/source/config.
C++ uses the independently resolved float maximum over diffmap; each
ComputeScoreImpl call invokes SetReferenceImage then CompareWith. Unknown
historical rejected-batch column identity is an exclusion, not a new-cohort gate.
No old row or executable is retroactively rebound.

Reviewed adapter content: a52d8e32e747dc9f5e4be2a6bfb8215069337aba.
Expected C++ source SHA256:
0633dd2017517483d9825c1d9294180fdb12cf91867cd83fc15643ff275bb574.
Final landed commit and exact three-file equivalence remain pending. Retained
0845/de04cd control binary is not a cohort binary. Config report review is
separate and pending; source similarity does not establish numerical equivalence.

Ordered build procedure, with no scorer execution:

1. Resolve final remote commit, verify reviewed adapter/control/doc hashes and
   bind exact Rust baseline revision, dispatch to butteraugli_max, crate revision,
   feature resolution, sibling pins and benchmark-only overlays.
2. Freeze paired RGB8 input hashes, dimensions, packed/strided layout and color
   metadata; declare rgb-input lifecycle boundaries and explicit scope identifier
   in both executable protocols. Raw loading and reusable pool remain outside;
   metadata, both bundles/conversions, per-call comparator and destruction inside.
3. Create fresh isolated target-compatible pinned snapshots. Admit using both
   existing locks in order, then run-heavy. Locked dependency validation precedes
   compilation; refuse lock changes rather than regenerate. Build both binaries
   fresh, retaining exact commands, compiler/linker/header/library hashes, features,
   overlays, logs and output hashes. No SCORE, warmup, parity or pilot invocation.
4. Bind transitive ELF closure and loader config for both executables, plus used
   Python controller dependencies. Do not introduce unrelated PyIQA/Triton roots;
   retain the failed Python cohort independently. Compile/link and failed-input
   nonmetric protocol controls cannot qualify a numerical path.

Remaining gates: independent complete cohort/scope/protocol/closure review;
separate generic diagnostic approval/full controls; actual same-input numerical
parity at unchanged bounds; real scorer-path gate/ownership evidence; defensible
predeclared order/selection/dependence model and strict precision semantics.
FR emits one scalar; BRISQUE paired-NR is not a dependency. No timings/scoring,
qualification, acquisition, push or landing. Coverage remains 0/19.
