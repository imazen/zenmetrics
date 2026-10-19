# New native Butteraugli max cohort — build preparation only

Owner selects `butteraugli_max`, 512x512, one thread, score only. The Rust
emitted column must be bound explicitly to its frozen dispatch/source/config.
C++ uses the independently resolved float maximum over diffmap; each
ComputeScoreImpl call invokes SetReferenceImage then CompareWith. Historical Rust max-column identity is receipt-resolved; the historical
canonical executable SHA remains unbound. Current d500 cannot retrobind it.
No old row or executable is retroactively rebound.

Reviewed adapter content: a52d8e32e747dc9f5e4be2a6bfb8215069337aba.
Expected C++ source SHA256:
0633dd2017517483d9825c1d9294180fdb12cf91867cd83fc15643ff275bb574.
Final landed commit: 62aaf7e94acaee2ccbe9b4f6c16f0ec3416a3559.
All three Git blobs independently checked equal the coordinator-provided IDs. Retained
0845/de04cd control binary is not a cohort binary. CMS build/callgraph audit is supervisor-owned; runtime path selection is
unobserved (normal-init fastpath and conditional LCMS fallback are source facts); source similarity does not establish numerical equivalence.

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

## Fresh preparation status

Frozen Rust base: 73c04e413fddc0e95e66b4147785910253772192, not oldopt.
Fresh sibling exports from exact ci/sibling-pins.tsv revisions pass
`cargo metadata --locked --offline`; Cargo.lock remains unchanged. The
benchmark-only overlay selects the returned butteraugli_max column by name,
limits this cohort to 512x512/1t/score, and emits literal rgb-input scope plus
scalar name. Scope declaration does not establish matched cost or parity.
Metric crate arithmetic remains unchanged. No NR/NN feature is requested.

Fresh copied libjxl source and packed reference/distorted input files have
individual hashes. Compilation has not run: the new production wrapper was
independently rejected, so further production use of that wrapper stopped.
Its in-flight source staging completed and retained reap/unlock evidence.
Native build preparation remains independent of its bounded repair.
