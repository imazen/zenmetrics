# Frozen oracle identity and separate native proposals

BRISQUE's declared derivation oracle is PIQ 0.8.0, not PyIQA. Frozen
`crates/brisque/README.md`, `tests/golden_brisque.py`, `tests/parity.rs` and
`goldens_brisque.tsv` agree. The frozen header identifies torch 2.14.1+cu130 CPU
AVX2, one thread, Python 3.12.14, numpy 2.5.4, OpenCV 5.0.0, glibc 2.43 and
MKL 2024.2. The generator decodes BGR-u8, converts RGB, casts f32 without /255,
NCHW, then calls PIQ with kernel_size7, sigma7/6, data_range255, reduction=none.
The tensor is CPU; score is one scalar per image. PairedNoReference retains two
NR evaluations. No metric route is changed.

The frozen JND constant15.138428 is documented as median absolute reference–
distorted difference over 1008 KonJND pairs (JPEG504/BPG504). The parity test
uses JND*1e-3 =0.015138428 against170 PIQ goldens, including oracle error mapping.
This is the declared fixed gate and its derivation, not a new measurement. The
older JND note's bit-exact claim is historical; later numerical review documents
nonzero errors. We do not copy that historical exactness claim into this cohort.

The retained rejected pilot instead used PyIQA0.1.16, torch2.14.1, Python3.12.12,
paired RGB-u8→float/255 NCHW→model() calls. Its explicit model file SHA256 is
e5b048ef5325bae6c57965cf077fadfa5834fcefb02e13611b4ea2d36cf3c52c, equal to the
PIQ model SHA in frozen params.rs. Model-file equality does NOT establish equality
of defaults, preprocessing, feature extraction, reductions or output behavior.
The pilot used create_metric defaults beyond cpu/pretrained_model_path; their
full equivalence to PIQ is unverified. Its one distorted-pair residual passing
the numeric threshold does not qualify it as the declared oracle. No new PyIQA
execution, import or substitution-specific measurement follows.

A future cohort must be named `brisque-piq-0.8.0-cpu-paired-nr` and independently
bind interpreter, installed PIQ/torch sources and versions, model/cache, exact
kwargs/input ingress, both output keys, native/bytecode closures and a new
binary/observer boundary. Existing PyIQA manifests/rows remain historical and
separately named. The currently prepared matched-observer wiring is a prototype
for its explicitly named PyIQA route; it does not become PIQ by changing a label.
PIQ execution adapter and numerical admission remain pending; no gate changed.

## Native subset of SAME19

Supervisor findings identify exactly butteraugli-original, butteraugli and ssim2.
The source-only contract in native_contract.py declares their separate executable
root, source/build/features/overlays/toolchain/input and loader bindings. This is
not a filter applied to the failed Python environment. It does not inherit PyIQA
or Triton scorer dependencies. The controller and supervisor still need their own
bound runtime/helper identities; their cost is outside metric timing.

For butteraugli-original the cited source is butteraugli_orig_canonical.cc:
to_planes, f64 sRGB LUT conversion and ButteraugliInterface are timed; Rust's
frozen in-tree71b18b6 path times f32 LUT/planes/compute/max. Exact declared bound0
is retained but parity/config/version/source-to-binary mapping remain unqualified.
For butteraugli and ssim2 the cited libjxl_compute.cpp uses the same iqa-jxl
shim; BufferToImageBundle x2 and metadata/pool/comparator setup are untimed.
Resident-RGB Rust calls include ingress. No negligible-cost waiver applies.
Butteraugli's ComputeScore norm versus Rust max remains unverified; declared
relative0.001 is retained, not validated. SSIM2 exact0 is retained, score-only.
No norm or metric route edits are proposed from empirical closeness.

Synthetic native dependency tests start with only the named executable, follow
an injected transitive dependency graph and reject mutated artifacts/unresolved
nodes. They make no installed-loader or compiled-shim claim. Proposal admission
and scope qualification stay false. Source-only plan is ready for review; actual
native closure, matched ingress/lifecycle adapters and numeric controls remain
future separately authorized gates.

Supervisor current binary hashes/pins are observations, not historical row
bindings and not necessarily the frozen73c04e41 cohort. We retain the corrected
supervisor report privately and do not adopt its broad claims or route-changing
suggestions. No Triton census or duplicate adapter audit was performed.
