# CUDA constructor and first evaluation

Missing: no native GPU cold comparison or accepted regression is established
by these adapters. Original-call controls qualify reference wiring separately
from native parity, runtime cleanliness and timings. GPU maps remain separate.

`just bench-iqa-canonical-gpu-first PLAN NEW_OUTPUT --blocks 3` runs explicit
CUDA scalar first calls. Each alternating block acquires both host locks and
a 12 GiB resource scope. Give the outer controller no heavy lock. The plan
must bind every helper, source contract, raw RGB input and executable, including
this controller and `native_gpu_first_cell.py`. The original CPU controllers
retain their GPU refusal and hidden-device behavior.

The GPU adapter forwards `gpu_runtime=cuda` to the unchanged native worker
and requires that value in its returned cold observation. The controller
admits only the five enabled GPU metric names. A reviewed source contract must
declare constructor, parameter, runtime and teardown boundaries. No numerical
bound is assigned or changed; null bounds remain numerically unqualified.

The [retained reference source](canonical_cuda_first_reference_20261020.py)
is an exact copy of the local candidate tested against the original CUDA
IW-SSIM and CVVDP wrappers. Four 512×512 CID22 cases at one and 28 threads
passed original/candidate/candidate/original score-bit controls in fresh
processes. The wrapper recorded 56 s, 1.61 GiB peak RSS, minimum available
13133 MiB and peak load 6.52. This is reference-wiring evidence, with no native
Rust worker execution in those controls. The
[control receipt](../canonical_recheck_2026-10-19/cuda-first-reference-controls.json)
retains the source/result hashes and explicit coverage without publishing
the reference scores.

Raw RGB and immutable CVVDP JSON text are loaded before clocks. JSON parsing,
scorer construction, first CUDA context use and pair evaluation remain inside
the original callable; scalar `.item()` completes before elapsed capture.
The scorer remains alive past the clock. The reference requires PyTorch's CUDA
context to be uninitialized immediately before the first call; imported author
modules may query device availability. This does not promise an uninitialized
driver or empty OS/JIT caches. Device metadata and eigenvalue reality checks
follow timing. Subsequent SAMPLE rows support controls, not first-call ratios.

The retained source keeps its historical local harness import path. All package
and author-source paths remain explicit command arguments; the adapter does not
download a reference or select a substitute implementation.

Run `just bench-iqa-canonical-gpu-first-check` under shared heavy admission with
explicit disk-backed `TMPDIR`. Its synthetic protocol controls launch no scorer
or CUDA process; passing them does not acquire a timing observation.
