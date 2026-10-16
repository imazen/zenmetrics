# Timing and qualification boundaries

This document describes the adapters. Completed coverage is recorded separately;
source availability and a successful build do not establish benchmark completion.

| Adapter | Resident input | Excluded setup | Timed operation |
|---|---|---|---|
| Rust CLI CPU | RGB8 bytes | file read, initial model/scorer warmup | CLI metric call, internal color conversion, output release |
| Rust native maps | RGB8 or native plane images | file read, scorer construction | full pair evaluate with native maps and output release |
| Rust CUDA/WGPU | RGB8 bytes | device instance/buffer allocation, initial JIT warmup | full pair compute, internal upload/conversion, reduction/readback; map readback when offered |
| libvmaf | u8 YUV/gray planes | file read, RGB component conversion, context/model/feature registration, picture allocation/fill | submit pictures, extract features, flush, predict score |
| libjxl | normalized encoded-sRGB ImageBundle | raw file read/unpack, metadata and worker pool | ComputeScore or ComputeSSIMULACRA2, internal color conversion and output release |
| numpy/PyTorch | ndarray or normalized tensors on selected device | import, raw read, normalization/upload, model construction | public metric call, internal conversions, synchronization, scalar extraction/output release |

The RGB-to-native-buffer preparation differs between APIs and is stated in each
row. The benchmark follows each implementation's native input type rather than
adding an artificial shared input conversion inside its call. GPU instances
retain buffers, without caching reference features. CUDA timing includes the
synchronous readback needed to return a scalar, not only kernel submission.

Rust uses interleaved zenbench arms, warmup and at least ten rounds. Percentiles
are over round batch means; iteration counts and raw rounds are retained.
Other adapters use one warmup and ten individual timed calls, retaining every
sample. All cells have an outer process timeout below twenty minutes and a
host memory limit. Timeout or memory refusal is an explicit row, not a timing.
Admission is separate from measurement: a high-load refusal is retained and
retried after releasing the heavy-job lock.

The VMAF snapshot adapter caches immutable parsed model parameters. Its owned
clones remain timed. The original parser and every scoring expression remain
unchanged; the committed bitwise receipt covers scalar and native-map outputs.
PIQ BRISQUE also reads SVM parameters through its loader on every call. Its
benchmark loader caches the deserialized values and returns fresh tensor clones,
preserving the public function's in-place support-vector transpose. Each child
compares original-loader and cached-loader score bits before timing. Neither
adapter modifies a shipped crate or an installed oracle environment.

Every competitor compares its headline with the same-pair Rust score. Existing
parity gates determine qualification; exact comparison is used when no applicable
cross-version bound is documented. Bound values and their source are recorded.
For Butteraugli C++, the pinned crate's reference_parity.rs uses a 0.001 relative
gate. CUDA mismatches repeat a complete cell and retain both attempts. A residual
that reproduces remains a variant mismatch and is graphed separately. A variant
comparison is not evidence of matching algorithm semantics; the libvmaf VIF and
PSNR-HVS chroma conventions are explicitly labeled.

The Python package APIs and conventions are documented by their projects:
[PIQ functions](https://piq.readthedocs.io/en/latest/functions.html),
[scikit-image metrics](https://scikit-image.org/docs/stable/api/skimage.metrics.html),
[TorchMetrics SSIM](https://lightning.ai/docs/torchmetrics/latest/image/structural_similarity),
[pytorch-msssim](https://github.com/VainF/pytorch-msssim),
[LPIPS](https://github.com/richzhang/PerceptualSimilarity),
[ST-LPIPS](https://github.com/abhijay9/ShiftTolerant-LPIPS),
[DeepDC](https://github.com/h4nwei/DeepDC),
[FLIP](https://github.com/NVlabs/flip), and
[ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP).

Third-party code is confined to this benchmark harness and external tools.
No aggregate IQA package is installed or imported by these adapters. The legacy
CLI metric names containing a package name are identifiers of the existing Rust
implementations; they do not cause a Python dependency to load.
