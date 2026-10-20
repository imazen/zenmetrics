"""Source review notes; data only. No metric code is imported or executed."""

CLI = 'crates/zenmetrics-cli/src/metrics/'
API = 'crates/zenmetrics-api/src/'
BENCH = 'benchmarks/iqa_compete_2026-10-16/'

COMMON = {
    'native_boundary': 'EA iqa_compete.rs:98-149,345-382,476-507: resident RGB files and closure/display setup outside; untimed initial score and zenbench warmup/calibration before samples. b.iter clocks the whole run_metric_display CPU callback, scalar extraction, error handling and temporary destruction. No reference-image feature cache is selected. Native warm stage fixes ten rounds; reference requests its explicit samples count. These are different sampling protocols, not a boundary or numerical proof.',
    'native_thread_budget': 'Consumer writes requested 1/8 into native config and RAYON_NUM_THREADS; native global Rayon pool is primed before clocks and row records rayon_threads_observed. metrics-parallel feature enabled. A configured budget does not imply every kernel uses eight workers.',
    'canonical_outside': 'Raw RGB file reads, imports, script construction and fixed configuration outside sample clocks. One COLD/prime call precedes requested SAMPLE calls unless stated. No decoded-image or encoded-file timing is claimed.',
    'destruction': 'Function-local pixel/tensor/map/scratch temporaries are released before the wrapper stops its sample clock; immutable runtime/package state and input arrays remain resident. Python reference-count destruction does not imply a forced cyclic GC or OS allocator trim.',
    'thread_environment': 'compare_canonical.py run_arm supplies RAYON_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS and OPENBLAS_NUM_THREADS from the exact case budget before child launch. Every selected argv contains literal {threads}. No requirement for an existing t1/t8 row is added when the callable templates it.',
    'audit_limit': 'Source review only. No scorer, build, model import, CI, runtime worker census or numerical recheck was run. Existing bounds and observations remain unchanged. Null bound means unassigned, not automatic numerical acceptance.',
}

PROFILES = {
    'pyiqa': {
        'reference_kind': 'Alternative PyIQA implementation with explicitly hash-bound fixed-factor ssim function; not Wang original MATLAB.',
        'native_sources': [CLI+'mod.rs', CLI+'ssim_pyiqa.rs'],
        'native_call': 'metrics/mod.rs:2967-2981 -> ssim_pyiqa.rs:16-62,116-143: RGB8 -> owned f64 unrounded .299/.587/.114 luma, per-scoring-view fixed-factor pool, five-level twofold pyramid and reductions all timed. The f32 name denotes unrounded ingress here; actual adapter/native arithmetic is double.',
        'canonical_call': 'canonical_cpu_cell.py:80-102,107-114: fixed-manifest check and AST installation outside clocks. Each call copies RGB, permutes/contiguates tensors, casts to double, forms unrounded .299/.587/.114 luma, then ms_ssim(data_range=255,downsample=True). PyIQA ssim_arch.py:30-67,126-181 allocates window/weights and pools original per-level views inside call. Local tensors release on return.',
        'parameters': 'No learned model or spectral parameters. Fixed function source is installed before prime; Gaussian/window and weights arithmetic remains in the timed functions. Native Gaussian/pyramid preparation remains timed.',
        'library_threads': 'torch.set_num_threads(1/8), interop=1 and no_grad before calls. Native fixed-factor code has no Rayon dispatch in this file; configured eight is a ceiling. Effective Torch operator concurrency was not independently measured.',
        'limits': '1152x1152 selected to preserve all five valid 11x11 scoring windows even at k6. Earlier too-small refusals are not bypassed. Fixed source is a declared controlled variant, not unchanged upstream ssim.',
        'entry_names': ['canonical_cpu_cell.py', 'check_fixed_factor_oracle.py'],
    },
    'vmaf-model': {
        'reference_kind': 'Original libvmaf C library with owned resident-RGB C++ wrapper; model convention explicit in command.',
        'native_sources': [CLI+'mod.rs', CLI+'vmaf.rs', 'crates/vmaf/src/lib.rs', 'crates/vmaf/src/v0.rs', 'crates/vmaf/src/score.rs', 'crates/vmaf/src/v0_score.rs'],
        'native_call': 'metrics/vmaf.rs:37-86,107-172: convert both RGB8 images to even-cropped limited-range YUV420 and widen u8 to u16 inside callback; score one frame. v0_score.rs:332-340 or score.rs:402-410 constructs/destroys a scorer per call.',
        'canonical_call': 'libvmaf_rgb_compare.cpp:96-129: raw reads and vmaf_model_load outside warm loop. Clock109-126 includes both RGB->YUV420 conversions, picture allocations/copies, VmafContext init with features, frame submit/flush, scalar score, close and vector/RAII destruction. One COLD then N SAMPLE calls.',
        'parameters': 'ASYMMETRIC: canonical VmafModel load at97-103/destruction129 outside all sample clocks. Native constructor calls JSON/SVM parsers on every timed call (v0.rs:61-95; lib.rs:171-205). Embedded native files avoid filesystem I/O but do not avoid parsing. Warm priming does not remove this work.',
        'library_threads': 'C++ VmafConfiguration.n_threads=1/8 at115 controls libvmaf workers; actual frame/feature scheduling may use less. Native global Rayon uses feature scopes in score.rs:110 and v0_score.rs:117; RGB conversion remains serial at512 below1<<20. No new thread census.',
        'limits': 'VMAF-MODEL-PREP: model lifetime/preparation differs. Retain scores and describe scope; source alone supplies no timing effect or exception approval. Selected model names distinguish0.6.1,NEG,4K and1.0.16. Feature-only libvmaf routes excluded from this finding.',
        'entry_names': ['libvmaf_rgb_compare.cpp'],
    },
    'vmaf-feature': {
        'reference_kind': 'Original libvmaf feature extractor via owned resident-RGB C++ wrapper.',
        'native_sources': [CLI+'mod.rs', CLI+'vmaf.rs', CLI+'classical.rs', 'crates/msssim/src/libvmaf.rs', 'crates/vmaf/src/cambi.rs'],
        'native_call': 'metrics/mod.rs selects float_ssim/float_ms_ssim over studio601-f32 rounded Y, limited709 Y PSNR, or distortion-only CAMBI. Conversion, feature scratch and scalar reduction are timed. CAMBI ordinary ingress calls to_yuv420_even (including unused chroma allocation), whereas AIC integer ingress builds only Y (vmaf.rs:248-277).',
        'canonical_call': 'libvmaf_rgb_compare.cpp:88-129: mono YUV400 feature path; both luma planes converted inside clock except CAMBI, which converts distorted only and copies it into two VmafPictures. Context init/feature registration/submit/flush/read/close and buffers release all timed. No VmafModel load for these routes.',
        'parameters': 'No VmafModel resident asymmetry on feature-only path. Constants and library startup primed by COLD. Gaussian/feature contexts still prepared in each call. Reference output pictures and native internal buffers differ as normal adapter/API work.',
        'library_threads': 'C++ sets VmafConfiguration.n_threads=1/8 explicitly. Native kernels use global Rayon where enabled; CAMBI column path threshold512x512 (cambi.rs:5653) can use eight; other feature kernels/ingress have size gates. Worker-budget propagation verified in source, actual utilization not measured.',
        'limits': 'Declared per-route luma convention must be kept: 601-f32 for float SSIM/MS-SSIM,709 for Y PSNR,601-f64 or integer601 for CAMBI. CAMBI scores distorted only, not a two-image NR pair. Do not apply VMAF model finding to these rows.',
        'entry_names': ['libvmaf_rgb_compare.cpp'],
    },
    'octave': {
        'reference_kind': 'Original author M-file functions with owned RGB8 ingress wrapper; VIF-vector also binds PyrTools and oracle compatibility files.',
        'native_sources': [CLI+'mod.rs', 'crates/msssim/src/lib.rs', 'crates/msssim/src/kernel.rs', 'crates/vif/src/lib.rs', 'crates/vif/src/kernel.rs', 'crates/vif/src/vifvec.rs'],
        'native_call': 'mod.rs:3273-3324 -> msssim_rgb8/vif_rgb8/vifvec_rgb8; owned luma conversion, multiscale/pyramid scratch, core score and destruction inside callback. MSSSIM has f32 luma coefficients .2989/.587/.114; VIF uses its source-defined unrounded luma; VIFvec explicitly rounds .299/.587/.114. No sRGB EOTF.',
        'canonical_call': 'canonical_octave_cell.py:31-60: path/function checks, raw fread, level/weights setup outside sample clocks. reshape/permute RGB8, double luma conversion and original M call inside tic/toc. MSSSIM also constructs fspecial(gaussian11,1.5) inside; clear ref/dist before toc. One COLD then N SAMPLE calls.',
        'parameters': 'No learned/spectral files. Original MSSSIM/VIF construct Gaussian filters inside evaluator; VIFvec creates original steerable pyramid inside. MSSSIM level/weights configuration is outside canonical clock, native determines level inside; declare this small configuration placement without a timing estimate.',
        'library_threads': 'argv templates1/8; Octave wrapper does not call a thread-count API, inherits OMP/BLAS env. Generic adapter invokes both --no-gui and --no-window-system. Root coordinator already verified FFTW budgets1/8; source-only review adds no new runtime evidence. Native MSSSIM255x257 is below2^21 parallel threshold; VIF/VIFvec feature-gated Rayon kernels can dispatch.',
        'limits': 'Selected odd255x257 pair, not photo512. Numeric luma precision and function/library runtime behavior remain subject to original per-route bounds; names alone do not prove equivalent ingress. Original M function hash and spec path bound in each row.',
        'entry_names': ['canonical_octave_cell.py'],
    },
    'nr': {
        'reference_kind': 'Alternative PIQ BRISQUE / pypiqe / BasicSR NIQE implementations, not the authors original MATLAB.',
        'native_sources': [CLI+'mod.rs', CLI+'nr.rs', 'crates/piqe/src/lib.rs', 'crates/niqe/src/lib.rs', 'crates/brisque/src/lib.rs'],
        'native_call': 'metrics/nr.rs constructs stateless native scorer and evaluates both reference and distorted RGB8 inside callback; returns distorted plus reference scalar. Both image conversions/statistics and local scratch destruction timed. Normal current40 worker has no per-call model file loads (also verified by root).',
        'canonical_call': 'canonical_cpu_cell.py:40-72,107-114: BRISQUE copies CHW float RGB and evaluates r then d; PIQE makes BGR copies and calls pypiqe twice, whose COLOR_BGR2GRAY is inside. NIQE selected resident wrapper:63-82 evaluates both BGR copies through original BasicSR calculate_niqe. Each SAMPLE includes both NR image evaluations and conversion.',
        'parameters': 'BRISQUE torch.load outside clocks; exact support-vector clones returned by load_url shim inside each call because PIQ mutates its owned tensor. NIQE npz parameters read/unpacked before clock, exact-key shim returns owned raw-array copies during call; original features/resize unchanged. Native NR parameters embedded/stateless. PIQE has no model file.',
        'library_threads': 'Both selected Python wrappers set torch intraop1/8, interop1, gradient recording disabled; OpenBLAS/MKL/OMP inherited1/8. NumPy/SciPy/OpenCV operators may use different or fewer workers; no runtime census claimed. Native uses global Rayon through compiled metric features/size gates.',
        'limits': 'NIQE resident wrapper additionally evaluates original/default and cached variants and checks exact bits outside clocks before COLD; not a process-first-call measurement. BRISQUE clones/model parameter preparation differs from native embedded representation; no I/O asymmetry is claimed for these selected resident wrappers.',
        'entry_names': ['canonical_cpu_cell.py', 'canonical_niqe_resident_cell.py'],
    },
    'ssim2': {
        'reference_kind': 'Cloudinary primary SSIMULACRA2 v2.1 callable C++ wrapper.',
        'native_sources': [CLI+'mod.rs', API+'cpu_dispatch.rs'],
        'native_call': 'mod.rs:2683-2685 invokes umbrella CPU SSIMULACRA2 on resident RGB8; constructor, conversion/score and temporary destruction inside each callback. Dependency implementation is selected by frozen build/lock, not a GPU or prototype path.',
        'canonical_call': 'ssimulacra2_primary_compute.cpp:79-114: raw inputs, JxlThreadParallelRunner/ThreadPool outside; per clock ImageMetadata/ImageBundle, BufferToImageBundle RGB8 conversion, ComputeSSIMULACRA2 and Score inside RAII scope, destroyed before clock stop. One COLD then N SAMPLE calls.',
        'parameters': 'No learned/spectral files. Thread pool and CMS static infrastructure outside canonical sample; image representation and metric scratch inside. Native global pool primed before samples; no reference feature caching selected.',
        'library_threads': 'JxlThreadParallelRunner uses0 helpers/caller execution for budget1 and8 workers for budget8; runner waits at its barrier while workers process ingress. THREADS ingress is explicit; canonical ComputeSSIMULACRA2 has no pool argument and wrapper reports caller-only score. Native global Rayon budget1/8; actual utilization unmeasured. No oversubscription finding inferred.',
        'limits': 'Selected plan binds primary executable hash but does not bind this C++ wrapper/build receipt. Retained compile_commands names this path, without a source hash witness. Observed current wrapper boundary is conditional on that provenance; do not claim wrong binary or full source-to-binary proof. GPU SSIM2 level/scalar issues are outside these40 CPU routes.',
        'entry_names': ['ssimulacra2_primary_compute.cpp'],
    },
    'butter': {
        'reference_kind': 'Original libjxl0.12 comparator with explicit proposal p3 scalar transform; owned RGB wrapper.',
        'native_sources': [CLI+'mod.rs', API+'cpu_dispatch.rs'],
        'native_call': 'mod.rs:2248-2264,2859-2864: constructs CPU Butteraugli umbrella per call, computes maximum and pnorm3 n, then applies proposal max(0.00459792*n,2.0938201*n-1.1033102). Conversion/core scratch and destruction timed.',
        'canonical_call': 'libjxl_compute.cpp:54-58,65-75,98-114 selected rgb-input mode: resident byte reads, thread pool and immutable comparator params outside. Timed scope creates metadata/bundles and diffmap, performs both BufferToImageBundle conversions, constructs comparator, computes distance and pnorm3/proposal result; RAII destruction before stop. Iteration0 primes, only later samples emitted.',
        'parameters': 'Immutable Butteraugli parameter object outside canonical, per-image comparator and buffers inside. No spectral/model file reads. Native fixed config creation remains per callback; no precomputed reference in either selected score route.',
        'library_threads': 'ThreadPoolInternal outside canonical clock wraps JxlThreadParallelRunner with0 helpers/caller execution at budget1 and8 workers at budget8; runner waits at its worker barrier. Native global Rayon budget1/8; source review supplies no utilization census or oversubscription finding.',
        'limits': 'C++ wrapper hash is plan-bound. Original library linked binary/library provenance remains in original build records; this source audit does not rebuild or prove transitive compiler output. Proposal transform is explicit, not plain original max Butteraugli.',
        'entry_names': ['libjxl_compute.cpp'],
    },
    'ssim1': {
        'reference_kind': 'Original Cloudinary SSIMULACRA1 body inside a source/receipt-bound OpenCV RGB callable.',
        'native_sources': [CLI+'mod.rs', CLI+'ssimulacra1.rs'],
        'native_call': 'mod.rs:2911-2920 -> ssimulacra1.rs:18-64,286-374: RGB8->linear RGB LUT and Lab, pyramid/score/local buffers all inside. Up to six levels; reference not precomputed.',
        'canonical_call': 'Generated reference.cpp:194-249,259-370,382-387: raw files and cv::setNumThreads outside. RGB->BGR copy, original per-call LUT/EOTF/Lab conversion, six-level body and Mat destruction inside reference_score before elapsed. One COLD then N SAMPLE calls.',
        'parameters': 'Only fixed constants. LUT rebuilt inside both calls; configuration/static coefficients outside or constant-folded. No learned files.',
        'library_threads': 'cv::setNumThreads(1/8) before prime; native uses Rayon channels when size>=32768. OpenCV implementation decides actual parallel operators; no runtime worker census.',
        'limits': 'Bound build.json hashes original source, extracted body, generated wrapper and binary. This is stronger wrapper provenance than SSIM2/DSSIM selected plans. No new numerical validation is inferred.',
        'entry_names': [],
    },
    'dssim': {
        'reference_kind': 'Author dssim-core3.5.1 at0e44c9b7; earlier fork comparison remains separate.',
        'native_sources': [CLI+'mod.rs', API+'cpu_dispatch.rs'],
        'native_call': 'mod.rs:2212-2220 constructs Metric::new per call -> cpu_dispatch.rs:231-236 boxed Dssim::new. Helpers1495-1533 create both images, compare and destroy local images/metric within callback; reference-image preparation not cached.',
        'canonical_call': 'Current dssim_reference.c:28-41: dssim_new before clocks31-37, free after all samples41; create both images32-33, compare35 and free images36 inside. Raw RGB reads outside; iteration0 prime before SAMPLE calls.',
        'parameters': 'Configuration lifetime differs: canonical Dssim context retained; native constructed and dropped per callback. Pixel conversion/preparation remains timed both sides. No model-file I/O in the inspected author library path.',
        'library_threads': 'C harness accepts the threads argv position but does not read it or set a thread API. Consumer supplies RAYON_NUM_THREADS and OMP/BLAS env1/8 before launch. Native global Rayon budget1/8 and join for image pair. Author library effective thread control is not proven by this C source; no runtime census.',
        'limits': 'Selected author binary/build.sha256 omit C harness hash (build_dssim_reference.sh:15-16). Boundary read from current source lacks an independent wrapper-to-binary witness. This is a provenance limit plus context placement difference, not evidence of wrong binary or measured overhead.',
        'entry_names': ['dssim_reference.c', 'build_dssim_reference.sh'],
    },
    'psnr': {
        'reference_kind': 'Explicit skimage/NumPy convention oracle; weighted611 is a declared weighted-MSE formula, not an original-author package.',
        'native_sources': [CLI+'mod.rs', CLI+'classical.rs', CLI+'classical/psnr_kernels.rs'],
        'native_call': 'classical.rs:53-162 selected RGB/luma/chroma conversion, SSE/MSE and logarithm inside callback. RGB integer SSE; unrounded709/601 f64 Y; studio601 f32 rounded Y; Cb/Cr and611 full-resolution rounded studio planes. No transfer EOTF or chroma subsampling.',
        'canonical_call': 'canonical_classical_cell.py:66-93 converts RGB to requested plane and skimage PSNR inside score. Studio601 branch computes all three rounded codes then selects Y. canonical_psnr_planes_cell.py:23-35 computes requested full-resolution studio plane(s), MSE and PSNR inside;611 weights Y/Cb/Cr MSE6:1:1. Raw reads/constants outside. COLD then N samples.',
        'parameters': 'Only fixed coefficients/constants; no model or parameter files. Native fused conversion versus NumPy materialization/all-plane studio-Y conversion is explicit implementation work within clock, not missing ingress.',
        'library_threads': 'classical wrapper sets Torch1/8/interOp1 and cv2.setNumThreads1/8 even though PSNR uses NumPy/skimage. Planes wrapper parses argv budget but uses env only. Native PSNR parallel RGB gate2^21 exceeds selected512x512; ordered unrounded Y paths serial. No claim of eight-way work from the row label.',
        'limits': 'All scopes use code values; do not infer linear light. psnr-y is full-range unrounded709; psnr-y601 is unrounded601; psnr-y-studio601 rounded limited601. libvmaf Y PSNR belongs to its separate feature profile. No external numerical bound assigned in these selected rows.',
        'entry_names': ['canonical_classical_cell.py', 'canonical_psnr_planes_cell.py'],
    },
    'flip': {
        'reference_kind': 'Official NVIDIA FLIP evaluator package; declared f64 mean for flip and original package mean for flip-v17.',
        'native_sources': [CLI+'mod.rs', CLI+'flip.rs'],
        'native_call': 'flip.rs:127-173,471-560: per-call byte/EOTF LUT, image Lab/YCxCz ingress, filters/error map and mean inside. flip uses f64 reduction; flip-v17 retains source-defined f32 reduction. Map/scratch destroyed before callback returns.',
        'canonical_call': 'canonical_flip_cell.py:17-23: both RGB->f32/255 conversions, evaluate(LDR,inputsRGB=True,applyMagma=False,computeMeanError=False), then error.mean(dtype=float64) inside. flip-v17 canonical_cpu_cell.py:74-79 requests package computeMeanError=True and returns original scalar. Package flip_python_api.py:36-100 accepts arrays, directly calls nbflip.evaluate; no file image loader on these calls.',
        'parameters': 'No learned files. Package load outside; default viewing params constructed within evaluate; native fixed display and LUT work inside. Per-call maps/normalized inputs released inside wrapper score lifetime; package/runtime remains resident.',
        'library_threads': 'flip wrapper parses1/8 but does not pass a library thread API; inherited OMP env is the only control visible here. flip-v17 also configures Torch, which does not establish nbflip thread count. Native Rayon gate65536 permits512x512. C-extension thread cap not independently proven by inspected Python source.',
        'limits': 'Selected plans bind Python adapter but not imported flip_python_api.py/nbflip source. Observed package source is recorded as unbound; C-extension body/parallel control remains a source-proof limit. Do not conflate distinct reduction conventions or infer numeric failure.',
        'entry_names': ['canonical_flip_cell.py', 'canonical_cpu_cell.py'],
    },
    'wshv': {
        'reference_kind': 'Original IVPLab WS_HV notebook function extracted with imports by owned wrapper; no function arithmetic rewrite.',
        'native_sources': [CLI+'mod.rs', CLI+'ws_hv.rs'],
        'native_call': 'ws_hv.rs:215-280: owned RGB gray ingress, wavelet detail pyramids/resize, original pooling and buffer destruction inside callback. score_order swap=false selects RGB convention.',
        'canonical_call': 'canonical_classical_cell.py:49-55,90-93 parses notebook and installs original function/import definitions outside. score invokes IQA_WS_HV(r,d), including cv2.COLOR_RGB2GRAY, PyWavelets three-level decomposition, cubic resize, saliency and pooling inside. Raw input reads outside; COLD then N samples.',
        'parameters': 'Fixed constants; no model files or cached per-image features. AST extraction setup outside both metric clocks; original function-local arrays destroyed on return.',
        'library_threads': 'Wrapper explicitly Torch1/8/interOp1 and cv2.setNumThreads1/8; PyWavelets/NumPy inherit env but no eight-worker proof. Native Rayon join for the two detail pyramids at225; additional row/column gates. Requested caps are not identical worker-utilization claims.',
        'limits': 'Original function explicitly COLOR_RGB2GRAY despite notebook imread example BGR. Selection supplies RGB arrays; variable names alone are not evidence of an ingress mismatch. Selected scalar512 only; no map qualification.',
        'entry_names': ['canonical_classical_cell.py'],
    },
    'hdr-aic': {
        'reference_kind': 'Official HDR-VDP2/3 evaluators under the separate AIC200nit/37.84ppd adapter convention; not base sRGB-display.',
        'native_sources': [CLI+'mod.rs', CLI+'hdrvdp_aic.rs', 'crates/hdrvdp/src/metric.rs', 'crates/hdrvdp/src/spectral.rs', 'crates/hdrvdp/src/v3/metric.rs', 'crates/hdrvdp/src/v3/pathway.rs', 'crates/hdrvdp/src/v3/spectral.rs'],
        'native_call': 'hdrvdp_aic.rs:41-108: RGB8 normalization/EOTF*200; v3 additionally rounds to10bit and converts XYZ; parameters and original port call/scalar extraction/scratch release inside callback. v2 uses rgb-bt70937.84; v3 quality XYZ37.8425 returning Q. Embedded spectral CSV parse/interpolation inside every call.',
        'canonical_call': 'canonical_hdrvdp_cell.py:31-49: raw fread outside; RGB reshaping/double/255, EOTF*200, v3 round10bit/XYZ and official call inside tic/toc, clear inputs before stop. One COLD then N samples. Statistics/path/script setup outside. This older helper only specifies --no-gui; startup overhead outside samples is not metric timing.',
        'parameters': 'AIC-PARAM-IO: no spectral resident shim installed. Original load_spectral_resp.m:10 dlmread and:16-18 PCHIP run inside sample. Native parses/interpolates embedded strings without filesystem reads. Matched resident-parameter claim is not supported by this selected source.',
        'library_threads': 'Explicit argv1/8 plus inherited OMP/BLAS caps; original Octave code has no per-call thread parameter. Root FFTW budget verification is separate retained evidence. Native global Rayon feature/size gates; no new effective-thread measurement.',
        'limits': 'Retain separate AIC convention, all scores and null bounds. Do not borrow base100nit30ppd/Q_JOD resident controls. Storage/I/O asymmetry is concrete; no performance effect size inferred. Existing matched_scope flag is recorded, not endorsed by audit.',
        'entry_names': ['canonical_hdrvdp_cell.py'],
    },
    'hdr-base': {
        'reference_kind': 'Official HDR-VDP2/3 unchanged evaluator bodies, sealed exact-key interpolated-parameter fixture; base100nit/1black/30ppd, v3 Q_JOD.',
        'native_sources': [CLI+'mod.rs', CLI+'hdrvdp_aic.rs', 'crates/hdrvdp/src/metric.rs', 'crates/hdrvdp/src/spectral.rs', 'crates/hdrvdp/src/v3/metric.rs', 'crates/hdrvdp/src/v3/pathway.rs', 'crates/hdrvdp/src/v3/spectral.rs'],
        'native_call': 'hdrvdp_aic.rs:116-151 exact production score_srgb_display: RGB8 unit conversion and native v2/v3 sRGB-display call inside warm callback; params creation, EOTF/display conversion, embedded CSV parse/PCHIP and temporary release all timed. Both bodies are in current40 worker; no prototype build relied on.',
        'canonical_call': 'first_boundary_hdrvdp_resident_cell.py composes hash-bound setup/inputs/body/suffix, input fread and original-helper fixture preload before clocks. Preserved M body times whole RGB8 conversion and official100nit/1black/30ppd sRGB-display evaluator, v3 Q_JOD, then clear temporary arrays. One COLD prime and explicit SAMPLE count. Headless --no-gui --no-window-system.',
        'parameters': 'BASE-HDR-PARAM-PREP: resident.py:105-116 loads and caches already interpolated original-helper outputs before clocks, shim46-60 returns them. Native spectral.rs:68-101,141-173 and v3 spectral.rs:84-97,148-152 parse/resample every call. File I/O excluded but canonical interpolation also hoisted; preparation placement asymmetric.',
        'library_threads': 'Exact case budgets1/8 explicit through cell argv/environment and sealed predecessor cases. Root verified FFTW effective budgets1/8. Native global Rayon configured1/8, no new utilization claim. Source/body/array/runtime gates retained; runtime snapshots not repeated by lane.',
        'limits': 'Only photo512 warm1/8 qualified by retained controls, not full104/first. V2 unchanged absolute0.005; V3 null, native_parity_qualified remains false: observed delta1.7763568394002505e-15 is descriptive, not invented acceptance. Coordinator/root provisional judgment keeps these timings asymmetric descriptive/unqualified, outside strict acceptance; no human-approved exception. Raw-table successor is design only.',
        'entry_names': [],
    },
}


def profile_for(metric):
    if metric.startswith('ms-ssim-pyiqa-'): return 'pyiqa'
    if metric.startswith('vmaf'): return 'vmaf-model'
    if metric in ('ssim-libvmaf', 'msssim-libvmaf', 'psnr-y-libvmaf', 'cambi', 'cambi-aic2026'): return 'vmaf-feature'
    if metric in ('msssim', 'vif', 'vifvec'): return 'octave'
    if metric in ('piqe', 'niqe', 'brisque'): return 'nr'
    if metric == 'ssim2': return 'ssim2'
    if metric == 'proposal-butteraugli': return 'butter'
    if metric == 'ssimulacra1': return 'ssim1'
    if metric == 'dssim': return 'dssim'
    if metric.startswith('psnr'): return 'psnr'
    if metric.startswith('flip'): return 'flip'
    if metric == 'ws-hv': return 'wshv'
    if metric.endswith('-aic2026'): return 'hdr-aic'
    if metric in ('hdrvdp','hdrvdp3'): return 'hdr-base'
    raise ValueError(metric)
