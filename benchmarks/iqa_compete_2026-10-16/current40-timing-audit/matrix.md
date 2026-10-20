# Current40 warm timing source matrix

40/40 detailed selected-entrypoint reviews; 80/80 original comparative cases represented. This is source coverage, not 40 timing qualifications. Each linked row contains exact plan/case/argv and hashes, input bindings, native frozen source paths/hashes, canonical source hashes with binding status, both clocks, conversion, parameter/destruction/prime placement, thread propagation and limits.

Native worker: source `ea767ba1a4029963bc56a74fb824e64455e4599e`, SHA256 `19898e546917c9b62badcf1733ec74aacb3e02b4e14d88af1e31923a88a54005`. The [shared source registry](source-registry.json) records the warm harness and consumer. Inspected native files match the retained verified-build hashes. This audit does not reproduce the build or requalify its dependency closure.

All selected commands template `{threads}`; no hardcoded-command thread substitution was found. Requested1/8 is a budget, not a claim of eight-way utilization. In particular, DSSIM C does not read argv[5]; only the outer runner supplies Rayon environment caps. FLIP extension scheduling remains unverified.

| Route / complete row | Selected geometry | Canonical timed entry | Principal limit |
| --- | --- | --- | --- |
| [ms-ssim-pyiqa-f32-k6](matrix/ms-ssim-pyiqa-f32-k6.json) | 1152x1152 | `canonical_cpu_cell.py` | Fixed-factor PyIQA alternative; actual tensors/native f64;1152 valid window scope. |
| [ms-ssim-pyiqa-f32-k5](matrix/ms-ssim-pyiqa-f32-k5.json) | 1152x1152 | `canonical_cpu_cell.py` | Fixed-factor PyIQA alternative; actual tensors/native f64;1152 valid window scope. |
| [ms-ssim-pyiqa-f32-k4](matrix/ms-ssim-pyiqa-f32-k4.json) | 1152x1152 | `canonical_cpu_cell.py` | Fixed-factor PyIQA alternative; actual tensors/native f64;1152 valid window scope. |
| [ssim-libvmaf](matrix/ssim-libvmaf.json) | 512x512 | `libvmaf-rgb-reference-v2` | Per-feature luma conventions; context/representation work explicit; no VmafModel. |
| [ms-ssim-pyiqa-f32-k3](matrix/ms-ssim-pyiqa-f32-k3.json) | 1152x1152 | `canonical_cpu_cell.py` | Fixed-factor PyIQA alternative; actual tensors/native f64;1152 valid window scope. |
| [cambi-aic2026](matrix/cambi-aic2026.json) | 512x512 | `libvmaf-rgb-reference-v2` | Per-feature luma conventions; context/representation work explicit; no VmafModel. |
| [ms-ssim-pyiqa-f32-k2](matrix/ms-ssim-pyiqa-f32-k2.json) | 1152x1152 | `canonical_cpu_cell.py` | Fixed-factor PyIQA alternative; actual tensors/native f64;1152 valid window scope. |
| [cambi](matrix/cambi.json) | 512x512 | `libvmaf-rgb-reference-v2` | Per-feature luma conventions; context/representation work explicit; no VmafModel. |
| [msssim](matrix/msssim.json) | 255x257 | `canonical_octave_cell.py` | Odd255x257; original M files; inherited thread caps, runtime evidence separate. |
| [vmaf-v1](matrix/vmaf-v1.json) | 512x512 | `libvmaf-rgb-reference-v2` | Native model JSON/SVM preparation timed; original model preload excluded. |
| [dssim](matrix/dssim.json) | 512x512 | `dssim-reference` | Context lifetime differs; harness-to-binary proof absent; outer env controls budget. |
| [vmaf](matrix/vmaf.json) | 512x512 | `libvmaf-rgb-reference-v2` | Native model JSON/SVM preparation timed; original model preload excluded. |
| [vmaf-4k](matrix/vmaf-4k.json) | 512x512 | `libvmaf-rgb-reference-v2` | Native model JSON/SVM preparation timed; original model preload excluded. |
| [vmaf-neg](matrix/vmaf-neg.json) | 512x512 | `libvmaf-rgb-reference-v2` | Native model JSON/SVM preparation timed; original model preload excluded. |
| [vmaf-neg-studio601](matrix/vmaf-neg-studio601.json) | 512x512 | `libvmaf-rgb-reference-v2` | Native model JSON/SVM preparation timed; original model preload excluded. |
| [vmaf-studio601](matrix/vmaf-studio601.json) | 512x512 | `libvmaf-rgb-reference-v2` | Native model JSON/SVM preparation timed; original model preload excluded. |
| [msssim-libvmaf](matrix/msssim-libvmaf.json) | 512x512 | `libvmaf-rgb-reference-v2` | Per-feature luma conventions; context/representation work explicit; no VmafModel. |
| [vif](matrix/vif.json) | 255x257 | `canonical_octave_cell.py` | Odd255x257; original M files; inherited thread caps, runtime evidence separate. |
| [piqe](matrix/piqe.json) | 512x512 | `canonical_cpu_cell.py` | Alternative implementations; both images timed; resident parameter copies explicit. |
| [ssim2](matrix/ssim2.json) | 512x512 | `iqa-ssim2-primary` | Executable bound; C++ wrapper hash missing from selected plan; score caller-only. |
| [vifvec](matrix/vifvec.json) | 255x257 | `canonical_octave_cell.py` | Odd255x257; original M files; inherited thread caps, runtime evidence separate. |
| [niqe](matrix/niqe.json) | 512x512 | `canonical_niqe_resident_cell.py` | Alternative implementations; both images timed; resident parameter copies explicit. |
| [proposal-butteraugli](matrix/proposal-butteraugli.json) | 512x512 | `iqa-jxl` | Original p3 proposal transform; raw-RGB scope; static pool outside. |
| [ssimulacra1](matrix/ssimulacra1.json) | 512x512 | `ssimulacra1-rgb` | Original body and generated wrapper hashes linked through bound build receipt. |
| [brisque](matrix/brisque.json) | 512x512 | `canonical_cpu_cell.py` | Alternative implementations; both images timed; resident parameter copies explicit. |
| [hdrvdp-aic2026](matrix/hdrvdp-aic2026.json) | 512x512 | `canonical_hdrvdp_cell.py` | AIC200nit convention; canonical timed parameter-file reads versus native embedded tables. |
| [hdrvdp3-aic2026](matrix/hdrvdp3-aic2026.json) | 512x512 | `canonical_hdrvdp_cell.py` | AIC200nit convention; canonical timed parameter-file reads versus native embedded tables. |
| [psnr](matrix/psnr.json) | 512x512 | `canonical_classical_cell.py` | Explicit code-value oracle;512 native kernels often serial; null bound retained. |
| [psnr-cb](matrix/psnr-cb.json) | 512x512 | `canonical_psnr_planes_cell.py` | Explicit code-value oracle;512 native kernels often serial; null bound retained. |
| [psnr-cr](matrix/psnr-cr.json) | 512x512 | `canonical_psnr_planes_cell.py` | Explicit code-value oracle;512 native kernels often serial; null bound retained. |
| [psnr-y](matrix/psnr-y.json) | 512x512 | `canonical_classical_cell.py` | Explicit code-value oracle;512 native kernels often serial; null bound retained. |
| [psnr-y-libvmaf](matrix/psnr-y-libvmaf.json) | 512x512 | `libvmaf-rgb-reference-v2` | Per-feature luma conventions; context/representation work explicit; no VmafModel. |
| [psnr-y-studio601](matrix/psnr-y-studio601.json) | 512x512 | `canonical_classical_cell.py` | Explicit code-value oracle;512 native kernels often serial; null bound retained. |
| [psnr-y601](matrix/psnr-y601.json) | 512x512 | `canonical_classical_cell.py` | Explicit code-value oracle;512 native kernels often serial; null bound retained. |
| [psnr-ycbcr611](matrix/psnr-ycbcr611.json) | 512x512 | `canonical_psnr_planes_cell.py` | Explicit code-value oracle;512 native kernels often serial; null bound retained. |
| [flip](matrix/flip.json) | 512x512 | `canonical_flip_cell.py` | Declared f64 versus package f32 mean; imported extension source/thread proof incomplete. |
| [flip-v17](matrix/flip-v17.json) | 512x512 | `canonical_cpu_cell.py` | Declared f64 versus package f32 mean; imported extension source/thread proof incomplete. |
| [ws-hv](matrix/ws-hv.json) | 512x512 | `canonical_classical_cell.py` | Original function explicitly RGB2GRAY; notebook setup outside; scalar512 only. |
| [hdrvdp](matrix/hdrvdp.json) | 512x512 | `first_boundary_hdrvdp_resident_cell.py` | Base100nit/1black/30ppd; interpolated reference params preloaded, native preparation timed. |
| [hdrvdp3](matrix/hdrvdp3.json) | 512x512 | `first_boundary_hdrvdp_resident_cell.py` | Base100nit/1black/30ppd; interpolated reference params preloaded, native preparation timed. |

## Findings and disposition

- **AIC-PARAM-IO (2 routes):** original `dlmread` stays in canonical warm calls while native uses embedded CSV. Preserve the separate200nit/XYZ/Q convention and all observations.
- **BASE-HDR-PARAM-PREP (2 routes):** the resident fixture preloads interpolated spectral arrays; native parsing/interpolation stays timed. Coordinator/root provisional judgment keeps raw timings descriptive/unqualified and outside strict acceptance, with all scores retained. No human-approved exception is claimed.
- **VMAF-MODEL-PREP (6 routes):** original model loading is outside clocks; native JSON/SVM parsing remains inside each callback. Feature-only libvmaf routes are excluded. No effect size inferred.
- **DSSIM:** native constructs/destroys context per callback; inspected C wrapper retains context across calls. Selected build digest omits its C source, so exact wrapper-to-binary correspondence remains unproved. The C source SHA is recorded; thread caps come from the outer runner, not argv parsing.
- **Source limits:** SSIM2 selected plan lacks the inspected C++ wrapper hash; FLIP plan lacks imported Python/C-extension body binding. Observed source hashes are labeled unbound. No wrong-binary or runtime-thread failure is inferred.

V3 HDR bounds remain null. The observed photo512 native/reference difference1.7763568394002505e-15 does not create a bound or native-parity qualification. Existing numerical controls/scores are retained; source timing findings do not invalidate them. The coordinator scope-aware successor is separate from the immutable80-case manifest audited here.

## Reproduction and exclusions

`selection.py` is a stdlib read-only intake which reproduces preparer selection without executing its native inventory. `profiles.py` contains reviewed notes; `assemble.py` writes a new matrix from selection shards, exact comparative plan and retained verified build, failing on source drift or an existing output folder. No package runtime is imported. Each generated file is below30KB.

No inference, builds, CI, GPU maps, heavy admission or process changes were performed. No production metric, active helper, plan, score, bound or output was edited. The [raw-table-resident design](hdr-parameter-design.md) remains documentation only and explicitly retains native CSV parsing as residual scope. `first_boundary_hdrvdp.rs` is uncompiled/superseded and supplies no evidence; current40 uses the two exact production route bodies in the full worker. Root coordinator owns landing and workspace cleanup.
