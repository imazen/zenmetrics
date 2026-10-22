#![forbid(unsafe_code)]

//! Metric backend dispatch.
//!
//! `MetricKind` is the user-facing enum (parsed from the CLI's `--metric`
//! flag). `run_metric` resolves it to the actual backend and runs the
//! comparison on the supplied RGB8 image pair.
//!
//! ## GPU path: one umbrella, one switch
//!
//! Every GPU metric routes through the [`zenmetrics_api`] umbrella crate.
//! The umbrella exposes an opaque `Metric::new(MetricKind, Backend, w, h,
//! params)` constructor and a `Metric::compute_srgb_u8(&r, &d) -> Score`
//! method that hide the per-crate `<Metric><R: Runtime>::new` /
//! `<Metric><R: Runtime>::compute` typed surface. The CLI keeps a single
//! `match kind { ... }` to translate its [`MetricKind`] into the
//! umbrella's `MetricKind` + per-metric default `MetricParams`. There is
//! no per-metric `gpu` module any more; one [`run_gpu_via_umbrella`]
//! helper handles the GPU variants (ssim2-gpu / dssim-gpu / iwssim-gpu /
//! zensim-gpu / cvvdp-gpu single-shot and butter max-norm scoring). The
//! unsuffixed `cvvdp` / `iwssim` are the native SIMD CPU ports (the
//! `ssim2`/`dssim`/`zensim` unsuffixed=CPU convention), scored through
//! `run_cpu_native_via_umbrella` / `Backend::Cpu`.
//!
//! Two typed-path escape hatches stay for the cases the opaque API does
//! not cover today:
//!
//! - [`butter_pnorm3`] uses the typed `butteraugli_gpu::Butteraugli<R>`
//!   pipeline (reached via the umbrella's `zenmetrics_api::butter`
//!   re-export) to extract the libjxl 3-norm aggregation
//!   (`ButteraugliResult.pnorm_3`) alongside the max-norm in one
//!   `compute()` call. The umbrella's opaque `Score` only carries the
//!   max-norm value.
//! - [`cvvdp_gpu`] (this directory) keeps `CvvdpBatchScorer`, which
//!   caches a `cvvdp_gpu::Cvvdp<R>` instance across pairs of matching
//!   dims to avoid the ~200 MB / NVRTC compile per-pair cost (fleet
//!   OOMs at 100-pair chunks otherwise). The instance type comes from
//!   the umbrella's `zenmetrics_api::cvvdp` re-export, so the CLI
//!   still has no direct `cvvdp-gpu` dependency.

use clap::ValueEnum;

use crate::decode::Rgb8Image;
mod butteraugli_original;
mod cw_fft;
mod cw_ssim;
pub mod display;
mod flip;
#[cfg(feature = "cpu-hdrvdp")]
pub(crate) mod hdrvdp_aic;
mod ssim_pyiqa;
mod ssimulacra1;
mod ws_hv;
pub use display::CvvdpDisplay;
// VMAF adapter tests cross-check their Y plane against classical PSNR.
#[cfg(any(feature = "cpu-metrics", all(test, feature = "cpu-vmaf")))]
mod classical;
#[cfg(feature = "nn")]
mod nn;
#[cfg(feature = "nr")]
mod nr;
#[cfg(feature = "pyiqa-ports")]
mod pyiqa_ports;
#[cfg(any(feature = "cpu-metrics", feature = "cpu-vmaf"))]
pub mod vmaf;

// ssim2 / dssim / butteraugli one-shot CPU scoring now routes through the
// single `zenmetrics-api::cpu_dispatch` umbrella path (`run_metric` +
// `run_cpu_native_via_umbrella` / `run_cpu_butter_both_via_umbrella`); the
// former per-metric CPU shim modules are gone. zensim keeps its module for
// the warm-reference + 300-feature extended paths (not yet folded in).
#[cfg(feature = "cpu-metrics")]
pub(crate) mod zensim;

#[cfg(feature = "gpu-butteraugli")]
pub mod butter_pnorm3;
#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
pub(crate) mod cache;
// Callers live in the `zenmetrics` bin (`batch` / `score-pairs`); the lib
// target compiles this module without them.
#[cfg(feature = "cpu-cvvdp")]
#[allow(dead_code)]
pub(crate) mod cvvdp_cpu;
#[cfg(feature = "gpu-cvvdp")]
pub mod cvvdp_gpu;

/// Metric identifier exposed on the CLI.
// `Hash` keys the HDR sweep's process-static scorer cache
// (`sweep::hdr::HDR_SCORERS`) by CLI metric kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
pub enum MetricKind {
    /// VMAF v0.6.1 (in-tree pure-Rust `vmaf` crate, libvmaf v3.2.1 match).
    #[value(name = "vmaf")]
    Vmaf,
    /// VMAF v0.6.1neg, with enhancement gain disabled (in-tree `vmaf` crate).
    #[value(name = "vmaf-neg")]
    VmafNeg,
    /// VMAF v0.6.1 4K model (in-tree `vmaf` crate).
    #[value(name = "vmaf-4k")]
    Vmaf4k,
    /// VMAF v1.0.16 3d0h model (in-tree `vmaf` crate).
    #[value(name = "vmaf-v1")]
    VmafV1,
    /// `vmaf` on studio-swing BT.601 YUV (the `psnr-y-studio601` luma), the
    /// convention behind the JPEG AIC-4 `VMAF` column.
    #[value(name = "vmaf-studio601")]
    VmafStudio601,
    /// `vmaf-neg` on studio-swing BT.601 YUV (JPEG AIC-4 `VMAF-neg`).
    #[value(name = "vmaf-neg-studio601")]
    VmafNegStudio601,
    /// Peak signal-to-noise ratio on encoded RGB8, in decibels.
    #[value(name = "psnr")]
    Psnr,
    /// Peak signal-to-noise ratio on full-range BT.709 luma.
    #[value(name = "psnr-y")]
    PsnrY,
    /// PSNR-Y on full-range BT.601 luma (the MATLAB `rgb2gray` weights
    /// 0.299/0.587/0.114) — the most common "PSNR-Y" convention in the
    /// literature; NOT the same numbers as `psnr-y` (BT.709).
    #[value(name = "psnr-y601")]
    PsnrY601,
    /// PSNR-Y on studio-swing BT.601 luma — `round(16 + (65.481R +
    /// 128.553G + 24.966B)/255)` per-pixel u8 — the JPEG/JFIF YUV420 `Y`
    /// convention and the JPEG AIC-4 `PSNR-Y` column (the same plane
    /// `ssim-libvmaf`/`msssim-libvmaf`/`psnrhvs-daala` build internally;
    /// equal to `--metric psnr-y --luma-ingress yuv601-studio`).
    #[value(name = "psnr-y-studio601")]
    PsnrYStudio601,
    /// PSNR-Y on studio-swing BT.709 luma — `round(16 + 219·L709)` with
    /// `L709 = (0.2126R + 0.7152G + 0.0722B)/255` — the Y-plane
    /// convention the CLI feeds the in-tree `vmaf` crate, i.e. what
    /// libvmaf's `psnr` aux feature reported under the removed exec
    /// adapter.
    #[value(name = "psnr-y-libvmaf")]
    PsnrYLibvmaf,
    /// PSNR-YCbCr 6:1:1 — weighted PSNR on studio-swing BT.601 YCbCr
    /// (6·MSE_Y + MSE_Cb + MSE_Cr)/8 converted to dB. Matches JPEG AIC-4
    /// `PSNR-YCbCr611` column. Studio-swing conversion per pixel:
    /// Y=round(16+(65.481R+128.553G+24.966B)/255),
    /// Cb=round(128+(-37.797R-74.203G+112B)/255),
    /// Cr=round(128+(112R-93.786G-18.214B)/255).
    #[value(name = "psnr-ycbcr611")]
    PsnrYCbCr611,
    /// PSNR of the studio-swing BT.601 Cb plane at full resolution (4:4:4),
    /// the `psnr-ycbcr611` conversion. JPEG AIC-4 `PSNR-Cb`: 61 pairs
    /// median |Δ| 1.9e-3 dB, max 1.7e-2 dB (max 2.0e-3 JND).
    #[value(name = "psnr-cb")]
    PsnrCb,
    /// PSNR of the studio-swing BT.601 Cr plane at full resolution (4:4:4),
    /// the `psnr-ycbcr611` conversion. JPEG AIC-4 `PSNR-Cr`: 61 pairs
    /// median |Δ| 2.1e-3 dB, max 5.7e-2 dB (max 8.8e-3 JND).
    #[value(name = "psnr-cr")]
    PsnrCr,
    /// Mean CIEDE2000 colour difference (ΔE00) — sRGB → linear → XYZ D65
    /// → CIELAB → ΔE00 per pixel → mean. Matches JPEG AIC-4 `CIEDE2000`
    /// column. 61-pair median|Δ|=3.0e-5, max|Δ|=2.96e-4 (target ≤ 1e-3).
    #[value(name = "ciede2000")]
    Ciede2000,
    /// JPEG AIC-4 `SSIMc`: SSIM on each RGB channel (11x11 Gaussian,
    /// sigma 1.5, valid region, no downsampling, contrast-structure term
    /// clipped at 0), averaged. Not the same as `ssim` (reflected borders,
    /// no clipping).
    #[value(name = "ssimc")]
    SsimC,
    /// NVIDIA LDR-FLIP with default viewing conditions (lower is better);
    /// the error map's mean summed in f64, as NVlabs FLIP computes it.
    #[value(name = "flip")]
    Flip,
    /// LDR-FLIP pooled as flip-evaluator 1.7 does (row-major f32 sum), which
    /// reproduces the JPEG AIC-4 `FLIP` column exactly. Reads low against
    /// `flip` on large images (about 0.1% at 2048², 1.7% at 4096²).
    #[value(name = "flip-v17")]
    FlipV17,
    /// Complex-wavelet SSIM: four levels, eight orientations, coarse 7x7 windows.
    #[value(name = "cw-ssim")]
    CwSsim,
    /// Contrast-aware multiscale banding index, studio-swing BT.601 ingress.
    #[value(name = "cambi")]
    Cambi,
    /// Original Google Butteraugli (revision 2019-03-19); lower is better.
    #[value(name = "butteraugli-original")]
    ButteraugliOriginal,
    /// Original Cloudinary SSIMULACRA; lower is better.
    #[value(name = "ssimulacra1")]
    Ssimulacra1,
    /// Wavelet horizontal/vertical similarity; higher is better.
    #[value(name = "ws-hv")]
    WsHv,
    /// Single-scale structural similarity.
    #[value(name = "ssim")]
    Ssim,
    /// Normalized Laplacian Pyramid Distance; lower is better.
    #[value(name = "nlpd")]
    Nlpd,
    /// NLPD with the **IQA_pytorch / JPEG AIC-4 ingress** — the
    /// `dingkeyan93/IQA-optimization` `NLPD(channels=1)` variant the
    /// published AIC-4 `NLPD` column was computed with, NOT the
    /// Laparra-author RGB configuration that `nlpd` mirrors. Scored on
    /// a single BT.709 luma plane (`0.2126/0.7152/0.0722`, `[0,1]`,
    /// re-quantized to the 8-bit grid), reflect-2 downsample, bilinear
    /// `align_corners` upsample, six loop bands, mean-of-RMS pooling —
    /// vs the `nlpd` 0.6-norm over RGB (which reads ≈20.5× higher).
    /// med|Δ| = 0.0026 vs `metrics_fullres.csv` (53 pairs, 2026-09-26).
    /// Emits `nlpd_iqa` so it never mixes with `nlpd`'s column.
    /// CPU-only.
    #[value(name = "nlpd-iqa")]
    NlpdIqa,
    /// LPIPS v0.1, AlexNet trunk (upstream `lpips` 0.1.4); lower is
    /// better. Needs the `nn` feature and an exported model directory in
    /// `ZENMETRICS_NN_MODEL_DIR` (see `crates/zenmetrics-nn`).
    #[value(name = "lpips-alex")]
    LpipsAlex,
    /// LPIPS v0.1, VGG16 trunk; as `lpips-alex`.
    #[value(name = "lpips-vgg")]
    LpipsVgg,
    /// LPIPS v0.1, SqueezeNet 1.1 trunk; as `lpips-alex`.
    #[value(name = "lpips-squeeze")]
    LpipsSqueeze,
    /// DISTS (upstream `DISTS_pytorch` 0.1, no resize); lower is better.
    /// As `lpips-alex`.
    #[value(name = "dists")]
    Dists,
    /// ST-LPIPS, AlexNet (upstream `stlpips_pytorch` 0.0.2,
    /// shift-tolerant); lower is better. As `lpips-alex`.
    #[value(name = "stlpips-alex")]
    StlpipsAlex,
    /// ST-LPIPS, VGG16; as `stlpips-alex`.
    #[value(name = "stlpips-vgg")]
    StlpipsVgg,
    /// DeepDC (upstream `DeepDC_PyTorch` 0.2 with its ImageNet input
    /// normalisation on, as the JPEG AIC-4 `DeepDC` column; no resize); lower
    /// is better. As `lpips-alex`.
    #[value(name = "deepdc")]
    Deepdc,
    /// CKDN (upstream `researchmm/CKDN` checkpoint, fed as
    /// `cat(dist, ref, dist, ref)` → `dr`; upstream eval transform:
    /// short side →329 then centre-crop 288, bicubic for the distorted
    /// image, nearest for the reference); **higher** is better —
    /// it is a sigmoid quality score, not a distance. As `lpips-alex`.
    #[value(name = "ckdn")]
    Ckdn,
    /// A-FINE — learned FR metric (upstream `ChrisDud0257/AFINE`, Apache-2.0;
    /// CLIP ViT-B/32 taps + naturalness/fidelity heads, raw `afine_all`,
    /// **lower** is better). Needs feature `nn`; its weights carry
    /// noncommercial terms (trained on DiffIQA data), so
    /// `ZENMETRICS_NN_ACCEPT_WEIGHTS=afine` as for `ckdn`.
    #[value(name = "afine")]
    Afine,
    /// DreamSim ensemble — learned perceptual similarity distance
    /// (upstream `ssundaram21/dreamsim`, MIT; DINO + CLIP + OpenCLIP
    /// ViT-B/16 ensemble, `1 - cosine`, **lower** is better). Needs
    /// feature `nn`; its weights state no licence, so
    /// `ZENMETRICS_NN_ACCEPT_WEIGHTS=dreamsim`.
    #[value(name = "dreamsim")]
    Dreamsim,
    /// ARNIQA — learned no-reference quality score (upstream
    /// `miccunifi/ARNIQA`, Apache-2.0; ResNet-50 SimCLR embeddings at full +
    /// PIL-bilinear half scale, kadid10k Ridge regressor); **higher** is
    /// better, `[0, 1]`. Scored through `PairedNoReference`: emits `arniqa`
    /// (distorted) and `arniqa_ref` (reference). As `lpips-alex` for
    /// `ZENMETRICS_NN_MODEL_DIR`/`ZENMETRICS_NN_ENGINE`.
    #[value(name = "arniqa")]
    Arniqa,
    /// MANIQA — learned no-reference quality score (upstream
    /// `IIGROUP/MANIQA`, Apache-2.0; ViT-B/8 + TABlocks + Swin stages,
    /// `ckpt_koniq10k`); **higher** is better, normalized `[0, 1]`
    /// (upstream min-max normalizes koniq10k MOS labels). Scored
    /// through `PairedNoReference`: emits
    /// `maniqa` (distorted) and `maniqa_ref` (reference). As `arniqa`
    /// for `ZENMETRICS_NN_MODEL_DIR`/`ZENMETRICS_NN_ENGINE`.
    #[value(name = "maniqa")]
    Maniqa,
    /// LIQE — learned no-reference quality score (upstream
    /// `zwx8981/LIQE`, MIT code / weights licence unknown:
    /// consent-gated; fine-tuned CLIP ViT-B/32, `LIQE.pt`); **higher**
    /// is better, MOS `[1, 5]`. Scored through `PairedNoReference`:
    /// emits `liqe` (distorted) and `liqe_ref` (reference). As `arniqa`
    /// for `ZENMETRICS_NN_MODEL_DIR`/`ZENMETRICS_NN_ENGINE`, plus
    /// `ZENMETRICS_NN_ACCEPT_WEIGHTS=liqe`.
    #[value(name = "liqe")]
    Liqe,
    /// MUSIQ — learned no-reference quality score (upstream
    /// `google-research/google-research/musiq`, Apache-2.0 incl.
    /// `koniq_ckpt.npz`; multi-scale transformer over 32x32 patches at
    /// 224/384/full resolution); **higher** is better, MOS-scale score.
    /// Scored through `PairedNoReference`: emits `musiq` (distorted) and
    /// `musiq_ref` (reference). As `arniqa` for
    /// `ZENMETRICS_NN_MODEL_DIR`/`ZENMETRICS_NN_ENGINE`.
    #[value(name = "musiq")]
    Musiq,
    /// HyperIQA — learned no-reference quality score (upstream
    /// `SSL92/hyperIQA`, MIT code; weights licence unknown —
    /// consent-gated; fine-tuned ResNet-50 hypernetwork,
    /// `koniq_pretrained.pkl`); **higher** is better, MOS-scale
    /// `[0, 100]`. Scored through `PairedNoReference`: emits `hyperiqa`
    /// (distorted) and `hyperiqa_ref` (reference). As `arniqa` for
    /// `ZENMETRICS_NN_MODEL_DIR`/`ZENMETRICS_NN_ENGINE`, plus
    /// `ZENMETRICS_NN_ACCEPT_WEIGHTS=hyperiqa`.
    #[value(name = "hyperiqa")]
    Hyperiqa,
    /// PaQ-2-PiQ — no-reference, authors' `baidut/paq2piq` (MIT code;
    /// CC-BY-NC-4.0 weights, consent-gated; ResNet-18 + 20x20-block
    /// RoIPool head, `RoIPoolModel-fit.10.bs.120.pth`); **higher** is
    /// better, MOS-scale `[0, 100]`. Scored through
    /// `PairedNoReference`: emits `paq2piq` (distorted) and
    /// `paq2piq_ref` (reference). As `arniqa` for
    /// `ZENMETRICS_NN_MODEL_DIR`/`ZENMETRICS_NN_ENGINE`, plus
    /// `ZENMETRICS_NN_ACCEPT_WEIGHTS=paq2piq`.
    #[value(name = "paq2piq")]
    Paq2piq,
    /// PieAPP v0.1, dense patches; noncommercial weights require consent.
    #[value(name = "pieapp")]
    Pieapp,
    /// WaDIQaM-FR LIVE — full-reference learned quality (upstream
    /// `dmaniry/deepIQA`, MIT code and weights; shared conv trunk over
    /// non-overlapping 32x32 patches + relu-weighted pooling,
    /// `fr_live_weighted.model`); distance-like — **higher** is more
    /// degraded versus the reference (an identical pair scores a
    /// nonzero image-dependent floor; KonJND median 0.019, range
    /// -0.29..0.70). Emits `wadiqam_fr_live`. As `arniqa` for
    /// `ZENMETRICS_NN_MODEL_DIR`/`ZENMETRICS_NN_ENGINE`.
    #[value(name = "wadiqam-fr-live")]
    WadiqamFrLive,
    /// WaDIQaM-FR TID — as `wadiqam-fr-live` with the TID2013-trained
    /// checkpoint (`fr_tid_weighted.model`; identical-pair floor
    /// median 12.5, range 9.7..20.6). Emits `wadiqam_fr_tid`.
    #[value(name = "wadiqam-fr-tid")]
    WadiqamFrTid,
    /// WaDIQaM-NR LIVE — no-reference learned quality (upstream
    /// `dmaniry/deepIQA`, MIT code and weights;
    /// `nr_live_weighted.model`); DMOS-like, **higher** is worse.
    /// Scored through `PairedNoReference`: emits `wadiqam_nr_live`
    /// (distorted) and `wadiqam_nr_live_ref` (reference). As `arniqa`
    /// for `ZENMETRICS_NN_MODEL_DIR`/`ZENMETRICS_NN_ENGINE`.
    #[value(name = "wadiqam-nr-live")]
    WadiqamNrLive,
    /// WaDIQaM-NR TID — as `wadiqam-nr-live` with the TID2013-trained
    /// checkpoint (`nr_tid_weighted.model`); measured **higher** is
    /// worse despite TID's MOS labels (median rho +0.79 on a JPEG
    /// q95..q3 ladder). Emits `wadiqam_nr_tid` and `wadiqam_nr_tid_ref`.
    #[value(name = "wadiqam-nr-tid")]
    WadiqamNrTid,
    /// Cool-Chic Wasserstein Distortion at sigma 2^1 through 2^5.
    /// Raw sRGB VGG16 pre-pool features; lower is closer. Emits five columns.
    #[value(name = "wd")]
    Wd,
    /// AHIQ — learned FR quality metric (upstream `IIGROUP/AHIQ`, MIT
    /// code; ViT-B/8 + ResNet-50 taps, deformable fusion, mean of nine
    /// 224×224 crops, **higher** is better). Needs feature `nn`; its
    /// weights state no licence, so `ZENMETRICS_NN_ACCEPT_WEIGHTS=ahiq`.
    #[value(name = "ahiq")]
    Ahiq,
    /// DMM — learned FR distance (upstream `Baoliang93/DMM`, MIT code;
    /// VGG16 L2-pooled taps, per-patch SVD head, **lower** is better).
    /// Needs feature `nn`; its only weights are torchvision VGG16
    /// (permissive), so no consent gate.
    #[value(name = "dmm")]
    Dmm,
    // pyiqa-ports: crates/zenmetrics-pyiqa-ports (pyiqa-derived code and
    // weights, non-commercial terms; feature `pyiqa-ports`).
    /// TOPIQ-FR, KADID-10k checkpoint (pyiqa `topiq_fr`; the AIC2026 `TOPIQ`
    /// column): CFANet on ResNet50, no resize, **higher** is better. Needs
    /// feature `pyiqa-ports`; code and weights are non-commercial, so
    /// `ZENMETRICS_NN_ACCEPT_WEIGHTS=topiq_fr`.
    #[value(name = "topiq-fr")]
    TopiqFr,
    /// TOPIQ-FR, PIPAL checkpoint (pyiqa `topiq_fr-pipal`; AIC2026
    /// `TOPIQ-PIPAL`). As `topiq-fr`, with
    /// `ZENMETRICS_NN_ACCEPT_WEIGHTS=topiq_fr_pipal`.
    #[value(name = "topiq-fr-pipal")]
    TopiqFrPipal,
    /// WaDIQaM-FR as pyiqa scores it (KADID checkpoint, pyiqa's 32×32 patch
    /// grid; the AIC2026 `WaDIQaM` column, **higher** is better). Needs
    /// feature `pyiqa-ports` and `ZENMETRICS_NN_ACCEPT_WEIGHTS=wadiqam_pyiqa`.
    #[value(name = "wadiqam-pyiqa")]
    WadiqamPyiqa,
    /// AHIQ as pyiqa scores it (20 uniform crops, timm 1.0.30 backbones,
    /// epoch-33 head; the AIC2026 `AHIQ` column, **higher** is better). Needs
    /// feature `pyiqa-ports` and `ZENMETRICS_NN_ACCEPT_WEIGHTS=ahiq_pyiqa`.
    #[value(name = "ahiq-pyiqa")]
    AhiqPyiqa,
    /// CW-SSIM as pyiqa computes it (coarsest of 4 levels, 8 orientations,
    /// f32 masks; the AIC2026 `CW-SSIM` column, **higher** is better, ≥ 64²).
    /// Needs feature `pyiqa-ports`; no model files.
    #[value(name = "cw-ssim-pyiqa")]
    CwSsimPyiqa,
    /// DMM as pyiqa scores it (square bilinear resize; close to the AIC2026
    /// `DMM` column, **lower** is better, ≥ 128²). Needs feature
    /// `pyiqa-ports`; torchvision VGG16 weights, no consent gate.
    #[value(name = "dmm-pyiqa")]
    DmmPyiqa,
    /// PieAPP as pyiqa scores it (64×64 patches at stride 27; the AIC2026
    /// `PieAPP` column, **lower** is better, ≥ 64²). Reads zenmetrics-nn's
    /// `pieapp.onnx`. Needs feature `pyiqa-ports` and
    /// `ZENMETRICS_NN_ACCEPT_WEIGHTS=pieapp_pyiqa`.
    #[value(name = "pieapp-pyiqa")]
    PieappPyiqa,
    /// SSIMULACRA2 — CPU implementation via Imazen's `fast-ssim2` crate.
    #[value(name = "ssim2")]
    Ssim2,
    /// SSIMULACRA2 — GPU implementation via the `ssim2-gpu` crate
    /// (dispatched through the `zenmetrics-api` umbrella).
    #[value(name = "ssim2-gpu")]
    Ssim2Gpu,
    /// Butteraugli — CPU implementation via the `butteraugli` crate.
    /// Emits **two** columns per cell: `butteraugli_max`
    /// (`ButteraugliResult::score`, the per-block maximum) and
    /// `butteraugli_pnorm3` (`ButteraugliResult::pnorm_3`, the libjxl-style
    /// 3-norm aggregation matching `butteraugli_main --pnorm` and the
    /// CID22 paper by Sneyers, Ben Baruch & Vaxman, Cloudinary). One `compute()` call yields both numbers,
    /// so emitting both is free.
    #[value(name = "butteraugli")]
    Butteraugli,
    /// Butteraugli — GPU implementation via the `butteraugli-gpu` crate.
    /// Emits two columns: `butteraugli_max_gpu` and `butteraugli_pnorm3_gpu`.
    /// Same single-`compute()` rationale as the CPU variant.
    #[value(name = "butteraugli-gpu")]
    ButteraugliGpu,
    /// DSSIM — CPU implementation via the `dssim-core` crate. Distance metric: 0 = identical.
    #[value(name = "dssim")]
    Dssim,
    /// DSSIM — GPU implementation via the `dssim-gpu` crate
    /// (dispatched through the `zenmetrics-api` umbrella).
    /// Distance metric: 0 = identical.
    #[value(name = "dssim-gpu")]
    DssimGpu,
    /// IW-SSIM — GPU implementation via the `iwssim-gpu` crate
    /// (dispatched through the `zenmetrics-api` umbrella).
    /// Range `[0, 1]`; 1.0 = identical.
    #[value(name = "iwssim-gpu")]
    IwssimGpu,
    /// Zensim — CPU implementation via the `zensim` crate.
    #[value(name = "zensim")]
    Zensim,
    /// Zensim — GPU implementation via the `zensim-gpu` crate
    /// (dispatched through the `zenmetrics-api` umbrella).
    #[value(name = "zensim-gpu")]
    ZensimGpu,
    /// ColorVideoVDP (still-image, JOD scale 0–10, 10 = imperceptible) —
    /// CPU implementation via the in-tree `cvvdp` crate (native SIMD port,
    /// dispatched through `zenmetrics_api::cpu_dispatch` / `Backend::Cpu`).
    /// This is the unsuffixed default, matching the `ssim2`/`dssim`/`zensim`
    /// convention (unsuffixed = CPU); use `cvvdp-gpu` for the GPU backend.
    /// Emits the CPU column name (`cvvdp_cpu_imazen_v*`), distinct from the
    /// GPU column so sidecars don't collide.
    #[value(name = "cvvdp")]
    Cvvdp,
    /// ColorVideoVDP (still-image, JOD scale 0–10, 10 = imperceptible) —
    /// GPU implementation via the `cvvdp-gpu` crate (dispatched through the
    /// umbrella by default; `score-pairs` uses the typed `CvvdpBatchScorer`
    /// for instance reuse across pairs). Emits the versioned GPU column
    /// (`cvvdp_imazen_v*`).
    #[value(name = "cvvdp-gpu")]
    CvvdpGpu,
    /// IW-SSIM (Information-Content Weighted SSIM, Wang & Li 2011) — CPU
    /// implementation via the in-tree `iwssim` crate (native SIMD port,
    /// dispatched through `zenmetrics_api::cpu_dispatch` / `Backend::Cpu`).
    /// Unsuffixed = CPU (the `ssim2`/`dssim`/`zensim` convention); use
    /// `iwssim-gpu` for the GPU backend. Score in `[0, 1]` where 1 =
    /// identical. Requires `min(W, H) >= 176` per the paper's 5-level
    /// pyramid + 11×11 valid-mode SSIM stats constraint; smaller images
    /// return an error. Emits the CPU column name (`iwssim_cpu_imazen_v*`).
    #[value(name = "iwssim")]
    Iwssim,
    /// IW-SSIM with the **piq / JPEG AIC-4 ingress** — identical
    /// in-tree `iwssim` algorithm, but the RGB→luma conversion is
    /// piq's unrounded `rgb2yiq` channel 0 (`0.299/0.587/0.114`,
    /// 0–255, no u8 rounding) instead of the Python-IW-SSIM
    /// reference's rounded BT.601. This is the convention the
    /// published AIC-4 `IW-SSIM` column was computed under
    /// (med|Δ| = 1.3e-6 vs `metrics_fullres.tab`, 53 pairs,
    /// 2026-09-25). Emits `iwssim_piq_imazen_v*` so it never mixes
    /// with `iwssim`'s column in a join. CPU-only.
    #[value(name = "iwssim-piq")]
    IwssimPiq,
    /// GMSD (Gradient Magnitude Similarity Deviation, Xue et al. 2014) —
    /// CPU implementation via the in-tree `gmsd` crate (a port of
    /// libgmsd). Distance: 0 = identical, larger = worse; typical values
    /// 0..0.35. Scored on BT.601 8-bit luma at half resolution, as the
    /// reference does. Emits `gmsd_cpu_imazen_v*`. Not routed through the
    /// umbrella or the orchestrator (CPU-only, no GPU twin).
    #[value(name = "gmsd")]
    Gmsd,
    /// MDSI default summation model (Nafchi et al. 2016), CPU, f64 L/H/M.
    /// Distance: smaller is better. Uses the `cpu-gmsd` build feature.
    #[value(name = "mdsi")]
    Mdsi,
    /// Paper-derived four-scale masked GMSD, with declared numerical conventions.
    #[value(name = "ms-gmsd")]
    MsGmsd,
    /// Paper-derived colour MS-GMSDc; not author-software parity.
    #[value(name = "ms-gmsdc")]
    MsGmsdc,
    /// HDR-VDP 2.2.2 (JOD scale 0–100, 100 = identical) — CPU implementation
    /// via the in-tree `hdrvdp` crate (native SIMD port of the official
    /// HDR-VDP-2.2.2; dispatched through `zenmetrics_api::cpu_dispatch` /
    /// `Backend::Cpu`). 8-bit sRGB pairs use upstream's own `sRGB-display`
    /// encoding, which `hdrvdp.m` documents for standard (LDR) colour images:
    /// the sRGB EOTF onto a 100 cd/m² peak, 1 cd/m² black display. Absolute
    /// luminance (cd/m², `ColorEncoding::RgbBt709` inside) goes through
    /// `score-pairs --hdr` / `sweep --hdr`. Viewing geometry defaults to
    /// `DEFAULT_PIX_PER_DEG` (30 — the measured UPIQ protocol). Emits
    /// `hdrvdp_imazen_v*`. No GPU twin.
    #[value(name = "hdrvdp")]
    Hdrvdp,
    /// HDR-VDP-3.0.7 (Q_JOD scale 0–10, 10 = identical) — CPU implementation
    /// via the in-tree `hdrvdp::v3` module (native port of the
    /// `jpeg-ai-qaf` `VDP3` NumPy reference, itself a port of the official
    /// HDR-VDP-3.0.7 MATLAB release). 8-bit sRGB pairs use upstream's
    /// `sRGB-display` encoding (as `hdrvdp`), task `quality`, the encoding's
    /// upstream emission (`led-lcd-srgb`), age 24, no surround, and the same
    /// 30 pixels per degree as v2. For absolute-luminance input (`--hdr`)
    /// there is **no default viewing geometry**: `--hdrvdp3-ppd` is required, and
    /// `--hdrvdp3-task`/`--hdrvdp3-input`/`--hdrvdp3-emission`/
    /// `--hdrvdp3-age`/`--hdrvdp3-surround` select the rest of the
    /// upstream parameter surface (defaults match the jpeg-ai-qaf
    /// `metrics.py` harness: task=quality, rgb-bt.709, led-lcd-srgb,
    /// age=24, surround=none). Emits `hdrvdp3_jod_imazen_v*`. No GPU twin.
    #[value(name = "hdrvdp3")]
    Hdrvdp3,
    /// AIC2026 two-segment calibration of the libjxl Butteraugli 3-norm.
    #[value(name = "proposal-butteraugli")]
    ProposalButteraugli,
    /// CAMBI with the AIC2026 swscale-compatible integer luma ingress.
    #[value(name = "cambi-aic2026")]
    CambiAic2026,
    /// HDR-VDP 3 res.Q and Q_JOD with 200-nit sRGB to XYZ, OLED, 37.842505866 ppd.
    #[value(name = "hdrvdp3-aic2026")]
    Hdrvdp3Aic2026,
    /// HDR-VDP 2 res.Q with unrounded sRGB at 200 nits and 37.84 ppd.
    #[value(name = "hdrvdp-aic2026")]
    HdrvdpAic2026,
    /// PSNR-HVS / PSNR-HVS-M (Egiazarian et al. VPQM-06, Ponomarenko et al.
    /// VPQM-07) — CPU implementation via the in-tree `psnrhvs` crate (a port
    /// of the authors' `psnrhvsm.m`). PSNR-like: dB scale, higher = better,
    /// `100000` = identical/visually indistinguishable. Scored per sRGB
    /// channel on the 0–255 scale and averaged (the multi-plane convention
    /// codec harnesses report). Emits both `psnrhvs_imazen_v*` and
    /// `psnrhvsm_imazen_v*` — the masked score is a byproduct of the same
    /// pass. Not routed through the umbrella or the orchestrator (CPU-only,
    /// no GPU twin).
    #[value(name = "psnrhvs")]
    Psnrhvs,
    /// PSNR-HVS-Y — the same metric on BT.601 luma only (the codec-benchmark
    /// staple; comparable to `psnr_hvs_y` in Daala/AV1-style reports).
    /// Emits `psnrhvsy_imazen_v*` and `psnrhvsym_imazen_v*`.
    #[value(name = "psnrhvs-y")]
    PsnrhvsY,
    /// HaarPSI (Reisenhofer et al., Signal Processing: Image
    /// Communication 61, 2018) — CPU implementation via the in-tree
    /// `haarpsi` crate (a port of the authors' MIT-licensed
    /// `HaarPSI.m`). Similarity on ~[0, 1]: 1 = identical (degenerate
    /// zero-weight inputs return NaN, like the reference). Scored on the
    /// sRGB pair through the reference's YIQ color path with the default
    /// subsampled preprocessing. Emits `haarpsi_imazen_v*`. Not routed
    /// through the umbrella or the orchestrator (CPU-only, no GPU twin).
    #[value(name = "haarpsi")]
    Haarpsi,
    /// HaarPSI on unrounded BT.601 luma only (the grayscale path — no
    /// I/Q color similarity). Emits `haarpsiy_imazen_v*`.
    #[value(name = "haarpsi-y")]
    HaarpsiY,
    /// FSIM / FSIMc (Zhang et al., IEEE TIP 20(8), 2011) — CPU
    /// implementation via the in-tree `fsim` crate (a port of the
    /// authors' `FR_FSIMc.m`: log-Gabor phase-congruency + Scharr
    /// gradient similarity, I/Q chroma term). Similarity on ~[0, 1]:
    /// 1 = identical (degenerate constant inputs return NaN, like the
    /// reference). One call emits both `fsim_imazen_v*` and
    /// `fsimc_imazen_v*`. Not routed through the umbrella or the
    /// orchestrator (CPU-only, no GPU twin).
    #[value(name = "fsim")]
    Fsim,
    /// AIC2026 ssim-pyiqa convention.
    #[value(name = "ssim-pyiqa")]
    SsimPyiqa,
    /// AIC2026 ssim-pyiqa-f32 convention.
    #[value(name = "ssim-pyiqa-f32")]
    SsimPyiqaF32,
    /// AIC2026 ssim-pyiqa-f32-k2 convention.
    #[value(name = "ssim-pyiqa-f32-k2")]
    SsimPyiqaF32K2,
    /// AIC2026 ssim-pyiqa-f32-k3 convention.
    #[value(name = "ssim-pyiqa-f32-k3")]
    SsimPyiqaF32K3,
    /// AIC2026 ssim-pyiqa-f32-k4 convention.
    #[value(name = "ssim-pyiqa-f32-k4")]
    SsimPyiqaF32K4,
    /// AIC2026 ssim-pyiqa-f32-k5 convention.
    #[value(name = "ssim-pyiqa-f32-k5")]
    SsimPyiqaF32K5,
    /// AIC2026 ssim-pyiqa-f32-k6 convention.
    #[value(name = "ssim-pyiqa-f32-k6")]
    SsimPyiqaF32K6,
    /// AIC2026 ssimc-pyiqa-f32 convention.
    #[value(name = "ssimc-pyiqa-f32")]
    SsimcPyiqaF32,
    /// AIC2026 ssimc-pyiqa-f32-k2 convention.
    #[value(name = "ssimc-pyiqa-f32-k2")]
    SsimcPyiqaF32K2,
    /// AIC2026 ssimc-pyiqa-f32-k3 convention.
    #[value(name = "ssimc-pyiqa-f32-k3")]
    SsimcPyiqaF32K3,
    /// AIC2026 ssimc-pyiqa-f32-k4 convention.
    #[value(name = "ssimc-pyiqa-f32-k4")]
    SsimcPyiqaF32K4,
    /// AIC2026 ssimc-pyiqa-f32-k5 convention.
    #[value(name = "ssimc-pyiqa-f32-k5")]
    SsimcPyiqaF32K5,
    /// AIC2026 ssimc-pyiqa-f32-k6 convention.
    #[value(name = "ssimc-pyiqa-f32-k6")]
    SsimcPyiqaF32K6,
    /// AIC2026 ms-ssim-pyiqa convention.
    #[value(name = "ms-ssim-pyiqa")]
    MsSsimPyiqa,
    /// AIC2026 ms-ssim-pyiqa-f32 convention.
    #[value(name = "ms-ssim-pyiqa-f32")]
    MsSsimPyiqaF32,
    /// AIC2026 ms-ssim-pyiqa-f32-k2 convention.
    #[value(name = "ms-ssim-pyiqa-f32-k2")]
    MsSsimPyiqaF32K2,
    /// AIC2026 ms-ssim-pyiqa-f32-k3 convention.
    #[value(name = "ms-ssim-pyiqa-f32-k3")]
    MsSsimPyiqaF32K3,
    /// AIC2026 ms-ssim-pyiqa-f32-k4 convention.
    #[value(name = "ms-ssim-pyiqa-f32-k4")]
    MsSsimPyiqaF32K4,
    /// AIC2026 ms-ssim-pyiqa-f32-k5 convention.
    #[value(name = "ms-ssim-pyiqa-f32-k5")]
    MsSsimPyiqaF32K5,
    /// AIC2026 ms-ssim-pyiqa-f32-k6 convention.
    #[value(name = "ms-ssim-pyiqa-f32-k6")]
    MsSsimPyiqaF32K6,

    /// FSIM/FSIMc with fixed factor 2.
    #[value(name = "fsim-k2")]
    FsimK2,
    /// FSIM/FSIMc with fixed factor 3.
    #[value(name = "fsim-k3")]
    FsimK3,
    /// FSIM/FSIMc with fixed factor 4.
    #[value(name = "fsim-k4")]
    FsimK4,
    /// FSIM/FSIMc with fixed factor 1.
    #[value(name = "fsim-nodownscale")]
    FsimNodownscale,

    /// FSIM on unrounded BT.601 luma only (the grayscale path — no I/Q
    /// chroma term). Emits `fsimy_imazen_v*`.
    #[value(name = "fsim-y")]
    FsimY,
    /// VSI — visual saliency-induced index, CPU-only in-tree `vsi`
    /// crate (port of the authors' `VSI.m`: SDSP saliency via a
    /// 256×256 log-Gabor + Lab pipeline, opponent-channel decimation,
    /// Scharr gradients, saliency-weighted pooling). Similarity on
    /// ~[0, 1]: 1 = identical (flat inputs return NaN, like the
    /// reference). Emits `vsi_imazen_v*`. Not routed through the
    /// umbrella or the orchestrator (CPU-only, no GPU twin).
    #[value(name = "vsi")]
    Vsi,
    /// VSI with fixed factor 2.
    #[value(name = "vsi-k2")]
    VsiK2,
    /// VSI with fixed factor 3.
    #[value(name = "vsi-k3")]
    VsiK3,
    /// VSI with fixed factor 4.
    #[value(name = "vsi-k4")]
    VsiK4,
    /// VSI with fixed factor 1.
    #[value(name = "vsi-nodownscale")]
    VsiNodownscale,

    /// MS-SSIM — multi-scale SSIM, CPU-only in-tree `msssim` crate
    /// (port of the authors' `msssim.m`: 5-level Gaussian-pyramid
    /// SSIM, weighted geometric mean of per-level means). Similarity
    /// on ~[0, 1]: 1 = identical (constant inputs also score 1.0 —
    /// no NaN). Emits `msssim_imazen_v*`. Not routed through the
    /// umbrella or the orchestrator (CPU-only, no GPU twin).
    #[value(name = "msssim")]
    Msssim,
    /// VIFp (pixel-domain Visual Information Fidelity, in-tree `vif`
    /// crate, port of the authors' `vifp_mscale.m`): ratio score
    /// ≥ 0, ~1 = identical; can exceed 1 on enhanced inputs; flat
    /// references yield NaN (denominator = 0 — matching the
    /// reference). Emits `vif_imazen_v*`. CPU-only — not routed
    /// through the umbrella or the orchestrator.
    #[value(name = "vif")]
    Vif,
    /// vifvec — the *original* steerable-pyramid vector-GSM VIF
    /// (in-tree `vif` crate, clean-room of the authors' `vifvec.m`
    /// release + matlabPyrTools `sp5Filters`/`buildSpyr`): 4-level
    /// steerable pyramid, 3×3 vector-GSM neighbourhoods, subbands
    /// [4 7 10 13 16 19 22 25], σn² = 0.4. **A different metric from
    /// `vif`** (VIFp) that shares the name in the literature — this
    /// is the implementation the JPEG AIC-4 `VIF` column was computed
    /// with (u8-rounded `round(0.299R+0.587G+0.114B)` luma; Δ ≤ 5e-5
    /// vs GNU Octave `vifvec.m`). Emits `vifvec_imazen_v*`. CPU-only.
    #[value(name = "vifvec")]
    VifVec,
    /// MAD — Most Apparent Distortion (in-tree `mad-iqa` crate, port
    /// of the authors' `hi_index.m`/`lo_index.m` + `ical_std`/
    /// `ical_stat` C-mex): DISTANCE score ≥ 0, 0 = identical,
    /// unbounded above; `min(w,h) < 34` yields NaN (the reference's
    /// `mp(17:end-17)` edge kill leaves nothing to pool). Emits
    /// `mad_imazen_v*` plus `mad_hi_imazen_v*` / `mad_lo_imazen_v*`
    /// (the two strategy indices). CPU-only — not routed through the
    /// umbrella or the orchestrator.
    #[value(name = "mad")]
    Mad,
    /// SSIM — the libvmaf `float_ssim` feature (Wang et al. 2004 as
    /// implemented by Z. Li / Netflix iqa: `round(min_dim/256)` box
    /// decimation, 11×11 Gaussian, zli l·c·s pooling, L=255) via
    /// `msssim::float_ssim`, FFI-verified against vendored libvmaf
    /// f85a8536. Scored on the studio-swing BT.601 luma plane a YUV420
    /// pipeline delivers to libvmaf
    /// (`Y = round(16 + (65.481R + 128.553G + 24.966B)/255)`), NOT the
    /// MATLAB `rgb2gray` luma `msssim`/`ssim` metrics use. This is the
    /// metric behind the JPEG AIC-4 `SSIM` column. `--luma-ingress`
    /// does not apply: the variant always builds its own studio-601 Y.
    #[value(name = "ssim-libvmaf")]
    SsimLibvmaf,
    /// MS-SSIM — the libvmaf `float_ms_ssim` feature (five-scale 9/7
    /// low-pass pyramid, 11×11 Gaussian, β/γ weights 0.0448/0.2856/
    /// 0.3001/0.2363/0.1333 with α=l only at the coarsest scale) via
    /// `msssim::float_ms_ssim`, FFI-verified against vendored libvmaf
    /// f85a8536. Same studio-601 luma ingress as `ssim-libvmaf`; the
    /// JPEG AIC-4 `MS-SSIM` column. NOT the Wang-MATLAB `msssim`
    /// variant (different pyramid, luma, and window handling).
    #[value(name = "msssim-libvmaf")]
    MsssimLibvmaf,
    /// PSNR-HVS — the Daala/libvmaf `dump_psnrhvs` feature (integer
    /// bin-DCT 8×8 blocks at step 7, CSF-masked error with variance-
    /// ratio masking, `0.8·Y + 0.1·(Cb + Cr)` combined) via
    /// `psnrhvs::daala`, FFI-verified against vendored libvmaf f85a8536.
    /// Input planes are studio-swing BT.601 Y + full-resolution Cb/Cr
    /// (YUV444 — the convention the JPEG AIC-4 `PSNR-HVS`/
    /// `PSNR-HVS-{Y,Cb,Cr}` columns reproduce under). NOT the Ponomarenko
    /// `psnrhvs` metric (floating DCT, different CSF + masking —
    /// ~7.6 dB stricter scoring). Emits four columns:
    /// `psnrhvs_daala_{y,cb,cr}` + combined `psnrhvs_daala`.
    #[value(name = "psnrhvs-daala")]
    PsnrhvsDaala,
    /// mDCT-PSNR (masked-DCT PSNR, Thomas Richter / University of
    /// Stuttgart — the JPEG AIC-4 `mDCT-PSNR` column) — CPU
    /// implementation via the in-tree `mdctpsnr` crate, a port of the
    /// reference `dctpsnr` binary verified to ~1e-5 dB on the AIC-4
    /// 53-pair subset. PSNR-like: dB scale, higher = better, `+inf` =
    /// identical. Scored on the decoded sRGB pair through the
    /// reference's linear-BT.601 YCbCr + sliding-8×8-DCT masking
    /// pipeline. Emits `mdctpsnr_imazen_v*`. Not routed through the
    /// umbrella or the orchestrator (CPU-only, no GPU twin).
    #[value(name = "mdctpsnr")]
    Mdctpsnr,
    /// PIQE — no-reference perception-based block distortion (in-tree
    /// `piqe` crate, port of pypiqe 1.2 / MATLAB `piqe`). Score in
    /// `[0, 100]`; higher is worse. Scored through `PairedNoReference`:
    /// emits `piqe` (distorted) and `piqe_ref` (reference). Needs `nr`.
    #[value(name = "piqe")]
    Piqe,
    /// NIQE — no-reference natural-image-quality evaluator (in-tree `niqe`
    /// crate, port of BasicSR `niqe.py` with LIVE's pristine-model params).
    /// Mahalanobis distance to the pristine MVG; higher is worse. Scored
    /// through `PairedNoReference`: emits `niqe` (distorted) and `niqe_ref`
    /// (reference). Needs `nr`.
    #[value(name = "niqe")]
    Niqe,
    /// BRISQUE — no-reference blind/referenceless image spatial quality
    /// evaluator (in-tree `brisque` crate, port of piq 0.8.0 `brisque()`
    /// with LIVE's released SVM model). SVR distance; higher is worse.
    /// Scored through `PairedNoReference`: emits `brisque` (distorted)
    /// and `brisque_ref` (reference). Needs `nr`.
    #[value(name = "brisque")]
    Brisque,
}

impl MetricKind {
    fn ssim_pyiqa_flags(self) -> Option<(bool, bool, bool, usize)> {
        match self {
            Self::SsimPyiqa => Some((false, false, true, 1)),
            Self::SsimPyiqaF32 => Some((false, false, false, 1)),
            Self::SsimPyiqaF32K2 => Some((false, false, false, 2)),
            Self::SsimPyiqaF32K3 => Some((false, false, false, 3)),
            Self::SsimPyiqaF32K4 => Some((false, false, false, 4)),
            Self::SsimPyiqaF32K5 => Some((false, false, false, 5)),
            Self::SsimPyiqaF32K6 => Some((false, false, false, 6)),
            Self::SsimcPyiqaF32 => Some((true, false, false, 1)),
            Self::SsimcPyiqaF32K2 => Some((true, false, false, 2)),
            Self::SsimcPyiqaF32K3 => Some((true, false, false, 3)),
            Self::SsimcPyiqaF32K4 => Some((true, false, false, 4)),
            Self::SsimcPyiqaF32K5 => Some((true, false, false, 5)),
            Self::SsimcPyiqaF32K6 => Some((true, false, false, 6)),
            Self::MsSsimPyiqa => Some((false, true, true, 1)),
            Self::MsSsimPyiqaF32 => Some((false, true, false, 1)),
            Self::MsSsimPyiqaF32K2 => Some((false, true, false, 2)),
            Self::MsSsimPyiqaF32K3 => Some((false, true, false, 3)),
            Self::MsSsimPyiqaF32K4 => Some((false, true, false, 4)),
            Self::MsSsimPyiqaF32K5 => Some((false, true, false, 5)),
            Self::MsSsimPyiqaF32K6 => Some((false, true, false, 6)),
            _ => None,
        }
    }

    pub fn all() -> &'static [MetricKind] {
        &[
            MetricKind::Vmaf,
            MetricKind::VmafNeg,
            MetricKind::Vmaf4k,
            MetricKind::VmafV1,
            MetricKind::VmafStudio601,
            MetricKind::VmafNegStudio601,
            MetricKind::Psnr,
            MetricKind::PsnrY,
            MetricKind::PsnrY601,
            MetricKind::PsnrYStudio601,
            MetricKind::PsnrYLibvmaf,
            MetricKind::PsnrYCbCr611,
            MetricKind::PsnrCb,
            MetricKind::PsnrCr,
            MetricKind::Ciede2000,
            MetricKind::SsimC,
            MetricKind::Flip,
            MetricKind::FlipV17,
            MetricKind::CwSsim,
            MetricKind::Cambi,
            MetricKind::ButteraugliOriginal,
            MetricKind::Ssimulacra1,
            MetricKind::WsHv,
            MetricKind::Ssim,
            MetricKind::Nlpd,
            MetricKind::NlpdIqa,
            MetricKind::LpipsAlex,
            MetricKind::LpipsVgg,
            MetricKind::LpipsSqueeze,
            MetricKind::Dists,
            MetricKind::StlpipsAlex,
            MetricKind::StlpipsVgg,
            MetricKind::Deepdc,
            MetricKind::Ckdn,
            MetricKind::Afine,
            MetricKind::Dreamsim,
            MetricKind::Arniqa,
            MetricKind::Maniqa,
            MetricKind::Liqe,
            MetricKind::Musiq,
            MetricKind::Hyperiqa,
            MetricKind::Paq2piq,
            MetricKind::Pieapp,
            MetricKind::WadiqamFrLive,
            MetricKind::WadiqamFrTid,
            MetricKind::WadiqamNrLive,
            MetricKind::WadiqamNrTid,
            MetricKind::Wd,
            MetricKind::Ahiq,
            MetricKind::Dmm,
            // pyiqa-ports
            MetricKind::TopiqFr,
            MetricKind::TopiqFrPipal,
            MetricKind::WadiqamPyiqa,
            MetricKind::AhiqPyiqa,
            MetricKind::CwSsimPyiqa,
            MetricKind::DmmPyiqa,
            MetricKind::PieappPyiqa,
            MetricKind::Ssim2,
            MetricKind::Ssim2Gpu,
            MetricKind::Butteraugli,
            MetricKind::ButteraugliGpu,
            MetricKind::Dssim,
            MetricKind::DssimGpu,
            MetricKind::IwssimGpu,
            MetricKind::Zensim,
            MetricKind::ZensimGpu,
            MetricKind::Cvvdp,
            MetricKind::CvvdpGpu,
            MetricKind::Iwssim,
            MetricKind::IwssimPiq,
            MetricKind::Gmsd,
            MetricKind::Mdsi,
            MetricKind::MsGmsd,
            MetricKind::MsGmsdc,
            MetricKind::Hdrvdp,
            MetricKind::Hdrvdp3,
            MetricKind::ProposalButteraugli,
            MetricKind::CambiAic2026,
            MetricKind::Hdrvdp3Aic2026,
            MetricKind::HdrvdpAic2026,
            MetricKind::Psnrhvs,
            MetricKind::PsnrhvsY,
            MetricKind::Haarpsi,
            MetricKind::HaarpsiY,
            MetricKind::Fsim,
            MetricKind::SsimPyiqa,
            MetricKind::SsimPyiqaF32,
            MetricKind::SsimPyiqaF32K2,
            MetricKind::SsimPyiqaF32K3,
            MetricKind::SsimPyiqaF32K4,
            MetricKind::SsimPyiqaF32K5,
            MetricKind::SsimPyiqaF32K6,
            MetricKind::SsimcPyiqaF32,
            MetricKind::SsimcPyiqaF32K2,
            MetricKind::SsimcPyiqaF32K3,
            MetricKind::SsimcPyiqaF32K4,
            MetricKind::SsimcPyiqaF32K5,
            MetricKind::SsimcPyiqaF32K6,
            MetricKind::MsSsimPyiqa,
            MetricKind::MsSsimPyiqaF32,
            MetricKind::MsSsimPyiqaF32K2,
            MetricKind::MsSsimPyiqaF32K3,
            MetricKind::MsSsimPyiqaF32K4,
            MetricKind::MsSsimPyiqaF32K5,
            MetricKind::MsSsimPyiqaF32K6,
            MetricKind::FsimK2,
            MetricKind::FsimK3,
            MetricKind::FsimK4,
            MetricKind::FsimNodownscale,
            MetricKind::FsimY,
            MetricKind::Vsi,
            MetricKind::VsiK2,
            MetricKind::VsiK3,
            MetricKind::VsiK4,
            MetricKind::VsiNodownscale,
            MetricKind::Msssim,
            MetricKind::Vif,
            MetricKind::VifVec,
            MetricKind::Mad,
            MetricKind::SsimLibvmaf,
            MetricKind::MsssimLibvmaf,
            MetricKind::PsnrhvsDaala,
            MetricKind::Mdctpsnr,
            MetricKind::Piqe,
            MetricKind::Niqe,
            MetricKind::Brisque,
        ]
    }

    pub fn name(self) -> &'static str {
        match self {
            MetricKind::Vmaf => "vmaf",
            MetricKind::VmafNeg => "vmaf-neg",
            MetricKind::Vmaf4k => "vmaf-4k",
            MetricKind::VmafV1 => "vmaf-v1",
            MetricKind::VmafStudio601 => "vmaf-studio601",
            MetricKind::VmafNegStudio601 => "vmaf-neg-studio601",
            MetricKind::Psnr => "psnr",
            MetricKind::PsnrY => "psnr-y",
            MetricKind::PsnrY601 => "psnr-y601",
            MetricKind::PsnrYStudio601 => "psnr-y-studio601",
            MetricKind::PsnrYLibvmaf => "psnr-y-libvmaf",
            MetricKind::PsnrYCbCr611 => "psnr-ycbcr611",
            MetricKind::PsnrCb => "psnr-cb",
            MetricKind::PsnrCr => "psnr-cr",
            MetricKind::Ciede2000 => "ciede2000",
            MetricKind::Flip => "flip",
            MetricKind::FlipV17 => "flip-v17",
            MetricKind::CwSsim => "cw-ssim",
            MetricKind::Cambi => "cambi",
            MetricKind::ButteraugliOriginal => "butteraugli-original",
            MetricKind::Ssimulacra1 => "ssimulacra1",
            MetricKind::WsHv => "ws-hv",
            MetricKind::SsimC => "ssimc",
            MetricKind::Ssim => "ssim",
            MetricKind::Nlpd => "nlpd",
            MetricKind::NlpdIqa => "nlpd-iqa",
            MetricKind::LpipsAlex => "lpips-alex",
            MetricKind::LpipsVgg => "lpips-vgg",
            MetricKind::LpipsSqueeze => "lpips-squeeze",
            MetricKind::Dists => "dists",
            MetricKind::StlpipsAlex => "stlpips-alex",
            MetricKind::StlpipsVgg => "stlpips-vgg",
            MetricKind::Deepdc => "deepdc",
            MetricKind::Ckdn => "ckdn",
            MetricKind::Afine => "afine",
            MetricKind::Dreamsim => "dreamsim",
            MetricKind::Arniqa => "arniqa",
            MetricKind::Maniqa => "maniqa",
            MetricKind::Liqe => "liqe",
            MetricKind::Musiq => "musiq",
            MetricKind::Hyperiqa => "hyperiqa",
            MetricKind::Paq2piq => "paq2piq",
            MetricKind::Pieapp => "pieapp",
            MetricKind::WadiqamFrLive => "wadiqam-fr-live",
            MetricKind::WadiqamFrTid => "wadiqam-fr-tid",
            MetricKind::WadiqamNrLive => "wadiqam-nr-live",
            MetricKind::WadiqamNrTid => "wadiqam-nr-tid",
            MetricKind::Wd => "wd",
            MetricKind::Ahiq => "ahiq",
            MetricKind::Dmm => "dmm",
            // pyiqa-ports
            MetricKind::TopiqFr => "topiq-fr",
            MetricKind::TopiqFrPipal => "topiq-fr-pipal",
            MetricKind::WadiqamPyiqa => "wadiqam-pyiqa",
            MetricKind::AhiqPyiqa => "ahiq-pyiqa",
            MetricKind::CwSsimPyiqa => "cw-ssim-pyiqa",
            MetricKind::DmmPyiqa => "dmm-pyiqa",
            MetricKind::PieappPyiqa => "pieapp-pyiqa",
            MetricKind::Ssim2 => "ssim2",
            MetricKind::Ssim2Gpu => "ssim2-gpu",
            MetricKind::Butteraugli => "butteraugli",
            MetricKind::ButteraugliGpu => "butteraugli-gpu",
            MetricKind::Dssim => "dssim",
            MetricKind::DssimGpu => "dssim-gpu",
            MetricKind::IwssimGpu => "iwssim-gpu",
            MetricKind::Zensim => "zensim",
            MetricKind::ZensimGpu => "zensim-gpu",
            MetricKind::Cvvdp => "cvvdp",
            MetricKind::CvvdpGpu => "cvvdp-gpu",
            MetricKind::Iwssim => "iwssim",
            MetricKind::IwssimPiq => "iwssim-piq",
            MetricKind::Gmsd => "gmsd",
            MetricKind::Mdsi => "mdsi",
            MetricKind::MsGmsd => "ms-gmsd",
            MetricKind::MsGmsdc => "ms-gmsdc",
            MetricKind::Hdrvdp => "hdrvdp",
            MetricKind::Hdrvdp3 => "hdrvdp3",
            MetricKind::ProposalButteraugli => "proposal-butteraugli",
            MetricKind::CambiAic2026 => "cambi-aic2026",
            MetricKind::Hdrvdp3Aic2026 => "hdrvdp3-aic2026",
            MetricKind::HdrvdpAic2026 => "hdrvdp-aic2026",
            MetricKind::Psnrhvs => "psnrhvs",
            MetricKind::PsnrhvsY => "psnrhvs-y",
            MetricKind::Haarpsi => "haarpsi",
            MetricKind::HaarpsiY => "haarpsi-y",
            MetricKind::Fsim => "fsim",
            MetricKind::SsimPyiqa => "ssim-pyiqa",
            MetricKind::SsimPyiqaF32 => "ssim-pyiqa-f32",
            MetricKind::SsimPyiqaF32K2 => "ssim-pyiqa-f32-k2",
            MetricKind::SsimPyiqaF32K3 => "ssim-pyiqa-f32-k3",
            MetricKind::SsimPyiqaF32K4 => "ssim-pyiqa-f32-k4",
            MetricKind::SsimPyiqaF32K5 => "ssim-pyiqa-f32-k5",
            MetricKind::SsimPyiqaF32K6 => "ssim-pyiqa-f32-k6",
            MetricKind::SsimcPyiqaF32 => "ssimc-pyiqa-f32",
            MetricKind::SsimcPyiqaF32K2 => "ssimc-pyiqa-f32-k2",
            MetricKind::SsimcPyiqaF32K3 => "ssimc-pyiqa-f32-k3",
            MetricKind::SsimcPyiqaF32K4 => "ssimc-pyiqa-f32-k4",
            MetricKind::SsimcPyiqaF32K5 => "ssimc-pyiqa-f32-k5",
            MetricKind::SsimcPyiqaF32K6 => "ssimc-pyiqa-f32-k6",
            MetricKind::MsSsimPyiqa => "ms-ssim-pyiqa",
            MetricKind::MsSsimPyiqaF32 => "ms-ssim-pyiqa-f32",
            MetricKind::MsSsimPyiqaF32K2 => "ms-ssim-pyiqa-f32-k2",
            MetricKind::MsSsimPyiqaF32K3 => "ms-ssim-pyiqa-f32-k3",
            MetricKind::MsSsimPyiqaF32K4 => "ms-ssim-pyiqa-f32-k4",
            MetricKind::MsSsimPyiqaF32K5 => "ms-ssim-pyiqa-f32-k5",
            MetricKind::MsSsimPyiqaF32K6 => "ms-ssim-pyiqa-f32-k6",

            MetricKind::FsimK2 => "fsim-k2",
            MetricKind::FsimK3 => "fsim-k3",
            MetricKind::FsimK4 => "fsim-k4",
            MetricKind::FsimNodownscale => "fsim-nodownscale",

            MetricKind::FsimY => "fsim-y",
            MetricKind::Vsi => "vsi",
            MetricKind::VsiK2 => "vsi-k2",
            MetricKind::VsiK3 => "vsi-k3",
            MetricKind::VsiK4 => "vsi-k4",
            MetricKind::VsiNodownscale => "vsi-nodownscale",

            MetricKind::Msssim => "msssim",
            MetricKind::Vif => "vif",
            MetricKind::VifVec => "vifvec",
            MetricKind::Mad => "mad",
            MetricKind::SsimLibvmaf => "ssim-libvmaf",
            MetricKind::MsssimLibvmaf => "msssim-libvmaf",
            MetricKind::PsnrhvsDaala => "psnrhvs-daala",
            MetricKind::Mdctpsnr => "mdctpsnr",
            MetricKind::Piqe => "piqe",
            MetricKind::Niqe => "niqe",
            MetricKind::Brisque => "brisque",
        }
    }

    pub fn backend(self) -> &'static str {
        match self {
            MetricKind::Ssim2Gpu
            | MetricKind::ButteraugliGpu
            | MetricKind::DssimGpu
            | MetricKind::IwssimGpu
            | MetricKind::ZensimGpu
            | MetricKind::CvvdpGpu => "GPU",
            // Unsuffixed metrics (incl. cvvdp / iwssim, which have in-tree
            // native SIMD CPU ports) are the CPU backend — matching the
            // `ssim2`/`dssim`/`zensim` unsuffixed=CPU convention.
            _ => "CPU",
        }
    }

    /// False when this build compiled the metric's scoring arm out: the
    /// learned metrics need the `nn` feature and the no-reference ports the
    /// `nr` feature. Such a metric parses but only returns a "disabled"
    /// error, so callers listing usable metrics (the browser module) skip it.
    // Called by zenmetrics-browser, which includes this file by path; the CLI
    // binary's private copy of the module has no caller.
    #[allow(dead_code)]
    pub fn scoring_compiled_in(self) -> bool {
        let nn = matches!(
            self,
            MetricKind::LpipsAlex
                | MetricKind::LpipsVgg
                | MetricKind::LpipsSqueeze
                | MetricKind::Dists
                | MetricKind::StlpipsAlex
                | MetricKind::StlpipsVgg
                | MetricKind::Deepdc
                | MetricKind::Ckdn
                | MetricKind::Afine
                | MetricKind::Dreamsim
                | MetricKind::Ahiq
                | MetricKind::Dmm
                | MetricKind::Arniqa
                | MetricKind::Maniqa
                | MetricKind::Liqe
                | MetricKind::Musiq
                | MetricKind::Hyperiqa
                | MetricKind::Paq2piq
                | MetricKind::WadiqamFrLive
                | MetricKind::WadiqamFrTid
                | MetricKind::WadiqamNrLive
                | MetricKind::WadiqamNrTid
                | MetricKind::Pieapp
                | MetricKind::Wd
        );
        let nr = matches!(
            self,
            MetricKind::Piqe | MetricKind::Niqe | MetricKind::Brisque
        );
        // pyiqa-ports
        let pyiqa = matches!(
            self,
            MetricKind::TopiqFr
                | MetricKind::TopiqFrPipal
                | MetricKind::WadiqamPyiqa
                | MetricKind::AhiqPyiqa
                | MetricKind::CwSsimPyiqa
                | MetricKind::DmmPyiqa
                | MetricKind::PieappPyiqa
        );
        !((nn && !cfg!(feature = "nn"))
            || (nr && !cfg!(feature = "nr"))
            || (pyiqa && !cfg!(feature = "pyiqa-ports"))
            || (matches!(self, MetricKind::HdrvdpAic2026 | MetricKind::Hdrvdp3Aic2026)
                && !cfg!(feature = "cpu-hdrvdp")))
    }

    pub fn requires_gpu(self) -> bool {
        matches!(
            self,
            MetricKind::Ssim2Gpu
                | MetricKind::ButteraugliGpu
                | MetricKind::DssimGpu
                | MetricKind::IwssimGpu
                | MetricKind::ZensimGpu
                | MetricKind::CvvdpGpu
        )
    }

    /// Output columns for this metric scored at `display`. cvvdp names its
    /// display in SDR output — `<impl column>_<display>`, e.g.
    /// `cvvdp_cpu_imazen_v0_1_0_standard_4k` — because cvvdp has no default
    /// display; `display = None` gives the display-less
    /// [`Self::column_names`] form, which only the HDR routes write (their
    /// display is the reference-peak HDR target, and `hdr_mode` qualifies
    /// the row). Display-less SDR cvvdp columns in existing sidecars predate
    /// 2026-09-25 and hold `standard_4k` scores. Every other metric ignores
    /// `display`.
    pub fn columns_for(self, display: Option<&CvvdpDisplay>) -> &'static [&'static str] {
        match (self, display) {
            (MetricKind::Cvvdp | MetricKind::CvvdpGpu, Some(d)) => {
                intern_display_column(self.column_names()[0], d.slug())
            }
            _ => self.column_names(),
        }
    }

    /// Whether this metric needs a display selection (`--display-model`)
    /// for SDR scoring.
    pub fn needs_display(self) -> bool {
        matches!(self, MetricKind::Cvvdp | MetricKind::CvvdpGpu)
    }

    /// Static list of TSV / parquet column suffixes emitted by this metric,
    /// in the order [`run_metric`] produces them. For most metrics this is a
    /// single column matching [`MetricKind::name`] with `-` rewritten to
    /// `_`. Butteraugli (CPU and GPU) emits two columns — `_max` and
    /// `_pnorm3` — because one `compute()` call yields both aggregations.
    ///
    /// Headers in the sweep TSV are formatted as `score_<column>` for each
    /// entry returned here.
    pub fn column_names(self) -> &'static [&'static str] {
        match self {
            MetricKind::Vmaf => &["vmaf"],
            MetricKind::VmafNeg => &["vmaf_neg"],
            MetricKind::Vmaf4k => &["vmaf_4k"],
            MetricKind::VmafV1 => &["vmaf_v1"],
            MetricKind::VmafStudio601 => &["vmaf_studio601"],
            MetricKind::VmafNegStudio601 => &["vmaf_neg_studio601"],
            MetricKind::Psnr => &["psnr"],
            MetricKind::PsnrY => &["psnr_y"],
            MetricKind::PsnrY601 => &["psnr_y601"],
            MetricKind::PsnrYStudio601 => &["psnr_y_studio601"],
            MetricKind::PsnrYLibvmaf => &["psnr_y_libvmaf"],
            MetricKind::PsnrYCbCr611 => &["psnr_ycbcr611"],
            MetricKind::PsnrCb => &["psnr_cb"],
            MetricKind::PsnrCr => &["psnr_cr"],
            MetricKind::Ciede2000 => &["ciede2000"],
            MetricKind::Flip => &["flip"],
            MetricKind::FlipV17 => &["flip_v17"],
            MetricKind::CwSsim => &["cw_ssim"],
            MetricKind::Cambi => &["cambi"],
            MetricKind::ButteraugliOriginal => &["butteraugli_original"],
            MetricKind::Ssimulacra1 => &["ssimulacra1"],
            MetricKind::WsHv => &["ws_hv"],
            MetricKind::SsimC => &["ssimc"],
            MetricKind::Ssim => &["ssim"],
            MetricKind::Nlpd => &["nlpd"],
            MetricKind::NlpdIqa => &["nlpd_iqa"],
            MetricKind::LpipsAlex => &["lpips_alex"],
            MetricKind::LpipsVgg => &["lpips_vgg"],
            MetricKind::LpipsSqueeze => &["lpips_squeeze"],
            MetricKind::Dists => &["dists"],
            MetricKind::StlpipsAlex => &["stlpips_alex"],
            MetricKind::StlpipsVgg => &["stlpips_vgg"],
            MetricKind::Deepdc => &["deepdc"],
            MetricKind::Ckdn => &["ckdn"],
            MetricKind::Afine => &["afine"],
            MetricKind::Dreamsim => &["dreamsim"],
            MetricKind::Arniqa => &["arniqa", "arniqa_ref"],
            MetricKind::Maniqa => &["maniqa", "maniqa_ref"],
            MetricKind::Liqe => &["liqe", "liqe_ref"],
            MetricKind::Musiq => &["musiq", "musiq_ref"],
            MetricKind::Hyperiqa => &["hyperiqa", "hyperiqa_ref"],
            MetricKind::Paq2piq => &["paq2piq", "paq2piq_ref"],
            MetricKind::Pieapp => &["pieapp"],
            MetricKind::WadiqamFrLive => &["wadiqam_fr_live"],
            MetricKind::WadiqamFrTid => &["wadiqam_fr_tid"],
            MetricKind::WadiqamNrLive => &["wadiqam_nr_live", "wadiqam_nr_live_ref"],
            MetricKind::WadiqamNrTid => &["wadiqam_nr_tid", "wadiqam_nr_tid_ref"],
            MetricKind::Wd => &["wd_s1", "wd_s2", "wd_s3", "wd_s4", "wd_s5"],
            MetricKind::Ahiq => &["ahiq"],
            MetricKind::Dmm => &["dmm"],
            // pyiqa-ports
            MetricKind::TopiqFr => &["topiq_fr"],
            MetricKind::TopiqFrPipal => &["topiq_fr_pipal"],
            MetricKind::WadiqamPyiqa => &["wadiqam_pyiqa"],
            MetricKind::AhiqPyiqa => &["ahiq_pyiqa"],
            MetricKind::CwSsimPyiqa => &["cw_ssim_pyiqa"],
            MetricKind::DmmPyiqa => &["dmm_pyiqa"],
            MetricKind::PieappPyiqa => &["pieapp_pyiqa"],
            MetricKind::Ssim2 => &["ssim2"],
            MetricKind::Ssim2Gpu => &["ssim2_gpu"],
            MetricKind::Butteraugli => &["butteraugli_max", "butteraugli_pnorm3"],
            MetricKind::ButteraugliGpu => &["butteraugli_max_gpu", "butteraugli_pnorm3_gpu"],
            MetricKind::Dssim => &["dssim"],
            MetricKind::DssimGpu => &["dssim_gpu"],
            MetricKind::IwssimGpu => &["iwssim_gpu"],
            MetricKind::Zensim => &["zensim"],
            MetricKind::ZensimGpu => &["zensim_gpu"],
            MetricKind::Cvvdp => &[CVVDP_CPU_BASE],
            MetricKind::CvvdpGpu => &[CVVDP_GPU_BASE],
            MetricKind::Iwssim => IWSSIM_CPU_COLUMNS,
            MetricKind::IwssimPiq => IWSSIM_PIQ_CPU_COLUMNS,
            MetricKind::Gmsd => GMSD_CPU_COLUMNS,
            MetricKind::Mdsi => MDSI_CPU_COLUMNS,
            MetricKind::MsGmsd => &["ms_gmsd_paper_cpu_imazen_v0_1_0"],
            MetricKind::MsGmsdc => &["ms_gmsdc_paper_cpu_imazen_v0_1_0"],
            MetricKind::Hdrvdp => HDRVDP_CPU_COLUMNS,
            MetricKind::Hdrvdp3 => HDRVDP3_CPU_COLUMNS,
            MetricKind::ProposalButteraugli => &["proposal_butteraugli"],
            MetricKind::CambiAic2026 => &["cambi_aic2026"],
            MetricKind::Hdrvdp3Aic2026 => &["hdrvdp3_aic2026_q", "hdrvdp3_aic2026_jod"],
            MetricKind::HdrvdpAic2026 => &["hdrvdp_aic2026_q"],
            MetricKind::Psnrhvs => PSNRHVS_CPU_COLUMNS,
            MetricKind::PsnrhvsY => PSNRHVSY_CPU_COLUMNS,
            MetricKind::Haarpsi => HAARPSI_CPU_COLUMNS,
            MetricKind::HaarpsiY => HAARPSIY_CPU_COLUMNS,
            MetricKind::Fsim => FSIM_CPU_COLUMNS,
            MetricKind::SsimPyiqa => &["ssim_pyiqa"],
            MetricKind::SsimPyiqaF32 => &["ssim_pyiqa_f32"],
            MetricKind::SsimPyiqaF32K2 => &["ssim_pyiqa_f32_k2"],
            MetricKind::SsimPyiqaF32K3 => &["ssim_pyiqa_f32_k3"],
            MetricKind::SsimPyiqaF32K4 => &["ssim_pyiqa_f32_k4"],
            MetricKind::SsimPyiqaF32K5 => &["ssim_pyiqa_f32_k5"],
            MetricKind::SsimPyiqaF32K6 => &["ssim_pyiqa_f32_k6"],
            MetricKind::SsimcPyiqaF32 => &["ssimc_pyiqa_f32"],
            MetricKind::SsimcPyiqaF32K2 => &["ssimc_pyiqa_f32_k2"],
            MetricKind::SsimcPyiqaF32K3 => &["ssimc_pyiqa_f32_k3"],
            MetricKind::SsimcPyiqaF32K4 => &["ssimc_pyiqa_f32_k4"],
            MetricKind::SsimcPyiqaF32K5 => &["ssimc_pyiqa_f32_k5"],
            MetricKind::SsimcPyiqaF32K6 => &["ssimc_pyiqa_f32_k6"],
            MetricKind::MsSsimPyiqa => &["ms_ssim_pyiqa"],
            MetricKind::MsSsimPyiqaF32 => &["ms_ssim_pyiqa_f32"],
            MetricKind::MsSsimPyiqaF32K2 => &["ms_ssim_pyiqa_f32_k2"],
            MetricKind::MsSsimPyiqaF32K3 => &["ms_ssim_pyiqa_f32_k3"],
            MetricKind::MsSsimPyiqaF32K4 => &["ms_ssim_pyiqa_f32_k4"],
            MetricKind::MsSsimPyiqaF32K5 => &["ms_ssim_pyiqa_f32_k5"],
            MetricKind::MsSsimPyiqaF32K6 => &["ms_ssim_pyiqa_f32_k6"],

            MetricKind::FsimK2 => intern_variant_columns(FSIM_CPU_COLUMNS, "k2"),
            MetricKind::FsimK3 => intern_variant_columns(FSIM_CPU_COLUMNS, "k3"),
            MetricKind::FsimK4 => intern_variant_columns(FSIM_CPU_COLUMNS, "k4"),
            MetricKind::FsimNodownscale => intern_variant_columns(FSIM_CPU_COLUMNS, "nodownscale"),

            MetricKind::FsimY => FSIMY_CPU_COLUMNS,
            MetricKind::Vsi => VSI_CPU_COLUMNS,
            MetricKind::VsiK2 => intern_variant_columns(VSI_CPU_COLUMNS, "k2"),
            MetricKind::VsiK3 => intern_variant_columns(VSI_CPU_COLUMNS, "k3"),
            MetricKind::VsiK4 => intern_variant_columns(VSI_CPU_COLUMNS, "k4"),
            MetricKind::VsiNodownscale => intern_variant_columns(VSI_CPU_COLUMNS, "nodownscale"),

            MetricKind::Msssim => MSSSIM_CPU_COLUMNS,
            MetricKind::Vif => VIF_CPU_COLUMNS,
            MetricKind::VifVec => VIFVEC_CPU_COLUMNS,
            MetricKind::Mad => MAD_CPU_COLUMNS,
            MetricKind::SsimLibvmaf => SSIM_LIBVMAF_CPU_COLUMNS,
            MetricKind::MsssimLibvmaf => MSSSIM_LIBVMAF_CPU_COLUMNS,
            MetricKind::PsnrhvsDaala => PSNRHVS_DAALA_CPU_COLUMNS,
            MetricKind::Mdctpsnr => MDCTPSNR_CPU_COLUMNS,
            MetricKind::Piqe => &["piqe", "piqe_ref"],
            MetricKind::Niqe => &["niqe", "niqe_ref"],
            MetricKind::Brisque => &["brisque", "brisque_ref"],
        }
    }

    /// Metrics that consume only a luminance plane (their house RGB→luma
    /// extraction is the whole colour path) and are therefore eligible for
    /// [`LumaIngress::Yuv601Studio`] substitution. Colour metrics — fsim
    /// (FSIMc chroma), vsi (SDSP colour saliency), haarpsi / psnrhvs
    /// (per-channel), dssim / ssim2 / butteraugli / zensim (native colour),
    /// cvvdp / hdrvdp (display-modelled), every GPU twin — return false and
    /// are always scored on the decoded RGB. The libvmaf-feature variants
    /// (`ssim-libvmaf`, `msssim-libvmaf`, `psnrhvs-daala`) also return
    /// false: they build their own studio-601 planes internally —
    /// applying the studio-601 ingress a second time would not be
    /// idempotent (the swing is affine, not a convex combination). The
    /// fixed-convention PSNR-Y variants (`psnr-y-studio601`,
    /// `psnr-y-libvmaf`) are excluded for the same reason; the
    /// house-luma ones (`psnr-y`, `psnr-y601`) accept the ingress.
    pub fn is_luma_only(self) -> bool {
        matches!(
            self,
            MetricKind::Gmsd
                | MetricKind::PsnrY
                | MetricKind::PsnrY601
                | MetricKind::PsnrhvsY
                | MetricKind::HaarpsiY
                | MetricKind::FsimY
                | MetricKind::Msssim
                | MetricKind::Vif
                | MetricKind::VifVec
                | MetricKind::Mad
                | MetricKind::Iwssim
                | MetricKind::IwssimPiq
        )
    }
}

/// Luminance-ingress convention for the luma-only metric family —
/// `--luma-ingress` on `score` / `batch`. Only metrics where
/// [`MetricKind::is_luma_only`] is true consume it; everything else runs on
/// the decoded sRGB regardless (see its doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Default)]
pub enum LumaIngress {
    /// Each metric's own documented luma extraction — the historical
    /// behaviour and the default.
    #[default]
    #[value(name = "house")]
    House,
    /// Studio-swing BT.601 luma — the ffmpeg / libvmaf YUV420 `Y` plane,
    /// `Y = round(16 + (65.481·R + 128.553·G + 24.966·B)/255)` per pixel,
    /// broadcast to a gray RGB image and even-cropped (the YUV420 constraint
    /// the pipeline it mimics runs). pyiqa's `to_y_channel` is the same
    /// transform unrounded and uncropped, so scores land within ~u8-rounding
    /// and an edge-row of that family too. This is the ingress the JPEG AIC-4
    /// published `SSIM`/`MS-SSIM`/`VMAF`/`VIF_vmaf` luma columns were
    /// computed with — see `docs/METRIC_PROVENANCE.md`. (The plain `VIF`
    /// column is *not* studio-601 — it is vifvec on u8-rounded full-range
    /// `0.299/0.587/0.114`; measured 2026-09-26.)
    #[value(name = "yuv601-studio")]
    Yuv601Studio,
}

/// Build the `Y = round(16 + (65.481·R + 128.553·G + 24.966·B)/255)`
/// (ITU-R BT.601 studio swing — the JPEG/JFIF and libvmaf YUV420 luma) of
/// `img`, broadcast to a gray RGB8 image even-cropped to `w & !1 × h & !1`.
/// The coefficients are the JPEG 6B / MPEG / libvmaf fixed-point weights in
/// float form; per-pixel u8 rounding matches the ffmpeg/libvmaf conversion.
///
/// The callers' luma-only metrics then re-derive luma of (y, y, y) — every
/// house luma extraction is a convex combination, so it returns `y` (within
/// ±1 LSB of its own rounding), i.e. the metric's formula is unchanged; only
/// the ingress plane differs.
pub fn studio601_gray(img: &Rgb8Image) -> Rgb8Image {
    let w = (img.width & !1) as usize;
    let h = (img.height & !1) as usize;
    let src_w = img.width as usize;
    let mut pixels = vec![0u8; w * h * 3];
    let fill = |orow: &mut [u8], srow: &[u8]| {
        for (o, px) in orow
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(srow.as_chunks::<3>().0)
        {
            let yv = (16.0
                + (65.481 * px[0] as f32 + 128.553 * px[1] as f32 + 24.966 * px[2] as f32) / 255.0)
                .round()
                .clamp(0.0, 255.0) as u8;
            o.copy_from_slice(&[yv, yv, yv]);
        }
    };
    if w * h >= (1 << 20) && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        pixels
            .par_chunks_mut(w * 3)
            .enumerate()
            .for_each(|(y, orow)| fill(orow, &img.pixels[y * src_w * 3..y * src_w * 3 + w * 3]));
    } else {
        pixels
            .chunks_mut(w * 3)
            .enumerate()
            .for_each(|(y, orow)| fill(orow, &img.pixels[y * src_w * 3..y * src_w * 3 + w * 3]));
    }
    Rgb8Image {
        pixels,
        width: w as u32,
        height: h as u32,
    }
}

/// The `Y = round(16 + (65.481R + 128.553G + 24.966B)/255)` studio-swing
/// BT.601 luma plane of `img` as a u16 buffer at full resolution (no
/// crop — the libvmaf SSIM features consume the Y plane as `&[u16]`).
/// Same per-pixel formula as [`studio601_gray`]; the u8→u16 widen is
/// lossless, so this is one fused pass. Row-parallel above a small
/// threshold; per-pixel math identical at any thread count.
#[cfg(feature = "cpu-msssim")]
fn studio601_y_plane_u16(img: &Rgb8Image) -> Vec<u16> {
    let (w, h) = (img.width as usize, img.height as usize);
    let mut y = vec![0u16; w * h];
    let fill = |orow: &mut [u16], srow: &[u8]| {
        for (o, px) in orow.iter_mut().zip(srow.as_chunks::<3>().0) {
            *o = u16::from(
                (16.0
                    + (65.481 * px[0] as f32 + 128.553 * px[1] as f32 + 24.966 * px[2] as f32)
                        / 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8,
            );
        }
    };
    if w * h >= (1 << 20) && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        y.par_chunks_mut(w)
            .enumerate()
            .for_each(|(row, orow)| fill(orow, &img.pixels[row * w * 3..row * w * 3 + w * 3]));
    } else {
        y.chunks_mut(w)
            .enumerate()
            .for_each(|(row, orow)| fill(orow, &img.pixels[row * w * 3..row * w * 3 + w * 3]));
    }
    y
}

/// Studio-swing BT.601 YUV444 planes of `img` at full resolution — `Y`,
/// `Cb`, `Cr` all `w`×`h`:
/// `Y  = round(16 + (65.481R + 128.553G + 24.966B)/255)`,
/// `Cb = round(128 + (−37.797R − 74.203G + 112B)/255)`,
/// `Cr = round(128 + (112R − 93.786G − 18.214B)/255)`.
/// These are the planes the JPEG AIC-4 pipeline hands libvmaf's
/// `psnr_hvs` (its `PSNR-HVS-Cb/Cr` columns reproduce to ≤0.03 dB on
/// full-res chroma — see `docs/METRIC_PROVENANCE.md`).
#[cfg(feature = "cpu-psnrhvs")]
fn studio601_yuv444(img: &Rgb8Image) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    // Studio components stay positive (16..240). Adding half and truncating
    // preserves round-away-from-zero, checked against every RGB8 triple.
    let n = (img.width * img.height) as usize;
    let w = img.width as usize;
    let mut y = vec![0u8; n];
    let mut cb = vec![0u8; n];
    let mut cr = vec![0u8; n];
    let fill = |yrow: &mut [u8], cbrow: &mut [u8], crrow: &mut [u8], srow: &[u8]| {
        for (((yo, cbo), cro), px) in yrow
            .iter_mut()
            .zip(cbrow.iter_mut())
            .zip(crrow.iter_mut())
            .zip(srow.as_chunks::<3>().0)
        {
            let (r, g, b) = (px[0] as f32, px[1] as f32, px[2] as f32);
            *yo = (16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0 + 0.5) as u8;
            *cbo = (128.0 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0 + 0.5) as u8;
            *cro = (128.0 + (112.0 * r - 93.786 * g - 18.214 * b) / 255.0 + 0.5) as u8;
        }
    };
    if n >= (1 << 20) && rayon::current_num_threads() > 1 {
        use rayon::prelude::*;
        y.par_chunks_mut(w)
            .zip(cb.par_chunks_mut(w))
            .zip(cr.par_chunks_mut(w))
            .enumerate()
            .for_each(|(row, ((yrow, cbrow), crrow))| {
                fill(
                    yrow,
                    cbrow,
                    crrow,
                    &img.pixels[row * w * 3..row * w * 3 + w * 3],
                )
            });
    } else {
        y.chunks_mut(w)
            .zip(cb.chunks_mut(w))
            .zip(cr.chunks_mut(w))
            .enumerate()
            .for_each(|(row, ((yrow, cbrow), crrow))| {
                fill(
                    yrow,
                    cbrow,
                    crrow,
                    &img.pixels[row * w * 3..row * w * 3 + w * 3],
                )
            });
    }
    (y, cb, cr)
}

/// Substitute [`studio601_gray`] images when `ingress` applies to `kind`
/// (i.e. `ingress == Yuv601Studio` and [`MetricKind::is_luma_only`]).
/// Returns `None` for an image that should be used as decoded.
pub fn luma_ingress_pair(
    kind: MetricKind,
    ingress: LumaIngress,
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> (Option<Rgb8Image>, Option<Rgb8Image>) {
    if ingress == LumaIngress::Yuv601Studio && kind.is_luma_only() {
        (
            Some(studio601_gray(reference)),
            Some(studio601_gray(distorted)),
        )
    } else {
        (None, None)
    }
}

// Versioned **GPU** cvvdp column name (used by `MetricKind::CvvdpGpu`).
// With the `gpu-cvvdp` feature enabled, pulls the per-implementation tag
// from [`zenmetrics_api::cvvdp::CVVDP_COLUMN_NAME`] (the `cvvdp-gpu` crate;
// defaults to `cvvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>`; overridable at build
// time via `CVVDP_IMPL_TAG`). Without the feature, falls back to a bare
// `"cvvdp_gpu"` so callers that list metric names without invoking the
// backend still see a usable identifier.
//
// Why versioned: parquet sidecars store cvvdp scores from multiple
// implementations side-by-side (e.g. `cvvdp_pycvvdp_v054` for the
// pycvvdp reference, `cvvdp_imazen_v0_0_1` for the GPU crate, and
// `cvvdp_cpu_imazen_v*` for the in-tree CPU port). A bare `cvvdp`
// column would collide on join. See the PINNED TASK section in the
// repo-root `CLAUDE.md`.
#[cfg(feature = "gpu-cvvdp")]
const CVVDP_GPU_BASE: &str = zenmetrics_api::cvvdp::CVVDP_COLUMN_NAME;
#[cfg(not(feature = "gpu-cvvdp"))]
const CVVDP_GPU_BASE: &str = "cvvdp_gpu";

// Versioned **CPU** cvvdp column name (used by `MetricKind::Cvvdp`, the
// unsuffixed default). With the `cpu-cvvdp` feature enabled, pulls from
// [`zenmetrics_api::cvvdp_cpu::CVVDP_COLUMN_NAME`] (the in-tree `cvvdp`
// crate; defaults to `cvvdp_cpu_imazen_v<MAJOR>_<MINOR>_<PATCH>`,
// overridable via `CVVDP_CPU_IMPL_TAG`) — a DISTINCT namespace from the
// GPU column so CPU and GPU cvvdp scores never collide in a joined
// sidecar. Without the feature, falls back to a bare `"cvvdp"`.
#[cfg(feature = "cpu-cvvdp")]
const CVVDP_CPU_BASE: &str = zenmetrics_api::cvvdp_cpu::CVVDP_COLUMN_NAME;
#[cfg(not(feature = "cpu-cvvdp"))]
const CVVDP_CPU_BASE: &str = "cvvdp";

/// Refusal for a cvvdp SDR score without a display. There is no default:
/// the caller names one (`--display-model`, `cvvdp@<display>`).
pub fn display_required_msg(kind: MetricKind) -> String {
    format!(
        "metric '{}' needs an explicit display: pass --display-model <name> \
         (e.g. standard_4k, standard_fhd; jobs use `{}@<name>`) — there is no default",
        kind.name(),
        kind.name()
    )
}

/// `<base>_<display>` as a `&'static` single-column slice. Interned so the
/// many `&'static [&'static str]` column consumers (sweep headers, output
/// tables) can take display-named cvvdp columns; the set is bounded by the
/// number of distinct displays a process uses.
fn intern_display_column(base: &'static str, display: &str) -> &'static [&'static str] {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Interned = HashMap<(&'static str, String), &'static [&'static str]>;
    static TABLE: OnceLock<Mutex<Interned>> = OnceLock::new();
    let mut table = TABLE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    table.entry((base, display.to_string())).or_insert_with(|| {
        let col: &'static str = Box::leak(format!("{base}_{display}").into_boxed_str());
        Box::leak(Box::new([col]))
    })
}

// The bounded CLI variant list interns each base-column/suffix tuple once.
fn intern_variant_columns(
    bases: &'static [&'static str],
    suffix: &'static str,
) -> &'static [&'static str] {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type Key = (&'static [&'static str], &'static str);
    type Columns = &'static [&'static str];
    static TABLE: OnceLock<Mutex<HashMap<Key, Columns>>> = OnceLock::new();
    let mut table = TABLE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    table.entry((bases, suffix)).or_insert_with(|| {
        let columns: Vec<&'static str> = bases
            .iter()
            .map(|base| {
                let name: &'static str = Box::leak(format!("{base}_{suffix}").into_boxed_str());
                name
            })
            .collect();
        Box::leak(columns.into_boxed_slice())
    })
}

// Versioned **CPU** iwssim column name (used by `MetricKind::Iwssim`, the
// unsuffixed default) — mirrors the cpu-cvvdp pattern. With the
// `cpu-iwssim` feature on, pulls from
// [`zenmetrics_api::iwssim_cpu::IWSSIM_COLUMN_NAME`] (the in-tree `iwssim`
// crate; default `iwssim_cpu_imazen_v<MAJOR>_<MINOR>_<PATCH>`, overridable
// via `IWSSIM_CPU_IMPL_TAG`) — distinct from the GPU column. Without the
// feature, falls back to a bare `"iwssim"`.
#[cfg(feature = "cpu-iwssim")]
const IWSSIM_CPU_COLUMNS: &[&str] = &[zenmetrics_api::iwssim_cpu::IWSSIM_COLUMN_NAME];
#[cfg(not(feature = "cpu-iwssim"))]
const IWSSIM_CPU_COLUMNS: &[&str] = &["iwssim"];

// Versioned **CPU** GMSD column name (`gmsd::GMSD_COLUMN_NAME`, default
// `gmsd_cpu_imazen_v<MAJOR>_<MINOR>_<PATCH>`, overridable via
// `GMSD_CPU_IMPL_TAG`). Without `cpu-gmsd` a bare `"gmsd"`.
#[cfg(feature = "cpu-gmsd")]
const GMSD_CPU_COLUMNS: &[&str] = &[gmsd::GMSD_COLUMN_NAME];
#[cfg(not(feature = "cpu-gmsd"))]
const GMSD_CPU_COLUMNS: &[&str] = &["gmsd"];

#[cfg(feature = "cpu-gmsd")]
const MDSI_CPU_COLUMNS: &[&str] = &["mdsi_cpu_imazen_v0_1_0"];
#[cfg(not(feature = "cpu-gmsd"))]
const MDSI_CPU_COLUMNS: &[&str] = &["mdsi"];

// Versioned **CPU** HDR-VDP column name (`hdrvdp::HDRVDP_COLUMN_NAME`, default
// `hdrvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>`, overridable via `HDRVDP_IMPL_TAG` —
// no `_cpu_` infix because there is no GPU twin to collide with). Without
// `cpu-hdrvdp` a bare `"hdrvdp"`.
#[cfg(feature = "cpu-hdrvdp")]
const HDRVDP_CPU_COLUMNS: &[&str] = &[zenmetrics_api::hdrvdp_cpu::HDRVDP_COLUMN_NAME];
#[cfg(not(feature = "cpu-hdrvdp"))]
const HDRVDP_CPU_COLUMNS: &[&str] = &["hdrvdp"];

// Versioned **CPU** HDR-VDP-3 column name (`hdrvdp::v3::HDRVDP3_COLUMN_NAME`,
// `hdrvdp3_jod_imazen_v<MAJOR>_<MINOR>_<PATCH>` — `Q_JOD` units, distinct
// from v2's 0–100 correlate, overridable via `HDRVDP3_IMPL_TAG`).
#[cfg(feature = "cpu-hdrvdp")]
const HDRVDP3_CPU_COLUMNS: &[&str] = &[zenmetrics_api::hdrvdp_cpu::v3::HDRVDP3_COLUMN_NAME];
#[cfg(not(feature = "cpu-hdrvdp"))]
const HDRVDP3_CPU_COLUMNS: &[&str] = &["hdrvdp3_jod"];

// Versioned **CPU** PSNR-HVS column names (`psnrhvs::PSNRHVS_COLUMN_NAME` +
// `PSNRHVSM_COLUMN_NAME`; the masked score rides along from the same pass).
// No `_cpu_` infix — no GPU twin. Without `cpu-psnrhvs` bare names.
#[cfg(feature = "cpu-psnrhvs")]
const PSNRHVS_CPU_COLUMNS: &[&str] = &[psnrhvs::PSNRHVS_COLUMN_NAME, psnrhvs::PSNRHVSM_COLUMN_NAME];
#[cfg(not(feature = "cpu-psnrhvs"))]
const PSNRHVS_CPU_COLUMNS: &[&str] = &["psnrhvs", "psnrhvsm"];

// Versioned **CPU** PSNR-HVS-Y (luma) column names.
#[cfg(feature = "cpu-psnrhvs")]
const PSNRHVSY_CPU_COLUMNS: &[&str] = &[
    psnrhvs::PSNRHVSY_COLUMN_NAME,
    psnrhvs::PSNRHVSYM_COLUMN_NAME,
];
#[cfg(not(feature = "cpu-psnrhvs"))]
const PSNRHVSY_CPU_COLUMNS: &[&str] = &["psnrhvs_y", "psnrhvsym"];

// Versioned **CPU** HaarPSI column name (`haarpsi::HAARPSI_COLUMN_NAME`,
// default `haarpsi_imazen_v<MAJOR>_<MINOR>_<PATCH>`, overridable via
// `HAARPSI_IMPL_TAG`). No `_cpu_` infix — no GPU twin. Without
// `cpu-haarpsi` a bare `"haarpsi"`.
#[cfg(feature = "cpu-haarpsi")]
const HAARPSI_CPU_COLUMNS: &[&str] = &[haarpsi::HAARPSI_COLUMN_NAME];
#[cfg(not(feature = "cpu-haarpsi"))]
const HAARPSI_CPU_COLUMNS: &[&str] = &["haarpsi"];

// Versioned **CPU** HaarPSI-Y (luma) column name.
#[cfg(feature = "cpu-haarpsi")]
const HAARPSIY_CPU_COLUMNS: &[&str] = &[haarpsi::HAARPSIY_COLUMN_NAME];
#[cfg(not(feature = "cpu-haarpsi"))]
const HAARPSIY_CPU_COLUMNS: &[&str] = &["haarpsi_y"];

// Versioned **CPU** FSIM + FSIMc column names
// (`fsim::FSIM_COLUMN_NAME` / `fsim::FSIMC_COLUMN_NAME`, defaults
// `fsim_imazen_v<…>` / `fsimc_imazen_v<…>`, overridable via
// `FSIM_IMPL_TAG` / `FSIMC_IMPL_TAG`). One kernel pass produces both
// scores, so the `fsim` metric emits two columns. No `_cpu_` infix —
// no GPU twin. Without `cpu-fsim` bare `"fsim"`/`"fsimc"`.
#[cfg(feature = "cpu-fsim")]
const FSIM_CPU_COLUMNS: &[&str] = &[fsim::FSIM_COLUMN_NAME, fsim::FSIMC_COLUMN_NAME];
#[cfg(not(feature = "cpu-fsim"))]
const FSIM_CPU_COLUMNS: &[&str] = &["fsim", "fsimc"];

// Versioned **CPU** FSIM-Y (luma) column name.
#[cfg(feature = "cpu-fsim")]
const FSIMY_CPU_COLUMNS: &[&str] = &[fsim::FSIMY_COLUMN_NAME];
#[cfg(not(feature = "cpu-fsim"))]
const FSIMY_CPU_COLUMNS: &[&str] = &["fsim_y"];

// Versioned **CPU** VSI column name (`vsi::VSI_COLUMN_NAME`, default
// `vsi_imazen_v<…>`, overridable via `VSI_IMPL_TAG`). No `_cpu_` infix
// — no GPU twin. Without `cpu-vsi` bare `"vsi"`.
#[cfg(feature = "cpu-vsi")]
const VSI_CPU_COLUMNS: &[&str] = &[vsi::VSI_COLUMN_NAME];
#[cfg(not(feature = "cpu-vsi"))]
const VSI_CPU_COLUMNS: &[&str] = &["vsi"];

// Versioned **CPU** MS-SSIM column name (`msssim::MSSSIM_COLUMN_NAME`,
// default `msssim_imazen_v<…>`, overridable via `MSSSIM_IMPL_TAG`).
// Without `cpu-msssim` bare `"msssim"`.
#[cfg(feature = "cpu-msssim")]
const MSSSIM_CPU_COLUMNS: &[&str] = &[msssim::MSSSIM_COLUMN_NAME];
#[cfg(not(feature = "cpu-msssim"))]
const MSSSIM_CPU_COLUMNS: &[&str] = &["msssim"];

#[cfg(feature = "cpu-vif")]
const VIF_CPU_COLUMNS: &[&str] = &[vif::VIF_COLUMN_NAME];
#[cfg(not(feature = "cpu-vif"))]
const VIF_CPU_COLUMNS: &[&str] = &["vif"];

// Versioned CPU vifvec column (`vif::VIFVEC_COLUMN_NAME`, default
// `vifvec_imazen_v*`, overridable via `VIFVEC_IMPL_TAG`). Without
// `cpu-vifvec` a bare `"vifvec"`.
#[cfg(feature = "cpu-vifvec")]
const VIFVEC_CPU_COLUMNS: &[&str] = &[vif::VIFVEC_COLUMN_NAME];
#[cfg(not(feature = "cpu-vifvec"))]
const VIFVEC_CPU_COLUMNS: &[&str] = &["vifvec"];

// Versioned CPU iwssim-piq column (`iwssim::IWSSIM_PIQ_COLUMN_NAME`,
// default `iwssim_piq_imazen_v*`, overridable via `IWSSIM_PIQ_IMPL_TAG`).
// Without `cpu-iwssim-piq` a bare `"iwssim_piq"`.
#[cfg(feature = "cpu-iwssim-piq")]
const IWSSIM_PIQ_CPU_COLUMNS: &[&str] = &[iwssim::IWSSIM_PIQ_COLUMN_NAME];
#[cfg(not(feature = "cpu-iwssim-piq"))]
const IWSSIM_PIQ_CPU_COLUMNS: &[&str] = &["iwssim_piq"];

// libvmaf-feature columns — fixed names (they are the vendored libvmaf
// feature names, not versioned `*_imazen_v*` identifiers).
const SSIM_LIBVMAF_CPU_COLUMNS: &[&str] = &["ssim_libvmaf"];
const MSSSIM_LIBVMAF_CPU_COLUMNS: &[&str] = &["msssim_libvmaf"];
const PSNRHVS_DAALA_CPU_COLUMNS: &[&str] = &[
    "psnrhvs_daala_y",
    "psnrhvs_daala_cb",
    "psnrhvs_daala_cr",
    "psnrhvs_daala",
];

// MAD emits three columns: the blend plus the two strategy indices.
#[cfg(feature = "cpu-mad")]
const MAD_CPU_COLUMNS: &[&str] = &[
    mad_iqa::MAD_COLUMN_NAME,
    mad_iqa::MAD_HI_COLUMN_NAME,
    mad_iqa::MAD_LO_COLUMN_NAME,
];
#[cfg(not(feature = "cpu-mad"))]
const MAD_CPU_COLUMNS: &[&str] = &["mad", "mad_hi", "mad_lo"];

// Versioned **CPU** mDCT-PSNR column name
// (`mdctpsnr::MDCTPSNR_COLUMN_NAME`, default
// `mdctpsnr_imazen_v<MAJOR>_<MINOR>_<PATCH>`, overridable via
// `MDCTPSNR_IMPL_TAG`). No `_cpu_` infix — no GPU twin. Without
// `cpu-mdctpsnr` a bare `"mdctpsnr"`.
#[cfg(feature = "cpu-mdctpsnr")]
const MDCTPSNR_CPU_COLUMNS: &[&str] = &[mdctpsnr::MDCTPSNR_COLUMN_NAME];
#[cfg(not(feature = "cpu-mdctpsnr"))]
const MDCTPSNR_CPU_COLUMNS: &[&str] = &["mdctpsnr"];

/// CubeCL runtime selector for GPU metrics.
///
/// Maps onto [`zenmetrics_api::Backend`] inside [`run_gpu_via_umbrella`];
/// kept on the CLI side as its own enum so the `--gpu-runtime` flag
/// surface stays stable and the existing `auto` discovery logic does
/// not leak through to callers that only want a fixed backend.
// `Hash` keys the cvvdp batch scorer's per-runtime liveness-probe cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
pub enum GpuRuntime {
    /// First runtime that initialises successfully.
    Auto,
    /// CUDA. Requires the `gpu-cuda` cargo feature.
    Cuda,
    /// wgpu (Vulkan / Metal / DX12 / WebGPU). Requires `gpu-wgpu`.
    Wgpu,
    /// AMD HIP / ROCm. Requires `gpu-hip`.
    Hip,
    /// Native SIMD CPU (`Backend::Cpu` → `zenmetrics-api::cpu_dispatch`, the
    /// in-tree `cvvdp`/`iwssim` ports + `fast-ssim2`/`dssim`/`butteraugli`/
    /// `zensim`). The canonical CPU path and the auto-ladder fallback. This is
    /// NOT cubecl-cpu — `Backend::CubeclCpu` is never dispatched (panics on
    /// `atomic<f32>`, slow). Needs `cpu-metrics` (default).
    Cpu,
}

#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
/// The `Auto` ladder, in preference order.
///
/// **The final `Cpu` rung is a silent-degradation hazard.** A GPU that is
/// absent, busy, out of memory, or has a driver/toolkit mismatch makes every
/// GPU rung fail, and `Auto` then computes the answer on the CPU and returns
/// it under the *GPU* metric's column name with exit 0 and no log line. The
/// resulting row is indistinguishable from a real GPU measurement after the
/// fact. Measured 2026-08-05: with the GPU hidden, `--gpu-runtime cuda` exits
/// 1 and refuses, while bare `Auto` exits 0 and emits `cvvdp_imazen_v0_0_1`
/// from a CPU computation.
///
/// Set **`ZENMETRICS_REQUIRE_GPU=1`** to drop the CPU rung, so a GPU-less
/// machine fails loudly instead of quietly producing CPU numbers. This is the
/// one lever that covers all three entry points at once — the hand-run
/// `score`/`score-pairs` path, the sweep cache, and `jobexec` (the fleet
/// path) — because all of them resolve their ladder through this function.
/// The fleet cannot pass `--gpu-runtime` at all, so the env var is the only
/// way to make fleet scoring GPU-guaranteed.
pub(crate) fn auto_order() -> &'static [GpuRuntime] {
    const WITH_CPU: &[GpuRuntime] = &[
        GpuRuntime::Cuda,
        GpuRuntime::Wgpu,
        GpuRuntime::Hip,
        GpuRuntime::Cpu,
    ];
    const GPU_ONLY: &[GpuRuntime] = &[GpuRuntime::Cuda, GpuRuntime::Wgpu, GpuRuntime::Hip];
    if require_gpu() { GPU_ONLY } else { WITH_CPU }
}

/// True when `ZENMETRICS_REQUIRE_GPU` is set to anything other than `0`,
/// `false`, or the empty string. Read once per process and cached — the
/// value must not change mid-run, or two rows in one parquet could have been
/// produced under different guarantees.
///
/// Its only non-test caller is [`auto_order`], which carries the `gpu-*` gate
/// above, so in a build with every `gpu-*` feature off this is dead in the lib
/// while still being live in the unit tests below. Allow the lint exactly in
/// that slice rather than gating the function out, which would break those
/// tests.
#[cfg_attr(
    not(any(
        feature = "gpu-butteraugli",
        feature = "gpu-ssim2",
        feature = "gpu-dssim",
        feature = "gpu-iwssim",
        feature = "gpu-zensim",
        feature = "gpu-cvvdp"
    )),
    allow(dead_code)
)]
pub(crate) fn require_gpu() -> bool {
    static REQUIRE_GPU: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *REQUIRE_GPU
        .get_or_init(|| require_gpu_from_env_value(std::env::var("ZENMETRICS_REQUIRE_GPU").ok()))
}

/// Pure parse of the `ZENMETRICS_REQUIRE_GPU` value, split out from
/// [`require_gpu`] so it is unit-testable — `require_gpu` caches in a
/// `OnceLock`, so a test process can only ever observe one value of it.
///
/// Unset, empty, `0`, `false`, `no`, `off` (any case, surrounding
/// whitespace ignored) mean "allow the CPU rung"; anything else means
/// "require a GPU". Defaulting *unset* to false keeps existing behaviour
/// byte-identical for every caller that has not opted in.
///
/// Same dead-in-the-CPU-only-lib situation as [`require_gpu`], its only
/// non-test caller.
#[cfg_attr(
    not(any(
        feature = "gpu-butteraugli",
        feature = "gpu-ssim2",
        feature = "gpu-dssim",
        feature = "gpu-iwssim",
        feature = "gpu-zensim",
        feature = "gpu-cvvdp"
    )),
    allow(dead_code)
)]
pub(crate) fn require_gpu_from_env_value(v: Option<String>) -> bool {
    match v {
        Some(v) => {
            let v = v.trim().to_ascii_lowercase();
            !(v.is_empty() || v == "0" || v == "false" || v == "no" || v == "off")
        }
        None => false,
    }
}

/// The runtime that actually executed the most recent GPU-umbrella metric on
/// **this thread**, as a stable lowercase label (`cuda` / `wgpu` / `hip` /
/// `cpu`), or `"unknown"` before any run.
///
/// This is the durable half of the fix. `ZENMETRICS_REQUIRE_GPU` protects
/// *future runs*; a recorded runtime protects *future readers*, who would
/// otherwise have to infer the backend from a column name that lies when
/// `Auto` degraded. `MetricKind::backend()` cannot serve this purpose — it
/// returns a static string from the enum variant, so it reports "GPU" for a
/// run that fell through to the CPU rung.
///
/// Thread-local because `score-pairs` scores pairs in parallel; a process-wide
/// cell would race and attribute one thread's runtime to another's row.
///
/// `src/metrics/` is compiled into BOTH crate targets, and the only caller is
/// the `zenmetrics` bin (`main.rs`, the `score-pairs` row assembly). The lib
/// target therefore never calls it in any feature combination, so this is an
/// unconditional allow rather than a cfg-conditional one.
#[allow(dead_code)]
pub(crate) fn last_runtime_label() -> &'static str {
    LAST_RUNTIME.with(|c| c.get())
}

thread_local! {
    static LAST_RUNTIME: std::cell::Cell<&'static str> = const { std::cell::Cell::new("unknown") };
}

/// Record the runtime that just produced a score on this thread.
///
/// Gated exactly like [`runtime_label`], which it calls: both call sites
/// (`run_gpu_via_umbrella` here and the `gpu-cvvdp` cascade in `cvvdp_gpu.rs`)
/// live behind the same `gpu-*` gate, so a CPU-only build has no caller. Without
/// the gate this body still referenced the gated `runtime_label`, so any build
/// with none of the `gpu-*` features on — `--features wgpu,all-metrics`, which
/// is what the Lint and Doctests jobs use — failed with
/// `E0425: cannot find function \`runtime_label\` in this scope`.
#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
pub(crate) fn record_runtime(rt: GpuRuntime) {
    LAST_RUNTIME.with(|c| c.set(runtime_label(rt)));
}

#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
pub(crate) fn runtime_label(rt: GpuRuntime) -> &'static str {
    match rt {
        GpuRuntime::Auto => "auto",
        GpuRuntime::Cuda => "cuda",
        GpuRuntime::Wgpu => "wgpu",
        GpuRuntime::Hip => "hip",
        GpuRuntime::Cpu => "cpu",
    }
}

/// Translate a CLI [`GpuRuntime`] selection into the umbrella's
/// [`zenmetrics_api::Backend`]. `Auto` is expanded by the caller.
#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
pub(crate) fn gpu_runtime_to_backend(rt: GpuRuntime) -> Result<zenmetrics_api::Backend, String> {
    match rt {
        GpuRuntime::Auto => Err("Auto is expanded by the caller".to_string()),
        GpuRuntime::Cuda => Ok(zenmetrics_api::Backend::Cuda),
        GpuRuntime::Wgpu => Ok(zenmetrics_api::Backend::Wgpu),
        GpuRuntime::Hip => Ok(zenmetrics_api::Backend::Hip),
        // PROJECT RULE: cubecl-cpu (`Backend::CubeclCpu`) is NEVER dispatched
        // (it panics on `atomic<f32>` for cvvdp and is slow for the rest).
        // `GpuRuntime::Cpu` maps to the NATIVE SIMD CPU path (`Backend::Cpu`
        // → `zenmetrics-api::cpu_dispatch`). So both the auto-ladder's final
        // rung AND an explicit `--gpu-runtime cpu` score on the native ports,
        // never cubecl-cpu. (C2; see docs/METRIC_DISPATCH_CONSOLIDATION.md.)
        GpuRuntime::Cpu => Ok(zenmetrics_api::Backend::Cpu),
    }
}

/// Process-wide flag toggled by `score-pairs --allow-small-images`.
/// Read by [`resolve_default_params`] to switch iwssim into adaptive
/// reflect-pad mode at metric construction. Defaults to false; tests
/// never set it. Using `AtomicBool` so we can flip-then-spawn-threads
/// without a `&mut`.
#[allow(dead_code)] // used by the --allow-small-images CLI path (feature-gated build configs)
static ALLOW_SMALL_IMAGES: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Set the process-wide allow-small-images flag. Called once from
/// `cmd_score_pairs` when `--allow-small-images` is on the CLI. Safe
/// to call from non-main if scoping changes.
#[allow(dead_code)]
pub fn set_allow_small_images() {
    ALLOW_SMALL_IMAGES.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Read the flag. Used by [`resolve_default_params`] and re-exposed
/// for the integration tests' debug paths.
#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
fn allow_small_images() -> bool {
    ALLOW_SMALL_IMAGES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Resolve the default [`zenmetrics_api::MetricParams`] for a kind,
/// applying per-CLI overrides from the CLI's process-wide flags where
/// it exposes them. Currently this is just `--allow-small-images` for
/// the iwssim metric.
#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp",
    feature = "cpu-cvvdp",
    feature = "cpu-iwssim",
    feature = "cpu-hdrvdp"
))]
pub(crate) fn resolve_default_params(
    kind: zenmetrics_api::MetricKind,
) -> Result<zenmetrics_api::MetricParams, zenmetrics_api::Error> {
    #[cfg(feature = "gpu-iwssim")]
    {
        if matches!(kind, zenmetrics_api::MetricKind::Iwssim) && allow_small_images() {
            return Ok(zenmetrics_api::MetricParams::Iwssim(
                zenmetrics_api::iwssim::IwssimParams::allow_small(true),
            ));
        }
    }
    zenmetrics_api::MetricParams::try_default_for(kind)
}

/// Single dispatch point for every GPU metric that fits the
/// "construct → `compute_srgb_u8` → unwrap one `Score`" shape:
/// ssim2-gpu, dssim-gpu, iwssim-gpu, zensim-gpu, cvvdp single-shot,
/// and butter max-norm. `auto` walks the compiled-in runtime list and
/// returns the first that produces a finite score. Replaces the
/// per-metric `score_with_runtime` / `run::<R>` cascade that used to
/// live one file per metric.
#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
fn run_gpu_via_umbrella(
    umbrella_kind: zenmetrics_api::MetricKind,
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    gpu_runtime: GpuRuntime,
) -> Result<f64, Box<dyn std::error::Error>> {
    // ENTRYPOINT PANIC — GPU zensim kernel DISABLED (2026-07-19). No path should
    // reach the GPU zensim pipeline: zensim is CPU features-only pending v2
    // GPU-port validation. A Zensim dispatch here is a bug; fail LOUD rather than
    // run the stale v1 kernel and poison the training set.
    assert!(
        !matches!(umbrella_kind, zenmetrics_api::MetricKind::Zensim),
        "GPU zensim is DISABLED (2026-07-19): run_gpu_via_umbrella reached with \
         MetricKind::Zensim — zensim is CPU features-only, no GPU dispatch"
    );
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(format!(
            "{}: reference ({}×{}) and distorted ({}×{}) differ in size",
            umbrella_kind.tag(),
            reference.width,
            reference.height,
            distorted.width,
            distorted.height
        )
        .into());
    }
    let candidates: Vec<GpuRuntime> = match gpu_runtime {
        GpuRuntime::Auto => auto_order().to_vec(),
        other => vec![other],
    };
    // Collect EVERY runtime's failure so the user can see what each one
    // tried — surfacing only the last error hides the actual problem
    // when (e.g.) cuda fails for an interesting reason but cpu trips
    // on a feature-not-enabled error later in the fallback chain.
    let mut errors: Vec<String> = Vec::with_capacity(candidates.len());
    for rt in candidates {
        let backend = match gpu_runtime_to_backend(rt) {
            Ok(b) => b,
            Err(e) => {
                errors.push(format!("{}: {e}", runtime_label(rt)));
                continue;
            }
        };
        // Per-metric default params from the umbrella. The CLI does
        // not expose per-metric tuning today — the score subcommand is
        // a "is the metric wired in" smoke test.
        //
        // Exception: for iwssim, honour the `IWSSIM_ALLOW_SMALL=1`
        // env var by switching to `IwssimParams::allow_small(true)`.
        // The score-pairs `--allow-small-images` flag sets that env var
        // in-process before invoking this dispatcher. Default behaviour
        // (env unset) is unchanged — `IwssimParams::DEFAULT` rejects
        // sub-176 inputs exactly as before.
        let params = match resolve_default_params(umbrella_kind) {
            Ok(p) => p,
            Err(e) => {
                errors.push(format!("{}: {e}", runtime_label(rt)));
                continue;
            }
        };
        let metric = zenmetrics_api::Metric::new(
            umbrella_kind,
            backend,
            reference.width,
            reference.height,
            params,
        );
        let mut m = match metric {
            Ok(m) => m,
            Err(e) => {
                errors.push(format!("{}: {e}", runtime_label(rt)));
                continue;
            }
        };
        match m.compute_srgb_u8(&reference.pixels, &distorted.pixels) {
            Ok(score) => {
                if !score.value.is_finite() {
                    errors.push(format!(
                        "{}: non-finite score {}",
                        runtime_label(rt),
                        score.value
                    ));
                    continue;
                }
                record_runtime(rt);
                return Ok(score.value);
            }
            Err(e) => errors.push(format!("{}: {e}", runtime_label(rt))),
        }
    }
    // Name the reason when the CPU rung was deliberately withheld, so the
    // failure is self-explaining rather than looking like a broken build.
    let gate_note = if require_gpu() && matches!(gpu_runtime, GpuRuntime::Auto) {
        " (ZENMETRICS_REQUIRE_GPU is set, so the CPU fallback rung was \
         withheld — this run refused to silently compute on the CPU and \
         report it under a GPU column; unset it to allow CPU fallback)"
    } else {
        ""
    };
    Err(format!(
        "{}: no runtime succeeded; tried [{}]{gate_note}",
        umbrella_kind.tag(),
        if errors.is_empty() {
            "none".to_string()
        } else {
            errors.join("; ")
        }
    )
    .into())
}

/// Score `umbrella_kind` on the NATIVE CPU backend (`Backend::Cpu` →
/// `zenmetrics-api::cpu_dispatch`, the in-tree SIMD `cvvdp`/`iwssim` ports) —
/// NOT the cubecl-on-CPU path (which panics on `atomic<f32>` for cvvdp). This
/// is the failover target for cvvdp/iwssim when no GPU runtime succeeds, and
/// the sole path in a CPU-only build. Wired by the `cpu-cvvdp` / `cpu-iwssim`
/// features (pulled into the default `cpu-metrics` bundle). See
/// docs/METRIC_DISPATCH_CONSOLIDATION.md.
#[cfg(any(feature = "cpu-cvvdp", feature = "cpu-iwssim", feature = "cpu-hdrvdp"))]
pub(crate) fn run_cpu_native_via_umbrella(
    umbrella_kind: zenmetrics_api::MetricKind,
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(format!(
            "{}: reference ({}×{}) and distorted ({}×{}) differ in size",
            umbrella_kind.tag(),
            reference.width,
            reference.height,
            distorted.width,
            distorted.height
        )
        .into());
    }
    let params = resolve_default_params(umbrella_kind)?;
    let mut m = zenmetrics_api::Metric::new(
        umbrella_kind,
        zenmetrics_api::Backend::Cpu,
        reference.width,
        reference.height,
        params,
    )?;
    let score = m.compute_srgb_u8(&reference.pixels, &distorted.pixels)?;
    if !score.value.is_finite() {
        return Err(format!(
            "{} (cpu): non-finite score {}",
            umbrella_kind.tag(),
            score.value
        )
        .into());
    }
    Ok(score.value)
}

/// CPU butteraugli via the umbrella, returning BOTH the max-norm and the
/// libjxl 3-norm (`pnorm_3`) from one `compute_srgb_u8_multi` call so the
/// two-column TSV schema is preserved without a second pass. The single
/// CPU butteraugli construct+translate lives in `zenmetrics-api::cpu_dispatch`.
#[cfg(feature = "cpu-metrics")]
pub(crate) fn run_cpu_butter_both_via_umbrella(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(format!(
            "butteraugli: reference ({}×{}) and distorted ({}×{}) differ in size",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let params = resolve_default_params(zenmetrics_api::MetricKind::Butter)?;
    let mut m = zenmetrics_api::Metric::new(
        zenmetrics_api::MetricKind::Butter,
        zenmetrics_api::Backend::Cpu,
        reference.width,
        reference.height,
        params,
    )?;
    let scores = m.compute_srgb_u8_multi(&reference.pixels, &distorted.pixels)?;
    let max = scores.primary();
    let pnorm3 = scores
        .get("pnorm_3")
        .ok_or("butteraugli (cpu): pnorm_3 scalar missing from umbrella output")?;
    if !max.is_finite() || !pnorm3.is_finite() {
        return Err(format!("butteraugli (cpu): non-finite (max={max}, pnorm3={pnorm3})").into());
    }
    Ok((max, pnorm3))
}

/// A cvvdp scorer bound to one named display — the CPU port (`cvvdp`) or
/// the GPU pipeline (`cvvdp-gpu`). Caches its instance across same-sized
/// pairs. This is the only SDR cvvdp scoring route: there is no default
/// display.
pub(crate) enum CvvdpScorer {
    /// Native CPU port.
    #[cfg(feature = "cpu-cvvdp")]
    Cpu(cvvdp_cpu::CpuCvvdpDisplayScorer),
    /// GPU pipeline.
    #[cfg(feature = "gpu-cvvdp")]
    Gpu {
        /// The cached GPU scorer.
        scorer: Box<cvvdp_gpu::CvvdpBatchScorer>,
        /// Output column (`<gpu impl column>_<display>`).
        column: &'static str,
    },
}

impl CvvdpScorer {
    /// Build a scorer for `kind` (`Cvvdp` or `CvvdpGpu`) at the named
    /// upstream display preset. Errors for any other kind, an unknown
    /// preset, or a backend this build lacks.
    pub(crate) fn new(
        kind: MetricKind,
        display: &CvvdpDisplay,
        gpu_runtime: GpuRuntime,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let _ = (display, gpu_runtime);
        match kind {
            #[cfg(feature = "cpu-cvvdp")]
            MetricKind::Cvvdp => Ok(Self::Cpu(cvvdp_cpu::CpuCvvdpDisplayScorer::new(
                display.clone(),
            ))),
            #[cfg(not(feature = "cpu-cvvdp"))]
            MetricKind::Cvvdp => Err(disabled_msg("cvvdp", "cpu-cvvdp` (native SIMD CPU)")),
            #[cfg(feature = "gpu-cvvdp")]
            MetricKind::CvvdpGpu => Ok(Self::Gpu {
                scorer: Box::new(cvvdp_gpu::CvvdpBatchScorer::new_with_target(
                    gpu_runtime,
                    cvvdp_gpu::DisplayTarget::from_display(display),
                )?),
                column: kind.columns_for(Some(display))[0],
            }),
            #[cfg(not(feature = "gpu-cvvdp"))]
            MetricKind::CvvdpGpu => Err(disabled_msg("cvvdp-gpu", "gpu-cvvdp")),
            other => Err(format!("{} is not a cvvdp metric", other.name()).into()),
        }
    }

    /// The display-named output column.
    pub(crate) fn column(&self) -> &'static str {
        match *self {
            #[cfg(feature = "cpu-cvvdp")]
            Self::Cpu(ref s) => s.column(),
            #[cfg(feature = "gpu-cvvdp")]
            Self::Gpu { column, .. } => column,
        }
    }

    /// Score one sRGB8 pair (JOD, 10 = identical).
    pub(crate) fn score(
        &mut self,
        reference: &Rgb8Image,
        distorted: &Rgb8Image,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let _ = (reference, distorted);
        match *self {
            #[cfg(feature = "cpu-cvvdp")]
            Self::Cpu(ref mut s) => s.score(reference, distorted),
            #[cfg(feature = "gpu-cvvdp")]
            Self::Gpu { ref mut scorer, .. } => scorer.score(reference, distorted),
        }
    }
}

/// [`run_metric`] with the cvvdp display selection. cvvdp (CPU `cvvdp`,
/// GPU `cvvdp-gpu`) REQUIRES `display` — a named upstream preset
/// (photometry AND geometry) — and reports `<impl column>_<display>`
/// ([`MetricKind::columns_for`]); there is no default. Every other metric
/// ignores `display` and is exactly [`run_metric`]. Builds a fresh cvvdp
/// scorer per call; loops should hold a [`CvvdpScorer`].
pub fn run_metric_display(
    kind: MetricKind,
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    gpu_runtime: GpuRuntime,
    display: Option<&CvvdpDisplay>,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if !kind.needs_display() {
        return run_metric(kind, reference, distorted, gpu_runtime);
    }
    let display = display.ok_or_else(|| display_required_msg(kind))?;
    let mut scorer = CvvdpScorer::new(kind, display, gpu_runtime)?;
    let v = scorer.score(reference, distorted)?;
    Ok(vec![(scorer.column(), v)])
}

/// [`run_metric`] for the HDR **u8 shell** (reference-peak-normalized
/// sRGB8 for cvvdp, PU21 sRGB8 for the rest). cvvdp keeps its historical
/// `standard_4k` display here, named explicitly (cvvdp has no default), and
/// reports the display-less [`MetricKind::column_names`] column the HDR
/// routes have always written; every other metric is exactly [`run_metric`].
pub fn run_metric_hdr_u8(
    kind: MetricKind,
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    gpu_runtime: GpuRuntime,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    match kind {
        #[cfg(feature = "cpu-cvvdp")]
        MetricKind::Cvvdp => {
            let v =
                cvvdp_cpu::CpuCvvdpDisplayScorer::new(display::DisplayPreset::Standard4k.into())
                    .score(reference, distorted)?;
            Ok(vec![(kind.column_names()[0], v)])
        }
        #[cfg(feature = "gpu-cvvdp")]
        MetricKind::CvvdpGpu => {
            let v = cvvdp_gpu::CvvdpBatchScorer::new_with_target(
                gpu_runtime,
                cvvdp_gpu::DisplayTarget::from_display(&display::DisplayPreset::Standard4k.into()),
            )?
            .score(reference, distorted)?;
            Ok(vec![(kind.column_names()[0], v)])
        }
        _ => run_metric(kind, reference, distorted, gpu_runtime),
    }
}

/// Run `kind` on a `(reference, distorted)` RGB8 pair. GPU metrics route
/// `gpu_runtime` through the CubeCL backend; CPU metrics ignore it.
///
/// Returns `(column_name, value)` pairs in the order declared by
/// [`MetricKind::column_names`]. For most metrics this is a single pair;
/// for butteraugli (CPU and GPU) it is two pairs (max-norm + 3-norm) yielded
/// from a single `compute()` call so callers don't pay twice.
pub fn run_metric(
    kind: MetricKind,
    #[cfg_attr(
        not(any(
            feature = "cpu-metrics",
            feature = "nn",
            feature = "gpu-butteraugli",
            feature = "gpu-ssim2",
            feature = "gpu-dssim",
            feature = "gpu-iwssim",
            feature = "gpu-zensim",
            feature = "gpu-cvvdp"
        )),
        allow(unused_variables)
    )]
    reference: &Rgb8Image,
    #[cfg_attr(
        not(any(
            feature = "cpu-metrics",
            feature = "nn",
            feature = "gpu-butteraugli",
            feature = "gpu-ssim2",
            feature = "gpu-dssim",
            feature = "gpu-iwssim",
            feature = "gpu-zensim",
            feature = "gpu-cvvdp"
        )),
        allow(unused_variables)
    )]
    distorted: &Rgb8Image,
    #[cfg_attr(
        not(any(
            feature = "gpu-butteraugli",
            feature = "gpu-ssim2",
            feature = "gpu-dssim",
            feature = "gpu-iwssim",
            feature = "gpu-zensim",
            feature = "gpu-cvvdp"
        )),
        allow(unused_variables)
    )]
    gpu_runtime: GpuRuntime,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    match kind {
        #[cfg(feature = "cpu-vmaf")]
        MetricKind::Vmaf => Ok(vec![(
            "vmaf",
            vmaf::score(
                reference,
                distorted,
                vmaf::Model::V061,
                vmaf::Ingress::Bt709,
            )?,
        )]),
        #[cfg(not(feature = "cpu-vmaf"))]
        MetricKind::Vmaf => Err(disabled_msg("vmaf", "cpu-vmaf")),
        #[cfg(feature = "cpu-vmaf")]
        MetricKind::VmafNeg => Ok(vec![(
            "vmaf_neg",
            vmaf::score(
                reference,
                distorted,
                vmaf::Model::V061Neg,
                vmaf::Ingress::Bt709,
            )?,
        )]),
        #[cfg(not(feature = "cpu-vmaf"))]
        MetricKind::VmafNeg => Err(disabled_msg("vmaf-neg", "cpu-vmaf")),
        #[cfg(feature = "cpu-vmaf")]
        MetricKind::Vmaf4k => Ok(vec![(
            "vmaf_4k",
            vmaf::score(
                reference,
                distorted,
                vmaf::Model::V061FourK,
                vmaf::Ingress::Bt709,
            )?,
        )]),
        #[cfg(not(feature = "cpu-vmaf"))]
        MetricKind::Vmaf4k => Err(disabled_msg("vmaf-4k", "cpu-vmaf")),
        #[cfg(feature = "cpu-vmaf")]
        MetricKind::VmafV1 => Ok(vec![(
            "vmaf_v1",
            vmaf::score(reference, distorted, vmaf::Model::V1, vmaf::Ingress::Bt709)?,
        )]),
        #[cfg(not(feature = "cpu-vmaf"))]
        MetricKind::VmafV1 => Err(disabled_msg("vmaf-v1", "cpu-vmaf")),
        #[cfg(feature = "cpu-vmaf")]
        MetricKind::VmafStudio601 => Ok(vec![(
            "vmaf_studio601",
            vmaf::score(
                reference,
                distorted,
                vmaf::Model::V061,
                vmaf::Ingress::Studio601,
            )?,
        )]),
        #[cfg(not(feature = "cpu-vmaf"))]
        MetricKind::VmafStudio601 => Err(disabled_msg("vmaf-studio601", "cpu-vmaf")),
        #[cfg(feature = "cpu-vmaf")]
        MetricKind::VmafNegStudio601 => Ok(vec![(
            "vmaf_neg_studio601",
            vmaf::score(
                reference,
                distorted,
                vmaf::Model::V061Neg,
                vmaf::Ingress::Studio601,
            )?,
        )]),
        #[cfg(not(feature = "cpu-vmaf"))]
        MetricKind::VmafNegStudio601 => Err(disabled_msg("vmaf-neg-studio601", "cpu-vmaf")),
        #[cfg(feature = "cpu-metrics")]
        MetricKind::Psnr => Ok(vec![(
            "psnr",
            classical::score(classical::Kind::Psnr, reference, distorted)?,
        )]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Psnr => Err(disabled_msg("psnr", "cpu-metrics")),
        #[cfg(feature = "cpu-metrics")]
        MetricKind::PsnrY => Ok(vec![(
            "psnr_y",
            classical::score(classical::Kind::PsnrY, reference, distorted)?,
        )]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::PsnrY => Err(disabled_msg("psnr-y", "cpu-metrics")),
        #[cfg(feature = "cpu-metrics")]
        MetricKind::PsnrY601 => Ok(vec![(
            "psnr_y601",
            classical::score(classical::Kind::PsnrY601, reference, distorted)?,
        )]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::PsnrY601 => Err(disabled_msg("psnr-y601", "cpu-metrics")),
        #[cfg(feature = "cpu-metrics")]
        MetricKind::PsnrYStudio601 => Ok(vec![(
            "psnr_y_studio601",
            classical::score(classical::Kind::PsnrYStudio601, reference, distorted)?,
        )]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::PsnrYStudio601 => Err(disabled_msg("psnr-y-studio601", "cpu-metrics")),
        #[cfg(feature = "cpu-metrics")]
        MetricKind::PsnrYLibvmaf => Ok(vec![(
            "psnr_y_libvmaf",
            classical::score(classical::Kind::PsnrYLibvmaf, reference, distorted)?,
        )]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::PsnrYLibvmaf => Err(disabled_msg("psnr-y-libvmaf", "cpu-metrics")),
        #[cfg(feature = "cpu-metrics")]
        MetricKind::PsnrYCbCr611 => Ok(vec![(
            "psnr_ycbcr611",
            classical::score(classical::Kind::PsnrYCbCr611, reference, distorted)?,
        )]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::PsnrYCbCr611 => Err(disabled_msg("psnr-ycbcr611", "cpu-metrics")),
        #[cfg(feature = "cpu-metrics")]
        MetricKind::PsnrCb => Ok(vec![(
            "psnr_cb",
            classical::score(classical::Kind::PsnrCb, reference, distorted)?,
        )]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::PsnrCb => Err(disabled_msg("psnr-cb", "cpu-metrics")),
        #[cfg(feature = "cpu-metrics")]
        MetricKind::PsnrCr => Ok(vec![(
            "psnr_cr",
            classical::score(classical::Kind::PsnrCr, reference, distorted)?,
        )]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::PsnrCr => Err(disabled_msg("psnr-cr", "cpu-metrics")),
        #[cfg(feature = "cpu-metrics")]
        MetricKind::Ciede2000 => Ok(vec![(
            "ciede2000",
            classical::score(classical::Kind::Ciede2000, reference, distorted)?,
        )]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Ciede2000 => Err(disabled_msg("ciede2000", "cpu-metrics")),
        #[cfg(feature = "cpu-metrics")]
        MetricKind::SsimC => Ok(vec![(
            "ssimc",
            classical::score(classical::Kind::SsimC, reference, distorted)?,
        )]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::SsimC => Err(disabled_msg("ssimc", "cpu-metrics")),
        #[cfg(feature = "cpu-metrics")]
        MetricKind::Ssim => Ok(vec![(
            "ssim",
            classical::score(classical::Kind::Ssim, reference, distorted)?,
        )]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Ssim => Err(disabled_msg("ssim", "cpu-metrics")),
        #[cfg(feature = "cpu-metrics")]
        MetricKind::Nlpd => Ok(vec![(
            "nlpd",
            nlpd::score_rgb_u8(
                &reference.pixels,
                &distorted.pixels,
                reference.width as usize,
                reference.height as usize,
            )?,
        )]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Nlpd => Err(disabled_msg("nlpd", "cpu-metrics")),
        #[cfg(feature = "cpu-metrics")]
        MetricKind::NlpdIqa => Ok(vec![(
            "nlpd_iqa",
            nlpd::iqa::score_rgb_u8(
                &reference.pixels,
                &distorted.pixels,
                reference.width as usize,
                reference.height as usize,
            )?,
        )]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::NlpdIqa => Err(disabled_msg("nlpd-iqa", "cpu-metrics")),
        #[cfg(feature = "nn")]
        MetricKind::LpipsAlex
        | MetricKind::LpipsVgg
        | MetricKind::LpipsSqueeze
        | MetricKind::Dists
        | MetricKind::StlpipsAlex
        | MetricKind::StlpipsVgg
        | MetricKind::Deepdc
        | MetricKind::Ckdn
        | MetricKind::Afine
        | MetricKind::Dreamsim
        | MetricKind::Arniqa
        | MetricKind::Maniqa
        | MetricKind::Liqe
        | MetricKind::Musiq
        | MetricKind::Hyperiqa
        | MetricKind::Paq2piq
        | MetricKind::Pieapp
        | MetricKind::WadiqamFrLive
        | MetricKind::WadiqamFrTid
        | MetricKind::WadiqamNrLive
        | MetricKind::WadiqamNrTid
        | MetricKind::Wd
        | MetricKind::Ahiq
        | MetricKind::Dmm => nn::score(kind, reference, distorted),
        #[cfg(not(feature = "nn"))]
        MetricKind::LpipsAlex
        | MetricKind::LpipsVgg
        | MetricKind::LpipsSqueeze
        | MetricKind::Dists
        | MetricKind::StlpipsAlex
        | MetricKind::StlpipsVgg
        | MetricKind::Deepdc
        | MetricKind::Ckdn
        | MetricKind::Afine
        | MetricKind::Dreamsim
        | MetricKind::Arniqa
        | MetricKind::Maniqa
        | MetricKind::Liqe
        | MetricKind::Musiq
        | MetricKind::Hyperiqa
        | MetricKind::Paq2piq
        | MetricKind::Pieapp
        | MetricKind::WadiqamFrLive
        | MetricKind::WadiqamFrTid
        | MetricKind::WadiqamNrLive
        | MetricKind::WadiqamNrTid
        | MetricKind::Wd
        | MetricKind::Ahiq
        | MetricKind::Dmm => Err(disabled_msg(kind.name(), "nn")),
        // pyiqa-ports
        #[cfg(feature = "pyiqa-ports")]
        MetricKind::TopiqFr
        | MetricKind::TopiqFrPipal
        | MetricKind::WadiqamPyiqa
        | MetricKind::AhiqPyiqa
        | MetricKind::CwSsimPyiqa
        | MetricKind::DmmPyiqa
        | MetricKind::PieappPyiqa => pyiqa_ports::score(kind, reference, distorted),
        #[cfg(not(feature = "pyiqa-ports"))]
        MetricKind::TopiqFr
        | MetricKind::TopiqFrPipal
        | MetricKind::WadiqamPyiqa
        | MetricKind::AhiqPyiqa
        | MetricKind::CwSsimPyiqa
        | MetricKind::DmmPyiqa
        | MetricKind::PieappPyiqa => Err(disabled_msg(kind.name(), "pyiqa-ports")),
        #[cfg(feature = "cpu-metrics")]
        MetricKind::Ssim2 => Ok(vec![(
            "ssim2",
            run_cpu_native_via_umbrella(zenmetrics_api::MetricKind::Ssim2, reference, distorted)?,
        )]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Ssim2 => Err(disabled_msg("ssim2", "cpu-metrics")),

        #[cfg(feature = "gpu-ssim2")]
        MetricKind::Ssim2Gpu => Ok(vec![(
            "ssim2_gpu",
            run_gpu_via_umbrella(
                zenmetrics_api::MetricKind::Ssim2,
                reference,
                distorted,
                gpu_runtime,
            )?,
        )]),
        #[cfg(not(feature = "gpu-ssim2"))]
        MetricKind::Ssim2Gpu => Err(disabled_msg("ssim2-gpu", "gpu-ssim2")),

        #[cfg(feature = "cpu-metrics")]
        MetricKind::Butteraugli => {
            let (max, pnorm3) = run_cpu_butter_both_via_umbrella(reference, distorted)?;
            Ok(vec![
                ("butteraugli_max", max),
                ("butteraugli_pnorm3", pnorm3),
            ])
        }
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Butteraugli => Err(disabled_msg("butteraugli", "cpu-metrics")),

        #[cfg(feature = "gpu-butteraugli")]
        MetricKind::ButteraugliGpu => {
            // Butteraugli is the only GPU metric the CLI still drives
            // through the typed cubecl-types surface — the opaque
            // `Score` only carries the max-norm, but our TSV emits
            // both max-norm AND the libjxl 3-norm (`pnorm_3`) so the
            // sweep parquet schema stays unchanged. The typed call
            // reaches `ButteraugliResult.pnorm_3` directly. Lives in
            // `butter_pnorm3.rs` to keep this dispatch readable.
            let (max, pnorm3) = butter_pnorm3::score_both(reference, distorted, gpu_runtime)?;
            Ok(vec![
                ("butteraugli_max_gpu", max),
                ("butteraugli_pnorm3_gpu", pnorm3),
            ])
        }
        #[cfg(not(feature = "gpu-butteraugli"))]
        MetricKind::ButteraugliGpu => Err(disabled_msg("butteraugli-gpu", "gpu-butteraugli")),

        #[cfg(feature = "cpu-metrics")]
        MetricKind::Dssim => Ok(vec![(
            "dssim",
            run_cpu_native_via_umbrella(zenmetrics_api::MetricKind::Dssim, reference, distorted)?,
        )]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Dssim => Err(disabled_msg("dssim", "cpu-metrics")),

        #[cfg(feature = "gpu-dssim")]
        MetricKind::DssimGpu => Ok(vec![(
            "dssim_gpu",
            run_gpu_via_umbrella(
                zenmetrics_api::MetricKind::Dssim,
                reference,
                distorted,
                gpu_runtime,
            )?,
        )]),
        #[cfg(not(feature = "gpu-dssim"))]
        MetricKind::DssimGpu => Err(disabled_msg("dssim-gpu", "gpu-dssim")),

        #[cfg(feature = "gpu-iwssim")]
        MetricKind::IwssimGpu => Ok(vec![(
            "iwssim_gpu",
            run_gpu_via_umbrella(
                zenmetrics_api::MetricKind::Iwssim,
                reference,
                distorted,
                gpu_runtime,
            )?,
        )]),
        #[cfg(not(feature = "gpu-iwssim"))]
        MetricKind::IwssimGpu => Err(disabled_msg("iwssim-gpu", "gpu-iwssim")),

        #[cfg(feature = "cpu-metrics")]
        MetricKind::Zensim => Ok(vec![("zensim", zensim::score(reference, distorted)?)]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Zensim => Err(disabled_msg("zensim", "cpu-metrics")),

        // GPU zensim kernel DISABLED (2026-07-19) + zensim is FEATURES-ONLY (no
        // score): a `--metric zensim-gpu` SCORE request has no valid path, so fail
        // LOUD rather than silently produce a v1 number. Extract features via
        // `run_zensim_features` / the jobexec V2Ab path instead.
        MetricKind::ZensimGpu => Err(
            "zensim-gpu is DISABLED: the GPU zensim kernel is off pending v2 \
             GPU-port validation, and zensim is features-only (no score)"
                .into(),
        ),

        // Unsuffixed cvvdp = the native SIMD CPU port (the
        // `ssim2`/`dssim`/`zensim` unsuffixed=CPU convention). Pure CPU
        // path — no GPU fallback; the GPU backend is `cvvdp-gpu`. Errors
        // clearly when `cpu-cvvdp` isn't compiled.
        // cvvdp has no default display: SDR scoring goes through
        // `run_metric_display` (a named display), HDR through
        // `run_metric_hdr_u8` / the faithful HDR routes.
        MetricKind::Cvvdp | MetricKind::CvvdpGpu => Err(display_required_msg(kind).into()),

        // `cvvdp-gpu` = the GPU implementation (the prior `cvvdp`
        // behaviour). Pure GPU path through the umbrella; the
        // `score-pairs` typed `CvvdpBatchScorer` path also keys on this
        // variant. Errors clearly when `gpu-cvvdp` isn't compiled.

        // Unsuffixed iwssim = the native SIMD CPU port. Pure CPU path —
        // the GPU backend is `iwssim-gpu`. Errors when `cpu-iwssim`
        // isn't compiled.
        #[cfg(feature = "cpu-iwssim")]
        MetricKind::Iwssim => Ok(vec![(
            IWSSIM_CPU_COLUMNS[0],
            run_cpu_native_via_umbrella(zenmetrics_api::MetricKind::Iwssim, reference, distorted)?,
        )]),
        #[cfg(not(feature = "cpu-iwssim"))]
        MetricKind::Iwssim => Err(disabled_msg("iwssim", "cpu-iwssim` (native SIMD CPU)")),

        // GMSD: direct call into the in-tree crate (no umbrella, no GPU twin).
        #[cfg(feature = "cpu-gmsd")]
        MetricKind::Gmsd => Ok(vec![(
            GMSD_CPU_COLUMNS[0],
            run_cpu_gmsd(reference, distorted)?,
        )]),
        #[cfg(not(feature = "cpu-gmsd"))]
        MetricKind::Gmsd => Err(disabled_msg("gmsd", "cpu-gmsd")),
        #[cfg(feature = "cpu-gmsd")]
        MetricKind::Mdsi => {
            if reference.width != distorted.width || reference.height != distorted.height {
                return Err("mdsi: reference and distorted dimensions differ".into());
            }
            let (w, h) = (reference.width as usize, reference.height as usize);
            Ok(vec![(
                MDSI_CPU_COLUMNS[0],
                gmsd::mdsi_rgb8(&reference.pixels, &distorted.pixels, w, h, w * 3)?,
            )])
        }
        #[cfg(not(feature = "cpu-gmsd"))]
        MetricKind::Mdsi => Err(disabled_msg("mdsi", "cpu-gmsd")),
        #[cfg(feature = "cpu-gmsd")]
        MetricKind::MsGmsd | MetricKind::MsGmsdc => {
            if reference.width != distorted.width || reference.height != distorted.height {
                return Err("ms-gmsd: reference and distorted dimensions differ".into());
            }
            let (w, h) = (reference.width as usize, reference.height as usize);
            let value = if kind == MetricKind::MsGmsd {
                gmsd::ms_gmsd_rgb8(&reference.pixels, &distorted.pixels, w, h, w * 3)?
            } else {
                gmsd::ms_gmsdc_rgb8(&reference.pixels, &distorted.pixels, w, h, w * 3)?
            };
            Ok(vec![(kind.column_names()[0], value)])
        }
        #[cfg(not(feature = "cpu-gmsd"))]
        MetricKind::MsGmsd | MetricKind::MsGmsdc => Err(disabled_msg("ms-gmsd", "cpu-gmsd")),

        #[cfg(feature = "cpu-hdrvdp")]
        MetricKind::HdrvdpAic2026 | MetricKind::Hdrvdp3Aic2026 => {
            hdrvdp_aic::score(kind, reference, distorted)
        }
        #[cfg(not(feature = "cpu-hdrvdp"))]
        MetricKind::HdrvdpAic2026 | MetricKind::Hdrvdp3Aic2026 => {
            Err(disabled_msg(kind.name(), "cpu-hdrvdp"))
        }
        #[cfg(all(feature = "cpu-metrics", feature = "cpu-vmaf"))]
        MetricKind::CambiAic2026 => Ok(vec![(
            "cambi_aic2026",
            vmaf::cambi_aic2026(reference, distorted)?,
        )]),
        #[cfg(not(all(feature = "cpu-metrics", feature = "cpu-vmaf")))]
        MetricKind::CambiAic2026 => Err(disabled_msg(kind.name(), "cpu-metrics,cpu-vmaf")),
        #[cfg(feature = "cpu-metrics")]
        MetricKind::ProposalButteraugli => {
            let (_, n) = run_cpu_butter_both_via_umbrella(reference, distorted)?;
            Ok(vec![(
                "proposal_butteraugli",
                (0.00459792 * n).max(2.0938201 * n - 1.1033102),
            )])
        }
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::ProposalButteraugli => Err(disabled_msg(kind.name(), "cpu-metrics")),

        // HDR-VDP on 8-bit sRGB: upstream's own `sRGB-display` encoding (hdrvdp_aic.rs). Absolute
        // luminance keeps the `--hdr` sweep/score path.
        #[cfg(feature = "cpu-hdrvdp")]
        MetricKind::Hdrvdp | MetricKind::Hdrvdp3 => {
            hdrvdp_aic::score_srgb_display(kind, reference, distorted)
        }
        #[cfg(not(feature = "cpu-hdrvdp"))]
        MetricKind::Hdrvdp => Err(disabled_msg("hdrvdp", "cpu-hdrvdp` (native SIMD CPU)")),
        #[cfg(not(feature = "cpu-hdrvdp"))]
        MetricKind::Hdrvdp3 => Err(disabled_msg("hdrvdp3", "cpu-hdrvdp` (native SIMD CPU)")),

        // PSNR-HVS: direct call into the in-tree crate (no umbrella, no
        // GPU twin — same shape as GMSD). Both the plain and masked
        // scores come out of one pass, so one call emits both columns.
        #[cfg(feature = "cpu-psnrhvs")]
        MetricKind::Psnrhvs => run_cpu_psnrhvs(reference, distorted),
        #[cfg(not(feature = "cpu-psnrhvs"))]
        MetricKind::Psnrhvs => Err(disabled_msg("psnrhvs", "cpu-psnrhvs")),

        #[cfg(feature = "cpu-psnrhvs")]
        MetricKind::PsnrhvsY => run_cpu_psnrhvs_y(reference, distorted),
        #[cfg(not(feature = "cpu-psnrhvs"))]
        MetricKind::PsnrhvsY => Err(disabled_msg("psnrhvs-y", "cpu-psnrhvs")),

        // HaarPSI: direct call into the in-tree crate (no umbrella, no
        // GPU twin — same shape as GMSD). `haarpsi` scores through the
        // YIQ color path; `haarpsi-y` is the luma-only variant.
        #[cfg(feature = "cpu-haarpsi")]
        MetricKind::Haarpsi => run_cpu_haarpsi(reference, distorted),
        #[cfg(not(feature = "cpu-haarpsi"))]
        MetricKind::Haarpsi => Err(disabled_msg("haarpsi", "cpu-haarpsi")),

        #[cfg(feature = "cpu-haarpsi")]
        MetricKind::HaarpsiY => run_cpu_haarpsi_y(reference, distorted),
        #[cfg(not(feature = "cpu-haarpsi"))]
        MetricKind::HaarpsiY => Err(disabled_msg("haarpsi-y", "cpu-haarpsi")),

        // Standalone CAMBI uses the same studio-601 ingress as the VMAF variants.
        #[cfg(all(feature = "cpu-metrics", feature = "cpu-vmaf"))]
        MetricKind::Cambi => Ok(vec![("cambi", vmaf::cambi(reference, distorted)?)]),
        #[cfg(not(all(feature = "cpu-metrics", feature = "cpu-vmaf")))]
        MetricKind::Cambi => Err(disabled_msg("cambi", "cpu-metrics,cpu-vmaf")),
        MetricKind::ButteraugliOriginal | MetricKind::Ssimulacra1 | MetricKind::WsHv => {
            if (reference.width, reference.height) != (distorted.width, distorted.height) {
                return Err("metric: dimension mismatch".into());
            }
            let w = reference.width as usize;
            let h = reference.height as usize;
            let r = &reference.pixels;
            let d = &distorted.pixels;
            let value = match kind {
                MetricKind::ButteraugliOriginal => {
                    butteraugli_original::error_map(r, d, w, h, w * 3, w * 3)?
                        .into_iter()
                        .fold(0.0_f32, f32::max) as f64
                }
                MetricKind::Ssimulacra1 => ssimulacra1::score(r, d, w, h, w * 3, w * 3)?,
                MetricKind::WsHv => ws_hv::score(r, d, w, h, w * 3, w * 3)?,
                _ => unreachable!(),
            };
            Ok(vec![(kind.column_names()[0], value)])
        }
        MetricKind::CwSsim => {
            if (reference.width, reference.height) != (distorted.width, distorted.height) {
                return Err("CW-SSIM: dimension mismatch".into());
            }
            let w = reference.width as usize;
            let h = reference.height as usize;
            let value = cw_ssim::score(&reference.pixels, &distorted.pixels, w, h, w * 3, w * 3)?;
            Ok(vec![("cw_ssim", value)])
        }
        MetricKind::Flip | MetricKind::FlipV17 => {
            if (reference.width, reference.height) != (distorted.width, distorted.height) {
                return Err("FLIP: dimension mismatch".into());
            }
            let w = reference.width as usize;
            let h = reference.height as usize;
            let map = flip::error_map(&reference.pixels, &distorted.pixels, w, h, w * 3, w * 3)?;
            Ok(if kind == MetricKind::Flip {
                vec![("flip", flip::mean(&map))]
            } else {
                vec![("flip_v17", flip::mean_v17(&map))]
            })
        }
        // SSIM-pyiqa family: valid-window scoring with explicit convention flags.
        MetricKind::SsimPyiqa
        | MetricKind::SsimPyiqaF32
        | MetricKind::SsimPyiqaF32K2
        | MetricKind::SsimPyiqaF32K3
        | MetricKind::SsimPyiqaF32K4
        | MetricKind::SsimPyiqaF32K5
        | MetricKind::SsimPyiqaF32K6
        | MetricKind::SsimcPyiqaF32
        | MetricKind::SsimcPyiqaF32K2
        | MetricKind::SsimcPyiqaF32K3
        | MetricKind::SsimcPyiqaF32K4
        | MetricKind::SsimcPyiqaF32K5
        | MetricKind::SsimcPyiqaF32K6
        | MetricKind::MsSsimPyiqa
        | MetricKind::MsSsimPyiqaF32
        | MetricKind::MsSsimPyiqaF32K2
        | MetricKind::MsSsimPyiqaF32K3
        | MetricKind::MsSsimPyiqaF32K4
        | MetricKind::MsSsimPyiqaF32K5
        | MetricKind::MsSsimPyiqaF32K6 => {
            let (color, multiscale, rounded, factor) = kind
                .ssim_pyiqa_flags()
                .ok_or("missing SSIM-pyiqa configuration")?;
            let value = if multiscale && factor > 1 {
                ssim_pyiqa::score_fixed_multiscale(reference, distorted, color, rounded, factor)?
            } else {
                ssim_pyiqa::score(reference, distorted, color, multiscale, rounded, factor)?
            };
            Ok(vec![(kind.column_names()[0], value)])
        }
        // FSIM/FSIMc: direct call into the in-tree crate (no umbrella,
        // no GPU twin — same shape as GMSD). `fsim` emits both the
        // grayscale and color scores from one pass; `fsim-y` is the
        // luma-only variant.
        #[cfg(feature = "cpu-fsim")]
        MetricKind::Fsim => run_cpu_fsim(reference, distorted),
        #[cfg(feature = "cpu-fsim")]
        MetricKind::FsimK2 => run_cpu_fsim_fixed(reference, distorted, 2, kind.column_names()),
        #[cfg(not(feature = "cpu-fsim"))]
        MetricKind::FsimK2 => Err(disabled_msg("fsim-k2", "cpu-fsim")),
        #[cfg(feature = "cpu-fsim")]
        MetricKind::FsimK3 => run_cpu_fsim_fixed(reference, distorted, 3, kind.column_names()),
        #[cfg(not(feature = "cpu-fsim"))]
        MetricKind::FsimK3 => Err(disabled_msg("fsim-k3", "cpu-fsim")),
        #[cfg(feature = "cpu-fsim")]
        MetricKind::FsimK4 => run_cpu_fsim_fixed(reference, distorted, 4, kind.column_names()),
        #[cfg(not(feature = "cpu-fsim"))]
        MetricKind::FsimK4 => Err(disabled_msg("fsim-k4", "cpu-fsim")),
        #[cfg(feature = "cpu-fsim")]
        MetricKind::FsimNodownscale => {
            run_cpu_fsim_fixed(reference, distorted, 1, kind.column_names())
        }
        #[cfg(not(feature = "cpu-fsim"))]
        MetricKind::FsimNodownscale => Err(disabled_msg("fsim-nodownscale", "cpu-fsim")),

        #[cfg(not(feature = "cpu-fsim"))]
        MetricKind::Fsim => Err(disabled_msg("fsim", "cpu-fsim")),

        #[cfg(feature = "cpu-fsim")]
        MetricKind::FsimY => run_cpu_fsim_y(reference, distorted),
        #[cfg(not(feature = "cpu-fsim"))]
        MetricKind::FsimY => Err(disabled_msg("fsim-y", "cpu-fsim")),

        // VSI: direct call into the in-tree crate (no umbrella, no GPU
        // twin — same shape as GMSD/FSIM).
        #[cfg(feature = "cpu-vsi")]
        MetricKind::Vsi => run_cpu_vsi(reference, distorted),
        #[cfg(feature = "cpu-vsi")]
        MetricKind::VsiK2 => run_cpu_vsi_fixed(reference, distorted, 2, kind.column_names()[0]),
        #[cfg(not(feature = "cpu-vsi"))]
        MetricKind::VsiK2 => Err(disabled_msg("vsi-k2", "cpu-vsi")),
        #[cfg(feature = "cpu-vsi")]
        MetricKind::VsiK3 => run_cpu_vsi_fixed(reference, distorted, 3, kind.column_names()[0]),
        #[cfg(not(feature = "cpu-vsi"))]
        MetricKind::VsiK3 => Err(disabled_msg("vsi-k3", "cpu-vsi")),
        #[cfg(feature = "cpu-vsi")]
        MetricKind::VsiK4 => run_cpu_vsi_fixed(reference, distorted, 4, kind.column_names()[0]),
        #[cfg(not(feature = "cpu-vsi"))]
        MetricKind::VsiK4 => Err(disabled_msg("vsi-k4", "cpu-vsi")),
        #[cfg(feature = "cpu-vsi")]
        MetricKind::VsiNodownscale => {
            run_cpu_vsi_fixed(reference, distorted, 1, kind.column_names()[0])
        }
        #[cfg(not(feature = "cpu-vsi"))]
        MetricKind::VsiNodownscale => Err(disabled_msg("vsi-nodownscale", "cpu-vsi")),

        #[cfg(not(feature = "cpu-vsi"))]
        MetricKind::Vsi => Err(disabled_msg("vsi", "cpu-vsi")),

        // MS-SSIM: same direct-crate shape.
        #[cfg(feature = "cpu-msssim")]
        MetricKind::Msssim => run_cpu_msssim(reference, distorted),
        #[cfg(not(feature = "cpu-msssim"))]
        MetricKind::Msssim => Err(disabled_msg("msssim", "cpu-msssim")),
        #[cfg(feature = "cpu-vif")]
        MetricKind::Vif => run_cpu_vif(reference, distorted),
        #[cfg(not(feature = "cpu-vif"))]
        MetricKind::Vif => Err(disabled_msg("vif", "cpu-vif")),
        #[cfg(feature = "cpu-vifvec")]
        MetricKind::VifVec => run_cpu_vifvec(reference, distorted),
        #[cfg(not(feature = "cpu-vifvec"))]
        MetricKind::VifVec => Err(disabled_msg("vifvec", "cpu-vifvec")),
        // iwssim-piq: same in-tree crate, piq (unrounded YIQ) luma
        // ingress — direct call, same shape as vif/mad.
        #[cfg(feature = "cpu-iwssim-piq")]
        MetricKind::IwssimPiq => Ok(vec![(
            IWSSIM_PIQ_CPU_COLUMNS[0],
            run_cpu_iwssim_piq(reference, distorted)?,
        )]),
        #[cfg(not(feature = "cpu-iwssim-piq"))]
        MetricKind::IwssimPiq => Err(disabled_msg("iwssim-piq", "cpu-iwssim-piq")),
        #[cfg(feature = "cpu-mad")]
        MetricKind::Mad => run_cpu_mad(reference, distorted),
        #[cfg(not(feature = "cpu-mad"))]
        MetricKind::Mad => Err(disabled_msg("mad", "cpu-mad")),

        #[cfg(feature = "cpu-msssim")]
        MetricKind::SsimLibvmaf => run_cpu_ssim_libvmaf(reference, distorted),
        #[cfg(not(feature = "cpu-msssim"))]
        MetricKind::SsimLibvmaf => Err(disabled_msg("ssim-libvmaf", "cpu-msssim")),

        #[cfg(feature = "cpu-msssim")]
        MetricKind::MsssimLibvmaf => run_cpu_msssim_libvmaf(reference, distorted),
        #[cfg(not(feature = "cpu-msssim"))]
        MetricKind::MsssimLibvmaf => Err(disabled_msg("msssim-libvmaf", "cpu-msssim")),

        #[cfg(feature = "cpu-psnrhvs")]
        MetricKind::PsnrhvsDaala => run_cpu_psnrhvs_daala(reference, distorted),
        #[cfg(not(feature = "cpu-psnrhvs"))]
        MetricKind::PsnrhvsDaala => Err(disabled_msg("psnrhvs-daala", "cpu-psnrhvs")),

        // mDCT-PSNR: direct call into the in-tree crate (no umbrella, no
        // GPU twin — same shape as GMSD/MAD).
        #[cfg(feature = "cpu-mdctpsnr")]
        MetricKind::Mdctpsnr => Ok(vec![(
            MDCTPSNR_CPU_COLUMNS[0],
            run_cpu_mdctpsnr(reference, distorted)?,
        )]),
        #[cfg(not(feature = "cpu-mdctpsnr"))]
        MetricKind::Mdctpsnr => Err(disabled_msg("mdctpsnr", "cpu-mdctpsnr")),

        // No-reference metrics: in-tree ports scored through
        // `PairedNoReference` — `piqe` on the distorted image plus
        // `piqe_ref` on the reference (see metrics/nr.rs).
        #[cfg(feature = "nr")]
        MetricKind::Piqe | MetricKind::Niqe | MetricKind::Brisque => {
            nr::score(kind, reference, distorted)
        }
        #[cfg(not(feature = "nr"))]
        MetricKind::Piqe | MetricKind::Niqe | MetricKind::Brisque => {
            Err(disabled_msg(kind.name(), "nr"))
        }
    }
}

/// GMSD of two decoded sRGB8 images (`gmsd::gmsd_rgb8`: BT.601 luma,
/// half-resolution Prewitt GMS map, standard-deviation pooling).
#[cfg(feature = "cpu-gmsd")]
fn run_cpu_gmsd(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(format!(
            "gmsd: dimension mismatch {}x{} vs {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    Ok(gmsd::gmsd_rgb8(&reference.pixels, &distorted.pixels, w, h, w * 3)?.gmsd)
}

/// PSNR-HVS / PSNR-HVS-M of two decoded sRGB8 images — each channel
/// scored as a 0..255 plane, scores averaged across channels.
#[cfg(feature = "cpu-psnrhvs")]
fn run_cpu_psnrhvs(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(format!(
            "psnrhvs: dimension mismatch {}x{} vs {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    let s = psnrhvs::psnrhvs_rgb8(&reference.pixels, &distorted.pixels, w, h, w * 3)?;
    Ok(vec![
        (PSNRHVS_CPU_COLUMNS[0], s.psnr_hvs),
        (PSNRHVS_CPU_COLUMNS[1], s.psnr_hvs_m),
    ])
}

/// PSNR-HVS-Y of two decoded sRGB8 images on BT.601 luma.
#[cfg(feature = "cpu-psnrhvs")]
fn run_cpu_psnrhvs_y(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(format!(
            "psnrhvs-y: dimension mismatch {}x{} vs {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    let s = psnrhvs::psnrhvs_luma8(&reference.pixels, &distorted.pixels, w, h, w * 3)?;
    Ok(vec![
        (PSNRHVSY_CPU_COLUMNS[0], s.psnr_hvs),
        (PSNRHVSY_CPU_COLUMNS[1], s.psnr_hvs_m),
    ])
}

/// HaarPSI of two decoded sRGB8 images (`haarpsi::haarpsi_rgb8`: YIQ
/// color path, subsampled preprocessing).
#[cfg(feature = "cpu-haarpsi")]
fn run_cpu_haarpsi(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(format!(
            "haarpsi: dimension mismatch {}x{} vs {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    let s = haarpsi::haarpsi_rgb8(&reference.pixels, &distorted.pixels, w, h, w * 3)?;
    Ok(vec![(HAARPSI_CPU_COLUMNS[0], s)])
}

/// HaarPSI of two decoded sRGB8 images on unrounded BT.601 luma
/// (`haarpsi::haarpsi_luma8`: grayscale path, no color channel).
#[cfg(feature = "cpu-haarpsi")]
fn run_cpu_haarpsi_y(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(format!(
            "haarpsi-y: dimension mismatch {}x{} vs {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    let s = haarpsi::haarpsi_luma8(&reference.pixels, &distorted.pixels, w, h, w * 3)?;
    Ok(vec![(HAARPSIY_CPU_COLUMNS[0], s)])
}

/// FSIM/FSIMc of two decoded sRGB8 images (`fsim::fsim_rgb8`:
/// log-Gabor phase-congruency + Scharr gradient similarity on luma,
/// I/Q chroma term — one pass emits both columns).
#[cfg(feature = "cpu-fsim")]
fn run_cpu_fsim(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(format!(
            "fsim: dimension mismatch {}x{} vs {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    let s = fsim::fsim_rgb8(&reference.pixels, &distorted.pixels, w, h, w * 3)?;
    Ok(vec![
        (FSIM_CPU_COLUMNS[0], s.fsim),
        (FSIM_CPU_COLUMNS[1], s.fsimc),
    ])
}

/// FSIM of two decoded sRGB8 images on unrounded BT.601 luma
/// (`fsim::fsim_luma8`: grayscale path, no chroma term).
#[cfg(feature = "cpu-fsim")]
fn run_cpu_fsim_y(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(format!(
            "fsim-y: dimension mismatch {}x{} vs {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    let s = fsim::fsim_luma8(&reference.pixels, &distorted.pixels, w, h, w * 3)?;
    Ok(vec![(FSIMY_CPU_COLUMNS[0], s)])
}

/// VSI of two decoded sRGB8 images (`vsi::vsi_rgb8`: SDSP saliency +
/// opponent-channel decimation + Scharr gradients, saliency-weighted
/// pooling).
#[cfg(feature = "cpu-vsi")]
fn run_cpu_vsi(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(format!(
            "vsi: dimension mismatch {}x{} vs {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    let s = vsi::vsi_rgb8(&reference.pixels, &distorted.pixels, w, h, w * 3)?;
    Ok(vec![(VSI_CPU_COLUMNS[0], s)])
}

/// MS-SSIM of two decoded sRGB8 images (`msssim::msssim_rgb8`: 5-level
/// Gaussian-pyramid SSIM on unrounded `rgb2gray` luma).
#[cfg(feature = "cpu-msssim")]
fn run_cpu_msssim(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(format!(
            "msssim: dimension mismatch {}x{} vs {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    let s = msssim::msssim_rgb8(&reference.pixels, &distorted.pixels, w, h, w * 3)?;
    Ok(vec![(MSSSIM_CPU_COLUMNS[0], s)])
}

#[cfg(feature = "cpu-vif")]
fn run_cpu_vif(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(format!(
            "vif: dimension mismatch {}x{} vs {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    let s = vif::vif_rgb8(&reference.pixels, &distorted.pixels, w, h, w * 3)?;
    Ok(vec![(VIF_CPU_COLUMNS[0], s)])
}

/// vifvec of two decoded sRGB8 images (`vif::vifvec_rgb8` — the
/// original steerable-pyramid vector-GSM VIF; u8-rounded
/// `0.299/0.587/0.114` luma ingress matching the published AIC-4
/// `VIF` column).
#[cfg(feature = "cpu-vifvec")]
fn run_cpu_vifvec(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(format!(
            "vifvec: dimension mismatch {}x{} vs {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    let s = vif::vifvec_rgb8(&reference.pixels, &distorted.pixels, w, h, w * 3)?;
    Ok(vec![(VIFVEC_CPU_COLUMNS[0], s)])
}

/// IW-SSIM of two decoded sRGB8 images under the piq ingress —
/// `IwssimParams::piq_luma()` (unrounded `rgb2yiq` channel 0). Same
/// in-tree algorithm as `iwssim`; only the RGB→luma convention
/// differs. Subject to the same `min(W,H) >= 176` pyramid constraint.
#[cfg(feature = "cpu-iwssim-piq")]
fn run_cpu_iwssim_piq(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(format!(
            "iwssim-piq: dimension mismatch {}x{} vs {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let mut m = iwssim::Iwssim::with_params(
        reference.width,
        reference.height,
        iwssim::IwssimParams::piq_luma(),
    )?;
    Ok(m.score(&reference.pixels, &distorted.pixels)?.score)
}

#[cfg(feature = "cpu-mad")]
fn run_cpu_mad(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(format!(
            "mad: dimension mismatch {}x{} vs {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    let s = mad_iqa::mad_rgb8(&reference.pixels, &distorted.pixels, w, h, w * 3)?;
    Ok(vec![
        (MAD_CPU_COLUMNS[0], s.mad),
        (MAD_CPU_COLUMNS[1], s.hi),
        (MAD_CPU_COLUMNS[2], s.lo),
    ])
}

/// libvmaf `float_ssim` of two decoded sRGB8 images on the studio-601
/// luma plane (`msssim::float_ssim`, FFI-verified vs vendored libvmaf
/// f85a8536 — the JPEG AIC-4 `SSIM` column's implementation).
#[cfg(feature = "cpu-msssim")]
fn run_cpu_ssim_libvmaf(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(format!(
            "ssim-libvmaf: dimension mismatch {}x{} vs {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    let yr16 = studio601_y_plane_u16(reference);
    let yd16 = studio601_y_plane_u16(distorted);
    let s = msssim::float_ssim(&yr16, &yd16, w, h, w, 8)?;
    Ok(vec![(SSIM_LIBVMAF_CPU_COLUMNS[0], s)])
}

/// libvmaf `float_ms_ssim` of two decoded sRGB8 images on the studio-601
/// luma plane (`msssim::float_ms_ssim`, FFI-verified — the JPEG AIC-4
/// `MS-SSIM` column's implementation, distinct from `msssim`).
#[cfg(feature = "cpu-msssim")]
fn run_cpu_msssim_libvmaf(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(format!(
            "msssim-libvmaf: dimension mismatch {}x{} vs {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    let yr16 = studio601_y_plane_u16(reference);
    let yd16 = studio601_y_plane_u16(distorted);
    let s = msssim::float_ms_ssim(&yr16, &yd16, w, h, w, 8)?;
    Ok(vec![(MSSSIM_LIBVMAF_CPU_COLUMNS[0], s)])
}

/// Daala/libvmaf `psnr_hvs` of two decoded sRGB8 images on studio-601
/// YUV444 planes (`psnrhvs::daala`, FFI-verified — the JPEG AIC-4
/// `PSNR-HVS` column's implementation and convention: full-res chroma).
/// Emits `psnrhvs_daala_{y,cb,cr}` plus the combined `psnrhvs_daala`.
#[cfg(feature = "cpu-psnrhvs")]
fn run_cpu_psnrhvs_daala(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(format!(
            "psnrhvs-daala: dimension mismatch {}x{} vs {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    if reference.width.min(reference.height) < 8 {
        return Err("psnrhvs-daala: min(w,h) must be ≥ 8 (8×8 block DCT)".into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    let (ry, rcb, rcr) = studio601_yuv444(reference);
    let (dy, dcb, dcr) = studio601_yuv444(distorted);
    let s = psnrhvs::daala::psnr_hvs_daala_yuv444(&ry, &dy, &rcb, &dcb, &rcr, &dcr, w, h, w, w);
    Ok(vec![
        (PSNRHVS_DAALA_CPU_COLUMNS[0], s.psnr_hvs_y),
        (PSNRHVS_DAALA_CPU_COLUMNS[1], s.psnr_hvs_cb),
        (PSNRHVS_DAALA_CPU_COLUMNS[2], s.psnr_hvs_cr),
        (PSNRHVS_DAALA_CPU_COLUMNS[3], s.psnr_hvs),
    ])
}

/// mDCT-PSNR of two decoded sRGB8 images (`mdctpsnr::mdct_psnr_srgb8` —
/// linear BT.601 YCbCr ingress, sliding 8×8 masked-DCT bands, Ahumada
/// detection pooling). dB scale; `+inf` = identical.
#[cfg(feature = "cpu-mdctpsnr")]
fn run_cpu_mdctpsnr(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(format!(
            "mdctpsnr: dimension mismatch {}x{} vs {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    Ok(mdctpsnr::mdct_psnr_srgb8(&reference.pixels, &distorted.pixels, w, h)? as f64)
}

#[allow(dead_code)]
fn disabled_msg(metric: &str, feature: &str) -> Box<dyn std::error::Error> {
    format!("metric '{metric}' is disabled (rebuild with `--features {feature}`)").into()
}

/// Run the zensim metric and additionally return the 300-feature extended
/// vector that the score is derived from. Only the zensim metric exposes a
/// feature vector — other metrics return `None` so callers can still use a
/// uniform call site. Callers that want feature output must therefore include
/// `MetricKind::Zensim` in their metric list.
///
/// Score values match what [`run_metric`] would return for `MetricKind::Zensim`,
/// so a sweep that scores zensim today and migrates to this entry point
/// later sees no shift in the TSV `score_zensim` column.
///
/// Not `sweep`-gated: the jobexec-only executor (`--features jobexec`) scores
/// persisted variants through this entry too — it has no sweep dependency.
#[allow(unused_variables)]
#[allow(dead_code)] // some feature shapes (e.g. jobexec-only bin) don't call it
pub fn run_zensim_with_features(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<(f64, Vec<f64>), Box<dyn std::error::Error>> {
    #[cfg(feature = "cpu-metrics")]
    {
        zensim::score_with_features(reference, distorted)
    }
    #[cfg(not(feature = "cpu-metrics"))]
    {
        Err(disabled_msg("zensim", "cpu-metrics"))
    }
}

/// Selects which zensim feature regime the GPU path emits.
///
/// Maps onto [`zenmetrics_api::zensim::ZensimFeatureRegime`]. Each
/// variant determines the per-cell feature-vector width (228 / 300 /
/// 372) and the parquet sidecar's `feat_<i>` column count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ZensimFeatureRegime {
    /// 228 features — basic + peak only. Matches the legacy
    /// `compute_features` output of CPU zensim's default profile.
    Basic,
    /// 300 features — basic + peak + masked. Matches CPU
    /// `compute_extended_features` (pre-v26 sweep schema).
    Extended,
    /// 372 features — basic + peak + masked + IW. v26+ default for
    /// picker training; supersedes Extended.
    WithIw,
    /// 720 features — the v2 "append-only" regime: v1 `WithIw` (372) ++ the
    /// bounded v2 `feature_v2` block (348). CPU-only (the GPU kernel implements
    /// only v1). The fleet default while GPU zensim is disabled pending v2
    /// GPU-port validation (2026-07-19). Score stays the canonical PreviewV0_2.
    #[value(name = "v2-ab")]
    V2Ab,
    /// 924 features — the folded+append STREAMING regime (zensim C5,
    /// 2026-07-26): folded-720 (v1-basic bands folded into the v2 walk,
    /// f156..f371 emitted as 0) ++ the 204-slot append block. Computed by
    /// `Zensim::compute_folded720_append_features_streaming` with
    /// `ZensimProfile::codec_target()` + default toggles + RAW unpadded
    /// slices — byte-identical to the `v2_ab_extract` foldapp driver.
    /// REGIME PURITY: never column-mix these rows into V2Ab/ext-720/v1
    /// parquets (padded-width divergence + zeroed pools).
    #[value(name = "folded720append")]
    Folded720Append,
    /// 944 features — `Folded720Append` (924) ++ the 20-slot append2 block
    /// ([`zensim` `idx_append2`]: BANDVIS gain/loss, luminance conditioner,
    /// HDR-gated highlight bins — SDR rows carry the HL bins as exact 0).
    /// Same streaming entry as `Folded720Append` with
    /// `V2NewFeatureToggles { append2_block: true, ..default() }` — all other
    /// toggles at their defaults, so the BANDVIS dst-activity plane stays OFF
    /// per the SOTA-944 P1.5 adjudication (zensim
    /// `docs/PLAN_SOTA944_CAMPAIGN_2026-08-01.md`; zensim ≥ `d0616362`).
    /// Slots f0..f923 are BITWISE-IDENTICAL to the `Folded720Append` output
    /// (append2 is additive-only — gated by `folded944_matches_driver_args`).
    /// REGIME PURITY: never column-mix with 924/720/v1 parquets.
    #[value(name = "folded720append2")]
    Folded720Append2,
    /// 944 features — `Folded720Append2` with v1's ten CARRIER slots emitted
    /// LIVE in `f156..372` (`V1PoolsMode::Carriers`, zensim `f19b8469`:
    /// art_l8 f178/190/196/226, masked_art_4th f231/237/243, iw_art_4th
    /// f303/321/333 — bit-identical to v1's 372 at exact widths; every other
    /// pool slot stays 0). Same width as `Folded720Append2`, a DIFFERENT
    /// regime (the `fused944native` one) — REGIME PURITY: never column-mix
    /// with zeroed-block 944 rows; the JSONL `regime` tag is the marker.
    /// SDR-only (no HDR route yet; the HDR arm rejects it explicitly).
    #[value(name = "folded720append2carriers")]
    Folded720Append2Carriers,
    /// 944 features — `Folded720Append2` with **all 216** of v1's pool slots
    /// emitted LIVE in `f156..372` (`V1PoolsMode::Full`, zensim `f19b8469`:
    /// peaks 72 + masked 72 + IW 72 — bit-identical to v1's 372 at exact
    /// widths, gated by zensim's `folded720_v1_pools_match_v1_path`). Same
    /// width as `Folded720Append2` / `Folded720Append2Carriers`, a THIRD
    /// regime: this is the all-944-live extraction (`f0..155` basic ++
    /// `f156..371` pools LIVE ++ v2-348 ++ append2-20) — one pass, one
    /// width, every slot. REGIME PURITY: never column-mix with zeroed-block
    /// or carriers-only 944 rows; the JSONL `regime` tag is the marker.
    /// SDR-only (no HDR route yet; the HDR arm rejects it explicitly).
    #[value(name = "folded720append2pools")]
    Folded720Append2Pools,
}

impl ZensimFeatureRegime {
    /// Feature-vector length (228 / 300 / 372 / 720).
    #[allow(dead_code)]
    pub fn total_features(self) -> usize {
        match self {
            ZensimFeatureRegime::Basic => 228,
            ZensimFeatureRegime::Extended => 300,
            ZensimFeatureRegime::WithIw => 372,
            ZensimFeatureRegime::V2Ab => 372 + 348,
            ZensimFeatureRegime::Folded720Append => 720 + 204,
            ZensimFeatureRegime::Folded720Append2 => 720 + 204 + 20,
            ZensimFeatureRegime::Folded720Append2Carriers => 720 + 204 + 20,
            ZensimFeatureRegime::Folded720Append2Pools => 720 + 204 + 20,
        }
    }
}

#[cfg(feature = "gpu-zensim")]
impl From<ZensimFeatureRegime> for zenmetrics_api::zensim::ZensimFeatureRegime {
    fn from(r: ZensimFeatureRegime) -> Self {
        match r {
            ZensimFeatureRegime::Basic => zenmetrics_api::zensim::ZensimFeatureRegime::Basic,
            ZensimFeatureRegime::Extended => zenmetrics_api::zensim::ZensimFeatureRegime::Extended,
            ZensimFeatureRegime::WithIw => zenmetrics_api::zensim::ZensimFeatureRegime::WithIw,
            // V2Ab has no GPU-umbrella equivalent (the v2 348-feature block is
            // CPU-only). This `From` feeds the GPU path, which is disabled; the
            // CPU `run_zensim_*` path handles V2Ab directly and never converts.
            ZensimFeatureRegime::V2Ab => zenmetrics_api::zensim::ZensimFeatureRegime::WithIw,
            // Folded720Append(2) likewise have no GPU-umbrella equivalent (the
            // folded streaming walk is CPU-only); the CPU path handles them
            // directly and never converts.
            ZensimFeatureRegime::Folded720Append
            | ZensimFeatureRegime::Folded720Append2
            | ZensimFeatureRegime::Folded720Append2Carriers
            | ZensimFeatureRegime::Folded720Append2Pools => {
                zenmetrics_api::zensim::ZensimFeatureRegime::WithIw
            }
        }
    }
}

/// Extract the regime-appropriate zensim FEATURE vector (228 / 300 / 372 / 720)
/// on the **CPU** `zensim` crate — FEATURES ONLY, no score. The GPU zensim kernel
/// is disabled pending v2 GPU-port validation (2026-07-19) and the v2 backfill
/// wants only features (a v2 score head is trained later). Delegates to
/// [`crate::metrics::zensim::extract_features_regime`].
#[cfg(feature = "cpu-metrics")]
#[allow(dead_code)] // library entry point (jobexec + lib consumers); some bins don't call it
pub fn run_zensim_features(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    regime: ZensimFeatureRegime,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    crate::metrics::zensim::extract_features_regime(reference, distorted, regime)
}

#[cfg(not(feature = "cpu-metrics"))]
#[allow(unused_variables, dead_code)]
pub fn run_zensim_features(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    regime: ZensimFeatureRegime,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    Err(disabled_msg("zensim", "cpu-metrics"))
}

#[cfg(feature = "cpu-fsim")]
fn run_cpu_fsim_fixed(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    factor: u32,
    columns: &'static [&'static str],
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err("fsim: dimension mismatch".into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    let scorer = fsim::Fsim::with_downscale(fsim::Downscale::Factor(
        core::num::NonZeroU32::new(factor).expect("fixed nonzero factor"),
    ));
    let scores = scorer.score_rgb8(&reference.pixels, &distorted.pixels, w, h, w * 3)?;
    Ok(vec![(columns[0], scores.fsim), (columns[1], scores.fsimc)])
}

#[cfg(feature = "cpu-vsi")]
fn run_cpu_vsi_fixed(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    factor: u32,
    column: &'static str,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err("vsi: dimension mismatch".into());
    }
    let (w, h) = (reference.width as usize, reference.height as usize);
    let scorer = vsi::Vsi::with_downscale(vsi::Downscale::Factor(
        core::num::NonZeroU32::new(factor).expect("fixed nonzero factor"),
    ));
    Ok(vec![(
        column,
        scorer.score_rgb8(&reference.pixels, &distorted.pixels, w, h, w * 3)?,
    )])
}

#[cfg(test)]
mod tests {
    #[test]
    fn disabled_zensim_gpu_returns_error_without_panicking() {
        use super::*;
        let image = Rgb8Image {
            pixels: vec![128; 64 * 64 * 3],
            width: 64,
            height: 64,
        };
        let error = run_metric(MetricKind::ZensimGpu, &image, &image, GpuRuntime::Cpu)
            .expect_err("disabled scoring must return an error");
        assert!(error.to_string().contains("zensim-gpu is DISABLED"));
    }

    #[test]
    fn all_twenty_ssim_names_select_expected_conventions() {
        use super::*;
        let cases = [
            ("ssim-pyiqa", (false, false, true, 1)),
            ("ssim-pyiqa-f32", (false, false, false, 1)),
            ("ssim-pyiqa-f32-k2", (false, false, false, 2)),
            ("ssim-pyiqa-f32-k3", (false, false, false, 3)),
            ("ssim-pyiqa-f32-k4", (false, false, false, 4)),
            ("ssim-pyiqa-f32-k5", (false, false, false, 5)),
            ("ssim-pyiqa-f32-k6", (false, false, false, 6)),
            ("ssimc-pyiqa-f32", (true, false, false, 1)),
            ("ssimc-pyiqa-f32-k2", (true, false, false, 2)),
            ("ssimc-pyiqa-f32-k3", (true, false, false, 3)),
            ("ssimc-pyiqa-f32-k4", (true, false, false, 4)),
            ("ssimc-pyiqa-f32-k5", (true, false, false, 5)),
            ("ssimc-pyiqa-f32-k6", (true, false, false, 6)),
            ("ms-ssim-pyiqa", (false, true, true, 1)),
            ("ms-ssim-pyiqa-f32", (false, true, false, 1)),
            ("ms-ssim-pyiqa-f32-k2", (false, true, false, 2)),
            ("ms-ssim-pyiqa-f32-k3", (false, true, false, 3)),
            ("ms-ssim-pyiqa-f32-k4", (false, true, false, 4)),
            ("ms-ssim-pyiqa-f32-k5", (false, true, false, 5)),
            ("ms-ssim-pyiqa-f32-k6", (false, true, false, 6)),
        ];
        assert_eq!(cases.len(), 20);
        for (name, expected) in cases {
            let kind = MetricKind::from_str(name, false).unwrap();
            assert_eq!(kind.ssim_pyiqa_flags(), Some(expected), "{name}");
        }
        assert_eq!(MetricKind::Fsim.ssim_pyiqa_flags(), None);
    }

    #[test]
    fn fixed_factor_columns_follow_versioned_base_and_overrides() {
        use super::*;
        for (kind, base, suffix) in [
            (MetricKind::FsimK2, MetricKind::Fsim, "k2"),
            (MetricKind::FsimK3, MetricKind::Fsim, "k3"),
            (MetricKind::FsimK4, MetricKind::Fsim, "k4"),
            (MetricKind::FsimNodownscale, MetricKind::Fsim, "nodownscale"),
            (MetricKind::VsiK2, MetricKind::Vsi, "k2"),
            (MetricKind::VsiK3, MetricKind::Vsi, "k3"),
            (MetricKind::VsiK4, MetricKind::Vsi, "k4"),
            (MetricKind::VsiNodownscale, MetricKind::Vsi, "nodownscale"),
        ] {
            let expected: Vec<String> = base
                .column_names()
                .iter()
                .map(|base| format!("{base}_{suffix}"))
                .collect();
            assert_eq!(kind.column_names(), expected);
        }
        assert_eq!(
            MetricKind::SsimPyiqaF32K2.column_names(),
            &["ssim_pyiqa_f32_k2"]
        );
    }

    use super::*;

    /// `ZENMETRICS_REQUIRE_GPU` value parsing. Unset MUST stay false so the
    /// gate is strictly opt-in and no existing caller changes behaviour.
    #[test]
    fn require_gpu_env_value_parsing() {
        // Off / unset — the CPU rung stays available.
        assert!(!require_gpu_from_env_value(None));
        for off in ["", "  ", "0", "false", "FALSE", "no", "off", " Off "] {
            assert!(
                !require_gpu_from_env_value(Some(off.to_string())),
                "{off:?} should NOT require gpu"
            );
        }
        // On — anything else is opt-in.
        for on in ["1", "true", "TRUE", "yes", "on", "anything"] {
            assert!(
                require_gpu_from_env_value(Some(on.to_string())),
                "{on:?} SHOULD require gpu"
            );
        }
    }

    /// The gate's whole point: with it set, the `Auto` ladder must not end in
    /// a CPU rung, so a GPU-less host fails loudly instead of quietly
    /// reporting CPU numbers under a GPU column name. Asserted structurally
    /// on the two ladders so it holds without any hardware.
    #[cfg(any(
        feature = "gpu-butteraugli",
        feature = "gpu-ssim2",
        feature = "gpu-dssim",
        feature = "gpu-iwssim",
        feature = "gpu-zensim",
        feature = "gpu-cvvdp"
    ))]
    #[test]
    fn auto_order_drops_cpu_rung_only_when_gpu_required() {
        // `auto_order()` reads the process-cached `require_gpu()`, which a
        // test cannot flip twice, so assert on the two ladders the function
        // selects between. Keep in lockstep with `auto_order`.
        const WITH_CPU: &[GpuRuntime] = &[
            GpuRuntime::Cuda,
            GpuRuntime::Wgpu,
            GpuRuntime::Hip,
            GpuRuntime::Cpu,
        ];
        const GPU_ONLY: &[GpuRuntime] = &[GpuRuntime::Cuda, GpuRuntime::Wgpu, GpuRuntime::Hip];

        assert!(
            WITH_CPU.contains(&GpuRuntime::Cpu),
            "permissive ladder must keep the CPU fallback"
        );
        assert!(
            !GPU_ONLY.contains(&GpuRuntime::Cpu),
            "required-GPU ladder must NOT contain a CPU rung — that rung is \
             exactly the silent-degradation path this gate closes"
        );
        // GPU preference order is identical; the gate only truncates.
        assert_eq!(&WITH_CPU[..GPU_ONLY.len()], GPU_ONLY);
        // And the live function returns whichever matches the cached flag.
        let live = auto_order();
        assert_eq!(live, if require_gpu() { GPU_ONLY } else { WITH_CPU });
    }

    #[cfg(feature = "cpu-psnrhvs")]
    #[test]
    fn studio601_yuv444_matches_round_for_every_rgb_triple() {
        for r in 0..=255u8 {
            let mut pixels = Vec::with_capacity(256 * 256 * 3);
            for g in 0..=255u8 {
                for b in 0..=255u8 {
                    pixels.extend_from_slice(&[r, g, b]);
                }
            }
            let img = Rgb8Image {
                pixels,
                width: 256,
                height: 256,
            };
            let (y, cb, cr) = studio601_yuv444(&img);
            for (i, px) in img.pixels.as_chunks::<3>().0.iter().enumerate() {
                let (r, g, b) = (px[0] as f32, px[1] as f32, px[2] as f32);
                let expected = [
                    (16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0)
                        .round()
                        .clamp(0.0, 255.0) as u8,
                    (128.0 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0)
                        .round()
                        .clamp(0.0, 255.0) as u8,
                    (128.0 + (112.0 * r - 93.786 * g - 18.214 * b) / 255.0)
                        .round()
                        .clamp(0.0, 255.0) as u8,
                ];
                assert_eq!([y[i], cb[i], cr[i]], expected, "RGB {px:?}");
            }
        }
    }

    /// `studio601_gray` — the libvmaf/JPEG AIC studio-swing BT.601 luma.
    /// Primary luma values are the classic colour-bar numbers (R=81, G=145,
    /// B=41, white=235, black=16); odd dims even-crop.
    #[test]
    fn studio601_gray_primaries_and_crop() {
        // 3x3: one pixel each of white, black, R / G, B, mid-gray.
        let mut px = Vec::new();
        px.extend_from_slice(&[255, 255, 255]); // white
        px.extend_from_slice(&[0, 0, 0]); // black
        px.extend_from_slice(&[255, 0, 0]); // R
        px.extend_from_slice(&[0, 255, 0]); // G
        px.extend_from_slice(&[0, 0, 255]); // B
        px.extend_from_slice(&[128, 128, 128]); // mid gray
        px.extend_from_slice(&[10, 20, 30]);
        px.extend_from_slice(&[200, 100, 50]);
        px.extend_from_slice(&[1, 2, 3]);
        let img = Rgb8Image {
            pixels: px,
            width: 3,
            height: 3,
        };
        let g = studio601_gray(&img);
        // 3x3 even-crops to 2x2 — the YUV420 constraint this plane mimics.
        assert_eq!((g.width, g.height), (2, 2));
        let y_of = |i: usize| g.pixels[i * 3];
        // Output is the top-left 2x2 of the source: row0 = [white, black],
        // row1 = [G, B] (R / mid-gray were the cropped-away column/row).
        assert_eq!(y_of(0), 235); // white: 16 + 219
        assert_eq!(y_of(1), 16); // black: 16
        assert_eq!(y_of(2), 145); // G: 16 + 128.553
        assert_eq!(y_of(3), 41); // B: 16 + 24.966
        // Broadcast: every channel carries the same Y.
        for px in g.pixels.as_chunks::<3>().0 {
            assert!(px[0] == px[1] && px[1] == px[2]);
        }
    }

    /// `studio601_gray` on a full 2x2 — B and mid-gray landings.
    #[test]
    fn studio601_gray_blue_and_midgray() {
        let mut px = Vec::new();
        px.extend_from_slice(&[0, 0, 255]); // B
        px.extend_from_slice(&[128, 128, 128]); // mid gray
        px.extend_from_slice(&[255, 255, 0]);
        px.extend_from_slice(&[0, 255, 255]);
        let img = Rgb8Image {
            pixels: px,
            width: 2,
            height: 2,
        };
        let g = studio601_gray(&img);
        assert_eq!((g.width, g.height), (2, 2));
        assert_eq!(g.pixels[0], 41); // B: 16 + 24.966
        assert_eq!(g.pixels[3], 126); // mid gray: 16 + 219*128/255 = 125.867 → 126
    }

    /// `luma_ingress_pair` substitutes only for luma-only kinds under
    /// `yuv601-studio`; colour metrics and `house` pass through untouched.
    #[test]
    fn luma_ingress_scoping() {
        let img = || Rgb8Image {
            pixels: vec![255u8, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30],
            width: 2,
            height: 2,
        };
        let (r, d) = (img(), img());
        // Luma-only + studio → substituted.
        for k in [
            MetricKind::Gmsd,
            MetricKind::Msssim,
            MetricKind::Vif,
            MetricKind::Iwssim,
            MetricKind::PsnrY,
            MetricKind::PsnrY601,
        ] {
            let (rs, ds) = luma_ingress_pair(k, LumaIngress::Yuv601Studio, &r, &d);
            assert!(rs.is_some() && ds.is_some(), "{k:?} should substitute");
        }
        // Colour metric + studio → untouched; luma metric + house →
        // untouched; fixed-convention luma variants + studio → untouched
        // (they embed their own swing — a second one isn't idempotent).
        for (k, ing) in [
            (MetricKind::Fsim, LumaIngress::Yuv601Studio),
            (MetricKind::Vsi, LumaIngress::Yuv601Studio),
            (MetricKind::Ssim2, LumaIngress::Yuv601Studio),
            (MetricKind::Cvvdp, LumaIngress::Yuv601Studio),
            (MetricKind::Gmsd, LumaIngress::House),
            (MetricKind::PsnrYStudio601, LumaIngress::Yuv601Studio),
            (MetricKind::PsnrYLibvmaf, LumaIngress::Yuv601Studio),
        ] {
            let (rs, ds) = luma_ingress_pair(k, ing, &r, &d);
            assert!(
                rs.is_none() && ds.is_none(),
                "{k:?}/{ing:?} must NOT substitute"
            );
        }
    }

    /// Regression guard for docs/METRIC_DISPATCH_CONSOLIDATION.md (C1): in ANY
    /// build that compiles the native CPU port (the default — `cpu-metrics`
    /// now forwards to `cpu-cvvdp`/`cpu-iwssim`), cvvdp + iwssim MUST produce a
    /// finite score through `run_metric` on native CPU. This makes "metric X
    /// looks GPU-only / its CPU path is unwired" structurally impossible: if
    /// the wiring regresses, CI fails right here (not at a fleet box at $/hr).
    #[allow(dead_code)]
    fn synth_rgb8(w: u32, h: u32, fill: u8) -> Rgb8Image {
        Rgb8Image {
            pixels: vec![fill; (w as usize) * (h as usize) * 3],
            width: w,
            height: h,
        }
    }

    #[test]
    fn round4_dispatch_columns_and_mismatch() {
        let r = Rgb8Image {
            pixels: (0..64 * 64 * 3)
                .map(|i| ((i * 37 + i / 11) % 256) as u8)
                .collect(),
            width: 64,
            height: 64,
        };
        let wrong = Rgb8Image {
            pixels: vec![0; 63 * 64 * 3],
            width: 63,
            height: 64,
        };
        for (name, kind, column) in [
            (
                "butteraugli-original",
                MetricKind::ButteraugliOriginal,
                "butteraugli_original",
            ),
            ("ssimulacra1", MetricKind::Ssimulacra1, "ssimulacra1"),
            ("ws-hv", MetricKind::WsHv, "ws_hv"),
        ] {
            assert_eq!(MetricKind::from_str(name, false).unwrap(), kind);
            assert!(MetricKind::all().contains(&kind));
            assert_eq!(kind.name(), name);
            assert_eq!(kind.column_names(), &[column]);
            let same = run_metric(kind, &r, &r, GpuRuntime::Auto).unwrap();
            assert_eq!(same[0].0, column);
            assert!((same[0].1 - if kind == MetricKind::WsHv { 1.0 } else { 0.0 }).abs() < 1e-6);
            assert!(run_metric(kind, &r, &wrong, GpuRuntime::Auto).is_err());
        }
    }

    #[test]
    fn flip_and_cw_ssim_dispatch_and_column_contract() {
        let r = Rgb8Image {
            pixels: (0..64 * 64 * 3)
                .map(|i| ((i * 37 + i / 11) % 256) as u8)
                .collect(),
            width: 64,
            height: 64,
        };
        let mut d = Rgb8Image {
            pixels: r.pixels.clone(),
            width: 64,
            height: 64,
        };
        for pixel in d.pixels.iter_mut().step_by(17) {
            *pixel ^= 63;
        }
        for (name, kind, column, identity) in [
            ("flip", MetricKind::Flip, "flip", 0.0),
            ("flip-v17", MetricKind::FlipV17, "flip_v17", 0.0),
            ("cw-ssim", MetricKind::CwSsim, "cw_ssim", 1.0),
        ] {
            assert_eq!(MetricKind::from_str(name, false).unwrap(), kind);
            assert!(MetricKind::all().contains(&kind));
            assert_eq!(kind.name(), name);
            assert_eq!(kind.column_names(), &[column]);
            let same = run_metric(kind, &r, &r, GpuRuntime::Auto).unwrap();
            assert_eq!(same[0].0, column);
            assert!((same[0].1 - identity).abs() < 1e-12);
            let changed = run_metric(kind, &r, &d, GpuRuntime::Auto).unwrap()[0].1;
            assert!(changed.is_finite());
            assert!(if matches!(kind, MetricKind::Flip | MetricKind::FlipV17) {
                changed > identity
            } else {
                changed < identity
            });
        }
        d.width = 63;
        for kind in [MetricKind::Flip, MetricKind::FlipV17, MetricKind::CwSsim] {
            assert!(run_metric(kind, &r, &d, GpuRuntime::Auto).is_err());
        }
    }

    /// FLIP on the dispatch test's 64×64 pair against flip-evaluator 1.7
    /// (`flip_evaluator.evaluate(ref, dist, "LDR")` on the same pixels saved as
    /// PNG): its mean is the f32 row-major sum (`flip-v17`); the f64 sum of its
    /// returned map is `flip`. The two differ by 8.3e-8 here, so 1e-9 separates
    /// the conventions.
    #[test]
    fn flip_and_cw_ssim_match_pinned_reference_values() {
        let r = Rgb8Image {
            pixels: (0..64 * 64 * 3)
                .map(|i| ((i * 37 + i / 11) % 256) as u8)
                .collect(),
            width: 64,
            height: 64,
        };
        let mut d = Rgb8Image {
            pixels: r.pixels.clone(),
            width: 64,
            height: 64,
        };
        for pixel in d.pixels.iter_mut().step_by(17) {
            *pixel ^= 63;
        }
        let v17 = run_metric(MetricKind::FlipV17, &r, &d, GpuRuntime::Auto).unwrap()[0].1;
        let f64_mean = run_metric(MetricKind::Flip, &r, &d, GpuRuntime::Auto).unwrap()[0].1;
        assert!((v17 - 0.07714099436998367).abs() < 1e-9, "flip-v17 {v17}");
        assert!(
            (f64_mean - 0.07714107689184857).abs() < 1e-9,
            "flip {f64_mean}"
        );
        // CW-SSIM against the round-3 Python reference (benchmarks/aic_flip_2026-10-15/
        // cw_variants.py: `fftfreq` grid, explicit f64 BT.601 luma, ties to even).
        let cw = run_metric(MetricKind::CwSsim, &r, &d, GpuRuntime::Auto).unwrap()[0].1;
        assert!((cw - 0.8767816402237913).abs() < 1e-9, "cw-ssim {cw}");
    }

    #[cfg(feature = "cpu-cvvdp")]
    #[test]
    fn cvvdp_scores_on_native_cpu_in_cpu_build() {
        let r = synth_rgb8(176, 176, 128);
        let d = synth_rgb8(176, 176, 116);
        // No default display: plain `run_metric` refuses cvvdp …
        let err = run_metric(MetricKind::Cvvdp, &r, &d, GpuRuntime::Auto)
            .expect_err("cvvdp has no default display");
        assert!(err.to_string().contains("--display-model"), "{err}");
        assert!(run_metric_display(MetricKind::Cvvdp, &r, &d, GpuRuntime::Auto, None).is_err());
        // … and a named display scores on native CPU (no GPU required).
        let out = run_metric_display(
            MetricKind::Cvvdp,
            &r,
            &d,
            GpuRuntime::Auto,
            Some(&display::DisplayPreset::Standard4k.into()),
        )
        .expect("cvvdp must score on native CPU when cpu-cvvdp is compiled (no GPU required)");
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].0,
            format!("{}_standard_4k", MetricKind::Cvvdp.column_names()[0])
        );
        assert!(
            out[0].1.is_finite(),
            "cvvdp native-CPU score must be finite, got {}",
            out[0].1
        );
    }

    #[cfg(feature = "cpu-iwssim")]
    #[test]
    fn iwssim_scores_on_native_cpu_in_cpu_build() {
        let r = synth_rgb8(176, 176, 128);
        let d = synth_rgb8(176, 176, 116);
        let out = run_metric(MetricKind::Iwssim, &r, &d, GpuRuntime::Auto).expect(
            "iwssim must score on native CPU when cpu-iwssim is compiled (no GPU required)",
        );
        assert_eq!(out.len(), 1);
        assert!(
            out[0].1.is_finite(),
            "iwssim native-CPU score must be finite, got {}",
            out[0].1
        );
    }

    /// PROJECT RULE (C2): cubecl-cpu (`Backend::CubeclCpu`) is NEVER dispatched.
    /// No CLI `GpuRuntime` may map to it — `--gpu-runtime cpu` and the auto
    /// ladder's CPU rung both resolve to the native `Backend::Cpu`. CI fails
    /// here if anyone re-points the CPU runtime at cubecl-cpu.
    #[cfg(any(
        feature = "gpu-butteraugli",
        feature = "gpu-ssim2",
        feature = "gpu-dssim",
        feature = "gpu-iwssim",
        feature = "gpu-zensim",
        feature = "gpu-cvvdp"
    ))]
    #[test]
    fn no_gpu_runtime_maps_to_cubecl_cpu() {
        for rt in [
            GpuRuntime::Cuda,
            GpuRuntime::Wgpu,
            GpuRuntime::Hip,
            GpuRuntime::Cpu,
        ] {
            if let Ok(b) = gpu_runtime_to_backend(rt) {
                assert_ne!(
                    b,
                    zenmetrics_api::Backend::CubeclCpu,
                    "{rt:?} must NOT map to cubecl-cpu (project rule: never dispatch it)"
                );
            }
        }
        assert_eq!(
            gpu_runtime_to_backend(GpuRuntime::Cpu).unwrap(),
            zenmetrics_api::Backend::Cpu,
            "--gpu-runtime cpu must resolve to the native CPU path"
        );
    }

    #[test]
    fn cvvdp_gpu_column_name_is_versioned_when_feature_on() {
        // `cvvdp-gpu` carries the GPU implementation tag so multiple cvvdp
        // variants (GPU / CPU / pycvvdp) don't collide in joined parquet
        // sidecars. The user-facing CLI flag (`--metric cvvdp-gpu`) stays
        // stable.
        let cols = MetricKind::CvvdpGpu.column_names();
        assert_eq!(cols.len(), 1);
        #[cfg(feature = "gpu-cvvdp")]
        {
            assert!(
                cols[0].starts_with("cvvdp_imazen_v")
                    || std::env::var("CVVDP_IMPL_TAG")
                        .map(|t| cols[0].starts_with(&t))
                        .unwrap_or(false),
                "expected cvvdp-gpu column to start with cvvdp_imazen_v or match \
                 CVVDP_IMPL_TAG override, got {:?}",
                cols[0]
            );
        }
        #[cfg(not(feature = "gpu-cvvdp"))]
        {
            assert_eq!(cols[0], "cvvdp_gpu");
        }
        // SDR output names the display on top of the implementation tag
        // (display presets exist only when a cvvdp backend is compiled).
        #[cfg(any(feature = "cpu-cvvdp", feature = "gpu-cvvdp"))]
        assert_eq!(
            MetricKind::CvvdpGpu.columns_for(Some(&display::DisplayPreset::Standard4k.into()))[0],
            format!("{}_standard_4k", cols[0])
        );
    }

    #[test]
    fn cvvdp_cpu_column_name_is_distinct_from_gpu() {
        // The unsuffixed `cvvdp` (CPU) emits a DISTINCT column namespace
        // (`cvvdp_cpu_imazen_v*`) from the GPU variant so CPU and GPU
        // cvvdp scores never collide on a sidecar join.
        let cols = MetricKind::Cvvdp.column_names();
        assert_eq!(cols.len(), 1);
        #[cfg(feature = "cpu-cvvdp")]
        {
            assert!(
                cols[0].starts_with("cvvdp_cpu_imazen_v")
                    || std::env::var("CVVDP_CPU_IMPL_TAG")
                        .map(|t| cols[0].starts_with(&t))
                        .unwrap_or(false),
                "expected cvvdp (CPU) column to start with cvvdp_cpu_imazen_v or \
                 match CVVDP_CPU_IMPL_TAG override, got {:?}",
                cols[0]
            );
        }
        #[cfg(not(feature = "cpu-cvvdp"))]
        {
            assert_eq!(cols[0], "cvvdp");
        }
        // SDR output names the display; HDR (`None`) keeps the base.
        // Display presets exist only when a cvvdp backend is compiled.
        #[cfg(any(feature = "cpu-cvvdp", feature = "gpu-cvvdp"))]
        {
            assert_eq!(
                MetricKind::Cvvdp.columns_for(Some(&display::DisplayPreset::StandardFhd.into()))[0],
                format!("{}_standard_fhd", cols[0])
            );
            assert_eq!(
                MetricKind::Ssim2.columns_for(Some(&display::DisplayPreset::StandardFhd.into())),
                MetricKind::Ssim2.column_names()
            );
        }
        assert_eq!(MetricKind::Cvvdp.columns_for(None), cols);
        // In any build where BOTH backends are compiled, the columns must
        // differ — that is the whole point of the CPU/GPU split.
        #[cfg(all(feature = "cpu-cvvdp", feature = "gpu-cvvdp"))]
        {
            assert_ne!(
                MetricKind::Cvvdp.column_names()[0],
                MetricKind::CvvdpGpu.column_names()[0],
                "cvvdp (CPU) and cvvdp-gpu must emit distinct columns"
            );
        }
    }

    #[test]
    fn cvvdp_cli_flag_names_are_stable() {
        // User-facing identifiers stay stable regardless of which
        // implementation is wired in — only the parquet column changes.
        assert_eq!(MetricKind::Cvvdp.name(), "cvvdp");
        assert_eq!(MetricKind::CvvdpGpu.name(), "cvvdp-gpu");
    }

    /// The unsuffixed cvvdp / iwssim follow the `ssim2`/`dssim`/`zensim`
    /// convention: backend == CPU, requires_gpu == false. Their `-gpu`
    /// siblings are GPU / requires_gpu. This is the deceptive-label fix —
    /// the registry must report the honest backend.
    #[test]
    fn unsuffixed_cvvdp_iwssim_are_cpu_gpu_siblings_require_gpu() {
        assert_eq!(MetricKind::Cvvdp.backend(), "CPU");
        assert!(!MetricKind::Cvvdp.requires_gpu());
        assert_eq!(MetricKind::CvvdpGpu.backend(), "GPU");
        assert!(MetricKind::CvvdpGpu.requires_gpu());

        assert_eq!(MetricKind::Iwssim.backend(), "CPU");
        assert!(!MetricKind::Iwssim.requires_gpu());
        assert_eq!(MetricKind::IwssimGpu.backend(), "GPU");
        assert!(MetricKind::IwssimGpu.requires_gpu());
    }

    /// The libvmaf-feature variants carry qualified names (they are NOT
    /// the Wang/Ponomarenko implementations the unqualified metrics
    /// bind), emit fixed non-versioned columns, and exempt themselves
    /// from `--luma-ingress` because they build their own studio-601
    /// planes — a second studio swing is not idempotent.
    #[test]
    fn libvmaf_variants_qualified_names_columns_ingress() {
        assert_eq!(MetricKind::SsimLibvmaf.name(), "ssim-libvmaf");
        assert_eq!(MetricKind::MsssimLibvmaf.name(), "msssim-libvmaf");
        assert_eq!(MetricKind::PsnrhvsDaala.name(), "psnrhvs-daala");

        assert_eq!(MetricKind::SsimLibvmaf.column_names(), &["ssim_libvmaf"]);
        assert_eq!(
            MetricKind::MsssimLibvmaf.column_names(),
            &["msssim_libvmaf"]
        );
        assert_eq!(
            MetricKind::PsnrhvsDaala.column_names(),
            &[
                "psnrhvs_daala_y",
                "psnrhvs_daala_cb",
                "psnrhvs_daala_cr",
                "psnrhvs_daala"
            ]
        );

        for k in [
            MetricKind::SsimLibvmaf,
            MetricKind::MsssimLibvmaf,
            MetricKind::PsnrhvsDaala,
        ] {
            assert!(
                !k.is_luma_only(),
                "{} does its own studio-601 ingress",
                k.name()
            );
            assert_eq!(k.backend(), "CPU");
            assert!(!k.requires_gpu());
        }
    }
}
