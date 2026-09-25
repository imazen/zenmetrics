//! Consolidated integration-test entry point.
//!
//! Every former `tests/<name>.rs` is a submodule here, compiled into one
//! `it` test binary instead of N separate binaries (one link step, not N).
//! Per-test gating that used to live in `[[test]] required-features` is now
//! a `#[cfg(...)]` on each `mod` line. Select a former target with a module
//! filter: `cargo test --test it <name>::`.

#[cfg(all(feature = "all-metrics", feature = "cpu-metrics"))]
mod backend_matrix;
mod backend_resolve;
mod cached_ref_parity;
mod cancel;
#[cfg(all(feature = "cubecl-types", feature = "cuda"))]
mod compute_handles;
#[cfg(all(feature = "butter", feature = "cuda", feature = "zensim"))]
mod compute_multi;
#[cfg(all(
    feature = "butter",
    feature = "cpu-butter",
    feature = "cuda",
    feature = "pixels"
))]
mod cpu_butter_linear;
#[cfg(all(feature = "cpu-cvvdp", feature = "cuda", feature = "cvvdp"))]
mod cpu_cvvdp_linear;
#[cfg(all(
    feature = "cpu-butter",
    feature = "cpu-cvvdp",
    feature = "cpu-dssim",
    feature = "cpu-iwssim",
    feature = "cpu-ssim2",
    feature = "cpu-zensim"
))]
mod cpu_dispatch;
#[cfg(all(feature = "cpu-hdrvdp", feature = "hdr"))]
mod cpu_hdrvdp_pu;
#[cfg(all(feature = "cpu-ssim2", feature = "hdr", feature = "ssim2"))]
mod cpu_ssim2_pu;
#[cfg(all(feature = "cpu-zensim", feature = "hdr", feature = "zensim"))]
mod cpu_zensim_pu;
#[cfg(all(feature = "cuda", feature = "cvvdp"))]
mod cvvdp_display;
mod dispatch;
#[cfg(all(
    any(feature = "cuda", feature = "wgpu"),
    feature = "hdr",
    feature = "zensim"
))]
mod gpu_zensim_pu;
#[cfg(all(
    feature = "butter",
    feature = "cuda",
    feature = "hdr",
    feature = "ssim2",
    feature = "zensim"
))]
mod hdr_scorer;
#[cfg(all(
    feature = "cpu-butter",
    feature = "cpu-ssim2",
    feature = "cpu-zensim",
    feature = "hdr"
))]
mod hdr_scorer_cpu;
mod metric_base_hdr;
mod pixels_smoke;
mod score_pair;
mod session_alloc_flat;
#[cfg(all(feature = "cuda", feature = "cvvdp"))]
mod session_cap;
mod session_owned;
mod session_owned_cap;
#[cfg(all(feature = "cuda", feature = "cvvdp"))]
mod session_parity;
mod session_reclaim_non_cvvdp;
#[cfg(all(feature = "cuda", feature = "cvvdp"))]
mod session_vram_isolation;
#[cfg(all(
    feature = "butter",
    feature = "cuda",
    feature = "hdr",
    feature = "pixels",
    feature = "ssim2"
))]
mod unified_pixels;

/// Test parameters for `kind`. cvvdp has no default display in the umbrella
/// (`MetricParams::try_default_for` refuses it), so the suite names the
/// `standard_4k` preset explicitly — the display every parity golden was
/// captured at.
#[allow(dead_code)]
pub(crate) fn try_params_for(
    kind: zenmetrics_api::MetricKind,
) -> zenmetrics_api::Result<zenmetrics_api::MetricParams> {
    #[cfg(feature = "cvvdp")]
    if kind == zenmetrics_api::MetricKind::Cvvdp {
        return Ok(zenmetrics_api::MetricParams::cvvdp(
            zenmetrics_api::cvvdp::params::DisplayPreset::Standard4k,
        ));
    }
    zenmetrics_api::MetricParams::try_default_for(kind)
}

/// [`try_params_for`], panicking on a disabled metric (the old
/// `MetricParams::default_for` contract).
#[allow(dead_code)]
pub(crate) fn params_for(kind: zenmetrics_api::MetricKind) -> zenmetrics_api::MetricParams {
    try_params_for(kind).unwrap_or_else(|e| panic!("{e}"))
}
