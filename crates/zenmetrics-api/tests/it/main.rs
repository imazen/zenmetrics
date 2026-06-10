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
#[cfg(all(feature = "cuda", feature = "cvvdp"))]
mod cvvdp_display;
mod dispatch;
#[cfg(all(
    feature = "butter",
    feature = "cuda",
    feature = "hdr",
    feature = "ssim2",
    feature = "zensim"
))]
mod hdr_scorer;
mod metric_base_hdr;
mod pixels_smoke;
mod score_pair;
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
