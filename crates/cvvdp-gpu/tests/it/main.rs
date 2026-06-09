//! Consolidated integration-test entry point.
//!
//! Every former `tests/<name>.rs` is a submodule here, compiled into one
//! `it` test binary instead of N separate binaries (one link step, not N).
//! Per-test gating that used to live in `[[test]] required-features` is now
//! a `#[cfg(...)]` on each `mod` line. Select a former target with a module
//! filter: `cargo test --test it <name>::`.

mod common;

mod auto_fallback;
#[cfg(feature = "cubecl-types")]
mod band_weights_invariants;
mod capped_pyramid_smoke;
#[cfg(feature = "cubecl-types")]
mod clamp_phase_uncertainty_invariants;
#[cfg(feature = "cubecl-types")]
mod color_kernel;
#[cfg(feature = "cubecl-types")]
mod color_kernel_display_dispatch;
#[cfg(feature = "cubecl-types")]
mod color_scalar;
#[cfg(feature = "cubecl-types")]
mod column_name;
#[cfg(feature = "cubecl-types")]
mod const_str_helpers;
#[cfg(feature = "cubecl-types")]
mod cpu_backend;
#[cfg(feature = "cubecl-types")]
mod csf_axes_invariants;
#[cfg(feature = "cubecl-types")]
mod csf_channel_invariants;
#[cfg(feature = "cubecl-types")]
mod csf_kernel;
#[cfg(feature = "cubecl-types")]
mod csf_scalar;
#[cfg(feature = "cubecl-types")]
mod diffmap_dispatch;
#[cfg(feature = "cubecl-types")]
mod diffmap_invariants;
#[cfg(feature = "cubecl-types")]
mod display_geometry;
#[cfg(feature = "cubecl-types")]
mod do_pooling_invariants;
#[cfg(feature = "cubecl-types")]
mod eotf_primaries_invariants;
#[cfg(feature = "cubecl-types")]
mod error_traits;
#[cfg(feature = "cubecl-types")]
mod gaussian_blur_sigma3_invariants;
#[cfg(feature = "cubecl-types")]
mod gausspyr_expand_invariants;
#[cfg(feature = "cubecl-types")]
mod gausspyr_reduce_invariants;
#[cfg(feature = "cubecl-types")]
mod goldens_metadata;
#[cfg(feature = "cubecl-types")]
mod laplacian_pyramid_invariants;
#[cfg(feature = "cubecl-types")]
mod lib_constants;
#[cfg(feature = "cubecl-types")]
mod lib_reexports;
#[cfg(feature = "cubecl-types")]
mod mask_pool_pixel_invariants;
#[cfg(feature = "cubecl-types")]
mod masking_constants;
#[cfg(feature = "cubecl-types")]
mod masking_kernel;
#[cfg(feature = "cubecl-types")]
mod masking_safe_pow;
#[cfg(feature = "cubecl-types")]
mod masking_scalar;
mod memory_mode;
#[cfg(feature = "cubecl-types")]
mod met2jod_invariants;
mod mode_b_walker_parity;
#[cfg(feature = "cubecl-types")]
mod mult_mutual_band_invariants;
#[cfg(feature = "cubecl-types")]
mod mult_mutual_pixel_invariants;
#[cfg(feature = "cubecl-types")]
mod opaque;
mod opaque_geometry_api;
mod params_match_upstream_json;
#[cfg(feature = "cubecl-types")]
mod params_placeholder;
#[cfg(feature = "cubecl-types")]
mod params_placeholder_non_display;
#[cfg(feature = "cubecl-types")]
mod parity;
#[cfg(feature = "cubecl-types")]
mod perf_mode_invariants;
#[cfg(feature = "cubecl-types")]
mod phase_uncertainty_band_invariants;
#[cfg(feature = "cubecl-types")]
mod pipeline_color;
#[cfg(feature = "cubecl-types")]
mod pipeline_score;
#[cfg(feature = "cubecl-types")]
mod pool_scalar;
#[cfg(feature = "cubecl-types")]
mod precompute_logs_row_invariants;
#[cfg(feature = "cubecl-types")]
mod predict_jod_invariants;
#[cfg(feature = "cubecl-types")]
mod pyramid_kernel;
#[cfg(feature = "cubecl-types")]
mod pyramid_scalar;
#[cfg(feature = "cubecl-types")]
mod shadow_jod;
#[cfg(feature = "cubecl-types")]
mod srgb_byte_to_dkl_invariants;
#[cfg(feature = "cubecl-types")]
mod state_machine_independence;
mod strip_kernel_parity;
mod strip_kernel_parity_pyramid;
mod strip_kernel_parity_upscale;
mod strip_mode_b_csf_halo_parity;
mod strip_mode_b_parity;
mod strip_mode_e_parity;
mod strip_mode_e_phase3;
mod sub_min_reflect_pad;
mod typed_sub_min_pad;
#[cfg(feature = "cubecl-types")]
mod version_lockstep;
#[cfg(feature = "cubecl-types")]
mod weber_pyramid_invariants;
