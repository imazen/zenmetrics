//! Emit pyramid + Gaussian-window constants for the IW-SSIM kernels.
//!
//! Bit-identical to `iwssim/build.rs` by construction: both crates'
//! build scripts call the same helper
//! `iwssim_filter_codegen::emit_filters_rs` (`crates/iwssim-filter-
//! codegen/src/lib.rs`). Why each crate still emits its own copy:
//! cube-macros in this crate resolve `crate::filters::*` paths at
//! macro-expansion time. Pulling them via `iwssim::filters::*` (or
//! `pub use iwssim::filters::BINOM5;`) is not name-resolvable inside
//! a `#[cube]` body — the cube codegen captures `crate`-relative
//! paths and re-emits them on the device side. So each crate owns
//! its own local `crate::filters` module, and the shared helper
//! guarantees the constants stay bit-identical.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dst = out_dir.join("filters.rs");
    let mut f = fs::File::create(&dst).expect("create filters.rs");
    iwssim_filter_codegen::emit_filters_rs(&mut f).expect("emit filters.rs");

    println!("cargo:rerun-if-changed=build.rs");
}
