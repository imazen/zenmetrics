//! Emit pyramid + Gaussian-window constants for the CPU IW-SSIM port.
//!
//! Bit-identical to `iwssim-gpu/build.rs` by construction: both
//! crates' build scripts call the same helper
//! `iwssim_filter_codegen::emit_filters_rs` (`crates/iwssim-filter-
//! codegen/src/lib.rs`). Any drift between the emitted constants
//! would silently break parity with the GPU port AND the Python
//! reference, so the shared helper keeps a single source of truth.

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
