//! Demonstrate the memory savings of `MemoryMode::Strip` for zensim-gpu.
//!
//! Builds a synthetic image at the configured size, scores it through
//! Strip mode, and prints the strip-mode estimator's working-set bytes
//! at every regime so the user can sanity-check vs the Full estimate.
//!
//! Run with:
//!     cargo run --release -p zensim-gpu --example strip_memory_demo --features cuda

use zensim_gpu::{
    ZensimFeatureRegime,
    estimate_gpu_memory_bytes,
    memory_mode::{
        CUBECL_OVERHEAD_BYTES, estimate_strip_gpu_memory_bytes_with_regime,
    },
    pipeline::{STRIP_DEFAULT_BODY, STRIP_DEFAULT_HALO},
};

fn main() {
    let halo = STRIP_DEFAULT_HALO;
    let body = STRIP_DEFAULT_BODY;
    eprintln!("zensim-gpu strip vs full memory estimate");
    eprintln!(
        "(halo = {halo}, h_body = {body}; CUBECL_OVERHEAD = {} MB)",
        CUBECL_OVERHEAD_BYTES / (1024 * 1024)
    );
    eprintln!();
    eprintln!(
        "{:^10} {:^8} {:^12} {:^12} {:^10}",
        "size", "regime", "full MB", "strip MB", "ratio"
    );
    eprintln!("{}", "-".repeat(60));

    let sizes = [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192)];
    let regimes = [
        ("Basic", ZensimFeatureRegime::Basic),
        ("Extended", ZensimFeatureRegime::Extended),
        ("WithIw", ZensimFeatureRegime::WithIw),
    ];

    for &(w, h) in &sizes {
        for &(name, regime) in &regimes {
            let full_bytes = estimate_gpu_memory_bytes(w, h, regime);
            let strip_bytes = estimate_strip_gpu_memory_bytes_with_regime(w, body, regime)
                .unwrap_or(0);
            let full_mb = (full_bytes as f64) / (1024.0 * 1024.0);
            let strip_mb = (strip_bytes as f64) / (1024.0 * 1024.0);
            let ratio = if strip_mb > 0.0 { full_mb / strip_mb } else { 0.0 };
            eprintln!(
                "{:>4}x{:<4} {:<8} {:>10.1} {:>10.1} {:>8.2}x",
                w, h, name, full_mb, strip_mb, ratio
            );
        }
    }
}
