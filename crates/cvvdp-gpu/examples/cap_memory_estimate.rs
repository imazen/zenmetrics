//! Print estimate_gpu_memory_bytes at canonical sizes + with capped
//! levels. Used to populate `docs/STRIP_PROCESSING.md`'s memory
//! tables with numbers actually computed by the codebase (not
//! pencil-math).
//!
//! Run with:
//!
//!     cargo run -p cvvdp-gpu --features cubecl-types \
//!         --example cap_memory_estimate --release

use cvvdp_gpu::estimate_gpu_memory_bytes;
use cvvdp_gpu::kernels::pyramid::band_frequencies;
use cvvdp_gpu::params::DisplayGeometry;

fn mb(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() {
    println!("estimate_gpu_memory_bytes (Full mode, MB):");
    println!("size,natural_n_levels,bytes_mb");
    let cases = [
        (1024u32, 1024u32),
        (2048, 2048),
        (4000, 3000),
        (4096, 4096),
        (4900, 4900), // ~24 MP square
        (1024, 8192), // tall panorama
        (8192, 1024), // wide panorama
    ];
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    for (w, h) in cases {
        let bytes = estimate_gpu_memory_bytes(w, h).expect("estimate");
        let n_levels = band_frequencies(ppd, w as usize, h as usize).len();
        println!("{w}x{h},{n_levels},{:.1}", mb(bytes));
    }
}
