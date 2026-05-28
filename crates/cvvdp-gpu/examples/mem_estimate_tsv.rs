//! Machine-readable per-mode VRAM-estimate dump for the memory-audit
//! join script (`scripts/memory_audit/join_estimates_2026-05-28.py`).
//!
//! Prints a TSV of `(mode, size_w, size_h, estimate_bytes)` for the
//! canonical sweep sizes, sourcing every number from the **real** Rust
//! estimator functions in `pipeline.rs` (not a hand-copied proxy
//! formula). The Python join script shells out to this example so the
//! estimate/gap TSVs can never silently drift from the Rust source —
//! when an estimator formula changes, this output changes with it.
//!
//! Run:
//! ```text
//! cargo run --release --example mem_estimate_tsv -p cvvdp-gpu --no-default-features
//! ```
//!
//! Output columns (tab-separated, with a header row):
//!   mode  size_w  size_h  estimate_bytes
//!
//! `mode` values mirror the sweep TSV's `mode` column for cvvdp-gpu:
//! `full`, `warm_ref`, `warm_ref_strip`, `strip_pair`, `capped`,
//! `auto`. The `auto` row reports the Full estimate (Auto resolves to
//! Full whenever it fits, which is the common case at these sizes).

use cvvdp_gpu::{
    estimate_gpu_memory_bytes, estimate_gpu_memory_bytes_capped,
    estimate_gpu_memory_bytes_strip, estimate_gpu_memory_bytes_strip_pair,
};

/// Default strip body the memory-audit harness uses, matching the
/// Python `BODY = 256` constant. Must be a power of two
/// (`is_valid_strip_h_body`).
const BODY: u32 = 256;

fn emit(mode: &str, w: u32, h: u32, bytes: Option<usize>) {
    match bytes {
        Some(b) => println!("{mode}\t{w}\t{h}\t{b}"),
        // Mirror the sweep TSV: absent estimate -> empty cell. The
        // Python side treats a missing/blank value as "no estimate".
        None => println!("{mode}\t{w}\t{h}\t"),
    }
}

fn main() {
    // Canonical sweep sizes (must match `SIZES` in the join script).
    let sizes: &[(u32, u32)] = &[
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (7680, 5184),
    ];

    println!("mode\tsize_w\tsize_h\testimate_bytes");
    for &(w, h) in sizes {
        let full = estimate_gpu_memory_bytes(w, h);
        // Mode E (warm_ref_strip) keeps the full RefFullState on device
        // — see `estimate_gpu_memory_bytes_strip`'s doc; it is NOT a
        // memory win vs Full.
        let strip_e = estimate_gpu_memory_bytes_strip(w, h, BODY);
        // Mode B (strip_pair) — strip-pair walker.
        let strip_pair = estimate_gpu_memory_bytes_strip_pair(w, h, BODY);
        // Capped pyramid: report at the natural depth cap (use a large
        // `levels` so the estimator clamps to the natural depth — same
        // value the sweep's `capped` mode exercises by default).
        let capped = estimate_gpu_memory_bytes_capped(w, h, u32::MAX);

        // Full-family modes (full / warm_ref / auto) all hold the full
        // working set; report the Full estimate for each so the gap
        // table has coverage for every sweep mode.
        emit("full", w, h, full);
        emit("warm_ref", w, h, full);
        emit("auto", w, h, full);
        emit("warm_ref_strip", w, h, strip_e);
        emit("strip_pair", w, h, strip_pair);
        emit("capped", w, h, capped);
    }
}
