//! Shared test image corpus for zenmetrics' `-gpu` parity tests.
//!
//! The corpus lives in `crates/zenmetrics-corpus/data/`:
//! - `source.png` — original 256×256 PNG (RGB).
//! - `q{1,5,20,45,70,90}.jpg` — JPEG-compressed variants of the same
//!   image at the listed quality levels. Quality grid spans the
//!   visually-broken (q1) to near-transparent (q90) range so parity
//!   tests can lock the GPU↔CPU agreement across the full distortion
//!   spectrum.
//!
//! ## Usage
//!
//! ```rust,no_run
//! let dir = zenmetrics_corpus::corpus_dir();
//! let src = std::fs::read(dir.join("source.png")).unwrap();
//! let q70 = std::fs::read(dir.join("q70.jpg")).unwrap();
//! ```

#![forbid(unsafe_code)]

use std::path::PathBuf;

/// Absolute path to the corpus `data/` directory.
///
/// Resolved at runtime from `CARGO_MANIFEST_DIR`, so this works from
/// every consuming crate's tests regardless of how they're invoked.
pub fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data")
}

/// Path to a specific JPEG-quality file in the corpus. `q` must be one
/// of `1`, `5`, `20`, `45`, `70`, `90`.
pub fn jpeg_at_quality(q: u32) -> PathBuf {
    corpus_dir().join(format!("q{q}.jpg"))
}

/// Path to the original PNG reference.
pub fn source_png() -> PathBuf {
    corpus_dir().join("source.png")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_files_exist() {
        assert!(source_png().exists(), "source.png missing");
        for q in [1, 5, 20, 45, 70, 90] {
            assert!(jpeg_at_quality(q).exists(), "q{q}.jpg missing");
        }
    }
}
