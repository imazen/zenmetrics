//! Backend liveness diagnostic (imazen/zenmetrics#37).
//!
//! Run on any box to see how explicit-backend construction and `Auto`
//! resolution behave against the real GPU stack:
//!
//! ```bash
//! cargo run -p zenmetrics-api --example backend_liveness
//! ```
//!
//! On a healthy CUDA box: explicit `Backend::Cuda` constructs and scores a
//! synthetic pair (prints a real ssim2 value well below 100). On a
//! driver-without-toolkit box (the #37 trap): explicit construction must
//! print `Error::BackendUnavailable` — never `Ok` with a degenerate 100.0
//! — and `Auto` must resolve away from the broken GPU.
//!
//! Exit code 0 = behavior is sane for this host (either a working GPU
//! producing a real score, or a broken/absent GPU surfacing an error).
//! Exit code 1 = the #37 bug: construction+compute "succeeded" on a
//! non-operational backend (score would be garbage).

use zenmetrics_api::{Backend, Error, MemoryMode, Metric, MetricKind, MetricParams};

fn main() {
    let (w, h) = (512u32, 512u32);
    let n = (w * h * 3) as usize;
    // Deterministic synthetic pair with real structure: a gradient vs the
    // same gradient with visible noise — genuinely distinct images, so a
    // working ssim2 must score clearly below 100.
    let reference: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
    let distorted: Vec<u8> = reference
        .iter()
        .enumerate()
        .map(|(i, &v)| v.wrapping_add(((i * 37) % 23) as u8))
        .collect();

    println!("Auto resolves to: {:?}", Backend::resolve_auto());

    print!("explicit Backend::Cuda construction: ");
    match Metric::new_with_memory_mode(
        MetricKind::Ssim2,
        Backend::Cuda,
        w,
        h,
        MetricParams::default_for(MetricKind::Ssim2),
        MemoryMode::Full,
    ) {
        Err(Error::BackendUnavailable { backend, reason }) => {
            println!("Err(BackendUnavailable)");
            println!("  backend: {backend}");
            println!("  reason:  {reason}");
            println!("OK: broken runtime surfaced as an error, not a score (#37 fixed)");
        }
        Err(other) => {
            println!("Err({other})");
            println!("OK: construction refused (no silent fallback)");
        }
        Ok(mut m) => match m.compute_srgb_u8(&reference, &distorted) {
            Ok(score) => {
                println!("Ok; compute -> {} = {:.4}", score.metric_name, score.value);
                // A working backend on genuinely different images scores
                // well below the identical-pair value. 100.0 here would be
                // the #37 zero-readback artifact.
                if score.value >= 99.99 {
                    println!(
                        "FAIL: degenerate {:.2} on distinct images — kernels almost \
                         certainly never ran (#37 regression)",
                        score.value
                    );
                    std::process::exit(1);
                }
                println!("OK: real score from a live backend");
            }
            Err(e) => {
                println!("Ok; compute -> Err({e})");
                println!("OK: compute failed loudly rather than fabricating a score");
            }
        },
    }
}
