//! Drift guard: the embedded `WEIGHTS_PREVIEW_V0_2` constant in
//! `zensim-gpu::weights` MUST be byte-equal to the CPU
//! `zensim::profile::WEIGHTS_PREVIEW_V0_2` static at all times. If
//! the CPU crate ever rotates the basic-regime default, this test
//! fails — the fix is to re-copy the array into
//! `src/weights.rs` and update the doc / `default_weights()`
//! documentation to point at the new version tag.

use zensim_gpu::WEIGHTS_PREVIEW_V0_2 as GPU_WEIGHTS;

#[test]
fn gpu_weights_match_cpu_zensim_v0_2_byte_for_byte() {
    let cpu = &zensim::profile::WEIGHTS_PREVIEW_V0_2;
    assert_eq!(
        cpu.len(),
        GPU_WEIGHTS.len(),
        "weight array length mismatch — CPU has {}, GPU has {}",
        cpu.len(),
        GPU_WEIGHTS.len()
    );

    // Compare bit-for-bit via to_bits() — two f64 values that compare
    // PartialEq but differ in NaN payload would slip past `==`. The
    // weights are non-NaN finite constants, so to_bits equality is
    // the strictest possible check.
    let mut mismatched = Vec::new();
    for (i, (&g, &c)) in GPU_WEIGHTS.iter().zip(cpu.iter()).enumerate() {
        if g.to_bits() != c.to_bits() {
            mismatched.push((i, g, c));
        }
    }
    assert!(
        mismatched.is_empty(),
        "{} weights drifted from zensim::profile::WEIGHTS_PREVIEW_V0_2; first 4: {:?}",
        mismatched.len(),
        &mismatched[..mismatched.len().min(4)]
    );
}
