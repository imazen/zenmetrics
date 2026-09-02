//! DEPTH-SENSITIVITY GATE: the fleet HDR pair-scoring layer must be able to
//! SEE the difference between a 10-bit signal and its 8-bit quantization.
//!
//! This is the property that "least sufficient bit depth" selection is built
//! on. If the instrument cannot distinguish 10-bit from 8-bit-banded content,
//! then no amount of encoding at 10 bits can be evaluated, and a picker
//! trained on those scores would be choosing depth on noise.
//!
//! The construction is exact rather than empirical. Both images are built in
//! **PQ code space**: the reference from 10-bit codes, the distorted from the
//! same codes snapped to the 8-bit ladder — which is precisely the banding an
//! 8-bit encode of HDR content produces. The `--hdr-transfer pq` u8 shell is
//! `round(pq_inverse_eotf(nits) * 255)`, so it maps
//!
//!   reference  → round(c10 / 1023 * 255) = c8
//!   distorted  → round(c8  /  255 * 255) = c8
//!
//! MEASURED on this construction: **99.75 %** of the f32 samples differ, and
//! only **5.82 %** of the u8 bytes do — the shell erases **94.17 %** of the
//! difference. (Not 100 %: f32 round-trip error through pq_eotf/pq_inverse_eotf
//! pushes near-half-code samples across a rounding boundary. The residual is
//! rounding noise, not signal.) `u8_shell_is_structurally_blind_to_10_bit_banding`
//! asserts that collapse; it is a property of the SHELL, so it passes both
//! before and after the routing fix, which is what makes it a valid ruler.
//!
//! The two gates the fix moves are `fleet_hdr_path_is_not_the_u8_shell` (the
//! fleet's value must stop being bit-identical to the shell's) and
//! `fleet_hdr_path_sees_more_banding_than_the_u8_shell` (it must land FURTHER
//! from the identity score — it sees more of the artifact, not merely
//! something different).
#![cfg(all(feature = "hdr", feature = "cpu-metrics"))]

use zenmetrics_api::hdr::{pq_eotf, pq_inverse_eotf};
use zenmetrics_cli::hdr::{
    HdrImageFeeds, HdrPairScorers, HdrTransfer, NitsImage, score_hdr_pair_per_score_pairs,
    to_sdr_rgb8,
};
use zenmetrics_cli::metrics::{GpuRuntime, MetricKind, run_metric};

/// 192x192: above iwssim's 176-pixel minimum, small enough for CI.
const W: u32 = 192;
const H: u32 = 192;
/// PQ, so the u8 shell is exactly `round(pq_inverse_eotf(nits) * 255)` and the
/// code-space arithmetic in the module docs is the literal shell.
const TRANSFER: HdrTransfer = HdrTransfer::Pq;

/// The 10-bit PQ code for pixel `(x, y)`, channel `c` — a smooth diagonal ramp
/// over the mid-tones (PQ codes 300..=800, roughly 8..600 cd/m²), which is
/// where banding is visible and where every real gradient sits. Deliberately
/// smooth: banding is a SMOOTH-REGION artifact, and adding texture would let a
/// metric score the texture instead of the banding.
fn code10(x: u32, y: u32, c: usize) -> f32 {
    let t = (x + y) as f32 / ((W + H - 2) as f32);
    // A small per-channel offset keeps the ramp from being pure gray (which
    // some metrics special-case) without introducing high-frequency content.
    300.0 + 500.0 * t + (c as f32) * 3.0
}

/// Build the pair. `eight_bit` snaps each 10-bit code to the 8-bit ladder
/// first — the banding an 8-bit encode introduces.
fn pq_image(eight_bit: bool) -> NitsImage {
    let mut rgb = Vec::with_capacity((W * H * 3) as usize);
    for y in 0..H {
        for x in 0..W {
            for c in 0..3 {
                let c10 = code10(x, y, c);
                let normalized = if eight_bit {
                    // 10-bit code → nearest 8-bit code → back to [0,1].
                    (c10 / 1023.0 * 255.0).round() / 255.0
                } else {
                    c10.round() / 1023.0
                };
                rgb.push(pq_eotf(normalized));
            }
        }
    }
    NitsImage {
        rgb,
        width: W,
        height: H,
    }
}

/// The blindness itself, MEASURED. A property of the u8 shell, so this holds
/// both before and after the routing fix — it is the ruler the other two gates
/// are read against, not a consequence of them.
#[test]
fn u8_shell_is_structurally_blind_to_10_bit_banding() {
    let r = pq_image(false);
    let d = pq_image(true);

    // Anti-vacuity: the two images must genuinely differ in f32. Without this,
    // a bug that made both sides identical would satisfy every assertion below
    // for entirely the wrong reason.
    let differing = r.rgb.iter().zip(&d.rgb).filter(|(a, b)| a != b).count();
    assert!(
        differing > r.rgb.len() * 9 / 10,
        "the 10-bit and 8-bit-quantized images must actually differ in f32 nits; \
         only {differing} of {} samples differ (measured: 110315/110592)",
        r.rgb.len()
    );

    let ru8 = to_sdr_rgb8(&r, TRANSFER);
    let du8 = to_sdr_rgb8(&d, TRANSFER);
    let shell_diff = ru8
        .pixels
        .iter()
        .zip(&du8.pixels)
        .filter(|(a, b)| a != b)
        .count();
    let collapsed = 1.0 - shell_diff as f64 / differing as f64;
    assert!(
        collapsed > 0.90,
        "the PQ u8 shell must collapse the great majority of 10-bit banding \
         (measured 94.17 %); got {:.2} % — {shell_diff} of {} bytes survived",
        collapsed * 100.0,
        ru8.pixels.len()
    );
}

/// Every metric's fleet-path value, its identity value, and the value the u8
/// shell would have produced for the same pair.
fn fleet_vs_shell(metric: MetricKind) -> (f64, f64, f64) {
    let rf = HdrImageFeeds::new(pq_image(false), TRANSFER);
    let df = HdrImageFeeds::new(pq_image(true), TRANSFER);
    let mut scorers = HdrPairScorers::new(GpuRuntime::Auto);
    let fleet = score_hdr_pair_per_score_pairs(metric, &rf, &df, &mut scorers)
        .unwrap_or_else(|e| panic!("{metric:?} fleet: {e}"))[0]
        .1;
    let identity = score_hdr_pair_per_score_pairs(metric, &rf, &rf, &mut scorers)
        .unwrap_or_else(|e| panic!("{metric:?} identity: {e}"))[0]
        .1;
    let shell = run_metric(metric, rf.sdr_u8(), df.sdr_u8(), GpuRuntime::Auto)
        .unwrap_or_else(|e| panic!("{metric:?} shell: {e}"))[0]
        .1;
    (fleet, identity, shell)
}

/// The metrics the fleet path used to blanket-route through the u8 shell.
/// cvvdp keeps its own documented reference-peak-anchored u8 arm and is out of
/// scope; dssim has no HDR path by design.
const SHELLED: [MetricKind; 4] = [
    MetricKind::Ssim2,
    MetricKind::Zensim,
    MetricKind::Iwssim,
    MetricKind::Butteraugli,
];

/// GATE 1 — the fleet path must no longer BE the u8 shell.
///
/// Anti-vacuity: pre-fix every one of these was bit-identical to the shell
/// (measured: ssim2 95.97785891904992 on both sides), so this fails on the
/// first entry.
#[test]
fn fleet_hdr_path_is_not_the_u8_shell() {
    for metric in SHELLED {
        let (fleet, _identity, shell) = fleet_vs_shell(metric);
        assert!(
            fleet.to_bits() != shell.to_bits(),
            "{metric:?}: the fleet HDR path returned {fleet}, bit-identical to the \
             u8 shell — it is still narrowing a 10-bit pair to 8 bits before scoring"
        );
    }
}

/// GATE 2 — and it must see MORE of the banding, not merely something else.
///
/// Direction-agnostic: ssim2/zensim are 0..100 with 100 = identical, iwssim is
/// 0..1 with 1 = identical, butteraugli is a distance with 0 = identical. So
/// the comparison is on |score - identity|, which grows with distortion in
/// every one of them.
///
/// MEASURED (192x192 PQ ramp, this construction) — deviation from identity,
/// u8 shell → fleet f32: ssim2 4.022 → 6.631 (1.65x), zensim 5.564 → 6.711
/// (1.21x), iwssim 7.258e-4 → 9.359e-4 (1.29x), butteraugli 0.3858 → 1.1697
/// (3.03x). The bar below is deliberately far under the smallest of those:
/// the claim is the SIGN, and the ratios are recorded rather than pinned.
#[test]
fn fleet_hdr_path_sees_more_banding_than_the_u8_shell() {
    for metric in SHELLED {
        let (fleet, identity, shell) = fleet_vs_shell(metric);
        let fleet_dev = (fleet - identity).abs();
        let shell_dev = (shell - identity).abs();
        assert!(
            fleet_dev > shell_dev * 1.05,
            "{metric:?}: the f32 route must register MORE distortion than the u8 \
             shell on a 10-bit-vs-8-bit pair; got fleet dev {fleet_dev} \
             (score {fleet}) vs shell dev {shell_dev} (score {shell}), identity {identity}"
        );
    }
}

/// The construction's own premise. If pq_eotf/pq_inverse_eotf ever stop being
/// mutual inverses to well within half a u8 code, the byte-collapse above would
/// move for a reason that has nothing to do with the shell, and this says so.
#[test]
fn pq_code_space_round_trip_is_exact_enough_for_the_construction() {
    for c10 in [300u32, 437, 512, 800] {
        let n = pq_eotf(c10 as f32 / 1023.0);
        let back = pq_inverse_eotf(n) * 1023.0;
        assert!(
            (back - c10 as f32).abs() < 0.05,
            "pq round-trip drifted at code {c10}: got {back}"
        );
    }
}
