//! **D14 regression** — `zenmetrics` must be able to score an ARBITRARY zensim
//! bake, not just the hard-coded shipped default.
//!
//! Before the `zensim_profile` selector landed, every zensim construction site
//! in this crate built `zensim::ZensimProfile::latest_preview()` as a literal
//! (`cpu_dispatch.rs:149`, `:269`, `metric.rs`, the GPU/CPU test paths), and no
//! CLI flag or parameter could change it. A candidate bake therefore could not
//! be scored through the fleet metric path at all — which is where any large
//! re-scoring has to happen.
//!
//! This test scores a **non-preview** bake through `zenmetrics_api::Metric` on
//! `Backend::Cpu` — the same entry point `zenmetrics jobexec` / `score-pairs`
//! reach — and asserts the umbrella's number is bit-identical to calling the
//! native `zensim` crate with that same profile.
//!
//! **No graceful skip.** The bake path comes from `ZENMETRICS_TEST_ZENSIM_BAKE`
//! (default [`DEFAULT_BAKE`]); if the file is absent the test FAILS loudly with
//! the path and the override name. It is never silently skipped.
//!
//! The bake is referenced BY PATH, never vendored — nothing >30 KB in git.
#![cfg(feature = "cpu-zensim")]

use zenmetrics_api::zensim_cpu::ZensimProfile;
use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams, zensim_profile};

/// ADD156 (`ADD156_safesyn_only_raw_lasso.bin`, 3,575 B, sha256
/// `51437a34f04887ce850b25eff4f72a6bcd12926873ce060a12878d558a7517db`) — a
/// 372-input basic-only linear bake carrying `zentrain.output_calibration_spline`.
const DEFAULT_BAKE: &str = "/mnt/v/output/zensim/corr-lq/ADD156_safesyn_only_raw_lasso.bin";

/// Env override for the bake path. Set it to point the test at another bake;
/// the decision is the CALLER's and is visible in the invocation, never buried
/// in a runtime `if !exists { return }`.
const BAKE_ENV: &str = "ZENMETRICS_TEST_ZENSIM_BAKE";

fn bake_path() -> String {
    let p = std::env::var(BAKE_ENV).unwrap_or_else(|_| DEFAULT_BAKE.to_string());
    assert!(
        std::path::Path::new(&p).is_file(),
        "zensim bake not found at {p:?}. This test does NOT skip itself — point \
         it at a readable ZNPR bake with {BAKE_ENV}=<path>, or restore the \
         default at {DEFAULT_BAKE}"
    );
    p
}

/// A deterministic (reference, distorted) sRGB8 pair: a smooth two-axis
/// gradient and a coarsely re-quantized copy of it.
fn pair(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    let (w, h) = (w as usize, h as usize);
    let mut r = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            r[i] = ((x * 255) / w) as u8;
            r[i + 1] = ((y * 255) / h) as u8;
            r[i + 2] = (((x + y) * 255) / (w + h)) as u8;
        }
    }
    // Quantize to 5 levels per channel — a visible, unambiguous distortion.
    let d: Vec<u8> = r.iter().map(|&v| (v / 51) * 51).collect();
    (r, d)
}

/// Score through the native `zensim` crate with an explicit profile — the
/// reference number the umbrella must reproduce.
fn direct_zensim_score(profile: ZensimProfile, r: &[u8], d: &[u8], w: u32, h: u32) -> f64 {
    use zenmetrics_api::zensim_cpu::{PixelFormat, StridedBytes, Zensim};
    let (wu, hu) = (w as usize, h as usize);
    let z = Zensim::new(profile);
    let src = StridedBytes::try_new(r, wu, hu, wu * 3, PixelFormat::Srgb8Rgb).expect("ref slice");
    let dst = StridedBytes::try_new(d, wu, hu, wu * 3, PixelFormat::Srgb8Rgb).expect("dist slice");
    z.compute(&src, &dst).expect("zensim compute").score()
}

fn umbrella_score(w: u32, h: u32, r: &[u8], d: &[u8]) -> (f64, f64) {
    let params = MetricParams::try_default_for(MetricKind::Zensim).expect("zensim params");
    let mut m = Metric::new(MetricKind::Zensim, Backend::Cpu, w, h, params)
        .expect("Metric::new zensim on Backend::Cpu");
    let one_shot = m.compute_srgb_u8(r, d).expect("compute_srgb_u8").value;
    // Cached-reference path — what `score-pairs` / jobexec use per reference.
    m.set_reference_srgb_u8(r).expect("set_reference_srgb_u8");
    let cached = m
        .compute_with_reference_srgb_u8(d)
        .expect("compute_with_reference_srgb_u8")
        .value;
    (one_shot, cached)
}

/// THE D14 assertion: the umbrella honours a runtime-selected bake.
///
/// Fails before the fix — `cpu_dispatch` ignored the selector entirely and
/// returned the `latest_preview()` (shipped `B`) score for every profile.
///
/// Single `#[test]` on purpose: it drives the PROCESS-WIDE default override,
/// so splitting it would let libtest's parallel threads race the global.
#[test]
fn cpu_umbrella_scores_a_runtime_selected_bake() {
    let (w, h) = (256u32, 192u32);
    let (r, d) = pair(w, h);
    let path = bake_path();

    // Default behaviour, unchanged: no override installed ⇒ exactly the
    // literal every call site used to hard-code.
    assert!(
        !zensim_profile::has_default_override(),
        "no override should be installed at test start"
    );
    assert_eq!(
        zensim_profile::default_profile().name(),
        ZensimProfile::latest_preview().name(),
        "default_profile() must be latest_preview() when unset"
    );

    let bake = zensim_profile::from_bake_path(&path).expect("resolve bake path");
    let shipped = ZensimProfile::latest_preview();

    // What the native zensim crate says for each profile, computed directly.
    let direct_bake = direct_zensim_score(bake, &r, &d, w, h);
    let direct_shipped = direct_zensim_score(shipped, &r, &d, w, h);
    eprintln!("D14: direct bake={direct_bake:.6}  direct shipped={direct_shipped:.6}  bake={path}");

    assert!(
        direct_bake.is_finite() && direct_shipped.is_finite(),
        "both direct scores must be finite: bake={direct_bake} shipped={direct_shipped}"
    );
    // The bake must actually be a different model, else the equalities below
    // would be vacuous.
    assert!(
        (direct_bake - direct_shipped).abs() > 1e-6,
        "the selected bake scores identically to the shipped default \
         ({direct_bake} vs {direct_shipped}) — pick a bake that differs, or the \
         umbrella assertion below proves nothing"
    );
    // The 0.000000 trap (a spline bake configured without skip_score_mapping)
    // would land exactly here.
    assert!(
        direct_bake != 0.0,
        "bake scored exactly 0.000000 — that is the \
         spline-without-skip_score_mapping signature, not a real score"
    );

    // No override ⇒ the umbrella is still the shipped-default score.
    let (default_one_shot, default_cached) = umbrella_score(w, h, &r, &d);
    assert_eq!(
        default_one_shot, direct_shipped,
        "with no override the umbrella must still be the shipped-default score \
         (byte-identical default behaviour)"
    );
    assert_eq!(default_cached, direct_shipped, "cached-ref default behaviour");

    // Override installed ⇒ the umbrella scores the SELECTED bake.
    zensim_profile::set_default(bake);
    let selected = std::panic::catch_unwind(|| umbrella_score(w, h, &r, &d));
    zensim_profile::clear_default();
    let (bake_one_shot, bake_cached) = selected.expect("umbrella score under override");

    assert_eq!(
        bake_one_shot, direct_bake,
        "umbrella (Backend::Cpu) must score the SELECTED bake, not the shipped \
         default (shipped would be {direct_shipped})"
    );
    assert_eq!(
        bake_cached, direct_bake,
        "cached-reference path must select the same profile as the one-shot path"
    );

    // …and clearing restores the default, so the override is not sticky.
    assert!(!zensim_profile::has_default_override());
    let (restored, _) = umbrella_score(w, h, &r, &d);
    assert_eq!(restored, direct_shipped, "clear_default() must restore the default");
}

/// Re-resolving the same bake path reuses one runtime slot (the
/// `fn() -> &'static [u8]` cap is on DISTINCT bakes, not on calls).
/// Touches no global default, so it is parallel-safe.
#[test]
fn repeat_resolution_of_one_path_is_stable() {
    let path = bake_path();
    let a = zensim_profile::from_bake_path(&path).expect("first resolve");
    for _ in 0..(zensim_profile::MAX_RUNTIME_BAKES * 4) {
        let b = zensim_profile::from_bake_path(&path).expect("repeat resolve");
        assert_eq!(a, b, "repeat resolution must return the same profile");
    }
}
