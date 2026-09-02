//! AVIF HDR tripwire — the twin of the PNG `cICP` refusal in `decode.rs`.
//!
//! **The defect this gates (zenmetrics#25 failure class, zero-tolerance).**
//! Before the fix, an AVIF whose container `colr`/`nclx` box signals an
//! absolute-luminance HDR transfer (PQ = 16, HLG = 18) was decoded, narrowed
//! to 8 bits and relabelled sRGB by the `RowConverter` funnel in
//! `decode::pixel_slice_to_rgb8` — with **no error and no warning**. Measured
//! on the pre-fix release binary: a PQ-signalled copy of `ref_64.avif` scored
//! `ssim2=96.137450`, *bit-identical* to the sRGB original, i.e. the transfer
//! signalling was ignored end to end. PNG had refused this since
//! `decode.rs`'s cICP tripwire; AVIF had no equivalent.
//!
//! The second-line `zenpixels` `HdrSourceRequiresPeak` guard cannot catch it:
//! `ManagedAvifDecoder`'s buffered path never calls `descriptor_with_cicp`
//! (only the row-sink paths do), so the buffer reaches the converter tagged
//! `TransferFunction::Unknown` and the conversion is a byte passthrough.
//!
//! **Tiled vs plain.** The fix reads the transfer from
//! `ManagedAvifDecoder::decode_full`, which is the single entry serving *both*
//! the grid (tiled) and non-grid shapes — so one policy covers both by
//! construction. Measured pre-fix, *both* shapes scored silently (a real
//! grid-tiled AVIF patched to `tc=16` also returned a plausible number), so
//! there is no correct-for-grid behaviour to preserve. Grid fixtures are
//! multi-item ISOBMFF and cannot be synthesised without a muxer, so the
//! shape-independence is covered structurally (one call site) plus the
//! exhaustive transfer-code unit test in `decode.rs`.

#![cfg(feature = "avif")]

use std::io::Write;

use zenmetrics_cli::decode::decode_image_to_rgb8;

/// The committed 64x64 SDR fixture. Its `colr`/`nclx` box carries
/// `colour_primaries=1, transfer_characteristics=13 (sRGB), matrix=6`.
const REF_64_AVIF: &[u8] = include_bytes!("fixtures/ref_64.avif");

/// Rewrite the `nclx` `transfer_characteristics` field in place.
///
/// The `colr` box payload is `colour_type(4) | primaries(u16) |
/// transfer(u16) | matrix(u16) | full_range_flag(1)`, so the transfer field
/// sits 10 bytes past the start of the box *type*. Patching it changes only
/// the colour signalling — the box length, the `av1C` config and the `mdat`
/// payload are untouched, so every variant stays a byte-valid, decodable
/// AVIF that differs from the original in signalling alone.
fn with_transfer(tc: u16) -> Vec<u8> {
    let mut bytes = REF_64_AVIF.to_vec();
    let colr = bytes
        .windows(4)
        .position(|w| w == b"colr")
        .expect("fixture must carry a colr box");
    assert_eq!(
        &bytes[colr + 4..colr + 8],
        b"nclx",
        "fixture's colr box must be nclx-typed"
    );
    let at = colr + 10;
    bytes[at..at + 2].copy_from_slice(&tc.to_be_bytes());
    bytes
}

/// Run bytes through the real, path-based decode entry point — the same one
/// `score`, `score-pairs`, the sweep and the jobexec feature backfill call.
fn decode(bytes: &[u8]) -> Result<zenmetrics_cli::decode::Rgb8Image, String> {
    let mut f = tempfile::Builder::new()
        .suffix(".avif")
        .tempfile()
        .expect("tempfile");
    f.write_all(bytes).expect("write fixture");
    f.flush().expect("flush fixture");
    decode_image_to_rgb8(f.path()).map_err(|e| e.to_string())
}

/// Assert the tripwire fires, returning its message. `Rgb8Image` is not
/// `Debug`, and adding a derive to a public type to satisfy a test would be
/// a public-surface change for no product reason — so unwrap by hand.
fn expect_refusal(bytes: &[u8], what: &str) -> String {
    match decode(bytes) {
        Ok(img) => panic!(
            "{what}: expected a loud refusal, got a silent {}x{} decode \
             ({} bytes) — this is the zenmetrics#25 failure class",
            img.width,
            img.height,
            img.pixels.len()
        ),
        Err(e) => e,
    }
}

#[test]
fn committed_sdr_fixture_decodes() {
    let img = decode(REF_64_AVIF).expect("the committed SDR fixture must decode");
    assert_eq!((img.width, img.height), (64, 64));
    assert_eq!(img.pixels.len(), 64 * 64 * 3);
}

#[test]
fn patching_transfer_alone_does_not_change_pixels() {
    // Guards the fixture-construction helper itself: rewriting the transfer
    // code to another SDR value must not perturb the decoded pixels, so a
    // refusal below is attributable to the tripwire and nothing else.
    let base = decode(REF_64_AVIF).expect("sRGB fixture decodes");
    let bt709 = decode(&with_transfer(1)).expect("BT.709 transfer must still decode");
    assert_eq!(base.pixels, bt709.pixels);
}

#[test]
fn pq_transfer_is_refused() {
    let err = expect_refusal(
        &with_transfer(16),
        "a PQ-signalled AVIF must NOT be silently narrowed to 8-bit sRGB",
    );
    assert!(
        err.contains("16"),
        "error must name the transfer code: {err}"
    );
    assert!(err.contains("PQ"), "error must name the transfer: {err}");
    assert!(err.contains("--hdr"), "error must name the route: {err}");
}

#[test]
fn hlg_transfer_is_refused() {
    let err = expect_refusal(
        &with_transfer(18),
        "an HLG-signalled AVIF must NOT be silently narrowed to 8-bit sRGB",
    );
    assert!(
        err.contains("18"),
        "error must name the transfer code: {err}"
    );
    assert!(err.contains("HLG"), "error must name the transfer: {err}");
    assert!(err.contains("--hdr"), "error must name the route: {err}");
}

#[test]
fn refusal_is_scoped_to_hdr_transfers_only() {
    // The 8-vs-10-bit SDR AVIF track scores 10-bit encodes of 8-bit sources
    // through this very path, and BT.2020 SDR transfers (14, 15) sit next to
    // PQ/HLG in the CICP table. An over-broad guard would break that track,
    // so pin the boundary: only 16 and 18 refuse.
    for tc in [1u16, 4, 6, 8, 13, 14, 15] {
        assert!(
            decode(&with_transfer(tc)).is_ok(),
            "transfer {tc} is SDR and must keep decoding"
        );
    }
}
