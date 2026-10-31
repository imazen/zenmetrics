//! CAMBI c-value maps on the generic surface, against libvmaf's own
//! heatmap dump and the declared support.

use std::ffi::CString;
use std::mem::MaybeUninit;
use std::ptr;

use vmaf::{ModelVariant, Vmaf};
use vmaf_head_sys::*;
use zenmetrics_abstractions::PlaneLayout;
use zenmetrics_abstractions::zenpixels::{ChannelType, PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{self as za, Metric as _};

const W: usize = 640;
const H: usize = 360;

/// 10-bit 4:2:0 planes: a smooth ramp quantized to 4-step bands on the
/// left third (visible banding) and texture elsewhere.
fn planes(seed: usize) -> [Vec<u16>; 3] {
    let y = (0..W * H)
        .map(|i| {
            let (x, yy) = (i % W, i / W);
            if x < W / 3 {
                ((64 + x * 420 / W) & !3) as u16
            } else {
                (64 + ((x * 3 + yy * 5 + seed) % 220) * 4) as u16
            }
        })
        .collect();
    let c = (0..(W / 2) * (H / 2))
        .map(|i| (512 + (i % 37) * 3) as u16)
        .collect::<Vec<_>>();
    [y, c.clone(), c]
}

fn gray16(d: &[u8], w: usize, h: usize) -> PixelSlice<'_> {
    PixelSlice::new(d, w as u32, h as u32, w * 2, PixelDescriptor::GRAY16).unwrap()
}

fn image(p: &[Vec<u8>; 3]) -> za::Image<'_> {
    za::Image::from(
        za::Planes::new(
            PlaneLayout::ycbcr_420(ChannelType::U16),
            vec![
                gray16(&p[0], W, H),
                gray16(&p[1], W / 2, H / 2),
                gray16(&p[2], W / 2, H / 2),
            ],
        )
        .unwrap()
        .with_significant_bits(10)
        .unwrap(),
    )
}

fn bytes(p: &[Vec<u16>; 3]) -> [Vec<u8>; 3] {
    p.clone()
        .map(|v| v.iter().flat_map(|s| s.to_ne_bytes()).collect())
}

fn cambi_maps(r: &[Vec<u16>; 3], d: &[Vec<u16>; 3]) -> Vec<(String, Vec<f32>, za::MapGeometry)> {
    let (rb, db) = (bytes(r), bytes(d));
    let e = Vmaf::new(ModelVariant::Standard1080p)
        .evaluate(
            image(&rb),
            image(&db),
            za::EvalOpts::SCORE_ONLY.with_native_map(),
        )
        .unwrap();
    e.maps()
        .into_iter()
        .filter(|m| m.name.starts_with("cambi"))
        .map(|m| (m.name.to_string(), m.values().into_owned(), m.geometry))
        .collect()
}

fn picture(p: &[Vec<u16>; 3]) -> VmafPicture {
    let mut pic = MaybeUninit::<VmafPicture>::uninit();
    // SAFETY: `pic` is valid writable storage for one `VmafPicture`.
    let rc = unsafe {
        vmaf_picture_alloc(
            pic.as_mut_ptr(),
            VmafPixelFormat_VMAF_PIX_FMT_YUV420P,
            10,
            W as u32,
            H as u32,
        )
    };
    assert_eq!(rc, 0);
    // SAFETY: `vmaf_picture_alloc` returned 0, so it initialized `pic`.
    let pic = unsafe { pic.assume_init() };
    for plane in 0..3 {
        let (w, h) = if plane == 0 { (W, H) } else { (W / 2, H / 2) };
        assert!(pic.stride[plane] >= (w * 2) as isize);
        for y in 0..h {
            for x in 0..w {
                let offset = y * pic.stride[plane] as usize + x * 2;
                // SAFETY: the plane holds `h` rows of `stride` bytes and
                // `x * 2 + 2 <= w * 2 <= stride`, so the 2-byte write is in
                // bounds; `write_unaligned` needs no alignment.
                unsafe {
                    ptr::write_unaligned(
                        pic.data[plane].add(offset).cast::<u16>(),
                        p[plane][y * w + x],
                    );
                }
            }
        }
    }
    pic
}

/// libvmaf's CAMBI writes each scale's c-values as `u16(scale·c)` with
/// `scale = 65535 / (max_diff_weight·window²/4)` when given
/// `heatmaps_path`. The served maps, quantized the same way, match its
/// dump.
#[test]
fn cambi_maps_match_libvmaf_heatmaps() {
    let (r, d) = (planes(1), planes(2));
    let maps = cambi_maps(&r, &d);
    assert_eq!(maps.len(), 5);

    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("cambi-heatmaps-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = VmafConfiguration {
        log_level: VmafLogLevel_VMAF_LOG_LEVEL_NONE,
        n_threads: 1,
        n_subsample: 1,
        cpumask: 0,
        gpumask: 0,
    };
    let mut context = ptr::null_mut();
    // SAFETY: `context` is a valid out-pointer; `cfg` is plain data.
    let rc = unsafe { vmaf_init(&mut context, cfg) };
    assert_eq!(rc, 0);
    let mut options = ptr::null_mut();
    for (key, value) in [
        ("cambi_vis_lum_threshold", "0.06"),
        ("cambi_max_val", "17"),
        ("heatmaps_path", dir.to_str().unwrap()),
    ] {
        let (k, v) = (CString::new(key).unwrap(), CString::new(value).unwrap());
        // SAFETY: both strings are NUL-terminated and outlive the call,
        // which copies them into the dictionary.
        let rc = unsafe { vmaf_feature_dictionary_set(&mut options, k.as_ptr(), v.as_ptr()) };
        assert_eq!(rc, 0);
    }
    let feature = CString::new("cambi").unwrap();
    // SAFETY: `context` is initialized; libvmaf takes ownership of `options`.
    let rc = unsafe { vmaf_use_feature(context, feature.as_ptr(), options) };
    assert_eq!(rc, 0);
    let (mut rp, mut dp) = (picture(&r), picture(&d));
    // SAFETY: both pictures were allocated by libvmaf; it takes ownership.
    let rc = unsafe { vmaf_read_pictures(context, &mut rp, &mut dp, 0) };
    assert_eq!(rc, 0);
    // SAFETY: null pictures flush the context, as libvmaf documents.
    let rc = unsafe { vmaf_read_pictures(context, ptr::null_mut(), ptr::null_mut(), 0) };
    assert_eq!(rc, 0);
    // SAFETY: `context` is initialized and not used afterwards.
    let rc = unsafe { vmaf_close(context) };
    assert_eq!(rc, 0);

    // The scaling libvmaf applies: max diff weight over the 4 contrast
    // levels (1, 2, 3, 4) and the adjusted window.
    let window = (((65 * (W + H)) / 375) >> 4) | 1;
    let max_c = 4 * window * window / 4;
    let scaling = 65535.0f64 / max_c as f64;
    let mut worst = 0u32;
    let mut differ = 0usize;
    let mut nonzero = 0usize;
    for (s, (name, values, g)) in maps.iter().enumerate() {
        assert_eq!(name, &format!("cambi_{s}"));
        let (sw, sh) = (g.cells_x as usize, g.cells_y as usize);
        let file = dir.join(format!("cambi_heatmap_scale_{s}_{sw}x{sh}_16b.gray"));
        let raw = std::fs::read(&file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
        assert_eq!(raw.len(), sw * sh * 2, "{}", file.display());
        for (i, chunk) in raw.as_chunks::<2>().0.iter().enumerate() {
            let libvmaf = u16::from_ne_bytes(*chunk);
            nonzero += usize::from(libvmaf != 0);
            let ours = (scaling * values[i] as f64) as u16;
            let delta = (i32::from(ours) - i32::from(libvmaf)).unsigned_abs();
            worst = worst.max(delta);
            differ += usize::from(delta != 0);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    eprintln!(
        "CAMBI heatmaps vs libvmaf: {differ} samples differ, max |Δ| {worst} LSB, {nonzero} nonzero"
    );
    assert!(nonzero > 1000, "the banded fixture must exercise CAMBI");
    assert_eq!(
        (differ, worst),
        (0, 0),
        "CAMBI heatmaps differ from libvmaf"
    );
}

/// CAMBI reads the distorted frame only, through local windows: changing
/// the distorted frame outside a cell's declared support leaves the cell
/// bit-identical, and changing the reference changes nothing at all.
#[test]
fn cambi_support_bounds_a_change_and_ignores_the_reference() {
    let (r, d) = (planes(1), planes(2));
    let base = cambi_maps(&r, &d);
    let other_ref = cambi_maps(&planes(9), &d);
    for ((_, a, _), (_, b, _)) in base.iter().zip(&other_ref) {
        assert_eq!(a, b, "CAMBI must not read the reference");
    }
    let (px, py, size) = (300usize, 150usize, 6usize);
    let mut d2 = d.clone();
    for y in py..py + size {
        for x in px..px + size {
            d2[0][y * W + x] = 1023 - d2[0][y * W + x];
        }
    }
    let changed = cambi_maps(&r, &d2);
    for ((name, a, g), (_, b, _)) in base.iter().zip(&changed) {
        for j in 0..g.cells_y {
            for i in 0..g.cells_x {
                let k = (j * g.cells_x + i) as usize;
                let hits = g.support_rect(i, j).is_some_and(|s| {
                    s.x < (px + size) as u32
                        && px < (s.x + s.width) as usize
                        && s.y < (py + size) as u32
                        && py < (s.y + s.height) as usize
                });
                if !hits {
                    assert_eq!(a[k].to_bits(), b[k].to_bits(), "{name} ({i},{j})");
                }
            }
        }
    }
}
