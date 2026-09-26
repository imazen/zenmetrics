//! Feature-level parity tests against real libvmaf (`vmaf-head-sys`,
//! vendored Netflix vmaf @ f85a8536). Registers the standalone
//! `float_ssim`, `float_ms_ssim`, and `psnr_hvs` extractors on a bare
//! `VmafContext` — no model — and compares per-frame feature scores
//! against `msssim::float_ssim`, `msssim::float_ms_ssim`, and
//! `psnrhvs::daala` (the Daala PSNR-HVS the libvmaf feature implements).

use msssim::{float_ms_ssim_scales, float_ssim_lcs};
use std::ffi::CString;
use std::mem::MaybeUninit;
use std::ptr;
use vmaf_head_sys::*;

struct FeatureCtx {
    ctx: *mut VmafContext,
}

impl FeatureCtx {
    fn new() -> Self {
        let cfg = VmafConfiguration {
            log_level: VmafLogLevel_VMAF_LOG_LEVEL_NONE,
            n_threads: 1,
            n_subsample: 1,
            cpumask: 0,
            gpumask: 0,
        };
        let mut ctx = ptr::null_mut();
        assert_eq!(unsafe { vmaf_init(&mut ctx, cfg) }, 0);

        // `enable_lcs` gives us the l/c/s sub-features to gate against.
        let key = CString::new("enable_lcs").unwrap();
        let val = CString::new("true").unwrap();
        for feature in ["float_ssim", "float_ms_ssim"] {
            let mut opts = ptr::null_mut();
            let rc = unsafe { vmaf_feature_dictionary_set(&mut opts, key.as_ptr(), val.as_ptr()) };
            assert_eq!(rc, 0, "vmaf_feature_dictionary_set failed: {rc}");
            let name = CString::new(feature).unwrap();
            let rc = unsafe { vmaf_use_feature(ctx, name.as_ptr(), opts) };
            assert_eq!(rc, 0, "vmaf_use_feature({feature}) failed: {rc}");
        }
        let name = CString::new("psnr_hvs").unwrap();
        let rc = unsafe { vmaf_use_feature(ctx, name.as_ptr(), ptr::null_mut()) };
        assert_eq!(rc, 0, "vmaf_use_feature(psnr_hvs) failed: {rc}");

        Self { ctx }
    }

    fn feature(&self, name: &str, index: usize) -> f64 {
        let name = CString::new(name).unwrap();
        let mut value = f64::NAN;
        let rc = unsafe {
            vmaf_feature_score_at_index(self.ctx, name.as_ptr(), &mut value, index as u32)
        };
        assert_eq!(rc, 0, "vmaf_feature_score_at_index({name:?}) failed: {rc}");
        value
    }
}

impl Drop for FeatureCtx {
    fn drop(&mut self) {
        unsafe {
            assert_eq!(vmaf_close(self.ctx), 0);
        }
    }
}

struct Frame {
    planes: [Vec<u16>; 3],
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
}

/// Synthetic content: patterned luma + chroma, several distortion
/// flavors so different code paths (masking thresholds, sign clamps,
/// decimation edges) get exercised.
fn frame(seed: usize, w: usize, h: usize, mode: u8) -> Frame {
    frame_impl(seed, w, h, mode, w / 2, h / 2)
}

/// Same generator as [`frame`] but with full-resolution chroma planes —
/// the YUV444P picture layout (`psnr_hvs` AIC-4 convention).
fn frame_444(seed: usize, w: usize, h: usize, mode: u8) -> Frame {
    frame_impl(seed, w, h, mode, w, h)
}

fn frame_impl(seed: usize, w: usize, h: usize, mode: u8, cw: usize, ch: usize) -> Frame {
    let mut planes = [vec![0u16; w * h], vec![0u16; cw * ch], vec![0u16; cw * ch]];
    for y in 0..h {
        for x in 0..w {
            let base = (x * 3 + y * 5 + seed * 7 + (x / 17) * 9 + ((x + y) / 29) * 31) % 224 + 16;
            let v = match mode {
                // reference
                0 => base,
                // coarse quantization on the left half (JPEG-ish)
                1 => {
                    if x < w / 2 {
                        (base / 12) * 12
                    } else {
                        base
                    }
                }
                // additive dither
                2 => base
                    .saturating_add(((x * 7 + y * 11 + seed) % 9).saturating_sub(4))
                    .min(255),
                // partial block smoothing
                3 => {
                    if x % 23 < 4 {
                        (base + 4) / 8 * 8
                    } else {
                        base
                    }
                }
                _ => base,
            };
            planes[0][y * w + x] = v as u16;
        }
    }
    for y in 0..ch {
        for x in 0..cw {
            let u = ((x * 7 + y * 3 + seed * 11 + (x / 11) * 5) % 200) + 28;
            let v = ((x * 2 + y * 9 + seed * 13 + (y / 7) * 3) % 200) + 28;
            let (du, dv) = if mode == 0 {
                (u, v)
            } else {
                (
                    u.saturating_sub((x + seed) % 6),
                    (v + (y + seed) % 7).min(240),
                )
            };
            planes[1][y * cw + x] = du as u16;
            planes[2][y * cw + x] = dv as u16;
        }
    }
    Frame {
        planes,
        w,
        h,
        cw,
        ch,
    }
}

fn picture(frame: &Frame, bit_depth: u32) -> VmafPicture {
    let mut picture = MaybeUninit::<VmafPicture>::uninit();
    let rc = unsafe {
        vmaf_picture_alloc(
            picture.as_mut_ptr(),
            if frame.planes[1].len() == frame.planes[0].len() {
                VmafPixelFormat_VMAF_PIX_FMT_YUV444P
            } else {
                VmafPixelFormat_VMAF_PIX_FMT_YUV420P
            },
            bit_depth,
            frame.w as u32,
            frame.h as u32,
        )
    };
    assert_eq!(rc, 0, "vmaf_picture_alloc failed: {rc}");
    let picture = unsafe { picture.assume_init() };
    let bytes = if bit_depth == 8 { 1 } else { 2 };
    for plane in 0..3 {
        let (pw, ph) = if plane == 0 {
            (frame.w, frame.h)
        } else {
            (frame.cw, frame.ch)
        };
        for y in 0..ph {
            for x in 0..pw {
                let sample = frame.planes[plane][y * pw + x];
                let dest = unsafe {
                    picture.data[plane].add(y * picture.stride[plane] as usize + x * bytes)
                };
                unsafe {
                    if bit_depth == 8 {
                        *dest.cast::<u8>() = sample as u8;
                    } else {
                        ptr::write_unaligned(dest.cast::<u16>(), sample);
                    }
                }
            }
        }
    }
    picture
}

fn frame_u8(frame: &Frame) -> [Vec<u8>; 3] {
    [
        frame.planes[0].iter().map(|&v| v as u8).collect(),
        frame.planes[1].iter().map(|&v| v as u8).collect(),
        frame.planes[2].iter().map(|&v| v as u8).collect(),
    ]
}

fn run_pair(ctx: &FeatureCtx, reference: &Frame, distorted: &Frame, index: usize) {
    let mut r = picture(reference, 8);
    let mut d = picture(distorted, 8);
    let rc = unsafe { vmaf_read_pictures(ctx.ctx, &mut r, &mut d, index as u32) };
    if rc != 0 {
        unsafe {
            vmaf_picture_unref(&mut r);
            vmaf_picture_unref(&mut d);
        }
        panic!("vmaf_read_pictures failed at frame {index}: {rc}");
    }
}

fn flush(ctx: &FeatureCtx) {
    let rc = unsafe { vmaf_read_pictures(ctx.ctx, ptr::null_mut(), ptr::null_mut(), 0) };
    assert_eq!(rc, 0, "vmaf flush failed: {rc}");
}

/// `(width, height, distortion-mode)` cases covering auto-scales 1, 2,
/// 4, 5, 7 plus an even-width/non-divisible decimate quirk and several
/// distortion shapes.
const SSIM_CASES: &[(usize, usize, u8)] = &[
    (384, 288, 1),   // scale 1
    (640, 384, 2),   // scale 2 (384/256 = 1.5 → 2)
    (900, 1100, 1),  // scale 4 (900/256 = 3.5 → 4)
    (1770, 1988, 2), // scale 7, even w not divisible by 7 (w&1 quirk)
    (1769, 1400, 3), // scale 5, odd dims
];

#[test]
fn float_ssim_parity() {
    for (case, &(w, h, mode)) in SSIM_CASES.iter().enumerate() {
        let ctx = FeatureCtx::new();
        let reference = frame(case, w, h, 0);
        let distorted = frame(case, w, h, mode);
        run_pair(&ctx, &reference, &distorted, 0);
        flush(&ctx);

        let (s, l, c, st) =
            float_ssim_lcs(&reference.planes[0], &distorted.planes[0], w, h, w, 8).unwrap();
        for (name, ours) in [
            ("float_ssim", s),
            ("float_ssim_l", l),
            ("float_ssim_c", c),
            ("float_ssim_s", st),
        ] {
            let oracle = ctx.feature(name, 0);
            let delta = (ours - oracle).abs();
            assert!(
                delta <= 2e-6,
                "{name} {w}x{h} mode{mode}: ours {ours} vs libvmaf {oracle} (|d| {delta})"
            );
        }
    }
}

#[test]
fn float_ms_ssim_parity() {
    // MS-SSIM needs min(w,h) ≥ ~176 (five halvings each ≥ 11).
    const MS_CASES: &[(usize, usize, u8)] = &[
        (384, 288, 1),
        (640, 384, 2),
        (900, 1100, 1),
        (1770, 1988, 2),
    ];
    for (case, &(w, h, mode)) in MS_CASES.iter().enumerate() {
        let ctx = FeatureCtx::new();
        let reference = frame(case, w, h, 0);
        let distorted = frame(case, w, h, mode);
        run_pair(&ctx, &reference, &distorted, 0);
        flush(&ctx);

        let (ls, cs, ss, msssim) =
            float_ms_ssim_scales(&reference.planes[0], &distorted.planes[0], w, h, w, 8).unwrap();
        let oracle = ctx.feature("float_ms_ssim", 0);
        assert!(
            (msssim - oracle).abs() <= 2e-6,
            "float_ms_ssim {w}x{h} mode{mode}: ours {msssim} vs libvmaf {oracle}"
        );
        for i in 0..5 {
            for (tag, ours) in [("l", ls[i]), ("c", cs[i]), ("s", ss[i])] {
                let name = format!("float_ms_ssim_{tag}_scale{i}");
                let oracle = ctx.feature(&name, 0);
                let delta = (ours - oracle).abs();
                assert!(
                    delta <= 2e-6,
                    "{name} {w}x{h} mode{mode}: ours {ours} vs libvmaf {oracle} (|d| {delta})"
                );
            }
        }
    }
}

#[test]
fn psnr_hvs_parity() {
    const HVS_CASES: &[(usize, usize, u8)] = &[
        (384, 288, 1),
        (640, 384, 2),
        (900, 1100, 1),
        (1770, 1988, 2),
        (512, 400, 3),
    ];
    for (case, &(w, h, mode)) in HVS_CASES.iter().enumerate() {
        let ctx = FeatureCtx::new();
        let reference = frame(case, w, h, 0);
        let distorted = frame(case, w, h, mode);
        run_pair(&ctx, &reference, &distorted, 0);
        flush(&ctx);

        let r8 = frame_u8(&reference);
        let d8 = frame_u8(&distorted);
        let ours = psnrhvs::daala::psnr_hvs_daala_yuv420(
            &r8[0],
            &d8[0],
            &r8[1],
            &d8[1],
            &r8[2],
            &d8[2],
            w,
            h,
            w,
            w / 2,
        );
        for (name, ours) in [
            ("psnr_hvs_y", ours.psnr_hvs_y),
            ("psnr_hvs_cb", ours.psnr_hvs_cb),
            ("psnr_hvs_cr", ours.psnr_hvs_cr),
            ("psnr_hvs", ours.psnr_hvs),
        ] {
            let oracle = ctx.feature(name, 0);
            let delta = (ours - oracle).abs();
            assert!(
                delta <= 2e-4,
                "{name} {w}x{h} mode{mode}: ours {ours} vs libvmaf {oracle} (|d| {delta})"
            );
        }
    }
}

/// YUV444P pictures — the convention the JPEG AIC-4 `PSNR-HVS` columns
/// were computed with (full-resolution chroma).
#[test]
fn psnr_hvs_parity_444() {
    const HVS_CASES: &[(usize, usize, u8)] =
        &[(384, 288, 1), (640, 384, 2), (900, 1100, 1), (512, 400, 3)];
    for (case, &(w, h, mode)) in HVS_CASES.iter().enumerate() {
        let ctx = FeatureCtx::new();
        let reference = frame_444(case, w, h, 0);
        let distorted = frame_444(case, w, h, mode);
        run_pair(&ctx, &reference, &distorted, 0);
        flush(&ctx);

        let r8 = frame_u8(&reference);
        let d8 = frame_u8(&distorted);
        let ours = psnrhvs::daala::psnr_hvs_daala_yuv444(
            &r8[0], &d8[0], &r8[1], &d8[1], &r8[2], &d8[2], w, h, w, w,
        );
        for (name, ours) in [
            ("psnr_hvs_y", ours.psnr_hvs_y),
            ("psnr_hvs_cb", ours.psnr_hvs_cb),
            ("psnr_hvs_cr", ours.psnr_hvs_cr),
            ("psnr_hvs", ours.psnr_hvs),
        ] {
            let oracle = ctx.feature(name, 0);
            let delta = (ours - oracle).abs();
            assert!(
                delta <= 2e-4,
                "{name} {w}x{h} mode{mode} (444): ours {ours} vs libvmaf {oracle} (|d| {delta})"
            );
        }
    }
}

/// Identical frames: SSIM family must be exactly 1.0; PSNR-HVS is +inf
/// on both sides (zero masked error).
#[test]
fn identical_frames() {
    let ctx = FeatureCtx::new();
    let reference = frame(7, 384, 288, 0);
    run_pair(&ctx, &reference, &reference, 0);
    flush(&ctx);

    assert_eq!(ctx.feature("float_ssim", 0), 1.0);
    assert_eq!(ctx.feature("float_ms_ssim", 0), 1.0);
    assert!(ctx.feature("psnr_hvs", 0).is_infinite());

    let (s, _, _, _) =
        float_ssim_lcs(&reference.planes[0], &reference.planes[0], 384, 288, 384, 8).unwrap();
    assert_eq!(s, 1.0);
    let (_, _, _, msssim) =
        float_ms_ssim_scales(&reference.planes[0], &reference.planes[0], 384, 288, 384, 8).unwrap();
    assert_eq!(msssim, 1.0);
    let r8 = frame_u8(&reference);
    let ours = psnrhvs::daala::psnr_hvs_daala_yuv420(
        &r8[0], &r8[0], &r8[1], &r8[1], &r8[2], &r8[2], 384, 288, 384, 192,
    );
    assert!(ours.psnr_hvs.is_infinite());
}

/// 10-bit input: libvmaf keeps `L = 255` constants regardless of bpc;
/// our `&[u16]` path must match that exact behavior.
#[test]
fn float_ssim_parity_10bit() {
    let ctx = FeatureCtx::new();
    let mut reference = frame(3, 640, 384, 0);
    let mut distorted = frame(3, 640, 384, 2);
    for p in reference
        .planes
        .iter_mut()
        .chain(distorted.planes.iter_mut())
    {
        for v in p.iter_mut() {
            *v = *v * 4 + (*v % 3);
        }
    }
    let mut r = picture(&reference, 10);
    let mut d = picture(&distorted, 10);
    let rc = unsafe { vmaf_read_pictures(ctx.ctx, &mut r, &mut d, 0) };
    assert_eq!(rc, 0);
    flush(&ctx);

    let (s, _, _, _) = float_ssim_lcs(
        &reference.planes[0],
        &distorted.planes[0],
        640,
        384,
        640,
        10,
    )
    .unwrap();
    let oracle = ctx.feature("float_ssim", 0);
    assert!(
        (s - oracle).abs() <= 2e-6,
        "float_ssim 10-bit: ours {s} vs libvmaf {oracle}"
    );
}
