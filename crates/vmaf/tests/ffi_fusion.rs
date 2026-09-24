use std::ffi::CString;
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "parallel")]
use vmaf::VmafV1Scorer;
use vmaf::{
    ModelVariant, PoolingMethod, VmafFeatures, VmafModel, VmafV0Features, VmafV0Model,
    VmafV0Stream, VmafV0Variant, VmafV1Stream, Yuv420Frame, adm2_v0_from_luma, adm3_v1_from_luma,
    cambi_v1_from_luma, motion2_v0_from_luma, motion3_from_luma, pool_v0_scores, pool_v1_scores,
    score_v0_420, score_v1_420, speed_v1_chroma_420, vif_v0_from_luma,
};
use vmaf_head_sys::*;

const WIDTH: usize = 640;
const HEIGHT: usize = 360;
const FRAME_COUNT: usize = 5;
static TRACE_ID: AtomicU64 = AtomicU64::new(0);

struct Session {
    context: *mut VmafContext,
    model: *mut vmaf_head_sys::VmafModel,
}

impl Session {
    fn new(version: &str) -> Self {
        let cfg = VmafConfiguration {
            log_level: VmafLogLevel_VMAF_LOG_LEVEL_NONE,
            n_threads: 1,
            n_subsample: 1,
            cpumask: 0,
            gpumask: 0,
        };
        let mut context = ptr::null_mut();
        assert_eq!(unsafe { vmaf_init(&mut context, cfg) }, 0);
        let name = CString::new("vmaf_v1_oracle").unwrap();
        let version = CString::new(version).unwrap();
        let mut model_cfg = VmafModelConfig {
            name: name.as_ptr(),
            flags: VmafModelFlags_VMAF_MODEL_FLAGS_DEFAULT as u64,
        };
        let mut model = ptr::null_mut();
        let rc = unsafe { vmaf_model_load(&mut model, &mut model_cfg, version.as_ptr()) };
        assert_eq!(rc, 0, "vmaf_model_load failed: {rc}");
        let rc = unsafe { vmaf_use_features_from_model(context, model) };
        assert_eq!(rc, 0, "vmaf_use_features_from_model failed: {rc}");
        Self { context, model }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            vmaf_model_destroy(self.model);
            assert_eq!(vmaf_close(self.context), 0);
        }
    }
}

struct Frame {
    planes: [Vec<u16>; 3],
}

fn frame(index: usize, bit_depth: u32, distorted: bool) -> Frame {
    let mut planes = [
        vec![0; WIDTH * HEIGHT],
        vec![0; WIDTH * HEIGHT / 4],
        vec![0; WIDTH * HEIGHT / 4],
    ];
    let multiplier = 1 << (bit_depth - 8);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let base = 16 + ((x * 3 + y * 5 + index * 7 + (x / 17) * 9) % 220);
            let signal = if distorted {
                let quantized = if x < WIDTH / 2 {
                    (base / 12) * 12
                } else {
                    base
                };
                quantized.saturating_sub((index + y / 32) % 6)
            } else {
                base
            };
            let extra = if bit_depth == 10 {
                (x + y + index) % 4
            } else {
                0
            };
            planes[0][y * WIDTH + x] = (signal * multiplier + extra) as u16;
        }
    }
    for y in 0..HEIGHT / 2 {
        for x in 0..WIDTH / 2 {
            let slot = y * WIDTH / 2 + x;
            let u = 16 + ((x * 7 + y * 3 + index * 11) % 224);
            let v = 16 + ((x * 2 + y * 9 + index * 13) % 224);
            planes[1][slot] = ((u + usize::from(distorted) * 3).min(240) * multiplier) as u16;
            planes[2][slot] = (v.saturating_sub(usize::from(distorted) * 5) * multiplier) as u16;
        }
    }
    Frame { planes }
}

fn banded_frame(index: usize, bit_depth: u32, distorted: bool) -> Frame {
    let mut frame = frame(index, bit_depth, distorted);
    let (offset, range) = if bit_depth == 8 { (16, 105) } else { (64, 420) };
    for y in 0..HEIGHT {
        for x in 0..WIDTH / 3 {
            let v = offset + x * range / WIDTH;
            frame.planes[0][y * WIDTH + x] = if bit_depth == 10 && distorted {
                (v & !3) as u16
            } else {
                v as u16
            };
        }
    }
    frame
}

fn large_frame(
    index: usize,
    bit_depth: u32,
    distorted: bool,
    width: usize,
    height: usize,
) -> Frame {
    let mut planes = [
        vec![0; width * height],
        vec![0; width * height / 4],
        vec![0; width * height / 4],
    ];
    let multiplier = 1 << (bit_depth - 8);
    for y in 0..height {
        for x in 0..width {
            let base = 16 + ((x * 3 + y * 5 + index * 7 + (x / 17) * 9) % 220);
            let value = if distorted && x < width / 2 {
                (base / 12) * 12
            } else {
                base
            };
            planes[0][y * width + x] = (value * multiplier + (x + y + index) % multiplier) as u16;
        }
    }
    for y in 0..height / 2 {
        for x in 0..width / 2 {
            let slot = y * width / 2 + x;
            let u = 16 + ((x * 7 + y * 3 + (x / 9) * (y / 9) * 13 + index * 11) % 224);
            let v = 16 + ((x * 2 + y * 9 + (x / 13) * (y / 7) * 5 + index * 13) % 224);
            let du = if distorted {
                (u + (x + y + index) % 17).min(240)
            } else {
                u
            };
            let dv = if distorted {
                v.saturating_sub((x + 2 * y + index) % 19)
            } else {
                v
            };
            planes[1][slot] = (du * multiplier) as u16;
            planes[2][slot] = (dv * multiplier) as u16;
        }
    }
    Frame { planes }
}

fn sharpened_frame(index: usize, bit_depth: u32, distorted: bool) -> Frame {
    let mut frame = frame(index, bit_depth, false);
    if !distorted {
        return frame;
    }
    let original = frame.planes[0].clone();
    let maximum = (1 << bit_depth) - 1;
    for y in 1..HEIGHT - 1 {
        for x in 1..WIDTH - 1 {
            let pos = y * WIDTH + x;
            let mid = original[pos] as i32;
            let laplacian = 4 * mid
                - original[pos - 1] as i32
                - original[pos + 1] as i32
                - original[pos - WIDTH] as i32
                - original[pos + WIDTH] as i32;
            frame.planes[0][pos] = (mid + laplacian / 2).clamp(0, maximum) as u16;
        }
    }
    frame
}

fn picture(frame: &Frame, bit_depth: u32, image_width: usize, image_height: usize) -> VmafPicture {
    let mut picture = MaybeUninit::<VmafPicture>::uninit();
    let rc = unsafe {
        vmaf_picture_alloc(
            picture.as_mut_ptr(),
            VmafPixelFormat_VMAF_PIX_FMT_YUV420P,
            bit_depth,
            image_width as u32,
            image_height as u32,
        )
    };
    assert_eq!(rc, 0, "vmaf_picture_alloc failed: {rc}");
    let picture = unsafe { picture.assume_init() };
    let bytes = if bit_depth == 8 { 1 } else { 2 };
    for plane in 0..3 {
        let (width, height) = if plane == 0 {
            (image_width, image_height)
        } else {
            (image_width / 2, image_height / 2)
        };
        assert_eq!(picture.w[plane], width as u32);
        assert_eq!(picture.h[plane], height as u32);
        assert!(picture.stride[plane] >= (width * bytes) as isize);
        for y in 0..height {
            for x in 0..width {
                let sample = frame.planes[plane][y * width + x];
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

fn exact_feature(
    context: *mut VmafContext,
    keys: &serde_json::Map<String, serde_json::Value>,
    index: usize,
    alias: &str,
    original: &str,
) -> f64 {
    let matches: Vec<_> = keys
        .keys()
        .filter(|key| key.starts_with(alias) || key.starts_with(original))
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "expected exactly one {alias} feature at frame {index}, got {matches:?} from {:?}",
        keys.keys().collect::<Vec<_>>()
    );
    let key = matches[0];
    let raw = if key == alias { original } else { key };
    let raw = CString::new(raw).unwrap();
    let mut value = f64::NAN;
    let rc =
        unsafe { vmaf_feature_score_at_index(context, raw.as_ptr(), &mut value, index as u32) };
    assert_eq!(rc, 0, "{alias}: vmaf_feature_score_at_index failed: {rc}");
    assert!(value.is_finite(), "{alias}: non-finite at frame {index}");
    value
}

fn oracle_in_dimensions<F: Fn(usize, u32, bool, usize, usize) -> Frame>(
    variant: ModelVariant,
    bit_depth: u32,
    distorted: bool,
    width: usize,
    height: usize,
    frame_count: usize,
    make_frame: F,
) -> Vec<(VmafFeatures, f64)> {
    let session = Session::new(variant.built_in_name());
    for index in 0..frame_count {
        let reference = make_frame(index, bit_depth, false, width, height);
        let distortion = make_frame(index, bit_depth, distorted, width, height);
        let mut reference = picture(&reference, bit_depth, width, height);
        let mut distortion = picture(&distortion, bit_depth, width, height);
        let rc = unsafe {
            vmaf_read_pictures(
                session.context,
                &mut reference,
                &mut distortion,
                index as u32,
            )
        };
        if rc != 0 {
            unsafe {
                vmaf_picture_unref(&mut reference);
                vmaf_picture_unref(&mut distortion);
            }
        }
        assert_eq!(rc, 0, "vmaf_read_pictures failed at frame {index}: {rc}");
    }
    let rc = unsafe { vmaf_read_pictures(session.context, ptr::null_mut(), ptr::null_mut(), 0) };
    assert_eq!(rc, 0, "vmaf flush failed: {rc}");
    let trace: PathBuf = std::env::temp_dir().join(format!(
        "zenmetrics-vmaf-v1-{}-{}.json",
        std::process::id(),
        TRACE_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let trace_path = CString::new(trace.to_str().unwrap()).unwrap();
    let rc = unsafe {
        vmaf_write_output(
            session.context,
            trace_path.as_ptr(),
            VmafOutputFormat_VMAF_OUTPUT_FORMAT_JSON,
        )
    };
    assert_eq!(rc, 0, "vmaf_write_output failed: {rc}");
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&trace).unwrap()).unwrap();
    let frames = json["frames"].as_array().unwrap();
    assert_eq!(frames.len(), frame_count);
    let mut out = Vec::with_capacity(frame_count);
    for (index, item) in frames.iter().enumerate() {
        assert_eq!(item["frameNum"].as_u64(), Some(index as u64));
        let keys = item["metrics"].as_object().unwrap();
        let features = VmafFeatures {
            cambi: exact_feature(
                session.context,
                keys,
                index,
                "cambi",
                "Cambi_feature_cambi_score",
            ),
            speed_chroma_uv: exact_feature(
                session.context,
                keys,
                index,
                "speed_chroma_uv",
                "Speed_chroma_feature_speed_chroma_uv_score",
            ),
            adm3: exact_feature(
                session.context,
                keys,
                index,
                "integer_adm3",
                "VMAF_integer_feature_adm3_score",
            ),
            motion3: exact_feature(
                session.context,
                keys,
                index,
                "integer_motion3",
                "VMAF_integer_feature_motion3_score",
            ),
        };
        let mut score = f64::NAN;
        let rc = unsafe {
            vmaf_score_at_index(session.context, session.model, &mut score, index as u32)
        };
        assert_eq!(rc, 0, "vmaf_score_at_index failed at frame {index}: {rc}");
        assert!(score.is_finite(), "non-finite VMAF at frame {index}");
        out.push((features, score));
    }
    out
}

fn oracle_with(
    variant: ModelVariant,
    bit_depth: u32,
    distorted: bool,
    make_frame: fn(usize, u32, bool) -> Frame,
) -> Vec<(VmafFeatures, f64)> {
    oracle_in_dimensions(
        variant,
        bit_depth,
        distorted,
        WIDTH,
        HEIGHT,
        FRAME_COUNT,
        |index, depth, distorted, _, _| make_frame(index, depth, distorted),
    )
}

fn oracle(variant: ModelVariant, bit_depth: u32, distorted: bool) -> Vec<(VmafFeatures, f64)> {
    oracle_with(variant, bit_depth, distorted, frame)
}

#[test]
fn v1_fusion_matches_v321_for_both_depths_stills_and_hfr() {
    for variant in [
        ModelVariant::Standard1080p,
        ModelVariant::Phone,
        ModelVariant::Default4k,
        ModelVariant::Consumer4k,
        ModelVariant::HfrStandard1080p,
        ModelVariant::HfrPhone,
        ModelVariant::HfrDefault4k,
        ModelVariant::HfrConsumer4k,
    ] {
        let model = VmafModel::new(variant).unwrap();
        for bit_depth in [8, 10] {
            for distorted in [false, true] {
                for (index, (features, expected)) in oracle(variant, bit_depth, distorted)
                    .into_iter()
                    .enumerate()
                {
                    let actual = model.predict(features).unwrap();
                    assert!(
                        (actual - expected).abs() <= 1e-5,
                        "{variant:?}, {bit_depth} bit, distorted={distorted}, frame={index}: \
                         features={features:?}, pure Rust={actual}, libvmaf={expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn motion3_matches_v321_for_standard_and_hfr_frames() {
    for (variant, hfr) in [
        (ModelVariant::Standard1080p, false),
        (ModelVariant::HfrStandard1080p, true),
    ] {
        for bit_depth in [8, 10] {
            let expected = oracle(variant, bit_depth, true);
            let owned: Vec<_> = (0..FRAME_COUNT)
                .map(|index| frame(index, bit_depth, false).planes[0].clone())
                .collect();
            let frames: Vec<&[u16]> = owned.iter().map(Vec::as_slice).collect();
            let actual = motion3_from_luma(&frames, WIDTH, HEIGHT, bit_depth as u8, hfr).unwrap();
            assert_eq!(actual.len(), expected.len());
            for (index, (actual, (features, _))) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (actual - features.motion3).abs() <= 1e-9,
                    "{variant:?}, {bit_depth} bit, frame={index}: \
                     pure Rust motion3={actual}, libvmaf motion3={}",
                    features.motion3
                );
            }
        }
    }
}

#[test]
fn oracle_detects_smooth_ramp_banding() {
    for bit_depth in [8, 10] {
        let result = oracle_with(ModelVariant::Standard1080p, bit_depth, true, banded_frame);
        assert!(
            result
                .iter()
                .all(|(features, _)| features.cambi > 0.0 && features.cambi < 17.0),
            "CAMBI signal absent or clipped at {bit_depth} bits: {:?}",
            result
                .iter()
                .map(|(features, _)| features.cambi)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn cambi_matches_v321_for_textured_and_banded_frames() {
    for bit_depth in [8, 10] {
        for (banded, make_frame) in [
            (false, frame as fn(usize, u32, bool) -> Frame),
            (true, banded_frame as fn(usize, u32, bool) -> Frame),
        ] {
            let expected = oracle_with(ModelVariant::Standard1080p, bit_depth, true, make_frame);
            for (index, (features, _)) in expected.into_iter().enumerate() {
                let distorted = make_frame(index, bit_depth, true);
                let actual =
                    cambi_v1_from_luma(&distorted.planes[0], WIDTH, HEIGHT, bit_depth as u8)
                        .unwrap();
                assert!(
                    (actual - features.cambi).abs() <= 1e-3,
                    "{bit_depth} bit, frame={index}: pure Rust CAMBI={actual}, \
                     libvmaf CAMBI={}, banded={banded}",
                    features.cambi,
                );
            }
        }
    }
}

#[test]
fn motion3_rejects_oversized_dimensions_without_overflow() {
    assert!(motion3_from_luma(&[&[]], usize::MAX, 8, 8, false).is_err());
}

#[test]
fn cambi_high_resolution_speedup_matches_v321() {
    let width = 1920;
    let height = 1080;
    let mut rendered = Frame {
        planes: [
            vec![0; width * height],
            vec![512; width * height / 4],
            vec![512; width * height / 4],
        ],
    };
    for y in 0..height {
        for x in 0..width {
            rendered.planes[0][y * width + x] = if x < width / 3 {
                ((64 + x * 420 / width) & !3) as u16
            } else {
                (64 + ((x * 3 + y * 5) % 220) * 4) as u16
            };
        }
    }
    let cfg = VmafConfiguration {
        log_level: VmafLogLevel_VMAF_LOG_LEVEL_NONE,
        n_threads: 1,
        n_subsample: 1,
        cpumask: 0,
        gpumask: 0,
    };
    let mut context = ptr::null_mut();
    assert_eq!(unsafe { vmaf_init(&mut context, cfg) }, 0);
    let mut options = ptr::null_mut();
    for (key, value) in [
        ("cambi_high_res_speedup", "1080"),
        ("cambi_vis_lum_threshold", "0.06"),
        ("cambi_max_val", "17"),
    ] {
        let key = CString::new(key).unwrap();
        let value = CString::new(value).unwrap();
        assert_eq!(
            unsafe { vmaf_feature_dictionary_set(&mut options, key.as_ptr(), value.as_ptr()) },
            0
        );
    }
    let feature = CString::new("cambi").unwrap();
    let rc = unsafe { vmaf_use_feature(context, feature.as_ptr(), options) };
    if rc != 0 {
        unsafe { vmaf_feature_dictionary_free(&mut options) };
    }
    assert_eq!(rc, 0, "vmaf_use_feature(cambi) failed: {rc}");
    let mut reference = picture(&rendered, 10, width, height);
    let mut distortion = picture(&rendered, 10, width, height);
    let rc = unsafe { vmaf_read_pictures(context, &mut reference, &mut distortion, 0) };
    if rc != 0 {
        unsafe {
            vmaf_picture_unref(&mut reference);
            vmaf_picture_unref(&mut distortion);
        }
    }
    assert_eq!(rc, 0, "vmaf_read_pictures failed: {rc}");
    assert_eq!(
        unsafe { vmaf_read_pictures(context, ptr::null_mut(), ptr::null_mut(), 0) },
        0
    );
    let trace = std::env::temp_dir().join(format!(
        "zenmetrics-vmaf-cambi-1080-{}-{}.json",
        std::process::id(),
        TRACE_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let trace_path = CString::new(trace.to_str().unwrap()).unwrap();
    assert_eq!(
        unsafe {
            vmaf_write_output(
                context,
                trace_path.as_ptr(),
                VmafOutputFormat_VMAF_OUTPUT_FORMAT_JSON,
            )
        },
        0
    );
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&trace).unwrap()).unwrap();
    let metrics = json["frames"][0]["metrics"].as_object().unwrap();
    let expected = exact_feature(context, metrics, 0, "cambi", "Cambi_feature_cambi_score");
    assert_eq!(unsafe { vmaf_close(context) }, 0);
    let actual = cambi_v1_from_luma(&rendered.planes[0], width, height, 10).unwrap();
    assert!(
        (0.0..17.0).contains(&expected) && expected > 0.0,
        "high-resolution CAMBI fixture is clipped or invalid: {expected}"
    );
    assert!(
        (actual - expected).abs() <= 1e-3,
        "1920x1080 speedup CAMBI: pure Rust={actual}, libvmaf={expected}"
    );
}

#[test]
fn prescaled_chroma_oracle_is_nontrivial() {
    for variant in [
        ModelVariant::Standard1080p,
        ModelVariant::Phone,
        ModelVariant::Consumer4k,
    ] {
        for bit_depth in [8, 10] {
            let result = oracle_in_dimensions(variant, bit_depth, true, 1280, 720, 2, large_frame);
            assert!(
                result.iter().any(|(features, _)| {
                    features.speed_chroma_uv > 0.0 && features.speed_chroma_uv < 45.0
                }),
                "{variant:?} at {bit_depth} bits has no unclipped SpEED signal: {:?}",
                result
                    .iter()
                    .map(|(features, _)| features.speed_chroma_uv)
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn chroma_speed_matches_v321_with_and_without_prescale() {
    let width = 1280;
    let height = 720;
    for variant in [
        ModelVariant::Standard1080p,
        ModelVariant::Phone,
        ModelVariant::Consumer4k,
    ] {
        for bit_depth in [8, 10] {
            for distorted in [false, true] {
                let expected = oracle_in_dimensions(
                    variant,
                    bit_depth,
                    distorted,
                    width,
                    height,
                    2,
                    large_frame,
                );
                for (index, (features, _)) in expected.iter().enumerate() {
                    let reference = large_frame(index, bit_depth, false, width, height);
                    let distortion = large_frame(index, bit_depth, distorted, width, height);
                    let actual = speed_v1_chroma_420(
                        &reference.planes[1],
                        &reference.planes[2],
                        &distortion.planes[1],
                        &distortion.planes[2],
                        width,
                        height,
                        bit_depth as u8,
                        variant,
                    )
                    .unwrap();
                    assert!(
                        (actual - features.speed_chroma_uv).abs() <= 1e-3,
                        "{variant:?}, {bit_depth} bit, distorted={distorted}, frame={index}: \
                         pure Rust SpEED={actual}, libvmaf SpEED={}",
                        features.speed_chroma_uv
                    );
                }
            }
        }
    }
}

#[test]
fn adm3_matches_v321_for_v1_viewing_distances_and_neg() {
    for variant in [
        ModelVariant::Standard1080p,
        ModelVariant::Phone,
        ModelVariant::Default4k,
        ModelVariant::Consumer4k,
    ] {
        for bit_depth in [8, 10] {
            for (scene, make_frame) in [
                (false, frame as fn(usize, u32, bool) -> Frame),
                (true, sharpened_frame as fn(usize, u32, bool) -> Frame),
            ] {
                let expected = oracle_with(variant, bit_depth, true, make_frame);
                for (index, (features, _)) in expected.into_iter().enumerate() {
                    let reference = make_frame(index, bit_depth, false);
                    let distortion = make_frame(index, bit_depth, true);
                    let actual = adm3_v1_from_luma(
                        &reference.planes[0],
                        &distortion.planes[0],
                        WIDTH,
                        HEIGHT,
                        bit_depth as u8,
                        variant,
                    )
                    .unwrap();
                    assert!(
                        (actual - features.adm3).abs() <= 1e-4,
                        "{variant:?}, {bit_depth} bit, sharpened={scene}, frame={index}: \
                         pure Rust ADM3={actual}, libvmaf ADM3={}",
                        features.adm3
                    );
                }
            }
        }
    }
}

fn yuv(frame: &Frame) -> Yuv420Frame<'_> {
    Yuv420Frame {
        y: &frame.planes[0],
        u: &frame.planes[1],
        v: &frame.planes[2],
    }
}

#[test]
fn full_v1_pixel_scores_match_v321() {
    for variant in [
        ModelVariant::Standard1080p,
        ModelVariant::Phone,
        ModelVariant::Default4k,
        ModelVariant::Consumer4k,
        ModelVariant::HfrStandard1080p,
        ModelVariant::HfrPhone,
        ModelVariant::HfrDefault4k,
        ModelVariant::HfrConsumer4k,
    ] {
        for bit_depth in [8, 10] {
            for distorted in [false, true] {
                let expected = oracle(variant, bit_depth, distorted);
                let reference: Vec<_> = (0..FRAME_COUNT)
                    .map(|i| frame(i, bit_depth, false))
                    .collect();
                let distortion: Vec<_> = (0..FRAME_COUNT)
                    .map(|i| frame(i, bit_depth, distorted))
                    .collect();
                let actual = score_v1_420(
                    &reference.iter().map(yuv).collect::<Vec<_>>(),
                    &distortion.iter().map(yuv).collect::<Vec<_>>(),
                    WIDTH,
                    HEIGHT,
                    bit_depth as u8,
                    variant,
                )
                .unwrap();
                for (i, (ours, (oracle_features, oracle_score))) in
                    actual.iter().zip(expected.iter()).enumerate()
                {
                    assert!(
                        (ours.features.cambi - oracle_features.cambi).abs() <= 1e-3
                            && (ours.features.speed_chroma_uv - oracle_features.speed_chroma_uv)
                                .abs()
                                <= 1e-3
                            && (ours.features.adm3 - oracle_features.adm3).abs() <= 1e-4
                            && (ours.features.motion3 - oracle_features.motion3).abs() <= 1e-8,
                        "{variant:?}, {bit_depth} bit, distorted={distorted}, frame={i}: \
                         pure Rust={:?}, libvmaf={oracle_features:?}",
                        ours.features
                    );
                    assert!(
                        (ours.score - oracle_score).abs() <= 0.02,
                        "{variant:?}, {bit_depth} bit, distorted={distorted}, frame={i}: \
                         pure Rust={}, libvmaf={oracle_score}",
                        ours.score
                    );
                }
            }
        }
    }
}

#[test]
fn full_v1_pixel_scores_with_nonzero_prescaled_chroma_match_v321() {
    let width = 1280;
    let height = 720;
    for variant in [ModelVariant::Phone, ModelVariant::Consumer4k] {
        for bit_depth in [8, 10] {
            let expected =
                oracle_in_dimensions(variant, bit_depth, true, width, height, 2, large_frame);
            let reference: Vec<_> = (0..2)
                .map(|i| large_frame(i, bit_depth, false, width, height))
                .collect();
            let distortion: Vec<_> = (0..2)
                .map(|i| large_frame(i, bit_depth, true, width, height))
                .collect();
            let actual = score_v1_420(
                &reference.iter().map(yuv).collect::<Vec<_>>(),
                &distortion.iter().map(yuv).collect::<Vec<_>>(),
                width,
                height,
                bit_depth as u8,
                variant,
            )
            .unwrap();
            for (i, (ours, (features, score))) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    features.speed_chroma_uv > 0.0,
                    "{variant:?} at {bit_depth} bits has zero chroma SpEED"
                );
                assert!(
                    (ours.score - score).abs() <= 0.02,
                    "{variant:?}, {bit_depth} bit, frame={i}: pure Rust={}, libvmaf={score}, \
                     pure Rust features={:?}, libvmaf features={features:?}",
                    ours.score,
                    ours.features
                );
            }
        }
    }
}

#[test]
fn v1_pooling_matches_libvmaf() {
    for (variant, bit_depth) in [
        (ModelVariant::Standard1080p, 8),
        (ModelVariant::HfrPhone, 10),
    ] {
        let session = Session::new(variant.built_in_name());
        for index in 0..FRAME_COUNT {
            let mut reference = picture(&frame(index, bit_depth, false), bit_depth, WIDTH, HEIGHT);
            let mut distortion = picture(&frame(index, bit_depth, true), bit_depth, WIDTH, HEIGHT);
            assert_eq!(
                unsafe {
                    vmaf_read_pictures(
                        session.context,
                        &mut reference,
                        &mut distortion,
                        index as u32,
                    )
                },
                0
            );
        }
        assert_eq!(
            unsafe { vmaf_read_pictures(session.context, ptr::null_mut(), ptr::null_mut(), 0) },
            0
        );
        let scores: Vec<_> = (0..FRAME_COUNT)
            .map(|index| {
                let mut value = f64::NAN;
                assert_eq!(
                    unsafe {
                        vmaf_score_at_index(
                            session.context,
                            session.model,
                            &mut value,
                            index as u32,
                        )
                    },
                    0
                );
                value
            })
            .collect();
        for (method, c_method) in [
            (PoolingMethod::Mean, VmafPoolingMethod_VMAF_POOL_METHOD_MEAN),
            (PoolingMethod::Min, VmafPoolingMethod_VMAF_POOL_METHOD_MIN),
            (PoolingMethod::Max, VmafPoolingMethod_VMAF_POOL_METHOD_MAX),
            (
                PoolingMethod::HarmonicMean,
                VmafPoolingMethod_VMAF_POOL_METHOD_HARMONIC_MEAN,
            ),
        ] {
            let mut expected = f64::NAN;
            assert_eq!(
                unsafe {
                    vmaf_score_pooled(
                        session.context,
                        session.model,
                        c_method,
                        &mut expected,
                        0,
                        (FRAME_COUNT - 1) as u32,
                    )
                },
                0
            );
            let actual = pool_v1_scores(&scores, method).unwrap();
            assert!(
                (actual - expected).abs() <= 1e-8,
                "{variant:?}, {method:?}: pure Rust={actual}, libvmaf={expected}"
            );
        }
    }
    assert!(pool_v1_scores(&[], PoolingMethod::Mean).is_err());
    assert!(pool_v1_scores(&[f64::NAN], PoolingMethod::Mean).is_err());
}

#[test]
fn v1_rejects_invalid_planar_input() {
    let frame = Yuv420Frame {
        y: &[],
        u: &[],
        v: &[],
    };
    assert!(score_v1_420(&[], &[], WIDTH, HEIGHT, 8, ModelVariant::Standard1080p).is_err());
    assert!(score_v1_420(&[frame], &[], WIDTH, HEIGHT, 8, ModelVariant::Standard1080p).is_err());
    assert!(
        score_v1_420(
            &[frame],
            &[frame],
            WIDTH,
            HEIGHT,
            8,
            ModelVariant::Standard1080p
        )
        .is_err()
    );
    assert!(adm3_v1_from_luma(&[], &[], 16, 16, 8, ModelVariant::Standard1080p).is_err());
    assert!(adm3_v1_from_luma(&[], &[], 17, 17, 12, ModelVariant::Standard1080p).is_err());
    assert!(speed_v1_chroma_420(&[], &[], &[], &[], 80, 80, 8, ModelVariant::Phone).is_err());
}

#[cfg(feature = "parallel")]
#[test]
fn bounded_parallel_frames_match_serial_bit_for_bit() {
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    assert!(
        VmafV1Scorer::new(WIDTH, HEIGHT, 8, ModelVariant::Standard1080p)
            .unwrap()
            .with_threads(0)
            .is_err()
    );
    assert!(
        VmafV1Scorer::new(WIDTH, HEIGHT, 8, ModelVariant::Standard1080p)
            .unwrap()
            .with_threads(available + 1)
            .is_err()
    );
    for variant in [
        ModelVariant::Standard1080p,
        ModelVariant::Phone,
        ModelVariant::Default4k,
        ModelVariant::Consumer4k,
        ModelVariant::HfrStandard1080p,
        ModelVariant::HfrPhone,
        ModelVariant::HfrDefault4k,
        ModelVariant::HfrConsumer4k,
    ] {
        for bit_depth in [8, 10] {
            let reference: Vec<_> = (0..FRAME_COUNT)
                .map(|i| frame(i, bit_depth, false))
                .collect();
            let distortion: Vec<_> = (0..FRAME_COUNT)
                .map(|i| frame(i, bit_depth, true))
                .collect();
            let ref_frames: Vec<_> = reference.iter().map(yuv).collect();
            let dis_frames: Vec<_> = distortion.iter().map(yuv).collect();
            let serial = score_v1_420(
                &ref_frames,
                &dis_frames,
                WIDTH,
                HEIGHT,
                bit_depth as u8,
                variant,
            )
            .unwrap();
            for thread_count in [1, 2, 4].into_iter().filter(|&n| n <= available) {
                let scorer = VmafV1Scorer::new(WIDTH, HEIGHT, bit_depth as u8, variant)
                    .unwrap()
                    .with_threads(thread_count)
                    .unwrap();
                for _ in 0..2 {
                    let actual = scorer.score(&ref_frames, &dis_frames).unwrap();
                    assert_eq!(actual.len(), serial.len());
                    for (i, (expected, got)) in serial.iter().zip(&actual).enumerate() {
                        for (feature_name, a, b) in [
                            ("cambi", expected.features.cambi, got.features.cambi),
                            (
                                "chroma speed",
                                expected.features.speed_chroma_uv,
                                got.features.speed_chroma_uv,
                            ),
                            ("adm3", expected.features.adm3, got.features.adm3),
                            ("motion3", expected.features.motion3, got.features.motion3),
                            ("score", expected.score, got.score),
                        ] {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "{variant:?}, {bit_depth} bit, {thread_count} threads, frame {i}, {feature_name}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn streaming_scores_match_batch_across_motion_boundaries() {
    for variant in [
        ModelVariant::Standard1080p,
        ModelVariant::Phone,
        ModelVariant::Default4k,
        ModelVariant::Consumer4k,
        ModelVariant::HfrStandard1080p,
        ModelVariant::HfrPhone,
        ModelVariant::HfrDefault4k,
        ModelVariant::HfrConsumer4k,
    ] {
        for bit_depth in [8, 10] {
            let reference: Vec<_> = (0..FRAME_COUNT)
                .map(|i| frame(i, bit_depth, false))
                .collect();
            let distortion: Vec<_> = (0..FRAME_COUNT)
                .map(|i| frame(i, bit_depth, true))
                .collect();
            let ref_frames: Vec<_> = reference.iter().map(yuv).collect();
            let dis_frames: Vec<_> = distortion.iter().map(yuv).collect();
            for count in [1, 2, 3, FRAME_COUNT] {
                let expected = score_v1_420(
                    &ref_frames[..count],
                    &dis_frames[..count],
                    WIDTH,
                    HEIGHT,
                    bit_depth as u8,
                    variant,
                )
                .unwrap();
                let mut stream =
                    VmafV1Stream::new(WIDTH, HEIGHT, bit_depth as u8, variant).unwrap();
                let mut actual = Vec::new();
                for i in 0..count {
                    actual.extend(stream.push(ref_frames[i], dis_frames[i]).unwrap());
                    assert!(actual.len() <= i + 1);
                }
                actual.extend(stream.finish().unwrap());
                assert_eq!(actual.len(), expected.len());
                for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                    for (name, got, want) in [
                        ("cambi", a.features.cambi, b.features.cambi),
                        (
                            "chroma speed",
                            a.features.speed_chroma_uv,
                            b.features.speed_chroma_uv,
                        ),
                        ("adm3", a.features.adm3, b.features.adm3),
                        ("motion3", a.features.motion3, b.features.motion3),
                        ("score", a.score, b.score),
                    ] {
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "{variant:?}, {bit_depth} bit, {count} frames, index {i}, {name}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn streaming_rejects_invalid_frame_without_advancing() {
    let reference = frame(0, 8, false);
    let distorted = frame(0, 8, true);
    let ref_frame = yuv(&reference);
    let dis_frame = yuv(&distorted);
    let invalid = Yuv420Frame {
        y: &[],
        u: &[],
        v: &[],
    };
    let mut stream = VmafV1Stream::new(WIDTH, HEIGHT, 8, ModelVariant::Standard1080p).unwrap();
    assert!(stream.push(ref_frame, invalid).is_err());
    assert!(stream.push(ref_frame, dis_frame).unwrap().is_empty());
    let got = stream.finish().unwrap();
    let expected = score_v1_420(
        &[ref_frame],
        &[dis_frame],
        WIDTH,
        HEIGHT,
        8,
        ModelVariant::Standard1080p,
    )
    .unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].score.to_bits(), expected[0].score.to_bits());
}

#[test]
fn explicit_v1_neg_alias_matches_official_v1_model() {
    let reference = frame(0, 8, false);
    let distorted = banded_frame(0, 8, true);
    let reference = [yuv(&reference)];
    let distorted = [yuv(&distorted)];
    let official = score_v1_420(
        &reference,
        &distorted,
        WIDTH,
        HEIGHT,
        8,
        ModelVariant::Standard1080p,
    )
    .unwrap();
    let neg = score_v1_420(
        &reference,
        &distorted,
        WIDTH,
        HEIGHT,
        8,
        ModelVariant::V1_NEG,
    )
    .unwrap();
    assert_eq!(
        ModelVariant::V1_NEG.built_in_name(),
        ModelVariant::Standard1080p.built_in_name()
    );
    assert_eq!(neg[0].score.to_bits(), official[0].score.to_bits());
    assert_eq!(
        neg[0].features.adm3.to_bits(),
        official[0].features.adm3.to_bits()
    );
}

fn oracle_v0(
    name: &str,
    bit_depth: u32,
    distorted: bool,
    make_frame: fn(usize, u32, bool) -> Frame,
) -> Vec<([f64; 6], f64)> {
    let session = Session::new(name);
    for index in 0..3 {
        let reference = make_frame(index, bit_depth, false);
        let distortion = make_frame(index, bit_depth, distorted);
        let mut reference = picture(&reference, bit_depth, WIDTH, HEIGHT);
        let mut distortion = picture(&distortion, bit_depth, WIDTH, HEIGHT);
        let rc = unsafe {
            vmaf_read_pictures(
                session.context,
                &mut reference,
                &mut distortion,
                index as u32,
            )
        };
        if rc != 0 {
            unsafe {
                vmaf_picture_unref(&mut reference);
                vmaf_picture_unref(&mut distortion);
            }
        }
        assert_eq!(rc, 0, "{name}: vmaf_read_pictures failed at {index}: {rc}");
    }
    assert_eq!(
        unsafe { vmaf_read_pictures(session.context, ptr::null_mut(), ptr::null_mut(), 0) },
        0
    );
    let trace = std::env::temp_dir().join(format!(
        "zenmetrics-vmaf-v0-{}-{}.json",
        std::process::id(),
        TRACE_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let trace_path = CString::new(trace.to_str().unwrap()).unwrap();
    assert_eq!(
        unsafe {
            vmaf_write_output(
                session.context,
                trace_path.as_ptr(),
                VmafOutputFormat_VMAF_OUTPUT_FORMAT_JSON,
            )
        },
        0
    );
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&trace).unwrap()).unwrap();
    std::fs::remove_file(&trace).unwrap();
    let frames = json["frames"].as_array().unwrap();
    assert_eq!(frames.len(), 3);
    let names = [
        ("integer_adm2", "VMAF_integer_feature_adm2_score"),
        ("integer_motion2", "VMAF_integer_feature_motion2_score"),
        (
            "integer_vif_scale0",
            "VMAF_integer_feature_vif_scale0_score",
        ),
        (
            "integer_vif_scale1",
            "VMAF_integer_feature_vif_scale1_score",
        ),
        (
            "integer_vif_scale2",
            "VMAF_integer_feature_vif_scale2_score",
        ),
        (
            "integer_vif_scale3",
            "VMAF_integer_feature_vif_scale3_score",
        ),
    ];
    frames
        .iter()
        .enumerate()
        .map(|(index, item)| {
            assert_eq!(item["frameNum"].as_u64(), Some(index as u64));
            let keys = item["metrics"].as_object().unwrap();
            let features = std::array::from_fn(|i| {
                exact_feature(session.context, keys, index, names[i].0, names[i].1)
            });
            let mut score = f64::NAN;
            assert_eq!(
                unsafe {
                    vmaf_score_at_index(session.context, session.model, &mut score, index as u32)
                },
                0
            );
            assert!(score.is_finite());
            (features, score)
        })
        .collect()
}

#[test]
fn v0_oracle_exercises_neg_gain_on_sharpening() {
    let original = oracle_v0("vmaf_v0.6.1", 8, true, sharpened_frame);
    let neg = oracle_v0("vmaf_v0.6.1neg", 8, true, sharpened_frame);
    assert!(original.iter().zip(&neg).any(|((a, _), (b, _))| {
        (a[0] - b[0]).abs() > 1e-5
            || a[2..]
                .iter()
                .zip(&b[2..])
                .any(|(x, y)| (x - y).abs() > 1e-5)
    }));
}

#[test]
fn v0_fusion_matches_v321_for_both_depths_and_neg() {
    for variant in [
        VmafV0Variant::Standard,
        VmafV0Variant::StandardNeg,
        VmafV0Variant::FourK,
        VmafV0Variant::FourKNeg,
    ] {
        let model = VmafV0Model::new(variant).unwrap();
        for bit_depth in [8, 10] {
            for distorted in [false, true] {
                for (index, (features, expected)) in oracle_v0(
                    variant.built_in_name(),
                    bit_depth,
                    distorted,
                    sharpened_frame,
                )
                .into_iter()
                .enumerate()
                {
                    let actual = model
                        .predict(VmafV0Features {
                            adm2: features[0],
                            motion2: features[1],
                            vif_scales: [features[2], features[3], features[4], features[5]],
                        })
                        .unwrap();
                    assert!(
                        (actual - expected).abs() <= 1e-5,
                        "{variant:?}, {bit_depth} bit, distorted={distorted}, frame={index}: \
                         features={features:?}, pure Rust={actual}, libvmaf={expected}"
                    );
                }
            }
        }
    }
    let model = VmafV0Model::new(VmafV0Variant::Standard).unwrap();
    assert!(
        model
            .predict(VmafV0Features {
                adm2: f64::NAN,
                motion2: 0.0,
                vif_scales: [0.0; 4]
            })
            .is_err()
    );
}

#[test]
fn v0_motion2_matches_v321_for_both_depths_and_neg() {
    for name in ["vmaf_v0.6.1", "vmaf_v0.6.1neg"] {
        for bit_depth in [8, 10] {
            let expected = oracle_v0(name, bit_depth, true, frame);
            let owned: Vec<_> = (0..3)
                .map(|index| frame(index, bit_depth, false).planes[0].clone())
                .collect();
            let frames: Vec<&[u16]> = owned.iter().map(Vec::as_slice).collect();
            let actual = motion2_v0_from_luma(&frames, WIDTH, HEIGHT, bit_depth as u8).unwrap();
            for (index, (&ours, &(features, _))) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (ours - features[1]).abs() <= 1e-8,
                    "{name}, {bit_depth} bit, frame {index}: Rust {ours}, libvmaf {}",
                    features[1]
                );
            }
        }
    }
    assert!(motion2_v0_from_luma(&[&[]], usize::MAX, 8, 8).is_err());
}

#[test]
fn v0_adm2_matches_v321_for_watson97_gain_and_neg() {
    let original = oracle_v0("vmaf_v0.6.1", 8, true, sharpened_frame);
    let neg = oracle_v0("vmaf_v0.6.1neg", 8, true, sharpened_frame);
    assert!(
        original
            .iter()
            .zip(&neg)
            .any(|((a, _), (b, _))| (a[0] - b[0]).abs() > 1e-6)
    );
    for variant in [
        VmafV0Variant::Standard,
        VmafV0Variant::StandardNeg,
        VmafV0Variant::FourK,
        VmafV0Variant::FourKNeg,
    ] {
        for bit_depth in [8, 10] {
            for distorted in [false, true] {
                let expected = oracle_v0(
                    variant.built_in_name(),
                    bit_depth,
                    distorted,
                    sharpened_frame,
                );
                for (index, (features, _)) in expected.into_iter().enumerate() {
                    let reference = sharpened_frame(index, bit_depth, false);
                    let distortion = sharpened_frame(index, bit_depth, distorted);
                    let actual = adm2_v0_from_luma(
                        &reference.planes[0],
                        &distortion.planes[0],
                        WIDTH,
                        HEIGHT,
                        bit_depth as u8,
                        variant,
                    )
                    .unwrap();
                    assert!(
                        (actual - features[0]).abs() <= 1e-4,
                        "{variant:?}, {bit_depth} bit, distorted={distorted}, frame={index}: \
                         pure Rust={actual}, libvmaf={}",
                        features[0]
                    );
                }
            }
        }
    }
}

#[test]
fn v0_vif_scales_match_v321_for_both_depths_and_neg() {
    let normal = oracle_v0("vmaf_v0.6.1", 8, true, sharpened_frame);
    let neg = oracle_v0("vmaf_v0.6.1neg", 8, true, sharpened_frame);
    assert!(normal.iter().zip(&neg).any(|((a, _), (b, _))| {
        a[2..]
            .iter()
            .zip(&b[2..])
            .any(|(x, y)| (x - y).abs() > 1e-6)
    }));
    for variant in [
        VmafV0Variant::Standard,
        VmafV0Variant::StandardNeg,
        VmafV0Variant::FourK,
        VmafV0Variant::FourKNeg,
    ] {
        for bit_depth in [8, 10] {
            for distorted in [false, true] {
                let expected = oracle_v0(
                    variant.built_in_name(),
                    bit_depth,
                    distorted,
                    sharpened_frame,
                );
                for (index, (features, _)) in expected.into_iter().enumerate() {
                    let reference = sharpened_frame(index, bit_depth, false);
                    let distortion = sharpened_frame(index, bit_depth, distorted);
                    let actual = vif_v0_from_luma(
                        &reference.planes[0],
                        &distortion.planes[0],
                        WIDTH,
                        HEIGHT,
                        bit_depth as u8,
                        variant,
                    )
                    .unwrap();
                    for (scale, (&ours, &oracle)) in actual.iter().zip(&features[2..]).enumerate() {
                        assert!(
                            (ours - oracle).abs() <= 1e-4,
                            "{variant:?}, {bit_depth} bit, distorted={distorted}, frame={index}, \
                             scale={scale}: pure Rust={ours}, libvmaf={oracle}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn full_v0_pixel_scores_match_v321() {
    for variant in [
        VmafV0Variant::Standard,
        VmafV0Variant::StandardNeg,
        VmafV0Variant::FourK,
        VmafV0Variant::FourKNeg,
    ] {
        for bit_depth in [8, 10] {
            for distorted in [false, true] {
                let expected = oracle_v0(
                    variant.built_in_name(),
                    bit_depth,
                    distorted,
                    sharpened_frame,
                );
                let reference: Vec<_> = (0..3)
                    .map(|i| sharpened_frame(i, bit_depth, false))
                    .collect();
                let distortion: Vec<_> = (0..3)
                    .map(|i| sharpened_frame(i, bit_depth, distorted))
                    .collect();
                let actual = score_v0_420(
                    &reference.iter().map(yuv).collect::<Vec<_>>(),
                    &distortion.iter().map(yuv).collect::<Vec<_>>(),
                    WIDTH,
                    HEIGHT,
                    bit_depth as u8,
                    variant,
                )
                .unwrap();
                for (index, (ours, (features, oracle))) in
                    actual.iter().zip(expected.iter()).enumerate()
                {
                    for (name, ours, oracle, tolerance) in [
                        ("adm2", ours.features.adm2, features[0], 1e-4),
                        ("motion2", ours.features.motion2, features[1], 1e-8),
                        ("vif0", ours.features.vif_scales[0], features[2], 1e-4),
                        ("vif1", ours.features.vif_scales[1], features[3], 1e-4),
                        ("vif2", ours.features.vif_scales[2], features[4], 1e-4),
                        ("vif3", ours.features.vif_scales[3], features[5], 1e-4),
                        ("score", ours.score, *oracle, 0.02),
                    ] {
                        assert!(
                            (ours - oracle).abs() <= tolerance,
                            "{variant:?}, {bit_depth} bit, distorted={distorted}, \
                             frame={index}, {name}: Rust {ours}, libvmaf {oracle}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn v0_stream_matches_batch_through_motion_lookahead() {
    for variant in [VmafV0Variant::StandardNeg, VmafV0Variant::FourK] {
        for bit_depth in [8, 10] {
            let reference: Vec<_> = (0..3).map(|i| frame(i, bit_depth, false)).collect();
            let distortion: Vec<_> = (0..3).map(|i| frame(i, bit_depth, true)).collect();
            let ref_frames: Vec<_> = reference.iter().map(yuv).collect();
            let dis_frames: Vec<_> = distortion.iter().map(yuv).collect();
            for count in [1, 2, 3] {
                let expected = score_v0_420(
                    &ref_frames[..count],
                    &dis_frames[..count],
                    WIDTH,
                    HEIGHT,
                    bit_depth as u8,
                    variant,
                )
                .unwrap();
                let mut stream =
                    VmafV0Stream::new(WIDTH, HEIGHT, bit_depth as u8, variant).unwrap();
                let mut actual = Vec::new();
                for i in 0..count {
                    actual.extend(stream.push(ref_frames[i], dis_frames[i]).unwrap());
                    assert!(actual.len() <= i);
                }
                actual.extend(stream.finish().unwrap());
                assert_eq!(actual.len(), count);
                for (i, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
                    for (name, got, want) in [
                        ("adm2", got.features.adm2, want.features.adm2),
                        ("motion2", got.features.motion2, want.features.motion2),
                        (
                            "vif0",
                            got.features.vif_scales[0],
                            want.features.vif_scales[0],
                        ),
                        (
                            "vif1",
                            got.features.vif_scales[1],
                            want.features.vif_scales[1],
                        ),
                        (
                            "vif2",
                            got.features.vif_scales[2],
                            want.features.vif_scales[2],
                        ),
                        (
                            "vif3",
                            got.features.vif_scales[3],
                            want.features.vif_scales[3],
                        ),
                        ("score", got.score, want.score),
                    ] {
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "{variant:?}, {bit_depth} bit, {count} frames, index {i}, {name}"
                        );
                    }
                }
            }
        }
    }
    assert!(
        VmafV0Stream::new(WIDTH, HEIGHT, 8, VmafV0Variant::Standard)
            .unwrap()
            .finish()
            .is_err()
    );
}

#[test]
fn v0_pooling_matches_v321() {
    for variant in [VmafV0Variant::StandardNeg, VmafV0Variant::FourK] {
        let session = Session::new(variant.built_in_name());
        for index in 0..3 {
            let mut reference = picture(&frame(index, 8, false), 8, WIDTH, HEIGHT);
            let mut distorted = picture(&frame(index, 8, true), 8, WIDTH, HEIGHT);
            assert_eq!(
                unsafe {
                    vmaf_read_pictures(
                        session.context,
                        &mut reference,
                        &mut distorted,
                        index as u32,
                    )
                },
                0
            );
        }
        assert_eq!(
            unsafe { vmaf_read_pictures(session.context, ptr::null_mut(), ptr::null_mut(), 0) },
            0
        );
        let scores: Vec<_> = (0..3)
            .map(|index| {
                let mut value = f64::NAN;
                assert_eq!(
                    unsafe {
                        vmaf_score_at_index(session.context, session.model, &mut value, index)
                    },
                    0
                );
                value
            })
            .collect();
        for (method, c_method) in [
            (PoolingMethod::Mean, VmafPoolingMethod_VMAF_POOL_METHOD_MEAN),
            (PoolingMethod::Min, VmafPoolingMethod_VMAF_POOL_METHOD_MIN),
            (PoolingMethod::Max, VmafPoolingMethod_VMAF_POOL_METHOD_MAX),
            (
                PoolingMethod::HarmonicMean,
                VmafPoolingMethod_VMAF_POOL_METHOD_HARMONIC_MEAN,
            ),
        ] {
            let mut oracle = f64::NAN;
            assert_eq!(
                unsafe {
                    vmaf_score_pooled(session.context, session.model, c_method, &mut oracle, 0, 2)
                },
                0
            );
            let actual = pool_v0_scores(&scores, method).unwrap();
            assert!(
                (actual - oracle).abs() <= 1e-8,
                "{variant:?}, {method:?}: Rust {actual}, libvmaf {oracle}"
            );
        }
    }
}

#[test]
fn v0_rejects_invalid_frames_without_advancing_stream() {
    let reference = frame(0, 8, false);
    let distorted = frame(0, 8, true);
    let ref_frame = yuv(&reference);
    let dis_frame = yuv(&distorted);
    let bad = Yuv420Frame {
        y: &distorted.planes[0],
        u: &[],
        v: &distorted.planes[2],
    };
    assert!(
        score_v0_420(
            &[ref_frame],
            &[bad],
            WIDTH,
            HEIGHT,
            8,
            VmafV0Variant::Standard,
        )
        .is_err()
    );
    let mut stream = VmafV0Stream::new(WIDTH, HEIGHT, 8, VmafV0Variant::Standard).unwrap();
    assert!(stream.push(ref_frame, bad).is_err());
    assert!(stream.push(ref_frame, dis_frame).unwrap().is_empty());
    let got = stream.finish().unwrap();
    let expected = score_v0_420(
        &[ref_frame],
        &[dis_frame],
        WIDTH,
        HEIGHT,
        8,
        VmafV0Variant::Standard,
    )
    .unwrap();
    assert_eq!(got[0].score.to_bits(), expected[0].score.to_bits());
    assert!(score_v0_420(&[], &[], WIDTH, HEIGHT, 8, VmafV0Variant::Standard).is_err());
}

#[test]
fn small_adm_frames_are_rejected_without_panicking() {
    for (width, height) in [(18, 18), (32, 64), (64, 32)] {
        let luma = vec![128u16; width * height];
        let chroma = vec![128u16; width * height / 4];
        let frame = Yuv420Frame {
            y: &luma,
            u: &chroma,
            v: &chroma,
        };
        assert!(
            adm3_v1_from_luma(&luma, &luma, width, height, 8, ModelVariant::Standard1080p).is_err()
        );
        assert!(
            score_v0_420(
                &[frame],
                &[frame],
                width,
                height,
                8,
                VmafV0Variant::Standard
            )
            .is_err()
        );
        assert!(VmafV0Stream::new(width, height, 8, VmafV0Variant::Standard).is_err());
        assert!(VmafV1Stream::new(width, height, 8, ModelVariant::Standard1080p).is_err());
    }
    let luma = vec![128u16; 34 * 34];
    let chroma = vec![128u16; 17 * 17];
    let frame = Yuv420Frame {
        y: &luma,
        u: &chroma,
        v: &chroma,
    };
    assert!(score_v0_420(&[frame], &[frame], 34, 34, 8, VmafV0Variant::Standard).is_ok());
}

#[cfg(feature = "parallel")]
#[test]
fn v0_parallel_preserves_serial_features_and_order() {
    use vmaf::VmafV0Scorer;
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let reference: Vec<_> = (0..3).map(|i| frame(i, 10, false)).collect();
    let distorted: Vec<_> = (0..3).map(|i| frame(i, 10, true)).collect();
    let ref_frames: Vec<_> = reference.iter().map(yuv).collect();
    let dis_frames: Vec<_> = distorted.iter().map(yuv).collect();
    let serial = score_v0_420(
        &ref_frames,
        &dis_frames,
        WIDTH,
        HEIGHT,
        10,
        VmafV0Variant::StandardNeg,
    )
    .unwrap();
    for threads in [1, 2, 4].into_iter().filter(|&n| n <= available) {
        let scorer = VmafV0Scorer::new(WIDTH, HEIGHT, 10, VmafV0Variant::StandardNeg)
            .unwrap()
            .with_threads(threads)
            .unwrap();
        for (want, got) in serial
            .iter()
            .zip(scorer.score(&ref_frames, &dis_frames).unwrap())
        {
            assert_eq!(want.score.to_bits(), got.score.to_bits());
            assert_eq!(want.features.adm2.to_bits(), got.features.adm2.to_bits());
            assert_eq!(
                want.features.motion2.to_bits(),
                got.features.motion2.to_bits()
            );
            for (a, b) in want.features.vif_scales.iter().zip(got.features.vif_scales) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }
    assert!(
        VmafV0Scorer::new(WIDTH, HEIGHT, 10, VmafV0Variant::StandardNeg)
            .unwrap()
            .with_threads(0)
            .is_err()
    );
}
