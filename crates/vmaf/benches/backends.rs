use std::ffi::CString;
use std::hint::black_box;
use std::mem::MaybeUninit;
use std::ptr;
use std::time::Instant;
use vmaf::{
    ModelVariant, VmafFeatures, VmafModel, VmafV0Features, VmafV0Model, VmafV0Variant, Yuv420Frame,
    adm2_v0_from_luma, adm3_v1_from_luma, cambi_v1_from_luma, motion2_v0_from_luma,
    motion3_from_luma, score_v0_420, score_v1_420, speed_v1_chroma_420, vif_v0_from_luma,
};
use vmaf_head_sys::*;

const WIDTH: usize = 1280;
const HEIGHT: usize = 720;
const FRAMES: usize = 2;
const ROUNDS: usize = 3;

struct Frame {
    planes: [Vec<u16>; 3],
}

fn fixture(index: usize, distorted: bool) -> Frame {
    let mut planes = [
        vec![0; WIDTH * HEIGHT],
        vec![0; WIDTH * HEIGHT / 4],
        vec![0; WIDTH * HEIGHT / 4],
    ];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let value = 16 + ((x * 3 + y * 5 + index * 7 + (x / 17) * 9) % 220);
            planes[0][y * WIDTH + x] = if distorted && x < WIDTH / 2 {
                (value / 12 * 12) as u16
            } else {
                value as u16
            };
        }
    }
    for y in 0..HEIGHT / 2 {
        for x in 0..WIDTH / 2 {
            let slot = y * WIDTH / 2 + x;
            let u = 16 + ((x * 7 + y * 3 + (x / 9) * (y / 9) * 13 + index * 11) % 224);
            let v = 16 + ((x * 2 + y * 9 + (x / 13) * (y / 7) * 5 + index * 13) % 224);
            planes[1][slot] = if distorted {
                (u + (x + y + index) % 17).min(240) as u16
            } else {
                u as u16
            };
            planes[2][slot] = if distorted {
                v.saturating_sub((x + 2 * y + index) % 19) as u16
            } else {
                v as u16
            };
        }
    }
    Frame { planes }
}

fn yuv(frame: &Frame) -> Yuv420Frame<'_> {
    Yuv420Frame {
        y: &frame.planes[0],
        u: &frame.planes[1],
        v: &frame.planes[2],
    }
}

fn picture(frame: &Frame) -> VmafPicture {
    let mut picture = MaybeUninit::<VmafPicture>::uninit();
    assert_eq!(
        unsafe {
            vmaf_picture_alloc(
                picture.as_mut_ptr(),
                VmafPixelFormat_VMAF_PIX_FMT_YUV420P,
                8,
                WIDTH as u32,
                HEIGHT as u32,
            )
        },
        0
    );
    let picture = unsafe { picture.assume_init() };
    for (plane, data) in frame.planes.iter().enumerate() {
        let (width, height) = if plane == 0 {
            (WIDTH, HEIGHT)
        } else {
            (WIDTH / 2, HEIGHT / 2)
        };
        for y in 0..height {
            for x in 0..width {
                unsafe {
                    *picture.data[plane]
                        .add(y * picture.stride[plane] as usize + x)
                        .cast::<u8>() = data[y * width + x] as u8;
                }
            }
        }
    }
    picture
}

fn score_c_version(
    reference: &[Frame],
    distorted: &[Frame],
    cpumask: u64,
    version: &str,
) -> Vec<f64> {
    let cfg = VmafConfiguration {
        log_level: VmafLogLevel_VMAF_LOG_LEVEL_NONE,
        n_threads: 1,
        n_subsample: 1,
        cpumask,
        gpumask: 0,
    };
    let mut ctx = ptr::null_mut();
    assert_eq!(unsafe { vmaf_init(&mut ctx, cfg) }, 0);
    let name = CString::new("vmaf_bench").unwrap();
    let version = CString::new(version).unwrap();
    let mut model_cfg = VmafModelConfig {
        name: name.as_ptr(),
        flags: VmafModelFlags_VMAF_MODEL_FLAGS_DEFAULT as u64,
    };
    let mut model = ptr::null_mut();
    assert_eq!(
        unsafe { vmaf_model_load(&mut model, &mut model_cfg, version.as_ptr()) },
        0
    );
    assert_eq!(unsafe { vmaf_use_features_from_model(ctx, model) }, 0);
    for (i, (reference, distorted)) in reference.iter().zip(distorted).enumerate() {
        let mut reference = picture(reference);
        let mut distorted = picture(distorted);
        assert_eq!(
            unsafe { vmaf_read_pictures(ctx, &mut reference, &mut distorted, i as u32) },
            0
        );
    }
    assert_eq!(
        unsafe { vmaf_read_pictures(ctx, ptr::null_mut(), ptr::null_mut(), 0) },
        0
    );
    let mut result = Vec::with_capacity(reference.len());
    for i in 0..reference.len() {
        let mut score = f64::NAN;
        assert_eq!(
            unsafe { vmaf_score_at_index(ctx, model, &mut score, i as u32) },
            0
        );
        result.push(score);
    }
    unsafe {
        vmaf_model_destroy(model);
        assert_eq!(vmaf_close(ctx), 0);
    }
    result
}

fn score_c(reference: &[Frame], distorted: &[Frame], cpumask: u64) -> Vec<f64> {
    score_c_version(
        reference,
        distorted,
        cpumask,
        ModelVariant::Phone.built_in_name(),
    )
}

fn score_rust(reference: &[Frame], distorted: &[Frame]) -> Vec<f64> {
    score_v1_420(
        &reference.iter().map(yuv).collect::<Vec<_>>(),
        &distorted.iter().map(yuv).collect::<Vec<_>>(),
        WIDTH,
        HEIGHT,
        8,
        ModelVariant::Phone,
    )
    .unwrap()
    .into_iter()
    .map(|frame| frame.score)
    .collect()
}

fn score_rust_v0(reference: &[Frame], distorted: &[Frame], variant: VmafV0Variant) -> Vec<f64> {
    score_v0_420(
        &reference.iter().map(yuv).collect::<Vec<_>>(),
        &distorted.iter().map(yuv).collect::<Vec<_>>(),
        WIDTH,
        HEIGHT,
        8,
        variant,
    )
    .unwrap()
    .into_iter()
    .map(|frame| frame.score)
    .collect()
}

#[derive(Clone, Copy)]
enum Backend {
    RustScalar,
    LibvmafAuto,
    LibvmafScalar,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::RustScalar => {
                if cfg!(feature = "simd") {
                    "rust_simd"
                } else {
                    "rust_scalar"
                }
            }
            Self::LibvmafAuto => "libvmaf_cpu_auto",
            Self::LibvmafScalar => "libvmaf_scalar",
        }
    }

    fn score(self, reference: &[Frame], distorted: &[Frame]) -> Vec<f64> {
        match self {
            Self::RustScalar => score_rust(reference, distorted),
            Self::LibvmafAuto => score_c(reference, distorted, 0),
            Self::LibvmafScalar => score_c(reference, distorted, u64::MAX),
        }
    }

    fn score_v0(
        self,
        reference: &[Frame],
        distorted: &[Frame],
        variant: VmafV0Variant,
    ) -> Vec<f64> {
        match self {
            Self::RustScalar => score_rust_v0(reference, distorted, variant),
            Self::LibvmafAuto => score_c_version(reference, distorted, 0, variant.built_in_name()),
            Self::LibvmafScalar => {
                score_c_version(reference, distorted, u64::MAX, variant.built_in_name())
            }
        }
    }
}

fn bench_v0(reference: &[Frame], distorted: &[Frame], variant: VmafV0Variant) {
    let backends = [
        Backend::RustScalar,
        Backend::LibvmafAuto,
        Backend::LibvmafScalar,
    ];
    let baseline = Backend::LibvmafAuto.score_v0(reference, distorted, variant);
    for backend in backends {
        let warmup = backend.score_v0(reference, distorted, variant);
        for (&expected, &actual) in baseline.iter().zip(&warmup) {
            assert!(
                (actual - expected).abs() <= 0.02,
                "{} {} vs libvmaf auto: {actual} vs {expected}",
                variant.built_in_name(),
                backend.name()
            );
        }
        black_box(warmup);
    }
    let mut times = [Vec::new(), Vec::new(), Vec::new()];
    println!("model\tbackend\tmedian_ms_per_frame\tmin_ms_per_frame\tframe_count\trounds");
    for round in 0..ROUNDS {
        for offset in 0..backends.len() {
            let idx = (round + offset) % backends.len();
            let start = Instant::now();
            let scores = backends[idx].score_v0(reference, distorted, variant);
            let elapsed = start.elapsed();
            for (&expected, &actual) in baseline.iter().zip(&scores) {
                assert!(
                    (actual - expected).abs() <= 0.02,
                    "{} {} vs libvmaf auto: {actual} vs {expected}",
                    variant.built_in_name(),
                    backends[idx].name()
                );
            }
            black_box(scores);
            times[idx].push(elapsed.as_secs_f64() * 1000.0 / FRAMES as f64);
        }
    }
    for (backend, measured) in backends.into_iter().zip(times.iter_mut()) {
        measured.sort_by(f64::total_cmp);
        println!(
            "{}\t{}\t{:.3}\t{:.3}\t{}\t{}",
            variant.built_in_name(),
            backend.name(),
            measured[ROUNDS / 2],
            measured[0],
            FRAMES,
            ROUNDS
        );
    }
}

fn v0_stage_timings(reference: &[Frame], distorted: &[Frame], variant: VmafV0Variant) -> [f64; 4] {
    let start = Instant::now();
    let motion = motion2_v0_from_luma(
        &reference
            .iter()
            .map(|frame| frame.planes[0].as_slice())
            .collect::<Vec<_>>(),
        WIDTH,
        HEIGHT,
        8,
    )
    .unwrap();
    let motion_time = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let adm: Vec<_> = reference
        .iter()
        .zip(distorted)
        .map(|(reference, distorted)| {
            adm2_v0_from_luma(
                &reference.planes[0],
                &distorted.planes[0],
                WIDTH,
                HEIGHT,
                8,
                variant,
            )
            .unwrap()
        })
        .collect();
    let adm_time = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let vif: Vec<_> = reference
        .iter()
        .zip(distorted)
        .map(|(reference, distorted)| {
            vif_v0_from_luma(
                &reference.planes[0],
                &distorted.planes[0],
                WIDTH,
                HEIGHT,
                8,
                variant,
            )
            .unwrap()
        })
        .collect();
    let vif_time = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let model = VmafV0Model::new(variant).unwrap();
    let scores: Vec<_> = (0..reference.len())
        .map(|i| {
            model
                .predict(VmafV0Features {
                    adm2: adm[i],
                    motion2: motion[i],
                    vif_scales: vif[i],
                })
                .unwrap()
        })
        .collect();
    black_box(scores);
    let fusion_time = start.elapsed().as_secs_f64();
    [motion_time, adm_time, vif_time, fusion_time]
}

fn bench_v0_stages(reference: &[Frame], distorted: &[Frame], variant: VmafV0Variant) {
    let _ = v0_stage_timings(reference, distorted, variant);
    let mut times: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    for _ in 0..ROUNDS {
        let stages = v0_stage_timings(reference, distorted, variant);
        for (samples, elapsed) in times.iter_mut().zip(stages) {
            samples.push(elapsed * 1000.0 / FRAMES as f64);
        }
    }
    println!("model\tstage\tmedian_ms_per_frame\tmin_ms_per_frame");
    for (stage, measurements) in ["motion2", "adm2", "vif4", "fusion"]
        .into_iter()
        .zip(times.iter_mut())
    {
        measurements.sort_by(f64::total_cmp);
        println!(
            "{}\t{stage}\t{:.3}\t{:.3}",
            variant.built_in_name(),
            measurements[ROUNDS / 2],
            measurements[0]
        );
    }
}

#[cfg(feature = "parallel")]
fn bench_v0_parallel(reference: &[Frame], distorted: &[Frame], variant: VmafV0Variant) {
    use vmaf::VmafV0Scorer;

    let ref_frames: Vec<_> = reference.iter().map(yuv).collect();
    let dis_frames: Vec<_> = distorted.iter().map(yuv).collect();
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let baseline = VmafV0Scorer::new(WIDTH, HEIGHT, 8, variant)
        .unwrap()
        .score(&ref_frames, &dis_frames)
        .unwrap();
    let runners: Vec<_> = [1, 2, 4]
        .into_iter()
        .filter(|&threads| threads <= available)
        .map(|threads| {
            (
                threads,
                VmafV0Scorer::new(WIDTH, HEIGHT, 8, variant)
                    .unwrap()
                    .with_threads(threads)
                    .unwrap(),
            )
        })
        .collect();
    for (_, scorer) in &runners {
        let warm = scorer.score(&ref_frames, &dis_frames).unwrap();
        for (actual, expected) in warm.iter().zip(&baseline) {
            assert_eq!(actual.score.to_bits(), expected.score.to_bits());
        }
        black_box(warm);
    }
    let mut times = vec![Vec::new(); runners.len()];
    println!("model\tthreads\tmedian_ms_per_frame\tmin_ms_per_frame\tframe_count\trounds");
    for round in 0..ROUNDS {
        for offset in 0..runners.len() {
            let idx = (round + offset) % runners.len();
            let start = Instant::now();
            let scores = runners[idx].1.score(&ref_frames, &dis_frames).unwrap();
            let elapsed = start.elapsed();
            for (actual, expected) in scores.iter().zip(&baseline) {
                assert_eq!(actual.score.to_bits(), expected.score.to_bits());
            }
            black_box(scores);
            times[idx].push(elapsed.as_secs_f64() * 1000.0 / FRAMES as f64);
        }
    }
    for ((threads, _), measurements) in runners.iter().zip(&mut times) {
        measurements.sort_by(f64::total_cmp);
        println!(
            "{}\t{threads}\t{:.3}\t{:.3}\t{}\t{}",
            variant.built_in_name(),
            measurements[ROUNDS / 2],
            measurements[0],
            FRAMES,
            ROUNDS
        );
    }
}

fn rust_stage_timings(reference: &[Frame], distorted: &[Frame]) -> [f64; 5] {
    let start = Instant::now();
    let motion = motion3_from_luma(
        &reference
            .iter()
            .map(|frame| frame.planes[0].as_slice())
            .collect::<Vec<_>>(),
        WIDTH,
        HEIGHT,
        8,
        false,
    )
    .unwrap();
    let motion_time = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let cambi: Vec<_> = distorted
        .iter()
        .map(|frame| cambi_v1_from_luma(&frame.planes[0], WIDTH, HEIGHT, 8).unwrap())
        .collect();
    let cambi_time = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let speed: Vec<_> = reference
        .iter()
        .zip(distorted)
        .map(|(reference, distorted)| {
            speed_v1_chroma_420(
                &reference.planes[1],
                &reference.planes[2],
                &distorted.planes[1],
                &distorted.planes[2],
                WIDTH,
                HEIGHT,
                8,
                ModelVariant::Phone,
            )
            .unwrap()
        })
        .collect();
    let speed_time = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let adm: Vec<_> = reference
        .iter()
        .zip(distorted)
        .map(|(reference, distorted)| {
            adm3_v1_from_luma(
                &reference.planes[0],
                &distorted.planes[0],
                WIDTH,
                HEIGHT,
                8,
                ModelVariant::Phone,
            )
            .unwrap()
        })
        .collect();
    let adm_time = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let model = VmafModel::new(ModelVariant::Phone).unwrap();
    let scores: Vec<_> = (0..reference.len())
        .map(|i| {
            model
                .predict(VmafFeatures {
                    cambi: cambi[i],
                    speed_chroma_uv: speed[i],
                    adm3: adm[i],
                    motion3: motion[i],
                })
                .unwrap()
        })
        .collect();
    black_box(scores);
    let fusion_time = start.elapsed().as_secs_f64();
    [motion_time, cambi_time, speed_time, adm_time, fusion_time]
}

#[cfg(feature = "parallel")]
fn bench_parallel(reference: &[Frame], distorted: &[Frame]) {
    use vmaf::VmafV1Scorer;

    let ref_frames: Vec<_> = reference.iter().map(yuv).collect();
    let dis_frames: Vec<_> = distorted.iter().map(yuv).collect();
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let baseline = VmafV1Scorer::new(WIDTH, HEIGHT, 8, ModelVariant::Phone)
        .unwrap()
        .score(&ref_frames, &dis_frames)
        .unwrap();
    let runners: Vec<_> = [1, 2, 4]
        .into_iter()
        .filter(|&threads| threads <= available)
        .map(|threads| {
            (
                threads,
                VmafV1Scorer::new(WIDTH, HEIGHT, 8, ModelVariant::Phone)
                    .unwrap()
                    .with_threads(threads)
                    .unwrap(),
            )
        })
        .collect();
    for (_, scorer) in &runners {
        let warm = scorer.score(&ref_frames, &dis_frames).unwrap();
        for (actual, expected) in warm.iter().zip(&baseline) {
            assert_eq!(actual.score.to_bits(), expected.score.to_bits());
        }
        black_box(warm);
    }
    let mut times = vec![Vec::new(); runners.len()];
    println!("threads\tmedian_ms_per_frame\tmin_ms_per_frame\tframe_count\trounds");
    for round in 0..ROUNDS {
        for offset in 0..runners.len() {
            let idx = (round + offset) % runners.len();
            let start = Instant::now();
            let scores = runners[idx].1.score(&ref_frames, &dis_frames).unwrap();
            let elapsed = start.elapsed();
            for (actual, expected) in scores.iter().zip(&baseline) {
                assert_eq!(actual.score.to_bits(), expected.score.to_bits());
            }
            black_box(scores);
            times[idx].push(elapsed.as_secs_f64() * 1000.0 / FRAMES as f64);
        }
    }
    for ((threads, _), measurements) in runners.iter().zip(&mut times) {
        measurements.sort_by(f64::total_cmp);
        println!(
            "{threads}\t{:.3}\t{:.3}\t{}\t{}",
            measurements[ROUNDS / 2],
            measurements[0],
            FRAMES,
            ROUNDS
        );
    }
}

fn main() {
    let reference: Vec<_> = (0..FRAMES).map(|i| fixture(i, false)).collect();
    let distorted: Vec<_> = (0..FRAMES).map(|i| fixture(i, true)).collect();
    let args: Vec<_> = std::env::args().collect();
    let v0 = if args.iter().any(|arg| arg == "--v0-neg") {
        Some(VmafV0Variant::StandardNeg)
    } else if args.iter().any(|arg| arg == "--v0") {
        Some(VmafV0Variant::Standard)
    } else {
        None
    };
    if let Some(variant) = v0 {
        #[cfg(feature = "parallel")]
        if args.iter().any(|arg| arg == "--parallel") {
            bench_v0_parallel(&reference, &distorted, variant);
            return;
        }
        if args.iter().any(|arg| arg == "--stages") {
            bench_v0_stages(&reference, &distorted, variant);
        } else {
            bench_v0(&reference, &distorted, variant);
        }
        return;
    }
    #[cfg(feature = "parallel")]
    if std::env::args().any(|arg| arg == "--parallel") {
        bench_parallel(&reference, &distorted);
        return;
    }
    if std::env::args().any(|arg| arg == "--stages") {
        let _ = rust_stage_timings(&reference, &distorted);
        let mut times: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::new());
        for _ in 0..ROUNDS {
            let stages = rust_stage_timings(&reference, &distorted);
            for (samples, elapsed) in times.iter_mut().zip(stages) {
                samples.push(elapsed * 1000.0 / FRAMES as f64);
            }
        }
        println!("stage\tmedian_ms_per_frame\tmin_ms_per_frame");
        for (stage, measurements) in ["motion", "cambi", "speed", "adm3", "fusion"]
            .into_iter()
            .zip(times.iter_mut())
        {
            measurements.sort_by(f64::total_cmp);
            println!(
                "{stage}\t{:.3}\t{:.3}",
                measurements[ROUNDS / 2],
                measurements[0]
            );
        }
        return;
    }
    let backends = [
        Backend::RustScalar,
        Backend::LibvmafAuto,
        Backend::LibvmafScalar,
    ];
    let baseline = Backend::LibvmafAuto.score(&reference, &distorted);
    for backend in backends {
        let warmup = backend.score(&reference, &distorted);
        for (&expected, &actual) in baseline.iter().zip(&warmup) {
            assert!(
                (actual - expected).abs() <= 0.02,
                "{} vs libvmaf auto: {actual} vs {expected}",
                backend.name()
            );
        }
        black_box(warmup);
    }
    let mut times = [Vec::new(), Vec::new(), Vec::new()];
    println!("backend\tmedian_ms_per_frame\tmin_ms_per_frame\tframe_count\trounds");
    for round in 0..ROUNDS {
        for offset in 0..backends.len() {
            let idx = (round + offset) % backends.len();
            let start = Instant::now();
            let scores = backends[idx].score(&reference, &distorted);
            let elapsed = start.elapsed();
            for (&expected, &actual) in baseline.iter().zip(&scores) {
                assert!(
                    (actual - expected).abs() <= 0.02,
                    "{} vs libvmaf auto: {actual} vs {expected}",
                    backends[idx].name()
                );
            }
            black_box(scores);
            times[idx].push(elapsed.as_secs_f64() * 1000.0 / FRAMES as f64);
        }
    }
    for (backend, measured) in backends.into_iter().zip(times.iter_mut()) {
        measured.sort_by(f64::total_cmp);
        println!(
            "{}\t{:.3}\t{:.3}\t{}\t{}",
            backend.name(),
            measured[ROUNDS / 2],
            measured[0],
            FRAMES,
            ROUNDS
        );
    }
}
