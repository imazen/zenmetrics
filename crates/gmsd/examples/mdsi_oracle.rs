//! Differential-oracle driver. All files are raw RGB8 / little-endian f64.
//! Input TSV (header): id, width, height, ref_rgb, dist_rgb, each tab-separated.
//! Calls the public scorer, then includes the *same* private implementation
//! to export CS/GCS without adding map access to the public API.
#![allow(clippy::too_many_arguments)]
extern crate alloc;
#[macro_use]
#[path = "../src/chroma_gradient.rs"]
mod chroma_gradient;
#[path = "../src/mdsi.rs"]
#[allow(dead_code)]
mod mdsi;
#[path = "../src/ms_gmsd.rs"]
#[allow(dead_code)]
mod ms_gmsd;
use gmsd::{Error, Result};
const BAND_ROWS: usize = 64;
mod kernel {
    pub(crate) fn padded_pitch(w: usize) -> usize {
        w + 10
    }
}
fn sqrt_f64(v: f64) -> f64 {
    v.sqrt()
}
fn check_rgb8(rgb: &[u8], w: usize, h: usize, stride: usize) -> Result<()> {
    // Delegate ingress validation to the public crate, outside measured work.
    let mut gray = vec![0.0; w * h];
    gmsd::rgb8_to_gray(rgb, w, h, stride, &mut gray)
}
fn write_map(path: &std::path::Path, map: &[f64]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for value in map {
        f.write_all(&value.to_le_bytes())?;
    }
    f.flush()
}
fn verify_tiers(s: &mdsi::Source<'_>, gcs: &[f64], cs: &[f64]) -> String {
    fn same_bits(a: &[f64], b: &[f64], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}");
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{what}: sample {i}");
        }
    }
    let params = mdsi::Params::default();
    let oh = s.height();
    let row_len = s.row_len();
    let mut checked = Vec::new();
    macro_rules! check {
        ($name:literal, $prep:path, $maps:path, $token:expr) => {{
            let mut data = vec![0.0f64; (oh + 2) * row_len];
            for (i, chunk) in data[row_len..(oh + 1) * row_len]
                .chunks_mut(BAND_ROWS * row_len)
                .enumerate()
            {
                $prep($token, s, i * BAND_ROWS, chunk);
            }
            let pl = mdsi::Planes::from_parts(data, s);
            let (mut q, mut c) = (vec![0.0; gcs.len()], vec![0.0; cs.len()]);
            for (i, (a, b)) in q
                .chunks_mut(BAND_ROWS * pl.width)
                .zip(c.chunks_mut(BAND_ROWS * pl.width))
                .enumerate()
            {
                $maps($token, &pl, i * BAND_ROWS, &params, a, b);
            }
            same_bits(&q, gcs, concat!($name, " gcs"));
            same_bits(&c, cs, concat!($name, " cs"));
            assert_eq!(mdsi::pool(&q).to_bits(), mdsi::pool(gcs).to_bits());
            checked.push($name.to_string());
        }};
    }
    check!(
        "scalar",
        mdsi::prepare_band_scalar,
        mdsi::maps_band_scalar,
        archmage::ScalarToken
    );
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::X64V3Token::summon() {
            check!("v3", mdsi::prepare_band_v3, mdsi::maps_band_v3, t);
        }
        #[cfg(feature = "avx512")]
        if let Some(t) = archmage::X64V4Token::summon() {
            check!("v4", mdsi::prepare_band_v4, mdsi::maps_band_v4, t);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::NeonToken::summon() {
            check!("neon", mdsi::prepare_band_neon, mdsi::maps_band_neon, t);
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::Wasm128Token::summon() {
            check!(
                "wasm128",
                mdsi::prepare_band_wasm128,
                mdsi::maps_band_wasm128,
                t
            );
        }
    }
    #[cfg(feature = "parallel")]
    for threads in [1, 8] {
        let (q, c) = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| mdsi::maps(&mdsi::prepare(s), &params));
        same_bits(&q, gcs, "threads gcs");
        same_bits(&c, cs, "threads cs");
        checked.push(format!("threads{threads}"));
    }
    checked.join(",")
}

fn verify_ms_tiers(p: &ms_gmsd::Planes, map: &[f64]) -> String {
    let mut checked = vec![];
    let bits = |a: &[f64]| a.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    macro_rules! check {
        ($name:literal,$kernel:path,$token:expr) => {{
            let mut q = vec![0.0; map.len()];
            $kernel($token, p, 0, 170.0, &mut q);
            assert_eq!(bits(&q), bits(map), "MS-GMSD {}", $name);
            checked.push($name.to_string());
        }};
    }
    check!("scalar", ms_gmsd::ms_band_scalar, archmage::ScalarToken);
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::X64V3Token::summon() {
            check!("v3", ms_gmsd::ms_band_v3, t);
        }
        #[cfg(feature = "avx512")]
        if let Some(t) = archmage::X64V4Token::summon() {
            check!("v4", ms_gmsd::ms_band_v4, t);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::NeonToken::summon() {
            check!("neon", ms_gmsd::ms_band_neon, t);
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::Wasm128Token::summon() {
            check!("wasm128", ms_gmsd::ms_band_wasm128, t);
        }
    }
    #[cfg(feature = "parallel")]
    for threads in [1, 8] {
        let q = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| ms_gmsd::map(p, 170.0));
        assert_eq!(bits(&q), bits(map), "MS-GMSD threads {threads}");
        checked.push(format!("threads{threads}"));
    }
    checked.join(",")
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut args = std::env::args().skip(1);
    let input = std::fs::read_to_string(args.next().ok_or("input TSV required")?)?;
    let out = std::path::PathBuf::from(args.next().ok_or("output directory required")?);
    std::fs::create_dir_all(&out)?;
    let mut table = std::fs::File::create(out.join("rust.tsv"))?;
    let mut ms_table = std::fs::File::create(out.join("ms_rust.tsv"))?;
    writeln!(
        ms_table,
        "id\tms_gmsd\tms_gmsdc\twrong_constant\tbitwise_checks"
    )?;
    writeln!(
        table,
        "id\twidth\theight\tmdsi\twrong_constant\tstrided_mdsi\tbitwise_checks"
    )?;
    for line in input.lines().skip(1) {
        let fields: Vec<_> = line.split('\t').collect();
        let (id, w, h) = (
            fields[0],
            fields[1].parse::<usize>()?,
            fields[2].parse::<usize>()?,
        );
        let r = std::fs::read(fields[3])?;
        let d = std::fs::read(fields[4])?;
        assert_eq!(r.len(), w * h * 3);
        assert_eq!(d.len(), w * h * 3);
        let score = gmsd::mdsi_rgb8(&r, &d, w, h, w * 3)?;
        let src = mdsi::source(&r, &d, w, h, w * 3);
        let p = mdsi::prepare(&src);
        let (gcs, cs) = mdsi::maps(&p, &mdsi::Params::default());
        let checks = verify_tiers(&src, &gcs, &cs);
        assert_eq!(score.to_bits(), mdsi::pool(&gcs).to_bits());
        // The straight-line equations must agree bit for bit with the fast path.
        let (slow, slow_map) =
            mdsi::reference::score_and_map(&r, &d, w, h, w * 3, &mdsi::Params::default())?;
        assert_eq!(score.to_bits(), slow.to_bits());
        assert!(
            gcs.iter()
                .zip(&slow_map)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        let wrong_params = mdsi::Params {
            c3: 55.0,
            ..mdsi::Params::default()
        };
        let (wrong, _) = mdsi::maps(&p, &wrong_params);
        let stride = w * 3 + 7;
        let padded = |data: &[u8]| {
            let mut result = vec![251; stride * h];
            for y in 0..h {
                result[y * stride..y * stride + 3 * w]
                    .copy_from_slice(&data[y * 3 * w..(y + 1) * 3 * w]);
            }
            result
        };
        let strided = gmsd::mdsi_rgb8(&padded(&r), &padded(&d), w, h, stride)?;
        assert_eq!(score.to_bits(), strided.to_bits());
        write_map(&out.join(format!("{id}.gcs.f64")), &gcs)?;
        write_map(&out.join(format!("{id}.cs.f64")), &cs)?;
        writeln!(
            table,
            "{id}\t{}\t{}\t{score:.17e}\t{:.17e}\t{strided:.17e}\t{checks}",
            p.width,
            p.height,
            mdsi::pool(&wrong)
        )?;
        eprintln!("scored {id} {w}x{h} mdsi={score:.17e}");

        let mut p = ms_gmsd::prepare(&r, &d, w, h, w * 3);
        let mut variances = [0.0; 4];
        let mut bad_variances = [0.0; 4];
        let mut checks = String::new();
        for (scale, (variance, bad_variance)) in
            variances.iter_mut().zip(&mut bad_variances).enumerate()
        {
            let q = ms_gmsd::map(&p, 170.0);
            checks = verify_ms_tiers(&p, &q);
            *variance = ms_gmsd::variance(&q);
            *bad_variance = ms_gmsd::variance(&ms_gmsd::map(&p, 17.0));
            write_map(&out.join(format!("{id}.ms_s{scale}.f64")), &q)?;
            if scale != 3 {
                p = p.half();
            }
        }
        let (ms, colour) = ms_gmsd::finish(variances, &p);
        let (_, wrong) = ms_gmsd::finish(bad_variances, &p);
        assert_eq!(
            ms.to_bits(),
            gmsd::ms_gmsd_rgb8(&r, &d, w, h, w * 3)?.to_bits()
        );
        assert_eq!(
            colour.to_bits(),
            gmsd::ms_gmsdc_rgb8(&r, &d, w, h, w * 3)?.to_bits()
        );
        assert_eq!(
            colour.to_bits(),
            gmsd::ms_gmsdc_rgb8(&padded(&r), &padded(&d), w, h, stride)?.to_bits()
        );
        writeln!(
            ms_table,
            "{id}\t{ms:.17e}\t{colour:.17e}\t{wrong:.17e}\t{checks}"
        )?;
    }
    Ok(())
}
