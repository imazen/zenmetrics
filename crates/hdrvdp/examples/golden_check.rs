//! Validate against official HDR-VDP-2.2.2 goldens generated under Octave.
//!
//! ```text
//! cargo run --release -p hdrvdp --example golden_check -- /path/to/goldens
//! ```
//!
//! The goldens directory is produced by `gen_goldens.m` (Octave +
//! hdrvdp-2.2.2): `goldens.tsv` plus raw little-endian f64 row-major
//! image/P_map binaries. For every case we compare the convention-free
//! visibility outputs (P_map, P_det, C_max) — the true port-parity
//! signal — and print the quality fields side by side.

#![forbid(unsafe_code)]

use hdrvdp::bands::decompose;
use hdrvdp::display::{ColorEncoding, to_nits};
use hdrvdp::masking::{self, diff_mask};
use hdrvdp::pathway::{surround_per_channel, visual_pathway};
use hdrvdp::photoreceptor::Photoreceptor;
use hdrvdp::pool::visibility;
use hdrvdp::spectral::{emission_spectra, lmsr_matrix};
use hdrvdp::{Params, hdrvdp};
use std::fs;
use std::path::{Path, PathBuf};

fn read_f64(path: &Path) -> Vec<f64> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(bytes.len() % 8, 0, "{path:?} not f64-aligned");
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

struct Row {
    case: String,
    w: usize,
    h: usize,
    ppd: f64,
    q: f64,
    p_det: f64,
    c_max: f64,
    qmos_disabled: f64,
}

/// Column-major (MATLAB) plane → row-major.
fn un_cm(v: &[f64], h: usize, w: usize) -> Vec<f64> {
    let mut out = vec![0.0; h * w];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = v[x * h + y];
        }
    }
    out
}

/// Isolate the reconstruction stage: run upstream's own `D_bands.pyr`
/// through Rust's `reconstruct` and compare with upstream's `S_map`.
fn diag_reconstruct(dir: &Path) {
    use hdrvdp::spyr::{Band, SteerablePyramid, reconstruct};
    let pind = read_f64(&dir.join("diag_pind.f64"));
    let nb = pind.len() / 2;
    let dims: Vec<(usize, usize)> = (0..nb)
        .map(|i| (pind[i] as usize, pind[nb + i] as usize))
        .collect();
    let pyr = read_f64(&dir.join("diag_dpyr.f64"));

    // Slice pyr into planes per pind, column-major → row-major.
    let mut off = 0;
    let planes: Vec<Vec<f64>> = dims
        .iter()
        .map(|&(h, w)| {
            let p = un_cm(&pyr[off..off + h * w], h, w);
            off += h * w;
            p
        })
        .collect();
    let (h, w) = dims[0];
    let band = |i: usize| Band {
        data: planes[i].iter().map(|v| *v as f32).collect(),
        width: dims[i].1,
        height: dims[i].0,
    };
    let sp = SteerablePyramid {
        high_pass: band(0),
        levels: (1..nb - 1)
            .step_by(4)
            .map(|s| [band(s), band(s + 1), band(s + 2), band(s + 3)])
            .collect(),
        low_pass: band(nb - 1),
        width: w,
        height: h,
    };
    let rec = reconstruct(&sp);
    let smap_up = read_f64(&dir.join("diag_smap.f64")); // col-major, POST-pool
    let smap_up = un_cm(&smap_up, h, w);
    // Undo spatial pooling: pre = post / (Σpost/max_post).
    let su: f64 = smap_up.iter().sum();
    let mu = smap_up.iter().fold(0.0f64, |a, b| a.max(*b));
    let k = su / (mu + 1e-12);
    let pre: Vec<f64> = smap_up.iter().map(|v| v / k).collect();
    let maxd = rec
        .data
        .iter()
        .zip(&pre)
        .map(|(a, b)| (f64::from(a.abs()) - b).abs())
        .fold(0.0, f64::max);
    let sum_r: f64 = rec.data.iter().map(|v| f64::from(v.abs())).sum();
    let sum_pre: f64 = pre.iter().sum();
    println!(
        "reconstruct(upstream D_bands): |rec| Σ={sum_r:.6e} upstream |S_map_pre| Σ={sum_pre:.6e} max|Δ|={maxd:.3e}"
    );
}

/// Run Rust's pipeline on the same case and compare d_bands + quality terms
/// against the upstream dumps.
fn diag_pipeline(dir: &Path) {
    let (w, h) = (128usize, 128usize);
    let test = read_f64(&dir.join("noise_128x128_p30.f64"));
    let refr = read_f64(&dir.join("ref_128x128.f64"));
    let par = Params::new(30.0);
    let channels = 1;

    let ref_nits = to_nits(&refr, w, h, ColorEncoding::Luminance).unwrap();
    let test_nits = to_nits(&test, w, h, ColorEncoding::Luminance).unwrap();
    let surround = surround_per_channel(&ref_nits, channels, par.surround_l);
    let lmsr = lmsr_matrix(&emission_spectra(
        ColorEncoding::Luminance.spectra(),
        channels,
    ));
    let pn = Photoreceptor::new(&par);
    let path_ref = visual_pathway(&ref_nits, w, h, &par, &pn, &lmsr, &surround);
    let path_test = visual_pathway(&test_nits, w, h, &par, &pn, &lmsr, &surround);
    let (bands_ref, pad) = decompose(&path_ref, &par, None);
    let (bands_test, _) = decompose(&path_test, &par, Some(pad));
    let l_adapt: Vec<f32> = path_ref
        .l_adapt
        .iter()
        .zip(&path_test.l_adapt)
        .map(|(a, b)| 0.5 * (a + b))
        .collect();
    let dm = diff_mask(&test_nits, &ref_nits, channels);
    let m = masking::run(&bands_test, &bands_ref, &l_adapt, &dm, &par);

    // Compare per-band |D| sums with upstream's pyr slices.
    let pind = read_f64(&dir.join("diag_pind.f64"));
    let nb = pind.len() / 2;
    let pyr = read_f64(&dir.join("diag_dpyr.f64"));
    let mut off = 0;
    println!("band: upstream dims / rust dims, upstream Σ|D| vs rust Σ|D|, max|Δplane|");
    let mut pi = 0usize;
    for b in 0..m.d_bands.count() {
        for o in 0..m.d_bands.orientations(b) {
            let (uh, uw) = (pind[pi] as usize, pind[nb + pi] as usize);
            let up = un_cm(&pyr[off..off + uh * uw], uh, uw);
            off += uh * uw;
            let rband = m.d_bands.band(b, o);
            let us: f64 = up.iter().map(|v| v.abs()).sum();
            let rs: f64 = rband.data.iter().map(|v| f64::from(v.abs())).sum();
            let md = if up.len() == rband.data.len() {
                up.iter()
                    .zip(&rband.data)
                    .map(|(a, b)| (a - f64::from(*b)).abs())
                    .fold(0.0, f64::max)
            } else {
                f64::NAN
            };
            println!(
                "  b{b}o{o}: {}x{} / {}x{}  {us:.6e} vs {rs:.6e}  maxΔ={md:.3e}",
                uh, uw, rband.height, rband.width
            );
            pi += 1;
        }
    }

    // Reconstruct official `res.Q = 100 − Σ(ln(msre+ε)−ln ε)·w_f` from Rust's
    // quality_terms — which already ARE the per-plane upstream terms.
    let mut q_raw = 0.0f64;
    for &t in &m.quality_terms {
        q_raw += t;
    }
    println!(
        "official res.Q reconstructed from rust quality_terms: {:.6} (golden noise p30 = 76.27267)",
        100.0 - q_raw
    );

    // And Rust's own S_map/c_map from these bands.
    let v = visibility(&m.d_bands, &par);
    let smap_up = un_cm(&read_f64(&dir.join("diag_smap.f64")), h, w);
    let md = v
        .c_map
        .iter()
        .zip(&smap_up)
        .map(|(a, b)| (f64::from(*a) - b).abs())
        .fold(0.0, f64::max);
    let sr: f64 = v.c_map.iter().map(|v| f64::from(*v)).sum();
    let su: f64 = smap_up.iter().sum();
    println!(
        "rust visibility on rust d_bands: Σc_map={sr:.6e} (pooled={:.6e}) vs upstream Σ={su:.6e} maxΔ={md:.3e}",
        v.c_max
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(
        args.iter()
            .skip(1)
            .find(|a| !a.starts_with('-'))
            .map(|s| s.as_str())
            .unwrap_or("/home/lilith/tmp/hdrvdp-ref/goldens"),
    );
    if args.iter().any(|a| a == "--diag") {
        diag_reconstruct(&dir);
        diag_pipeline(&dir);
        return;
    }
    let tsv = fs::read_to_string(dir.join("goldens.tsv")).expect("goldens.tsv");
    let mut rows: Vec<Row> = Vec::new();
    for line in tsv.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 9 {
            continue;
        }
        rows.push(Row {
            case: f[0].into(),
            w: f[2].parse().unwrap(),
            h: f[3].parse().unwrap(),
            ppd: f[4].parse().unwrap(),
            q: f[5].parse().unwrap(),
            p_det: f[6].parse().unwrap(),
            c_max: f[7].parse().unwrap(),
            qmos_disabled: f[8].parse().unwrap(),
        });
    }

    println!(
        "{:<9} {:>3}x{:<3} {:>3} | {:>10} {:>10} | {:>10} {:>10} | {:>9} {:>9} | {:>9}",
        "case",
        "w",
        "h",
        "ppd",
        "Q_oct",
        "Q_rust",
        "Pdet_oct",
        "Pdet_rs",
        "Cmax_oct",
        "Cmax_rs",
        "Pmap_max"
    );

    let mut worst_pdet = 0.0f64;
    let mut worst_cmax_rel = 0.0f64;
    let mut worst_pmap = 0.0f64;
    let mut worst_q = 0.0f64;

    for r in &rows {
        let ppd_i = r.ppd as usize;
        let test = read_f64(&dir.join(format!("{}_{}x{}_p{ppd_i}.f64", r.case, r.h, r.w)));
        let refr = read_f64(&dir.join(format!("ref_{}x{}.f64", r.h, r.w)));
        let pmap_ref = read_f64(&dir.join(format!("{}_{}x{}_p{ppd_i}_pmap.f64", r.case, r.h, r.w)));
        assert_eq!(test.len(), r.w * r.h, "{} size", r.case);

        let par = Params {
            pix_per_deg: r.ppd,
            ..Default::default()
        };
        let res = hdrvdp(&test, &refr, r.w, r.h, ColorEncoding::Luminance, &par)
            .unwrap_or_else(|e| panic!("{}: {e:?}", r.case));

        let pmap_max = res
            .p_map
            .iter()
            .zip(&pmap_ref)
            .map(|(a, b)| (f64::from(*a) - b).abs())
            .fold(0.0, f64::max);
        let cmax_rel = (res.c_max - r.c_max).abs() / r.c_max.max(1e-30);
        worst_pdet = worst_pdet.max((res.p_det - r.p_det).abs());
        worst_cmax_rel = worst_cmax_rel.max(cmax_rel);
        worst_pmap = worst_pmap.max(pmap_max);
        worst_q = worst_q.max((res.q - r.q).abs());
        let qmos_d = (res.q_mos - r.qmos_disabled).abs();
        if qmos_d > 1e-5 {
            println!(
                "    ^ removed-Q_MOS Δ = {qmos_d:.3e} (rust {} vs upstream-disabled {})",
                res.q_mos, r.qmos_disabled
            );
        }

        println!(
            "{:<9} {:>3}x{:<3} {:>3} | {:>10.5} {:>10.5} | {:>10.6} {:>10.6} | {:>9.4} {:>9.4} | {:>9.2e}",
            r.case, r.w, r.h, ppd_i, r.q, res.q, r.p_det, res.p_det, r.c_max, res.c_max, pmap_max
        );
    }

    println!("\nworst |ΔP_det|        = {worst_pdet:.3e}");
    println!("worst rel |ΔC_max|  = {worst_cmax_rel:.3e}");
    println!("worst |ΔP_map|      = {worst_pmap:.3e}");
    println!("worst |Δres.Q|      = {worst_q:.3e}");
}
