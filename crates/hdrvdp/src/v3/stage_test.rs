//! Staged parity checks against `dump_vdp3_interm.py` intermediates.
//!
//! The end-to-end goldens in `tests/v3_reference.rs` only see the final
//! `Q`/`P_map`; when one breaks, these tests bisect the pipeline —
//! display-native values → JND LUTs → `L_adapt` → `P` → sp0 bands — so
//! the diverging stage is visible without rerunning the Python reference.
//!
//! Stage dumps exist only for `lum_ramp_96` and `srgb_disp_96` (the two
//! ingress families: absolute luminance vs EOTF-decoded display RGB).

#[cfg(test)]
mod tests {
    use super::super::fft64::Pad;
    use super::super::params::{
        Emission, InputEncoding, Options, Surround, Task, ViewingConditions,
    };
    use super::super::pathway::visual_pathway;
    use super::super::{Params, ingress, spectral};
    use std::fs;
    use std::path::{Path, PathBuf};

    /// Stage-level budget: measured stage deltas are ≤ 1e-12; the looser
    /// 1e-9 leaves headroom for platform FP variation while still
    /// catching algorithmic drift at the stage where it enters.
    const TOL: f64 = 1e-9;

    fn dir(case: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/v3")
            .join(case)
    }

    fn read_f64(path: &Path) -> Vec<f64> {
        let b = fs::read(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        b.as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect()
    }

    fn assert_close(what: &str, a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        let d = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(d <= TOL, "{what}: max|Δ| = {d:e}");
    }

    /// `sRGB-display` ingress — exercises `display_model_srgb`,
    /// `itu2native('rgb-bt.709')` on the `led-lcd-srgb` emission, and
    /// `fix_out_of_gamut`, then the shared pathway stages.
    #[test]
    fn srgb_disp_96_stages() {
        let d = dir("srgb_disp_96");
        let (w, h) = (96usize, 96usize);
        let test = read_f64(&d.join("test.f64le"));
        let refr = read_f64(&d.join("ref.f64le"));

        let par = Params::new(
            Task::Quality,
            ViewingConditions::new(30.0, Surround::None, 24),
            InputEncoding::SrgbDisplay,
            Emission::default_for(InputEncoding::SrgbDisplay),
            Options::reference(Task::Quality),
        )
        .unwrap();
        let mp = par.task_par();
        let channels = 3;
        let mut test_n = ingress::display_model_srgb(&test);
        let mut ref_n = ingress::display_model_srgb(&refr);
        let img_e = spectral::emission_columns(&par.emission).unwrap();
        let m = spectral::itu2native(par.encoding, &img_e)
            .unwrap()
            .expect("srgb-display maps through bt.709→native");
        ingress::colorspace_transform(&mut test_n, &m);
        ingress::colorspace_transform(&mut ref_n, &m);
        let _ = ingress::fix_out_of_gamut(&mut test_n, channels);
        let _ = ingress::fix_out_of_gamut(&mut ref_n, channels);

        assert_close(
            "test_native",
            &test_n,
            &read_f64(&d.join("dump_test_native.f64le")),
        );
        assert_close(
            "ref_native",
            &ref_n,
            &read_f64(&d.join("dump_ref_native.f64le")),
        );

        let pad_at = |_k: usize| Pad::Symmetric;
        let pr =
            visual_pathway(&ref_n, w, h, channels, &mp, &par.viewing, &img_e, &pad_at).unwrap();
        assert_close(
            "L_adapt_R",
            &pr.l_adapt,
            &read_f64(&d.join("dump_L_adapt_R.f64le")),
        );
        for b in 0..pr.bands.band_count() {
            assert_close(
                &format!("band{b}_R"),
                &pr.bands.band(b),
                &read_f64(&d.join(format!("dump_band{b}_R.f64le"))),
            );
        }
    }

    /// `luminance` ingress — JND LUTs, `L_adapt`, `P`, and every sp0 band
    /// for both operands.
    #[test]
    fn lum_ramp_96_stages() {
        let d = dir("lum_ramp_96");
        let (w, h) = (96usize, 96usize);
        let test = read_f64(&d.join("test.f64le"));
        let refr = read_f64(&d.join("ref.f64le"));

        let par = Params::new(
            Task::Quality,
            ViewingConditions::new(30.0, Surround::None, 24),
            InputEncoding::Luminance,
            Emission::default_for(InputEncoding::Luminance),
            Options::reference(Task::Quality),
        )
        .unwrap();
        let mp = par.task_par();
        let channels = 1;
        // luminance: no display model applied upstream.
        let test_n = test.clone();
        let ref_n = refr.clone();
        let img_e = spectral::emission_columns(&par.emission).unwrap();

        // Stage 0: JND LUTs. The dump captured `create_pn_jnd`'s output —
        // pre-`insert(0,0)` (2047 entries) and pre-`×10^bsc`; ours carry
        // both the leading 0 and the correction, so the want arrays get
        // the same scaling before comparing.
        let bsc = 10f64.powf(mp.base_sensitivity_correction);
        let pn_want_j1: Vec<f64> = read_f64(&d.join("dump_jnd1.f64le"))
            .iter()
            .map(|v| v * bsc)
            .collect();
        let pn_want_j2: Vec<f64> = read_f64(&d.join("dump_jnd2.f64le"))
            .iter()
            .map(|v| v * bsc)
            .collect();
        let c_l: Vec<f64> = (0..2048)
            .map(|i| 10f64.powf(-5.0 + 10.0 * i as f64 / 2047.0))
            .collect();
        let s_a: Vec<f64> = c_l
            .iter()
            .map(|&l| ingress::joint_rod_cone_sens(l, &mp.csf_sa))
            .collect();
        let s_r: Vec<f64> = c_l
            .iter()
            .map(|&l| ingress::rod_sens(l, &mp.csf_sr) * 10f64.powf(mp.rod_sensitivity))
            .collect();
        let (_y_lut, jnd_c, jnd_r) =
            ingress::pn_jnd_luts(&s_a, &s_r, &c_l, mp.base_sensitivity_correction);
        assert_close("jnd_cone", &jnd_c[1..], &pn_want_j1);
        assert_close("jnd_rod", &jnd_r[1..], &pn_want_j2);

        // Stage 1: pathway outputs.
        let pad_at = |_k: usize| Pad::Symmetric;
        let pr =
            visual_pathway(&ref_n, w, h, channels, &mp, &par.viewing, &img_e, &pad_at).unwrap();
        let pt =
            visual_pathway(&test_n, w, h, channels, &mp, &par.viewing, &img_e, &pad_at).unwrap();

        assert_close(
            "L_adapt_R",
            &pr.l_adapt,
            &read_f64(&d.join("dump_L_adapt_R.f64le")),
        );
        assert_close("P_R", &pr.p, &read_f64(&d.join("dump_P_R.f64le")));
        assert_close("P_T", &pt.p, &read_f64(&d.join("dump_P_T.f64le")));

        // Stage 2: every sp0 band, both operands.
        for b in 0..pr.bands.band_count() {
            assert_close(
                &format!("band{b}_R"),
                &pr.bands.band(b),
                &read_f64(&d.join(format!("dump_band{b}_R.f64le"))),
            );
            assert_close(
                &format!("band{b}_T"),
                &pt.bands.band(b),
                &read_f64(&d.join(format!("dump_band{b}_T.f64le"))),
            );
        }
    }
}
