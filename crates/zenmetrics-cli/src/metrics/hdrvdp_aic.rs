//! HDR-VDP on 8-bit sRGB: the explicit AIC2026 SDR conventions (`*-aic2026`) and the plain routes'
//! upstream `sRGB-display` encoding, both separate from the absolute-nits (`--hdr`) routes.
use super::MetricKind;
use crate::decode::Rgb8Image;
use zenmetrics_api::hdrvdp_cpu as vdp;

static CLI: std::sync::OnceLock<Args> = std::sync::OnceLock::new();

/// Overrides for the explicit SDR display routes only.
#[derive(Debug, Clone, Default, clap::Args)]
pub(crate) struct Args {
    /// SDR display peak in cd/m² for hdrvdp-aic2026/hdrvdp3-aic2026 (default 200).
    #[arg(long, value_parser = positive)]
    hdrvdp_sdr_peak: Option<f64>,
    /// Pixels per degree for the SDR routes (v2 default 37.84; v3 37.842505866).
    #[arg(long, value_parser = positive)]
    hdrvdp_sdr_ppd: Option<f64>,
    /// Code quantization before the EOTF: 0 disables it (v2 default); v3 defaults to 10.
    #[arg(long, value_parser = clap::value_parser!(u8).range(0..=16))]
    hdrvdp_sdr_code_bits: Option<u8>,
    /// Use strict arithmetic for the HDR-VDP 2 SDR route (reference comparison).
    #[arg(long)]
    hdrvdp_sdr_strict: bool,
}

fn positive(s: &str) -> Result<f64, String> {
    let value: f64 = s.parse().map_err(|_| "expected a number".to_string())?;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err("expected a finite positive value".into())
    }
}

// The binary compiles this module independently of the library.
#[allow(dead_code)]
pub(crate) fn set_args(args: Args) {
    let _ = CLI.set(args);
}

fn nits(image: &Rgb8Image, xyz: bool, peak: f64, bits: u8) -> Vec<f64> {
    let mut out: Vec<f64> = image
        .pixels
        .iter()
        .map(|&b| {
            let mut v = f64::from(b) / 255.0;
            if bits > 0 {
                let q = f64::from((1u32 << bits) - 1);
                v = (v * q).round() / q;
            }
            peak * if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        })
        .collect();
    if xyz {
        for p in out.as_chunks_mut::<3>().0 {
            let [r, g, b] = [p[0], p[1], p[2]];
            p[0] = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
            p[1] = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
            p[2] = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
        }
    }
    out
}

pub(super) fn score(
    kind: MetricKind,
    r: &Rgb8Image,
    d: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if (r.width, r.height) != (d.width, d.height) {
        return Err("HDR-VDP AIC2026 reference/distorted dimensions differ".into());
    }
    let (w, h) = (r.width as usize, r.height as usize);
    let v3 = kind == MetricKind::Hdrvdp3Aic2026;
    let args = CLI.get().cloned().unwrap_or_default();
    let peak = args.hdrvdp_sdr_peak.unwrap_or(200.0);
    let ppd = args
        .hdrvdp_sdr_ppd
        .unwrap_or(if v3 { 37.842505866 } else { 37.84 });
    let bits = args.hdrvdp_sdr_code_bits.unwrap_or(if v3 { 10 } else { 0 });
    let (r, d) = (nits(r, v3, peak, bits), nits(d, v3, peak, bits));
    if v3 {
        use vdp::v3::*;
        let p = Params::new(
            Task::Quality,
            ViewingConditions::new(ppd, Surround::None, 24),
            InputEncoding::Xyz,
            Emission::Preset(DisplayPreset::Oled),
            Options::reference(Task::Quality),
        )?;
        let result = hdrvdp3(&d, &r, w, h, &p)?;
        Ok(vec![
            ("hdrvdp3_aic2026_q", result.q),
            ("hdrvdp3_aic2026_jod", result.q_jod),
        ])
    } else {
        let mut params = vdp::Params::new(ppd);
        if args.hdrvdp_sdr_strict {
            params = params.with_strict();
        }
        Ok(vec![(
            "hdrvdp_aic2026_q",
            vdp::score(&d, &r, w, h, vdp::ColorEncoding::RgbBt709, &params)?,
        )])
    }
}

/// Plain `hdrvdp` / `hdrvdp3` on 8-bit sRGB: upstream's own `sRGB-display` encoding, which `hdrvdp.m`
/// documents "for standard (LDR) color images" (sRGB EOTF onto a 100 cd/m² peak, 1 cd/m² black
/// display), at the crate's default 30 pixels per degree. v3 adds task `quality`, the encoding's
/// upstream emission (`led-lcd-srgb`), age 24 and no surround, and reports Q_JOD.
pub(super) fn score_srgb_display(
    kind: MetricKind,
    r: &Rgb8Image,
    d: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    if (r.width, r.height) != (d.width, d.height) {
        return Err(format!("{}: reference/distorted dimensions differ", kind.name()).into());
    }
    let (w, h) = (r.width as usize, r.height as usize);
    let unit = |image: &Rgb8Image| {
        image
            .pixels
            .iter()
            .map(|&b| f64::from(b) / 255.0)
            .collect::<Vec<f64>>()
    };
    let (r, d) = (unit(r), unit(d));
    if kind == MetricKind::Hdrvdp3 {
        use vdp::v3::*;
        let p = Params::new(
            Task::Quality,
            ViewingConditions::new(vdp::DEFAULT_PIX_PER_DEG, Surround::None, 24),
            InputEncoding::SrgbDisplay,
            Emission::default_for(InputEncoding::SrgbDisplay),
            Options::reference(Task::Quality),
        )?;
        Ok(vec![(
            kind.column_names()[0],
            vdp::v3::score(&d, &r, w, h, &p)?,
        )])
    } else {
        let params = vdp::Params::new(vdp::DEFAULT_PIX_PER_DEG);
        Ok(vec![(
            kind.column_names()[0],
            vdp::score(&d, &r, w, h, vdp::ColorEncoding::SrgbDisplay, &params)?,
        )])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdr_display_black_white_and_primary_units() {
        let image = Rgb8Image {
            width: 3,
            height: 1,
            pixels: vec![0, 0, 0, 255, 255, 255, 255, 0, 0],
        };
        let rgb = nits(&image, false, 200.0, 0);
        assert_eq!(&rgb[..3], &[0.0; 3]);
        assert_eq!(&rgb[3..6], &[200.0; 3]);
        assert_eq!(&rgb[6..], &[200.0, 0.0, 0.0]);
        let xyz = nits(&image, true, 200.0, 10);
        // CIE XYZ values for the declared D65 sRGB red primary at this peak.
        for (actual, expected) in xyz[6..].iter().zip([82.49128, 42.53458, 3.86678]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert!(positive("NaN").is_err());
        assert!(positive("inf").is_err());
        assert!(positive("0").is_err());
        assert!(positive("-1").is_err());
        assert_eq!(positive("37.84").unwrap(), 37.84);
    }

    #[test]
    fn plain_routes_score_srgb_through_srgb_display() {
        let (w, h) = (48usize, 40usize);
        let px = |seed: u32| -> Vec<u8> {
            (0..w * h * 3)
                .map(|i| ((i as u32 * 37 + seed * 11 + (i as u32 / 7) * 5) % 256) as u8)
                .collect()
        };
        let reference = Rgb8Image {
            width: w as u32,
            height: h as u32,
            pixels: px(1),
        };
        let distorted = Rgb8Image {
            width: w as u32,
            height: h as u32,
            pixels: px(2),
        };
        let unit = |p: &[u8]| {
            p.iter()
                .map(|&b| f64::from(b) / 255.0)
                .collect::<Vec<f64>>()
        };
        let (rf, df) = (unit(&reference.pixels), unit(&distorted.pixels));
        let v2 = score_srgb_display(MetricKind::Hdrvdp, &reference, &distorted).unwrap();
        let want = vdp::score(
            &df,
            &rf,
            w,
            h,
            vdp::ColorEncoding::SrgbDisplay,
            &vdp::Params::new(vdp::DEFAULT_PIX_PER_DEG),
        )
        .unwrap();
        assert_eq!(v2[0].1.to_bits(), want.to_bits());
        assert!(want < 100.0);
        let same = score_srgb_display(MetricKind::Hdrvdp, &reference, &reference).unwrap();
        assert_eq!(same[0].1, 100.0);
        let v3 = score_srgb_display(MetricKind::Hdrvdp3, &reference, &distorted).unwrap();
        assert!(v3[0].1.is_finite() && v3[0].1 < 10.0, "{v3:?}");
        let short = Rgb8Image {
            width: 4,
            height: 4,
            pixels: vec![0; 48],
        };
        assert!(score_srgb_display(MetricKind::Hdrvdp, &reference, &short).is_err());
    }
}
