//! Byte-exact output lock for the whole HDR-VDP-2.2 pipeline.
//!
//! ## Why this exists
//!
//! hdrvdp's outputs are metric scores: a 1-ULP change is a silently wrong
//! number, not a rounding detail. This lock freezes the **naive reference
//! semantics** of every pipeline stage as verbatim copies of the original
//! (pre-optimisation, 2026-08-29) implementations, and asserts that the
//! production code produces **bit-identical** output — `f64::to_bits`
//! equality, element by element — on fixtures covering every stage, odd and
//! even sizes, all colour encodings, and every `do_*` toggle.
//!
//! Any optimisation of the production code must keep this green. An
//! optimisation that cannot be bit-identical (e.g. a float reassociation) is a
//! **score change** and needs the owner's sign-off — never weaken this lock to
//! admit one.
//!
//! ## Why dual-path rather than golden constants
//!
//! `sin/cos/exp/ln/powf` come from the platform libm and are not correctly
//! rounded, so a hex golden captured on one platform is not portable to CI's
//! other platforms. Running the frozen reference and the production code **in
//! the same process** compares like with like on every platform, forever.
//!
//! The one thing dual-path cannot catch is a change to a *shared constant
//! table* (both paths would read the new value). Those are pure literals, so
//! they ARE portable — `constant_tables_are_frozen` pins them with an FNV-1a
//! hash over their exact bit patterns.
//!
//! ## Scope
//!
//! Everything from `to_nits` to `Q_MOS` / `P_map` is dual-pathed, including
//! the full end-to-end glue. NOT dual-pathed: `spectral.rs` (`emission_spectra`
//! / `lmsr_matrix` — one-time setup, microseconds, shared by both paths) and
//! `interp::interp1_pchip` (only used by spectral). If spectral is ever
//! optimised, extend this lock first.

#![forbid(unsafe_code)]

use hdrvdp::display::ColorEncoding;
use hdrvdp::params::Params;
use hdrvdp::spyr::Band;

// ═══════════════════════════════════════════════════════════════════════════
// Frozen reference implementations — verbatim copies of the 2026-08-29
// pre-optimisation source. DO NOT "sync" these with later production changes:
// their entire value is that they do not move. A deliberate semantic change to
// the metric (owner-approved) is the only reason to touch them.
// ═══════════════════════════════════════════════════════════════════════════
#[allow(clippy::needless_range_loop)]
mod reference {
    use core::f64::consts::PI;
    use hdrvdp::display::{ColorEncoding, SDR_BLACK_NITS, SDR_GAMMA, SDR_PEAK_NITS};
    use hdrvdp::params::Params;
    use hdrvdp::spectral::{emission_spectra, lmsr_matrix};
    use hdrvdp::spyr::{Band, ORIENTATIONS, SteerablePyramid};
    use hdrvdp::{Error, HdrVdpResult};

    // ── interp.rs ─────────────────────────────────────────────────────────

    pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
        if x.is_nan() || x < lo {
            lo
        } else if x > hi {
            hi
        } else {
            x
        }
    }

    pub fn interp1_linear(xs: &[f64], ys: &[f64], x: f64) -> f64 {
        debug_assert_eq!(xs.len(), ys.len());
        debug_assert!(xs.len() >= 2);
        if x <= xs[0] {
            return ys[0];
        }
        let n = xs.len();
        if x >= xs[n - 1] {
            return ys[n - 1];
        }
        let i = match xs.binary_search_by(|probe| probe.partial_cmp(&x).expect("finite grid")) {
            Ok(i) => return ys[i],
            Err(i) => i - 1,
        };
        let t = (x - xs[i]) / (xs[i + 1] - xs[i]);
        ys[i] + t * (ys[i + 1] - ys[i])
    }

    pub fn point_op(lut: &[f64], x0: f64, dx: f64, x: f64) -> f64 {
        debug_assert!(lut.len() >= 2);
        debug_assert!(dx > 0.0);
        let pos = (x - x0) / dx;
        if !matches!(pos.partial_cmp(&0.0), Some(core::cmp::Ordering::Greater)) {
            return lut[0];
        }
        let last = lut.len() - 1;
        if pos >= last as f64 {
            return lut[last];
        }
        let i = pos as usize;
        let t = pos - i as f64;
        lut[i] + t * (lut[i + 1] - lut[i])
    }

    pub fn cumtrapz(x: &[f64], y: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x.len(), y.len());
        let mut out = vec![0.0; x.len()];
        let mut acc = 0.0;
        for i in 1..x.len() {
            acc += 0.5 * (x[i] - x[i - 1]) * (y[i] + y[i - 1]);
            out[i] = acc;
        }
        out
    }

    // ── csf.rs ────────────────────────────────────────────────────────────

    pub fn mtf(rho: f64, par: &Params) -> f64 {
        let mut m = 0.0;
        for k in 0..4 {
            m += par.mtf_params_a[k] * (-par.mtf_params_b[k] * rho).exp();
        }
        m
    }

    pub fn ncsf(rho: f64, lum: f64, par: &Params) -> f64 {
        let lum_lut: Vec<f64> = par.csf_lums.iter().map(|l| l.log10()).collect();
        let log_lum = clamp(lum.log10(), lum_lut[0], lum_lut[lum_lut.len() - 1]);

        let mut p = [0.0f64; 4];
        for (k, pk) in p.iter_mut().enumerate() {
            let col: Vec<f64> = par.csf_params.iter().map(|row| row[k + 1]).collect();
            *pk = interp1_linear(&lum_lut, &col, log_lum);
        }
        ncsf_from_coeffs(rho, &p)
    }

    pub fn ncsf_from_coeffs(rho: f64, p: &[f64; 4]) -> f64 {
        let b = 1.0 - (-(rho / 7.0).powi(2)).exp();
        let a = 1.0 + (p[0] * rho).powf(p[1]);
        p[3] * b.powf(p[2] / 2.0) / a.sqrt()
    }

    pub fn joint_rod_cone_sens(la: f64, par: &Params) -> f64 {
        let [peak, cvi_sens_drop, cvi_trans_slope, cvi_low_slope] = par.csf_sa;
        peak * ((cvi_sens_drop / la).powf(cvi_trans_slope) + 1.0).powf(-cvi_low_slope)
    }

    pub fn rod_sens(la: f64, par: &Params) -> f64 {
        let [peak_l, low_s, low_exp, high_s, high_exp, rod_sens_scale] = par.csf_sr_par;
        let t = (la / peak_l).log10().abs();
        let s = if la > peak_l {
            (-t.powf(high_exp) / high_s).exp()
        } else {
            (-t.powf(low_exp) / low_s).exp()
        };
        s * 10f64.powf(rod_sens_scale)
    }

    // ── display.rs ────────────────────────────────────────────────────────

    pub fn display_model(v: f64, gamma: f64, peak: f64, black_level: f64) -> f64 {
        peak * v.powf(gamma) + black_level
    }

    pub fn display_model_srgb(srgb: f64) -> f64 {
        const A: f64 = 0.055;
        const THR: f64 = 0.04045;
        let lin = if srgb <= THR {
            srgb / 12.92
        } else {
            ((srgb + A) / (1.0 + A)).powf(2.4)
        };
        SDR_PEAK_NITS * lin + SDR_BLACK_NITS
    }

    pub fn xyz_to_rgb(xyz: [f64; 3]) -> [f64; 3] {
        const M: [[f64; 3]; 3] = [
            [3.240708, -1.537259, -0.498570],
            [-0.969257, 1.875995, 0.041555],
            [0.055636, -0.203996, 1.057069],
        ];
        let mut out = [0.0; 3];
        for (o, row) in out.iter_mut().zip(M) {
            *o = row[0] * xyz[0] + row[1] * xyz[1] + row[2] * xyz[2];
        }
        out
    }

    pub fn to_nits(
        pixels: &[f64],
        width: usize,
        height: usize,
        encoding: ColorEncoding,
    ) -> Result<Vec<f64>, Error> {
        let ch = encoding.channels();
        let want = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(ch))
            .ok_or(Error::DimensionOverflow)?;
        if pixels.len() != want {
            return Err(Error::ChannelMismatch {
                expected: want,
                got: pixels.len(),
            });
        }

        let out: Vec<f64> = match encoding {
            ColorEncoding::Luminance | ColorEncoding::RgbBt709 => pixels.to_vec(),
            ColorEncoding::LumaDisplay => pixels
                .iter()
                .map(|&v| display_model(v, SDR_GAMMA, SDR_PEAK_NITS, SDR_BLACK_NITS))
                .collect(),
            ColorEncoding::SrgbDisplay => pixels.iter().map(|&v| display_model_srgb(v)).collect(),
            ColorEncoding::Xyz => {
                let mut o = Vec::with_capacity(pixels.len());
                let (triples, _) = pixels.as_chunks::<3>();
                for p in triples {
                    o.extend_from_slice(&xyz_to_rgb([p[0], p[1], p[2]]));
                }
                o
            }
        };

        if let Some(bad) = out.iter().find(|v| !v.is_finite()) {
            return Err(Error::ImpossibleValues(*bad));
        }
        Ok(out)
    }

    pub fn looks_relative(nits: &[f64], channels: usize, encoding: ColorEncoding) -> bool {
        if !encoding.expects_absolute_input() {
            return false;
        }
        let probe: f64 = if channels == 3 {
            nits.as_chunks::<3>()
                .0
                .iter()
                .map(|p| p[1])
                .fold(f64::MIN, f64::max)
        } else {
            nits.iter().copied().fold(f64::MIN, f64::max)
        };
        probe <= 1.0
    }

    // ── fft.rs ────────────────────────────────────────────────────────────

    #[derive(Debug, Clone, Copy, PartialEq, Default)]
    pub struct Complex {
        pub re: f64,
        pub im: f64,
    }

    impl Complex {
        pub const fn new(re: f64, im: f64) -> Self {
            Self { re, im }
        }
        pub fn expi(theta: f64) -> Self {
            let (s, c) = theta.sin_cos();
            Self { re: c, im: s }
        }
        pub fn conj(self) -> Self {
            Self {
                re: self.re,
                im: -self.im,
            }
        }
        fn mul(self, o: Self) -> Self {
            Self {
                re: self.re * o.re - self.im * o.im,
                im: self.re * o.im + self.im * o.re,
            }
        }
        fn add(self, o: Self) -> Self {
            Self {
                re: self.re + o.re,
                im: self.im + o.im,
            }
        }
        fn sub(self, o: Self) -> Self {
            Self {
                re: self.re - o.re,
                im: self.im - o.im,
            }
        }
        fn scale(self, s: f64) -> Self {
            Self {
                re: self.re * s,
                im: self.im * s,
            }
        }
    }

    pub fn fft(buf: &mut [Complex]) {
        let n = buf.len();
        if n <= 1 {
            return;
        }
        if n.is_power_of_two() {
            fft_radix2(buf);
        } else {
            let out = fft_bluestein(buf);
            buf.copy_from_slice(&out);
        }
    }

    pub fn ifft(buf: &mut [Complex]) {
        let n = buf.len();
        if n == 0 {
            return;
        }
        for v in buf.iter_mut() {
            *v = v.conj();
        }
        fft(buf);
        let s = 1.0 / n as f64;
        for v in buf.iter_mut() {
            *v = v.conj().scale(s);
        }
    }

    fn fft_radix2(buf: &mut [Complex]) {
        let n = buf.len();
        debug_assert!(n.is_power_of_two());

        let bits = n.trailing_zeros();
        for i in 0..n {
            let j = (i as u32).reverse_bits() >> (32 - bits);
            let j = j as usize;
            if j > i {
                buf.swap(i, j);
            }
        }

        let mut len = 2;
        while len <= n {
            let ang = -2.0 * PI / len as f64;
            let half = len / 2;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let w = Complex::expi(ang * k as f64);
                    let u = buf[start + k];
                    let v = buf[start + k + half].mul(w);
                    buf[start + k] = u.add(v);
                    buf[start + k + half] = u.sub(v);
                }
            }
            len <<= 1;
        }
    }

    fn fft_bluestein(buf: &[Complex]) -> Vec<Complex> {
        let n = buf.len();
        let m = (2 * n - 1).next_power_of_two();

        let chirp: Vec<Complex> = (0..n)
            .map(|k| {
                let kk = (k as u128 * k as u128 % (2 * n as u128)) as f64;
                Complex::expi(-PI * kk / n as f64)
            })
            .collect();

        let mut a = vec![Complex::default(); m];
        for k in 0..n {
            a[k] = buf[k].mul(chirp[k]);
        }

        let mut b = vec![Complex::default(); m];
        for k in 0..n {
            b[k] = chirp[k].conj();
            if k > 0 {
                b[m - k] = chirp[k].conj();
            }
        }

        fft_radix2(&mut a);
        fft_radix2(&mut b);
        for (x, y) in a.iter_mut().zip(&b) {
            *x = x.mul(*y);
        }
        for v in a.iter_mut() {
            *v = v.conj();
        }
        fft_radix2(&mut a);
        let s = 1.0 / m as f64;
        for v in a.iter_mut() {
            *v = v.conj().scale(s);
        }

        (0..n).map(|k| a[k].mul(chirp[k])).collect()
    }

    pub fn fft2(buf: &mut [Complex], width: usize, height: usize) {
        debug_assert_eq!(buf.len(), width * height);
        for row in buf.chunks_exact_mut(width) {
            fft(row);
        }
        let mut col = vec![Complex::default(); height];
        for x in 0..width {
            for (y, c) in col.iter_mut().enumerate() {
                *c = buf[y * width + x];
            }
            fft(&mut col);
            for (y, c) in col.iter().enumerate() {
                buf[y * width + x] = *c;
            }
        }
    }

    pub fn ifft2(buf: &mut [Complex], width: usize, height: usize) {
        debug_assert_eq!(buf.len(), width * height);
        for v in buf.iter_mut() {
            *v = v.conj();
        }
        fft2(buf, width, height);
        let s = 1.0 / (width * height) as f64;
        for v in buf.iter_mut() {
            *v = v.conj().scale(s);
        }
    }

    pub fn cycles_per_degree_grid(width: usize, height: usize, pix_per_deg: f64) -> Vec<f64> {
        let fx = dft_bin_frequencies(width, pix_per_deg);
        let fy = dft_bin_frequencies(height, pix_per_deg);
        let mut out = Vec::with_capacity(width * height);
        for &v in &fy {
            for &u in &fx {
                out.push((u * u + v * v).sqrt());
            }
        }
        out
    }

    fn dft_bin_frequencies(n: usize, pix_per_deg: f64) -> Vec<f64> {
        (0..n)
            .map(|k| {
                let k = if k * 2 <= n {
                    k as f64
                } else {
                    k as f64 - n as f64
                };
                k / n as f64 * pix_per_deg
            })
            .collect()
    }

    pub fn conv_fft_real(
        x: &[f64],
        width: usize,
        height: usize,
        filter: &[f64],
        pad_w: usize,
        pad_h: usize,
        pad_value: f64,
    ) -> Vec<f64> {
        assert_eq!(
            x.len(),
            width * height,
            "conv_fft_real: image size mismatch"
        );
        assert_eq!(
            filter.len(),
            pad_w * pad_h,
            "conv_fft_real: filter size mismatch"
        );
        assert!(
            pad_w >= width && pad_h >= height,
            "conv_fft_real: padded size must not be smaller than the image"
        );

        let mut buf = vec![Complex::new(pad_value, 0.0); pad_w * pad_h];
        for y in 0..height {
            for xi in 0..width {
                buf[y * pad_w + xi] = Complex::new(x[y * width + xi], 0.0);
            }
        }

        fft2(&mut buf, pad_w, pad_h);
        for (b, f) in buf.iter_mut().zip(filter) {
            *b = b.scale(*f);
        }
        ifft2(&mut buf, pad_w, pad_h);

        let mut out = Vec::with_capacity(width * height);
        for y in 0..height {
            for xi in 0..width {
                out.push(buf[y * pad_w + xi].re);
            }
        }
        out
    }

    // ── spyr.rs ───────────────────────────────────────────────────────────

    use hdrvdp::sp3_filters::{BFILTS, HI0FILT, LO0FILT, LOFILT};

    pub fn reflect1(i: isize, n: usize) -> usize {
        debug_assert!(n > 0);
        if n == 1 {
            return 0;
        }
        let period = 2 * (n as isize - 1);
        let mut i = i % period;
        if i < 0 {
            i += period;
        }
        if i >= n as isize {
            i = period - i;
        }
        i as usize
    }

    pub fn corr_dn(
        image: &[f64],
        width: usize,
        height: usize,
        filter: &[f64],
        fs: usize,
        step: usize,
    ) -> Band {
        assert_eq!(image.len(), width * height, "corr_dn: image size mismatch");
        assert_eq!(filter.len(), fs * fs, "corr_dn: filter size mismatch");
        assert!(step >= 1);

        let out_w = width.div_ceil(step);
        let out_h = height.div_ceil(step);
        let c = (fs / 2) as isize;
        let mut out = vec![0.0; out_w * out_h];

        for oy in 0..out_h {
            let base_y = (oy * step) as isize;
            for ox in 0..out_w {
                let base_x = (ox * step) as isize;
                let mut acc = 0.0;
                for ky in 0..fs {
                    let sy = reflect1(base_y + ky as isize - c, height);
                    let row = &image[sy * width..sy * width + width];
                    let frow = &filter[ky * fs..ky * fs + fs];
                    for (kx, &w) in frow.iter().enumerate() {
                        let sx = reflect1(base_x + kx as isize - c, width);
                        acc += w * row[sx];
                    }
                }
                out[oy * out_w + ox] = acc;
            }
        }
        Band {
            width: out_w,
            height: out_h,
            data: out,
        }
    }

    pub fn up_conv(
        band: &Band,
        filter: &[f64],
        fs: usize,
        step: usize,
        out_width: usize,
        out_height: usize,
        res: &mut [f64],
    ) {
        assert_eq!(filter.len(), fs * fs, "up_conv: filter size mismatch");
        assert_eq!(
            res.len(),
            out_width * out_height,
            "up_conv: res size mismatch"
        );
        assert_eq!(
            (band.width, band.height),
            (out_width.div_ceil(step), out_height.div_ceil(step)),
            "up_conv: band size does not match the requested output size"
        );

        let c = (fs / 2) as isize;
        let sample = |y: isize, x: isize| -> f64 {
            let y = reflect1(y, out_height);
            let x = reflect1(x, out_width);
            if !y.is_multiple_of(step) || !x.is_multiple_of(step) {
                return 0.0;
            }
            band.data[(y / step) * band.width + (x / step)]
        };

        for py in 0..out_height {
            for px in 0..out_width {
                let mut acc = 0.0;
                for ky in 0..fs {
                    let sy = py as isize - ky as isize + c;
                    let frow = &filter[ky * fs..ky * fs + fs];
                    for (kx, &w) in frow.iter().enumerate() {
                        if w == 0.0 {
                            continue;
                        }
                        acc += w * sample(sy, px as isize - kx as isize + c);
                    }
                }
                res[py * out_width + px] += acc;
            }
        }
    }

    fn flat<const N: usize>(f: &[[f64; N]; N]) -> Vec<f64> {
        f.iter().flat_map(|r| r.iter().copied()).collect()
    }

    pub fn max_pyr_height(width: usize, height: usize, filter_size: usize) -> usize {
        let (mut w, mut h, mut n) = (width, height, 0usize);
        while w >= filter_size && h >= filter_size {
            n += 1;
            w /= 2;
            h /= 2;
        }
        n
    }

    pub fn build(
        image: &[f64],
        width: usize,
        height: usize,
        levels: Option<usize>,
    ) -> SteerablePyramid {
        assert_eq!(image.len(), width * height, "build: image size mismatch");
        let lo0 = flat(&LO0FILT);
        let hi0 = flat(&HI0FILT);
        let lo = flat(&LOFILT);
        let bfilts: Vec<Vec<f64>> = BFILTS.iter().map(flat).collect();

        let max_ht = max_pyr_height(width, height, 17);
        let ht = levels.unwrap_or(max_ht);
        assert!(
            ht <= max_ht,
            "build: cannot build {ht} levels on a {width}×{height} image (max {max_ht})"
        );

        let high_pass = corr_dn(image, width, height, &hi0, 15, 1);
        let mut current = corr_dn(image, width, height, &lo0, 9, 1);

        let mut out_levels: Vec<[Band; ORIENTATIONS]> = Vec::with_capacity(ht);
        for _ in 0..ht {
            let bands: Vec<Band> = bfilts
                .iter()
                .map(|f| corr_dn(&current.data, current.width, current.height, f, 9, 1))
                .collect();
            let bands: [Band; ORIENTATIONS] = bands
                .try_into()
                .unwrap_or_else(|_| unreachable!("BFILTS has ORIENTATIONS entries"));
            out_levels.push(bands);
            current = corr_dn(&current.data, current.width, current.height, &lo, 17, 2);
        }

        SteerablePyramid {
            high_pass,
            levels: out_levels,
            low_pass: current,
            width,
            height,
        }
    }

    pub fn reconstruct(pyr: &SteerablePyramid) -> Band {
        let lo0 = flat(&LO0FILT);
        let hi0 = flat(&HI0FILT);
        let lo = flat(&LOFILT);
        let bfilts: Vec<Vec<f64>> = BFILTS.iter().map(flat).collect();

        let mut acc = pyr.low_pass.clone();
        for level in pyr.levels.iter().rev() {
            let (w, h) = (level[0].width, level[0].height);
            let mut res = vec![0.0; w * h];
            up_conv(&acc, &lo, 17, 2, w, h, &mut res);
            for (band, f) in level.iter().zip(&bfilts) {
                up_conv(band, f, 9, 1, w, h, &mut res);
            }
            acc = Band {
                width: w,
                height: h,
                data: res,
            };
        }

        let mut out = vec![0.0; pyr.width * pyr.height];
        up_conv(&acc, &lo0, 9, 1, pyr.width, pyr.height, &mut out);
        up_conv(&pyr.high_pass, &hi0, 15, 1, pyr.width, pyr.height, &mut out);
        Band {
            width: pyr.width,
            height: pyr.height,
            data: out,
        }
    }

    // ── resize.rs ─────────────────────────────────────────────────────────

    pub fn cubic(x: f64) -> f64 {
        let a = x.abs();
        let a2 = a * a;
        let a3 = a2 * a;
        if a <= 1.0 {
            1.5 * a3 - 2.5 * a2 + 1.0
        } else if a <= 2.0 {
            -0.5 * a3 + 2.5 * a2 - 4.0 * a + 2.0
        } else {
            0.0
        }
    }

    struct Contributions {
        weights: Vec<f64>,
        indices: Vec<usize>,
        taps: usize,
    }

    fn contributions(in_length: usize, out_length: usize) -> Contributions {
        assert!(in_length > 0 && out_length > 0);
        let scale = out_length as f64 / in_length as f64;
        let (kernel_width, stretch) = if scale < 1.0 {
            (4.0 / scale, scale)
        } else {
            (4.0, 1.0)
        };
        let taps = (kernel_width.ceil() as usize) + 2;

        let mut weights = Vec::with_capacity(out_length * taps);
        let mut indices = Vec::with_capacity(out_length * taps);

        let aux_len = 2 * in_length;
        let aux = |i: isize| -> usize {
            let m = i.rem_euclid(aux_len as isize) as usize;
            if m < in_length { m } else { aux_len - 1 - m }
        };

        for x in 0..out_length {
            let u = (x as f64 + 1.0) / scale + 0.5 * (1.0 - 1.0 / scale);
            let left = (u - kernel_width / 2.0).floor();

            let mut row: Vec<f64> = Vec::with_capacity(taps);
            let mut sum = 0.0;
            for t in 0..taps {
                let idx = left + t as f64;
                let w = stretch * cubic(stretch * (u - idx));
                row.push(w);
                sum += w;
            }
            for w in &mut row {
                *w /= sum;
            }
            for (t, w) in row.into_iter().enumerate() {
                let idx1 = left as isize + t as isize;
                indices.push(aux(idx1 - 1));
                weights.push(w);
            }
        }

        Contributions {
            weights,
            indices,
            taps,
        }
    }

    pub fn imresize(
        image: &[f64],
        width: usize,
        height: usize,
        out_width: usize,
        out_height: usize,
    ) -> Vec<f64> {
        assert_eq!(image.len(), width * height, "imresize: image size mismatch");
        assert!(width > 0 && height > 0 && out_width > 0 && out_height > 0);
        if (width, height) == (out_width, out_height) {
            return image.to_vec();
        }

        let hc = contributions(width, out_width);
        let mut tmp = vec![0.0; out_width * height];
        for y in 0..height {
            let src = &image[y * width..y * width + width];
            for x in 0..out_width {
                let base = x * hc.taps;
                let mut acc = 0.0;
                for t in 0..hc.taps {
                    acc += hc.weights[base + t] * src[hc.indices[base + t]];
                }
                tmp[y * out_width + x] = acc;
            }
        }

        let vc = contributions(height, out_height);
        let mut out = vec![0.0; out_width * out_height];
        for y in 0..out_height {
            let base = y * vc.taps;
            for x in 0..out_width {
                let mut acc = 0.0;
                for t in 0..vc.taps {
                    acc += vc.weights[base + t] * tmp[vc.indices[base + t] * out_width + x];
                }
                out[y * out_width + x] = acc;
            }
        }
        out
    }

    // ── photoreceptor.rs ──────────────────────────────────────────────────

    const LUT_N: usize = 2048;
    const LUT_LOG_MIN: f64 = -5.0;
    const LUT_LOG_MAX: f64 = 5.0;

    pub struct Photoreceptor {
        pub log_lum: Vec<f64>,
        pub jnd_cone: Vec<f64>,
        pub jnd_rod: Vec<f64>,
    }

    impl Photoreceptor {
        pub fn new(par: &Params) -> Self {
            let log_lum: Vec<f64> = (0..LUT_N)
                .map(|i| {
                    LUT_LOG_MIN + (LUT_LOG_MAX - LUT_LOG_MIN) * (i as f64) / ((LUT_N - 1) as f64)
                })
                .collect();
            let lum: Vec<f64> = log_lum.iter().map(|l| 10f64.powf(*l)).collect();

            let s_a: Vec<f64> = lum.iter().map(|&l| joint_rod_cone_sens(l, par)).collect();
            let rod_scale = 10f64.powf(par.rod_sensitivity);
            let s_r: Vec<f64> = lum.iter().map(|&l| rod_sens(l, par) * rod_scale).collect();

            let cone_src: Vec<f64> = s_a
                .iter()
                .zip(&s_r)
                .map(|(a, r)| (a - r).max(1e-3))
                .collect();
            let l_max = lum[LUT_N - 1];
            let s_c: Vec<f64> = lum
                .iter()
                .map(|&l| 0.5 * interp1_linear(&lum, &cone_src, (l * 2.0).min(l_max)))
                .collect();

            let sc = par.sensitivity_correction;
            let jnd_cone: Vec<f64> = build_jnd_space(&log_lum, &s_c)
                .into_iter()
                .map(|v| v * sc)
                .collect();
            let jnd_rod: Vec<f64> = build_jnd_space(&log_lum, &s_r)
                .into_iter()
                .map(|v| v * sc)
                .collect();

            Self {
                log_lum,
                jnd_cone,
                jnd_rod,
            }
        }

        pub fn lum_min(&self) -> f64 {
            10f64.powf(self.log_lum[0])
        }

        pub fn lum_max(&self) -> f64 {
            10f64.powf(self.log_lum[self.log_lum.len() - 1])
        }

        fn step(&self) -> f64 {
            self.log_lum[1] - self.log_lum[0]
        }

        pub fn cone(&self, lum: f64) -> f64 {
            let l = clamp(lum, self.lum_min(), self.lum_max()).log10();
            point_op(&self.jnd_cone, self.log_lum[0], self.step(), l)
        }

        pub fn rod(&self, lum: f64) -> f64 {
            let l = clamp(lum, self.lum_min(), self.lum_max()).log10();
            point_op(&self.jnd_rod, self.log_lum[0], self.step(), l)
        }
    }

    fn build_jnd_space(log_lum: &[f64], s: &[f64]) -> Vec<f64> {
        debug_assert_eq!(log_lum.len(), s.len());
        let ln10 = core::f64::consts::LN_10;
        let d: Vec<f64> = s.iter().map(|v| v * ln10).collect();
        cumtrapz(log_lum, &d)
    }

    // ── pathway.rs ────────────────────────────────────────────────────────

    pub const MIN_NITS: f64 = 1e-5;
    pub const MAX_NITS: f64 = 1e10;

    pub struct Pathway {
        pub achromatic: Vec<f64>,
        pub l_adapt: Vec<f64>,
        pub width: usize,
        pub height: usize,
    }

    pub fn visual_pathway(
        nits: &[f64],
        width: usize,
        height: usize,
        par: &Params,
        pn: &Photoreceptor,
        lmsr: &[[f64; 4]],
        surround: &[f64],
    ) -> Pathway {
        let channels = lmsr.len();
        assert_eq!(
            nits.len(),
            width * height * channels,
            "visual_pathway: pixel buffer does not match {width}×{height}×{channels}"
        );
        assert_eq!(
            surround.len(),
            channels,
            "visual_pathway: need one surround value per channel"
        );
        let n = width * height;

        let (pad_w, pad_h) = (width * 2, height * 2);
        let mtf_filter: Option<Vec<f64>> = par.do_mtf.then(|| {
            cycles_per_degree_grid(pad_w, pad_h, par.pix_per_deg)
                .into_iter()
                .map(|rho| mtf(rho, par))
                .collect()
        });

        let mut optical: Vec<Vec<f64>> = Vec::with_capacity(channels);
        for c in 0..channels {
            let plane: Vec<f64> = (0..n).map(|i| nits[i * channels + c]).collect();
            let filtered = match &mtf_filter {
                Some(f) => conv_fft_real(&plane, width, height, f, pad_w, pad_h, surround[c])
                    .into_iter()
                    .map(|v| clamp(v, MIN_NITS, MAX_NITS))
                    .collect(),
                None => plane,
            };
            optical.push(filtered);
        }

        let mut r_lmsr = [vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]];
        for (c, plane) in optical.iter().enumerate() {
            for (k, out) in r_lmsr.iter_mut().enumerate() {
                let m = lmsr[c][k];
                for (o, v) in out.iter_mut().zip(plane) {
                    *o += m * v;
                }
            }
        }

        let l_adapt: Vec<f64> = r_lmsr[0]
            .iter()
            .zip(&r_lmsr[1])
            .map(|(l, m)| l + m)
            .collect();

        let p_l: Vec<f64> = r_lmsr[0].iter().map(|&v| pn.cone(v)).collect();
        let p_m: Vec<f64> = r_lmsr[1].iter().map(|&v| pn.cone(v)).collect();
        let p_r: Vec<f64> = r_lmsr[3].iter().map(|&v| pn.rod(v)).collect();

        let mut cones: Vec<f64> = p_l.iter().zip(&p_m).map(|(a, b)| a + b).collect();
        remove_mean(&mut cones);
        let mut rods = p_r;
        remove_mean(&mut rods);

        let achromatic: Vec<f64> = cones.iter().zip(&rods).map(|(a, b)| a + b).collect();

        Pathway {
            achromatic,
            l_adapt,
            width,
            height,
        }
    }

    pub fn surround_per_channel(
        nits: &[f64],
        channels: usize,
        configured: Option<f64>,
    ) -> Vec<f64> {
        if let Some(v) = configured {
            return vec![v; channels];
        }
        (0..channels)
            .map(|c| {
                let mut sum = 0.0;
                let mut count = 0usize;
                for p in nits.chunks_exact(channels) {
                    if p[c] > 0.0 {
                        sum += p[c].ln();
                        count += 1;
                    }
                }
                if count == 0 {
                    MIN_NITS
                } else {
                    (sum / count as f64).exp()
                }
            })
            .collect()
    }

    fn remove_mean(v: &mut [f64]) {
        if v.is_empty() {
            return;
        }
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        for x in v.iter_mut() {
            *x -= mean;
        }
    }

    // ── bands.rs ──────────────────────────────────────────────────────────

    pub struct BandPyramid {
        pub bands: Vec<Vec<Band>>,
        pub width: usize,
        pub height: usize,
    }

    impl BandPyramid {
        pub fn count(&self) -> usize {
            self.bands.len()
        }
        pub fn orientations(&self, b: usize) -> usize {
            self.bands[b].len()
        }
        pub fn total_planes(&self) -> usize {
            self.bands.iter().map(Vec::len).sum()
        }
        pub fn band(&self, b: usize, o: usize) -> &Band {
            let planes = &self.bands[b];
            &planes[o.min(planes.len() - 1)]
        }
        pub fn band_mut(&mut self, b: usize, o: usize) -> &mut Band {
            let planes = &mut self.bands[b];
            let i = o.min(planes.len() - 1);
            &mut planes[i]
        }
        pub fn frequencies(&self, pix_per_deg: f64) -> Vec<f64> {
            (0..self.count())
                .map(|b| 2f64.powi(-(b as i32)) * pix_per_deg / 2.0)
                .collect()
        }
        pub fn zeros_like(&self) -> Self {
            Self {
                bands: self
                    .bands
                    .iter()
                    .map(|planes| {
                        planes
                            .iter()
                            .map(|b| Band::zeros(b.width, b.height))
                            .collect()
                    })
                    .collect(),
                width: self.width,
                height: self.height,
            }
        }
    }

    pub fn decompose(
        pathway: &Pathway,
        par: &Params,
        bb_padvalue: Option<f64>,
    ) -> (BandPyramid, f64) {
        let pyr: SteerablePyramid = build(&pathway.achromatic, pathway.width, pathway.height, None);

        let mut bands: Vec<Vec<Band>> = Vec::with_capacity(pyr.height_levels() + 2);
        bands.push(vec![pyr.high_pass.clone()]);
        for level in &pyr.levels {
            bands.push(level.to_vec());
        }
        bands.push(vec![pyr.low_pass.clone()]);

        let l_mean = pathway.l_adapt.iter().sum::<f64>() / pathway.l_adapt.len() as f64;
        let last = bands.len() - 1;
        let bb = &bands[last][0];
        let pad = bb_padvalue
            .unwrap_or_else(|| bb.data.iter().sum::<f64>() / bb.data.len().max(1) as f64);

        let band_freq_last = 2f64.powi(-((bands.len() - 1) as i32)) * par.pix_per_deg / 2.0;
        let (pw, ph) = (bb.width * 2, bb.height * 2);
        let ppd_bb = band_freq_last * 2.0 * core::f64::consts::SQRT_2;
        let csf_bb: Vec<f64> = cycles_per_degree_grid(pw, ph, ppd_bb)
            .into_iter()
            .map(|rho| ncsf(rho, l_mean, par))
            .collect();
        let filtered = conv_fft_real(&bb.data, bb.width, bb.height, &csf_bb, pw, ph, pad);
        bands[last][0].data = filtered;

        (
            BandPyramid {
                bands,
                width: pathway.width,
                height: pathway.height,
            },
            pad,
        )
    }

    // ── masking.rs ────────────────────────────────────────────────────────

    const CSF_LUT_N: usize = 256;
    pub const DIFF_MASK_THRESHOLD: f64 = 0.001;

    pub fn sign_pow(x: f64, e: f64) -> f64 {
        x.signum() * x.abs().powf(e)
    }

    pub fn mutual_masking(test: &Band, reference: &Band) -> Band {
        debug_assert_eq!(
            (test.width, test.height),
            (reference.width, reference.height)
        );
        let (w, h) = (test.width, test.height);
        let m: Vec<f64> = test
            .data
            .iter()
            .zip(&reference.data)
            .map(|(t, r)| t.abs().min(r.abs()))
            .collect();
        Band {
            width: w,
            height: h,
            data: box3x3(&m, w, h),
        }
    }

    fn box3x3(src: &[f64], w: usize, h: usize) -> Vec<f64> {
        let mut out = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                for dy in -1isize..=1 {
                    let sy = y as isize + dy;
                    if sy < 0 || sy >= h as isize {
                        continue;
                    }
                    for dx in -1isize..=1 {
                        let sx = x as isize + dx;
                        if sx < 0 || sx >= w as isize {
                            continue;
                        }
                        acc += src[sy as usize * w + sx as usize];
                    }
                }
                out[y * w + x] = acc / 9.0;
            }
        }
        out
    }

    pub struct Masking {
        pub d_bands: BandPyramid,
        pub quality_terms: Vec<f64>,
    }

    pub fn masking_run(
        test: &BandPyramid,
        reference: &BandPyramid,
        l_adapt: &[f64],
        diff_mask: &[f64],
        par: &Params,
    ) -> Masking {
        assert_eq!(test.count(), reference.count(), "band count mismatch");
        let (w, h) = (test.width, test.height);
        assert_eq!(l_adapt.len(), w * h, "l_adapt does not match the image");
        assert_eq!(diff_mask.len(), w * h, "diff_mask does not match the image");

        let b_count = test.count();
        let total_planes = test.total_planes();
        let band_freq = test.frequencies(par.pix_per_deg);

        let csf_la: Vec<f64> = (0..CSF_LUT_N)
            .map(|i| 10f64.powf(-5.0 + 10.0 * i as f64 / (CSF_LUT_N - 1) as f64))
            .collect();
        let csf_log_la: Vec<f64> = csf_la.iter().map(|v| v.log10()).collect();
        let csf: Vec<Vec<f64>> = band_freq
            .iter()
            .map(|f| csf_la.iter().map(|la| ncsf(*f, *la, par)).collect())
            .collect();

        let log_la: Vec<f64> = l_adapt
            .iter()
            .map(|v| clamp(*v, csf_la[0], csf_la[CSF_LUT_N - 1]).log10())
            .collect();

        let qf: Vec<f64> = par.quality_band_freq.iter().rev().copied().collect();
        let qw: Vec<f64> = par.quality_band_w.iter().rev().copied().collect();

        let p = par.transducer_p();
        let q = par.transducer_q();
        let pf = par.transducer_pf();
        let k_self = 10f64.powf(par.mask_self);
        let k_xo = 10f64.powf(par.mask_xo);
        let k_xn = 10f64.powf(par.mask_xn);

        let mut d_bands = test.zeros_like();
        let mut quality_terms = Vec::with_capacity(total_planes);

        let mm: Vec<Vec<Band>> = (0..b_count)
            .map(|b| {
                (0..test.orientations(b))
                    .map(|o| mutual_masking(test.band(b, o), reference.band(b, o)))
                    .collect()
            })
            .collect();

        for b in 0..b_count {
            let (bw, bh) = (test.band(b, 0).width, test.band(b, 0).height);
            let band_norm = 2f64.powi(b as i32);

            let mut mask_xo_total = vec![0.0; bw * bh];
            for plane in &mm[b] {
                for (a, v) in mask_xo_total.iter_mut().zip(&plane.data) {
                    *a += v;
                }
            }

            let log_la_rs = imresize(&log_la, w, h, bw, bh);
            let csf_b: Vec<f64> = log_la_rs
                .iter()
                .map(|l| {
                    let l = clamp(*l, csf_log_la[0], csf_log_la[CSF_LUT_N - 1]);
                    interp1_linear(&csf_log_la, &csf[b], l)
                })
                .collect();

            let f_b = clamp(band_freq[b], qf[0], qf[qf.len() - 1]);
            let w_f = interp1_linear(&qf, &qw, f_b);
            let diff_mask_b = imresize(diff_mask, w, h, bw, bh);

            for o in 0..test.orientations(b) {
                let t = test.band(b, o);
                let r = reference.band(b, o);
                let self_mask = &mm[b][o];

                let mut mask_xn = vec![0.0; bw * bh];
                if b > 0 {
                    let src = &mm[b - 1][o.min(mm[b - 1].len() - 1)];
                    let rs = imresize(&src.data, src.width, src.height, bw, bh);
                    for (a, v) in mask_xn.iter_mut().zip(&rs) {
                        *a += v.max(0.0) / (band_norm / 2.0);
                    }
                }
                if b + 2 < b_count {
                    let src = &mm[b + 1][o.min(mm[b + 1].len() - 1)];
                    let rs = imresize(&src.data, src.width, src.height, bw, bh);
                    for (a, v) in mask_xn.iter_mut().zip(&rs) {
                        *a += v.max(0.0) / (band_norm * 2.0);
                    }
                }

                let mut d = vec![0.0; bw * bh];
                for i in 0..bw * bh {
                    let band_diff = t.data[i] - r.data[i];
                    let ex_diff = sign_pow(band_diff / band_norm, p) * band_norm;

                    let n_ncsf = if b == b_count - 1 {
                        1.0
                    } else {
                        1.0 / csf_b[i]
                    };

                    d[i] = if par.do_masking {
                        let sm = self_mask.data[i];
                        let xo = (mask_xo_total[i] - sm).max(0.0);
                        let n_mask = band_norm
                            * (k_self * (sm / n_ncsf / band_norm).powf(q)
                                + k_xo * (xo / n_ncsf / band_norm).powf(q)
                                + k_xn * (mask_xn[i] / n_ncsf).powf(q));
                        ex_diff / (n_ncsf.powf(2.0 * p) + n_mask * n_mask).sqrt()
                    } else {
                        ex_diff / n_ncsf.powf(p)
                    };
                }

                let msre = {
                    let s: f64 = d
                        .iter()
                        .zip(&diff_mask_b)
                        .map(|(v, m)| (v * m) * (v * m))
                        .sum();
                    s.sqrt() / (bw * bh) as f64
                };
                quality_terms.push((msre + 1e-12).ln() * w_f / total_planes as f64);

                let out = d_bands.band_mut(b, o);
                for (dst, v) in out.data.iter_mut().zip(&d) {
                    *dst = sign_pow(v / band_norm, pf) * band_norm;
                }
            }
        }

        Masking {
            d_bands,
            quality_terms,
        }
    }

    pub fn diff_mask(test: &[f64], reference: &[f64], channels: usize) -> Vec<f64> {
        assert_eq!(test.len(), reference.len());
        assert!(channels >= 1);
        test.chunks(channels)
            .zip(reference.chunks(channels))
            .map(|(t, r)| {
                let any = t
                    .iter()
                    .zip(r)
                    .any(|(a, b)| ((a - b) / b).abs() > DIFF_MASK_THRESHOLD);
                f64::from(any)
            })
            .collect()
    }

    // ── pool.rs ───────────────────────────────────────────────────────────

    pub struct Visibility {
        pub p_map: Vec<f64>,
        pub p_det: f64,
        pub c_map: Vec<f64>,
        pub c_max: f64,
        // Kept so the copy stays verbatim; the harness compares via the maps.
        #[allow(dead_code)]
        pub width: usize,
        #[allow(dead_code)]
        pub height: usize,
    }

    pub fn to_steerable(bp: &BandPyramid) -> SteerablePyramid {
        assert!(
            bp.count() >= 2,
            "band list needs a high-pass and a low-pass"
        );
        let high_pass = bp.bands[0][0].clone();
        let low_pass = bp.bands[bp.count() - 1][0].clone();
        let levels: Vec<[Band; ORIENTATIONS]> = bp.bands[1..bp.count() - 1]
            .iter()
            .map(|planes| {
                let v: Vec<Band> = planes.clone();
                v.try_into()
                    .unwrap_or_else(|_| panic!("oriented band must carry {ORIENTATIONS} planes"))
            })
            .collect();
        SteerablePyramid {
            high_pass,
            levels,
            low_pass,
            width: bp.width,
            height: bp.height,
        }
    }

    pub fn visibility(d_bands: &BandPyramid, par: &Params) -> Visibility {
        let pyr = to_steerable(d_bands);
        let rec = reconstruct(&pyr);
        let mut s_map: Vec<f64> = rec.data.iter().map(|v| v.abs()).collect();

        if par.do_spatial_pooling {
            let sum: f64 = s_map.iter().sum();
            let max = s_map.iter().fold(0.0f64, |a, b| a.max(*b));
            let k = sum / (max + 1e-12);
            for v in &mut s_map {
                *v *= k;
            }
        }

        let ln_half = 0.5f64.ln();
        let p_map: Vec<f64> = s_map.iter().map(|c| 1.0 - (ln_half * c).exp()).collect();
        let p_det = p_map.iter().fold(0.0f64, |a, b| a.max(*b));
        let c_max = s_map.iter().fold(0.0f64, |a, b| a.max(*b));

        Visibility {
            p_map,
            p_det,
            c_map: s_map,
            c_max,
            width: rec.width,
            height: rec.height,
        }
    }

    pub fn quality_correlate(terms: &[f64]) -> f64 {
        terms.iter().sum()
    }

    pub fn quality_mos(q: f64, par: &Params) -> f64 {
        100.0 / (1.0 + (par.quality_logistic_q1 * (q + par.quality_logistic_q2)).exp())
    }

    // ── metric.rs glue ────────────────────────────────────────────────────

    pub fn hdrvdp(
        test: &[f64],
        reference: &[f64],
        width: usize,
        height: usize,
        encoding: ColorEncoding,
        par: &Params,
    ) -> Result<HdrVdpResult, Error> {
        if !(par.pix_per_deg.is_finite() && par.pix_per_deg > 0.0) {
            return Err(Error::InvalidResolution(par.pix_per_deg));
        }
        if test.len() != reference.len() {
            return Err(Error::SizeMismatch {
                reference: (width, height),
                distorted: (width, height),
            });
        }
        let channels = encoding.channels();

        let ref_nits = to_nits(reference, width, height, encoding)?;
        let test_nits = to_nits(test, width, height, encoding)?;

        let surround = surround_per_channel(&ref_nits, channels, par.surround_l);

        // Spectral setup is shared with production (documented scope cut).
        let lmsr = lmsr_matrix(&emission_spectra(encoding.spectra(), channels));
        let pn = Photoreceptor::new(par);

        let path_ref = visual_pathway(&ref_nits, width, height, par, &pn, &lmsr, &surround);
        let path_test = visual_pathway(&test_nits, width, height, par, &pn, &lmsr, &surround);

        let (bands_ref, pad) = decompose(&path_ref, par, None);
        let (bands_test, _) = decompose(&path_test, par, Some(pad));

        let l_adapt: Vec<f64> = path_ref
            .l_adapt
            .iter()
            .zip(&path_test.l_adapt)
            .map(|(a, b)| 0.5 * (a + b))
            .collect();
        let dm = diff_mask(&test_nits, &ref_nits, channels);

        let m = masking_run(&bands_test, &bands_ref, &l_adapt, &dm, par);
        let vis = visibility(&m.d_bands, par);

        let q = quality_correlate(&m.quality_terms);
        Ok(HdrVdpResult {
            p_map: vis.p_map,
            p_det: vis.p_det,
            c_map: vis.c_map,
            c_max: vis.c_max,
            q,
            q_mos: quality_mos(q, par),
            width,
            height,
            input_looks_relative: looks_relative(&ref_nits, channels, encoding),
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Harness
// ═══════════════════════════════════════════════════════════════════════════

/// Bitwise slice equality with a first-divergence report.
#[track_caller]
fn assert_bits_eq(name: &str, got: &[f64], want: &[f64]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{name}: length {} vs {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "{name}[{i}]: {g:e} ({:#018x}) != {w:e} ({:#018x}) — production diverged from the \
             frozen reference by {} ULP-ish; a metric output changed",
            g.to_bits(),
            w.to_bits(),
            (*g - *w).abs()
        );
    }
}

#[track_caller]
fn assert_bit_eq(name: &str, got: f64, want: f64) {
    assert!(
        got.to_bits() == want.to_bits(),
        "{name}: {got:e} ({:#018x}) != {want:e} ({:#018x})",
        got.to_bits(),
        want.to_bits()
    );
}

/// Deterministic pseudo-random stream (xorshift64*-ish), dependency-free.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A structured HDR luminance field: textured base + noise + specular blob,
/// spanning ~0.05 to ~3000 cd/m².
fn hdr_field(w: usize, h: usize, seed: u64) -> Vec<f64> {
    let pi = core::f64::consts::PI;
    let mut rng = Rng(seed | 1);
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            let base = 60.0
                * (1.0 + 0.4 * (2.0 * pi * x / 17.0).sin() * (2.0 * pi * y / 23.0).cos())
                + 6.0 * (rng.next() - 0.5);
            let (cx, cy) = (w as f64 * 0.7, h as f64 * 0.3);
            let r2 = ((x - cx).powi(2) + (y - cy).powi(2)) / (0.02 * (w * h) as f64 + 1.0);
            (base + 3000.0 * (-r2).exp()).max(0.05)
        })
        .collect()
}

/// A distorted counterpart: multiplicative grating + a local additive dent.
fn distorted(reference: &[f64], w: usize, _h: usize, amp: f64) -> Vec<f64> {
    let pi = core::f64::consts::PI;
    reference
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            let g = 1.0 + amp * (2.0 * pi * x / 5.0).sin() * (2.0 * pi * y / 7.0).sin();
            let dent = if i % 97 == 0 { 0.98 } else { 1.0 };
            v * g * dent
        })
        .collect()
}

/// Code values in [0, 1] for the display-model encodings.
fn code_field(len: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed | 1);
    let pi = core::f64::consts::PI;
    (0..len)
        .map(|i| {
            let s = 0.5 + 0.35 * (2.0 * pi * (i as f64) / 41.0).sin() + 0.1 * (rng.next() - 0.5);
            s.clamp(0.0, 1.0)
        })
        .collect()
}

// ── L1: shared constant tables (pure literals → portable golden hash) ──────

/// FNV-1a-64 over the little-endian bit patterns of an f64 stream.
fn fnv1a(values: impl IntoIterator<Item = f64>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in values {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The shared constant tables both paths read. Dual-path cannot catch a change
/// to these (both sides would agree on the new wrong value), so their exact bit
/// patterns are pinned here. Every value hashed is a literal or a product of
/// literals — no libm — so the hash is portable across platforms.
#[test]
fn constant_tables_are_frozen() {
    use hdrvdp::sp3_filters::{BFILTS, HI0FILT, LO0FILT, LOFILT};

    let taps = fnv1a(
        HI0FILT
            .iter()
            .flatten()
            .chain(LO0FILT.iter().flatten())
            .chain(LOFILT.iter().flatten())
            .chain(BFILTS.iter().flatten().flatten())
            .copied(),
    );
    assert_eq!(
        taps, 0x9af1_d49c_1898_755b,
        "sp3 filter taps changed ({taps:#018x}) — every pyramid output moves; if deliberate, \
         re-derive this hash and say so in the commit"
    );

    let par = Params::default();
    let mut stream: Vec<f64> = Vec::new();
    stream.extend_from_slice(&par.mtf_params_a);
    stream.extend_from_slice(&par.mtf_params_b);
    for row in &par.csf_params {
        stream.extend_from_slice(row);
    }
    stream.extend_from_slice(&par.csf_lums);
    stream.extend_from_slice(&par.csf_sa);
    stream.extend_from_slice(&par.csf_sr_par);
    stream.extend_from_slice(&[
        par.rod_sensitivity,
        par.mask_p,
        par.mask_self,
        par.mask_xo,
        par.mask_xn,
        par.mask_q,
    ]);
    stream.extend_from_slice(&par.quality_band_freq);
    stream.extend_from_slice(&par.quality_band_w);
    stream.extend_from_slice(&[par.quality_logistic_q1, par.quality_logistic_q2]);
    // Deliberately NOT hashed: sensitivity_correction and psych_func_slope
    // (both go through libm powf/log10 → not portable) — they are locked by
    // the dual-path end-to-end tests instead.
    let cal = fnv1a(stream);
    assert_eq!(
        cal, 0x6a35_1635_bdc5_1b0d,
        "Params calibration constants changed ({cal:#018x}) — the metric is recalibrated; if \
         deliberate, re-derive this hash and say so in the commit"
    );
}

// ── L2: kernel-level dual-path locks ───────────────────────────────────────

#[test]
fn lock_fft_paths() {
    for n in [1usize, 2, 4, 8, 64, 256, 3, 5, 7, 12, 15, 33, 100, 122, 194] {
        let mut rng = Rng(0x5eed + n as u64);
        let src: Vec<(f64, f64)> = (0..n)
            .map(|_| (rng.next() - 0.5, rng.next() - 0.5))
            .collect();

        let mut a: Vec<hdrvdp::fft::Complex> = src
            .iter()
            .map(|&(re, im)| hdrvdp::fft::Complex::new(re, im))
            .collect();
        let mut b: Vec<reference::Complex> = src
            .iter()
            .map(|&(re, im)| reference::Complex::new(re, im))
            .collect();

        hdrvdp::fft::fft(&mut a);
        reference::fft(&mut b);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_bit_eq(&format!("fft(n={n}).re[{i}]"), x.re, y.re);
            assert_bit_eq(&format!("fft(n={n}).im[{i}]"), x.im, y.im);
        }

        hdrvdp::fft::ifft(&mut a);
        reference::ifft(&mut b);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_bit_eq(&format!("ifft(n={n}).re[{i}]"), x.re, y.re);
            assert_bit_eq(&format!("ifft(n={n}).im[{i}]"), x.im, y.im);
        }
    }
}

#[test]
fn lock_fft2_and_conv() {
    for (w, h) in [(4usize, 4usize), (8, 5), (13, 7), (32, 16), (24, 20)] {
        let mut rng = Rng(0xf00d + (w * h) as u64);
        let src: Vec<(f64, f64)> = (0..w * h)
            .map(|_| (rng.next() - 0.5, rng.next() - 0.5))
            .collect();
        let mut a: Vec<hdrvdp::fft::Complex> = src
            .iter()
            .map(|&(re, im)| hdrvdp::fft::Complex::new(re, im))
            .collect();
        let mut b: Vec<reference::Complex> = src
            .iter()
            .map(|&(re, im)| reference::Complex::new(re, im))
            .collect();
        hdrvdp::fft::fft2(&mut a, w, h);
        reference::fft2(&mut b, w, h);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_bit_eq(&format!("fft2({w}x{h}).re[{i}]"), x.re, y.re);
            assert_bit_eq(&format!("fft2({w}x{h}).im[{i}]"), x.im, y.im);
        }
        hdrvdp::fft::ifft2(&mut a, w, h);
        reference::ifft2(&mut b, w, h);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_bit_eq(&format!("ifft2({w}x{h}).re[{i}]"), x.re, y.re);
            assert_bit_eq(&format!("ifft2({w}x{h}).im[{i}]"), x.im, y.im);
        }
    }

    // conv_fft_real over even and odd (Bluestein) padded sizes, non-trivial
    // pad values, and a filter with structure.
    for (w, h) in [(12usize, 10usize), (9, 7), (16, 16)] {
        let (pw, ph) = (2 * w, 2 * h);
        let im = hdr_field(w, h, 77);
        let ppd = 21.3;
        let filt_a: Vec<f64> = hdrvdp::fft::cycles_per_degree_grid(pw, ph, ppd)
            .into_iter()
            .map(|rho| 1.0 / (1.0 + rho))
            .collect();
        let filt_b: Vec<f64> = reference::cycles_per_degree_grid(pw, ph, ppd)
            .into_iter()
            .map(|rho| 1.0 / (1.0 + rho))
            .collect();
        assert_bits_eq(&format!("cycdeg({pw}x{ph})"), &filt_a, &filt_b);
        let got = hdrvdp::fft::conv_fft_real(&im, w, h, &filt_a, pw, ph, 0.125);
        let want = reference::conv_fft_real(&im, w, h, &filt_b, pw, ph, 0.125);
        assert_bits_eq(&format!("conv_fft_real({w}x{h})"), &got, &want);
    }
}

#[test]
fn lock_corr_dn_and_up_conv() {
    use hdrvdp::sp3_filters::{BFILTS, HI0FILT, LO0FILT, LOFILT};
    let hi0: Vec<f64> = HI0FILT.iter().flatten().copied().collect();
    let lo0: Vec<f64> = LO0FILT.iter().flatten().copied().collect();
    let lo: Vec<f64> = LOFILT.iter().flatten().copied().collect();
    let b1: Vec<f64> = BFILTS[1].iter().flatten().copied().collect();
    let b3: Vec<f64> = BFILTS[3].iter().flatten().copied().collect();

    let cases: [(&str, &Vec<f64>, usize, usize); 5] = [
        ("hi0/15/1", &hi0, 15, 1),
        ("lo0/9/1", &lo0, 9, 1),
        ("lofilt/17/2", &lo, 17, 2),
        ("bfilt1/9/1", &b1, 9, 1),
        ("bfilt3/9/1", &b3, 9, 1),
    ];

    for (w, h) in [
        (7usize, 5usize),
        (16, 16),
        (33, 17),
        (1, 9),
        (64, 48),
        (17, 1),
    ] {
        let im = hdr_field(w, h, (w * 31 + h) as u64);
        for (name, filt, fs, step) in &cases {
            let got = hdrvdp::spyr::corr_dn(&im, w, h, filt, *fs, *step);
            let want = reference::corr_dn(&im, w, h, filt, *fs, *step);
            assert_eq!((got.width, got.height), (want.width, want.height), "{name}");
            assert_bits_eq(&format!("corr_dn[{name}]({w}x{h})"), &got.data, &want.data);

            // up_conv accumulates into a non-zero res, from the band we just
            // made — sizes then satisfy the div_ceil contract by construction.
            let band = want.clone();
            let seed_res: Vec<f64> = (0..w * h).map(|i| (i as f64) * 1e-3 - 0.7).collect();
            let mut res_got = seed_res.clone();
            let mut res_want = seed_res;
            hdrvdp::spyr::up_conv(&band, filt, *fs, *step, w, h, &mut res_got);
            reference::up_conv(&band, filt, *fs, *step, w, h, &mut res_want);
            assert_bits_eq(&format!("up_conv[{name}]({w}x{h})"), &res_got, &res_want);
        }
        // Delta-ish asymmetric filter, step 3: exercises the generic path.
        let mut f = vec![0.0; 25];
        f[7] = 0.5;
        f[12] = 1.0;
        f[13] = -0.25;
        let got = hdrvdp::spyr::corr_dn(&im, w, h, &f, 5, 3);
        let want = reference::corr_dn(&im, w, h, &f, 5, 3);
        assert_bits_eq(
            &format!("corr_dn[custom/5/3]({w}x{h})"),
            &got.data,
            &want.data,
        );
        let seed_res: Vec<f64> = vec![0.25; w * h];
        let mut res_got = seed_res.clone();
        let mut res_want = seed_res;
        hdrvdp::spyr::up_conv(&want, &f, 5, 3, w, h, &mut res_got);
        reference::up_conv(&want, &f, 5, 3, w, h, &mut res_want);
        assert_bits_eq(
            &format!("up_conv[custom/5/3]({w}x{h})"),
            &res_got,
            &res_want,
        );
    }
}

#[test]
fn lock_pyramid_build_and_reconstruct() {
    for (w, h) in [(32usize, 24usize), (97, 61), (64, 64)] {
        let im = hdr_field(w, h, 0xbead + w as u64);
        let got = hdrvdp::spyr::build(&im, w, h, None);
        let want = reference::build(&im, w, h, None);
        assert_eq!(got.height_levels(), want.height_levels(), "{w}x{h} levels");
        assert_bits_eq(
            &format!("build({w}x{h}).high"),
            &got.high_pass.data,
            &want.high_pass.data,
        );
        assert_bits_eq(
            &format!("build({w}x{h}).low"),
            &got.low_pass.data,
            &want.low_pass.data,
        );
        for (l, (gl, wl)) in got.levels.iter().zip(&want.levels).enumerate() {
            for (o, (gb, wb)) in gl.iter().zip(wl).enumerate() {
                assert_bits_eq(&format!("build({w}x{h}).l{l}o{o}"), &gb.data, &wb.data);
            }
        }
        let rec_got = hdrvdp::spyr::reconstruct(&got);
        let rec_want = reference::reconstruct(&want);
        assert_bits_eq(
            &format!("reconstruct({w}x{h})"),
            &rec_got.data,
            &rec_want.data,
        );
    }
}

#[test]
fn lock_imresize() {
    for (w, h, ow, oh) in [
        (7usize, 5usize, 3usize, 2usize),
        (4, 4, 16, 16),
        (16, 9, 5, 12),
        (1, 1, 8, 8),
        (33, 17, 33, 17),
        (64, 48, 32, 24),
        (97, 61, 25, 16),
        (12, 90, 47, 13),
    ] {
        let im = hdr_field(w, h, (w * 7 + h * 3) as u64);
        let got = hdrvdp::resize::imresize(&im, w, h, ow, oh);
        let want = reference::imresize(&im, w, h, ow, oh);
        assert_bits_eq(&format!("imresize({w}x{h}->{ow}x{oh})"), &got, &want);
    }
}

#[test]
fn lock_pointwise_helpers() {
    // interp / clamp / point_op / cumtrapz on stressed grids.
    let xs: Vec<f64> = (0..64).map(|i| -3.0 + 0.11 * i as f64).collect();
    let mut rng = Rng(42);
    let ys: Vec<f64> = (0..64).map(|_| rng.next() * 5.0 - 1.0).collect();
    for q in [-10.0, -3.0, -2.95, 0.0, 1.234, 3.6, 3.93, 50.0, f64::NAN] {
        let g = hdrvdp::interp::interp1_linear(&xs, &ys, hdrvdp::interp::clamp(q, xs[0], xs[63]));
        let w = reference::interp1_linear(&xs, &ys, reference::clamp(q, xs[0], xs[63]));
        assert_bit_eq(&format!("interp1_linear({q})"), g, w);
        let g = hdrvdp::interp::point_op(&ys, -3.0, 0.11, q);
        let w = reference::point_op(&ys, -3.0, 0.11, q);
        assert_bit_eq(&format!("point_op({q})"), g, w);
    }
    assert_bits_eq(
        "cumtrapz",
        &hdrvdp::interp::cumtrapz(&xs, &ys),
        &reference::cumtrapz(&xs, &ys),
    );

    // csf family over a (rho, luminance) grid.
    let par = Params::new(30.0);
    for rho in [0.0, 0.1, 0.9375, 3.75, 7.5, 15.0, 31.0, 60.0] {
        assert_bit_eq(
            &format!("mtf({rho})"),
            hdrvdp::csf::mtf(rho, &par),
            reference::mtf(rho, &par),
        );
        for lum in [1e-6, 2e-3, 0.02, 1.0, 42.0, 150.0, 9e4] {
            assert_bit_eq(
                &format!("ncsf({rho},{lum})"),
                hdrvdp::csf::ncsf(rho, lum, &par),
                reference::ncsf(rho, lum, &par),
            );
        }
    }
    for la in [1e-5, 1e-3, 0.5, 1.1732, 20.0, 1e4] {
        assert_bit_eq(
            &format!("joint_rod_cone_sens({la})"),
            hdrvdp::csf::joint_rod_cone_sens(la, &par),
            reference::joint_rod_cone_sens(la, &par),
        );
        assert_bit_eq(
            &format!("rod_sens({la})"),
            hdrvdp::csf::rod_sens(la, &par),
            reference::rod_sens(la, &par),
        );
    }

    // sign_pow + quality_mos.
    for x in [-8.0, -0.3, 0.0, 1e-9, 0.7, 5.0] {
        for e in [0.25, 1.0, 2.7] {
            assert_bit_eq(
                &format!("sign_pow({x},{e})"),
                hdrvdp::masking::sign_pow(x, e),
                reference::sign_pow(x, e),
            );
        }
    }
    for q in [-30.0, -2.0, -0.8886, 0.0, 4.0] {
        assert_bit_eq(
            &format!("quality_mos({q})"),
            hdrvdp::pool::quality_mos(q, &par),
            reference::quality_mos(q, &par),
        );
    }
}

#[test]
fn lock_photoreceptor_tables() {
    for par in [
        Params::new(30.0),
        Params::new(30.0).with_peak_sensitivity(2.7),
    ] {
        let got = hdrvdp::photoreceptor::Photoreceptor::new(&par);
        let want = reference::Photoreceptor::new(&par);
        let (glog, gcone) = got.cone_table();
        let (_, grod) = got.rod_table();
        assert_bits_eq("pn.log_lum", glog, &want.log_lum);
        assert_bits_eq("pn.jnd_cone", gcone, &want.jnd_cone);
        assert_bits_eq("pn.jnd_rod", grod, &want.jnd_rod);
        for lum in [
            0.0,
            1e-7,
            1e-5,
            0.004,
            0.9,
            88.0,
            4000.0,
            1e5,
            1e9,
            f64::NAN,
        ] {
            assert_bit_eq(&format!("pn.cone({lum})"), got.cone(lum), want.cone(lum));
            assert_bit_eq(&format!("pn.rod({lum})"), got.rod(lum), want.rod(lum));
        }
    }
}

#[test]
fn lock_display_and_masking_primitives() {
    // to_nits across all encodings.
    let (w, h) = (13usize, 9usize);
    let lum = hdr_field(w, h, 5);
    let code1 = code_field(w * h, 6);
    let code3 = code_field(w * h * 3, 7);
    let rgb: Vec<f64> = hdr_field(w * 3, h, 8);
    for (name, pixels, enc) in [
        ("Luminance", &lum, ColorEncoding::Luminance),
        ("LumaDisplay", &code1, ColorEncoding::LumaDisplay),
        ("SrgbDisplay", &code3, ColorEncoding::SrgbDisplay),
        ("RgbBt709", &rgb, ColorEncoding::RgbBt709),
        ("Xyz", &rgb, ColorEncoding::Xyz),
    ] {
        let got = hdrvdp::display::to_nits(pixels, w, h, enc).unwrap();
        let want = reference::to_nits(pixels, w, h, enc).unwrap();
        assert_bits_eq(&format!("to_nits[{name}]"), &got, &want);
        assert_eq!(
            hdrvdp::display::looks_relative(&got, enc.channels(), enc),
            reference::looks_relative(&want, enc.channels(), enc),
            "looks_relative[{name}]"
        );
    }

    // mutual_masking (min + 3×3 box) on assorted sizes incl. 1-wide.
    for (w, h) in [(3usize, 3usize), (16, 12), (33, 1), (1, 8), (40, 27)] {
        let a = Band {
            width: w,
            height: h,
            data: hdr_field(w, h, 21).iter().map(|v| v - 60.0).collect(),
        };
        let b = Band {
            width: w,
            height: h,
            data: hdr_field(w, h, 22).iter().map(|v| 55.0 - v).collect(),
        };
        let got = hdrvdp::masking::mutual_masking(&a, &b);
        let want = reference::mutual_masking(&a, &b);
        assert_bits_eq(&format!("mutual_masking({w}x{h})"), &got.data, &want.data);
    }

    // diff_mask, 1 and 3 channels.
    let r = hdr_field(11, 7, 31);
    let t = distorted(&r, 11, 7, 0.004);
    assert_bits_eq(
        "diff_mask/1ch",
        &hdrvdp::masking::diff_mask(&t, &r, 1),
        &reference::diff_mask(&t, &r, 1),
    );
    let r3 = hdr_field(33, 7, 32);
    let t3 = distorted(&r3, 33, 7, 0.004);
    assert_bits_eq(
        "diff_mask/3ch",
        &hdrvdp::masking::diff_mask(&t3, &r3, 3),
        &reference::diff_mask(&t3, &r3, 3),
    );

    // surround_per_channel: configured and geometric-mean (incl. a zero pixel).
    let mut nits3 = hdr_field(10 * 3, 6, 33);
    nits3[4] = 0.0;
    for cfg in [Some(0.25), None] {
        assert_bits_eq(
            &format!("surround({cfg:?})"),
            &hdrvdp::pathway::surround_per_channel(&nits3, 3, cfg),
            &reference::surround_per_channel(&nits3, 3, cfg),
        );
    }
}

// ── L3: stage-composition and end-to-end dual-path locks ───────────────────

fn assert_pathway_eq(name: &str, got: &hdrvdp::pathway::Pathway, want: &reference::Pathway) {
    assert_bits_eq(
        &format!("{name}.achromatic"),
        &got.achromatic,
        &want.achromatic,
    );
    assert_bits_eq(&format!("{name}.l_adapt"), &got.l_adapt, &want.l_adapt);
}

#[test]
fn lock_visual_pathway_and_decompose() {
    use hdrvdp::spectral::{DisplaySpectra, emission_spectra, lmsr_matrix};
    for (w, h, mtf_on) in [(16usize, 12usize, true), (33, 21, true), (16, 12, false)] {
        let mut par = Params::new(27.5);
        par.do_mtf = mtf_on;
        let pn_got = hdrvdp::photoreceptor::Photoreceptor::new(&par);
        let pn_want = reference::Photoreceptor::new(&par);
        for channels in [1usize, 3] {
            let spectra = if channels == 1 {
                DisplaySpectra::D65
            } else {
                DisplaySpectra::CcflLcd
            };
            let lmsr = lmsr_matrix(&emission_spectra(spectra, channels));
            let nits = hdr_field(w * channels, h, 0xace + (w + channels) as u64);
            let surround = reference::surround_per_channel(&nits, channels, None);
            let got = hdrvdp::pathway::visual_pathway(&nits, w, h, &par, &pn_got, &lmsr, &surround);
            let want = reference::visual_pathway(&nits, w, h, &par, &pn_want, &lmsr, &surround);
            let name = format!("pathway({w}x{h},ch{channels},mtf={mtf_on})");
            assert_pathway_eq(&name, &got, &want);

            // decompose on top of the same pathway, both pad-value modes.
            if w >= 17 && h >= 17 {
                for pad in [None, Some(0.037)] {
                    let (bp_got, pad_got) = hdrvdp::bands::decompose(&got, &par, pad);
                    let want_path = reference::Pathway {
                        achromatic: want.achromatic.clone(),
                        l_adapt: want.l_adapt.clone(),
                        width: w,
                        height: h,
                    };
                    let (bp_want, pad_want) = reference::decompose(&want_path, &par, pad);
                    assert_bit_eq(&format!("{name}.pad"), pad_got, pad_want);
                    assert_eq!(bp_got.count(), bp_want.count());
                    for b in 0..bp_got.count() {
                        for o in 0..bp_got.orientations(b) {
                            assert_bits_eq(
                                &format!("{name}.decomp[b{b}o{o}]"),
                                &bp_got.band(b, o).data,
                                &bp_want.band(b, o).data,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// End-to-end lock: every field of `HdrVdpResult`, bit for bit, across sizes
/// (even, odd/Bluestein), encodings, surround modes, and `do_*` toggles.
#[test]
fn lock_end_to_end() {
    struct Case {
        name: &'static str,
        w: usize,
        h: usize,
        enc: ColorEncoding,
        ppd: f64,
        amp: f64,
        tweak: fn(&mut Params),
    }
    let cases = [
        Case {
            name: "lum-32x24",
            w: 32,
            h: 24,
            enc: ColorEncoding::Luminance,
            ppd: 30.0,
            amp: 0.05,
            tweak: |_| {},
        },
        Case {
            name: "lum-97x61-geo-surround",
            w: 97,
            h: 61,
            enc: ColorEncoding::Luminance,
            ppd: 21.7,
            amp: 0.03,
            tweak: |p| p.surround_l = None,
        },
        Case {
            name: "srgb-96x72",
            w: 96,
            h: 72,
            enc: ColorEncoding::SrgbDisplay,
            ppd: 30.0,
            amp: 0.15,
            tweak: |_| {},
        },
        Case {
            name: "xyz-48x40",
            w: 48,
            h: 40,
            enc: ColorEncoding::Xyz,
            // Low ppd: the interleaved-index grating lands at ~1/3 the spatial
            // period, and at 45 ppd that aliases into a CSF-dead band and the
            // vacuity check below trips. 14 ppd keeps it visible.
            ppd: 14.0,
            amp: 0.25,
            tweak: |_| {},
        },
        Case {
            name: "luma-33x29",
            w: 33,
            h: 29,
            enc: ColorEncoding::LumaDisplay,
            ppd: 30.0,
            amp: 0.15,
            tweak: |_| {},
        },
        Case {
            name: "lum-32x24-no-mtf",
            w: 32,
            h: 24,
            enc: ColorEncoding::Luminance,
            ppd: 30.0,
            amp: 0.05,
            tweak: |p| p.do_mtf = false,
        },
        Case {
            name: "lum-32x24-no-masking",
            w: 32,
            h: 24,
            enc: ColorEncoding::Luminance,
            ppd: 30.0,
            amp: 0.05,
            tweak: |p| p.do_masking = false,
        },
        Case {
            name: "lum-32x24-no-pooling",
            w: 32,
            h: 24,
            enc: ColorEncoding::Luminance,
            ppd: 30.0,
            amp: 0.05,
            tweak: |p| p.do_spatial_pooling = false,
        },
    ];

    for c in cases {
        let mut par = Params::new(c.ppd);
        (c.tweak)(&mut par);
        let ch = c.enc.channels();
        let (reference_im, test_im): (Vec<f64>, Vec<f64>) = match c.enc {
            ColorEncoding::Luminance => {
                let r = hdr_field(c.w, c.h, 0xd00d + c.w as u64);
                let t = distorted(&r, c.w, c.h, c.amp);
                (r, t)
            }
            ColorEncoding::RgbBt709 | ColorEncoding::Xyz => {
                let r = hdr_field(c.w * ch, c.h, 0xd00d + c.w as u64);
                let t = distorted(&r, c.w * ch, c.h, c.amp);
                (r, t)
            }
            ColorEncoding::LumaDisplay | ColorEncoding::SrgbDisplay => {
                let r = code_field(c.w * c.h * ch, 0xd00d + c.w as u64);
                let t: Vec<f64> = distorted(&r, c.w * ch, c.h, c.amp)
                    .into_iter()
                    .map(|v| v.clamp(0.0, 1.0))
                    .collect();
                (r, t)
            }
        };

        let got = hdrvdp::hdrvdp(&test_im, &reference_im, c.w, c.h, c.enc, &par)
            .unwrap_or_else(|e| panic!("{}: production path failed: {e}", c.name));
        let want = reference::hdrvdp(&test_im, &reference_im, c.w, c.h, c.enc, &par)
            .unwrap_or_else(|e| panic!("{}: reference path failed: {e}", c.name));

        assert_bit_eq(&format!("{}: q", c.name), got.q, want.q);
        assert_bit_eq(&format!("{}: q_mos", c.name), got.q_mos, want.q_mos);
        assert_bit_eq(&format!("{}: p_det", c.name), got.p_det, want.p_det);
        assert_bit_eq(&format!("{}: c_max", c.name), got.c_max, want.c_max);
        assert_bits_eq(&format!("{}: p_map", c.name), &got.p_map, &want.p_map);
        assert_bits_eq(&format!("{}: c_map", c.name), &got.c_map, &want.c_map);
        assert_eq!((got.width, got.height), (c.w, c.h), "{}", c.name);
        assert_eq!(
            got.input_looks_relative, want.input_looks_relative,
            "{}: input_looks_relative",
            c.name
        );

        // The distortion must actually register, or this lock is vacuous:
        // an all-zero p_map would bit-match trivially.
        assert!(
            want.p_det > 1e-6,
            "{}: fixture produced no visible difference (p_det = {}) — lock is vacuous",
            c.name,
            want.p_det
        );
        assert!(
            want.q_mos < 99.99,
            "{}: fixture scored as identical (q_mos = {}) — lock is vacuous",
            c.name,
            want.q_mos
        );
    }
}

/// The masking band loop and visibility pooling, locked stage-direct (the
/// end-to-end test covers them too; this isolates a failure to the stage).
#[test]
fn lock_masking_and_visibility() {
    let (w, h) = (64usize, 48usize);
    let par = Params::new(30.0);
    use hdrvdp::spectral::{DisplaySpectra, emission_spectra, lmsr_matrix};
    let lmsr = lmsr_matrix(&emission_spectra(DisplaySpectra::D65, 1));
    let pn_got = hdrvdp::photoreceptor::Photoreceptor::new(&par);
    let pn_want = reference::Photoreceptor::new(&par);

    let r_im = hdr_field(w, h, 3);
    let t_im = distorted(&r_im, w, h, 0.08);
    let surround = reference::surround_per_channel(&r_im, 1, par.surround_l);

    let path_r_got = hdrvdp::pathway::visual_pathway(&r_im, w, h, &par, &pn_got, &lmsr, &surround);
    let path_t_got = hdrvdp::pathway::visual_pathway(&t_im, w, h, &par, &pn_got, &lmsr, &surround);
    let path_r_want = reference::visual_pathway(&r_im, w, h, &par, &pn_want, &lmsr, &surround);
    let path_t_want = reference::visual_pathway(&t_im, w, h, &par, &pn_want, &lmsr, &surround);

    let (br_got, pad_got) = hdrvdp::bands::decompose(&path_r_got, &par, None);
    let (bt_got, _) = hdrvdp::bands::decompose(&path_t_got, &par, Some(pad_got));
    let (br_want, pad_want) = reference::decompose(&path_r_want, &par, None);
    let (bt_want, _) = reference::decompose(&path_t_want, &par, Some(pad_want));

    let l_adapt_got: Vec<f64> = path_r_got
        .l_adapt
        .iter()
        .zip(&path_t_got.l_adapt)
        .map(|(a, b)| 0.5 * (a + b))
        .collect();
    let dm_got = hdrvdp::masking::diff_mask(&t_im, &r_im, 1);
    let l_adapt_want: Vec<f64> = path_r_want
        .l_adapt
        .iter()
        .zip(&path_t_want.l_adapt)
        .map(|(a, b)| 0.5 * (a + b))
        .collect();
    let dm_want = reference::diff_mask(&t_im, &r_im, 1);
    assert_bits_eq("l_adapt", &l_adapt_got, &l_adapt_want);

    for do_masking in [true, false] {
        let mut p2 = par.clone();
        p2.do_masking = do_masking;
        let m_got = hdrvdp::masking::run(&bt_got, &br_got, &l_adapt_got, &dm_got, &p2);
        let m_want = reference::masking_run(&bt_want, &br_want, &l_adapt_want, &dm_want, &p2);
        assert_bits_eq(
            &format!("quality_terms(mask={do_masking})"),
            &m_got.quality_terms,
            &m_want.quality_terms,
        );
        for b in 0..m_got.d_bands.count() {
            for o in 0..m_got.d_bands.orientations(b) {
                assert_bits_eq(
                    &format!("d_bands[b{b}o{o}](mask={do_masking})"),
                    &m_got.d_bands.band(b, o).data,
                    &m_want.d_bands.band(b, o).data,
                );
            }
        }

        for pooling in [true, false] {
            let mut p3 = p2.clone();
            p3.do_spatial_pooling = pooling;
            let v_got = hdrvdp::pool::visibility(&m_got.d_bands, &p3);
            let v_want = reference::visibility(&m_want.d_bands, &p3);
            let tag = format!("vis(mask={do_masking},pool={pooling})");
            assert_bit_eq(&format!("{tag}.p_det"), v_got.p_det, v_want.p_det);
            assert_bit_eq(&format!("{tag}.c_max"), v_got.c_max, v_want.c_max);
            assert_bits_eq(&format!("{tag}.p_map"), &v_got.p_map, &v_want.p_map);
            assert_bits_eq(&format!("{tag}.c_map"), &v_got.c_map, &v_want.c_map);
        }
    }
}
