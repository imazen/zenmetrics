//! Google Butteraugli (pre-libjxl), Apache-2.0 port of revision
//! 71b18b636b9c7d1ae0c1d3730b85b3c127eb4511. See LICENSE-butteraugli-original.
//! Uses the original polynomial opsin response, frequency bands, Malta
//! neighbourhoods, visual masking and maximum-error pooling, not libjxl tuning.
#![allow(clippy::excessive_precision)]
type Planes = [Vec<f32>; 3];
fn convolve(a: &[f32], w: usize, h: usize, k: &[f32], border: f32) -> Vec<f32> {
    const TILE: usize = 32;
    let radius = k.len() / 2;
    let sum = k.iter().sum::<f32>();
    let scaled: Vec<_> = k.iter().map(|&v| v * (1.0 / sum)).collect();
    let mut out = vec![0.0; w * h];
    let start = radius.min(w);
    let end = w.saturating_sub(radius).max(start);
    let mut border_column = |x: usize| {
        let lo = x.saturating_sub(radius);
        let hi = (x + radius).min(w - 1);
        let mut z = 0.0;
        for j in lo..=hi {
            z += k[j + radius - x];
        }
        let weight = 1.0 / ((1.0 - border) * z + border * sum);
        for y in 0..h {
            let mut v = 0.0;
            for j in lo..=hi {
                v += a[y * w + j] * k[j + radius - x];
            }
            out[x * h + y] = v * weight;
        }
    };
    for x in 0..start {
        border_column(x);
    }
    for x in end..w {
        border_column(x);
    }
    // Every lane retains the original left-to-right tap sequence. Tiles
    // keep source rows and the transposed output local; they do not combine
    // or reassociate a pixel's sum with any other pixel.
    for y0 in (0..h).step_by(TILE) {
        let rows = (h - y0).min(TILE);
        for x0 in (start..end).step_by(TILE) {
            let columns = (end - x0).min(TILE);
            let mut tile = [[0.0f32; TILE]; TILE];
            for (y, accum) in tile[..rows].iter_mut().enumerate() {
                let source = &a[(y0 + y) * w..(y0 + y + 1) * w];
                for (tap, &weight) in scaled.iter().enumerate() {
                    let lo = x0 + tap - radius;
                    for (v, &sample) in accum[..columns].iter_mut().zip(&source[lo..lo + columns]) {
                        *v += sample * weight;
                    }
                }
            }
            for x in 0..columns {
                let lo = (x0 + x) * h + y0;
                for (dst, row) in out[lo..lo + rows].iter_mut().zip(&tile[..rows]) {
                    *dst = row[x] * 1.0;
                }
            }
        }
    }
    out
}

fn blur(a: &[f32], w: usize, h: usize, sigma: f32, border: f32) -> Vec<f32> {
    let r = (2.25 * sigma.abs()).max(1.0) as i32;
    let scale = -1.0 / (2.0 * sigma * sigma);
    let k: Vec<_> = (-r..=r)
        .map(|i| (scale * i as f32 * i as f32).exp())
        .collect();
    convolve(&convolve(a, w, h, &k, border), h, w, &k, border)
}
fn opsin(p: [f32; 3]) -> [f32; 3] {
    [
        0.254462330846 * p[0] + 0.488238255095 * p[1] + 0.0635278003854 * p[2] + 1.01681026909,
        0.195214015766 * p[0] + 0.568019861857 * p[1] + 0.0860755536007 * p[2] + 1.1510118369,
        0.07374607900105684 * p[0]
            + 0.06142425304154509 * p[1]
            + 0.24416850520714256 * p[2]
            + 1.20481945273,
    ]
}
fn gamma(v: f32) -> f32 {
    let x = 2.0 * ((f64::from(v) - 0.971783) / (590.188894 - 0.971783)) - 1.0;
    let evaluate = |c: [f64; 6]| {
        let mut b1 = 0.0;
        let mut b2 = 0.0;
        for i in (1..6).rev() {
            let xb = x * b1;
            let t = (xb + xb) - b2 + c[i];
            b2 = b1;
            b1 = t;
        }
        x * b1 - b2 + c[0]
    };
    (evaluate([
        98.7821300963361,
        164.273222212631,
        92.948112871376,
        33.8165311212688,
        6.91626704983562,
        0.556380877028234,
    ]) / evaluate([
        1.0,
        1.64339473427892,
        0.89392405219969,
        0.298947051776379,
        0.0507146002577288,
        0.00226495093949756,
    ])) as f32
}
fn xyb(rgb: &Planes, w: usize, h: usize) -> Planes {
    let sm: Planes = std::array::from_fn(|i| blur(&rgb[i], w, h, 1.2, 0.0));
    let mut out = std::array::from_fn(|_| vec![0.0; w * h]);
    for i in 0..w * h {
        let p = opsin([rgb[0][i], rgb[1][i], rgb[2][i]]);
        let s = opsin([sm[0][i], sm[1][i], sm[2][i]]);
        let v: [f32; 3] = std::array::from_fn(|c| p[c] * (gamma(s[c]) / s[c]));
        out[0][i] = v[0] - v[1];
        out[1][i] = v[0] + v[1];
        out[2][i] = v[2];
    }
    out
}
fn remove(w: f32, x: f32) -> f32 {
    if x > w {
        x - w
    } else if x < -w {
        x + w
    } else {
        0.0
    }
}
fn amplify(w: f32, x: f32) -> f32 {
    if x > w {
        x + w
    } else if x < -w {
        x - w
    } else {
        2.0 * x
    }
}
fn clamp(v: f32, max: f32) -> f32 {
    if v >= max {
        (((v - max) as f64 * 0.688059627878) as f32) + max
    } else if v < -max {
        (((v + max) as f64 * 0.688059627878) as f32) - max
    } else {
        v
    }
}
struct Psycho {
    lf: Planes,
    mf: Planes,
    hf: [Vec<f32>; 2],
    uhf: [Vec<f32>; 2],
}
fn separate(rgb: &Planes, w: usize, h: usize) -> Psycho {
    let src = xyb(rgb, w, h);
    let mut lf: Planes =
        std::array::from_fn(|c| blur(&src[c], w, h, 7.46953768697, -0.00457628248637));
    let mut mf: Planes =
        std::array::from_fn(|c| src[c].iter().zip(&lf[c]).map(|(&a, &b)| a - b).collect());
    let mut hf: [Vec<f32>; 2] = std::array::from_fn(|c| mf[c].clone());
    for c in 0..3 {
        mf[c] = blur(&mf[c], w, h, 3.734768843485, -0.271277366628);
        if c < 2 {
            for i in 0..w * h {
                hf[c][i] -= mf[c][i];
                mf[c][i] = if c == 0 {
                    remove(0.120079806822, mf[c][i])
                } else {
                    amplify(0.03430529365, mf[c][i])
                };
            }
        }
    }
    for i in 0..w * h {
        let y = f64::from(hf[1][i]);
        let s = 0.745954517135 + (2.96534974403 * (1.0 - 0.745954517135)) / (2.96534974403 + y * y);
        hf[0][i] = (s * f64::from(hf[0][i])) as f32;
    }
    let mut uhf = hf.clone();
    for c in 0..2 {
        hf[c] = blur(&hf[c], w, h, 1.8673844217425, 0.147068973249);
        for i in 0..w * h {
            uhf[c][i] -= hf[c][i];
            if c == 0 {
                hf[c][i] = remove(0.0287615200377, hf[c][i]);
            } else {
                hf[c][i] = clamp(hf[c][i], 78.8223237675);
                uhf[c][i] = clamp(uhf[c][i], 5.8907152736);
                let reg_h = 2000.0f32 * 0.478741530298f32;
                let reg_u = 2000.0f32 * 0.310148420674f32;
                uhf[c][i] *= 1.76905001176f32 * reg_u / (reg_u + lf[1][i]);
                hf[c][i] *= 1.10684769012f32 * reg_h / (reg_h + lf[1][i]);
            }
        }
    }
    for i in 0..w * h {
        lf[2][i] = (lf[2][i] + -0.628811683685f32 * lf[1][i]) * 6.08319517575f32;
        lf[0][i] *= 5.57547552483f32;
        lf[1][i] *= 1.20828034498f32;
    }
    Psycho { lf, mf, hf, uhf }
}
fn malta(
    a: &[f32],
    b: &[f32],
    w: usize,
    h: usize,
    weight: f64,
    norm: f64,
    low: bool,
    out: &mut [f32],
) {
    let mul = if low { 0.405371989604 } else { 0.354191303559 };
    let n0 = (mul * (0.5f32 as f64 * weight).sqrt() / 8.5 * norm) as f32;
    let n1 = (mul * (0.33f32 as f64 * weight).sqrt() / 8.5 * norm) as f32;
    let mut dif = Vec::with_capacity(w * h);
    for i in 0..w * h {
        let abs = (0.5 * f64::from(a[i].abs()) + 0.5 * f64::from(b[i].abs())) as f32;
        let delta = a[i] - b[i];
        let den = norm as f32 + abs;
        let mut value = (n0 / den) * delta;
        let aa = f64::from(a[i].abs());
        let bb = f64::from(b[i]);
        let impact = if a[i] < 0.0 {
            if bb > -0.55 * aa {
                bb + 0.55 * aa
            } else if bb < -1.05 * aa {
                -bb - 1.05 * aa
            } else {
                0.0
            }
        } else if bb < 0.55 * aa {
            0.55 * aa - bb
        } else if bb > 1.05 * aa {
            bb - 1.05 * aa
        } else {
            0.0
        };
        let impact = f64::from(n1 / den) * impact;
        value = (f64::from(value) + if delta < 0.0 { -impact } else { impact }) as f32;
        dif.push(value);
    }
    let paths = if low { MALTA_LOW } else { MALTA_HIGH };
    malta_accumulate(&dif, w, h, paths, out);
}
fn malta_accumulate(dif: &[f32], w: usize, h: usize, paths: &[&[(i8, i8)]], out: &mut [f32]) {
    let scalar = |x: usize, y: usize| {
        let mut value = 0.0;
        for path in paths {
            let mut sum = 0.0;
            for &(dx, dy) in *path {
                let xx = x as isize + isize::from(dx);
                let yy = y as isize + isize::from(dy);
                if xx >= 0 && yy >= 0 && xx < w as isize && yy < h as isize {
                    sum += dif[yy as usize * w + xx as usize];
                }
            }
            value += sum * sum;
        }
        value
    };
    for y in 0..h {
        let mut x = 0;
        if y >= 4 && y + 4 < h {
            for xx in 0..w.min(4) {
                out[y * w + xx] += scalar(xx, y);
            }
            x = w.min(4);
            // Both fixed path sets have radius four. A contiguous eight-
            // sample load handles eight independent sums in the same order
            // as the scalar traversal, with one bounds check per neighbor.
            while x + 8 <= w.saturating_sub(4) {
                let mut value = [0.0f32; 8];
                for path in paths {
                    let mut sum = [0.0f32; 8];
                    for &(dx, dy) in *path {
                        let lo = (y as isize + isize::from(dy)) as usize * w
                            + (x as isize + isize::from(dx)) as usize;
                        let samples: &[f32; 8] = dif[lo..lo + 8].try_into().unwrap();
                        for i in 0..8 {
                            sum[i] += samples[i];
                        }
                    }
                    for i in 0..8 {
                        value[i] += sum[i] * sum[i];
                    }
                }
                let dst: &mut [f32; 8] = (&mut out[y * w + x..y * w + x + 8]).try_into().unwrap();
                for i in 0..8 {
                    dst[i] += value[i];
                }
                x += 8;
            }
        }
        for xx in x..w {
            out[y * w + xx] += scalar(xx, y);
        }
    }
}

// Interior Malta traversal reserves a four-pixel halo on every side.
const _: () = {
    let banks = [MALTA_LOW, MALTA_HIGH];
    let mut bank = 0;
    while bank < banks.len() {
        let mut path = 0;
        while path < banks[bank].len() {
            let mut step = 0;
            while step < banks[bank][path].len() {
                let (x, y) = banks[bank][path][step];
                assert!(x >= -4 && x <= 4 && y >= -4 && y <= 4);
                step += 1;
            }
            path += 1;
        }
        bank += 1;
    }
};

#[cfg(test)]
fn malta_accumulate_ordered(
    dif: &[f32],
    w: usize,
    h: usize,
    paths: &[&[(i8, i8)]],
    out: &mut [f32],
) {
    for y in 0..h {
        for x in 0..w {
            let mut value = 0.0;
            for path in paths {
                let mut sum = 0.0;
                for &(dx, dy) in *path {
                    let xx = x as isize + isize::from(dx);
                    let yy = y as isize + isize::from(dy);
                    if xx >= 0 && yy >= 0 && xx < w as isize && yy < h as isize {
                        sum += dif[yy as usize * w + xx as usize];
                    }
                }
                value += sum * sum;
            }
            out[y * w + x] += value;
        }
    }
}

fn l2(a: &[f32], b: &[f32], weight: f64, out: &mut [f32]) {
    if weight == 0.0 {
        return;
    }
    for i in 0..a.len() {
        let d = f64::from(a[i] - b[i]);
        out[i] = (f64::from(out[i]) + weight * d * d) as f32;
    }
}
fn asymmetric(a: &[f32], b: &[f32], weight: f64, out: &mut [f32]) {
    if weight == 0.0 {
        return;
    }
    let weight = weight * 0.8;
    for i in 0..a.len() {
        let d = f64::from(a[i] - b[i]);
        out[i] = (f64::from(out[i]) + weight * d * d) as f32;
        let aa = f64::from(a[i].abs());
        let bb = f64::from(b[i]);
        let impact = if a[i] < 0.0 {
            if bb > -0.4 * aa {
                bb + 0.4 * aa
            } else if bb < -aa {
                -bb - aa
            } else {
                0.0
            }
        } else if bb < 0.4 * aa {
            0.4 * aa - bb
        } else if bb > aa {
            bb - aa
        } else {
            0.0
        };
        out[i] = (f64::from(out[i]) + weight * impact * impact) as f32;
    }
}
fn noise(a: &[f32], b: &[f32], w: usize, h: usize, out: &mut [f32]) {
    let d: Vec<_> = a
        .iter()
        .zip(b)
        .map(|(&a, &b)| {
            (f64::from(a.abs()).min(85.7047444518) - f64::from(b.abs()).min(85.7047444518)) as f32
        })
        .collect();
    let d = blur(&d, w, h, 10.6666499623, 0.0);
    for i in 0..w * h {
        let x = f64::from(d[i]);
        out[i] = (f64::from(out[i]) + 884.809801415 * x * x) as f32;
    }
}
fn diffpre(a: &[f32], b: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        let yy = if y + 1 < h {
            y + 1
        } else {
            y.saturating_sub(1)
        };
        for x in 0..w {
            let xx = if x + 1 < w {
                x + 1
            } else {
                x.saturating_sub(1)
            };
            let i = y * w + x;
            let p = f64::from((a[i] - a[y * w + xx]).abs() + (a[i] - a[yy * w + x]).abs());
            let q = f64::from((b[i] - b[y * w + xx]).abs() + (b[i] - b[yy * w + x]).abs());
            out[i] = (0.918416534734 * p.min(q)) as f32;
            if out[i] >= 55.0184555849 {
                out[i] = 55.0184555849;
            }
        }
    }
    out
}
fn lut(params: [f64; 5]) -> [f64; 512] {
    let [ext, off, mul, offset, scale] = params;
    std::array::from_fn(|i| {
        let c = mul / (0.01 * scale * i as f64 + offset);
        let v = ((1.0 / 20.35) * (1.0 + ext * (c + off))).max(1e-5);
        v * v
    })
}
fn lookup(a: &[f64; 512], x: f64) -> f64 {
    let x = x.max(0.0);
    let k = x as usize;
    if k >= 511 {
        a[511]
    } else {
        a[k] + (x - k as f64) * (a[k + 1] - a[k])
    }
}
fn masks(a: &Psycho, b: &Psycho, w: usize, h: usize) -> (Planes, Planes) {
    let mix = |p: &Psycho, c: usize| {
        let (a, b) = if c == 0 {
            (0.0, 1.64178305129)
        } else {
            (0.831081703362, 3.23680933546)
        };
        p.uhf[c]
            .iter()
            .zip(&p.hf[c])
            .map(|(&u, &h)| (a * f64::from(u) + b * f64::from(h)) as f32)
            .collect::<Vec<_>>()
    };
    let x = diffpre(&mix(a, 0), &mix(b, 0), w, h);
    let y = diffpre(&mix(a, 1), &mix(b, 1), w, h);
    let x = blur(&x, w, h, 9.24456601467, -0.0724948220913);
    let y0 = blur(&y, w, h, 2.3770330432, -0.0724948220913);
    let y1 = blur(&y, w, h, 9.04353323561, -0.0724948220913);
    let tables = [
        lut([
            2.59885507073,
            3.08805636789,
            5.62939030582,
            0.315424196682,
            16.2770141832,
        ]),
        lut([
            0.9613705131,
            -0.581933100068,
            6.64307621174,
            1.00846207765,
            2.2342321176,
        ]),
        lut([
            10.0470705878,
            3.18472654033,
            0.373092999662,
            0.0551512255218,
            70.0,
        ]),
        lut([
            0.0115640939227,
            45.9483175519,
            2.52611324247,
            0.0142290066313,
            5.0,
        ]),
    ];
    let mut ac: Planes = std::array::from_fn(|_| vec![0.0; w * h]);
    let mut dc = ac.clone();
    for i in 0..w * h {
        let y = ((1.0 / (0.207017089891 + 0.267138152891))
            * (0.207017089891 * f64::from(y0[i]) + 0.267138152891 * f64::from(y1[i])))
            as f32;
        let p1 = 2.1364621982 * 2.1887170895 * f64::from(y);
        let p0 = 16.6963293877 * 36.4671237619 * f64::from(x[i]) + 0.0513061271723 * p1;
        ac[0][i] = lookup(&tables[0], p0) as f32;
        ac[1][i] = lookup(&tables[1], p1) as f32;
        ac[2][i] = (0.086624184478 * lookup(&tables[1], p1)) as f32;
        dc[0][i] = lookup(&tables[2], p0) as f32;
        dc[1][i] = lookup(&tables[3], p1) as f32;
        dc[2][i] = (21.6804277046 * lookup(&tables[3], p1)) as f32;
    }
    (ac, dc)
}
fn compute(r: &Planes, d: &Planes, w: usize, h: usize) -> Vec<f32> {
    let a = separate(r, w, h);
    let b = separate(d, w, h);
    let mut ac: Planes = std::array::from_fn(|_| vec![0.0; w * h]);
    let mut dc = ac.clone();
    malta(
        &a.uhf[1],
        &b.uhf[1],
        w,
        h,
        5.1409625726,
        58.5001247061,
        false,
        &mut ac[1],
    );
    malta(
        &a.uhf[0],
        &b.uhf[0],
        w,
        h,
        4.91743441556,
        687196.39002,
        false,
        &mut ac[0],
    );
    malta(
        &a.hf[1],
        &b.hf[1],
        w,
        h,
        153.671655716,
        83150785.9592,
        true,
        &mut ac[1],
    );
    malta(
        &a.hf[0],
        &b.hf[0],
        w,
        h,
        668.358918152,
        0.882954368025,
        true,
        &mut ac[0],
    );
    malta(
        &a.mf[1],
        &b.mf[1],
        w,
        h,
        6841.81248144,
        0.0135134962487,
        true,
        &mut ac[1],
    );
    malta(
        &a.mf[0],
        &b.mf[0],
        w,
        h,
        813.901703816,
        16792.9322251,
        true,
        &mut ac[0],
    );
    noise(&a.hf[1], &b.hf[1], w, h, &mut ac[1]);
    asymmetric(&a.hf[1], &b.hf[1], 32.4449876135, &mut ac[1]);
    // All original MF L2 weights, the X HF L2 weight, and Y DC weight are zero.
    l2(&a.lf[0], &b.lf[0], 1.01370836411, &mut dc[0]);
    l2(&a.lf[2], &b.lf[2], 1.74566011615, &mut dc[2]);
    let (mask, mask_dc) = masks(&a, &b, w, h);
    (0..w * h)
        .map(|i| {
            let d = dc[0][i] * mask_dc[0][i] + dc[1][i] * mask_dc[1][i] + dc[2][i] * mask_dc[2][i];
            let c = ac[0][i] * mask[0][i] + ac[1][i] * mask[1][i] + ac[2][i] * mask[2][i];
            let v = d + c;
            if v < 1.0 / (100.0 * 100.0) {
                100.0 * v
            } else {
                v.sqrt()
            }
        })
        .collect()
}
pub(super) fn error_map(
    r: &[u8],
    d: &[u8],
    w: usize,
    h: usize,
    rs: usize,
    ds: usize,
) -> Result<Vec<f32>, &'static str> {
    let ingress = |rgb: &[u8], stride: usize| -> Result<Planes, &'static str> {
        let need = h
            .checked_sub(1)
            .and_then(|n| n.checked_mul(stride))
            .and_then(|n| n.checked_add(w.checked_mul(3)?));
        if w == 0 || h == 0 || stride < w.saturating_mul(3) || need.is_none_or(|n| rgb.len() < n) {
            return Err("Butteraugli-original: invalid RGB geometry");
        }
        let lut: [f32; 256] = std::array::from_fn(|i| {
            let s = i as f64;
            if s <= 10.31475 {
                (s / 12.92) as f32
            } else {
                (255.0 * ((s / 255.0 + 0.055) / 1.055).powf(2.4)) as f32
            }
        });
        let mut out = std::array::from_fn(|_| Vec::with_capacity(w * h));
        for y in 0..h {
            for p in rgb[y * stride..y * stride + w * 3].as_chunks::<3>().0 {
                for c in 0..3 {
                    out[c].push(lut[usize::from(p[c])]);
                }
            }
        }
        Ok(out)
    };
    let r = ingress(r, rs)?;
    let d = ingress(d, ds)?;
    if w >= 8 && h >= 8 {
        return Ok(compute(&r, &d, w, h));
    }
    let nw = w.max(8);
    let nh = h.max(8);
    let bx = (nw - w) / 2;
    let by = (nh - h) / 2;
    let extend = |p: &Planes| {
        std::array::from_fn(|c| {
            (0..nw * nh)
                .map(|i| {
                    p[c][(i / nw).saturating_sub(by).min(h - 1) * w
                        + (i % nw).saturating_sub(bx).min(w - 1)]
                })
                .collect()
        })
    };
    let map = compute(&extend(&r), &extend(&d), nw, nh);
    Ok((0..w * h)
        .map(|i| map[(i / w + by) * nw + i % w + bx])
        .collect())
}

// Fixed line-integral neighbourhoods from the pinned original reference.
const MALTA_LOW: &[&[(i8, i8)]] = &[
    &[(-4, 0), (-2, 0), (0, 0), (2, 0), (4, 0)],
    &[(0, -4), (0, -2), (0, 0), (0, 2), (0, 4)],
    &[(-3, -3), (-2, -2), (0, 0), (2, 2), (3, 3)],
    &[(3, -3), (2, -2), (0, 0), (-2, 2), (-3, 3)],
    &[(1, -4), (1, -2), (0, 0), (-1, 2), (-1, 4)],
    &[(-1, -4), (-1, -2), (0, 0), (1, 2), (1, 4)],
    &[(-4, -1), (-2, -1), (0, 0), (2, 1), (4, 1)],
    &[(-4, 1), (-2, 1), (0, 0), (2, -1), (4, -1)],
    &[(-2, -3), (-1, -2), (0, 0), (1, 2), (2, 3)],
    &[(2, -3), (1, -2), (0, 0), (-1, 2), (-2, 3)],
    &[(-3, -2), (-2, -1), (0, 0), (2, 1), (3, 2)],
    &[(3, -2), (2, -1), (0, 0), (-2, 1), (-3, 2)],
    &[(-4, 2), (-2, 1), (0, 0), (2, -1), (4, -2)],
    &[(-4, -2), (-2, -1), (0, 0), (2, 1), (4, 2)],
    &[(-2, -4), (-1, -2), (0, 0), (1, 2), (2, 4)],
    &[(2, -4), (1, -2), (0, 0), (-1, 2), (-2, 4)],
];

// Fixed line-integral neighbourhoods from the pinned original reference.
const MALTA_HIGH: &[&[(i8, i8)]] = &[
    &[
        (-4, 0),
        (-3, 0),
        (-2, 0),
        (-1, 0),
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
    ],
    &[
        (0, -4),
        (0, -3),
        (0, -2),
        (0, -1),
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
    ],
    &[(-3, -3), (-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2), (3, 3)],
    &[(3, -3), (2, -2), (1, -1), (0, 0), (-1, 1), (-2, 2), (-3, 3)],
    &[
        (1, -4),
        (1, -3),
        (1, -2),
        (0, -1),
        (0, 0),
        (0, 1),
        (-1, 2),
        (-1, 3),
        (-1, 4),
    ],
    &[
        (-1, -4),
        (-1, -3),
        (-1, -2),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, 2),
        (1, 3),
        (1, 4),
    ],
    &[
        (-4, -1),
        (-3, -1),
        (-2, -1),
        (-1, 0),
        (0, 0),
        (1, 0),
        (2, 1),
        (3, 1),
        (4, 1),
    ],
    &[
        (-4, 1),
        (-3, 1),
        (-2, 1),
        (-1, 0),
        (0, 0),
        (1, 0),
        (2, -1),
        (3, -1),
        (4, -1),
    ],
    &[(-2, -3), (-1, -2), (-1, -1), (0, 0), (1, 1), (1, 2), (2, 3)],
    &[(2, -3), (1, -2), (1, -1), (0, 0), (-1, 1), (-1, 2), (-2, 3)],
    &[(-3, -2), (-2, -1), (-1, -1), (0, 0), (1, 1), (2, 1), (3, 2)],
    &[(3, -2), (2, -1), (1, -1), (0, 0), (-1, 1), (-2, 1), (-3, 2)],
    &[
        (-4, 2),
        (-3, 2),
        (-2, 1),
        (-1, 1),
        (0, 0),
        (1, 0),
        (2, -1),
        (3, -1),
    ],
    &[
        (-4, -2),
        (-3, -2),
        (-2, -1),
        (-1, -1),
        (0, 0),
        (1, 0),
        (2, 1),
        (3, 1),
    ],
    &[
        (-2, -4),
        (-2, -3),
        (-1, -2),
        (-1, -1),
        (0, 0),
        (0, 1),
        (1, 2),
        (1, 3),
    ],
    &[
        (2, -4),
        (2, -3),
        (1, -2),
        (1, -1),
        (0, 0),
        (0, 1),
        (-1, 2),
        (-1, 3),
    ],
];

#[cfg(test)]
fn convolve_ordered(a: &[f32], w: usize, h: usize, k: &[f32], border: f32) -> Vec<f32> {
    let radius = k.len() / 2;
    let sum = k.iter().sum::<f32>();
    let scaled: Vec<_> = k.iter().map(|&v| v * (1.0 / sum)).collect();
    let mut out = vec![0.0; w * h];
    for x in 0..w {
        let lo = x.saturating_sub(radius);
        let hi = (x + radius).min(w - 1);
        let interior = x >= radius && x + radius < w;
        let weight = if interior {
            1.0
        } else {
            let mut z = 0.0;
            for j in lo..=hi {
                z += k[j + radius - x];
            }
            1.0 / ((1.0 - border) * z + border * sum)
        };
        for y in 0..h {
            let mut v = 0.0;
            for j in lo..=hi {
                v += a[y * w + j]
                    * if interior {
                        scaled[j + radius - x]
                    } else {
                        k[j + radius - x]
                    };
            }
            out[x * h + y] = v * weight;
        }
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malta_blocks_match_ordered_paths_at_borders_and_tails() {
        for paths in [MALTA_LOW, MALTA_HIGH] {
            assert!(
                paths
                    .iter()
                    .flat_map(|path| path.iter())
                    .all(|&(x, y)| (-4..=4).contains(&x) && (-4..=4).contains(&y))
            );
            for (w, h) in [
                (1, 1),
                (7, 5),
                (8, 9),
                (16, 9),
                (17, 15),
                (31, 19),
                (32, 33),
                (65, 35),
                (129, 67),
            ] {
                let dif: Vec<f32> = (0..w * h)
                    .map(|i| (((i * 7919 + i / w * 13) % 8191) as i32 - 4095) as f32 * 0.0625)
                    .collect();
                let mut expected: Vec<f32> =
                    (0..w * h).map(|i| (i % 19) as f32 * 0.015625).collect();
                let mut actual = expected.clone();
                malta_accumulate_ordered(&dif, w, h, paths, &mut expected);
                malta_accumulate(&dif, w, h, paths, &mut actual);
                for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(a.to_bits(), b.to_bits(), "{w}x{h} pixel{index}");
                }
            }
        }
    }

    #[test]
    fn tiled_convolution_matches_ordered_taps_at_borders_and_tails() {
        for (w, h) in [
            (1, 1),
            (1, 5),
            (2, 3),
            (7, 2),
            (9, 17),
            (31, 19),
            (32, 33),
            (65, 35),
            (129, 67),
        ] {
            let mut input: Vec<f32> = (0..w * h)
                .map(|i| (((i * 7919 + i / w * 13) % 8191) as i32 - 4095) as f32 * 0.0625)
                .collect();
            input[0] = -0.0;
            for radius in [0, 1, 3, 7, 15] {
                let kernel: Vec<f32> = (0..2 * radius + 1)
                    .map(|i| ((i * 7 + 3) % 17 + 1) as f32 * 0.03125)
                    .collect();
                for border in [0.0, 0.25, 1.0] {
                    let expected = convolve_ordered(&input, w, h, &kernel, border);
                    let actual = convolve(&input, w, h, &kernel, border);
                    for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{w}x{h} r{radius} border{border} pixel{index}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn stride_identity_and_tiny_padding() {
        for (w, h) in [(1, 1), (7, 5), (37, 41)] {
            let r: Vec<_> = (0..w * h * 3)
                .map(|i| ((i * 37 + i / 11) % 256) as u8)
                .collect();
            let mut d = r.clone();
            for p in d.iter_mut().step_by(17) {
                *p ^= 63;
            }
            assert!(
                error_map(&r, &r, w, h, w * 3, w * 3)
                    .unwrap()
                    .iter()
                    .all(|&v| v == 0.0)
            );
            let expected = error_map(&r, &d, w, h, w * 3, w * 3).unwrap();
            assert!(expected.iter().any(|&v| v > 0.0));
            let stride = w * 3 + 7;
            let mut sr = vec![222; stride * h];
            let mut sd = sr.clone();
            for y in 0..h {
                sr[y * stride..y * stride + w * 3].copy_from_slice(&r[y * w * 3..(y + 1) * w * 3]);
                sd[y * stride..y * stride + w * 3].copy_from_slice(&d[y * w * 3..(y + 1) * w * 3]);
            }
            assert_eq!(error_map(&sr, &sd, w, h, stride, stride).unwrap(), expected);
        }
        assert!(error_map(&[], &[], 0, 1, 0, 0).is_err());
    }
}

#[cfg(test)]
mod oracle_tests {
    use super::*;
    // Fixture and original-author values: benchmarks/aic_review_2026-10-15/pin_oracles.py.
    #[test]
    fn original_author_synthetic_pin() {
        let (w, h) = (64, 64);
        let r: Vec<u8> = (0..h)
            .flat_map(|y| {
                (0..w).flat_map(move |x| {
                    (0..3).map(move |c| ((x * 13 + y * 17 + c * 71 + (x * y) % 29) % 256) as u8)
                })
            })
            .collect();
        let mut d = r.clone();
        for p in d.iter_mut().step_by(17) {
            *p ^= 63;
        }
        let actual = error_map(&r, &d, w, h, w * 3, w * 3)
            .unwrap()
            .into_iter()
            .fold(0.0_f32, f32::max);
        // C++ Google 71b18b63, printf increased to 17 significant digits.
        // 1e-7 is below one f32 ULP at 2.20: a one-ULP drift fails.
        assert!((f64::from(actual) - 2.201789379119873).abs() < 1e-7);
    }
}
