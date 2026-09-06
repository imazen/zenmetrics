//! **G-GPU.1's instrument** — a raw `to_bits()` dump of every GPU feature
//! slot, over a fixed fixture set, at whatever arithmetic revision the process
//! is running.
//!
//! Deliberately carries NO reference to `formula_rev`, `FormulaRevision` or
//! `with_formula_revision`: it has to compile and run UNCHANGED on the
//! pre-port tree, so that "revision 1 through the ported path is byte-identical
//! to today's path" is a `cmp` of two files rather than an argument about what
//! the diff does. The revision is selected by `ZENSIM_FORMULA_REV` alone
//! (unset = the shipped revision = 1).
//!
//! Every fixture is generated in-process from a fixed LCG, so the dump is a
//! pure function of the binary — no corpus files, no decode era, no clock.
//!
//! Coverage, chosen so all four v1 SSIM kernels and the F17 host finalize are
//! exercised: `Basic` (228) drives `fused_features_kernel`; `Extended` (300)
//! and `WithIw` (372) drive `fused_features_kernel_persist` +
//! `masked_iw_strip_kernel`; a strip-mode instance drives the strip walker's
//! body/halo gating; and `hf_energy_gain` (basic block-local slot 11) is
//! computed host-side for every (scale, channel) of all of them.
//!
//! Usage: `cargo run --example formula_rev_dump --features wgpu,cubecl-types
//! -- <out.txt>`

use cubecl::Runtime;
#[cfg(all(feature = "cpu", not(feature = "cuda"), not(feature = "wgpu")))]
use cubecl::cpu::CpuRuntime as Backend;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;
use std::io::Write;
use zensim_gpu::{Zensim, ZensimFeatureRegime};

/// Deterministic LCG — the same generator the crate's own parity tests use,
/// so the fixtures live in the same distribution the tolerances were tuned on.
struct Lcg(u32);
impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) as u8
    }
}

fn gradient(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            v.push(((x * 255) / w.max(1)) as u8);
            v.push(((y * 255) / h.max(1)) as u8);
            v.push((((x + y) * 255) / (w + h).max(1)) as u8);
        }
    }
    v
}

fn add_noise(data: &[u8], amount: i16) -> Vec<u8> {
    let mut lcg = Lcg(12345);
    data.iter()
        .map(|&v| {
            let n = (lcg.next_u8() as i16 % (amount * 2 + 1)) - amount;
            (v as i16 + n).clamp(0, 255) as u8
        })
        .collect()
}

fn solid(w: usize, h: usize, c: [u8; 3]) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for _ in 0..w * h {
        v.extend_from_slice(&c);
    }
    v
}

/// A FLAT reference against a NOISY distorted image.
///
/// This is the fixture shape that drives `var_dst >> var_src` — F17's
/// `contrast_inc` regime. Not flat enough to trip the `var_src > 1e-10` gate
/// (which would return a hard 0 on both revisions and prove nothing): a
/// low-amplitude ramp keeps the source variance alive while the distorted
/// side carries real HF.
fn flat_ref_noisy_dist(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    let mut r = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            // Amplitude 2/255 — enough variance to clear the gate, small
            // enough that any HF in the distorted side dwarfs it.
            let v = 120u8 + (((x + y) % 3) as u8);
            r.extend_from_slice(&[v, v, v]);
        }
    }
    let d = add_noise(&r, 60);
    (r, d)
}

/// A HIGH-CONTRAST checkerboard against its inverse — drives the local mean
/// difference `|mu1 - mu2|` as far as 8-bit sRGB allows, which is F4's regime.
fn checker_pair(w: usize, h: usize, block: usize) -> (Vec<u8>, Vec<u8>) {
    let mut a = Vec::with_capacity(w * h * 3);
    let mut b = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let on = ((x / block) + (y / block)).is_multiple_of(2);
            let (p, q) = if on { (255u8, 0u8) } else { (0u8, 255u8) };
            a.extend_from_slice(&[p, p, p]);
            b.extend_from_slice(&[q, q, q]);
        }
    }
    (a, b)
}

/// One dump case: a label, the dimensions, and the (reference, distorted)
/// sRGB byte buffers.
type Fixture = (&'static str, usize, usize, Vec<u8>, Vec<u8>);

fn fixtures() -> Vec<Fixture> {
    let mut out: Vec<Fixture> = Vec::new();
    for &(w, h) in &[(64usize, 64usize), (128, 96), (256, 256)] {
        let name: &'static str = match (w, h) {
            (64, 64) => "64x64",
            (128, 96) => "128x96",
            _ => "256x256",
        };
        let g = gradient(w, h);
        out.push((name, w, h, g.clone(), add_noise(&g, 8)));
        let (fr, fd) = flat_ref_noisy_dist(w, h);
        out.push((name, w, h, fr, fd));
        let (ca, cb) = checker_pair(w, h, 8);
        out.push((name, w, h, ca, cb));
        out.push((
            name,
            w,
            h,
            solid(w, h, [0, 0, 0]),
            solid(w, h, [255, 255, 255]),
        ));
        out.push((name, w, h, g.clone(), g));
    }
    out
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: formula_rev_dump <out>");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).expect("create dump"));
    writeln!(
        f,
        "# zensim-gpu feature bit-dump; ZENSIM_FORMULA_REV={}",
        std::env::var("ZENSIM_FORMULA_REV").unwrap_or_else(|_| "<unset>".into())
    )
    .unwrap();

    let regimes = [
        ("basic", ZensimFeatureRegime::Basic),
        ("extended", ZensimFeatureRegime::Extended),
        ("withiw", ZensimFeatureRegime::WithIw),
    ];

    for (idx, (name, w, h, r, d)) in fixtures().into_iter().enumerate() {
        for (rname, regime) in regimes {
            let client = Backend::client(&Default::default());
            let mut z = Zensim::<Backend>::new_with_regime(client, w as u32, h as u32, regime)
                .expect("construct");
            let feats = z.compute_features_vec(&r, &d).expect("compute");
            for (i, v) in feats.iter().enumerate() {
                writeln!(f, "{idx}\t{name}\t{rname}\tf{i}\t{:016x}", v.to_bits()).unwrap();
            }
            // Warm-reference route: same kernels, different upload path.
            let client2 = Backend::client(&Default::default());
            let mut z2 = Zensim::<Backend>::new_with_regime(client2, w as u32, h as u32, regime)
                .expect("construct warm");
            z2.set_reference(&r).expect("set_reference");
            let warm = z2.compute_with_reference_vec(&d).expect("warm compute");
            for (i, v) in warm.iter().enumerate() {
                writeln!(f, "{idx}\t{name}\t{rname}\twarm_f{i}\t{:016x}", v.to_bits()).unwrap();
            }
        }
    }

    // Strip mode — exercises `masked_iw_strip_kernel`'s body/halo gating and
    // the persist kernel at a non-full-image `y_body` range.
    for &(w, h, body) in &[(256u32, 320u32, 256u32)] {
        for (rname, regime) in regimes {
            let client = Backend::client(&Default::default());
            let mut z =
                Zensim::<Backend>::new_strip_with_halo_and_regime(client, w, h, body, 40, regime)
                    .expect("construct strip");
            let g = gradient(w as usize, h as usize);
            let n = add_noise(&g, 8);
            let feats = z.compute_features_vec(&g, &n).expect("strip compute");
            for (i, v) in feats.iter().enumerate() {
                writeln!(f, "strip\t{w}x{h}\t{rname}\tf{i}\t{:016x}", v.to_bits()).unwrap();
            }
        }
    }

    // ── HDR / PU-XYB fixtures ──
    //
    // The 8-bit sRGB fixtures above cannot reach F4's regime: after the
    // opsin + cbrt transform an SDR XYB channel mean stays well inside
    // [-1, 1], so `(mu1 - mu2)^2 > 1` never happens and revision 2's `Clamp`
    // arm is bit-identical to revision 1 there — which is exactly what the
    // CPU lane measured over 217,756 real rows.
    //
    // The PU-XYB (HDR) route encodes absolute luminance through PU21 into a
    // range ~[0, 600] before normalising by PU_WHITE = 256.3, so a
    // near-black reference against a 10,000 cd/m^2 distorted image drives
    // |mu1 - mu2| past 1 and makes the ported branch LIVE. Without this
    // block the F4 half of the port would be untested rather than merely
    // unmoved.
    for &(w, h) in &[(64usize, 64usize), (128usize, 96usize)] {
        for (rname, regime) in regimes {
            // Reference: near the PU floor. Distorted: near the PU ceiling,
            // with a spatial pattern so the blur windows straddle the swing.
            let n = w * h;
            let mut lo = vec![0.005_f32; n];
            let mut hi = vec![0.005_f32; n];
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    lo[i] = if (x / 16 + y / 16) % 2 == 0 {
                        0.005
                    } else {
                        0.02
                    };
                    hi[i] = if (x / 16 + y / 16) % 2 == 0 {
                        10000.0
                    } else {
                        4000.0
                    };
                }
            }
            let client = Backend::client(&Default::default());
            let mut z = Zensim::<Backend>::new_with_regime(client, w as u32, h as u32, regime)
                .expect("construct pu");
            let feats = z.compute_features_pu_linear_nits([&lo, &lo, &lo], [&hi, &hi, &hi]);
            for (i, v) in feats.iter().enumerate() {
                writeln!(f, "pu\t{w}x{h}\t{rname}\tf{i}\t{:016x}", v.to_bits()).unwrap();
            }
        }
    }

    f.flush().unwrap();
    eprintln!("wrote {path}");
}
