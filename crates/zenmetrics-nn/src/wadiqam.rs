//! WaDIQaM (Bosse, Maniry, Müller, Wiegand, Samek, "Deep Neural
//! Networks for No-Reference and Full-Reference Image Quality
//! Assessment", TIP 2018; upstream `dmaniry/deepIQA` — MIT code **and**
//! MIT Chainer weights, `models/{fr,nr}_{live,tid}_*.model`).
//! `WeightsLicence::Permissive`; parity target: upstream `evaluate.py`
//! run on CPU fp32, `top="weighted"` (the documented default).
//!
//! The eval transform is upstream's deterministic one (README: the
//! paper used randomly sampled overlapping patches; `evaluate.py`
//! uses non-overlapping 32×32 stride-32 patches at native resolution,
//! raw u8 → f32, no normalisation). Partial edge patches are dropped
//! (the `as_strided` grid is `⌊(dim−32)/32⌋+1` per axis).
//!
//! A shared VGG-ish trunk (five blocks of 2× conv3×3+relu + maxpool2)
//! maps each 32² patch to 512 features. FR concatenates
//! `(h − h_ref, h, h_ref)` to 1536; NR uses `h` alone. `fc1 → relu →
//! fc2` yields the per-patch score `y`; a parallel `fc1_a → relu →
//! fc2_a → relu + 1e-6` yields the patch weight `a`. The image score
//! is `Σ(y·a)/Σa`, pooled host-side (upstream's `weighted_loss`
//! computes the same expression over `model.y`/`model.a`).
//!
//! `models/wadiqam/export.py` loads the official HDF5 checkpoints with
//! h5py (no chainer needed). NR ships whole:
//! `wadiqam_nr_{live,tid}.onnx` takes `x[P,3,32,32]` → `ya[2,P]`. FR
//! ships split — `wadiqam_fr_{live,tid}_trunk.onnx` (`x` → `h[P,512]`)
//! and `wadiqam_fr_{live,tid}_head.onnx` (`f[P,1536]` → `ya`) — so the
//! reference's patch features are computed once under
//! [`za::Metric::warm_reference`] and the host owns the concat.

use std::borrow::Cow;
use std::path::Path;

use crate::common::read_meta;
#[cfg(feature = "ort")]
use crate::engine::ort_engine::OrtCudaImage;
use crate::engine::{self, ImageGraph, NamedInput};
use crate::{Engine, Error, WeightsLicence, WeightsTerms, za};

/// Bump whenever any output can change.
const SCORE_REVISION: u32 = 1;
/// Patch side and grid stride (upstream `extract_patches` defaults).
const PATCH: usize = 32;
/// Trunk output width per patch.
const FEAT: usize = 512;
/// Patch-count bound, applied before the patch tensor allocates
/// (32,768 patches ≈ 400 MiB of f32 input; a 4096² image uses 16,384).
/// Chunked graph calls keep the per-conv intermediates bounded too —
/// see [`PATCH_CHUNK`].
const MAX_PATCHES: usize = 32768;

/// Graph batch size: upstream's evaluate.py scores 2000 patches per
/// forward call. Without chunking, one `[P,3,32,32]` trunk call
/// materialises ~2 GiB per conv activation at 4096²; chunking keeps
/// that ≤ 256 MiB. Per-patch outputs are independent, so chunked
/// scores equal single-call scores bit for bit.
const PATCH_CHUNK: usize = 2000;

/// The released checkpoints: `{fr,nr}_{live,tid}_weighted.model`,
/// trained on the full LIVE or TID2013 database. `patchwise` tops are
/// fetched alongside but not registered (upstream's documented default
/// top is `weighted`).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WadiqamKind {
    /// `fr_live_weighted.model` — full-reference, LIVE-trained.
    FrLive,
    /// `fr_tid_weighted.model` — full-reference, TID2013-trained.
    FrTid,
    /// `nr_live_weighted.model` — no-reference, LIVE-trained.
    NrLive,
    /// `nr_tid_weighted.model` — no-reference, TID2013-trained.
    NrTid,
}

impl WadiqamKind {
    fn column(self) -> &'static str {
        match self {
            Self::FrLive => "wadiqam_fr_live",
            Self::FrTid => "wadiqam_fr_tid",
            Self::NrLive => "wadiqam_nr_live",
            Self::NrTid => "wadiqam_nr_tid",
        }
    }

    /// Whether this is one of the full-reference checkpoints.
    pub fn is_fr(self) -> bool {
        matches!(self, Self::FrLive | Self::FrTid)
    }

    fn score_kind(self) -> za::ScoreKind {
        match self {
            // FR checkpoints are distance-like (identical pairs score a
            // nonzero floor but distorted pairs score above it).
            Self::FrLive | Self::FrTid => za::ScoreKind::Difference,
            // Measured on a 12-source KonJND JPEG q95..q3 ladder
            // (benchmarks/wadiqam_port_2026-10-16.md): nr_live and
            // nr_tid both RISE with distortion — DMOS-like despite
            // TID2013's MOS labels (nr_tid median rho +0.79).
            Self::NrLive | Self::NrTid => za::ScoreKind::Distortion,
        }
    }
}

/// WaDIQaM-FR (pair scorer: distorted against reference). Distance-
/// like: higher predicts more degradation relative to the reference —
/// an identical pair scores a nonzero, image-dependent floor (KonJND
/// identical-pair medians 0.019 live / 12.5 tid, ranges
/// −0.29..0.70 / 9.7..20.6; the `(0, h, h)` concat is not the head's
/// zero).
/// Input: packed sRGB8 (any stride); both images ≥ 32×32 and equal
/// size; at most 32,768 grid patches.
pub struct WadiqamFr {
    trunk: Box<dyn ImageGraph>,
    head: Box<dyn ImageGraph>,
    kind: WadiqamKind,
    min_size: usize,
}

/// WaDIQaM-NR. Same trunk and pooling as FR but trained no-reference;
/// both checkpoints are DMOS-like (higher = worse) — `nr_tid` is
/// TID2013 MOS-trained yet measured rising with distortion (median
/// Spearman rho +0.79 on a 12-source JPEG q95..q3 ladder;
/// benchmarks/wadiqam_port_2026-10-16.md). Wrapped in
/// [`za::PairedNoReference`] for pair scoring: `<col>` on the
/// distorted image, `<col>_ref` on the reference.
pub struct WadiqamNr {
    graph: Box<dyn ImageGraph>,
    kind: WadiqamKind,
    min_size: usize,
}

/// `⌊(dim−32)/32⌋+1` grid count; `0` when `dim < 32`.
fn grid(dim: usize) -> usize {
    if dim < PATCH {
        0
    } else {
        (dim - PATCH) / PATCH + 1
    }
}

/// `evaluate.py`'s `extract_patches`: non-overlapping 32×32 stride-32
/// grid, row-major `(gy, gx)` patch order, `[P,3,32,32]` f32 from raw
/// u8 (no normalisation).
fn extract_patches(img: &za::Rgb8In<'_>, min_size: usize) -> Result<(Vec<f32>, usize), Error> {
    let (h, w) = (img.height, img.width);
    if h < min_size || w < min_size {
        return Err(Error::TooSmall { min: min_size });
    }
    let (gh, gw) = (grid(h), grid(w));
    let n = gh * gw;
    if n == 0 {
        return Err(Error::TooSmall { min: min_size });
    }
    if n > MAX_PATCHES {
        return Err(Error::TooLarge { limit: MAX_PATCHES });
    }
    let mut out = vec![0.0f32; n * 3 * PATCH * PATCH];
    let stride = img.stride;
    for gy in 0..gh {
        for gx in 0..gw {
            let d = &mut out[(gy * gw + gx) * 3 * PATCH * PATCH..][..3 * PATCH * PATCH];
            for py in 0..PATCH {
                let row = &img.pixels[(gy * PATCH + py) * stride + (gx * PATCH) * 3..];
                for px in 0..PATCH {
                    let base = py * PATCH + px;
                    d[base] = row[px * 3] as f32;
                    d[PATCH * PATCH + base] = row[px * 3 + 1] as f32;
                    d[2 * PATCH * PATCH + base] = row[px * 3 + 2] as f32;
                }
            }
        }
    }
    Ok((out, n))
}

/// Upstream's pooling, `Σ(y·a)/Σa` over the `ya[2,P]` output in f64.
fn pool(ya: &[f32], p: usize) -> f64 {
    let (y, a) = ya.split_at(p);
    let (num, den) = y
        .iter()
        .zip(a)
        .fold((0.0f64, 0.0f64), |(ns, ds), (&y, &a)| {
            (ns + f64::from(y) * f64::from(a), ds + f64::from(a))
        });
    num / den
}

fn load_graph(
    dir: &Path,
    file: &str,
    engine: Engine,
    #[allow(unused_variables)] output: &'static str,
) -> Result<Box<dyn ImageGraph>, Error> {
    let path = dir.join(file);
    match engine {
        Engine::Rten => engine::load_image(&path),
        #[cfg(feature = "ort")]
        Engine::OrtCuda => OrtCudaImage::load(&path, output).map(|g| Box::new(g) as _),
    }
}

/// Run `input` through `graph`; the output tensor's shape must equal
/// `want` exactly (`trunk` → `[P,512]`, `head`/NR graph → `ya[2,P]`).
fn run(
    graph: &dyn ImageGraph,
    input: NamedInput,
    want: &[usize],
    what: &str,
) -> Result<Vec<f32>, Error> {
    let (shape, data) = graph.tensor(&[input])?;
    if shape != want {
        return Err(Error::BadModel(format!(
            "{what}: want {want:?}, got {shape:?}"
        )));
    }
    Ok(data)
}

/// Run `graph` on `x` (`n` patches, `in_row` f32 per patch, input name
/// `input_name`, input shape `in_shape(cn)`) in ≤[`PATCH_CHUNK`]-patch
/// calls — upstream's evaluate.py batch size — so the conv
/// intermediates stay bounded. Output must be `ya[2, cn]`; the two
/// halves of `ya` each concatenate in patch order, so the result is
/// bit-identical to a single call.
fn run_patches_ya(
    graph: &dyn ImageGraph,
    input_name: &'static str,
    in_shape: impl Fn(usize) -> Vec<usize>,
    x: &[f32],
    n: usize,
    in_row: usize,
    what: &str,
) -> Result<Vec<f32>, Error> {
    let mut ya = vec![0.0f32; 2 * n];
    let mut p = 0usize;
    for chunk in x.chunks(PATCH_CHUNK * in_row) {
        let cn = chunk.len() / in_row;
        let out = run(
            graph,
            NamedInput {
                name: input_name,
                shape: in_shape(cn),
                data: chunk.to_vec(),
            },
            &[2, cn],
            what,
        )?;
        ya[p..p + cn].copy_from_slice(&out[..cn]);
        ya[n + p..n + p + cn].copy_from_slice(&out[cn..]);
        p += cn;
    }
    Ok(ya)
}

/// As [`run_patches_ya`] for a `x[cn,3,32,32]` → `h[cn,FEAT]` trunk:
/// row-major per-patch outputs append in order.
fn run_patches_rows(
    graph: &dyn ImageGraph,
    x: &[f32],
    n: usize,
    what: &str,
) -> Result<Vec<f32>, Error> {
    let mut h = Vec::with_capacity(n * FEAT);
    for chunk in x.chunks(PATCH_CHUNK * 3 * PATCH * PATCH) {
        let cn = chunk.len() / (3 * PATCH * PATCH);
        h.extend(run(
            graph,
            NamedInput {
                name: "x",
                shape: vec![cn, 3, PATCH, PATCH],
                data: chunk.to_vec(),
            },
            &[cn, FEAT],
            what,
        )?);
    }
    Ok(h)
}

impl WadiqamFr {
    /// The released weights' published terms: the `models/*.model`
    /// files ship inside the MIT-licensed `dmaniry/deepIQA` repository
    /// (upstream licence in `crates/zenmetrics-nn/LICENSE-wadiqam`).
    pub const WEIGHTS: WeightsTerms = WeightsTerms {
        model: "wadiqam_fr",
        licence: WeightsLicence::Permissive,
        terms_url: "https://github.com/dmaniry/deepIQA/tree/master/models",
    };

    /// Load `wadiqam_fr_{live,tid}_{trunk,head}.onnx` + `.meta` from
    /// `dir` on `engine`.
    ///
    /// # Errors
    /// As [`crate::Lpips::from_dir`]; `BadModel` when `kind` is an
    /// `nr_*` checkpoint.
    pub fn from_dir(
        dir: impl AsRef<Path>,
        engine: Engine,
        kind: WadiqamKind,
    ) -> Result<Self, Error> {
        if !kind.is_fr() {
            return Err(Error::BadModel(format!(
                "WadiqamFr wants a fr_* kind, got {}",
                kind.column()
            )));
        }
        let dir = dir.as_ref();
        let column = kind.column();
        let (_channels, min_size) = read_meta(&dir.join(format!("{column}.meta")))?;
        Ok(Self {
            trunk: load_graph(dir, &format!("{column}_trunk.onnx"), engine, "h")?,
            head: load_graph(dir, &format!("{column}_head.onnx"), engine, "ya")?,
            kind,
            min_size,
        })
    }

    /// Trunk features `[P,512]` for one image — the reference-side
    /// work `warm_reference` hoists. Runs in ≤[`PATCH_CHUNK`]-patch
    /// calls like upstream's evaluate.py batching.
    fn features(&self, img: &za::Rgb8In<'_>) -> Result<(Vec<f32>, usize), Error> {
        let (x, n) = extract_patches(img, self.min_size)?;
        let h = run_patches_rows(
            &*self.trunk,
            &x,
            n,
            &format!("{} trunk", self.kind.column()),
        )?;
        Ok((h, n))
    }

    /// Head on the upstream concat `(h − h_ref, h, h_ref)` → pooled score.
    fn score_against(&self, d: &za::Rgb8In<'_>, hr: &[f32], n: usize) -> Result<f64, Error> {
        let (h, nd) = self.features(d)?;
        if nd != n {
            return Err(Error::SizeMismatch);
        }
        let mut f = vec![0.0f32; n * 3 * FEAT];
        for p in 0..n {
            let d3 = &mut f[p * 3 * FEAT..(p + 1) * 3 * FEAT];
            let (h3, r3) = (&h[p * FEAT..(p + 1) * FEAT], &hr[p * FEAT..(p + 1) * FEAT]);
            for i in 0..FEAT {
                d3[i] = h3[i] - r3[i];
                d3[FEAT + i] = h3[i];
                d3[2 * FEAT + i] = r3[i];
            }
        }
        let ya = run_patches_ya(
            &*self.head,
            "f",
            |cn| vec![cn, 3 * FEAT],
            &f,
            n,
            3 * FEAT,
            &format!("{} head", self.kind.column()),
        )?;
        Ok(pool(&ya, n))
    }
}

impl za::Metric for WadiqamFr {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(self.kind.column(), za::implementation!(SCORE_REVISION)).with_reference(
            format!(
                "dmaniry/deepIQA {} weighted, native-res 32x32 grid; engine {}",
                self.kind.column(),
                self.trunk.engine()
            ),
        )
    }

    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        Cow::Owned(vec![za::ScoreSpec::new(
            self.kind.column(),
            self.kind.score_kind(),
        )])
    }

    fn accepts(&self, format: &za::InputFormat) -> bool {
        za::is_rgb8_srgb(format)
    }

    fn evaluate(
        &mut self,
        r: za::Image<'_>,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map(self.kind.column())?;
        let (r, d) = za::rgb8_pair(&r, &d)?;
        let (hr, n) = self
            .features(&r)
            .metric_err_with(self.kind.column(), Error::kind)?;
        let s = self
            .score_against(&d, &hr, n)
            .metric_err_with(self.kind.column(), Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(self.kind.column(), s)))
    }

    fn warm_reference<'m>(
        &'m mut self,
        r: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::WarmReference + 'm>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        let view = za::rgb8_view(&r)?;
        let (hr, n) = self
            .features(&view)
            .metric_err_with(self.kind.column(), Error::kind)?;
        Ok(Box::new(WarmWadiqamFr {
            metric: self,
            reference: hr,
            n,
            size: (view.width, view.height),
        }))
    }
}

/// The reference's patch features, computed once: every candidate then
/// needs only its own trunk pass, the host concat and the head.
struct WarmWadiqamFr<'m> {
    metric: &'m WadiqamFr,
    reference: Vec<f32>,
    n: usize,
    size: (usize, usize),
}

impl za::WarmReference for WarmWadiqamFr<'_> {
    fn evaluate(
        &mut self,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map(self.metric.kind.column())?;
        let d = za::rgb8_view(&d)?;
        if (d.width, d.height) != self.size {
            return Err(Error::SizeMismatch)
                .metric_err_with(self.metric.kind.column(), Error::kind);
        }
        let s = self
            .metric
            .score_against(&d, &self.reference, self.n)
            .metric_err_with(self.metric.kind.column(), Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(
            self.metric.kind.column(),
            s,
        )))
    }

    fn reuses_reference_work(&self) -> bool {
        true
    }
}

impl WadiqamNr {
    /// The released weights' published terms (as [`WadiqamFr::WEIGHTS`]).
    pub const WEIGHTS: WeightsTerms = WeightsTerms {
        model: "wadiqam_nr",
        licence: WeightsLicence::Permissive,
        terms_url: "https://github.com/dmaniry/deepIQA/tree/master/models",
    };

    /// Load `wadiqam_nr_{live,tid}.onnx` + `.meta` from `dir` on `engine`.
    ///
    /// # Errors
    /// As [`crate::Lpips::from_dir`]; `BadModel` when `kind` is an
    /// `fr_*` checkpoint.
    pub fn from_dir(
        dir: impl AsRef<Path>,
        engine: Engine,
        kind: WadiqamKind,
    ) -> Result<Self, Error> {
        if kind.is_fr() {
            return Err(Error::BadModel(format!(
                "WadiqamNr wants an nr_* kind, got {}",
                kind.column()
            )));
        }
        let dir = dir.as_ref();
        let column = kind.column();
        let (_channels, min_size) = read_meta(&dir.join(format!("{column}.meta")))?;
        Ok(Self {
            graph: load_graph(dir, &format!("{column}.onnx"), engine, "ya")?,
            kind,
            min_size,
        })
    }

    fn score_image(&self, img: &za::Rgb8In<'_>) -> Result<f64, Error> {
        let (x, n) = extract_patches(img, self.min_size)?;
        let ya = run_patches_ya(
            &*self.graph,
            "x",
            |cn| vec![cn, 3, PATCH, PATCH],
            &x,
            n,
            3 * PATCH * PATCH,
            self.kind.column(),
        )?;
        Ok(pool(&ya, n))
    }
}

impl za::NoReferenceMetric for WadiqamNr {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(self.kind.column(), za::implementation!(SCORE_REVISION)).with_reference(
            format!(
                "dmaniry/deepIQA {} weighted, native-res 32x32 grid; engine {}",
                self.kind.column(),
                self.graph.engine()
            ),
        )
    }

    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        Cow::Owned(vec![za::ScoreSpec::new(
            self.kind.column(),
            self.kind.score_kind(),
        )])
    }

    fn accepts(&self, format: &za::InputFormat) -> bool {
        za::is_rgb8_srgb(format)
    }

    fn evaluate_image(
        &mut self,
        image: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map(self.kind.column())?;
        let img = za::rgb8_view(&image)?;
        let s = self
            .score_image(&img)
            .metric_err_with(self.kind.column(), Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(self.kind.column(), s)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view<'a>(px: &'a [u8], w: usize, h: usize) -> za::Rgb8In<'a> {
        za::Rgb8In {
            pixels: px,
            width: w,
            height: h,
            stride: w * 3,
        }
    }

    #[test]
    fn patch_grid_matches_as_strided() {
        // deepIQA: ((shape-32)//32)+1 per axis, edge partials dropped.
        assert_eq!(grid(32), 1);
        assert_eq!(grid(63), 1);
        assert_eq!(grid(64), 2);
        assert_eq!(grid(320), 10);
        assert_eq!(grid(333), 10); // 333-32=301, //32=9, +1=10
        assert_eq!(grid(31), 0);
    }

    #[test]
    // Indices are written p*(3*1024) + c*1024 + i to document the
    // [P,3,32,32] layout; clippy's identity_op/erasing_op would fold
    // away the layout terms.
    #[allow(clippy::identity_op, clippy::erasing_op)]
    fn extract_patches_is_nchw_patch_major() {
        // 64x48 -> gh=2, gw=1 -> two patches; [P,3,32,32] layout:
        // out[p*3072 + c*1024 + y*32 + x].
        let (w, h) = (64usize, 48usize);
        let mut px = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                px[(y * w + x) * 3] = (x % 256) as u8;
                px[(y * w + x) * 3 + 1] = (y % 256) as u8;
                px[(y * w + x) * 3 + 2] = 200;
            }
        }
        let (out, n) = extract_patches(&view(&px, w, h), 32).unwrap();
        assert_eq!(n, 2);
        assert_eq!(out.len(), 2 * 3 * 32 * 32);
        // patch 1, channel 0, pixel (0,0) = image[0, 32].r = 32
        assert_eq!(out[1 * 3 * 1024 + 0 * 1024 + 0], 32.0);
        // patch 0, channel 0, (0,0) = image[0,0].r = 0
        assert_eq!(out[0], 0.0);
        // patch 0, channel 1, (1,1) = image[1,1].g = 1
        assert_eq!(out[1024 + 32 + 1], 1.0);
        // patch 1, channel 2, (15,15) = image[15,47].b = 200
        assert_eq!(out[1 * 3 * 1024 + 2 * 1024 + 15 * 32 + 15], 200.0);
    }

    /// A padded-stride view of a wider buffer extracts the same patches
    /// as the packed view; the stride tail is never read.
    #[test]
    fn extract_patches_honours_stride() {
        let (w, h, stride) = (96usize, 64usize, 96 * 3 + 17);
        let mut wide = vec![0u8; stride * h];
        let mut packed = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let v = ((x * 7 + y * 11 + c * 13) % 251) as u8;
                    wide[y * stride + x * 3 + c] = v;
                    packed[(y * w + x) * 3 + c] = v;
                }
            }
        }
        let strided = za::Rgb8In {
            pixels: &wide,
            width: w,
            height: h,
            stride,
        };
        assert_eq!(
            extract_patches(&strided, 32).unwrap(),
            extract_patches(&view(&packed, w, h), 32).unwrap()
        );
    }

    #[test]
    fn size_bounds() {
        let px = vec![0u8; 4096];
        for (w, h) in [(31, 31), (1, 64), (16, 32)] {
            let img = view(&px[..w * h * 3], w, h);
            assert!(matches!(
                extract_patches(&img, 32),
                Err(Error::TooSmall { min: 32 })
            ));
        }
        // 8000x8000 -> 249*249 = 62001 patches > 32768
        assert!(matches!(
            extract_patches(
                &za::Rgb8In {
                    pixels: &[],
                    width: 8000,
                    height: 8000,
                    stride: 8000 * 3,
                },
                32
            ),
            Err(Error::TooLarge { limit: MAX_PATCHES })
        ));
    }

    #[test]
    fn pool_is_weighted_mean() {
        let ya = [1.0f32, 2.0, 3.0, 2.0, 1.0, 1.0]; // y=[1,2,3], a=[2,1,1]
        assert_eq!(pool(&ya, 3), (2.0 + 2.0 + 3.0) / 4.0);
    }

    /// A fake `x[cn, R]` → `ya[2, cn]` graph whose patch outputs carry
    /// their index, so a chunked concat can be checked for order and
    /// completeness. Out[0][p] = in[p][0], Out[1][p] = in[p][R-1].
    struct FakeYa;
    impl crate::engine::ImageGraph for FakeYa {
        fn engine(&self) -> &'static str {
            "test"
        }
        fn score(&self, _: &[NamedInput]) -> Result<f64, Error> {
            unreachable!()
        }
        fn tensor(&self, inputs: &[NamedInput]) -> Result<(Vec<usize>, Vec<f32>), Error> {
            let i = &inputs[0];
            let cn = i.shape[0];
            let row = i.data.len() / cn;
            let mut out = vec![0.0f32; 2 * cn];
            for p in 0..cn {
                out[p] = i.data[p * row];
                out[cn + p] = i.data[p * row + row - 1];
            }
            Ok((vec![2, cn], out))
        }
    }

    /// A fake `x[cn,3,32,32]` → `h[cn,512]` trunk: row p echoes the
    /// patch's first FEAT inputs — identical per-patch outputs for any
    /// batch split (outputs must be input-derived, not index-derived,
    /// or a chunked call's local `p` would shift the encoding).
    struct FakeTrunk;
    impl crate::engine::ImageGraph for FakeTrunk {
        fn engine(&self) -> &'static str {
            "test"
        }
        fn score(&self, _: &[NamedInput]) -> Result<f64, Error> {
            unreachable!()
        }
        fn tensor(&self, inputs: &[NamedInput]) -> Result<(Vec<usize>, Vec<f32>), Error> {
            let i = &inputs[0];
            let cn = i.shape[0];
            let mut h = vec![0.0f32; cn * FEAT];
            for p in 0..cn {
                for j in 0..FEAT {
                    h[p * FEAT + j] = i.data[p * 3 * PATCH * PATCH + j];
                }
            }
            Ok((vec![cn, FEAT], h))
        }
    }

    /// The 2000-patch batching must be a pure re-batch: chunked output
    /// equals the single-call output bit for bit, in patch order.
    #[test]
    fn chunking_preserves_patch_order_and_values() {
        let n = PATCH_CHUNK + 37; // two chunks: 2000 + 37
        // ya path (NR graph / FR head): x is [n, in_row].
        let in_row = 5usize;
        let x: Vec<f32> = (0..n * in_row)
            .map(|i| (i * 31 % 977) as f32 - 488.0)
            .collect();
        let chunked =
            run_patches_ya(&FakeYa, "x", |cn| vec![cn, in_row], &x, n, in_row, "t").unwrap();
        let want: Vec<f32> = (0..n)
            .map(|p| x[p * in_row])
            .chain((0..n).map(|p| x[p * in_row + in_row - 1]))
            .collect();
        assert_eq!(chunked, want);
        // h path (FR trunk): x is [n, 3*32*32]; each patch's leading
        // values encode (global patch, column) so a mis-ordered or
        // truncated chunk concat fails the equality below.
        let row = 3 * PATCH * PATCH;
        let xr: Vec<f32> = (0..n * row)
            .map(|i| {
                let (p, j) = (i / row, i % row);
                if j < FEAT {
                    (p * 1024 + j) as f32
                } else {
                    -7.0
                }
            })
            .collect();
        let chunked = run_patches_rows(&FakeTrunk, &xr, n, "t").unwrap();
        let whole = FakeTrunk
            .tensor(&[NamedInput {
                name: "x",
                shape: vec![n, 3, PATCH, PATCH],
                data: xr.clone(),
            }])
            .unwrap()
            .1;
        assert_eq!(chunked, whole);
    }
}
