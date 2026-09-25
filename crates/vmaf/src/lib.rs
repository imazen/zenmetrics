#![forbid(unsafe_code)]

mod adm;
mod cambi;
mod score;
mod speed;
mod v0;
mod v0_score;
mod vif;

pub use adm::{adm2_v0_from_luma, adm3_v1_from_luma};
pub use cambi::cambi_v1_from_luma;
pub use score::{
    PoolingMethod, VmafFrameResult, VmafV1Scorer, VmafV1Stream, Yuv420Frame, pool_v1_scores,
    score_v1_420,
};
pub use speed::speed_v1_chroma_420;
pub use v0::{VmafV0Features, VmafV0Model, VmafV0Variant};
pub use v0_score::{VmafV0FrameResult, VmafV0Scorer, VmafV0Stream, pool_v0_scores, score_v0_420};
pub use vif::vif_v0_from_luma;

use std::error::Error as StdError;
use std::fmt;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use archmage::intrinsics::x86_64::*;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use archmage::{SimdToken, X64V3Token, arcane, rite};
#[cfg(all(feature = "simd", feature = "avx512", target_arch = "x86_64"))]
use archmage::X64V4Token;
#[cfg(feature = "simd")]
use archmage::{autoversion, magetypes};

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline(always)]
fn a8<T, const N: usize>(s: &[T]) -> &[T; N] {
    s.try_into().unwrap()
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline(always)]
fn a8m<T, const N: usize>(s: &mut [T]) -> &mut [T; N] {
    s.try_into().unwrap()
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[inline(always)]
fn v3_token() -> Option<X64V3Token> {
    X64V3Token::summon()
}

#[cfg(all(feature = "simd", feature = "avx512", target_arch = "x86_64"))]
#[inline(always)]
fn v4_token() -> Option<X64V4Token> {
    X64V4Token::summon()
}

#[derive(Debug)]
pub enum Error {
    InvalidModel(&'static str),
    NonFiniteInput(&'static str),
    InvalidInput(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidModel(msg) => write!(f, "invalid model: {msg}"),
            Error::NonFiniteInput(name) => write!(f, "non-finite input feature: {name}"),
            Error::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl StdError for Error {}

#[derive(Copy, Clone, Debug)]
pub enum ModelVariant {
    Standard1080p,
    Phone,
    Default4k,
    Consumer4k,
    HfrStandard1080p,
    HfrPhone,
    HfrDefault4k,
    HfrConsumer4k,
}

impl ModelVariant {
    pub const V1_NEG: Self = Self::Standard1080p;

    pub fn built_in_name(&self) -> &'static str {
        match self {
            ModelVariant::Standard1080p => "vmaf_v1.0.16_3d0h",
            ModelVariant::Phone => "vmaf_v1.0.16_5d0h",
            ModelVariant::Default4k => "vmaf_v1.0.16_1d5h_2160",
            ModelVariant::Consumer4k => "vmaf_v1.0.16_3d0h_2160",
            ModelVariant::HfrStandard1080p => "vmaf_v1.0.16_hfr_3d0h",
            ModelVariant::HfrPhone => "vmaf_v1.0.16_hfr_5d0h",
            ModelVariant::HfrDefault4k => "vmaf_v1.0.16_hfr_1d5h_2160",
            ModelVariant::HfrConsumer4k => "vmaf_v1.0.16_hfr_3d0h_2160",
        }
    }

    fn json(&self) -> &'static str {
        match self {
            ModelVariant::Standard1080p => {
                include_str!("../models/vmaf_v1.0.16_3d0h.json")
            }
            ModelVariant::Phone => include_str!("../models/vmaf_v1.0.16_5d0h.json"),
            ModelVariant::Default4k => {
                include_str!("../models/vmaf_v1.0.16_1d5h_2160.json")
            }
            ModelVariant::Consumer4k => {
                include_str!("../models/vmaf_v1.0.16_3d0h_2160.json")
            }
            ModelVariant::HfrStandard1080p => {
                include_str!("../models/vmaf_v1.0.16_hfr_3d0h.json")
            }
            ModelVariant::HfrPhone => {
                include_str!("../models/vmaf_v1.0.16_hfr_5d0h.json")
            }
            ModelVariant::HfrDefault4k => {
                include_str!("../models/vmaf_v1.0.16_hfr_1d5h_2160.json")
            }
            ModelVariant::HfrConsumer4k => {
                include_str!("../models/vmaf_v1.0.16_hfr_3d0h_2160.json")
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VmafFeatures {
    pub cambi: f64,
    pub speed_chroma_uv: f64,
    pub adm3: f64,
    pub motion3: f64,
}

const FEATURE_NAMES: [&str; 4] = [
    "Cambi_feature_cambi_score",
    "Speed_chroma_feature_speed_chroma_uv_score",
    "VMAF_integer_feature_adm3_score",
    "VMAF_integer_feature_motion3_score",
];

const SPEED_CHROMA: usize = 1;
const ADM3: usize = 2;

struct SupportVector<const N: usize> {
    alpha: f64,
    features: [f64; N],
}

pub struct VmafModel {
    slopes: [f64; 5],
    intercepts: [f64; 5],
    gamma: f64,
    rho: f64,
    support_vectors: Vec<SupportVector<4>>,
    chroma_correction_parameter: Option<f64>,
    score_transform: Option<[f64; 3]>,
    score_clip: [f64; 2],
}

impl VmafModel {
    pub fn new(variant: ModelVariant) -> Result<Self, Error> {
        let root: serde_json::Value = serde_json::from_str(variant.json())
            .map_err(|_| Error::InvalidModel("malformed JSON"))?;
        let model_dict = root
            .get("model_dict")
            .ok_or(Error::InvalidModel("missing model_dict"))?;

        if model_dict.get("model_type").and_then(|v| v.as_str()) != Some("LIBSVMNUSVR") {
            return Err(Error::InvalidModel("model_type is not LIBSVMNUSVR"));
        }
        if model_dict.get("norm_type").and_then(|v| v.as_str()) != Some("linear_rescale") {
            return Err(Error::InvalidModel("norm_type is not linear_rescale"));
        }

        let feature_names = model_dict
            .get("feature_names")
            .and_then(|v| v.as_array())
            .ok_or(Error::InvalidModel("missing feature_names"))?;
        if feature_names.len() != 4
            || feature_names
                .iter()
                .zip(FEATURE_NAMES.iter())
                .any(|(a, b)| a.as_str() != Some(*b))
        {
            return Err(Error::InvalidModel("unexpected feature_names"));
        }

        let slopes = get_f64_array::<5>(model_dict, "slopes")?;
        let intercepts = get_f64_array::<5>(model_dict, "intercepts")?;

        let svm_text = model_dict
            .get("model")
            .and_then(|v| v.as_str())
            .ok_or(Error::InvalidModel("missing svm model text"))?;
        let (gamma, rho, support_vectors) = parse_svm::<4>(svm_text)?;

        let chroma_correction_parameter = model_dict
            .get("chroma_correction_parameter")
            .map(|v| {
                let c = v.as_f64().ok_or(Error::InvalidModel(
                    "chroma_correction_parameter is not a number",
                ))?;
                if !c.is_finite() {
                    return Err(Error::InvalidModel(
                        "chroma_correction_parameter is not finite",
                    ));
                }
                Ok(c)
            })
            .transpose()?;

        let score_transform = model_dict
            .get("score_transform")
            .map(|st| {
                let enabled = st
                    .get("enabled")
                    .and_then(|v| v.as_bool())
                    .ok_or(Error::InvalidModel("score_transform.enabled missing"))?;
                if !enabled {
                    return Ok(None);
                }
                let mut p = [0.0; 3];
                for (i, key) in ["p0", "p1", "p2"].iter().enumerate() {
                    p[i] = st
                        .get(key)
                        .and_then(|v| v.as_f64())
                        .ok_or(Error::InvalidModel("score_transform p missing"))?;
                    if !p[i].is_finite() {
                        return Err(Error::InvalidModel("score_transform p non-finite"));
                    }
                }
                Ok(Some(p))
            })
            .transpose()?
            .flatten();

        let clip = model_dict
            .get("score_clip")
            .and_then(|v| v.as_array())
            .ok_or(Error::InvalidModel("missing score_clip"))?;
        if clip.len() != 2 {
            return Err(Error::InvalidModel("score_clip must have 2 bounds"));
        }
        let lo = clip[0]
            .as_f64()
            .ok_or(Error::InvalidModel("bad score_clip"))?;
        let hi = clip[1]
            .as_f64()
            .ok_or(Error::InvalidModel("bad score_clip"))?;
        if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
            return Err(Error::InvalidModel("bad score_clip bounds"));
        }

        Ok(Self {
            slopes,
            intercepts,
            gamma,
            rho,
            support_vectors,
            chroma_correction_parameter,
            score_transform,
            score_clip: [lo, hi],
        })
    }

    pub fn predict(&self, features: VmafFeatures) -> Result<f64, Error> {
        let raw = [
            features.cambi,
            features.speed_chroma_uv,
            features.adm3,
            features.motion3,
        ];
        for (i, value) in raw.iter().enumerate() {
            if !value.is_finite() {
                return Err(Error::NonFiniteInput(FEATURE_NAMES[i]));
            }
        }

        let mut node = [0.0f64; 4];
        for i in 0..4 {
            node[i] = self.slopes[i + 1] * raw[i] + self.intercepts[i + 1];
        }

        if let Some(ccp) = self.chroma_correction_parameter {
            let guided_denorm = (node[SPEED_CHROMA] - self.intercepts[SPEED_CHROMA + 1])
                / self.slopes[SPEED_CHROMA + 1];
            if guided_denorm == 0.0 {
                let guiding_denorm =
                    (node[ADM3] - self.intercepts[ADM3 + 1]) / self.slopes[ADM3 + 1];
                let corrected = (-ccp * guiding_denorm) + ccp;
                node[SPEED_CHROMA] =
                    self.slopes[SPEED_CHROMA + 1] * corrected + self.intercepts[SPEED_CHROMA + 1];
            }
        }

        let mut svm = 0.0;
        for sv in &self.support_vectors {
            let d0 = sv.features[0] - node[0];
            let d1 = sv.features[1] - node[1];
            let d2 = sv.features[2] - node[2];
            let d3 = sv.features[3] - node[3];
            let dist2 = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
            svm += sv.alpha * (-self.gamma * dist2).exp();
        }
        svm -= self.rho;

        let mut prediction = (svm - self.intercepts[0]) / self.slopes[0];

        if let Some([p0, p1, p2]) = self.score_transform {
            prediction = p0 + p1 * prediction + p2 * prediction * prediction;
        }

        if prediction < self.score_clip[0] {
            prediction = self.score_clip[0];
        }
        if prediction > self.score_clip[1] {
            prediction = self.score_clip[1];
        }

        Ok(prediction)
    }
}

pub(crate) fn get_f64_array<const N: usize>(
    model_dict: &serde_json::Value,
    key: &str,
) -> Result<[f64; N], Error> {
    let arr = model_dict
        .get(key)
        .and_then(|v| v.as_array())
        .ok_or(Error::InvalidModel("missing numeric array"))?;
    if arr.len() != N {
        return Err(Error::InvalidModel("unexpected array length"));
    }
    let mut out = [0.0; N];
    for (i, v) in arr.iter().enumerate() {
        out[i] = v
            .as_f64()
            .ok_or(Error::InvalidModel("non-numeric array element"))?;
        if !out[i].is_finite() {
            return Err(Error::InvalidModel("non-finite array element"));
        }
    }
    Ok(out)
}

fn parse_svm<const N: usize>(text: &str) -> Result<(f64, f64, Vec<SupportVector<N>>), Error> {
    let mut gamma = None;
    let mut rho = None;
    let mut total_sv = None;
    let mut svm_type = None;
    let mut kernel_type = None;
    let mut svs = Vec::new();
    let mut in_sv = false;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if in_sv {
            let mut parts = line.split_whitespace();
            let alpha: f64 = parts
                .next()
                .and_then(|t| t.parse::<f64>().ok())
                .filter(|v| v.is_finite())
                .ok_or(Error::InvalidModel("bad SV coefficient"))?;
            let mut features = [0.0f64; N];
            let mut seen = [false; N];
            for token in parts {
                let (idx, val) = token
                    .split_once(':')
                    .ok_or(Error::InvalidModel("bad SV pair"))?;
                let idx: usize = idx
                    .parse()
                    .ok()
                    .ok_or(Error::InvalidModel("bad SV index"))?;
                if !(1..=N).contains(&idx) {
                    return Err(Error::InvalidModel("SV index out of range"));
                }
                if seen[idx - 1] {
                    return Err(Error::InvalidModel("duplicate SV index"));
                }
                let val: f64 = val
                    .parse()
                    .ok()
                    .filter(|v: &f64| v.is_finite())
                    .ok_or(Error::InvalidModel("bad SV value"))?;
                seen[idx - 1] = true;
                features[idx - 1] = val;
            }
            svs.push(SupportVector { alpha, features });
            continue;
        }
        if line == "SV" {
            in_sv = true;
            continue;
        }
        let (key, value) = line
            .split_once(' ')
            .ok_or(Error::InvalidModel("bad svm header line"))?;
        match key {
            "svm_type" => svm_type = Some(value),
            "kernel_type" => kernel_type = Some(value),
            "gamma" => {
                gamma = Some(
                    value
                        .parse::<f64>()
                        .ok()
                        .filter(|v| v.is_finite())
                        .ok_or(Error::InvalidModel("bad gamma"))?,
                )
            }
            "rho" => {
                rho = Some(
                    value
                        .parse::<f64>()
                        .ok()
                        .filter(|v| v.is_finite())
                        .ok_or(Error::InvalidModel("bad rho"))?,
                )
            }
            "total_sv" => {
                total_sv = Some(
                    value
                        .parse::<usize>()
                        .ok()
                        .ok_or(Error::InvalidModel("bad total_sv"))?,
                )
            }
            _ => {}
        }
    }

    if svm_type != Some("nu_svr") {
        return Err(Error::InvalidModel("svm_type is not nu_svr"));
    }
    if kernel_type != Some("rbf") {
        return Err(Error::InvalidModel("kernel_type is not rbf"));
    }
    let gamma = gamma.ok_or(Error::InvalidModel("missing gamma"))?;
    let rho = rho.ok_or(Error::InvalidModel("missing rho"))?;
    let total_sv = total_sv.ok_or(Error::InvalidModel("missing total_sv"))?;
    if svs.len() != total_sv {
        return Err(Error::InvalidModel("SV count does not match total_sv"));
    }
    Ok((gamma, rho, svs))
}

const MOTION_FILTER: [i64; 5] = [3571, 16004, 26386, 16004, 3571];
pub(crate) const MOTION_MAX_VAL: f64 = 18.0;
const MOTION_V0_MAX_VAL: f64 = 10000.0;

fn mirror(idx: isize, size: usize) -> usize {
    let size = size as isize;
    if idx < 0 {
        (-idx) as usize
    } else if idx >= size {
        (2 * size - idx - 2) as usize
    } else {
        idx as usize
    }
}

#[cfg(feature = "simd")]
#[magetypes(define(u16x16, u32x8, i32x8), v3, neon, wasm128, scalar)]
fn motion_vertical_simd(
    token: Token,
    prev_rows: &[&[u16]; 5],
    cur_rows: &[&[u16]; 5],
    coef: &[i64; 5],
    y_round: i32,
    bpc: u32,
    out: &mut [i32; 16],
) {
    let mut acc_l = i32x8::zero(token);
    let mut acc_h = i32x8::zero(token);
    for k in 0..5 {
        let p = u16x16::load(token, prev_rows[k][..16].try_into().unwrap());
        let c = u16x16::load(token, cur_rows[k][..16].try_into().unwrap());
        let dl = p.widen_low().bitcast_i32x8() - c.widen_low().bitcast_i32x8();
        let dh = p.widen_high().bitcast_i32x8() - c.widen_high().bitcast_i32x8();
        let w = i32x8::splat(token, coef[k] as i32);
        acc_l += w * dl;
        acc_h += w * dh;
    }
    let rnd = i32x8::splat(token, y_round);
    let yl = (acc_l + rnd).shr_arithmetic_uniform(bpc);
    let yh = (acc_h + rnd).shr_arithmetic_uniform(bpc);
    out[..8].copy_from_slice(&yl.to_array());
    out[8..16].copy_from_slice(&yh.to_array());
}

/// Direct port of `motion_score_pipeline_8_avx2`'s phase-1 (vertical diff +
/// 5-tap convolution) for 8-bit planes: epi16 differences and mullo/mulhi
/// epi16 + unpack_epi16 products, `+128 >> 8` rounding, permute2x128 lane
/// reorder. Processes 16 columns per call; our u16 input holds values <= 255
/// so the direct u16 load equals C's `cvtepu8_epi16`.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
fn motion_vertical8_v3(
    _token: X64V3Token,
    prev_rows: &[&[u16]; 5],
    cur_rows: &[&[u16]; 5],
    out: &mut [i32; 16],
) {
    let f = [
        _mm256_set1_epi16(3571),
        _mm256_set1_epi16(16004),
        _mm256_set1_epi16(26386),
        _mm256_set1_epi16(16004),
        _mm256_set1_epi16(3571),
    ];
    let round8 = _mm256_set1_epi32(1 << 7);
    let mut acc_lo = _mm256_setzero_si256();
    let mut acc_hi = _mm256_setzero_si256();
    for k in 0..5 {
        let d = _mm256_sub_epi16(
            _mm256_loadu_si256(a8::<u16, 16>(&prev_rows[k][..16])),
            _mm256_loadu_si256(a8::<u16, 16>(&cur_rows[k][..16])),
        );
        let lo = _mm256_mullo_epi16(d, f[k]);
        let hi = _mm256_mulhi_epi16(d, f[k]);
        acc_lo = _mm256_add_epi32(acc_lo, _mm256_unpacklo_epi16(lo, hi));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_unpackhi_epi16(lo, hi));
    }
    acc_lo = _mm256_srai_epi32(_mm256_add_epi32(acc_lo, round8), 8);
    acc_hi = _mm256_srai_epi32(_mm256_add_epi32(acc_hi, round8), 8);
    _mm256_storeu_si256(
        a8m::<i32, 8>(&mut out[..8]),
        _mm256_permute2x128_si256(acc_lo, acc_hi, 0x20),
    );
    _mm256_storeu_si256(
        a8m::<i32, 8>(&mut out[8..16]),
        _mm256_permute2x128_si256(acc_lo, acc_hi, 0x31),
    );
}

/// Direct port of `motion_score_pipeline_16_avx2`'s phase-1: epi32
/// differences, mullo_epi32 products (each fits i32 for bpc <= 16), i64
/// accumulation via cvtepi32_epi64, `+round >> bpc` via srlv_epi64 (the low
/// 32 bits identical to arithmetic shift for bpc < 32), permutevar pack.
/// Processes 8 columns per call and is exact for any bpc <= 16.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
fn motion_vertical16_v3(
    _token: X64V3Token,
    prev_rows: &[&[u16]; 5],
    cur_rows: &[&[u16]; 5],
    bpc: u32,
    out: &mut [i32; 8],
) {
    let g = [
        _mm256_set1_epi32(3571),
        _mm256_set1_epi32(16004),
        _mm256_set1_epi32(26386),
        _mm256_set1_epi32(16004),
        _mm256_set1_epi32(3571),
    ];
    let round64 = _mm256_set1_epi64x(1i64 << (bpc - 1));
    let bpc_vec = _mm256_set1_epi64x(bpc as i64);
    let perm_idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
    let mut prod = [_mm256_setzero_si256(); 5];
    for k in 0..5 {
        let d = _mm256_sub_epi32(
            _mm256_cvtepu16_epi32(_mm_loadu_si128(a8::<u16, 8>(&prev_rows[k][..8]))),
            _mm256_cvtepu16_epi32(_mm_loadu_si128(a8::<u16, 8>(&cur_rows[k][..8]))),
        );
        prod[k] = _mm256_mullo_epi32(d, g[k]);
    }
    let mut acc_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(prod[0]));
    for p in prod.iter().take(5).skip(1) {
        acc_lo = _mm256_add_epi64(acc_lo, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(*p)));
    }
    let mut acc_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(prod[0], 1));
    for p in prod.iter().take(5).skip(1) {
        acc_hi = _mm256_add_epi64(
            acc_hi,
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256(*p, 1)),
        );
    }
    acc_lo = _mm256_srlv_epi64(_mm256_add_epi64(acc_lo, round64), bpc_vec);
    acc_hi = _mm256_srlv_epi64(_mm256_add_epi64(acc_hi, round64), bpc_vec);
    let res_lo = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(acc_lo, perm_idx));
    let res_hi = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(acc_hi, perm_idx));
    _mm256_storeu_si256(
        out,
        _mm256_inserti128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1),
    );
}

/// Direct port of `motion_score_pipeline_8_avx512`'s phase-1 for 8-bit
/// planes: epi16 differences of 32 columns per call (u16 loads stand in for
/// C's `cvtepu8_epi16`), mullo/mulhi_epi16 + unpack_epi16 products,
/// `+128 >> 8`, then the four-way `permute2x128` reorder before storing.
#[cfg(all(feature = "simd", feature = "avx512", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
fn motion_vertical8_v4(
    _token: X64V4Token,
    prev_rows: &[&[u16]; 5],
    cur_rows: &[&[u16]; 5],
    out: &mut [i32; 32],
) {
    let f = [
        _mm512_set1_epi16(3571),
        _mm512_set1_epi16(16004),
        _mm512_set1_epi16(26386),
        _mm512_set1_epi16(16004),
        _mm512_set1_epi16(3571),
    ];
    let round8 = _mm512_set1_epi32(1 << 7);
    let mut acc_lo = _mm512_setzero_si512();
    let mut acc_hi = _mm512_setzero_si512();
    for k in 0..5 {
        let d = _mm512_sub_epi16(
            _mm512_loadu_si512(a8::<u16, 32>(&prev_rows[k][..32])),
            _mm512_loadu_si512(a8::<u16, 32>(&cur_rows[k][..32])),
        );
        let lo = _mm512_mullo_epi16(d, f[k]);
        let hi = _mm512_mulhi_epi16(d, f[k]);
        acc_lo = _mm512_add_epi32(acc_lo, _mm512_unpacklo_epi16(lo, hi));
        acc_hi = _mm512_add_epi32(acc_hi, _mm512_unpackhi_epi16(lo, hi));
    }
    acc_lo = _mm512_srai_epi32(_mm512_add_epi32(acc_lo, round8), 8);
    acc_hi = _mm512_srai_epi32(_mm512_add_epi32(acc_hi, round8), 8);
    let lo_lo = _mm512_castsi512_si256(acc_lo);
    let lo_hi = _mm512_extracti64x4_epi64(acc_lo, 1);
    let hi_lo = _mm512_castsi512_si256(acc_hi);
    let hi_hi = _mm512_extracti64x4_epi64(acc_hi, 1);
    _mm256_storeu_si256(
        a8m::<i32, 8>(&mut out[..8]),
        _mm256_permute2x128_si256(lo_lo, hi_lo, 0x20),
    );
    _mm256_storeu_si256(
        a8m::<i32, 8>(&mut out[8..16]),
        _mm256_permute2x128_si256(lo_lo, hi_lo, 0x31),
    );
    _mm256_storeu_si256(
        a8m::<i32, 8>(&mut out[16..24]),
        _mm256_permute2x128_si256(lo_hi, hi_hi, 0x20),
    );
    _mm256_storeu_si256(
        a8m::<i32, 8>(&mut out[24..32]),
        _mm256_permute2x128_si256(lo_hi, hi_hi, 0x31),
    );
}

/// Direct port of `motion_score_pipeline_16_avx512`'s phase-1: epi32
/// differences over 16 columns per call, mullo_epi32 products, i64
/// accumulation, and native `srav_epi64` rounding (AVX-512F has the
/// variable i64 shift AVX2 lacked). Exact for any bpc <= 16.
#[cfg(all(feature = "simd", feature = "avx512", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
fn motion_vertical16_v4(
    _token: X64V4Token,
    prev_rows: &[&[u16]; 5],
    cur_rows: &[&[u16]; 5],
    bpc: u32,
    out: &mut [i32; 16],
) {
    let g = [
        _mm512_set1_epi32(3571),
        _mm512_set1_epi32(16004),
        _mm512_set1_epi32(26386),
        _mm512_set1_epi32(16004),
        _mm512_set1_epi32(3571),
    ];
    let round64 = _mm512_set1_epi64(1i64 << (bpc - 1));
    let bpc_vec = _mm512_set1_epi64(bpc as i64);
    let mut prod = [_mm512_setzero_si512(); 5];
    for k in 0..5 {
        let d = _mm512_sub_epi32(
            _mm512_cvtepu16_epi32(_mm256_loadu_si256(a8::<u16, 16>(&prev_rows[k][..16]))),
            _mm512_cvtepu16_epi32(_mm256_loadu_si256(a8::<u16, 16>(&cur_rows[k][..16]))),
        );
        prod[k] = _mm512_mullo_epi32(d, g[k]);
    }
    let mut acc_lo = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(prod[0]));
    let mut acc_hi = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(prod[0], 1));
    for p in prod.iter().take(5).skip(1) {
        acc_lo = _mm512_add_epi64(acc_lo, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(*p)));
        acc_hi = _mm512_add_epi64(
            acc_hi,
            _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(*p, 1)),
        );
    }
    acc_lo = _mm512_srav_epi64(_mm512_add_epi64(acc_lo, round64), bpc_vec);
    acc_hi = _mm512_srav_epi64(_mm512_add_epi64(acc_hi, round64), bpc_vec);
    let res_lo = _mm512_cvtsepi64_epi32(acc_lo);
    let res_hi = _mm512_cvtsepi64_epi32(acc_hi);
    _mm512_storeu_si512(
        a8m::<i32, 16>(out),
        _mm512_inserti64x4(_mm512_castsi256_si512(res_lo), res_hi, 1),
    );
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[rite]
fn srai_epi64_16(_token: X64V3Token, v: __m256i) -> __m256i {
    let lo = _mm256_srli_epi64(v, 16);
    let hi = _mm256_srai_epi32(v, 16);
    _mm256_blend_epi32(lo, hi, 0xAA)
}

/// Direct port of `x_conv_row_sad_avx2`: horizontal 5-tap convolution of the
/// i32 y_row with i64-pair accumulation, srai_epi64_16 rounding, permutevar
/// pack, abs_epi32, and a per-lane SAD sum reduced at the end. Edge columns
/// use the scalar mirror path exactly as in C.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
fn motion_xsad_v3(_token: X64V3Token, y_row: &[i32], w: usize) -> u64 {
    let g0 = _mm256_set1_epi32(3571);
    let g1 = _mm256_set1_epi32(16004);
    let g2 = _mm256_set1_epi32(26386);
    let round64 = _mm256_set1_epi64x(1 << 15);
    let perm_idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);

    let mut row_sad = 0u64;
    let mut j = 0usize;
    while j < 2 && j < w {
        let mut accum = 0i64;
        for (k, &coef) in MOTION_FILTER.iter().enumerate() {
            let col = mirror(j as isize - 2 + k as isize, w);
            accum += coef * y_row[col] as i64;
        }
        let val = ((accum + (1 << 15)) >> 16) as i32;
        row_sad += val.unsigned_abs() as u64;
        j += 1;
    }

    let mut sad_acc = _mm256_setzero_si256();
    while j + 10 <= w {
        let y0 = _mm256_loadu_si256(a8::<i32, 8>(&y_row[j - 2..j + 6]));
        let y1 = _mm256_loadu_si256(a8::<i32, 8>(&y_row[j - 1..j + 7]));
        let y2 = _mm256_loadu_si256(a8::<i32, 8>(&y_row[j..j + 8]));
        let y3 = _mm256_loadu_si256(a8::<i32, 8>(&y_row[j + 1..j + 9]));
        let y4 = _mm256_loadu_si256(a8::<i32, 8>(&y_row[j + 2..j + 10]));
        let p0 = _mm256_mullo_epi32(y0, g0);
        let p1 = _mm256_mullo_epi32(y1, g1);
        let p2 = _mm256_mullo_epi32(y2, g2);
        let p3 = _mm256_mullo_epi32(y3, g1);
        let p4 = _mm256_mullo_epi32(y4, g0);
        let s04 = _mm256_add_epi32(p0, p4);
        let s13 = _mm256_add_epi32(p1, p3);
        let mut acc_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(s04));
        acc_lo = _mm256_add_epi64(acc_lo, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(s13)));
        acc_lo = _mm256_add_epi64(acc_lo, _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p2)));
        let mut acc_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(s04, 1));
        acc_hi = _mm256_add_epi64(
            acc_hi,
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256(s13, 1)),
        );
        acc_hi = _mm256_add_epi64(
            acc_hi,
            _mm256_cvtepi32_epi64(_mm256_extracti128_si256(p2, 1)),
        );
        acc_lo = srai_epi64_16(_token, _mm256_add_epi64(acc_lo, round64));
        acc_hi = srai_epi64_16(_token, _mm256_add_epi64(acc_hi, round64));
        let res_lo = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(acc_lo, perm_idx));
        let res_hi = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(acc_hi, perm_idx));
        let result = _mm256_inserti128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1);
        sad_acc = _mm256_add_epi32(sad_acc, _mm256_abs_epi32(result));
        j += 8;
    }

    let lo128 = _mm256_castsi256_si128(sad_acc);
    let hi128 = _mm256_extracti128_si256(sad_acc, 1);
    let mut sum128 = _mm_add_epi32(lo128, hi128);
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b01001110));
    sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b00010001));
    row_sad += (_mm_cvtsi128_si32(sum128) as u32) as u64;

    while j < w {
        let mut accum = 0i64;
        for (k, &coef) in MOTION_FILTER.iter().enumerate() {
            let col = mirror(j as isize - 2 + k as isize, w);
            accum += coef * y_row[col] as i64;
        }
        let val = ((accum + (1 << 15)) >> 16) as i32;
        row_sad += val.unsigned_abs() as u64;
        j += 1;
    }
    row_sad
}

/// Direct port of `x_conv_row_sad_avx512`: the same horizontal 5-tap
/// convolution + abs + SAD as `_v3`, widened to 16 i32 lanes. AVX-512 gives
/// the two ops AVX2 had to emulate: native `srai_epi64` rounding and
/// `cvtsepi64_epi32` narrowing, plus a single-instruction epi32 reduction.
#[cfg(all(feature = "simd", feature = "avx512", target_arch = "x86_64"))]
#[arcane(import_intrinsics)]
fn motion_xsad_v4(_token: X64V4Token, y_row: &[i32], w: usize) -> u64 {
    let g0 = _mm512_set1_epi32(3571);
    let g1 = _mm512_set1_epi32(16004);
    let g2 = _mm512_set1_epi32(26386);
    let round64 = _mm512_set1_epi64(1 << 15);

    let mut row_sad = 0u64;
    let mut j = 0usize;
    while j < 2 && j < w {
        let mut accum = 0i64;
        for (k, &coef) in MOTION_FILTER.iter().enumerate() {
            let col = mirror(j as isize - 2 + k as isize, w);
            accum += coef * y_row[col] as i64;
        }
        let val = ((accum + (1 << 15)) >> 16) as i32;
        row_sad += val.unsigned_abs() as u64;
        j += 1;
    }

    let mut sad_acc = _mm512_setzero_si512();
    // Reads y_row[j-2 ..= j+17], so the vector loop needs j+18 <= w.
    while j + 18 <= w {
        let y0 = _mm512_loadu_si512(a8::<i32, 16>(&y_row[j - 2..j + 14]));
        let y1 = _mm512_loadu_si512(a8::<i32, 16>(&y_row[j - 1..j + 15]));
        let y2 = _mm512_loadu_si512(a8::<i32, 16>(&y_row[j..j + 16]));
        let y3 = _mm512_loadu_si512(a8::<i32, 16>(&y_row[j + 1..j + 17]));
        let y4 = _mm512_loadu_si512(a8::<i32, 16>(&y_row[j + 2..j + 18]));
        let p0 = _mm512_mullo_epi32(y0, g0);
        let p1 = _mm512_mullo_epi32(y1, g1);
        let p2 = _mm512_mullo_epi32(y2, g2);
        let p3 = _mm512_mullo_epi32(y3, g1);
        let p4 = _mm512_mullo_epi32(y4, g0);
        let s04 = _mm512_add_epi32(p0, p4);
        let s13 = _mm512_add_epi32(p1, p3);
        let mut acc_lo = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(s04));
        acc_lo = _mm512_add_epi64(acc_lo, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(s13)));
        acc_lo = _mm512_add_epi64(acc_lo, _mm512_cvtepi32_epi64(_mm512_castsi512_si256(p2)));
        let mut acc_hi = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(s04, 1));
        acc_hi = _mm512_add_epi64(
            acc_hi,
            _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(s13, 1)),
        );
        acc_hi = _mm512_add_epi64(
            acc_hi,
            _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64(p2, 1)),
        );
        acc_lo = _mm512_srai_epi64(_mm512_add_epi64(acc_lo, round64), 16);
        acc_hi = _mm512_srai_epi64(_mm512_add_epi64(acc_hi, round64), 16);
        let res_lo = _mm512_cvtsepi64_epi32(acc_lo);
        let res_hi = _mm512_cvtsepi64_epi32(acc_hi);
        let result = _mm512_inserti64x4(_mm512_castsi256_si512(res_lo), res_hi, 1);
        sad_acc = _mm512_add_epi32(sad_acc, _mm512_abs_epi32(result));
        j += 16;
    }

    row_sad += (_mm512_reduce_add_epi32(sad_acc) as u32) as u64;

    while j < w {
        let mut accum = 0i64;
        for (k, &coef) in MOTION_FILTER.iter().enumerate() {
            let col = mirror(j as isize - 2 + k as isize, w);
            accum += coef * y_row[col] as i64;
        }
        let val = ((accum + (1 << 15)) >> 16) as i32;
        row_sad += val.unsigned_abs() as u64;
        j += 1;
    }
    row_sad
}

#[cfg_attr(feature = "simd", autoversion)]
fn motion_horizontal_row(y_row: &[i32], width: usize) -> u64 {
    let x_round: i64 = 1 << 15;
    let mut row_sad: u64 = 0;
    for j in 0..2.min(width) {
        let mut accum: i64 = 0;
        for (k, &coef) in MOTION_FILTER.iter().enumerate() {
            let col = mirror(j as isize - 2 + k as isize, width);
            accum += coef * y_row[col] as i64;
        }
        let val = ((accum + x_round) >> 16) as i32;
        row_sad += val.unsigned_abs() as u64;
    }
    for j in 2..width.saturating_sub(2) {
        let mut accum: i64 = 0;
        for (k, &coef) in MOTION_FILTER.iter().enumerate() {
            accum += coef * y_row[j - 2 + k] as i64;
        }
        let val = ((accum + x_round) >> 16) as i32;
        row_sad += val.unsigned_abs() as u64;
    }
    for j in width.saturating_sub(2).max(2.min(width))..width {
        let mut accum: i64 = 0;
        for (k, &coef) in MOTION_FILTER.iter().enumerate() {
            let col = mirror(j as isize - 2 + k as isize, width);
            accum += coef * y_row[col] as i64;
        }
        let val = ((accum + x_round) >> 16) as i32;
        row_sad += val.unsigned_abs() as u64;
    }
    row_sad
}

pub(crate) fn motion_sad(prev: &[u16], cur: &[u16], width: usize, height: usize, bpc: u8) -> u64 {
    let y_round: i64 = 1 << (bpc - 1);
    let mut y_row = vec![0i32; width];
    let mut sad: u64 = 0;

    for i in 0..height {
        let mut any_nonzero: i32 = 0;
        let mut j = 0usize;
        #[cfg(feature = "simd")]
        if (2..height.saturating_sub(2)).contains(&i) {
            let prev_rows = |j: usize| {
                [
                    &prev[(i - 2) * width + j..],
                    &prev[(i - 1) * width + j..],
                    &prev[i * width + j..],
                    &prev[(i + 1) * width + j..],
                    &prev[(i + 2) * width + j..],
                ]
            };
            let cur_rows = |j: usize| {
                [
                    &cur[(i - 2) * width + j..],
                    &cur[(i - 1) * width + j..],
                    &cur[i * width + j..],
                    &cur[(i + 1) * width + j..],
                    &cur[(i + 2) * width + j..],
                ]
            };
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            if bpc == 8
                && let Some(token) = v4_token()
            {
                while j + 32 <= width {
                    let mut out = [0i32; 32];
                    motion_vertical8_v4(token, &prev_rows(j), &cur_rows(j), &mut out);
                    y_row[j..j + 32].copy_from_slice(&out);
                    any_nonzero |= out.iter().fold(0i32, |a, &b| a | b);
                    j += 32;
                }
            }
            #[cfg(target_arch = "x86_64")]
            if bpc == 8
                && let Some(token) = v3_token()
            {
                while j + 16 <= width {
                    let mut out = [0i32; 16];
                    motion_vertical8_v3(token, &prev_rows(j), &cur_rows(j), &mut out);
                    y_row[j..j + 16].copy_from_slice(&out);
                    any_nonzero |= out.iter().fold(0i32, |a, &b| a | b);
                    j += 16;
                }
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            if bpc == 16
                && let Some(token) = v4_token()
            {
                while j + 16 <= width {
                    let mut out = [0i32; 16];
                    motion_vertical16_v4(token, &prev_rows(j), &cur_rows(j), 16, &mut out);
                    y_row[j..j + 16].copy_from_slice(&out);
                    any_nonzero |= out.iter().fold(0i32, |a, &b| a | b);
                    j += 16;
                }
            }
            #[cfg(target_arch = "x86_64")]
            if bpc == 16
                && let Some(token) = v3_token()
            {
                while j + 8 <= width {
                    let mut out = [0i32; 8];
                    motion_vertical16_v3(token, &prev_rows(j), &cur_rows(j), 16, &mut out);
                    y_row[j..j + 8].copy_from_slice(&out);
                    any_nonzero |= out.iter().fold(0i32, |a, &b| a | b);
                    j += 8;
                }
            }
            if bpc < 16 && j == 0 {
                while j + 16 <= width {
                    let mut out = [0i32; 16];
                    archmage::incant!(
                        motion_vertical_simd(
                            &prev_rows(j),
                            &cur_rows(j),
                            &MOTION_FILTER,
                            y_round as i32,
                            bpc as u32,
                            &mut out
                        ),
                        [v3, neon, wasm128, scalar]
                    );
                    y_row[j..j + 16].copy_from_slice(&out);
                    any_nonzero |= out.iter().fold(0i32, |a, &b| a | b);
                    j += 16;
                }
            }
        }
        while j < width {
            let mut accum: i64 = 0;
            for (k, &coef) in MOTION_FILTER.iter().enumerate() {
                let row = mirror(i as isize - 2 + k as isize, height);
                let diff = prev[row * width + j] as i64 - cur[row * width + j] as i64;
                accum += coef * diff;
            }
            y_row[j] = ((accum + y_round) >> bpc) as i32;
            any_nonzero |= y_row[j];
            j += 1;
        }
        if any_nonzero == 0 {
            continue;
        }
        #[cfg(all(feature = "simd", feature = "avx512", target_arch = "x86_64"))]
        if let Some(token) = v4_token() {
            sad += motion_xsad_v4(token, &y_row, width);
        } else if let Some(token) = v3_token() {
            sad += motion_xsad_v3(token, &y_row, width);
        } else {
            sad += motion_horizontal_row(&y_row, width);
        }
        #[cfg(all(feature = "simd", not(feature = "avx512"), target_arch = "x86_64"))]
        if let Some(token) = v3_token() {
            sad += motion_xsad_v3(token, &y_row, width);
        } else {
            sad += motion_horizontal_row(&y_row, width);
        }
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        {
            sad += motion_horizontal_row(&y_row, width);
        }
    }
    sad
}

fn motion_raw_from_luma(
    frames: &[&[u16]],
    width: usize,
    height: usize,
    bit_depth: u8,
    hfr: bool,
    max_val: f64,
) -> Result<Vec<f64>, Error> {
    if !matches!(bit_depth, 8 | 10 | 12 | 16) {
        return Err(Error::InvalidInput("unsupported bit depth"));
    }
    if width == 0 || height == 0 {
        return Err(Error::InvalidInput("zero dimension"));
    }
    if width < 5 || height < 5 {
        return Err(Error::InvalidInput("dimension below filter width"));
    }
    let npix = width
        .checked_mul(height)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or(Error::InvalidInput("dimension overflow"))?;
    if frames.is_empty() {
        return Err(Error::InvalidInput("empty frame list"));
    }
    let max_sample = (1u32 << bit_depth) - 1;
    for frame in frames {
        if frame.len() != npix {
            return Err(Error::InvalidInput("frame length mismatch"));
        }
        if frame.iter().any(|&v| v as u32 > max_sample) {
            return Err(Error::InvalidInput("sample exceeds bit depth"));
        }
    }

    let min_idx: usize = if hfr { 2 } else { 1 };
    let mut raw = vec![0.0f64; frames.len()];
    for i in min_idx..frames.len() {
        let sad = motion_sad(frames[i - min_idx], frames[i], width, height, bit_depth);
        raw[i] = ((sad as f64) / 256.0 / (width * height) as f64).min(max_val);
    }
    Ok(raw)
}

pub fn motion2_v0_from_luma(
    frames: &[&[u16]],
    width: usize,
    height: usize,
    bit_depth: u8,
) -> Result<Vec<f64>, Error> {
    if !matches!(bit_depth, 8 | 10) {
        return Err(Error::InvalidInput("unsupported bit depth"));
    }
    let raw = motion_raw_from_luma(frames, width, height, bit_depth, false, MOTION_V0_MAX_VAL)?;
    let mut out = vec![0.0; frames.len()];
    for i in 1..frames.len() {
        out[i] = if i + 1 == frames.len() {
            raw[i]
        } else {
            raw[i].min(raw[i + 1])
        };
    }
    Ok(out)
}

pub fn motion3_from_luma(
    frames: &[&[u16]],
    width: usize,
    height: usize,
    bit_depth: u8,
    hfr: bool,
) -> Result<Vec<f64>, Error> {
    let raw = motion_raw_from_luma(frames, width, height, bit_depth, hfr, MOTION_MAX_VAL)?;
    let n = frames.len();
    let min_idx: usize = if hfr { 2 } else { 1 };
    let stride: usize = if hfr { 2 } else { 1 };

    let stamp = if n > min_idx { raw[min_idx] } else { 0.0 };
    let mut out = vec![0.0f64; n];
    let mut prev_processed = 0.0;
    for i in 0..n {
        if i < min_idx {
            out[i] = stamp;
            prev_processed = stamp;
            continue;
        }
        let hi = i + 1;
        let motion2 = if hi >= n {
            raw[i]
        } else {
            let lo = i as isize - (stride as isize - 1);
            if lo >= min_idx as isize {
                raw[lo as usize].min(raw[hi])
            } else {
                raw[hi]
            }
        };
        let processed = motion2.min(MOTION_MAX_VAL);
        out[i] = if hfr {
            (processed + prev_processed) / 2.0
        } else {
            processed
        };
        prev_processed = processed;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn motion_sad_scalar(prev: &[u16], cur: &[u16], width: usize, height: usize, bpc: u8) -> u64 {
        let y_round: i64 = 1 << (bpc - 1);
        let x_round: i64 = 1 << 15;
        let mut y_row = vec![0i32; width];
        let mut sad: u64 = 0;
        for i in 0..height {
            let mut any_nonzero: i32 = 0;
            for j in 0..width {
                let mut accum: i64 = 0;
                for (k, &coef) in MOTION_FILTER.iter().enumerate() {
                    let row = mirror(i as isize - 2 + k as isize, height);
                    let diff = prev[row * width + j] as i64 - cur[row * width + j] as i64;
                    accum += coef * diff;
                }
                y_row[j] = ((accum + y_round) >> bpc) as i32;
                any_nonzero |= y_row[j];
            }
            if any_nonzero == 0 {
                continue;
            }
            let mut row_sad: u64 = 0;
            for j in 0..width {
                let mut accum: i64 = 0;
                for (k, &coef) in MOTION_FILTER.iter().enumerate() {
                    let col = mirror(j as isize - 2 + k as isize, width);
                    accum += coef * y_row[col] as i64;
                }
                let val = ((accum + x_round) >> 16) as i32;
                row_sad += val.unsigned_abs() as u64;
            }
            sad += row_sad;
        }
        sad
    }

    #[test]
    fn motion_sad_matches_scalar_across_widths_and_depths() {
        for (width, height) in [
            (7usize, 3usize),
            (16, 5),
            (17, 9),
            (31, 12),
            (33, 33),
            (47, 20),
            (64, 8),
        ] {
            for bpc in [8u8, 10, 12, 16] {
                let max = (1u32 << bpc) - 1;
                let n = width * height;
                let prev: Vec<u16> = (0..n)
                    .map(|i| ((i * 37 + width) % (max as usize + 1)) as u16)
                    .collect();
                let cur: Vec<u16> = (0..n)
                    .map(|i| ((i * 53 + height + i / width) % (max as usize + 1)) as u16)
                    .collect();
                assert_eq!(
                    motion_sad(&prev, &cur, width, height, bpc),
                    motion_sad_scalar(&prev, &cur, width, height, bpc),
                    "width={width} height={height} bpc={bpc}"
                );
            }
        }
    }
}
