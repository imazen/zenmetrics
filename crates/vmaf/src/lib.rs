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

pub(crate) fn motion_sad(prev: &[u16], cur: &[u16], width: usize, height: usize, bpc: u8) -> u64 {
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
