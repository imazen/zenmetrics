use crate::{Error, get_f64_array, parse_svm};

const FEATURE_NAMES: [&str; 6] = [
    "VMAF_integer_feature_adm2_score",
    "VMAF_integer_feature_motion2_score",
    "VMAF_integer_feature_vif_scale0_score",
    "VMAF_integer_feature_vif_scale1_score",
    "VMAF_integer_feature_vif_scale2_score",
    "VMAF_integer_feature_vif_scale3_score",
];

#[derive(Copy, Clone, Debug)]
pub enum VmafV0Variant {
    Standard,
    StandardNeg,
    FourK,
    FourKNeg,
}

impl VmafV0Variant {
    pub fn built_in_name(&self) -> &'static str {
        match self {
            Self::Standard => "vmaf_v0.6.1",
            Self::StandardNeg => "vmaf_v0.6.1neg",
            Self::FourK => "vmaf_4k_v0.6.1",
            Self::FourKNeg => "vmaf_4k_v0.6.1neg",
        }
    }

    pub fn no_enhancement_gain(&self) -> bool {
        matches!(self, Self::StandardNeg | Self::FourKNeg)
    }

    fn json(&self) -> &'static str {
        match self {
            Self::Standard => include_str!("../models/vmaf_v0.6.1.json"),
            Self::StandardNeg => include_str!("../models/vmaf_v0.6.1neg.json"),
            Self::FourK => include_str!("../models/vmaf_4k_v0.6.1.json"),
            Self::FourKNeg => include_str!("../models/vmaf_4k_v0.6.1neg.json"),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VmafV0Features {
    pub adm2: f64,
    pub motion2: f64,
    pub vif_scales: [f64; 4],
}

pub struct VmafV0Model {
    slopes: [f64; 7],
    intercepts: [f64; 7],
    gamma: f64,
    rho: f64,
    support_vectors: Vec<crate::SupportVector<6>>,
    score_clip: [f64; 2],
}

impl VmafV0Model {
    pub fn new(variant: VmafV0Variant) -> Result<Self, Error> {
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
        if feature_names.len() != 6
            || feature_names
                .iter()
                .zip(FEATURE_NAMES.iter())
                .any(|(a, b)| a.as_str() != Some(*b))
        {
            return Err(Error::InvalidModel("unexpected feature_names"));
        }

        let slopes = get_f64_array::<7>(model_dict, "slopes")?;
        let intercepts = get_f64_array::<7>(model_dict, "intercepts")?;

        let svm_text = model_dict
            .get("model")
            .and_then(|v| v.as_str())
            .ok_or(Error::InvalidModel("missing svm model text"))?;
        let (gamma, rho, support_vectors) = parse_svm::<6>(svm_text)?;

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
            score_clip: [lo, hi],
        })
    }

    pub fn predict(&self, features: VmafV0Features) -> Result<f64, Error> {
        let raw = [
            features.adm2,
            features.motion2,
            features.vif_scales[0],
            features.vif_scales[1],
            features.vif_scales[2],
            features.vif_scales[3],
        ];
        for (i, value) in raw.iter().enumerate() {
            if !value.is_finite() {
                return Err(Error::NonFiniteInput(FEATURE_NAMES[i]));
            }
        }

        let mut node = [0.0f64; 6];
        for i in 0..6 {
            node[i] = self.slopes[i + 1] * raw[i] + self.intercepts[i + 1];
        }

        let mut svm = 0.0;
        for sv in &self.support_vectors {
            let mut dist2 = 0.0;
            for (&sv_feature, &node_value) in sv.features.iter().zip(&node) {
                let d = sv_feature - node_value;
                dist2 += d * d;
            }
            svm += sv.alpha * (-self.gamma * dist2).exp();
        }
        svm -= self.rho;

        let mut prediction = (svm - self.intercepts[0]) / self.slopes[0];

        if prediction < self.score_clip[0] {
            prediction = self.score_clip[0];
        }
        if prediction > self.score_clip[1] {
            prediction = self.score_clip[1];
        }

        Ok(prediction)
    }
}
