use crate::{
    Error, MOTION_MAX_VAL, ModelVariant, VmafFeatures, VmafModel, adm3_v1_from_luma,
    cambi_v1_from_luma, motion_sad, motion3_from_luma, speed_v1_chroma_420,
};
use std::collections::VecDeque;

#[derive(Clone, Copy)]
pub struct Yuv420Frame<'a> {
    pub y: &'a [u16],
    pub u: &'a [u16],
    pub v: &'a [u16],
}

#[derive(Clone, Copy, Debug)]
pub struct VmafFrameResult {
    pub features: VmafFeatures,
    pub score: f64,
}

#[derive(Clone, Copy, Debug)]
pub enum PoolingMethod {
    Mean,
    Min,
    Max,
    HarmonicMean,
}

pub struct VmafV1Scorer {
    width: usize,
    height: usize,
    bit_depth: u8,
    variant: ModelVariant,
    model: VmafModel,
    #[cfg(feature = "parallel")]
    pool: Option<rayon::ThreadPool>,
}

impl VmafV1Scorer {
    pub fn new(
        width: usize,
        height: usize,
        bit_depth: u8,
        variant: ModelVariant,
    ) -> Result<Self, Error> {
        Ok(Self {
            width,
            height,
            bit_depth,
            variant,
            model: VmafModel::new(variant)?,
            #[cfg(feature = "parallel")]
            pool: None,
        })
    }

    #[cfg(feature = "parallel")]
    pub fn with_threads(mut self, threads: usize) -> Result<Self, Error> {
        if threads == 0 {
            return Err(Error::InvalidInput("thread count must be positive"));
        }
        if threads == 1 {
            self.pool = None;
            return Ok(self);
        }
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        if threads > available {
            return Err(Error::InvalidInput(
                "thread count exceeds available parallelism",
            ));
        }
        self.pool = Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|_| Error::InvalidInput("failed to create thread pool"))?,
        );
        Ok(self)
    }

    fn score_frame(
        &self,
        reference: &Yuv420Frame<'_>,
        distorted: &Yuv420Frame<'_>,
        motion3: f64,
    ) -> Result<VmafFrameResult, Error> {
        let mut features = self.extract_features(reference, distorted)?;
        features.motion3 = motion3;
        self.predict_features(features)
    }

    fn extract_features(
        &self,
        reference: &Yuv420Frame<'_>,
        distorted: &Yuv420Frame<'_>,
    ) -> Result<VmafFeatures, Error> {
        Ok(VmafFeatures {
            cambi: cambi_v1_from_luma(distorted.y, self.width, self.height, self.bit_depth)?,
            speed_chroma_uv: speed_v1_chroma_420(
                reference.u,
                reference.v,
                distorted.u,
                distorted.v,
                self.width,
                self.height,
                self.bit_depth,
                self.variant,
            )?,
            adm3: adm3_v1_from_luma(
                reference.y,
                distorted.y,
                self.width,
                self.height,
                self.bit_depth,
                self.variant,
            )?,
            motion3: 0.0,
        })
    }

    fn predict_features(&self, features: VmafFeatures) -> Result<VmafFrameResult, Error> {
        Ok(VmafFrameResult {
            score: self.model.predict(features)?,
            features,
        })
    }

    pub fn score(
        &self,
        reference: &[Yuv420Frame<'_>],
        distorted: &[Yuv420Frame<'_>],
    ) -> Result<Vec<VmafFrameResult>, Error> {
        if reference.is_empty() || reference.len() != distorted.len() {
            return Err(Error::InvalidInput(
                "reference/distorted frame count mismatch or empty",
            ));
        }
        let hfr = matches!(
            self.variant,
            ModelVariant::HfrStandard1080p
                | ModelVariant::HfrPhone
                | ModelVariant::HfrDefault4k
                | ModelVariant::HfrConsumer4k
        );
        let motion = motion3_from_luma(
            &reference.iter().map(|frame| frame.y).collect::<Vec<_>>(),
            self.width,
            self.height,
            self.bit_depth,
            hfr,
        )?;
        #[cfg(feature = "parallel")]
        if let Some(pool) = &self.pool
            && reference.len() > 1
        {
            use rayon::prelude::*;
            return pool.install(|| {
                reference
                    .par_iter()
                    .zip(distorted.par_iter())
                    .zip(motion.par_iter())
                    .map(|((reference, distorted), &motion3)| {
                        self.score_frame(reference, distorted, motion3)
                    })
                    .collect()
            });
        }
        reference
            .iter()
            .zip(distorted)
            .zip(motion)
            .map(|((reference, distorted), motion3)| {
                self.score_frame(reference, distorted, motion3)
            })
            .collect()
    }
}

pub struct VmafV1Stream {
    scorer: VmafV1Scorer,
    references: VecDeque<Vec<u16>>,
    pending: VecDeque<VmafFeatures>,
    raw: [f64; 3],
    seen: usize,
    previous_processed: f64,
}

impl VmafV1Stream {
    pub fn new(
        width: usize,
        height: usize,
        bit_depth: u8,
        variant: ModelVariant,
    ) -> Result<Self, Error> {
        if !matches!(bit_depth, 8 | 10) {
            return Err(Error::InvalidInput("unsupported bit depth"));
        }
        if width < 5 || height < 5 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(Error::InvalidInput("invalid YUV420 dimensions"));
        }
        width
            .checked_mul(height)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or(Error::InvalidInput("dimension overflow"))?;
        Ok(Self {
            scorer: VmafV1Scorer::new(width, height, bit_depth, variant)?,
            references: VecDeque::new(),
            pending: VecDeque::new(),
            raw: [0.0; 3],
            seen: 0,
            previous_processed: 0.0,
        })
    }

    pub fn push(
        &mut self,
        reference: Yuv420Frame<'_>,
        distorted: Yuv420Frame<'_>,
    ) -> Result<Vec<VmafFrameResult>, Error> {
        let index = self.seen;
        let next = index
            .checked_add(1)
            .ok_or(Error::InvalidInput("frame count overflow"))?;
        let features = self.scorer.extract_features(&reference, &distorted)?;
        let hfr = matches!(
            self.scorer.variant,
            ModelVariant::HfrStandard1080p
                | ModelVariant::HfrPhone
                | ModelVariant::HfrDefault4k
                | ModelVariant::HfrConsumer4k
        );
        let distance = if hfr { 2 } else { 1 };
        if index >= distance {
            let previous = self
                .references
                .front()
                .ok_or(Error::InvalidInput("missing reference frame"))?;
            let sad = motion_sad(
                previous,
                reference.y,
                self.scorer.width,
                self.scorer.height,
                self.scorer.bit_depth,
            );
            self.raw[index % 3] =
                ((sad as f64) / 256.0 / (self.scorer.width * self.scorer.height) as f64)
                    .min(MOTION_MAX_VAL);
        }
        self.references.push_back(reference.y.to_vec());
        if self.references.len() > distance {
            self.references.pop_front();
        }
        self.pending.push_back(features);
        self.seen = next;
        let mut emitted = Vec::new();
        if index == distance {
            let stamp = self.raw[index % 3];
            self.previous_processed = stamp;
            for _ in 0..distance {
                let mut features = self
                    .pending
                    .pop_front()
                    .ok_or(Error::InvalidInput("missing pending frame"))?;
                features.motion3 = stamp;
                emitted.push(self.scorer.predict_features(features)?);
            }
        } else if index > distance {
            let frame = index - 1;
            let processed = if hfr && frame == distance {
                self.raw[index % 3]
            } else {
                self.raw[(frame - usize::from(hfr)) % 3].min(self.raw[index % 3])
            };
            let motion3 = if hfr {
                (processed + self.previous_processed) / 2.0
            } else {
                processed
            };
            self.previous_processed = processed;
            let mut features = self
                .pending
                .pop_front()
                .ok_or(Error::InvalidInput("missing pending frame"))?;
            features.motion3 = motion3;
            emitted.push(self.scorer.predict_features(features)?);
        }
        Ok(emitted)
    }

    pub fn finish(mut self) -> Result<Vec<VmafFrameResult>, Error> {
        if self.seen == 0 {
            return Err(Error::InvalidInput("empty frame list"));
        }
        let hfr = matches!(
            self.scorer.variant,
            ModelVariant::HfrStandard1080p
                | ModelVariant::HfrPhone
                | ModelVariant::HfrDefault4k
                | ModelVariant::HfrConsumer4k
        );
        let distance = if hfr { 2 } else { 1 };
        let processed = if self.seen > distance {
            self.raw[(self.seen - 1) % 3]
        } else {
            0.0
        };
        let motion3 = if hfr && self.seen > distance {
            (processed + self.previous_processed) / 2.0
        } else {
            processed
        };
        let mut remaining = Vec::with_capacity(self.pending.len());
        while let Some(mut features) = self.pending.pop_front() {
            features.motion3 = motion3;
            remaining.push(self.scorer.predict_features(features)?);
        }
        Ok(remaining)
    }
}

pub fn score_v1_420(
    reference: &[Yuv420Frame<'_>],
    distorted: &[Yuv420Frame<'_>],
    width: usize,
    height: usize,
    bit_depth: u8,
    variant: ModelVariant,
) -> Result<Vec<VmafFrameResult>, Error> {
    VmafV1Scorer::new(width, height, bit_depth, variant)?.score(reference, distorted)
}

pub fn pool_v1_scores(scores: &[f64], method: PoolingMethod) -> Result<f64, Error> {
    if scores.is_empty() {
        return Err(Error::InvalidInput("empty frame scores"));
    }
    let (mut sum, mut reciprocal_sum) = (0.0, 0.0);
    let (mut min, mut max) = (scores[0], scores[0]);
    for &score in scores {
        if !score.is_finite() {
            return Err(Error::NonFiniteInput("VMAF score"));
        }
        if score < 0.0 {
            return Err(Error::InvalidInput("negative VMAF score"));
        }
        sum += score;
        reciprocal_sum += 1.0 / (score + 1.0);
        min = min.min(score);
        max = max.max(score);
    }
    Ok(match method {
        PoolingMethod::Mean => sum / scores.len() as f64,
        PoolingMethod::Min => min,
        PoolingMethod::Max => max,
        PoolingMethod::HarmonicMean => scores.len() as f64 / reciprocal_sum - 1.0,
    })
}
