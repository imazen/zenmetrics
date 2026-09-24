use crate::adm::ADM_MIN_DIM;
use crate::{
    Error, PoolingMethod, VmafV0Features, VmafV0Model, VmafV0Variant, Yuv420Frame,
    adm2_v0_from_luma, motion_sad, motion2_v0_from_luma, pool_v1_scores, vif_v0_from_luma,
};

#[derive(Clone, Copy, Debug)]
pub struct VmafV0FrameResult {
    pub features: VmafV0Features,
    pub score: f64,
}

fn validate_dimensions(width: usize, height: usize, bit_depth: u8) -> Result<usize, Error> {
    if !matches!(bit_depth, 8 | 10) {
        return Err(Error::InvalidInput("unsupported bit depth"));
    }
    if width < ADM_MIN_DIM
        || height < ADM_MIN_DIM
        || !width.is_multiple_of(2)
        || !height.is_multiple_of(2)
    {
        return Err(Error::InvalidInput("invalid YUV420 dimensions"));
    }
    width
        .checked_mul(height)
        .filter(|&value| value <= isize::MAX as usize)
        .ok_or(Error::InvalidInput("dimension overflow"))
}

pub struct VmafV0Scorer {
    width: usize,
    height: usize,
    bit_depth: u8,
    variant: VmafV0Variant,
    model: VmafV0Model,
    #[cfg(feature = "parallel")]
    pool: Option<rayon::ThreadPool>,
}

impl VmafV0Scorer {
    pub fn new(
        width: usize,
        height: usize,
        bit_depth: u8,
        variant: VmafV0Variant,
    ) -> Result<Self, Error> {
        validate_dimensions(width, height, bit_depth)?;
        Ok(Self {
            width,
            height,
            bit_depth,
            variant,
            model: VmafV0Model::new(variant)?,
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

    fn validate_frame(&self, frame: &Yuv420Frame<'_>) -> Result<(), Error> {
        let pixels = self.width * self.height;
        if frame.y.len() != pixels || frame.u.len() != pixels / 4 || frame.v.len() != pixels / 4 {
            return Err(Error::InvalidInput("plane length mismatch"));
        }
        let max = (1u16 << self.bit_depth) - 1;
        if frame
            .y
            .iter()
            .chain(frame.u.iter())
            .chain(frame.v.iter())
            .any(|&sample| sample > max)
        {
            return Err(Error::InvalidInput("sample exceeds bit depth"));
        }
        Ok(())
    }

    fn extract_features(
        &self,
        reference: &Yuv420Frame<'_>,
        distorted: &Yuv420Frame<'_>,
    ) -> Result<VmafV0Features, Error> {
        self.validate_frame(reference)?;
        self.validate_frame(distorted)?;
        Ok(VmafV0Features {
            adm2: adm2_v0_from_luma(
                reference.y,
                distorted.y,
                self.width,
                self.height,
                self.bit_depth,
                self.variant,
            )?,
            motion2: 0.0,
            vif_scales: vif_v0_from_luma(
                reference.y,
                distorted.y,
                self.width,
                self.height,
                self.bit_depth,
                self.variant,
            )?,
        })
    }

    fn predict_features(&self, features: VmafV0Features) -> Result<VmafV0FrameResult, Error> {
        Ok(VmafV0FrameResult {
            score: self.model.predict(features)?,
            features,
        })
    }

    fn score_frame(
        &self,
        reference: &Yuv420Frame<'_>,
        distorted: &Yuv420Frame<'_>,
        motion2: f64,
    ) -> Result<VmafV0FrameResult, Error> {
        let mut features = self.extract_features(reference, distorted)?;
        features.motion2 = motion2;
        self.predict_features(features)
    }

    pub fn score(
        &self,
        reference: &[Yuv420Frame<'_>],
        distorted: &[Yuv420Frame<'_>],
    ) -> Result<Vec<VmafV0FrameResult>, Error> {
        if reference.is_empty() || reference.len() != distorted.len() {
            return Err(Error::InvalidInput(
                "reference/distorted frame count mismatch or empty",
            ));
        }
        for frame in reference.iter().chain(distorted.iter()) {
            self.validate_frame(frame)?;
        }
        let motion = motion2_v0_from_luma(
            &reference.iter().map(|frame| frame.y).collect::<Vec<_>>(),
            self.width,
            self.height,
            self.bit_depth,
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
                    .map(|((reference, distorted), &motion2)| {
                        self.score_frame(reference, distorted, motion2)
                    })
                    .collect()
            });
        }
        reference
            .iter()
            .zip(distorted)
            .zip(motion)
            .map(|((reference, distorted), motion2)| {
                self.score_frame(reference, distorted, motion2)
            })
            .collect()
    }
}

pub struct VmafV0Stream {
    scorer: VmafV0Scorer,
    previous_reference: Option<Vec<u16>>,
    pending: Option<VmafV0Features>,
    previous_raw: f64,
    seen: usize,
}

impl VmafV0Stream {
    pub fn new(
        width: usize,
        height: usize,
        bit_depth: u8,
        variant: VmafV0Variant,
    ) -> Result<Self, Error> {
        Ok(Self {
            scorer: VmafV0Scorer::new(width, height, bit_depth, variant)?,
            previous_reference: None,
            pending: None,
            previous_raw: 0.0,
            seen: 0,
        })
    }

    pub fn push(
        &mut self,
        reference: Yuv420Frame<'_>,
        distorted: Yuv420Frame<'_>,
    ) -> Result<Vec<VmafV0FrameResult>, Error> {
        let next = self
            .seen
            .checked_add(1)
            .ok_or(Error::InvalidInput("frame count overflow"))?;
        let features = self.scorer.extract_features(&reference, &distorted)?;
        let raw = if let Some(previous) = &self.previous_reference {
            let sad = motion_sad(
                previous,
                reference.y,
                self.scorer.width,
                self.scorer.height,
                self.scorer.bit_depth,
            );
            ((sad as f64) / 256.0 / (self.scorer.width * self.scorer.height) as f64).min(10000.0)
        } else {
            0.0
        };
        let emitted = self
            .pending
            .map(|mut pending| {
                pending.motion2 = if self.seen == 1 {
                    0.0
                } else {
                    self.previous_raw.min(raw)
                };
                self.scorer.predict_features(pending)
            })
            .transpose()?;
        self.previous_reference = Some(reference.y.to_vec());
        self.pending = Some(features);
        self.previous_raw = raw;
        self.seen = next;
        Ok(emitted.into_iter().collect())
    }

    pub fn finish(self) -> Result<Vec<VmafV0FrameResult>, Error> {
        let mut features = self
            .pending
            .ok_or(Error::InvalidInput("empty frame list"))?;
        features.motion2 = if self.seen > 1 {
            self.previous_raw
        } else {
            0.0
        };
        Ok(vec![self.scorer.predict_features(features)?])
    }
}

pub fn score_v0_420(
    reference: &[Yuv420Frame<'_>],
    distorted: &[Yuv420Frame<'_>],
    width: usize,
    height: usize,
    bit_depth: u8,
    variant: VmafV0Variant,
) -> Result<Vec<VmafV0FrameResult>, Error> {
    VmafV0Scorer::new(width, height, bit_depth, variant)?.score(reference, distorted)
}

pub fn pool_v0_scores(scores: &[f64], method: PoolingMethod) -> Result<f64, Error> {
    pool_v1_scores(scores, method)
}
