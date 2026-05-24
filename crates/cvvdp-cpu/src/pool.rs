//! Re-export the pool helpers from cvvdp-gpu so all impls share one
//! source of truth for `lp_norm_mean` / `lp_norm_sum` / `met2jod` /
//! `do_pooling_and_jod_still_3ch`.

pub(crate) use cvvdp_gpu::kernels::pool::{
    do_pooling_and_jod_still_3ch, lp_norm_mean, BASEBAND_W, BETA_BAND, BETA_CH, BETA_SPATIAL,
    IMAGE_INT, PER_CH_W,
};
