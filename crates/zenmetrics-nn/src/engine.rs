//! Engines that run an exported trunk graph.

use std::path::Path;

use crate::Error;

/// One `[c, h, w]` feature map (batch 1), row-major planes.
#[derive(Clone, Debug)]
pub(crate) struct Tap {
    pub(crate) c: usize,
    pub(crate) h: usize,
    pub(crate) w: usize,
    pub(crate) data: Vec<f32>,
}

impl Tap {
    pub(crate) fn plane(&self) -> usize {
        self.h * self.w
    }
}

/// A loaded trunk: one `[1, 3, h, w]` input, `n` feature-map outputs.
pub(crate) trait Trunk: Send + Sync {
    /// Engine name, recorded in the score's implementation tag.
    fn engine(&self) -> &'static str;
    /// Run on a planar `[3, h, w]` input.
    fn run(&self, input: Vec<f32>, h: usize, w: usize) -> Result<Vec<Tap>, Error>;

    /// Run two same-size inputs. Engines that can split their threads run
    /// the two concurrently.
    fn run_pair(
        &self,
        a: Vec<f32>,
        b: Vec<f32>,
        h: usize,
        w: usize,
    ) -> Result<(Vec<Tap>, Vec<Tap>), Error> {
        Ok((self.run(a, h, w)?, self.run(b, h, w)?))
    }
}

/// Load `path` with the default engine.
pub(crate) fn load(path: &Path, outputs: usize) -> Result<Box<dyn Trunk>, Error> {
    #[cfg(feature = "rten")]
    {
        Ok(Box::new(rten_engine::RtenTrunk::load(path, outputs)?))
    }
    #[cfg(not(feature = "rten"))]
    {
        let _ = (path, outputs);
        Err(Error::Engine("built without an engine feature".into()))
    }
}

#[cfg(feature = "rten")]
mod rten_engine {
    use std::path::Path;
    use std::sync::Arc;

    use rten::{Model, ModelOptions, NodeId, RunOptions, ThreadPool};
    use rten_tensor::NdTensor;
    use rten_tensor::prelude::*;

    use super::{Tap, Trunk};
    use crate::Error;

    pub(crate) struct RtenTrunk {
        model: Model,
        input: NodeId,
        outputs: Vec<NodeId>,
        /// Two pools splitting the threads, for [`Trunk::run_pair`]. One run
        /// of a conv trunk stops scaling well past ~8 threads (measured on
        /// i265), so two concurrent half-width runs beat two full-width ones
        /// in a row.
        halves: Option<[Arc<ThreadPool>; 2]>,
    }

    /// rten's own thread count: `RTEN_NUM_THREADS`, else available parallelism.
    fn thread_count() -> usize {
        std::env::var("RTEN_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .or_else(|| std::thread::available_parallelism().ok().map(usize::from))
            .unwrap_or(1)
            .max(1)
    }

    impl RtenTrunk {
        pub(crate) fn load(path: &Path, n_outputs: usize) -> Result<Self, Error> {
            // Weight prepacking was measured (i265, 64²–1024²) and did not
            // change run time, so it stays off to save the memory.
            let model = ModelOptions::with_all_ops()
                .load_file(path)
                .map_err(|e| Error::BadModel(format!("{}: {e}", path.display())))?;
            let input = *model
                .input_ids()
                .first()
                .ok_or_else(|| Error::BadModel("graph has no input".into()))?;
            let outputs = model.output_ids().to_vec();
            if outputs.len() != n_outputs {
                return Err(Error::BadModel(format!(
                    "expected {n_outputs} outputs, graph has {}",
                    outputs.len()
                )));
            }
            let n = thread_count();
            let halves = (n >= 2).then(|| {
                [
                    Arc::new(ThreadPool::with_num_threads(n / 2)),
                    Arc::new(ThreadPool::with_num_threads(n - n / 2)),
                ]
            });
            Ok(Self {
                model,
                input,
                outputs,
                halves,
            })
        }

        fn run_with(
            &self,
            input: Vec<f32>,
            h: usize,
            w: usize,
            pool: Option<Arc<ThreadPool>>,
        ) -> Result<Vec<Tap>, Error> {
            let x = NdTensor::from_data([1, 3, h, w], input);
            let opts = pool.map(|p| {
                let mut o = RunOptions::default();
                o.thread_pool = Some(p);
                o
            });
            let values = self
                .model
                .run(vec![(self.input, x.into())], &self.outputs, opts)
                .map_err(|e| Error::Engine(e.to_string()))?;
            values
                .into_iter()
                .map(|v| {
                    let t: NdTensor<f32, 4> = v
                        .try_into()
                        .map_err(|e| Error::Engine(format!("output: {e:?}")))?;
                    let [n, c, th, tw] = t.shape();
                    if n != 1 {
                        return Err(Error::Engine(format!("batch {n} output")));
                    }
                    Ok(Tap {
                        c,
                        h: th,
                        w: tw,
                        data: t.to_vec(),
                    })
                })
                .collect()
        }
    }

    impl Trunk for RtenTrunk {
        fn engine(&self) -> &'static str {
            "rten"
        }

        fn run(&self, input: Vec<f32>, h: usize, w: usize) -> Result<Vec<Tap>, Error> {
            self.run_with(input, h, w, None)
        }

        fn run_pair(
            &self,
            a: Vec<f32>,
            b: Vec<f32>,
            h: usize,
            w: usize,
        ) -> Result<(Vec<Tap>, Vec<Tap>), Error> {
            let Some([pa, pb]) = &self.halves else {
                return Ok((self.run(a, h, w)?, self.run(b, h, w)?));
            };
            std::thread::scope(|s| {
                let ha = s.spawn(|| self.run_with(a, h, w, Some(Arc::clone(pa))));
                let tb = self.run_with(b, h, w, Some(Arc::clone(pb)));
                let ta = ha
                    .join()
                    .map_err(|_| Error::Engine("paired run panicked".into()))?;
                Ok((ta?, tb?))
            })
        }
    }
}
