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

/// An extra trunk input, addressed by position in the graph's input list
/// (index 0 is the image input `run` feeds). Heads with more than one
/// tensor input take their other taps through these — e.g. ahiq's head
/// consumes `vit_d` as the primary input with `vit_r`, `cnn_d`, `cnn_r`
/// as aux.
#[cfg_attr(not(any(feature = "rten", feature = "ort")), expect(dead_code))]
pub(crate) struct AuxInput {
    /// Position in `model.input_ids()`.
    pub(crate) index: usize,
    /// Input shape, e.g. `[1, 1, n + 1, 768]`.
    pub(crate) shape: [usize; 4],
    /// Row-major tensor data, `shape` elements.
    pub(crate) data: Vec<f32>,
}

/// A loaded trunk: one `[1, 3, h, w]` input, `n` feature-map outputs.
pub(crate) trait Trunk: Send + Sync {
    /// Engine name, recorded in the score's implementation tag.
    fn engine(&self) -> &'static str;
    /// Run on a planar `[3, h, w]` input.
    fn run(&self, input: Vec<f32>, h: usize, w: usize) -> Result<Vec<Tap>, Error>;

    /// Run on a planar `[c, h, w]` input. Trunks whose graph takes a fixed
    /// 3-channel image get the default, which accepts `c == 3` only.
    fn run_c(&self, input: Vec<f32>, c: usize, h: usize, w: usize) -> Result<Vec<Tap>, Error> {
        if c != 3 {
            return Err(Error::Engine(format!(
                "{c}-channel input on a 3-channel trunk"
            )));
        }
        self.run(input, h, w)
    }

    /// Run on a planar `[3, h, w]` input plus auxiliary inputs. Needed when
    /// part of the host-side preprocessing feeds the middle of the graph —
    /// e.g. afine's positional embedding is bicubic-resampled on the host
    /// because rten's `Resize` has no cubic mode.
    fn run_aux(
        &self,
        input: Vec<f32>,
        h: usize,
        w: usize,
        aux: &[AuxInput],
    ) -> Result<Vec<Tap>, Error> {
        if aux.is_empty() {
            return self.run(input, h, w);
        }
        Err(Error::Engine(
            "auxiliary inputs unsupported on this trunk".into(),
        ))
    }

    /// [`Trunk::run_aux`] on a planar `[c, h, w]` input — for graphs whose
    /// primary input is a feature map, e.g. ahiq's head takes the distorted
    /// ViT tap plus the other three taps as [`AuxInput`]s.
    fn run_c_aux(
        &self,
        input: Vec<f32>,
        c: usize,
        h: usize,
        w: usize,
        aux: &[AuxInput],
    ) -> Result<Vec<Tap>, Error> {
        if c == 3 {
            return self.run_aux(input, h, w, aux);
        }
        if aux.is_empty() {
            return self.run_c(input, c, h, w);
        }
        Err(Error::Engine(
            "channel-flexible aux inputs unsupported on this trunk".into(),
        ))
    }

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

    /// [`Trunk::run_pair`] with auxiliary inputs — the same aux set feeds
    /// both runs (aux contents may differ per image).
    fn run_pair_aux(
        &self,
        a: Vec<f32>,
        b: Vec<f32>,
        h: usize,
        w: usize,
        aux_a: &[AuxInput],
        aux_b: &[AuxInput],
    ) -> Result<(Vec<Tap>, Vec<Tap>), Error> {
        Ok((self.run_aux(a, h, w, aux_a)?, self.run_aux(b, h, w, aux_b)?))
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
        inputs: Vec<NodeId>,
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
            let inputs = model.input_ids().to_vec();
            if inputs.is_empty() {
                return Err(Error::BadModel("graph has no input".into()));
            }
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
                inputs,
                outputs,
                halves,
            })
        }

        fn run_with(
            &self,
            input: Vec<f32>,
            c: usize,
            h: usize,
            w: usize,
            aux: &[super::AuxInput],
            pool: Option<Arc<ThreadPool>>,
        ) -> Result<Vec<Tap>, Error> {
            let x = NdTensor::from_data([1, c, h, w], input);
            let mut feeds = vec![(self.inputs[0], x.into())];
            for a in aux {
                let id = *self
                    .inputs
                    .get(a.index)
                    .ok_or_else(|| Error::BadModel(format!("graph has no input {}", a.index)))?;
                feeds.push((id, NdTensor::from_data(a.shape, a.data.clone()).into()));
            }
            let opts = pool.map(|p| {
                let mut o = RunOptions::default();
                o.thread_pool = Some(p);
                o
            });
            let values = self
                .model
                .run(feeds, &self.outputs, opts)
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
                    // Move the buffer out (rten's outputs are contiguous);
                    // a non-contiguous output would fall back to a copy.
                    Ok(Tap {
                        c,
                        h: th,
                        w: tw,
                        data: t.into_data(),
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
            self.run_with(input, 3, h, w, &[], None)
        }

        fn run_c(&self, input: Vec<f32>, c: usize, h: usize, w: usize) -> Result<Vec<Tap>, Error> {
            self.run_with(input, c, h, w, &[], None)
        }

        fn run_aux(
            &self,
            input: Vec<f32>,
            h: usize,
            w: usize,
            aux: &[super::AuxInput],
        ) -> Result<Vec<Tap>, Error> {
            self.run_with(input, 3, h, w, aux, None)
        }

        fn run_c_aux(
            &self,
            input: Vec<f32>,
            c: usize,
            h: usize,
            w: usize,
            aux: &[super::AuxInput],
        ) -> Result<Vec<Tap>, Error> {
            self.run_with(input, c, h, w, aux, None)
        }

        fn run_pair(
            &self,
            a: Vec<f32>,
            b: Vec<f32>,
            h: usize,
            w: usize,
        ) -> Result<(Vec<Tap>, Vec<Tap>), Error> {
            self.run_pair_aux(a, b, h, w, &[], &[])
        }

        fn run_pair_aux(
            &self,
            a: Vec<f32>,
            b: Vec<f32>,
            h: usize,
            w: usize,
            aux_a: &[super::AuxInput],
            aux_b: &[super::AuxInput],
        ) -> Result<(Vec<Tap>, Vec<Tap>), Error> {
            let Some([pa, pb]) = &self.halves else {
                return Ok((self.run_aux(a, h, w, aux_a)?, self.run_aux(b, h, w, aux_b)?));
            };
            std::thread::scope(|s| {
                let ha = s.spawn(|| self.run_with(a, 3, h, w, aux_a, Some(Arc::clone(pa))));
                let tb = self.run_with(b, 3, h, w, aux_b, Some(Arc::clone(pb)));
                let ta = ha
                    .join()
                    .map_err(|_| Error::Engine("paired run panicked".into()))?;
                Ok((ta?, tb?))
            })
        }
    }
}

/// A whole-metric graph: reference and distorted in, one score out. For
/// engines where shipping feature maps back to the host would cost more than
/// the network (a GPU behind PCIe).
pub(crate) trait PairGraph: Send + Sync {
    /// Engine name, recorded in the score's implementation tag.
    fn engine(&self) -> &'static str;
    /// Score two planar `[3, h, w]` inputs.
    fn score(&self, r: Vec<f32>, d: Vec<f32>, h: usize, w: usize) -> Result<f64, Error>;
}

#[cfg(feature = "ort")]
pub(crate) mod ort_engine {
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};

    use ort::environment::Environment;
    use ort::ep;
    use ort::session::Session;
    use ort::value::Tensor;

    use super::PairGraph;
    use crate::Error;

    /// ONNX Runtime is loaded once per process, from `ORT_DYLIB_PATH` when it
    /// is set, else from the platform's default library search.
    fn environment() -> Result<&'static Environment, Error> {
        static ENV: OnceLock<Result<Environment, String>> = OnceLock::new();
        ENV.get_or_init(|| {
            let builder = match std::env::var_os("ORT_DYLIB_PATH") {
                Some(p) => ort::init_from(p).map_err(|e| format!("loading ONNX Runtime: {e}"))?,
                None => ort::init(),
            };
            builder
                .with_name("zenmetrics-nn")
                .build()
                .map_err(|e| format!("ONNX Runtime environment: {e}"))
        })
        .as_ref()
        .map_err(|e| Error::Engine(e.clone()))
    }

    /// The CUDA execution provider with TF32 off (PyTorch's
    /// `allow_tf32 = False`), exhaustive cuDNN algorithm search, deterministic
    /// compute, and no silent fallback to the CPU provider.
    pub(crate) struct OrtCudaPair {
        session: Mutex<Session>,
        output: &'static str,
    }

    impl OrtCudaPair {
        pub(crate) fn load(path: &Path, output: &'static str) -> Result<Self, Error> {
            let env = environment()?;
            let cuda = ep::CUDA::default()
                .with_tf32(false)
                .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Exhaustive)
                .build()
                .error_on_failure();
            // Deterministic compute: without it cuDNN's algorithms return
            // last-bit-different scores for identical calls (measured;
            // benchmarks/nn_lpips_phase0_2026-10-15/ort_determinism.log), at
            // a cost within noise with exhaustive search.
            let session = Session::builder(env)
                .and_then(|b| b.with_deterministic_compute(true).map_err(Into::into))
                .and_then(|b| b.with_execution_providers([cuda]).map_err(Into::into))
                .and_then(|mut b| b.commit_from_file(path))
                .map_err(|e| Error::BadModel(format!("{}: {e}", path.display())))?;
            Ok(Self {
                session: Mutex::new(session),
                output,
            })
        }
    }

    impl PairGraph for OrtCudaPair {
        fn engine(&self) -> &'static str {
            "ort-cuda"
        }

        fn score(&self, r: Vec<f32>, d: Vec<f32>, h: usize, w: usize) -> Result<f64, Error> {
            let to_value = |v: Vec<f32>| {
                Tensor::from_array(([1usize, 3, h, w], v)).map_err(|e| Error::Engine(e.to_string()))
            };
            let (rv, dv) = (to_value(r)?, to_value(d)?);
            let mut session = self
                .session
                .lock()
                .map_err(|_| Error::Engine("session lock poisoned".into()))?;
            let out = session
                .run(ort::inputs!["r" => rv, "d" => dv])
                .map_err(|e| Error::Engine(e.to_string()))?;
            let s = out[self.output]
                .try_extract_scalar::<f32>()
                .map_err(|e| Error::Engine(e.to_string()))?;
            Ok(f64::from(s))
        }
    }

    /// A single-image trunk on CUDA: `[1, 3, h, w]` in, the graph's named
    /// feature maps out — for metrics whose head is host-side Rust (DMM's
    /// SVD), where a pair graph cannot express the score.
    // live only under `ort` (DMM is its only consumer).
    #[cfg_attr(not(feature = "ort"), allow(dead_code))]
    pub(crate) struct OrtCudaTrunk {
        session: Mutex<Session>,
        input: &'static str,
        outputs: Vec<&'static str>,
    }

    impl OrtCudaTrunk {
        #[cfg_attr(not(feature = "ort"), allow(dead_code))]
        pub(crate) fn load(
            path: &Path,
            input: &'static str,
            outputs: Vec<&'static str>,
        ) -> Result<Self, Error> {
            let env = environment()?;
            let cuda = ep::CUDA::default()
                .with_tf32(false)
                .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Exhaustive)
                .build()
                .error_on_failure();
            let session = Session::builder(env)
                .and_then(|b| b.with_deterministic_compute(true).map_err(Into::into))
                .and_then(|b| b.with_execution_providers([cuda]).map_err(Into::into))
                .and_then(|mut b| b.commit_from_file(path))
                .map_err(|e| Error::BadModel(format!("{}: {e}", path.display())))?;
            Ok(Self {
                session: Mutex::new(session),
                input,
                outputs,
            })
        }
    }

    impl super::Trunk for OrtCudaTrunk {
        fn engine(&self) -> &'static str {
            "ort-cuda"
        }

        fn run(&self, input: Vec<f32>, h: usize, w: usize) -> Result<Vec<super::Tap>, Error> {
            self.run_c(input, 3, h, w)
        }

        fn run_c(
            &self,
            input: Vec<f32>,
            c: usize,
            h: usize,
            w: usize,
        ) -> Result<Vec<super::Tap>, Error> {
            let x = Tensor::from_array(([1usize, c, h, w], input))
                .map_err(|e| Error::Engine(e.to_string()))?;
            let mut session = self
                .session
                .lock()
                .map_err(|_| Error::Engine("session lock poisoned".into()))?;
            let out = session
                .run(ort::inputs![self.input => x])
                .map_err(|e| Error::Engine(e.to_string()))?;
            self.outputs
                .iter()
                .map(|name| {
                    let (shape, data) = out[*name]
                        .try_extract_tensor::<f32>()
                        .map_err(|e| Error::Engine(e.to_string()))?;
                    let &[n, tc, th, tw] = &shape[..] else {
                        return Err(Error::BadModel(format!(
                            "tap {name} shape {shape:?}, expected [1, c, h, w]"
                        )));
                    };
                    if n != 1 {
                        return Err(Error::Engine(format!("batch {n} output")));
                    }
                    Ok(super::Tap {
                        c: tc as usize,
                        h: th as usize,
                        w: tw as usize,
                        data: data.to_vec(),
                    })
                })
                .collect()
        }
    }
}
