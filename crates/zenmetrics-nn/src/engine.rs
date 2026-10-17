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
/// (index 0 is the image input `run` feeds).
// Read only by the engine backends; a build with neither engine never reads it.
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
    /// Run on a planar `[3, h, w]` input. Owned: rten's in-place-capable
    /// operators reuse the input buffer as scratch — a borrowed view would
    /// force an internal copy instead.
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

/// Worker count for the Rust-side head passes: `RTEN_NUM_THREADS` when set
/// (the process's configured parallelism, and the knob kernel-level timing
/// checks turn down), else available parallelism.
pub(crate) fn workers() -> usize {
    std::env::var("RTEN_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .or_else(|| std::thread::available_parallelism().ok().map(usize::from))
        .unwrap_or(1)
        .max(1)
}
/// Load `path` as a named-input scalar-output graph with the default engine.
pub(crate) fn load_image(path: &Path) -> Result<Box<dyn ImageGraph>, Error> {
    #[cfg(feature = "rten")]
    {
        Ok(Box::new(rten_engine::RtenImage::load(path)?))
    }
    #[cfg(not(feature = "rten"))]
    {
        let _ = path;
        Err(Error::Engine("built without an engine feature".into()))
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
        /// in a row — and beat two runs sharing one pool (task-stealing
        /// overhead loses at small sizes, measured on i265).
        halves: Option<[Arc<ThreadPool>; 2]>,
    }

    /// rten's own thread count: `RTEN_NUM_THREADS`, else available parallelism.
    fn thread_count() -> usize {
        super::workers()
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

    /// [`ImageGraph`] on rten: inputs feed `model.input_ids()` in declared
    /// order, the single output is read as a scalar.
    pub(crate) struct RtenImage {
        model: Model,
        inputs: Vec<NodeId>,
        outputs: Vec<NodeId>,
    }

    impl RtenImage {
        pub(crate) fn load(path: &Path) -> Result<Self, Error> {
            let model = ModelOptions::with_all_ops()
                .load_file(path)
                .map_err(|e| Error::BadModel(format!("{}: {e}", path.display())))?;
            let inputs = model.input_ids().to_vec();
            if inputs.is_empty() {
                return Err(Error::BadModel("graph has no input".into()));
            }
            let outputs = model.output_ids().to_vec();
            if outputs.len() != 1 {
                return Err(Error::BadModel(format!(
                    "expected one output, graph has {}",
                    outputs.len()
                )));
            }
            Ok(Self {
                model,
                inputs,
                outputs,
            })
        }
    }

    impl super::ImageGraph for RtenImage {
        fn engine(&self) -> &'static str {
            "rten"
        }

        fn score(&self, inputs: &[super::NamedInput]) -> Result<f64, Error> {
            let (_, d) = self.tensor(inputs)?;
            d.first()
                .map(|&s| f64::from(s))
                .ok_or_else(|| Error::Engine("empty score output".into()))
        }

        fn tensor(&self, inputs: &[super::NamedInput]) -> Result<(Vec<usize>, Vec<f32>), Error> {
            if inputs.len() != self.inputs.len() {
                return Err(Error::BadModel(format!(
                    "graph takes {} inputs, got {}",
                    self.inputs.len(),
                    inputs.len()
                )));
            }
            let feeds: Vec<(NodeId, _)> = inputs
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    (
                        self.inputs[i],
                        rten_tensor::Tensor::from_data(&x.shape, x.data.clone()).into(),
                    )
                })
                .collect();
            let values = self
                .model
                .run(feeds, &self.outputs, None)
                .map_err(|e| Error::Engine(e.to_string()))?;
            let v = values.into_iter().next().unwrap();
            let t: rten_tensor::Tensor<f32> = v
                .try_into()
                .map_err(|e| Error::Engine(format!("output: {e:?}")))?;
            let shape = t.shape().to_vec();
            Ok((shape, t.into_data()))
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

/// A named tensor input to an [`ImageGraph`], any rank (e.g. `[n, c, h, w]`
/// for image models, `[1, l, 3075]` for MUSIQ's patch list). Constructed by
/// the model but read only by the engine backends.
#[cfg_attr(not(any(feature = "rten", feature = "ort")), allow(dead_code))]
pub(crate) struct NamedInput {
    /// The graph's input name. Read only by engines that feed by name.
    #[cfg_attr(not(feature = "ort"), allow(dead_code))]
    pub(crate) name: &'static str,
    /// Tensor shape; `data` has `shape` elements, row-major.
    pub(crate) shape: Vec<usize>,
    /// Row-major f32 tensor data.
    pub(crate) data: Vec<f32>,
}

/// A whole-metric graph over one image: named inputs in, one score out. For
/// no-reference models on whole-graph engines — the host still owns the eval
/// transform (crops, multi-scale views) and the graph is upstream's forward.
pub(crate) trait ImageGraph: Send + Sync {
    /// Engine name, recorded in the score's implementation tag.
    fn engine(&self) -> &'static str;
    /// Score the inputs.
    fn score(&self, inputs: &[NamedInput]) -> Result<f64, Error>;

    /// Run the inputs and return the first output tensor's `(shape,
    /// row-major data)`. For graphs whose intermediate features feed
    /// host-side logic (paq2piq's RoIPool) — `score` is `tensor().1[0]`.
    fn tensor(&self, inputs: &[NamedInput]) -> Result<(Vec<usize>, Vec<f32>), Error>;
}

#[cfg(feature = "ort")]
pub(crate) mod ort_engine {
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};

    use ort::environment::Environment;
    use ort::ep;
    use ort::session::Session;
    use ort::value::Tensor;

    use super::{ImageGraph, NamedInput, PairGraph};
    use crate::Error;

    /// GPU execution provider for the shared [`OrtGpuPair`],
    /// [`OrtGpuImage`] and [`OrtGpuTrunk`] session wrappers.
    #[derive(Clone, Copy)]
    pub(crate) enum Ep {
        /// CUDA + CPU providers (`ORT_DYLIB_PATH`, TF32 off).
        Cuda,
        /// WebGPU plugin EP (`ZENMETRICS_NN_WEBGPU_EP_LIB`), discrete
        /// adapter preferred, CPU fallback forbidden.
        #[cfg(feature = "ort-webgpu")]
        Webgpu,
    }

    impl Ep {
        /// The GPU provider an [`Engine`](crate::Engine) selects, if it is
        /// an ONNX Runtime GPU engine.
        pub(crate) fn of(engine: crate::Engine) -> Option<Self> {
            match engine {
                crate::Engine::OrtCuda => Some(Self::Cuda),
                #[cfg(feature = "ort-webgpu")]
                crate::Engine::OrtWebgpu => Some(Self::Webgpu),
                _ => None,
            }
        }

        fn name(self) -> &'static str {
            match self {
                Self::Cuda => "ort-cuda",
                #[cfg(feature = "ort-webgpu")]
                Self::Webgpu => "ort-webgpu",
            }
        }
    }

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

    /// Shared CUDA session config: TF32 off (PyTorch's
    /// `allow_tf32 = False`), exhaustive cuDNN algorithm search,
    /// deterministic compute, bounded arena retention, and no silent
    /// fallback to the CPU provider.
    ///
    /// Two accumulation modes must be bounded for shape-varying inputs
    /// (61-pair AIC sweep OOM'd the afine fused graph at pair 40/61,
    /// S46, 2026-10-16): the device-side memory-pattern block retains a
    /// per-shape intermediate arena, and the CPU arena buckets freed
    /// host buffers by size without returning them. Memory-pattern is
    /// off and the CPU provider's arena is disabled (same fix as
    /// `ae582a3b` for the WD trunk). Deterministic compute stays on:
    /// without it cuDNN's algorithms return last-bit-different scores
    /// for identical calls (measured;
    /// benchmarks/nn_lpips_phase0_2026-10-15/ort_determinism.log), at
    /// a cost within noise with exhaustive search.
    fn cuda_session(env: &Environment, path: &Path) -> Result<Session, Error> {
        let cuda = ep::CUDA::default()
            .with_tf32(false)
            .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Exhaustive)
            .build()
            .error_on_failure();
        let cpu = ep::CPU::default().with_arena_allocator(false).build();
        Session::builder(env)
            .and_then(|b| b.with_deterministic_compute(true).map_err(Into::into))
            .and_then(|b| b.with_memory_pattern(false).map_err(Into::into))
            .and_then(|b| b.with_execution_providers([cuda, cpu]).map_err(Into::into))
            .and_then(|mut b| b.commit_from_file(path))
            .map_err(|e| Error::BadModel(format!("{}: {e}", path.display())))
    }

    /// Session construction for the selected GPU execution provider.
    fn gpu_session(env: &Environment, path: &Path, ep: Ep) -> Result<Session, Error> {
        match ep {
            Ep::Cuda => cuda_session(env, path),
            #[cfg(feature = "ort-webgpu")]
            Ep::Webgpu => webgpu_session(env, path),
        }
    }

    /// Registers the WebGPU plugin EP library once per process. The path
    /// comes from `ZENMETRICS_NN_WEBGPU_EP_LIB` — the
    /// `libonnxruntime_providers_webgpu.so` shipped by the
    /// `onnxruntime-ep-webgpu` pip package / `Microsoft.ML.OnnxRuntime.EP.WebGpu`
    /// NuGet, version-matched to the loaded onnxruntime core. Like
    /// `ORT_DYLIB_PATH`, the variable names trusted code that is dlopen'd
    /// into this process — it is part of the trusted configuration
    /// surface, not a data input.
    #[cfg(feature = "ort-webgpu")]
    fn webgpu_library(env: &Environment) -> Result<(), Error> {
        static REG: OnceLock<Result<(), String>> = OnceLock::new();
        REG.get_or_init(|| {
            let lib = std::env::var("ZENMETRICS_NN_WEBGPU_EP_LIB").map_err(|_| {
                "ZENMETRICS_NN_WEBGPU_EP_LIB unset — point it at \
                 libonnxruntime_providers_webgpu.so (pip package \
                 onnxruntime-ep-webgpu matched to the loaded onnxruntime)"
                    .to_string()
            })?;
            if !Path::new(&lib).is_file() {
                return Err(format!(
                    "ZENMETRICS_NN_WEBGPU_EP_LIB={lib}: not a file — expected \
                     libonnxruntime_providers_webgpu.so"
                ));
            }
            env.register_ep_library("webgpu", lib)
                .map(|_| ())
                .map_err(|e| format!("registering the WebGPU EP library: {e}"))
        })
        .as_ref()
        .map_err(|e| Error::Engine(e.clone()))?;
        Ok(())
    }

    /// A GPU-type device belonging to the WebGPU plugin EP.
    #[cfg(feature = "ort-webgpu")]
    fn is_webgpu_gpu(d: &ort::device::Device) -> bool {
        matches!(d.ep(), Ok("WebGpuExecutionProvider"))
            && d.hardware_device().ty() == ort::memory::DeviceType::GPU
    }

    /// Auto-selection rank for a WebGPU EP GPU device: discrete-vendor
    /// adapters first — NVIDIA, then AMD/ATI — then anything else the EP
    /// offers. The safe ort API exposes no per-adapter discreteness flag,
    /// so the vendor-name rank and `ZENMETRICS_NN_WEBGPU_DEVICE` are the
    /// resolution; an AMD iGPU must not win over an NVIDIA dGPU.
    #[cfg(feature = "ort-webgpu")]
    fn webgpu_rank(d: &ort::device::Device) -> u8 {
        match d.hardware_device().vendor() {
            Ok(v) if v.contains("NVIDIA") => 0,
            Ok(v) if v.contains("AMD") || v.contains("ATI") => 1,
            _ => 2,
        }
    }

    /// Index into the WebGPU EP's GPU-device list of the adapter to run
    /// on. `ZENMETRICS_NN_WEBGPU_DEVICE` pins an index; otherwise the
    /// lowest-[`webgpu_rank`] GPU wins and more than one top-ranked
    /// candidate is an error that names every GPU, since a silent pick
    /// between equals is not reproducible. The index is cached for the
    /// process; the choice is reported through errors (below, and the
    /// session-commit path) rather than printed.
    #[cfg(feature = "ort-webgpu")]
    fn webgpu_device_index(env: &Environment) -> Result<usize, Error> {
        static IDX: OnceLock<Result<usize, String>> = OnceLock::new();
        IDX.get_or_init(|| {
            let gpus: Vec<_> = env.devices().filter(is_webgpu_gpu).collect();
            if gpus.is_empty() {
                return Err("the WebGPU EP exposes no GPU device".into());
            }
            let names: Vec<String> = gpus
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let hw = d.hardware_device();
                    format!(
                        "{i} (vendor {:?}, id {}, type {:?})",
                        hw.vendor().unwrap_or("<unknown>"),
                        hw.id(),
                        hw.ty()
                    )
                })
                .collect();
            let i = if let Ok(v) = std::env::var("ZENMETRICS_NN_WEBGPU_DEVICE") {
                v.parse::<usize>()
                    .map_err(|_| format!("ZENMETRICS_NN_WEBGPU_DEVICE={v}: expected an index"))?
            } else {
                let best = gpus.iter().map(webgpu_rank).min().unwrap_or(u8::MAX);
                let top: Vec<usize> = gpus
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| webgpu_rank(d) == best)
                    .map(|(i, _)| i)
                    .collect();
                if top.len() > 1 {
                    return Err(format!(
                        "ambiguous WebGPU adapter: {} equally-ranked GPUs; set \
                         ZENMETRICS_NN_WEBGPU_DEVICE to one of: {}",
                        top.len(),
                        names.join(", ")
                    ));
                }
                top[0]
            };
            if i >= gpus.len() {
                return Err(format!(
                    "the WebGPU EP exposes {} GPU device(s) ({}), index {i} out of range",
                    gpus.len(),
                    names.join(", ")
                ));
            }
            Ok(i)
        })
        .as_ref()
        .copied()
        .map_err(|e| Error::Engine(e.clone()))
    }

    /// The WebGPU adapter [`webgpu_device_index`] selected.
    #[cfg(feature = "ort-webgpu")]
    fn webgpu_device(env: &Environment) -> Result<ort::device::Device<'_>, Error> {
        env.devices()
            .filter(is_webgpu_gpu)
            .nth(webgpu_device_index(env)?)
            .ok_or_else(|| Error::Engine("the WebGPU EP exposes no GPU device".into()))
    }

    /// WebGPU plugin EP session: same graph and determinism as
    /// [`cuda_session`], but provider selection goes through the device
    /// API (plugin EPs are not named providers) and CPU fallback is
    /// disabled — a graph that leaves a node on the CPU fails to commit.
    /// Memory pattern stays ON here: the CUDA-side arena-retention bug it
    /// is disabled for does not exist in the WebGPU EP; measured
    /// (`benchmarks/ort_webgpu_2026-10-17/`): identical peaks on and off
    /// (5,298 MiB at DISTS 1024², 15,690 at the 2048² OOM attempt), and
    /// alternating shapes in one session are stable.
    #[cfg(feature = "ort-webgpu")]
    fn webgpu_session(env: &Environment, path: &Path) -> Result<Session, Error> {
        webgpu_library(env)?;
        let i = webgpu_device_index(env)?;
        let device = webgpu_device(env)?;
        Session::builder(env)
            .and_then(|b| b.with_deterministic_compute(true).map_err(Into::into))
            .and_then(|b| b.with_disable_cpu_fallback().map_err(Into::into))
            .and_then(|b| b.with_devices([device], None).map_err(Into::into))
            .and_then(|mut b| b.commit_from_file(path))
            .map_err(|e| {
                Error::BadModel(format!(
                    "{} (WebGPU EP GPU device {i}): {e}",
                    path.display()
                ))
            })
    }

    /// Extract the one named output from a session run, consuming each
    /// [`ort::session::SessionOutputs`] value as it is read so every
    /// original is released after its copy. Indexing a borrowed
    /// `SessionOutputs` retains every original until the collection
    /// drops (`ae582a3b`).
    fn take_output<T>(
        out: ort::session::SessionOutputs<'_>,
        want: &str,
        get: impl Fn(&ort::value::DynValue) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let mut found = None;
        for (name, value) in out {
            if name == want {
                found = Some(get(&value)?);
            }
        }
        found.ok_or_else(|| Error::BadModel(format!("missing output {want}")))
    }

    /// The fused pair-graph session on the selected GPU provider via
    /// [`gpu_session`].
    pub(crate) struct OrtGpuPair {
        session: Mutex<Session>,
        output: &'static str,
        ep: Ep,
    }

    impl OrtGpuPair {
        pub(crate) fn load(
            path: &Path,
            output: &'static str,
            engine: crate::Engine,
        ) -> Result<Self, Error> {
            let ep = Ep::of(engine)
                .ok_or_else(|| Error::Engine(format!("{engine:?} is not an ORT GPU engine")))?;
            let session = gpu_session(environment()?, path, ep)?;
            Ok(Self {
                session: Mutex::new(session),
                output,
                ep,
            })
        }
    }

    impl PairGraph for OrtGpuPair {
        fn engine(&self) -> &'static str {
            self.ep.name()
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
            take_output(out, self.output, |v| {
                v.try_extract_scalar::<f32>()
                    .map(f64::from)
                    .map_err(|e| Error::Engine(e.to_string()))
            })
        }
    }

    /// [`OrtGpuPair`] for a single-image graph with named inputs —
    /// [`ImageGraph`] on the same session configuration.
    pub(crate) struct OrtGpuImage {
        session: Mutex<Session>,
        output: &'static str,
        ep: Ep,
    }

    impl OrtGpuImage {
        pub(crate) fn load(
            path: &Path,
            output: &'static str,
            engine: crate::Engine,
        ) -> Result<Self, Error> {
            let ep = Ep::of(engine)
                .ok_or_else(|| Error::Engine(format!("{engine:?} is not an ORT GPU engine")))?;
            let session = gpu_session(environment()?, path, ep)?;
            Ok(Self {
                session: Mutex::new(session),
                output,
                ep,
            })
        }
    }

    impl ImageGraph for OrtGpuImage {
        fn engine(&self) -> &'static str {
            self.ep.name()
        }

        fn score(&self, inputs: &[NamedInput]) -> Result<f64, Error> {
            let feeds: Vec<(&'static str, Tensor<f32>)> = inputs
                .iter()
                .map(|i| {
                    Tensor::from_array((i.shape.clone(), i.data.clone()))
                        .map(|t| (i.name, t))
                        .map_err(|e| Error::Engine(e.to_string()))
                })
                .collect::<Result<_, _>>()?;
            let mut session = self
                .session
                .lock()
                .map_err(|_| Error::Engine("session lock poisoned".into()))?;
            let out = session
                .run(feeds)
                .map_err(|e| Error::Engine(e.to_string()))?;
            take_output(out, self.output, |v| {
                v.try_extract_scalar::<f32>()
                    .map(f64::from)
                    .map_err(|e| Error::Engine(e.to_string()))
            })
        }

        fn tensor(&self, inputs: &[NamedInput]) -> Result<(Vec<usize>, Vec<f32>), Error> {
            let feeds: Vec<(&'static str, Tensor<f32>)> = inputs
                .iter()
                .map(|i| {
                    Tensor::from_array((i.shape.clone(), i.data.clone()))
                        .map(|t| (i.name, t))
                        .map_err(|e| Error::Engine(e.to_string()))
                })
                .collect::<Result<_, _>>()?;
            let mut session = self
                .session
                .lock()
                .map_err(|_| Error::Engine("session lock poisoned".into()))?;
            let out = session
                .run(feeds)
                .map_err(|e| Error::Engine(e.to_string()))?;
            take_output(out, self.output, |v| {
                let (shape, data) = v
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::Engine(e.to_string()))?;
                Ok((shape.iter().map(|&d| d as usize).collect(), data.to_vec()))
            })
        }
    }
    /// Multi-output feature graph with bounded host output retention.
    /// `load` uses the WD graph's names (`x` -> `tap0..tap3`); `load_named`
    /// takes an explicit input name and ordered output list for trunks with
    /// different conventions (DMM's `x` -> `t3`,`t4`).
    pub(crate) struct OrtGpuTrunk {
        session: Mutex<Session>,
        input: &'static str,
        outputs: Vec<&'static str>,
        ep: Ep,
    }
    impl OrtGpuTrunk {
        pub(crate) fn load(path: &Path, engine: crate::Engine) -> Result<Self, Error> {
            Self::load_named(path, engine, "x", vec!["tap0", "tap1", "tap2", "tap3"])
        }
        pub(crate) fn load_named(
            path: &Path,
            engine: crate::Engine,
            input: &'static str,
            outputs: Vec<&'static str>,
        ) -> Result<Self, Error> {
            let ep = Ep::of(engine)
                .ok_or_else(|| Error::Engine(format!("{engine:?} is not an ORT GPU engine")))?;
            let session = gpu_session(environment()?, path, ep)?;
            Ok(Self {
                session: Mutex::new(session),
                input,
                outputs,
                ep,
            })
        }
    }
    impl super::Trunk for OrtGpuTrunk {
        fn engine(&self) -> &'static str {
            self.ep.name()
        }
        fn run(&self, input: Vec<f32>, h: usize, w: usize) -> Result<Vec<super::Tap>, Error> {
            let input = Tensor::from_array(([1usize, 3, h, w], input))
                .map_err(|e| Error::Engine(e.to_string()))?;
            let mut session = self
                .session
                .lock()
                .map_err(|_| Error::Engine("session lock poisoned".into()))?;
            let out = session
                .run(ort::inputs![self.input => input])
                .map_err(|e| Error::Engine(e.to_string()))?;
            let mut taps: Vec<Option<super::Tap>> = vec![None; self.outputs.len()];
            // Consume each ORT value, releasing it immediately after its copy.
            // Indexing a borrowed SessionOutputs retains every original until
            // the complete collection has been cloned.
            for (name, value) in out {
                let index = self
                    .outputs
                    .iter()
                    .position(|&o| o == name)
                    .ok_or_else(|| Error::BadModel(format!("unexpected trunk output {name}")))?;
                let (shape, data) = value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::Engine(e.to_string()))?;
                if shape.len() != 4 || shape[0] != 1 || shape[1..].iter().any(|&x| x <= 0) {
                    return Err(Error::BadModel(format!("trunk output shape {shape:?}")));
                }
                taps[index] = Some(super::Tap {
                    c: shape[1] as usize,
                    h: shape[2] as usize,
                    w: shape[3] as usize,
                    data: data.to_vec(),
                });
            }
            taps.into_iter()
                .map(|tap| tap.ok_or_else(|| Error::BadModel("missing trunk output".into())))
                .collect()
        }
    }
}
