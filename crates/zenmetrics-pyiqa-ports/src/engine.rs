//! Engines for this crate's exported graphs.
//!
//! A [`Graph`] takes its inputs positionally (the graph's declared input
//! order) and returns every declared output, as row-major f32 [`Tensor`]s.
//! Engines: [`Engine::Rten`] (pure-Rust CPU, default feature) and, with the
//! `ort` feature, ONNX Runtime on the CPU ([`Engine::OrtCpu`]) or on CUDA
//! with TF32 off ([`Engine::OrtCuda`]), loaded at run time from
//! `ORT_DYLIB_PATH`.
//!
//! Adapted from `crates/zenmetrics-nn/src/{engine,common,error}.rs` (commit
//! e9cd38d2), whose items are crate-private: the rten loading and the CUDA
//! session configuration follow it; the positional multi-input interface and
//! the ONNX Runtime CPU engine are new here.

use std::path::Path;

use zenmetrics_nn::{Error, WeightsConsent, WeightsTerms};

use crate::za;

/// `consent` accepts `terms` (zenmetrics-nn's `WeightsConsent::check`).
pub(crate) fn accept(consent: &WeightsConsent, terms: &WeightsTerms) -> Result<(), Error> {
    if consent.accepts(terms) {
        Ok(())
    } else {
        Err(Error::WeightsNotAccepted(*terms))
    }
}

/// The [`za::ErrorKind`] of an [`Error`] (zenmetrics-nn's `Error::kind`).
pub(crate) fn kind(e: &Error) -> za::ErrorKind {
    match e {
        Error::TooSmall { .. } => za::ErrorKind::TooSmall,
        Error::TooLarge { .. } | Error::TooManyPatches { .. } | Error::DegenerateAspectRatio => {
            za::ErrorKind::InvalidInput
        }
        Error::SizeMismatch => za::ErrorKind::DimensionMismatch,
        Error::Engine(_) => za::ErrorKind::Backend,
        _ => za::ErrorKind::InvalidConfig,
    }
}

/// Which engine runs a graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Engine {
    /// rten, pure-Rust CPU (feature `rten`, default).
    Rten,
    /// ONNX Runtime, CPU provider (feature `ort`).
    #[cfg(feature = "ort")]
    OrtCpu,
    /// ONNX Runtime, CUDA provider with TF32 off and no CPU fallback for
    /// supported nodes (feature `ort`).
    #[cfg(feature = "ort")]
    OrtCuda,
}

impl Engine {
    /// Parse `rten`, `ort-cpu` or `ort-cuda`.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "rten" => Some(Self::Rten),
            #[cfg(feature = "ort")]
            "ort-cpu" => Some(Self::OrtCpu),
            #[cfg(feature = "ort")]
            "ort-cuda" => Some(Self::OrtCuda),
            _ => None,
        }
    }

    /// The name [`Engine::parse`] accepts.
    pub fn name(self) -> &'static str {
        match self {
            Self::Rten => "rten",
            #[cfg(feature = "ort")]
            Self::OrtCpu => "ort-cpu",
            #[cfg(feature = "ort")]
            Self::OrtCuda => "ort-cuda",
        }
    }
}

/// A row-major f32 tensor.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Tensor {
    /// Dimensions, outermost first.
    pub(crate) shape: Vec<usize>,
    /// `shape.iter().product()` elements.
    pub(crate) data: Vec<f32>,
}

impl Tensor {
    /// A tensor; panics if `data` does not hold `shape`'s element count.
    pub(crate) fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        assert_eq!(shape.iter().product::<usize>(), data.len(), "tensor shape");
        Self { shape, data }
    }
}

/// A loaded graph.
pub(crate) trait Graph: Send + Sync {
    /// Engine name, for provenance.
    fn engine(&self) -> &'static str;
    /// Run with `inputs` in the graph's declared input order; returns every
    /// declared output in order. Owned inputs: rten reuses them as scratch.
    fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, Error>;
}

/// Worker count for host-side passes and the ONNX Runtime CPU provider:
/// `RTEN_NUM_THREADS` when set, else available parallelism.
pub(crate) fn workers() -> usize {
    std::env::var("RTEN_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .or_else(|| std::thread::available_parallelism().ok().map(usize::from))
        .unwrap_or(1)
        .max(1)
}

/// Load the ONNX graph at `path` on `engine`.
pub(crate) fn load(path: &Path, engine: Engine) -> Result<Box<dyn Graph>, Error> {
    match engine {
        Engine::Rten => {
            #[cfg(feature = "rten")]
            {
                Ok(Box::new(rten_engine::RtenGraph::load(path)?))
            }
            #[cfg(not(feature = "rten"))]
            {
                let _ = path;
                Err(Error::Engine("built without the rten feature".into()))
            }
        }
        #[cfg(feature = "ort")]
        Engine::OrtCpu => Ok(Box::new(ort_engine::OrtGraph::load(path, false)?)),
        #[cfg(feature = "ort")]
        Engine::OrtCuda => Ok(Box::new(ort_engine::OrtGraph::load(path, true)?)),
    }
}

/// `<prefix>.meta`: `channels c0,c1,...` and `min_size n`.
pub(crate) fn read_meta(path: &Path) -> Result<(Vec<usize>, usize), Error> {
    let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let bad = |m: &str| Error::BadModel(format!("{}: {m}", path.display()));
    let (mut channels, mut min_size) = (None, None);
    for line in text.lines() {
        match line.split_once(' ') {
            Some(("channels", v)) => {
                channels = Some(
                    v.split(',')
                        .map(|c| {
                            c.trim()
                                .parse::<usize>()
                                .map_err(|_| bad("bad channel count"))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            Some(("min_size", v)) => {
                min_size = Some(v.trim().parse::<usize>().map_err(|_| bad("bad min_size"))?);
            }
            _ => {}
        }
    }
    let channels = channels
        .filter(|c| !c.is_empty())
        .ok_or_else(|| bad("no channels line"))?;
    Ok((
        channels,
        min_size.ok_or_else(|| bad("no min_size line"))?.max(1),
    ))
}

/// Raw little-endian f32 weights, exactly `expected` values, all finite.
pub(crate) fn read_f32(path: &Path, expected: usize) -> Result<Vec<f32>, Error> {
    let raw = std::fs::read(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if raw.len() != expected * 4 {
        return Err(Error::BadModel(format!(
            "{}: {} bytes, expected {}",
            path.display(),
            raw.len(),
            expected * 4
        )));
    }
    let v: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    if v.iter().any(|x| !x.is_finite()) {
        return Err(Error::BadModel(format!(
            "{}: non-finite weight",
            path.display()
        )));
    }
    Ok(v)
}

/// Packed RGB8 rows (any stride ≥ `3·w`) → planar `[3, h, w]` f32 in
/// `[0, 1]`: `v as f32 / 255.0`, torchvision `to_tensor`'s
/// `img.float().div(255)`.
pub(crate) fn rgb8_to_planar_01(img: &za::Rgb8In<'_>) -> Vec<f32> {
    let (w, h, stride, rgb) = (img.width, img.height, img.stride, img.pixels);
    let plane = w * h;
    let mut out = vec![0.0f32; 3 * plane];
    for y in 0..h {
        let row = &rgb[y * stride..y * stride + 3 * w];
        for x in 0..w {
            for c in 0..3 {
                out[c * plane + y * w + x] = f32::from(row[3 * x + c]) / 255.0;
            }
        }
    }
    out
}

#[cfg(feature = "rten")]
mod rten_engine {
    use std::path::Path;

    use rten::{Model, ModelOptions, NodeId};
    use rten_tensor::Layout;

    use super::{Error, Graph, Tensor};

    pub(crate) struct RtenGraph {
        model: Model,
        inputs: Vec<NodeId>,
        outputs: Vec<NodeId>,
    }

    impl RtenGraph {
        pub(crate) fn load(path: &Path) -> Result<Self, Error> {
            let model = ModelOptions::with_all_ops()
                .load_file(path)
                .map_err(|e| Error::BadModel(format!("{}: {e}", path.display())))?;
            let inputs = model.input_ids().to_vec();
            let outputs = model.output_ids().to_vec();
            if inputs.is_empty() || outputs.is_empty() {
                return Err(Error::BadModel(format!(
                    "{}: no inputs or outputs",
                    path.display()
                )));
            }
            Ok(Self {
                model,
                inputs,
                outputs,
            })
        }
    }

    impl Graph for RtenGraph {
        fn engine(&self) -> &'static str {
            "rten"
        }

        fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, Error> {
            if inputs.len() != self.inputs.len() {
                return Err(Error::BadModel(format!(
                    "graph takes {} inputs, got {}",
                    self.inputs.len(),
                    inputs.len()
                )));
            }
            let feeds = self
                .inputs
                .iter()
                .zip(inputs)
                .map(|(&id, t)| (id, rten_tensor::Tensor::from_data(&t.shape, t.data).into()))
                .collect();
            let values = self
                .model
                .run(feeds, &self.outputs, None)
                .map_err(|e| Error::Engine(e.to_string()))?;
            values
                .into_iter()
                .map(|v| {
                    let t: rten_tensor::Tensor<f32> = v
                        .try_into()
                        .map_err(|e| Error::Engine(format!("output: {e:?}")))?;
                    let shape = t.shape().to_vec();
                    Ok(Tensor {
                        shape,
                        data: t.into_data(),
                    })
                })
                .collect()
        }
    }
}

#[cfg(feature = "ort")]
mod ort_engine {
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};

    use ort::environment::Environment;
    use ort::ep;
    use ort::session::Session;

    use super::{Error, Graph, Tensor};

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
                .with_name("zenmetrics-pyiqa-ports")
                .build()
                .map_err(|e| format!("ONNX Runtime environment: {e}"))
        })
        .as_ref()
        .map_err(|e| Error::Engine(e.clone()))
    }

    /// CPU: arena off (shape-varying inputs otherwise keep every size's
    /// buffers), intra-op threads = [`super::workers`]. CUDA: as zenmetrics-nn's
    /// `cuda_session` — TF32 off, exhaustive cuDNN search, deterministic
    /// compute, no memory pattern, CPU arena off, error if CUDA fails.
    fn session(env: &Environment, path: &Path, cuda: bool) -> Result<Session, Error> {
        let cpu = ep::CPU::default().with_arena_allocator(false).build();
        let threads = super::workers();
        let builder = Session::builder(env)
            .and_then(|b| b.with_deterministic_compute(true).map_err(Into::into))
            .and_then(|b| b.with_memory_pattern(false).map_err(Into::into));
        let builder = if cuda {
            let cuda = ep::CUDA::default()
                .with_tf32(false)
                .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Exhaustive)
                .build()
                .error_on_failure();
            builder.and_then(|b| b.with_execution_providers([cuda, cpu]).map_err(Into::into))
        } else {
            builder
                .and_then(|b| b.with_intra_threads(threads).map_err(Into::into))
                .and_then(|b| b.with_execution_providers([cpu]).map_err(Into::into))
        };
        builder
            .and_then(|mut b| b.commit_from_file(path))
            .map_err(|e| Error::BadModel(format!("{}: {e}", path.display())))
    }

    pub(crate) struct OrtGraph {
        session: Mutex<Session>,
        inputs: Vec<String>,
        outputs: Vec<String>,
        cuda: bool,
    }

    impl OrtGraph {
        pub(crate) fn load(path: &Path, cuda: bool) -> Result<Self, Error> {
            let session = session(environment()?, path, cuda)?;
            let inputs = session
                .inputs()
                .iter()
                .map(|o| o.name().to_string())
                .collect();
            let outputs = session
                .outputs()
                .iter()
                .map(|o| o.name().to_string())
                .collect();
            Ok(Self {
                session: Mutex::new(session),
                inputs,
                outputs,
                cuda,
            })
        }
    }

    impl Graph for OrtGraph {
        fn engine(&self) -> &'static str {
            if self.cuda { "ort-cuda" } else { "ort-cpu" }
        }

        fn run(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, Error> {
            if inputs.len() != self.inputs.len() {
                return Err(Error::BadModel(format!(
                    "graph takes {} inputs, got {}",
                    self.inputs.len(),
                    inputs.len()
                )));
            }
            let feeds: Vec<(String, ort::value::Tensor<f32>)> = self
                .inputs
                .iter()
                .zip(inputs)
                .map(|(name, t)| {
                    ort::value::Tensor::from_array((t.shape, t.data))
                        .map(|v| (name.clone(), v))
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
            let mut got: Vec<Option<Tensor>> = vec![None; self.outputs.len()];
            // Consume each value as it is read, so every original is released
            // after its copy (zenmetrics-nn `take_output`, ae582a3b).
            for (name, value) in out {
                let i = self
                    .outputs
                    .iter()
                    .position(|o| o == name)
                    .ok_or_else(|| Error::BadModel(format!("unexpected output {name}")))?;
                let (shape, data) = value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::Engine(e.to_string()))?;
                got[i] = Some(Tensor {
                    shape: shape.iter().map(|&d| d as usize).collect(),
                    data: data.to_vec(),
                });
            }
            got.into_iter()
                .map(|t| t.ok_or_else(|| Error::BadModel("missing output".into())))
                .collect()
        }
    }
}
