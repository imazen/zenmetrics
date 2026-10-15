use std::fmt;
use std::path::PathBuf;

use crate::za;

/// Why a model failed to load or score.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A model file could not be read.
    Io {
        /// The file.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// A model file is malformed or does not match what the head expects.
    BadModel(String),
    /// The engine failed while running the graph.
    Engine(String),
    /// The image is smaller than the trunk's receptive field allows.
    TooSmall {
        /// Smallest supported width and height.
        min: usize,
    },
    /// Reference and distorted images differ in size.
    SizeMismatch,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "reading {}: {source}", path.display()),
            Self::BadModel(m) => write!(f, "bad model: {m}"),
            Self::Engine(m) => write!(f, "engine: {m}"),
            Self::TooSmall { min } => write!(f, "image smaller than {min}x{min}"),
            Self::SizeMismatch => f.write_str("reference and distorted sizes differ"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl Error {
    pub(crate) fn kind(&self) -> za::ErrorKind {
        match self {
            Self::TooSmall { .. } => za::ErrorKind::TooSmall,
            Self::SizeMismatch => za::ErrorKind::DimensionMismatch,
            Self::Io { .. } | Self::BadModel(_) => za::ErrorKind::InvalidConfig,
            Self::Engine(_) => za::ErrorKind::Backend,
        }
    }
}
