//! Pieces shared by every model: exported-file readers and the engine split.

use std::path::Path;

use crate::Error;
use crate::engine::{ImageGraph, PairGraph, Trunk};

pub(crate) enum Backend {
    /// Trunk graph on the engine, head here: taps come back to the host.
    Trunk(Box<dyn Trunk>),
    /// The whole metric in one graph; only the score comes back.
    #[cfg_attr(not(feature = "ort"), allow(dead_code))]
    Pair(Box<dyn PairGraph>),
    /// A whole-image NR graph; only the score comes back.
    #[cfg_attr(not(feature = "ort"), allow(dead_code))]
    Image(Box<dyn ImageGraph>),
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
                min_size = Some(v.trim().parse::<usize>().map_err(|_| bad("bad min_size"))?)
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
