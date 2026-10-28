//! Opt-in archive CAS: atomic immutable files, independently of S3 claim/ledger transport.
use super::{Sha256Hex, sha256};
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
static SEQ: AtomicU64 = AtomicU64::new(0);
pub(super) fn path(root: &Path, sha: &Sha256Hex) -> PathBuf {
    let s = sha.as_str();
    root.join(&s[..2]).join(&s[2..4]).join(s)
}
pub(super) fn put(root: &Path, bytes: &[u8]) -> io::Result<Sha256Hex> {
    if !root.is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "archive CAS must be absolute",
        ));
    }
    let sha = sha256(bytes);
    let target = path(root, &sha);
    let parent = target.parent().unwrap();
    fs::create_dir_all(parent)?;
    if target.exists() {
        if sha256(&fs::read(&target)?) != sha {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "archive CAS bytes differ",
            ));
        }
        return Ok(sha);
    }
    let worker = std::env::var("ZEN_WORKER")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "local".into());
    let worker: String = worker
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let (temp, mut file) = loop {
        let temp = parent.join(format!(
            ".pending-{worker}-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        match fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)
        {
            Ok(file) => break (temp, file),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    };
    file.write_all(bytes)?;
    drop(file);
    match fs::hard_link(&temp, &target) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
            if sha256(&fs::read(&target)?) != sha {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "archive CAS collision/corruption",
                ));
            }
        }
        Err(e) => return Err(e),
    }
    fs::remove_file(temp)?;
    Ok(sha)
}
// The batch stage and CAS share one archive mount: hard-link the verified
// object instead of reading and writing the full container a second time.
pub(super) fn publish(root: &Path, source: &Path, expected: &Sha256Hex) -> io::Result<()> {
    let target = path(root, expected);
    fs::create_dir_all(target.parent().unwrap())?;
    match fs::hard_link(source, &target) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
            if sha256(&fs::read(target)?) == *expected {
                Ok(())
            } else {
                Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "archive CAS differs",
                ))
            }
        }
        Err(e) if e.raw_os_error() == Some(18) => {
            let actual = put(root, &fs::read(source)?)?;
            if actual == *expected {
                Ok(())
            } else {
                Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "archive copy differs",
                ))
            }
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parallel_publish_is_complete_and_corruption_refused() {
        let root = std::env::temp_dir().join(format!("archive-cas-{}", std::process::id()));
        let data = vec![37u8; 65536];
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    put(&root, &data).unwrap();
                });
            }
        });
        let sha = sha256(&data);
        assert_eq!(fs::read(path(&root, &sha)).unwrap(), data);
        fs::write(path(&root, &sha), b"bad").unwrap();
        assert!(put(&root, &data).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
