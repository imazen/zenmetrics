//! Collision-proof scratch files for staged transfers (s5cmd uploads/downloads).
//!
//! A temp name built only from `std::process::id()` and an in-process counter is NOT unique when
//! several containers share one `TMPDIR`: each container has its own PID namespace, so two workers
//! routinely carry the same pid, and their counters start at the same value. On 2026-10-03 two tower
//! fit workers (pids equal inside their containers, `TMPDIR=/scratch` bound to one host directory)
//! finished a chunk at the same moment and staged their ledger sidecars through the same
//! `zenledger_ul_{pid}_{n}.parquet`; one worker uploaded the other's bytes under its own sidecar key,
//! so its Done row was lost and the cell sat claimed-but-unrecorded for a full claim TTL before it
//! was re-run (zenmetrics#63). [`unique_temp_path`] adds wall-clock nanos to the name and creates the
//! file with `create_new` (O_EXCL), retrying under a fresh name on a collision, so no two writers can
//! ever share a staging file.

use std::fs::OpenOptions;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static N: AtomicU64 = AtomicU64::new(0);

/// Create a new, empty file in [`std::env::temp_dir`] that no other process or container can be
/// using, and return its path: `{tag}_{pid}_{counter}_{nanos}.{ext}`, created with `create_new`.
/// The caller owns the file (write it, hand the path to a downloader that overwrites it, then
/// remove it).
pub fn unique_temp_path(tag: &str, ext: &str) -> io::Result<PathBuf> {
    create_unique_in(&std::env::temp_dir(), |_| {
        let n = N.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("{tag}_{}_{n}_{nanos:x}.{ext}", std::process::id())
    })
}

/// Create the first name from `name(attempt)` that does not exist yet in `dir`, exclusively.
fn create_unique_in(dir: &Path, mut name: impl FnMut(u32) -> String) -> io::Result<PathBuf> {
    const ATTEMPTS: u32 = 64;
    for attempt in 0..ATTEMPTS {
        let path = dir.join(name(attempt));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(_) => return Ok(path),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        format!(
            "no free temp name in {} after {ATTEMPTS} attempts",
            dir.display()
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "zenfleet_tmp_test_{tag}_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// The 2026-10-03 failure: another process (same pid, same counter) already holds the name this
    /// process computes first. The helper must move on to a fresh name and leave the foreign file
    /// untouched — never open, truncate or reuse it.
    #[test]
    fn a_name_held_by_another_process_is_never_shared() {
        let dir = scratch("foreign");
        std::fs::write(
            dir.join("zenledger_ul_7_0.parquet"),
            b"other worker's sidecar",
        )
        .unwrap();
        let p = create_unique_in(&dir, |attempt| {
            if attempt == 0 {
                "zenledger_ul_7_0.parquet".into()
            } else {
                format!("zenledger_ul_7_0_{attempt}.parquet")
            }
        })
        .unwrap();
        assert_eq!(p.file_name().unwrap(), "zenledger_ul_7_0_1.parquet");
        assert_eq!(
            std::fs::read(dir.join("zenledger_ul_7_0.parquet")).unwrap(),
            b"other worker's sidecar"
        );
        assert_eq!(std::fs::read(&p).unwrap(), b"", "the new file starts empty");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Two writers computing the SAME name sequence (the shared-TMPDIR, equal-pid case) get two
    /// different files; the second never reuses the first's.
    #[test]
    fn equal_name_sequences_get_distinct_files() {
        let dir = scratch("equal");
        let seq = |attempt: u32| format!("zenledger_ul_1_0_{attempt}.parquet");
        let a = create_unique_in(&dir, seq).unwrap();
        let b = create_unique_in(&dir, seq).unwrap();
        assert_ne!(a, b);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn exhausting_attempts_is_an_error_not_a_shared_file() {
        let dir = scratch("exhaust");
        std::fs::write(dir.join("taken.bin"), b"x").unwrap();
        let e = create_unique_in(&dir, |_| "taken.bin".into()).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::AlreadyExists);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unique_temp_path_creates_an_empty_named_file() {
        let a = unique_temp_path("zenfleet_tmp_test", "parquet").unwrap();
        let b = unique_temp_path("zenfleet_tmp_test", "parquet").unwrap();
        assert_ne!(a, b);
        let name = a.file_name().unwrap().to_string_lossy().into_owned();
        assert!(
            name.starts_with("zenfleet_tmp_test_") && name.ends_with(".parquet"),
            "{name}"
        );
        assert_eq!(std::fs::metadata(&a).unwrap().len(), 0);
        std::fs::remove_file(&a).unwrap();
        std::fs::remove_file(&b).unwrap();
    }
}
