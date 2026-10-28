//! Bounded per-view staging avoids buffering every EVIF in a pipe at once.
use super::{BlobStore, DesiredJob, Sha256Hex, sha256};
use serde::Deserialize;
use std::{
    fs, io,
    path::{Path, PathBuf},
};
pub(super) const MAGIC: &[u8] = b"ZFEVIFSTAGE1\n";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    root: PathBuf,
    primary: String,
    objects: Vec<Object>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Object {
    path: String,
    sha256: String,
}
fn bad() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid staged EVIF output")
}
fn file(root: &Path, name: &str) -> io::Result<PathBuf> {
    let rel = Path::new(name);
    if rel.is_absolute()
        || rel
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err(bad());
    }
    let p = root.join(rel);
    if fs::symlink_metadata(&p)?.file_type().is_symlink() || !p.canonicalize()?.starts_with(root) {
        return Err(bad());
    }
    Ok(p)
}
pub(super) fn persist(
    store: &dyn BlobStore,
    job: &DesiredJob,
    bytes: &[u8],
) -> io::Result<Sha256Hex> {
    let scratch = PathBuf::from(std::env::var_os("TMPDIR").ok_or_else(bad)?);
    persist_with_scratch(store, job, bytes, &scratch)
}
fn persist_with_scratch(
    store: &dyn BlobStore,
    job: &DesiredJob,
    bytes: &[u8],
    scratch: &Path,
) -> io::Result<Sha256Hex> {
    if bytes.len() > 1_048_576 {
        return Err(bad());
    }
    let env: Envelope =
        serde_json::from_slice(bytes.strip_prefix(MAGIC).ok_or_else(bad)?).map_err(|_| bad())?;
    let scratch = scratch.canonicalize()?;
    let root = env.root.canonicalize()?;
    let archive = std::env::var_os("ZEN_ARCHIVE_BLOBS_DIR").map(PathBuf::from);
    if archive.as_ref().is_some_and(|p| !p.is_absolute()) {
        return Err(bad());
    }
    let archive_stage = archive
        .as_ref()
        .map(|p| p.join(".staging"))
        .and_then(|p| p.canonicalize().ok());
    let inside_archive = archive_stage
        .as_ref()
        .is_some_and(|p| root != *p && root.starts_with(p));
    if root == scratch
        || (!root.starts_with(&scratch) && !inside_archive)
        || env.objects.is_empty()
        || env.objects.len() > 512
    {
        return Err(bad());
    }
    let primary = file(&root, &env.primary)?;
    let primary_bytes = fs::read(primary)?;
    let manifest: serde_json::Value = serde_json::from_slice(&primary_bytes).map_err(|_| bad())?;
    if manifest["image_path"].as_str() != Some(job.cell.image_path.as_str())
        || manifest["knob_tuple_json"].as_str() != Some(job.cell.knob_tuple_json.as_str())
    {
        return Err(bad());
    }
    let mut total = 0u64;
    for object in &env.objects {
        let p = file(&root, &object.path)?;
        let size = fs::metadata(&p)?.len();
        total = total.checked_add(size).ok_or_else(bad)?;
        if size > 32 * 1024 * 1024 || total > 1024 * 1024 * 1024 {
            return Err(bad());
        }
        let data = fs::read(&p)?;
        let expected = Sha256Hex::parse(object.sha256.clone()).map_err(|_| bad())?;
        if sha256(&data) != expected {
            return Err(bad());
        }
        if let Some(archive) = &archive {
            if inside_archive {
                super::archive_store::publish(archive, &p, &expected)?;
            } else if store.put(&data)? != expected {
                return Err(bad());
            }
        } else if store.put(&data)? != expected {
            return Err(bad());
        }
    }
    let result = store.put(&primary_bytes)?;
    // Every scored raster, recipe and native bitstream now has a verified CAS
    // copy. Only duplicate transient transport files are removed after success.
    // NFS backed by FUSE may retain .fuse_hidden files until server handles
    // expire. Publication has already succeeded: transient cleanup cannot
    // turn a verified, durable result into an upload failure.
    if let Err(error) = fs::remove_dir_all(&root) {
        eprintln!(
            "zenfleet-worker: published staged output; cleanup retained {}: {error}",
            root.display()
        );
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LocalBlobStore;
    use zenfleet_core::{CellId, JobKind};
    #[test]
    fn staged_objects_commit_before_primary_and_escape_is_refused() {
        let scratch = std::env::var_os("TMPDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(
                    std::env::var_os("HOME")
                        .or_else(|| std::env::var_os("USERPROFILE"))
                        .expect("test home directory"),
                )
                .join("tmp")
            });
        fs::create_dir_all(&scratch).unwrap();
        let base = scratch.join(format!("staged-evif-{}", std::process::id()));
        fs::create_dir(&base).unwrap();
        let root = base.join("stage");
        fs::create_dir(&root).unwrap();
        fs::write(root.join("native"), b"jpeg").unwrap();
        fs::write(
            root.join("result.json"),
            br#"{"image_path":"view.png","knob_tuple_json":"{}"}"#,
        )
        .unwrap();
        let store = LocalBlobStore::new(base.join("store")).unwrap();
        let job = DesiredJob::new(
            JobKind::Metric {
                metric: "evif-chroma-batch-v1".into(),
            },
            vec![],
            CellId {
                image_path: "view.png".into(),
                codec: "evif:batch".into(),
                q: 95,
                knob_tuple_json: "{}".into(),
            },
        );
        let mut value = serde_json::json!({"root":root,"primary":"result.json","objects":[{"path":"../escape","sha256":sha256(b"jpeg").as_str()}]});
        let wire = |v: &serde_json::Value| {
            let mut b = MAGIC.to_vec();
            b.extend(serde_json::to_vec(v).unwrap());
            b
        };
        assert!(persist_with_scratch(&store, &job, &wire(&value), &base).is_err());
        assert!(root.exists());
        value["objects"][0]["path"] = "native".into();
        let primary = persist_with_scratch(&store, &job, &wire(&value), &base).unwrap();
        assert!(store.exists(&primary));
        assert!(store.exists(&sha256(b"jpeg")));
        assert!(!root.exists());
        fs::remove_dir_all(base).unwrap();
    }
}
