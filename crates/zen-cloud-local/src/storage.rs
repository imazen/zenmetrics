//! Local `BlobStorage` — a filesystem mirror of object storage.
//!
//! Every cloud backend's `BlobStorage` speaks `s3://bucket/key` URIs.
//! [`LocalFsStorage`] resolves those (and `file://…` URIs, and plain
//! relative paths) under a single base directory so the SAME compute
//! closure that `put`s `s3://zentrain/run/omni/c0.parquet` in the cloud
//! writes `<base>/zentrain/run/omni/c0.parquet` on disk locally — no
//! network, no spend.
//!
//! ## Key → path mapping
//!
//! | [`ArtifactKey`] form | resolves to |
//! |---|---|
//! | `s3://bucket/a/b.parquet` | `<base>/bucket/a/b.parquet` |
//! | `r2://bucket/a/b.parquet` | `<base>/bucket/a/b.parquet` |
//! | `gs://bucket/a/b.parquet` | `<base>/bucket/a/b.parquet` |
//! | `file:///abs/path` | `/abs/path` (absolute, base ignored) |
//! | `file://rel/path` | `<base>/rel/path` |
//! | `a/b.parquet` (plain) | `<base>/a/b.parquet` |
//!
//! For an `s3://…`-style URI the bucket becomes the first path segment,
//! so a chunk that references several buckets mirrors them as sibling
//! directories under one base — the natural "local mirror dir" layout.
//! `list(prefix)` accepts the same key forms and walks the resolved
//! sub-tree, returning keys in the SAME scheme the prefix used (so a
//! caller listing `s3://bucket/run/` gets `s3://bucket/run/…` keys back).
//!
//! Path traversal is contained: a resolved relative path is rejected if
//! it escapes the base dir via `..`.

use std::path::{Component, Path, PathBuf};

use zen_cloud_core::{ArtifactKey, BlobMeta, BlobStorage, CloudError};

/// Filesystem-backed [`BlobStorage`] rooted at a single base dir.
#[derive(Clone, Debug)]
pub struct LocalFsStorage {
    base: PathBuf,
}

/// The parsed scheme of an [`ArtifactKey`], remembered so `list` can
/// reconstruct keys in the caller's original scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scheme {
    /// `s3://`, `r2://`, `gs://`, … — bucket-prefixed object store.
    Bucketed,
    /// `file://` — a filesystem URI.
    File,
    /// A plain relative path with no scheme.
    Plain,
}

impl LocalFsStorage {
    /// Root a new filesystem store at `base`. The base dir is created on
    /// the first `put` (lazily) — constructing the store does no IO.
    pub fn new(base: impl Into<PathBuf>) -> Self {
        Self { base: base.into() }
    }

    /// The base directory this store mirrors under.
    pub fn base(&self) -> &Path {
        &self.base
    }

    /// Split a key into its scheme + the path portion relative to the
    /// base (or absolute, for `file:///…`).
    ///
    /// Returns `(scheme, scheme_prefix, rel_or_abs)`. `scheme_prefix` is
    /// the literal prefix (`"s3://"`, `"file://"`, or `""`) needed to
    /// reconstruct a key in the same scheme during `list`.
    fn split_scheme(key: &str) -> (Scheme, &'static str, &str) {
        // Bucketed object-store schemes all map identically (the scheme
        // is cosmetic locally; the bucket is just the first path segment).
        for prefix in ["s3://", "r2://", "gs://", "https://"] {
            if let Some(rest) = key.strip_prefix(prefix) {
                // Reconstruct as canonical `s3://` so list output is
                // stable regardless of the alias the caller used.
                let _ = prefix;
                return (Scheme::Bucketed, "s3://", rest);
            }
        }
        if let Some(rest) = key.strip_prefix("file://") {
            return (Scheme::File, "file://", rest);
        }
        (Scheme::Plain, "", key)
    }

    /// Resolve an [`ArtifactKey`] to an absolute filesystem path under
    /// (or, for `file:///abs`, ignoring) the base dir. Rejects `..`
    /// traversal out of the base for the relative forms.
    fn resolve(&self, key: &ArtifactKey) -> Result<PathBuf, CloudError> {
        let (scheme, _prefix, rest) = Self::split_scheme(key.as_str());

        // `file:///abs/path` → an absolute host path (base ignored).
        if scheme == Scheme::File && rest.starts_with('/') {
            return Ok(PathBuf::from(rest));
        }

        let rel = rest.trim_start_matches('/');
        if rel.is_empty() {
            return Err(CloudError::Storage(format!(
                "empty key after scheme: {key}"
            )));
        }

        // Contain traversal: reject any `..` component.
        let rel_path = PathBuf::from(rel);
        for comp in rel_path.components() {
            if matches!(comp, Component::ParentDir) {
                return Err(CloudError::Storage(format!(
                    "key escapes base via `..`: {key}"
                )));
            }
        }
        Ok(self.base.join(rel_path))
    }
}

impl BlobStorage for LocalFsStorage {
    fn put(&self, key: &ArtifactKey, bytes: &[u8]) -> Result<(), CloudError> {
        let path = self.resolve(key)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                CloudError::Storage(format!("create dir {}: {e}", parent.display()))
            })?;
        }
        std::fs::write(&path, bytes)
            .map_err(|e| CloudError::Storage(format!("write {}: {e}", path.display())))?;
        tracing::debug!(key = %key, path = %path.display(), bytes = bytes.len(), "local put");
        Ok(())
    }

    fn get(&self, key: &ArtifactKey) -> Result<Vec<u8>, CloudError> {
        let path = self.resolve(key)?;
        std::fs::read(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                CloudError::Storage(format!("not found: {key}"))
            } else {
                CloudError::Storage(format!("read {}: {e}", path.display()))
            }
        })
    }

    fn head(&self, key: &ArtifactKey) -> Result<Option<BlobMeta>, CloudError> {
        let path = self.resolve(key)?;
        match std::fs::metadata(&path) {
            Ok(meta) if meta.is_file() => Ok(Some(BlobMeta {
                size: meta.len(),
                // Local FS has no ETag — the vast.ai atomic-claim path
                // that relies on it is never exercised locally.
                etag: None,
            })),
            // A directory at the key path is "not a blob" → None.
            Ok(_) => Ok(None),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(CloudError::Storage(format!("head {}: {e}", path.display()))),
        }
    }

    fn list(&self, prefix: &str) -> Result<Vec<ArtifactKey>, CloudError> {
        let (_scheme, scheme_prefix, rest) = Self::split_scheme(prefix);
        let rel = rest.trim_start_matches('/');

        // The directory to walk + the key-prefix that every result must
        // start with (so `list("s3://b/run/c")` matches `c0`, `c1` files).
        let (walk_root, file_prefix) = {
            let rel_path = PathBuf::from(rel);
            for comp in rel_path.components() {
                if matches!(comp, Component::ParentDir) {
                    return Err(CloudError::Storage(format!(
                        "list prefix escapes base via `..`: {prefix}"
                    )));
                }
            }
            // Walk the parent dir of the last segment; the last segment
            // is a filename-prefix filter. A prefix ending in `/` walks
            // the whole dir (empty filename filter).
            if rel.is_empty() || rel.ends_with('/') {
                (self.base.join(rel), String::new())
            } else {
                let p = self.base.join(&rel_path);
                let parent = p
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| self.base.clone());
                let fname = p
                    .file_name()
                    .map(|f| f.to_string_lossy().into_owned())
                    .unwrap_or_default();
                (parent, fname)
            }
        };

        let mut out = Vec::new();
        self.walk(&walk_root, &file_prefix, scheme_prefix, &mut out)?;
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    fn delete(&self, key: &ArtifactKey) -> Result<(), CloudError> {
        let path = self.resolve(key)?;
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            // Idempotent delete: removing an absent key is success, matching
            // the object-store semantics the cloud impls present.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(CloudError::Storage(format!(
                "delete {}: {e}",
                path.display()
            ))),
        }
    }
}

impl LocalFsStorage {
    /// Recursively collect files under `dir` whose relative path (from
    /// the base) begins with `file_prefix` on its first matched segment,
    /// emitting keys in `scheme_prefix` scheme. The first path segment
    /// under the base is the bucket for the `s3://` scheme.
    fn walk(
        &self,
        dir: &Path,
        file_prefix: &str,
        scheme_prefix: &str,
        out: &mut Vec<ArtifactKey>,
    ) -> Result<(), CloudError> {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            // A prefix that names no existing dir lists empty (object
            // stores never error on an empty prefix).
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => {
                return Err(CloudError::Storage(format!("list {}: {e}", dir.display())));
            }
        };
        for entry in entries {
            let entry =
                entry.map_err(|e| CloudError::Storage(format!("list {}: {e}", dir.display())))?;
            let path = entry.path();
            let ftype = entry
                .file_type()
                .map_err(|e| CloudError::Storage(format!("file type {}: {e}", path.display())))?;
            if ftype.is_dir() {
                // The filename-prefix filter only applies to the *direct*
                // children of the walk root; once inside a matching dir we
                // recurse with an empty filter.
                let dir_name = entry.file_name();
                let dir_name = dir_name.to_string_lossy();
                if file_prefix.is_empty() || dir_name.starts_with(file_prefix) {
                    self.walk(&path, "", scheme_prefix, out)?;
                }
            } else if ftype.is_file() {
                let fname = entry.file_name();
                let fname = fname.to_string_lossy();
                if !file_prefix.is_empty() && !fname.starts_with(file_prefix) {
                    continue;
                }
                // Reconstruct the key: the path relative to the base, in
                // the caller's scheme.
                if let Ok(rel) = path.strip_prefix(&self.base) {
                    let rel_str = rel.to_string_lossy().replace('\\', "/");
                    out.push(ArtifactKey(format!("{scheme_prefix}{rel_str}")));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, LocalFsStorage) {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalFsStorage::new(dir.path());
        (dir, store)
    }

    #[test]
    fn s3_key_maps_to_bucket_subdir() {
        let (dir, s) = store();
        let key = ArtifactKey("s3://zentrain/run/omni/c0.parquet".into());
        s.put(&key, b"hello").unwrap();
        let on_disk = dir.path().join("zentrain/run/omni/c0.parquet");
        assert!(on_disk.is_file());
        assert_eq!(s.get(&key).unwrap(), b"hello");
    }

    #[test]
    fn r2_and_gs_aliases_share_layout() {
        let (dir, s) = store();
        s.put(&ArtifactKey("r2://b/x".into()), b"r").unwrap();
        s.put(&ArtifactKey("gs://b/y".into()), b"g").unwrap();
        assert!(dir.path().join("b/x").is_file());
        assert!(dir.path().join("b/y").is_file());
    }

    #[test]
    fn plain_path_key_maps_under_base() {
        let (dir, s) = store();
        s.put(&ArtifactKey("plain/file.bin".into()), b"p").unwrap();
        assert!(dir.path().join("plain/file.bin").is_file());
    }

    #[test]
    fn file_uri_relative_under_base_absolute_ignores_base() {
        let (dir, s) = store();
        s.put(&ArtifactKey("file://rel/r.bin".into()), b"rel")
            .unwrap();
        assert!(dir.path().join("rel/r.bin").is_file());

        let abs = dir.path().join("absdir").join("a.bin");
        let abs_key = ArtifactKey(format!("file://{}", abs.display()));
        s.put(&abs_key, b"abs").unwrap();
        assert!(abs.is_file());
        assert_eq!(s.get(&abs_key).unwrap(), b"abs");
    }

    #[test]
    fn head_reports_size_and_none_for_missing() {
        let (_dir, s) = store();
        let key = ArtifactKey("s3://b/h.bin".into());
        assert!(s.head(&key).unwrap().is_none());
        s.put(&key, b"12345").unwrap();
        let meta = s.head(&key).unwrap().expect("present");
        assert_eq!(meta.size, 5);
        assert!(meta.etag.is_none());
    }

    #[test]
    fn get_missing_is_not_found_error() {
        let (_dir, s) = store();
        let err = s.get(&ArtifactKey("s3://b/missing".into())).unwrap_err();
        assert!(matches!(err, CloudError::Storage(_)));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn delete_is_idempotent() {
        let (_dir, s) = store();
        let key = ArtifactKey("s3://b/d.bin".into());
        s.put(&key, b"x").unwrap();
        assert!(s.head(&key).unwrap().is_some());
        s.delete(&key).unwrap();
        assert!(s.head(&key).unwrap().is_none());
        // Deleting again is still Ok.
        s.delete(&key).unwrap();
    }

    #[test]
    fn list_returns_keys_in_caller_scheme() {
        let (_dir, s) = store();
        s.put(&ArtifactKey("s3://b/run/c0.parquet".into()), b"0")
            .unwrap();
        s.put(&ArtifactKey("s3://b/run/c1.parquet".into()), b"1")
            .unwrap();
        s.put(&ArtifactKey("s3://b/other/z.parquet".into()), b"z")
            .unwrap();

        // Trailing-slash prefix walks the whole dir.
        let mut keys: Vec<String> = s
            .list("s3://b/run/")
            .unwrap()
            .into_iter()
            .map(|k| k.0)
            .collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "s3://b/run/c0.parquet".to_string(),
                "s3://b/run/c1.parquet".to_string()
            ]
        );

        // Filename-prefix filter: `s3://b/run/c0` matches only c0.
        let keys: Vec<String> = s
            .list("s3://b/run/c0")
            .unwrap()
            .into_iter()
            .map(|k| k.0)
            .collect();
        assert_eq!(keys, vec!["s3://b/run/c0.parquet".to_string()]);
    }

    #[test]
    fn list_missing_prefix_is_empty_not_error() {
        let (_dir, s) = store();
        assert!(s.list("s3://nope/nothing/").unwrap().is_empty());
    }

    #[test]
    fn traversal_out_of_base_is_rejected() {
        let (_dir, s) = store();
        let err = s
            .put(&ArtifactKey("s3://b/../escape".into()), b"x")
            .unwrap_err();
        assert!(err.to_string().contains(".."));
        assert!(s.get(&ArtifactKey("file://../escape".into())).is_err());
    }
}
