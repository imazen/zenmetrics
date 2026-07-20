//! In-process pooled S3/R2 client for the worker's blob + claim I/O.
//!
//! Replaces the per-blob `s5cmd cp`/`ls` and per-claim `aws s3api put-object` CLI
//! spawns. At fleet job rates the CLI startup (aws-cli ~1.5s Python; s5cmd ~50ms Go)
//! dominated the box CPU. `object_store` gives SigV4 auth + a persistent HTTP
//! connection pool; conditional PUT (`PutMode::Create`) is the exactly-once claim
//! (`If-None-Match: *`) natively. One client per (endpoint, bucket) is cached for the
//! process lifetime. Creds come from the ambient `AWS_*` env the launcher injects.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use object_store::{
    ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload,
    aws::{AmazonS3, AmazonS3Builder},
    path::Path as OsPath,
};

fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build tokio runtime for s3io")
    })
}

fn store(endpoint: &str, bucket: &str) -> Result<Arc<AmazonS3>, String> {
    static STORES: OnceLock<Mutex<HashMap<String, Arc<AmazonS3>>>> = OnceLock::new();
    let m = STORES.get_or_init(|| Mutex::new(HashMap::new()));
    let cache_key = format!("{endpoint}|{bucket}");
    {
        let g = m.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(s) = g.get(&cache_key) {
            return Ok(s.clone());
        }
    }
    let ak = std::env::var("AWS_ACCESS_KEY_ID").map_err(|_| "AWS_ACCESS_KEY_ID unset")?;
    let sk = std::env::var("AWS_SECRET_ACCESS_KEY").map_err(|_| "AWS_SECRET_ACCESS_KEY unset")?;
    let mut b = AmazonS3Builder::new()
        .with_endpoint(endpoint)
        .with_bucket_name(bucket)
        .with_access_key_id(&ak)
        .with_secret_access_key(&sk)
        .with_region("auto")
        .with_virtual_hosted_style_request(false);
    if let Ok(tok) = std::env::var("AWS_SESSION_TOKEN") {
        if !tok.is_empty() {
            b = b.with_token(tok);
        }
    }
    let s: Arc<AmazonS3> = Arc::new(b.build().map_err(|e| format!("s3io build: {e}"))?);
    let mut g = m.lock().unwrap_or_else(|p| p.into_inner());
    Ok(g.entry(cache_key).or_insert(s).clone())
}

/// Upload (overwrite) an object.
pub fn put(endpoint: &str, bucket: &str, key: &str, bytes: &[u8]) -> Result<(), String> {
    let s = store(endpoint, bucket)?;
    let p = OsPath::from(key);
    let payload = PutPayload::from_bytes(bytes::Bytes::copy_from_slice(bytes));
    runtime()
        .block_on(async move { s.put(&p, payload).await })
        .map(|_| ())
        .map_err(|e| format!("s3io put {bucket}/{key}: {e}"))
}

/// Conditional create (`If-None-Match: *`) — the exactly-once claim. Returns
/// `Ok(true)` iff THIS caller created the object, `Ok(false)` if it already existed.
pub fn put_create(endpoint: &str, bucket: &str, key: &str, bytes: &[u8]) -> Result<bool, String> {
    let s = store(endpoint, bucket)?;
    let p = OsPath::from(key);
    let payload = PutPayload::from_bytes(bytes::Bytes::copy_from_slice(bytes));
    let opts = PutOptions {
        mode: PutMode::Create,
        ..Default::default()
    };
    match runtime().block_on(async move { s.put_opts(&p, payload, opts).await }) {
        Ok(_) => Ok(true),
        Err(object_store::Error::AlreadyExists { .. }) => Ok(false),
        Err(e) => Err(format!("s3io put_create {bucket}/{key}: {e}")),
    }
}

/// True iff the object exists (HEAD).
pub fn head_exists(endpoint: &str, bucket: &str, key: &str) -> bool {
    let Ok(s) = store(endpoint, bucket) else {
        return false;
    };
    let p = OsPath::from(key);
    runtime().block_on(async move { s.head(&p).await }).is_ok()
}

/// Delete an object (idempotent — NotFound is OK).
pub fn delete(endpoint: &str, bucket: &str, key: &str) -> Result<(), String> {
    let s = store(endpoint, bucket)?;
    let p = OsPath::from(key);
    match runtime().block_on(async move { s.delete(&p).await }) {
        Ok(_) | Err(object_store::Error::NotFound { .. }) => Ok(()),
        Err(e) => Err(format!("s3io delete {bucket}/{key}: {e}")),
    }
}
