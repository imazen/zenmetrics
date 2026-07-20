#![cfg(feature = "jobexec")]
//! In-process pooled S3/R2 object client for the jobexec fetch path.
//!
//! Replaces the per-variant `aws s3api get-object` CLI spawn (the aws-cli's ~1.5s
//! Python/boto3 startup, ×12 variants/chunk, was the dominant fleet cost — cores
//! idle in process spawn). `object_store` gives SigV4 auth + HTTP connection
//! pooling; a small tokio runtime drives it from the otherwise-sync executor. One
//! client per bucket is cached for the process lifetime, so a persistent (`--serve`)
//! executor reuses the same connection pool across every job on the box.

use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, Mutex, OnceLock};

use object_store::{
    GetOptions, ObjectStore, ObjectStoreExt, aws::AmazonS3Builder, path::Path as OsPath,
};

fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build tokio runtime for objstore")
    })
}

/// Build (once) + cache a pooled R2 client for `bucket`. Credentials + endpoint
/// come from the ambient env the fleet injects (`ZEN_R2_ENDPOINT`, `AWS_*`).
fn store_for(bucket: &str) -> Result<Arc<dyn ObjectStore>, Box<dyn Error>> {
    static STORES: OnceLock<Mutex<HashMap<String, Arc<dyn ObjectStore>>>> = OnceLock::new();
    let m = STORES.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let g = m.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(s) = g.get(bucket) {
            return Ok(s.clone());
        }
    }
    let endpoint =
        std::env::var("ZEN_R2_ENDPOINT").map_err(|_| "ZEN_R2_ENDPOINT unset (objstore)")?;
    let ak = std::env::var("AWS_ACCESS_KEY_ID").map_err(|_| "AWS_ACCESS_KEY_ID unset")?;
    let sk = std::env::var("AWS_SECRET_ACCESS_KEY").map_err(|_| "AWS_SECRET_ACCESS_KEY unset")?;
    let mut b = AmazonS3Builder::new()
        .with_endpoint(&endpoint)
        .with_bucket_name(bucket)
        .with_access_key_id(&ak)
        .with_secret_access_key(&sk)
        .with_region("auto")
        // R2 serves path-style (endpoint/bucket/key), not virtual-hosted.
        .with_virtual_hosted_style_request(false);
    if let Ok(tok) = std::env::var("AWS_SESSION_TOKEN") {
        if !tok.is_empty() {
            b = b.with_token(tok);
        }
    }
    let store: Arc<dyn ObjectStore> = Arc::new(b.build().map_err(|e| format!("objstore build: {e}"))?);
    let mut g = m.lock().unwrap_or_else(|p| p.into_inner());
    let s = g.entry(bucket.to_string()).or_insert(store).clone();
    Ok(s)
}

/// GET a whole object (`bucket`, `key`) into memory over the pooled connection.
pub fn get_object(bucket: &str, key: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let store = store_for(bucket)?;
    let p = OsPath::from(key);
    let bytes = runtime()
        .block_on(async move {
            let r = store.get_opts(&p, GetOptions::default()).await?;
            r.bytes().await
        })
        .map_err(|e| format!("objstore get {bucket}/{key}: {e}"))?;
    Ok(bytes.to_vec())
}

/// GET an object addressed by a full `s3://bucket/key` URI.
pub fn get_uri(uri: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let rest = uri
        .strip_prefix("s3://")
        .ok_or_else(|| format!("not an s3:// uri: {uri}"))?;
    let (bucket, key) = rest
        .split_once('/')
        .ok_or_else(|| format!("s3 uri has no key: {uri}"))?;
    get_object(bucket, key)
}

/// GET a single byte range `[offset, offset+len)` of an object (one R2 request, pooled connection).
/// Replaces the per-variant `aws s3api get-object --range` spawn on the byte-range tar path.
pub fn get_range(bucket: &str, key: &str, offset: u64, len: u64) -> Result<Vec<u8>, Box<dyn Error>> {
    let store = store_for(bucket)?;
    let p = OsPath::from(key);
    let range = offset..offset + len;
    let bytes = runtime()
        .block_on(async move { store.get_range(&p, range).await })
        .map_err(|e| format!("objstore get_range {bucket}/{key} [{offset}+{len}]: {e}"))?;
    Ok(bytes.to_vec())
}

/// GET MANY byte ranges of ONE object CONCURRENTLY — all in flight together over the pooled
/// connection, not one-at-a-time. `ranges` is `(offset, len)` pairs; the returned `Vec` is in the
/// same order. This is the concurrency win for a ScoreFile cell: its ~12 variants are all ranges of
/// the same `variants.tar`, so one call overlaps every fetch instead of 12 serial round-trips.
pub fn get_ranges(
    bucket: &str,
    key: &str,
    ranges: &[(u64, u64)],
) -> Result<Vec<Vec<u8>>, Box<dyn Error>> {
    if ranges.is_empty() {
        return Ok(Vec::new());
    }
    let store = store_for(bucket)?;
    let p = OsPath::from(key);
    let rs: Vec<std::ops::Range<u64>> = ranges.iter().map(|&(o, l)| o..o + l).collect();
    let out = runtime()
        .block_on(async move { store.get_ranges(&p, &rs).await })
        .map_err(|e| format!("objstore get_ranges {bucket}/{key} ({} ranges): {e}", ranges.len()))?;
    Ok(out.into_iter().map(|b| b.to_vec()).collect())
}
