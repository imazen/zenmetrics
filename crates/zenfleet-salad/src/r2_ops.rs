//! Operator-side R2 SigV4 client: HEAD / PUT / GET / LIST.
//!
//! Used by the launcher binary (`zenfleet-salad-sweep`) to read sidecars,
//! upload snapshots, and stitch the fleet_summary. Worker-side R2
//! access uses s5cmd with the scoped child credential — this module is
//! NEVER reachable on a worker.
//!
//! Why hand-rolled SigV4 rather than `aws-sdk-s3`: the SDK is heavy
//! (transitively pulls hyperion + h2 + tower + ...) and we only need
//! four call shapes. The implementation here is operator-side, so
//! correctness over completeness is acceptable.
//!
//! All four functions sign with the *parent* R2 cred (the operator
//! workstation's root key). Workers never see this key.

#![cfg(feature = "launcher")]

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use zenfleet_orchestrator::R2Operator;

use crate::launch::R2ParentCreds;

/// R2 operator wrapper. Holds the parent cred + a shared
/// `reqwest::Client` so connection pooling works across many polls.
pub struct R2OperatorImpl {
    parent: R2ParentCreds,
    http: reqwest::Client,
}

impl R2OperatorImpl {
    /// Construct with the operator's parent cred. The cred is cloned
    /// once and never logged in full.
    pub fn new(parent: R2ParentCreds) -> Self {
        Self {
            parent,
            http: reqwest::Client::new(),
        }
    }

    /// Borrow the parent cred (useful for pre-flight checks that
    /// reach outside the trait surface).
    pub fn parent(&self) -> &R2ParentCreds {
        &self.parent
    }

    fn endpoint(&self) -> String {
        format!(
            "https://{}.r2.cloudflarestorage.com",
            self.parent.account_id
        )
    }

    /// HEAD-check a single key. Public so the launcher's pre-flight
    /// can reuse it (it lives outside the `R2Operator` trait, which
    /// only carries the three call shapes the driver needs).
    pub async fn head(&self, bucket: &str, key: &str) -> Result<()> {
        let url = format!("{}/{bucket}/{key}", self.endpoint());
        let now = chrono_now();
        let auth = sigv4_auth_header(
            &self.parent,
            "HEAD",
            bucket,
            key,
            &url,
            &[("host", host_of(&url))],
            b"",
            "auto",
            "s3",
            &now,
        );
        let resp = self
            .http
            .head(&url)
            .header("Host", host_of(&url))
            .header("x-amz-content-sha256", empty_payload_hash())
            .header("x-amz-date", &now.amz)
            .header("Authorization", auth)
            .send()
            .await?;
        if !resp.status().is_success() {
            bail!("HEAD {url}: HTTP {}", resp.status());
        }
        Ok(())
    }

    /// HEAD-check an `s3://...` URI.
    pub async fn head_uri(&self, uri: &str) -> Result<()> {
        let (bucket, key) = split_s3_uri(uri)?;
        self.head(&bucket, &key).await
    }
}

impl R2Operator for R2OperatorImpl {
    async fn list(&self, bucket: &str, prefix: &str) -> Result<Vec<String>> {
        let endpoint = self.endpoint();
        let url = format!("{endpoint}/{bucket}/?list-type=2&prefix={prefix}");
        let now = chrono_now();
        let mut query: Vec<(String, String)> = vec![
            ("list-type".into(), "2".into()),
            ("prefix".into(), prefix.to_string()),
        ];
        query.sort();
        let auth = sigv4_auth_header_with_query(
            &self.parent,
            "GET",
            bucket,
            "",
            &endpoint,
            &[("host", host_of(&endpoint))],
            b"",
            "auto",
            "s3",
            &now,
            &query,
        );
        let resp = self
            .http
            .get(&url)
            .header("Host", host_of(&endpoint))
            .header("x-amz-content-sha256", empty_payload_hash())
            .header("x-amz-date", &now.amz)
            .header("Authorization", auth)
            .send()
            .await?;
        if !resp.status().is_success() {
            let s = resp.status();
            let b = resp.text().await.unwrap_or_default();
            bail!("LIST {url}: HTTP {s}: {b}");
        }
        let xml = resp.text().await?;
        let mut out = Vec::new();
        let mut rest = xml.as_str();
        while let Some(open) = rest.find("<Key>") {
            let after = &rest[open + 5..];
            if let Some(close) = after.find("</Key>") {
                out.push(after[..close].to_string());
                rest = &after[close..];
            } else {
                break;
            }
        }
        Ok(out)
    }

    async fn upload(&self, bucket: &str, key: &str, body: &[u8]) -> Result<()> {
        let url = format!("{}/{bucket}/{key}", self.endpoint());
        let now = chrono_now();
        let payload_hash = sha256_hex(body);
        let auth = sigv4_auth_header(
            &self.parent,
            "PUT",
            bucket,
            key,
            &url,
            &[("host", host_of(&url))],
            body,
            "auto",
            "s3",
            &now,
        );
        let resp = self
            .http
            .put(&url)
            .header("Host", host_of(&url))
            .header("x-amz-content-sha256", payload_hash)
            .header("x-amz-date", &now.amz)
            .header("Authorization", auth)
            .header("Content-Type", "application/octet-stream")
            .body(body.to_vec())
            .send()
            .await?;
        if !resp.status().is_success() {
            let s = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("PUT {url}: HTTP {s}: {body}");
        }
        Ok(())
    }

    async fn get_bytes(&self, bucket: &str, key: &str) -> Result<Vec<u8>> {
        let url = format!("{}/{bucket}/{key}", self.endpoint());
        let now = chrono_now();
        let auth = sigv4_auth_header(
            &self.parent,
            "GET",
            bucket,
            key,
            &url,
            &[("host", host_of(&url))],
            b"",
            "auto",
            "s3",
            &now,
        );
        let resp = self
            .http
            .get(&url)
            .header("Host", host_of(&url))
            .header("x-amz-content-sha256", empty_payload_hash())
            .header("x-amz-date", &now.amz)
            .header("Authorization", auth)
            .send()
            .await?;
        if !resp.status().is_success() {
            let s = resp.status();
            bail!("GET {url}: HTTP {s}");
        }
        Ok(resp.bytes().await?.to_vec())
    }
}

/// Split an `s3://bucket/key...` URI into `(bucket, key)`.
pub fn split_s3_uri(uri: &str) -> Result<(String, String)> {
    let rest = uri
        .strip_prefix("s3://")
        .with_context(|| format!("not an s3:// URI: {uri}"))?;
    let (bucket, key) = rest
        .split_once('/')
        .with_context(|| format!("URI missing key: {uri}"))?;
    Ok((bucket.to_string(), key.to_string()))
}

// ── Tiny SigV4 (just what we need: HEAD / PUT / GET / LIST) ──────────

struct AmzDate {
    amz: String,
    short: String,
}

fn chrono_now() -> AmzDate {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let (y, mo, d, h, mi, s) = epoch_to_ymdhms(secs);
    AmzDate {
        amz: format!("{y:04}{mo:02}{d:02}T{h:02}{mi:02}{s:02}Z"),
        short: format!("{y:04}{mo:02}{d:02}"),
    }
}

fn epoch_to_ymdhms(secs: u64) -> (i32, u32, u32, u32, u32, u32) {
    let days_total = (secs / 86_400) as i64;
    let secs_of_day = (secs % 86_400) as u32;
    let h = secs_of_day / 3600;
    let mi = (secs_of_day % 3600) / 60;
    let s = secs_of_day % 60;
    let z = days_total + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y_final = if m <= 2 { (y + 1) as i32 } else { y as i32 };
    (y_final, m as u32, d as u32, h, mi, s)
}

fn host_of(url: &str) -> String {
    let after_scheme = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    after_scheme.split('/').next().unwrap_or("").to_string()
}

fn empty_payload_hash() -> &'static str {
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
}

fn sha256_hex(b: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b);
    hex(&h.finalize())
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    let mut m = <Hmac<Sha256>>::new_from_slice(key).expect("hmac");
    m.update(data);
    m.finalize().into_bytes().to_vec()
}

#[allow(clippy::too_many_arguments)]
fn sigv4_auth_header(
    parent: &R2ParentCreds,
    method: &str,
    bucket: &str,
    key: &str,
    url: &str,
    headers: &[(&str, String)],
    body: &[u8],
    region: &str,
    service: &str,
    now: &AmzDate,
) -> String {
    sigv4_auth_header_with_query(
        parent,
        method,
        bucket,
        key,
        url,
        headers,
        body,
        region,
        service,
        now,
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn sigv4_auth_header_with_query(
    parent: &R2ParentCreds,
    method: &str,
    bucket: &str,
    key: &str,
    _url: &str,
    headers: &[(&str, String)],
    body: &[u8],
    region: &str,
    service: &str,
    now: &AmzDate,
    query: &[(String, String)],
) -> String {
    let canonical_uri = if key.is_empty() {
        format!("/{bucket}/")
    } else {
        format!("/{bucket}/{key}")
    };
    let canonical_query = query
        .iter()
        .map(|(k, v)| format!("{}={}", uri_encode(k, true), uri_encode(v, true)))
        .collect::<Vec<_>>()
        .join("&");
    let payload_hash = sha256_hex(body);
    let mut h = headers.to_vec();
    h.push(("x-amz-content-sha256", payload_hash.clone()));
    h.push(("x-amz-date", now.amz.clone()));
    h.sort_by(|a, b| a.0.cmp(b.0));
    let canonical_headers = h
        .iter()
        .map(|(k, v)| format!("{}:{}\n", k.to_ascii_lowercase(), v.trim()))
        .collect::<String>();
    let signed_headers = h
        .iter()
        .map(|(k, _)| k.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(";");
    let canonical_req = format!(
        "{method}\n{canonical_uri}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    );
    let cr_hash = sha256_hex(canonical_req.as_bytes());
    let credential_scope = format!("{}/{region}/{service}/aws4_request", now.short);
    let sts = format!(
        "AWS4-HMAC-SHA256\n{}\n{credential_scope}\n{cr_hash}",
        now.amz
    );
    let k_secret = format!("AWS4{}", parent.parent_secret_access_key);
    let k_date = hmac_sha256(k_secret.as_bytes(), now.short.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    let k_signing = hmac_sha256(&k_service, b"aws4_request");
    let sig = hex(&hmac_sha256(&k_signing, sts.as_bytes()));
    format!(
        "AWS4-HMAC-SHA256 Credential={}/{credential_scope}, SignedHeaders={signed_headers}, Signature={sig}",
        parent.parent_access_key_id
    )
}

fn uri_encode(s: &str, encode_slash: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.bytes() {
        let unreserved = c.is_ascii_alphanumeric()
            || c == b'-'
            || c == b'_'
            || c == b'.'
            || c == b'~'
            || (c == b'/' && !encode_slash);
        if unreserved {
            out.push(c as char);
        } else {
            out.push_str(&format!("%{c:02X}"));
        }
    }
    out
}

/// Format the current time as `YYYYMMDDTHHMMSSZ` for use in sweep ids.
pub fn short_ts() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let (y, mo, d, h, mi, s) = epoch_to_ymdhms(secs);
    format!("{y:04}{mo:02}{d:02}T{h:02}{mi:02}{s:02}")
}
