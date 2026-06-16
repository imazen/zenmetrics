//! Provider-agnostic minting of **scoped, auto-expiring Cloudflare R2
//! temporary access credentials**.
//!
//! A sweep launcher (Salad, RunPod, vast.ai — anywhere a container runs
//! on hardware the operator does not own) must NOT inject the root R2
//! key into a remote-fleet container. Instead it mints a credential
//! scoped to one bucket (optionally to prefixes), one permission, and a
//! short TTL, then injects that into the container-group / pod env. A
//! compromised consumer node's blast radius is then limited to that one
//! bucket's objects instead of the whole R2 account.
//!
//! This module is provider-agnostic on purpose — `zenfleet-salad`,
//! `zenfleet-runpod`, and `zenfleet-vastai` all call
//! [`mint_scoped_r2_cred`].
//!
//! ## API spec + gotchas (VERIFIED)
//!
//! Source of truth: `~/work/claudehints/topics/r2-credentials.md`
//! (Cloudflare R2 "temporary access credentials", verified live).
//!
//! ```text
//! POST https://api.cloudflare.com/client/v4/accounts/{account_id}/r2/temp-access-credentials
//! Authorization: Bearer <CF_API_TOKEN>
//! Content-Type: application/json
//! body: { bucket, parentAccessKeyId, parentSecretAccessKey,
//!         permission, ttlSeconds, prefixes? }
//! → result.{ accessKeyId, secretAccessKey, sessionToken }
//! ```
//!
//! - **GOTCHA — the path is account-level, NOT under the bucket.** The
//!   bucket goes in the *body*. The
//!   `.../r2/buckets/{bucket}/temp-access-credentials` form returns
//!   error **10015 "No route matches this url."** The correct path is
//!   `.../r2/temp-access-credentials`.
//! - **GOTCHA — the returned `sessionToken` (~550 chars) MUST reach the
//!   S3 client as `AWS_SESSION_TOKEN` / `aws_session_token`** (or
//!   `X-Amz-Security-Token`). Access-key + secret ALONE will 403. Any
//!   consumer that writes `~/.aws/credentials` must also write
//!   `aws_session_token = <sessionToken>` when one is present.
//! - `ttlSeconds` range is `900..=604800` (15 min .. 7 days).

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::CloudError;

/// Cloudflare API base — the account-level temp-creds endpoint hangs off
/// `{base}/accounts/{account_id}/r2/temp-access-credentials`.
pub const CF_API_BASE: &str = "https://api.cloudflare.com/client/v4";

/// Default TTL for a per-sweep worker credential (6 hours). Generous
/// enough that most sweeps finish inside one credential; long sweeps
/// that outlast the TTL need a re-mint (a documented follow-on).
pub const DEFAULT_TTL_SECONDS: u64 = 21_600;

/// Cloudflare's TTL floor for temp creds (15 minutes).
pub const MIN_TTL_SECONDS: u64 = 900;

/// Cloudflare's TTL ceiling for temp creds (7 days).
pub const MAX_TTL_SECONDS: u64 = 604_800;

/// The permission a minted credential carries. Serializes to the exact
/// Cloudflare wire strings.
///
/// Workers that read + write artifacts use [`Permission::ObjectReadWrite`].
/// The `admin-*` variants additionally allow bucket-level operations and
/// are not appropriate for an untrusted fleet node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Permission {
    /// `object-read-write` — object get/put/delete. Use this for workers.
    #[serde(rename = "object-read-write")]
    ObjectReadWrite,
    /// `object-read-only` — object get/list only.
    #[serde(rename = "object-read-only")]
    ObjectReadOnly,
    /// `admin-read-write` — object ops + bucket-level ops.
    #[serde(rename = "admin-read-write")]
    AdminReadWrite,
    /// `admin-read-only` — read object + bucket-level metadata.
    #[serde(rename = "admin-read-only")]
    AdminReadOnly,
}

impl Permission {
    /// The exact Cloudflare wire string for this permission.
    pub fn as_wire_str(self) -> &'static str {
        match self {
            Permission::ObjectReadWrite => "object-read-write",
            Permission::ObjectReadOnly => "object-read-only",
            Permission::AdminReadWrite => "admin-read-write",
            Permission::AdminReadOnly => "admin-read-only",
        }
    }
}

/// A minted scoped R2 credential. The three S3 fields go into the worker
/// env (`R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`);
/// `expires_at` is the wall-clock Unix-epoch second the credential stops
/// working, for the launcher's own re-mint bookkeeping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopedR2Cred {
    /// S3 access key id (goes to env as `R2_ACCESS_KEY_ID`).
    pub access_key_id: String,
    /// S3 secret (goes to env as `R2_SECRET_ACCESS_KEY`).
    pub secret_access_key: String,
    /// The session token — REQUIRED by the S3 client as
    /// `AWS_SESSION_TOKEN` / `aws_session_token`. Without it the S3 calls
    /// 403 even with a valid key+secret (see module gotchas).
    pub session_token: String,
    /// Unix-epoch second the credential expires (mint time + TTL). `None`
    /// if the launcher can't determine the local clock (best-effort).
    pub expires_at: Option<u64>,
}

/// The JSON request body for the temp-creds endpoint. Field names are the
/// exact camelCase Cloudflare expects.
#[derive(Debug, Serialize)]
struct TempCredsRequest<'a> {
    bucket: &'a str,
    #[serde(rename = "parentAccessKeyId")]
    parent_access_key_id: &'a str,
    #[serde(rename = "parentSecretAccessKey")]
    parent_secret_access_key: &'a str,
    permission: Permission,
    #[serde(rename = "ttlSeconds")]
    ttl_seconds: u64,
    /// Omitted entirely when empty so the account default (whole bucket)
    /// applies — an empty `prefixes: []` is NOT the same as absent.
    #[serde(skip_serializing_if = "<[String]>::is_empty")]
    prefixes: &'a [String],
}

/// The CF envelope wrapping every `client/v4` response.
#[derive(Debug, Deserialize)]
struct CfEnvelope<T> {
    success: bool,
    #[serde(default)]
    errors: Vec<CfError>,
    // `Option` is already treated as optional by serde (missing => None);
    // no `#[serde(default)]` so we don't force `T: Default`.
    result: Option<T>,
}

#[derive(Debug, Deserialize)]
struct CfError {
    #[serde(default)]
    code: i64,
    #[serde(default)]
    message: String,
}

/// The `result` block of a successful temp-creds response.
#[derive(Debug, Deserialize)]
struct TempCredsResult {
    #[serde(rename = "accessKeyId")]
    access_key_id: String,
    #[serde(rename = "secretAccessKey")]
    secret_access_key: String,
    #[serde(rename = "sessionToken")]
    session_token: String,
}

/// Clamp a requested TTL into Cloudflare's `[MIN, MAX]` range.
fn clamp_ttl(ttl_seconds: u64) -> u64 {
    ttl_seconds.clamp(MIN_TTL_SECONDS, MAX_TTL_SECONDS)
}

/// Build the account-level temp-creds endpoint URL. Account-level path —
/// the bucket goes in the body, NOT the URL (the bucket-scoped path
/// returns CF error 10015 "No route matches this url").
fn temp_creds_url(base: &str, account_id: &str) -> String {
    format!("{base}/accounts/{account_id}/r2/temp-access-credentials")
}

/// Mint a scoped, auto-expiring R2 temporary credential.
///
/// Hits the verified account-level Cloudflare endpoint:
/// `POST {CF_API_BASE}/accounts/{account_id}/r2/temp-access-credentials`
/// with `Authorization: Bearer {cf_api_token}`.
///
/// - `cf_api_token` — the Cloudflare REST API bearer token with R2
///   temp-cred-mint permission (operator-box secret; NEVER injected into
///   a worker).
/// - `parent_access_key_id` / `parent_secret_access_key` — the root R2
///   **S3** key+secret that scopes the child credential.
/// - `bucket` — the single bucket the credential is scoped to. R2 temp
///   creds are single-bucket; a sweep that reads bucket A and writes
///   bucket B needs a second cred or presigned source URLs.
/// - `prefixes` — optional tighter prefix scope (e.g.
///   `["runs/<SWEEP_ID>/"]`); empty means the whole bucket.
/// - `permission` — workers use [`Permission::ObjectReadWrite`].
/// - `ttl_seconds` — clamped into `[900, 604800]`.
///
/// `ttl_seconds` is clamped rather than rejected so callers can pass a
/// generous default without a fallible pre-check.
///
/// The argument count mirrors the Cloudflare request fields exactly; a
/// builder struct here would just shuffle the same fields one level up.
#[allow(clippy::too_many_arguments)]
pub async fn mint_scoped_r2_cred(
    cf_api_token: &str,
    account_id: &str,
    parent_access_key_id: &str,
    parent_secret_access_key: &str,
    bucket: &str,
    prefixes: &[String],
    permission: Permission,
    ttl_seconds: u64,
) -> Result<ScopedR2Cred, CloudError> {
    let http = reqwest::Client::builder()
        .build()
        .map_err(|e| CloudError::Credentials(format!("build reqwest client: {e}")))?;
    mint_scoped_r2_cred_with(
        &http,
        CF_API_BASE,
        cf_api_token,
        account_id,
        parent_access_key_id,
        parent_secret_access_key,
        bucket,
        prefixes,
        permission,
        ttl_seconds,
    )
    .await
}

/// Like [`mint_scoped_r2_cred`] but with a caller-supplied `reqwest::Client`
/// and API base — used by tests (mock server) and by callers that want to
/// reuse a client. Most callers want [`mint_scoped_r2_cred`].
#[allow(clippy::too_many_arguments)]
pub async fn mint_scoped_r2_cred_with(
    http: &reqwest::Client,
    api_base: &str,
    cf_api_token: &str,
    account_id: &str,
    parent_access_key_id: &str,
    parent_secret_access_key: &str,
    bucket: &str,
    prefixes: &[String],
    permission: Permission,
    ttl_seconds: u64,
) -> Result<ScopedR2Cred, CloudError> {
    let ttl = clamp_ttl(ttl_seconds);
    let body = TempCredsRequest {
        bucket,
        parent_access_key_id,
        parent_secret_access_key,
        permission,
        ttl_seconds: ttl,
        prefixes,
    };
    let url = temp_creds_url(api_base, account_id);

    let resp = http
        .post(&url)
        .bearer_auth(cf_api_token)
        .json(&body)
        .send()
        .await
        .map_err(|e| CloudError::Credentials(format!("POST {url}: {e}")))?;

    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(CloudError::Credentials(format!(
            "mint scoped R2 cred for bucket {bucket:?}: HTTP {status}: {text}"
        )));
    }

    let env: CfEnvelope<TempCredsResult> = serde_json::from_str(&text)
        .map_err(|e| CloudError::Credentials(format!("decode CF temp-creds response: {e}")))?;
    if !env.success {
        let detail = env
            .errors
            .iter()
            .map(|e| format!("[{}] {}", e.code, e.message))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(CloudError::Credentials(format!(
            "CF temp-creds for bucket {bucket:?} returned success=false: {detail}"
        )));
    }
    let result = env.result.ok_or_else(|| {
        CloudError::Credentials("CF temp-creds response had success=true but no result".into())
    })?;

    let expires_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_secs() + ttl);

    Ok(ScopedR2Cred {
        access_key_id: result.access_key_id,
        secret_access_key: result.secret_access_key,
        session_token: result.session_token,
        expires_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_serializes_to_exact_cf_strings() {
        assert_eq!(
            serde_json::to_value(Permission::ObjectReadWrite).unwrap(),
            "object-read-write"
        );
        assert_eq!(
            serde_json::to_value(Permission::ObjectReadOnly).unwrap(),
            "object-read-only"
        );
        assert_eq!(
            serde_json::to_value(Permission::AdminReadWrite).unwrap(),
            "admin-read-write"
        );
        assert_eq!(
            serde_json::to_value(Permission::AdminReadOnly).unwrap(),
            "admin-read-only"
        );
        // And the explicit accessor agrees.
        assert_eq!(
            Permission::ObjectReadWrite.as_wire_str(),
            "object-read-write"
        );
    }

    #[test]
    fn request_body_uses_camelcase_and_omits_empty_prefixes() {
        let no_prefixes: &[String] = &[];
        let body = TempCredsRequest {
            bucket: "zen-tuning-ephemeral",
            parent_access_key_id: "PARENT_KEY",
            parent_secret_access_key: "PARENT_SECRET",
            permission: Permission::ObjectReadWrite,
            ttl_seconds: 21_600,
            prefixes: no_prefixes,
        };
        let v = serde_json::to_value(&body).unwrap();
        assert_eq!(v["bucket"], "zen-tuning-ephemeral");
        assert_eq!(v["parentAccessKeyId"], "PARENT_KEY");
        assert_eq!(v["parentSecretAccessKey"], "PARENT_SECRET");
        assert_eq!(v["permission"], "object-read-write");
        assert_eq!(v["ttlSeconds"], 21_600);
        // Empty prefixes are omitted entirely (absent != []).
        assert!(v.get("prefixes").is_none());
    }

    #[test]
    fn request_body_includes_nonempty_prefixes() {
        let prefixes = vec!["runs/sweep-abc/".to_string(), "logs/".to_string()];
        let body = TempCredsRequest {
            bucket: "b",
            parent_access_key_id: "k",
            parent_secret_access_key: "s",
            permission: Permission::ObjectReadOnly,
            ttl_seconds: 900,
            prefixes: &prefixes,
        };
        let v = serde_json::to_value(&body).unwrap();
        assert_eq!(v["prefixes"][0], "runs/sweep-abc/");
        assert_eq!(v["prefixes"][1], "logs/");
    }

    #[test]
    fn ttl_is_clamped_to_cf_range() {
        assert_eq!(clamp_ttl(0), MIN_TTL_SECONDS);
        assert_eq!(clamp_ttl(100), MIN_TTL_SECONDS);
        assert_eq!(clamp_ttl(21_600), 21_600);
        assert_eq!(clamp_ttl(u64::MAX), MAX_TTL_SECONDS);
        assert_eq!(clamp_ttl(MAX_TTL_SECONDS + 1), MAX_TTL_SECONDS);
    }

    #[test]
    fn url_is_account_level_not_bucket_scoped() {
        // The bucket-scoped path returns CF error 10015; the correct path
        // is account-level with the bucket in the body.
        let url = temp_creds_url(CF_API_BASE, "ACCT123");
        assert_eq!(
            url,
            "https://api.cloudflare.com/client/v4/accounts/ACCT123/r2/temp-access-credentials"
        );
        assert!(!url.contains("/buckets/"));
    }

    #[test]
    fn success_envelope_decodes_to_scoped_cred() {
        // Construct the CF response struct (do NOT make live calls).
        let json = r#"{
            "success": true,
            "errors": [],
            "messages": [],
            "result": {
                "accessKeyId": "CHILD_AK",
                "secretAccessKey": "CHILD_SK",
                "sessionToken": "SESSION_TOKEN_550_CHARS"
            }
        }"#;
        let env: CfEnvelope<TempCredsResult> = serde_json::from_str(json).unwrap();
        assert!(env.success);
        let r = env.result.unwrap();
        assert_eq!(r.access_key_id, "CHILD_AK");
        assert_eq!(r.secret_access_key, "CHILD_SK");
        assert_eq!(r.session_token, "SESSION_TOKEN_550_CHARS");
    }

    #[test]
    fn failure_envelope_surfaces_code_10015() {
        // The well-known wrong-path error so a future caller recognizes it.
        let json = r#"{
            "success": false,
            "errors": [{"code": 10015, "message": "No route matches this url"}],
            "messages": [],
            "result": null
        }"#;
        let env: CfEnvelope<TempCredsResult> = serde_json::from_str(json).unwrap();
        assert!(!env.success);
        assert_eq!(env.errors[0].code, 10015);
        assert!(env.errors[0].message.contains("No route"));
        assert!(env.result.is_none());
    }
}
