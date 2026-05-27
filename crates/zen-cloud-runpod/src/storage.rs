//! RunPod `BlobStorage` — the shared S3-compatible store, BYO.
//!
//! RunPod has no native object store, so a RunPod pod brings its own
//! R2/S3 bucket — exactly the storage the vast.ai backend uses. Per spec
//! §1.10 (mirroring §1.9 item 4) we do NOT write a second S3 client:
//! this module is a thin constructor around the shared
//! [`zen_cloud_s3::S3BlobStorage`] (re-exported here as
//! [`RunpodBlobStorage`]) wiring it from the BYO credentials RunPod
//! injected into the pod env.

use zen_cloud_core::CloudError;
use zen_cloud_s3::{S3BlobStorage, S3Client};

/// `BlobStorage` for the RunPod backend — the shared S3-compatible impl.
pub type RunpodBlobStorage = S3BlobStorage;

/// Build a [`RunpodBlobStorage`] from explicit binary/endpoint/profile.
pub fn blob_storage(
    s5cmd_bin: impl Into<String>,
    endpoint: impl Into<String>,
    profile: impl Into<String>,
) -> RunpodBlobStorage {
    S3BlobStorage::new(S3Client::new(s5cmd_bin, endpoint, profile))
}

/// Build a [`RunpodBlobStorage`] from the resolved pod-env credentials
/// map (the output of [`crate::host::RunpodEnvCredentials::resolve`]).
///
/// Prefers an explicit `R2_ENDPOINT`; otherwise derives the R2 endpoint
/// from `R2_ACCOUNT_ID`. The s5cmd binary defaults to `s5cmd` on PATH
/// and the profile to `r2` — the same convention the rest of the fleet
/// uses.
pub fn blob_storage_from_credentials(
    creds: &std::collections::HashMap<String, String>,
) -> Result<RunpodBlobStorage, CloudError> {
    let endpoint = if let Some(ep) = creds.get("R2_ENDPOINT").filter(|s| !s.is_empty()) {
        ep.clone()
    } else if let Some(acct) = creds.get("R2_ACCOUNT_ID").filter(|s| !s.is_empty()) {
        S3Client::r2_endpoint_for_account(acct)
    } else {
        return Err(CloudError::Credentials(
            "neither R2_ENDPOINT nor R2_ACCOUNT_ID present in pod env".into(),
        ));
    };
    let profile = creds
        .get("S5CMD_PROFILE")
        .filter(|s| !s.is_empty())
        .cloned()
        .unwrap_or_else(|| "r2".to_string());
    Ok(blob_storage("s5cmd", endpoint, profile))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn from_credentials_uses_explicit_endpoint() {
        let mut c = HashMap::new();
        c.insert(
            "R2_ENDPOINT".to_string(),
            "https://explicit.example.com".to_string(),
        );
        let s = blob_storage_from_credentials(&c).expect("build");
        assert_eq!(s.client().endpoint, "https://explicit.example.com");
    }

    #[test]
    fn from_credentials_derives_r2_endpoint_from_account() {
        let mut c = HashMap::new();
        c.insert("R2_ACCOUNT_ID".to_string(), "acct123".to_string());
        let s = blob_storage_from_credentials(&c).expect("build");
        assert_eq!(
            s.client().endpoint,
            "https://acct123.r2.cloudflarestorage.com"
        );
    }

    #[test]
    fn from_credentials_errors_without_endpoint_info() {
        let c = HashMap::new();
        assert!(blob_storage_from_credentials(&c).is_err());
    }

    #[test]
    fn from_credentials_honors_profile() {
        let mut c = HashMap::new();
        c.insert(
            "R2_ENDPOINT".to_string(),
            "https://e.example.com".to_string(),
        );
        c.insert("S5CMD_PROFILE".to_string(), "myprofile".to_string());
        let s = blob_storage_from_credentials(&c).expect("build");
        assert_eq!(s.client().profile, "myprofile");
    }
}
