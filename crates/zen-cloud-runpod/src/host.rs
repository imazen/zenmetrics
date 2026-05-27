//! RunPod `CredentialSource` + `WorkerHost` — plain container env.
//!
//! Unlike vast.ai (which hides credentials on `/proc/1/environ`) or
//! SaladCloud (which uses a link-local IMDS), RunPod injects everything
//! the worker needs as ordinary environment variables in the container's
//! own process environment (spec §1.10): the `env` map from the
//! pod-create request, plus RunPod's reserved identity vars. So a plain
//! [`std::env::var`] read is all that is needed — no IMDS HTTP, no pid-1
//! trick.
//!
//! ## Reserved RunPod env vars
//!
//! RunPod sets `RUNPOD_POD_ID` (this pod's id) on every pod, and
//! commonly `RUNPOD_API_KEY` / `RUNPOD_POD_HOSTNAME` / `RUNPOD_GPU_COUNT`
//! depending on the template. We treat `RUNPOD_POD_ID` as the canonical
//! worker identity (the claim-token namespace), falling back to
//! `$WORKER_ID` then the hostname when running off a RunPod node.

use std::path::PathBuf;

use zen_cloud_core::{CloudError, CredentialSource, Credentials, WorkerHost, WorkerId};

/// Reserved RunPod env var: this pod's id (set on every pod).
pub const ENV_POD_ID: &str = "RUNPOD_POD_ID";
/// Reserved RunPod env var: RunPod-provided GPU count (when set by the
/// template). Used as a hint before falling back to `nvidia-smi`.
pub const ENV_GPU_COUNT: &str = "RUNPOD_GPU_COUNT";
/// RunPod-provided public hostname (when networking is enabled).
pub const ENV_POD_HOSTNAME: &str = "RUNPOD_POD_HOSTNAME";

/// `CredentialSource` reading the RunPod container environment.
///
/// RunPod injects the `env` map from the pod-create request directly
/// into the app process, so this reads the same BYO object-store creds
/// the vast.ai path expects (`R2_*` / `AWS_*` / `S5CMD_*`) plus the
/// sweep wiring and RunPod identity vars.
#[derive(Default)]
pub struct RunpodEnvCredentials;

impl CredentialSource for RunpodEnvCredentials {
    fn resolve(&self) -> Result<Credentials, CloudError> {
        let mut out = Credentials::new();
        let keys = [
            // BYO R2 / S3 storage creds (set on the pod via the
            // pod-create `env` map; same names the vast.ai path expects).
            "R2_ACCOUNT_ID",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_ENDPOINT",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "S5CMD_PROFILE",
            // Sweep wiring.
            "SWEEP_RUN_ID",
            "CHUNKS_R2",
            "WORKER_ID",
            // RunPod-reserved identity vars.
            ENV_POD_ID,
            ENV_POD_HOSTNAME,
            ENV_GPU_COUNT,
        ];
        for k in keys {
            if let Ok(v) = std::env::var(k) {
                out.insert(k.to_string(), v);
            }
        }
        Ok(out)
    }
}

/// `WorkerHost` over RunPod's pod identity.
///
/// The worker id is `RUNPOD_POD_ID` (falling back to `$WORKER_ID` then
/// the hostname); the scratch dir defaults to `/workspace` (RunPod's
/// conventional persistent-volume mount, also the pod-create default
/// `volumeMountPath`); GPU count prefers `RUNPOD_GPU_COUNT` and falls
/// back to an `nvidia-smi` line count, matching the vast.ai host probe.
pub struct RunpodWorkerHost {
    worker_id: WorkerId,
    scratch: PathBuf,
}

impl RunpodWorkerHost {
    /// Build from the environment: `RUNPOD_POD_ID` → `$WORKER_ID` →
    /// hostname for the id; `$WORKDIR` → `/workspace` for scratch.
    pub fn from_env() -> Self {
        let worker_id = std::env::var(ENV_POD_ID)
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| std::env::var("WORKER_ID").ok().filter(|s| !s.is_empty()))
            .or_else(hostname)
            .unwrap_or_else(|| "runpod-worker".to_string());
        let scratch = std::env::var("WORKDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/workspace"));
        Self {
            worker_id: WorkerId(worker_id),
            scratch,
        }
    }

    /// Explicit constructor (tests / non-RunPod hosts).
    pub fn new(worker_id: impl Into<String>, scratch: impl Into<PathBuf>) -> Self {
        Self {
            worker_id: WorkerId(worker_id.into()),
            scratch: scratch.into(),
        }
    }
}

impl WorkerHost for RunpodWorkerHost {
    fn worker_id(&self) -> WorkerId {
        self.worker_id.clone()
    }

    fn scratch_dir(&self) -> PathBuf {
        self.scratch.clone()
    }

    fn gpu_count(&self) -> usize {
        // Prefer RunPod's own count when the template exposes it.
        if let Ok(n) = std::env::var(ENV_GPU_COUNT)
            && let Ok(n) = n.trim().parse::<usize>()
            && n > 0
        {
            return n;
        }
        // Fall back to the nvidia-smi line count (one line per GPU),
        // identical to the vast.ai + salad host probes. Absent
        // nvidia-smi (dev box) → 0.
        match std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            .output()
        {
            Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter(|l| !l.trim().is_empty())
                .count(),
            _ => 0,
        }
    }
}

/// Read the system hostname via `HOSTNAME` (set in most container
/// runtimes) falling back to the `hostname` command.
fn hostname() -> Option<String> {
    if let Ok(h) = std::env::var("HOSTNAME")
        && !h.is_empty()
    {
        return Some(h);
    }
    std::process::Command::new("hostname")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_host_explicit() {
        let h = RunpodWorkerHost::new("pod-abc", "/workspace");
        assert_eq!(h.worker_id().as_str(), "pod-abc");
        assert_eq!(h.scratch_dir(), PathBuf::from("/workspace"));
        // gpu_count is env-dependent; must not panic.
        let _ = h.gpu_count();
    }

    #[test]
    fn credentials_resolve_collects_only_present_env() {
        // `resolve()` reads the live process env; in the test harness the
        // RunPod-specific keys are normally absent, so the map only ever
        // contains keys that genuinely exist (it never invents nulls).
        // Mutating the process env from a test would be both `unsafe`
        // (this crate is `#![forbid(unsafe_code)]`) and racy under the
        // parallel test runner, so we assert the no-invented-keys
        // invariant against whatever the ambient env happens to be.
        let creds = RunpodEnvCredentials.resolve().unwrap();
        for k in creds.keys() {
            assert!(
                std::env::var(k).is_ok(),
                "resolve() returned key {k:?} that is not actually set in the env"
            );
        }
    }

    #[test]
    fn pod_id_constants_are_runpod_reserved_names() {
        assert_eq!(ENV_POD_ID, "RUNPOD_POD_ID");
        assert_eq!(ENV_GPU_COUNT, "RUNPOD_GPU_COUNT");
        assert_eq!(ENV_POD_HOSTNAME, "RUNPOD_POD_HOSTNAME");
    }
}
