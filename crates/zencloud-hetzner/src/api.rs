//! Hetzner Cloud REST client — only the four endpoints `ProviderHandle`
//! needs: `GET /server_types`, `POST /servers`,
//! `GET /servers?label_selector=...`, `DELETE /servers/{id}`.
//!
//! Hetzner billing notes (verified 2026-05-28 via API probe):
//! - **CAX21**: 4 ARM cores (Ampere), 8 GB, 80 GB disk, €0.0152/hr (~€9.49/mo)
//! - **CCX13**: 2 dedicated AMD cores, 8 GB, 80 GB disk, €0.032/hr (~€19.99/mo)
//! - **CCX23**: 4 dedicated AMD cores, 16 GB, 160 GB disk, €0.0641/hr
//!
//! Hetzner charges per hour with a 1-hour minimum and continues billing
//! until the server is DELETEd. Workers must DELETE on teardown.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Hetzner Cloud API base URL.
const HETZNER_API_BASE: &str = "https://api.hetzner.cloud/v1";

/// Hetzner Cloud REST client.
///
/// Holds the bearer token + a shared `reqwest::Client` so connection
/// pooling works across many polls. Token is `Bearer <token>`.
#[derive(Clone)]
pub struct HetznerApi {
    token: String,
    http: reqwest::Client,
}

impl HetznerApi {
    /// Construct with a bearer token.
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            http: reqwest::Client::new(),
        }
    }

    /// `GET /server_types` — list available server types. Pages through
    /// up to 50 results in one call (Hetzner's max page size).
    ///
    /// Filter `cpu_type` ("dedicated" / "shared") for the CCX / CAX
    /// AMD-dedicated lines vs the CPX / CX shared-vCPU lines.
    pub async fn list_server_types(&self) -> Result<Vec<HetznerServerType>> {
        let url = format!("{HETZNER_API_BASE}/server_types?per_page=50");
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.token)
            .send()
            .await
            .context("GET /server_types")?;
        if !resp.status().is_success() {
            let s = resp.status();
            let b = resp.text().await.unwrap_or_default();
            bail!("GET /server_types: HTTP {s}: {b}");
        }
        let v: serde_json::Value = resp.json().await.context("parse /server_types JSON")?;
        let arr = v
            .get("server_types")
            .and_then(|x| x.as_array())
            .context("missing `server_types` array")?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            let st: HetznerServerType =
                serde_json::from_value(item.clone()).context("parse one server_type entry")?;
            out.push(st);
        }
        Ok(out)
    }

    /// `POST /servers` — create a server. Returns the created server's
    /// id + initial status. The server starts with status `initializing`,
    /// then `starting`, then `running` once cloud-init finishes booting.
    ///
    /// `user_data` is the cloud-init script the server boots into;
    /// `labels` is the `{group: <sweep_id>}` map for label-selector
    /// scoping; `ssh_keys` is the list of SSH key names/ids on the
    /// project (empty = no inbound SSH access — fine for our workers).
    #[allow(clippy::too_many_arguments)]
    pub async fn create_server(
        &self,
        name: &str,
        server_type: &str,
        image: &str,
        location: &str,
        user_data: &str,
        ssh_keys: &[String],
        labels: &HashMap<String, String>,
    ) -> Result<HetznerServer> {
        let url = format!("{HETZNER_API_BASE}/servers");
        let body = serde_json::json!({
            "name": name,
            "server_type": server_type,
            "image": image,
            "location": location,
            "user_data": user_data,
            "ssh_keys": ssh_keys,
            "labels": labels,
            "start_after_create": true,
        });
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.token)
            .json(&body)
            .send()
            .await
            .context("POST /servers")?;
        if !resp.status().is_success() {
            let s = resp.status();
            let b = resp.text().await.unwrap_or_default();
            bail!("POST /servers ({name}): HTTP {s}: {b}");
        }
        let v: serde_json::Value = resp.json().await.context("parse /servers response")?;
        let srv = v.get("server").context("missing `server` field")?;
        serde_json::from_value(srv.clone()).context("parse server entry")
    }

    /// `GET /servers?label_selector=group=<group_name>` — list all
    /// servers carrying the label. Used to enumerate the group for
    /// poll + teardown.
    pub async fn list_servers_by_label(&self, label_selector: &str) -> Result<Vec<HetznerServer>> {
        let url = format!(
            "{HETZNER_API_BASE}/servers?label_selector={}&per_page=50",
            urlencode(label_selector)
        );
        let resp = self
            .http
            .get(&url)
            .bearer_auth(&self.token)
            .send()
            .await
            .context("GET /servers (label_selector)")?;
        if !resp.status().is_success() {
            let s = resp.status();
            let b = resp.text().await.unwrap_or_default();
            bail!("GET /servers?label_selector={label_selector}: HTTP {s}: {b}");
        }
        let v: serde_json::Value = resp.json().await.context("parse /servers JSON")?;
        let arr = v
            .get("servers")
            .and_then(|x| x.as_array())
            .context("missing `servers` array")?;
        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            let s: HetznerServer =
                serde_json::from_value(item.clone()).context("parse one server entry")?;
            out.push(s);
        }
        Ok(out)
    }

    /// `DELETE /servers/{id}` — destroy a server. Idempotent: a 404
    /// (already deleted) is treated as success.
    pub async fn delete_server(&self, id: i64) -> Result<()> {
        let url = format!("{HETZNER_API_BASE}/servers/{id}");
        let resp = self
            .http
            .delete(&url)
            .bearer_auth(&self.token)
            .send()
            .await
            .with_context(|| format!("DELETE /servers/{id}"))?;
        if resp.status().is_success() || resp.status() == 404 {
            Ok(())
        } else {
            let s = resp.status();
            let b = resp.text().await.unwrap_or_default();
            bail!("DELETE /servers/{id}: HTTP {s}: {b}");
        }
    }
}

/// A Hetzner server type entry (returned from `GET /server_types`).
///
/// Only the fields we use are deserialized; the others are ignored
/// (Hetzner adds new fields over time and we want forward-compat).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HetznerServerType {
    /// Internal id (rarely used; `name` is the user-facing identifier).
    pub id: i64,
    /// Slug like "cax21" / "ccx13" / "cpx11". This is what
    /// `POST /servers.server_type` accepts.
    pub name: String,
    /// "shared" or "dedicated" — per Hetzner's own taxonomy. Note:
    /// CAX is listed as `shared` even though it's on dedicated ARM
    /// hardware; the truly-dedicated AMD lines are CCX.
    pub cpu_type: String,
    /// "x86" or "arm".
    pub architecture: String,
    /// Core count.
    pub cores: u32,
    /// Memory in GiB.
    pub memory: f64,
    /// Disk size in GB.
    pub disk: u32,
}

/// A Hetzner server entry (returned from `POST /servers` or
/// `GET /servers`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HetznerServer {
    /// Internal numeric id (load-bearing for `DELETE /servers/{id}`).
    pub id: i64,
    /// User-supplied name (set at create-time).
    pub name: String,
    /// Server lifecycle status. See [`HetznerServerStatus`] for the enum.
    pub status: String,
    /// Labels carried by the server (we set `group=<sweep_id>` at create).
    #[serde(default)]
    pub labels: HashMap<String, String>,
    /// Public IPv4 metadata (the address + dns_ptr live nested here).
    #[serde(default)]
    pub public_net: serde_json::Value,
}

impl HetznerServer {
    /// Public IPv4 address (if Hetzner assigned one — every CAX/CCX
    /// gets one by default).
    pub fn ipv4(&self) -> Option<String> {
        self.public_net
            .get("ipv4")
            .and_then(|v| v.get("ip"))
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
    }

    /// Parse the status string into the typed enum, defaulting to
    /// `Unknown` for forward-compat with Hetzner adding new statuses.
    pub fn parsed_status(&self) -> HetznerServerStatus {
        HetznerServerStatus::from_str(&self.status)
    }
}

/// Hetzner server status enum.
///
/// Source: Hetzner Cloud API docs `GET /servers/{id}.status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HetznerServerStatus {
    /// Just-created; cloud-init not yet started.
    Initializing,
    /// Booting; cloud-init may be running.
    Starting,
    /// Fully booted and running. Cloud-init script is typically still
    /// in flight here (docker pull + container start), but the VM is up.
    Running,
    /// Being stopped.
    Stopping,
    /// Stopped (still billable until DELETEd).
    Off,
    /// Being deleted.
    Deleting,
    /// Anything else Hetzner returns.
    Unknown,
}

impl HetznerServerStatus {
    /// Parse from the Hetzner JSON status string. Unknown values map
    /// to `Unknown` so we never blow up on forward-compat changes.
    // Intentionally an infallible inherent `from_str(&str) -> Self` (unknown →
    // Unknown), not the fallible `FromStr` trait (which returns Result).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "initializing" => Self::Initializing,
            "starting" => Self::Starting,
            "running" => Self::Running,
            "stopping" => Self::Stopping,
            "off" => Self::Off,
            "deleting" => Self::Deleting,
            _ => Self::Unknown,
        }
    }

    /// Map onto the orchestrator's `InstanceStatus.state` string.
    ///
    /// initializing/starting → "allocating"
    /// running               → "running"
    /// stopping/off/deleting → "stopping"
    pub fn as_orchestrator_state(&self) -> &'static str {
        match self {
            Self::Initializing | Self::Starting => "allocating",
            Self::Running => "running",
            Self::Stopping | Self::Off | Self::Deleting => "stopping",
            Self::Unknown => "unknown",
        }
    }
}

/// Hetzner location slug. Verified live 2026-05-28:
/// `fsn1` Falkenstein DE, `nbg1` Nuremberg DE, `hel1` Helsinki FI,
/// `ash` Ashburn VA US, `hil` Hillsboro OR US, `sin` Singapore SG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HetznerLocation {
    /// Falkenstein, Germany.
    Fsn1,
    /// Nuremberg, Germany.
    Nbg1,
    /// Helsinki, Finland.
    Hel1,
    /// Ashburn, VA, US.
    Ash,
    /// Hillsboro, OR, US.
    Hil,
    /// Singapore, SG.
    Sin,
}

impl HetznerLocation {
    /// Canonical Hetzner location slug.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fsn1 => "fsn1",
            Self::Nbg1 => "nbg1",
            Self::Hel1 => "hel1",
            Self::Ash => "ash",
            Self::Hil => "hil",
            Self::Sin => "sin",
        }
    }
}

/// Load Hetzner API token from `~/.config/hetzner/credentials` or the
/// `$HETZNER_API_TOKEN` env var.
///
/// File format: lines `key=value`, comments start with `#`. The token
/// key is `api_token=...` (matching the file the user provisioned).
pub fn load_token_from_file_or_env() -> Result<String> {
    if let Ok(t) = std::env::var("HETZNER_API_TOKEN")
        && !t.trim().is_empty()
    {
        return Ok(t.trim().to_string());
    }
    let home = std::env::var("HOME").context("HOME not set")?;
    let path = PathBuf::from(home).join(".config/hetzner/credentials");
    let contents = std::fs::read_to_string(&path)
        .with_context(|| format!("read {} (or set $HETZNER_API_TOKEN)", path.display()))?;
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(v) = line.strip_prefix("api_token=") {
            let v = v.trim();
            if !v.is_empty() {
                return Ok(v.to_string());
            }
        }
    }
    bail!(
        "no `api_token=...` line found in {} (or set $HETZNER_API_TOKEN)",
        path.display()
    )
}

/// Minimal URL-encoder for label_selector values. The `=` separator
/// inside the value (`group=<sweep_id>`) MUST survive, so we don't
/// percent-encode `=`; we only escape spaces (rare in our group names)
/// and any other reserved char Hetzner would reject.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.bytes() {
        match c {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'=' => {
                out.push(c as char);
            }
            _ => {
                out.push_str(&format!("%{c:02X}"));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_round_trip() {
        for (s, expect) in [
            ("initializing", HetznerServerStatus::Initializing),
            ("starting", HetznerServerStatus::Starting),
            ("running", HetznerServerStatus::Running),
            ("stopping", HetznerServerStatus::Stopping),
            ("off", HetznerServerStatus::Off),
            ("deleting", HetznerServerStatus::Deleting),
            ("garbage", HetznerServerStatus::Unknown),
        ] {
            assert_eq!(HetznerServerStatus::from_str(s), expect, "for {s:?}");
        }
    }

    #[test]
    fn status_to_orchestrator_state() {
        assert_eq!(
            HetznerServerStatus::Initializing.as_orchestrator_state(),
            "allocating"
        );
        assert_eq!(
            HetznerServerStatus::Running.as_orchestrator_state(),
            "running"
        );
        assert_eq!(HetznerServerStatus::Off.as_orchestrator_state(), "stopping");
        assert_eq!(
            HetznerServerStatus::Unknown.as_orchestrator_state(),
            "unknown"
        );
    }

    #[test]
    fn urlencode_preserves_label_selector_equals() {
        assert_eq!(urlencode("group=sweep-1"), "group=sweep-1");
        assert_eq!(urlencode("a b"), "a%20b");
    }

    #[test]
    fn location_slugs() {
        assert_eq!(HetznerLocation::Fsn1.as_str(), "fsn1");
        assert_eq!(HetznerLocation::Ash.as_str(), "ash");
    }

    #[test]
    fn server_ipv4_extracted_from_public_net() {
        let s = HetznerServer {
            id: 1,
            name: "n".into(),
            status: "running".into(),
            labels: HashMap::new(),
            public_net: serde_json::json!({
                "ipv4": { "ip": "203.0.113.10", "dns_ptr": "x" },
                "ipv6": { "ip": "2001:db8::1" }
            }),
        };
        assert_eq!(s.ipv4().as_deref(), Some("203.0.113.10"));
    }
}
