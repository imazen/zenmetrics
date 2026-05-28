//! Cloud-init `user_data` synthesis for Hetzner worker boxes.
//!
//! Hetzner has no managed job queue and no provider-supplied sidecar.
//! The cloud-init script is the entire worker-bootstrap surface:
//!
//! 1. Install docker (`apt-get install -y docker.io`).
//! 2. Login to ghcr.io if registry creds are provided.
//! 3. Pull the worker image.
//! 4. Run the worker container with `WORKER_BACKEND=hetzner` and the
//!    scoped R2 credentials in the env.
//!
//! The bake-everything mandate (zensim CLAUDE.md "BAKE EVERYTHING")
//! says workers should never apt-install at boot. We *partially* honour
//! that here: the docker image carries everything else. The docker
//! engine itself is the one apt install we cannot avoid on a vanilla
//! Ubuntu image — Hetzner doesn't ship a docker-pre-installed snapshot.
//! For this iter that's the right tradeoff vs building + maintaining
//! our own Hetzner base image. If apt stalls become a real cost, the
//! follow-up is to build a one-time Hetzner base image with docker
//! pre-baked and switch `image` from `ubuntu-24.04` to that snapshot.

use std::collections::BTreeMap;

/// Inputs needed to synthesize the cloud-init `user_data` script.
#[derive(Debug, Clone)]
pub struct WorkerBootstrap {
    /// Docker image (with tag) the worker runs.
    pub image: String,
    /// Sweep run id (also the R2 prefix scope).
    pub sweep_id: String,
    /// R2 account id (`<acct>.r2.cloudflarestorage.com`).
    pub r2_account_id: String,
    /// R2 working bucket (sweep sidecars + queue land here).
    pub r2_bucket: String,
    /// Scoped R2 access key id (per-sweep minted; NEVER the parent).
    pub r2_access_key_id: String,
    /// Scoped R2 secret access key.
    pub r2_secret_access_key: String,
    /// Scoped R2 session token (REQUIRED — temp keys 403 without it).
    pub r2_session_token: String,
    /// Optional ghcr.io / registry credentials. Empty = public image.
    pub registry_username: Option<String>,
    /// Optional ghcr.io / registry password (or PAT).
    pub registry_password: Option<String>,
    /// Optional ghcr.io / registry server (default `ghcr.io`).
    pub registry_server: Option<String>,
    /// Extra env vars to set on the container.
    pub extra_env: BTreeMap<String, String>,
    /// R2 chunks-queue prefix (`runs/<sweep_id>/queue/`). The worker
    /// LISTs this on a loop.
    pub chunks_queue_prefix: String,
    /// Optional SSH ed25519 public key to inject into the box's
    /// `/root/.ssh/authorized_keys`. When `Some`, the launcher can
    /// SSH in as `root` to pull `/var/log/cloud-init-output.log` +
    /// `/var/log/zen-bootstrap.log` + `docker logs worker` to
    /// disambiguate cloud-init stall / docker pull / R2 cred / worker
    /// crash failure modes. The key is appended to authorized_keys
    /// from the cloud-init script (which runs as root), so it works
    /// even when no provider-managed `ssh_keys` list is set.
    pub ssh_authorized_pubkey: Option<String>,
}

/// Synthesize a cloud-init script that boots the worker.
///
/// The script intentionally uses `#!/bin/bash` + `set -e` (loose mode
/// — we don't want a transient apt-update failure to kill the whole
/// boot; the docker pull retry handles steady-state recovery).
pub fn build_user_data(spec: &WorkerBootstrap) -> String {
    let mut script = String::with_capacity(4096);
    script.push_str("#!/bin/bash\n");
    script.push_str("# zencloud-hetzner worker bootstrap (cloud-init user_data)\n");
    script.push_str(&format!("# sweep_id={}\n", spec.sweep_id));
    script.push_str("set -eu\n");
    script.push_str("exec > >(tee -a /var/log/zen-bootstrap.log) 2>&1\n");
    script.push_str("echo \"[$(date -u +%Y-%m-%dT%H:%M:%SZ)] zen-bootstrap starting\"\n\n");

    // ── SSH pubkey inject (diagnostic surface) ───────────────────────
    // BEFORE the apt-install block — even if apt hangs we still have
    // SSH for log-pull. cloud-init runs as root, so we drop the key
    // straight into /root/.ssh/authorized_keys (Hetzner Cloud Ubuntu
    // images permit root login when an authorized_keys is present).
    if let Some(pubkey) = &spec.ssh_authorized_pubkey {
        // One trimmed line; reject anything that looks like an attempt
        // to inject extra commands (newlines / quotes).
        let cleaned = pubkey
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        if !cleaned.is_empty() && !cleaned.contains('\'') {
            script.push_str("# Diagnostic SSH key (launcher-side debugger).\n");
            script.push_str("mkdir -p /root/.ssh\n");
            script.push_str("chmod 700 /root/.ssh\n");
            script.push_str(&format!(
                "echo {key} >> /root/.ssh/authorized_keys\n",
                key = sh_squote(&cleaned),
            ));
            script.push_str("chmod 600 /root/.ssh/authorized_keys\n\n");
        }
    }

    // ── docker install ───────────────────────────────────────────────
    script.push_str("# Install docker (the one apt install we can't bake).\n");
    script.push_str("export DEBIAN_FRONTEND=noninteractive\n");
    script.push_str("apt-get update -qq\n");
    script.push_str("apt-get install -y -qq docker.io curl\n");
    script.push_str("systemctl enable --now docker\n\n");

    // ── registry login (optional) ────────────────────────────────────
    if let (Some(user), Some(pass)) = (&spec.registry_username, &spec.registry_password) {
        let server = spec
            .registry_server
            .clone()
            .unwrap_or_else(|| "ghcr.io".to_string());
        // `docker login --password-stdin` keeps the password out of `ps`.
        script.push_str("# Registry login (private image).\n");
        script.push_str(&format!(
            "printf '%s' {pass} | docker login {server} -u {user} --password-stdin\n\n",
            pass = sh_squote(pass),
            server = sh_squote(&server),
            user = sh_squote(user),
        ));
    } else {
        script.push_str("# (No registry login — assume image is public.)\n\n");
    }

    // ── pull image ───────────────────────────────────────────────────
    script.push_str("# Pull worker image (retry once on transient registry hiccup).\n");
    script.push_str(&format!(
        "docker pull {image} || (sleep 5 && docker pull {image})\n\n",
        image = sh_squote(&spec.image),
    ));

    // ── env file for the container ───────────────────────────────────
    script.push_str("# Write env file for the worker container (out of /proc/<pid>/cmdline).\n");
    script.push_str("mkdir -p /etc/zen\n");
    script.push_str("cat >/etc/zen/worker.env <<'ZEN_EOF'\n");
    push_env_line(
        &mut script,
        "SWEEP_RUN_ID",
        &spec.sweep_id,
    );
    push_env_line(&mut script, "WORKER_BACKEND", "hetzner");
    push_env_line(&mut script, "R2_ACCOUNT_ID", &spec.r2_account_id);
    push_env_line(&mut script, "R2_ACCESS_KEY_ID", &spec.r2_access_key_id);
    push_env_line(
        &mut script,
        "R2_SECRET_ACCESS_KEY",
        &spec.r2_secret_access_key,
    );
    push_env_line(&mut script, "AWS_SESSION_TOKEN", &spec.r2_session_token);
    push_env_line(&mut script, "R2_SESSION_TOKEN", &spec.r2_session_token);
    push_env_line(
        &mut script,
        "CHUNKS_QUEUE_PREFIX",
        &spec.chunks_queue_prefix,
    );
    push_env_line(
        &mut script,
        "CHUNKS_R2",
        &format!(
            "s3://{}/runs/{}/chunks.jsonl",
            spec.r2_bucket, spec.sweep_id
        ),
    );
    push_env_line(&mut script, "BUCKET", &spec.r2_bucket);
    push_env_line(&mut script, "RUST_LOG", "info,zencloud_hetzner=info");
    for (k, v) in &spec.extra_env {
        push_env_line(&mut script, k, v);
    }
    script.push_str("ZEN_EOF\n\n");

    // ── run container ────────────────────────────────────────────────
    // `--restart=on-failure:5` so transient docker failures retry; the
    // server itself is teardown-managed by the launcher.
    script.push_str("# Launch the worker.\n");
    let chunks_r2_uri = format!(
        "s3://{}/runs/{}/chunks.jsonl",
        spec.r2_bucket, spec.sweep_id
    );
    script.push_str(&format!(
        "docker run -d \\\n    --name=worker \\\n    --restart=on-failure:5 \\\n    --env-file=/etc/zen/worker.env \\\n    --hostname=zen-hetzner-$(hostname) \\\n    {image} \\\n    /usr/local/bin/zen-sweep-worker worker --backend hetzner \\\n        --run-id {sweep_id} \\\n        --chunks-r2 {chunks_r2}\n\n",
        image = sh_squote(&spec.image),
        sweep_id = sh_squote(&spec.sweep_id),
        chunks_r2 = sh_squote(&chunks_r2_uri),
    ));
    script.push_str("echo \"[$(date -u +%Y-%m-%dT%H:%M:%SZ)] zen-bootstrap done; worker container launched\"\n");
    script
}

/// Single-quote a shell argument so it survives `set -eu` substitution.
/// Hetzner cloud-init runs as `bash`, so `'...'` is a literal string;
/// embedded single quotes are escaped with `'\''`.
fn sh_squote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

/// Append a `KEY=VALUE` line to the env-file body. The value is NOT
/// quoted — docker's `--env-file` parser splits on the first `=` and
/// treats the rest of the line as the literal value (no shell parsing).
fn push_env_line(out: &mut String, k: &str, v: &str) {
    out.push_str(k);
    out.push('=');
    out.push_str(v);
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_spec() -> WorkerBootstrap {
        let mut extra = BTreeMap::new();
        extra.insert("METRICS".into(), "ssim2-gpu".into());
        WorkerBootstrap {
            image: "ghcr.io/imazen/zen-metrics-sweep-hetzner:v1".into(),
            sweep_id: "hetzner-iter1-2026-05-28".into(),
            r2_account_id: "abc123".into(),
            r2_bucket: "zen-tuning-ephemeral".into(),
            r2_access_key_id: "AKIA-temp".into(),
            r2_secret_access_key: "secret-temp".into(),
            r2_session_token: "session-token-xyz".into(),
            registry_username: None,
            registry_password: None,
            registry_server: None,
            extra_env: extra,
            chunks_queue_prefix: "runs/hetzner-iter1-2026-05-28/queue/".into(),
            ssh_authorized_pubkey: None,
        }
    }

    #[test]
    fn user_data_includes_critical_envs() {
        let s = build_user_data(&sample_spec());
        assert!(s.contains("SWEEP_RUN_ID=hetzner-iter1-2026-05-28"));
        assert!(s.contains("WORKER_BACKEND=hetzner"));
        assert!(s.contains("R2_ACCOUNT_ID=abc123"));
        assert!(s.contains("AWS_SESSION_TOKEN=session-token-xyz"));
        assert!(s.contains("CHUNKS_QUEUE_PREFIX=runs/hetzner-iter1-2026-05-28/queue/"));
        assert!(s.contains("METRICS=ssim2-gpu"));
        assert!(s.contains("apt-get install -y -qq docker.io"));
        assert!(s.contains("docker run -d"));
        assert!(s.contains("--backend hetzner"));
    }

    #[test]
    fn registry_login_skipped_when_no_creds() {
        let s = build_user_data(&sample_spec());
        assert!(s.contains("(No registry login"));
        assert!(!s.contains("docker login"));
    }

    #[test]
    fn registry_login_runs_when_creds_present() {
        let mut spec = sample_spec();
        spec.registry_username = Some("user".into());
        spec.registry_password = Some("pat-token".into());
        let s = build_user_data(&spec);
        assert!(s.contains("docker login 'ghcr.io' -u 'user' --password-stdin"));
        // Single-quoted to keep it out of `ps`.
        assert!(s.contains("printf '%s' 'pat-token'"));
    }

    #[test]
    fn ssh_pubkey_injected_into_authorized_keys_when_present() {
        let mut spec = sample_spec();
        spec.ssh_authorized_pubkey = Some(
            "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx zen-fleet"
                .into(),
        );
        let s = build_user_data(&spec);
        assert!(s.contains("mkdir -p /root/.ssh"));
        assert!(s.contains(">> /root/.ssh/authorized_keys"));
        assert!(s.contains("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5"));
        // Permission tightening lands too.
        assert!(s.contains("chmod 700 /root/.ssh"));
        assert!(s.contains("chmod 600 /root/.ssh/authorized_keys"));
    }

    #[test]
    fn ssh_pubkey_skipped_when_none() {
        let spec = sample_spec();
        let s = build_user_data(&spec);
        assert!(!s.contains("/root/.ssh/authorized_keys"));
    }

    #[test]
    fn ssh_pubkey_with_embedded_quote_is_rejected() {
        let mut spec = sample_spec();
        spec.ssh_authorized_pubkey =
            Some("ssh-ed25519 AAAAC3xxx 'comment'".into());
        let s = build_user_data(&spec);
        // The cleaned-line check drops the injection; nothing lands.
        assert!(!s.contains("/root/.ssh/authorized_keys"));
    }

    #[test]
    fn sh_squote_escapes_single_quotes() {
        assert_eq!(sh_squote("foo"), "'foo'");
        assert_eq!(sh_squote("a'b"), "'a'\\''b'");
        // Empty string is fine.
        assert_eq!(sh_squote(""), "''");
    }
}
