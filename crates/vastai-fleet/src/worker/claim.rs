//! Token-race claim primitive. Ports the proven
//! `onstart_cvvdp_backfill_imazen.sh` claim pattern to Rust.
//!
//! ## Why a custom race instead of S3 conditional puts
//!
//! Cloudflare R2 added conditional puts (`If-None-Match: *`) in late
//! 2024, but s5cmd's older versions in the docker image don't expose
//! the flag. Sticking with the token-race lets the same code run
//! against multiple R2 client tools and ages of s5cmd. When phase C
//! drops to native aws-sdk-s3 we can switch to conditional puts and
//! drop the read-back delay.
//!
//! ## Algorithm
//!
//! 1. **Pre-flight idempotency**: if the sidecar URL exists, return
//!    `Skip` — someone already finished this chunk.
//! 2. **Read existing claim**: if it's <`stale_secs` old AND owned by
//!    a different worker, return `Skip` — fresh claim by a peer.
//!    Otherwise we overwrite.
//! 3. **Write our claim**: file body = `<token>\t<epoch>\t<worker>`
//!    where `token = <worker>-<pid>-<nanos>` is process-unique. The
//!    PID + nanos disambiguate concurrent dispatchers on the same
//!    box (currently impossible, but cheap insurance).
//! 4. **Sleep `read_back_delay`**: lets near-simultaneous overlapping
//!    writes settle on the R2 side.
//! 5. **Read back**: if our token survived, we own it. Otherwise
//!    return `LostRace` — another worker wrote AFTER us.
//!
//! Tunables (with operationally-validated defaults from the bash
//! version) live on [`ClaimConfig`].

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use tokio::time::sleep;
use tracing::debug;

use super::r2::R2Client;

#[derive(Clone, Debug)]
pub struct ClaimConfig {
    /// How long a claim is considered "alive" before another worker
    /// is allowed to steal it. Default 600s — matches the bash
    /// onstart's CLAIM_STALE_SEC. Tuned so a worker that's mid-chunk
    /// (typical 4-5 min) doesn't get its claim stolen.
    pub stale_secs: u64,
    /// Delay between writing the claim and reading it back. Default
    /// 1.5s — covers R2's typical write-to-read propagation lag.
    /// Smaller values risk false-positive "I own it" verdicts when
    /// another worker's write hasn't propagated yet.
    pub read_back_delay: Duration,
}

impl Default for ClaimConfig {
    fn default() -> Self {
        Self {
            stale_secs: 600,
            read_back_delay: Duration::from_millis(1500),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimOutcome {
    /// We won the race; safe to start processing.
    Acquired { token: String },
    /// Sidecar already exists. No work to do.
    AlreadyDone,
    /// Another worker holds a fresh (<stale_secs) claim. Try the
    /// next chunk; this one is in flight.
    HeldByPeer,
    /// We wrote our claim but another worker's write landed after
    /// ours and overwrote it. The other worker now owns the chunk.
    LostRace,
    /// s5cmd / R2 error during the race. Caller can retry next iter.
    Errored,
}

/// Attempt to claim `chunk_id` for processing.
///
/// `worker_id` distinguishes peers; `sidecar_uri` is the R2 URL of
/// the would-be output (used for idempotency). `claim_uri` is the
/// R2 path to the claim file itself.
pub async fn try_claim(
    r2: &R2Client,
    worker_id: &str,
    chunk_id: &str,
    sidecar_uri: &str,
    claim_uri: &str,
    cfg: &ClaimConfig,
) -> Result<ClaimOutcome> {
    // 1. Idempotency: did someone already produce the sidecar?
    if r2.exists(sidecar_uri).await {
        debug!(chunk_id, "sidecar exists; skip");
        return Ok(ClaimOutcome::AlreadyDone);
    }

    // 2. Existing claim probe.
    let existing = r2.cat_string(claim_uri).await;
    let now = epoch_secs();
    if !existing.is_empty() {
        if let Some((_token, ep, owner)) = parse_claim(&existing) {
            let age = now.saturating_sub(ep);
            if age < cfg.stale_secs && owner != worker_id {
                debug!(chunk_id, age, owner, "held by peer");
                return Ok(ClaimOutcome::HeldByPeer);
            }
            // Stale or self-owned -> fall through and overwrite.
        }
    }

    // 3. Write our claim.
    let token = generate_token(worker_id);
    let body = format!("{token}\t{now}\t{worker_id}");
    let tmp = std::env::temp_dir().join(format!("claim-{chunk_id}.txt"));
    if let Err(e) = tokio::fs::write(&tmp, &body).await {
        tracing::warn!(error = %e, chunk_id, "tmp claim write failed");
        return Ok(ClaimOutcome::Errored);
    }
    if r2.upload(&tmp, claim_uri).await.is_err() {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Ok(ClaimOutcome::Errored);
    }
    let _ = tokio::fs::remove_file(&tmp).await;

    // 4. Read-back settle delay.
    sleep(cfg.read_back_delay).await;

    // 5. Verify.
    let verify = r2.cat_string(claim_uri).await;
    let Some((verified_token, _, _)) = parse_claim(&verify) else {
        // File disappeared or unreadable — treat as lost.
        return Ok(ClaimOutcome::LostRace);
    };
    if verified_token == token {
        Ok(ClaimOutcome::Acquired { token })
    } else {
        Ok(ClaimOutcome::LostRace)
    }
}

/// Generate a unique-per-process-per-instant token: `worker-pid-nanos`.
/// Matches the bash version exactly so a Rust worker can interoperate
/// with bash-onstart workers on the same fleet during the rollout.
fn generate_token(worker_id: &str) -> String {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{worker_id}-{pid}-{nanos}")
}

/// Parse a claim file body. Format: `<token>\t<epoch>\t<worker>`.
/// Returns None on any parse error so the caller can treat the claim
/// as "unparseable, overwrite".
fn parse_claim(s: &str) -> Option<(String, u64, String)> {
    let mut parts = s.splitn(3, '\t');
    let token = parts.next()?.to_string();
    let epoch: u64 = parts.next()?.parse().ok()?;
    let owner = parts.next()?.to_string();
    if token.is_empty() || owner.is_empty() {
        return None;
    }
    Some((token, epoch, owner))
}

fn epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_claim_roundtrip() {
        let body = "tok-A-1234-9876543210\t1779173814\tcvvdp-w2";
        let (t, e, w) = parse_claim(body).unwrap();
        assert_eq!(t, "tok-A-1234-9876543210");
        assert_eq!(e, 1779173814);
        assert_eq!(w, "cvvdp-w2");
    }

    #[test]
    fn parse_claim_rejects_empty_fields() {
        assert!(parse_claim("").is_none());
        assert!(parse_claim("\t\t").is_none());
        assert!(parse_claim("tok\t").is_none());
        assert!(parse_claim("tok\tnotnumeric\tw").is_none());
    }

    #[test]
    fn parse_claim_rejects_short() {
        assert!(parse_claim("tok").is_none());
        assert!(parse_claim("tok\t1234").is_none());
    }

    #[test]
    fn token_uniqueness() {
        let a = generate_token("w1");
        // generate_token reads nanos with ns precision — even back-
        // to-back calls should differ on any modern system clock.
        let b = generate_token("w1");
        assert_ne!(a, b, "consecutive tokens MUST differ");
    }
}
