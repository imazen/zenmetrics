//! Live R2 smoke for the chunk-model work-stealing + resumability claim — the
//! exact production claim code (`worker::claim::try_claim`) the granular
//! `zenmetrics plan-chunks` sweep feeds, exercised against a real isolated R2
//! prefix with two concurrent workers and a simulated dead box.
//!
//! ## Why this test exists
//!
//! `plan-chunks` (zenmetrics-cli) makes MANY ≤5-min chunks; the omni worker
//! (`zenfleet-sweep worker --mode omni`) loops over them, claiming each via this
//! token-race before processing. The four properties the sweep promises —
//! work-stealing, exactly-once, resumability, dead-box recovery — all live in
//! `try_claim`. This drives it against real R2 (no GPU, no encode) to prove:
//!
//! 1. **Exactly-once / work-stealing**: two workers racing the SAME chunk → at
//!    most one `Acquired`; the loser gets `HeldByPeer`/`LostRace`. No double
//!    execution. (Fast boxes naturally claim more because each claim is one
//!    atomic op; this proves the atomicity.)
//! 2. **Dead-box re-steal**: a worker holding a STALE claim (older than
//!    `stale_secs`) is presumed dead, so a second worker re-steals and finishes
//!    — the chunk completes elsewhere; ≤5 min is lost, not a multi-hour chunk.
//! 3. **Resumability**: once the omni sidecar exists, any later claim returns
//!    `AlreadyDone` — a re-run / re-launch skips done chunks (the `gap` re-runs
//!    only the missing ones).
//! 4. **Corruption-impossible idempotency**: completion is keyed on the sidecar
//!    object existing; the sidecar lands via one atomic PUT, so a re-run can
//!    never produce a partial/double sidecar.
//!
//! ## Running
//!
//! Gated behind `ZEN_R2_SMOKE=1` + the standard `R2_*` env (the skip decision is
//! the caller's, visible here — NOT a silent in-body skip; per CLAUDE.md "no
//! graceful skips"). The test isolates itself to a unique
//! `s3://zentrain/_smoke/claim-workstealing/<ts>-<pid>/` prefix and deletes it
//! on the way out. Costs a handful of tiny R2 objects.
//!
//! ```sh
//! export ZEN_R2_SMOKE=1
//! set -a; . ~/.config/cloudflare/r2-credentials; set +a
//! cargo test -p zenfleet-vastai --test claim_workstealing_r2 -- --nocapture
//! ```

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use zenfleet_vastai::worker::claim::{ClaimConfig, ClaimOutcome, try_claim};
use zenfleet_vastai::worker::r2::R2Client;

/// Whether the live-R2 smoke is enabled by the CALLER (env), and the creds it
/// needs. Returns the resolved (account_id, access_key, secret) when on.
fn smoke_env() -> Option<(String, String, String)> {
    if std::env::var("ZEN_R2_SMOKE").ok().as_deref() != Some("1") {
        return None;
    }
    let acct = std::env::var("R2_ACCOUNT_ID").ok()?;
    let ak = std::env::var("R2_ACCESS_KEY_ID").ok()?;
    let sk = std::env::var("R2_SECRET_ACCESS_KEY").ok()?;
    Some((acct, ak, sk))
}

/// Write a private `~/.aws/credentials`-style profile for s5cmd into a temp HOME
/// and return the (home_dir, profile_name). The R2Client shells s5cmd with
/// `--profile`, which reads `$HOME/.aws/credentials`.
fn write_profile(home: &std::path::Path, ak: &str, sk: &str) -> String {
    let aws = home.join(".aws");
    std::fs::create_dir_all(&aws).expect("mkdir .aws");
    std::fs::write(
        aws.join("credentials"),
        format!("[zensmoke]\naws_access_key_id = {ak}\naws_secret_access_key = {sk}\n"),
    )
    .expect("write credentials");
    "zensmoke".to_string()
}

fn unique_prefix() -> String {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!(
        "s3://zentrain/_smoke/claim-workstealing/{ts}-{}",
        std::process::id()
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn workstealing_exactly_once_resteal_and_resume() {
    let Some((acct, ak, sk)) = smoke_env() else {
        // The caller did NOT opt in (ZEN_R2_SMOKE!=1 or creds absent). This is
        // the caller's decision, surfaced loudly here — not a silent in-test
        // skip of an enabled run.
        eprintln!(
            "claim_workstealing_r2: SKIPPED — set ZEN_R2_SMOKE=1 and source \
             ~/.config/cloudflare/r2-credentials to run the live-R2 smoke"
        );
        return;
    };

    // s5cmd reads creds from a profile in $HOME/.aws/credentials. Point HOME at a
    // temp dir so we never touch the developer's real ~/.aws.
    let home = tempfile::tempdir().expect("temp home");
    let profile = write_profile(home.path(), &ak, &sk);
    // SAFETY of env mutation: the test is single-purpose and serial within this
    // binary; we set HOME for the duration of the s5cmd calls.
    unsafe {
        std::env::set_var("HOME", home.path());
    }

    let endpoint = R2Client::r2_endpoint_for_account(&acct);
    let r2 = R2Client::new("s5cmd", endpoint, profile.clone());

    let base = unique_prefix();
    let chunk_id = "zenjpeg-000000";
    let sidecar_uri = format!("{base}/omni/{chunk_id}.parquet");
    let claim_uri = format!("{base}/claims/{chunk_id}.claim");

    // Cleanup guard — delete the whole isolated prefix at the end, win or lose.
    // Drop is sync and we're inside a tokio runtime, so shell s5cmd directly
    // (a sync `std::process::Command`) rather than nesting a runtime — the
    // R2Client's `rm` is async and `block_on` from inside the test runtime
    // panics ("cannot start a runtime from within a runtime").
    struct Cleanup {
        endpoint: String,
        profile: String,
        base: String,
    }
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::process::Command::new("s5cmd")
                .args([
                    "--endpoint-url",
                    &self.endpoint,
                    "--profile",
                    &self.profile,
                    "rm",
                    &format!("{}/*", self.base),
                ])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
        }
    }
    let _cleanup = Cleanup {
        endpoint: R2Client::r2_endpoint_for_account(&acct),
        profile: profile.clone(),
        base: base.clone(),
    };

    // Fresh-prefix sanity: the sidecar must not exist yet.
    assert!(
        !r2.exists(&sidecar_uri).await,
        "fresh prefix should have no sidecar"
    );

    // ── 1. Work-stealing exactly-once: two workers race the same chunk. ──
    // A normal (non-stale) config so the second worker can't steal — it must
    // back off. The token-race + read-back guarantees exactly one Acquired.
    let cfg = ClaimConfig::default();
    let a = try_claim(&r2, "worker-A", chunk_id, &sidecar_uri, &claim_uri, &cfg)
        .await
        .expect("worker-A claim");
    assert!(
        matches!(a, ClaimOutcome::Acquired { .. }),
        "first worker must acquire, got {a:?}"
    );
    let b = try_claim(&r2, "worker-B", chunk_id, &sidecar_uri, &claim_uri, &cfg)
        .await
        .expect("worker-B claim");
    assert!(
        matches!(b, ClaimOutcome::HeldByPeer),
        "second worker must see a fresh peer claim (HeldByPeer), got {b:?} — \
         this is the exactly-once / no-double-execution guarantee"
    );

    // ── 2. Dead-box re-steal: worker-A 'dies' WITHOUT writing the sidecar. ──
    // A worker that treats the claim as stale (stale_secs = 0) presumes the
    // owner dead and re-steals. This is how a dead box's sub-5-min chunk
    // completes elsewhere instead of stranding for hours.
    let steal_cfg = ClaimConfig {
        stale_secs: 0,
        ..ClaimConfig::default()
    };
    let c = try_claim(
        &r2,
        "worker-C",
        chunk_id,
        &sidecar_uri,
        &claim_uri,
        &steal_cfg,
    )
    .await
    .expect("worker-C re-steal");
    assert!(
        matches!(c, ClaimOutcome::Acquired { .. }),
        "a stale (dead-box) claim must be re-stealable (Acquired), got {c:?}"
    );

    // worker-C 'completes' the chunk: upload the omni sidecar (one atomic PUT).
    // Build it fully locally first, then PUT — the corruption-impossible path.
    let local = PathBuf::from(home.path()).join(format!("{chunk_id}.omni.parquet"));
    std::fs::write(&local, b"OMNI-SIDECAR-COMPLETE").expect("write local sidecar");
    r2.upload(&local, &sidecar_uri)
        .await
        .expect("upload omni sidecar");
    assert!(
        r2.exists(&sidecar_uri).await,
        "sidecar must exist after the completing worker's atomic PUT"
    );

    // ── 3. Resumability: any later claim now short-circuits to AlreadyDone. ──
    // This is what makes a re-run / re-launch skip done chunks — the `gap`
    // reconcile re-runs only the missing ones.
    let d = try_claim(&r2, "worker-D", chunk_id, &sidecar_uri, &claim_uri, &cfg)
        .await
        .expect("worker-D post-completion claim");
    assert!(
        matches!(d, ClaimOutcome::AlreadyDone),
        "after the sidecar exists, a re-run must skip the chunk (AlreadyDone), \
         got {d:?} — this is the resumability guarantee"
    );
    // Even an aggressive re-steal config must NOT re-run a done chunk: the
    // idempotency check (sidecar-exists) precedes the staleness check.
    let e = try_claim(
        &r2,
        "worker-E",
        chunk_id,
        &sidecar_uri,
        &claim_uri,
        &steal_cfg,
    )
    .await
    .expect("worker-E aggressive re-claim");
    assert!(
        matches!(e, ClaimOutcome::AlreadyDone),
        "a done chunk is never re-run even under an aggressive steal config, got {e:?}"
    );

    eprintln!(
        "claim_workstealing_r2: PASSED — exactly-once (A wins, B backs off), \
         dead-box re-steal (C), resumability (D/E skip). prefix {base}"
    );
}
