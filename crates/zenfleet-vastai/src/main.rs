//! `zenfleet-vastai` — robust replacement for the bash+python destroyer +
//! status scripts in `scripts/sweep/`.
//!
//! Solves three operational pain points from the 2026-05-17 / 2026-05-18
//! cvvdp / iwssim / ssim2 backfill sessions:
//!
//! 1. **Destroyer crashes on JSON parse errors.** The ssim2 destroyer
//!    hit `json.decoder.JSONDecodeError: Expecting value: line 1 column 1`
//!    once the sidecar count crossed the target, leaving 15 boxes
//!    running for 1.5 hr after the work was done. This binary uses the
//!    tolerant parser in [`parse`] which never aborts on malformed
//!    rows — it warns and skips.
//! 2. **No one-shot status check.** Operators needed three commands +
//!    a jq invocation to answer "how many of my workers are still
//!    running and what's the burn rate?". This binary's `status`
//!    subcommand does it in one call.
//! 3. **No watch loop.** Auto-destroy logic lived in `bash + python`
//!    heredocs that combined into one fragile script per sweep.
//!    `zenfleet-vastai watch` does the same thing in 50 LOC of Rust,
//!    with proper error handling and a clean exit on target-hit or
//!    wall-cap.
//!
//! Scope is deliberately small: shell out to `vastai`, parse, decide.
//! No vast.ai API client; no SDK. The destroyer that crashed exited
//! before destroying anything; this one is structured so that if it
//! crashes, the partial state is recoverable from `vastai show
//! instances-v1` directly.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};

use zenfleet_vastai::parse::{
    Instance, filter_by_label, parse_instances, status_breakdown, total_dph,
};
#[cfg(feature = "worker")]
use zenfleet_vastai::worker;

/// CLI for managing vast.ai fleets of zenmetrics backfill workers.
#[derive(Parser, Debug)]
#[command(
    name = "zenfleet-vastai",
    version,
    about = "Robust vast.ai fleet status / destroy / watch CLI"
)]
struct Cli {
    /// Path to the `vastai` CLI. Defaults to `vastai` on PATH.
    #[arg(long, default_value = "vastai", global = true)]
    vastai_bin: String,
    /// Read `vastai show instances` raw output from this file instead of
    /// shelling out. Used by integration tests; not for production.
    #[arg(long, global = true)]
    raw_input: Option<std::path::PathBuf>,
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// One-shot status report for instances whose label contains
    /// `--label-prefix`. Prints count, $/hr, status breakdown.
    Status {
        /// Match instances whose label CONTAINS this substring. Use the
        /// sweep's run_id as the prefix.
        #[arg(long)]
        label_prefix: String,
        /// Output format. `text` is human-readable; `json` emits a single
        /// JSON object for piping into other tools.
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// Destroy every instance whose label contains `--label-prefix`.
    /// Defensive against malformed `vastai` output — bad rows are
    /// skipped with a warning, not fatal.
    Destroy {
        #[arg(long)]
        label_prefix: String,
        /// Skip the actual `vastai destroy` calls. Prints what would be
        /// destroyed. Used in tests + dry-run audits.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
    /// Self-destroy: upload an error log to R2 then destroy this
    /// vast.ai instance via the REST API. Called from the worker's
    /// EXIT trap on non-zero exit so a broken box doesn't keep
    /// burning money. Idempotent: if either step fails, the other
    /// still runs.
    ///
    /// Required env vars (or pass `--instance-id` + `--api-key`):
    ///   - `CONTAINER_ID`            — vast.ai sets this in the container
    ///   - `CONTAINER_API_KEY`       — container-scoped vast.ai key
    ///   - `R2_ACCOUNT_ID`           — R2 endpoint derivation
    ///   - `R2_ACCESS_KEY_ID` + `R2_SECRET_ACCESS_KEY` — for s5cmd
    ///
    /// On success: log uploaded to `<r2-prefix>/<instance-id>.log`,
    /// instance destroyed via DELETE
    /// `https://console.vast.ai/api/v0/instances/<id>/`.
    SelfDestroy {
        /// Path to the error log to upload before destroy. If the file
        /// doesn't exist, upload is skipped (still destroys).
        #[arg(long)]
        error_log: std::path::PathBuf,
        /// R2 prefix to upload the log under (e.g.
        /// `s3://zentrain/<run>/errors/`). The log is named
        /// `<instance-id>.log` under this prefix.
        #[arg(long)]
        r2_prefix: String,
        /// Instance ID. Defaults to `$CONTAINER_ID`. Fail if neither set.
        #[arg(long)]
        instance_id: Option<String>,
        /// vast.ai container API key. Defaults to `$CONTAINER_API_KEY`.
        /// Fail if neither set.
        #[arg(long)]
        api_key: Option<String>,
        /// R2 endpoint URL. Default derived from `$R2_ACCOUNT_ID`.
        #[arg(long)]
        r2_endpoint: Option<String>,
        /// s5cmd profile. Default `r2`.
        #[arg(long, default_value = "r2")]
        s5cmd_profile: String,
        /// Skip the actual destroy call. Used for testing the upload
        /// path without burning the box.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
        /// Path to the `curl` binary. Default `curl` on PATH.
        #[arg(long, default_value = "curl")]
        curl_bin: String,
        /// Path to the `s5cmd` binary. Default `s5cmd` on PATH.
        #[arg(long, default_value = "s5cmd")]
        s5cmd_bin: String,
    },
    /// Run as a sweep worker on a vast.ai box. Replaces the bash
    /// `onstart_omni_backfill.sh` dispatch loop. See the
    /// `worker::WorkerArgs` doc for env vars + CLI flags.
    ///
    /// Built into the `zenfleet-vastai` binary so the v22+ docker image
    /// can boot directly into `zenfleet-vastai worker --run-id $RUN_ID ...`
    /// with no bash glue.
    #[cfg(feature = "worker")]
    Worker(worker::WorkerArgs),
    /// Poll fleet until a target sidecar count is reached, then destroy
    /// every matching instance. Replaces the per-sweep
    /// `run_destroy_<metric>_<chunks>.sh` heredocs.
    Watch {
        #[arg(long)]
        label_prefix: String,
        /// Sidecar count to wait for. Watch exits + destroys at >=.
        #[arg(long)]
        target_sidecars: usize,
        /// R2 prefix to count sidecars under. Counting via `s5cmd ls
        /// $prefix | wc -l`. Required.
        #[arg(long)]
        r2_prefix: String,
        /// Hard wall-clock cap. Watch destroys + exits even if the
        /// target isn't met. Default 240 minutes.
        #[arg(long, default_value_t = 240)]
        max_wall_min: u64,
        /// Poll interval in seconds.
        #[arg(long, default_value_t = 90)]
        poll_secs: u64,
        /// R2 endpoint URL. Required if R2 sidecar counting is on.
        /// Defaults to the env var pattern.
        #[arg(long)]
        r2_endpoint: Option<String>,
        /// s5cmd profile to use. Default `r2`.
        #[arg(long, default_value = "r2")]
        s5cmd_profile: String,
        /// Skip the actual destroy at the end (useful for dry runs).
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Cmd::Status {
            label_prefix,
            format,
        } => cmd_status(
            &cli.vastai_bin,
            cli.raw_input.as_deref(),
            &label_prefix,
            &format,
        ),
        Cmd::Destroy {
            label_prefix,
            dry_run,
        } => cmd_destroy(
            &cli.vastai_bin,
            cli.raw_input.as_deref(),
            &label_prefix,
            dry_run,
        ),
        Cmd::SelfDestroy {
            error_log,
            r2_prefix,
            instance_id,
            api_key,
            r2_endpoint,
            s5cmd_profile,
            dry_run,
            curl_bin,
            s5cmd_bin,
        } => cmd_self_destroy(SelfDestroyArgs {
            error_log,
            r2_prefix,
            instance_id,
            api_key,
            r2_endpoint,
            s5cmd_profile,
            dry_run,
            curl_bin,
            s5cmd_bin,
        }),
        #[cfg(feature = "worker")]
        Cmd::Worker(wargs) => worker::cmd_worker(wargs),
        Cmd::Watch {
            label_prefix,
            target_sidecars,
            r2_prefix,
            max_wall_min,
            poll_secs,
            r2_endpoint,
            s5cmd_profile,
            dry_run,
        } => cmd_watch(WatchArgs {
            vastai_bin: cli.vastai_bin,
            raw_input: cli.raw_input,
            label_prefix,
            target_sidecars,
            r2_prefix,
            max_wall_min,
            poll_secs,
            r2_endpoint,
            s5cmd_profile,
            dry_run,
        }),
    }
}

/// Shell out to `vastai show instances-v1 --raw -a` (or read from
/// `--raw-input` if set) and parse. Warnings from the parser go to
/// stderr immediately so the operator sees them before any action.
fn fetch_instances(
    vastai_bin: &str,
    raw_input: Option<&std::path::Path>,
) -> anyhow::Result<Vec<Instance>> {
    let raw = if let Some(p) = raw_input {
        std::fs::read_to_string(p)?
    } else {
        let out = Command::new(vastai_bin)
            .args(["show", "instances-v1", "--raw", "-a"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()?;
        if !out.status.success() {
            anyhow::bail!(
                "vastai show instances-v1 exited {} stderr: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            );
        }
        String::from_utf8_lossy(&out.stdout).to_string()
    };

    let report = parse_instances(&raw)?;
    for w in &report.warnings {
        eprintln!("[zenfleet-vastai] WARN parser: {w}");
    }
    Ok(report.instances)
}

fn cmd_status(
    vastai_bin: &str,
    raw_input: Option<&std::path::Path>,
    label_prefix: &str,
    format: &str,
) -> anyhow::Result<()> {
    let all = fetch_instances(vastai_bin, raw_input)?;
    let matched = filter_by_label(&all, label_prefix);
    let total = matched.len();
    let dph = total_dph(&matched);
    let breakdown = status_breakdown(&matched);

    match format {
        "json" => {
            let mut obj = serde_json::Map::new();
            obj.insert(
                "label_prefix".into(),
                serde_json::Value::String(label_prefix.into()),
            );
            obj.insert("count".into(), serde_json::json!(total));
            obj.insert("dph_total".into(), serde_json::json!(dph));
            let mut bk = serde_json::Map::new();
            for (s, n) in &breakdown {
                bk.insert(s.clone(), serde_json::json!(n));
            }
            obj.insert("status_breakdown".into(), serde_json::Value::Object(bk));
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::Value::Object(obj))?
            );
        }
        _ => {
            println!("label_prefix:  {label_prefix}");
            // Clamp dph to >= 0 for display — empty fleet -0.000 looks
            // confusing.
            let dph_display = if dph.abs() < 1e-9 { 0.0 } else { dph.max(0.0) };
            println!("instances:     {total}");
            println!("burn rate:     ${dph_display:.3}/hr");
            if breakdown.is_empty() {
                println!("status:        (none)");
            } else {
                println!("status:");
                for (s, n) in breakdown {
                    println!("  {s:<12} {n}");
                }
            }
            if !matched.is_empty() {
                println!(
                    "ids: {}",
                    matched
                        .iter()
                        .map(|i| i.id.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
        }
    }
    Ok(())
}

fn cmd_destroy(
    vastai_bin: &str,
    raw_input: Option<&std::path::Path>,
    label_prefix: &str,
    dry_run: bool,
) -> anyhow::Result<()> {
    let all = fetch_instances(vastai_bin, raw_input)?;
    let matched = filter_by_label(&all, label_prefix);
    if matched.is_empty() {
        eprintln!(
            "[zenfleet-vastai] no instances match label substring {label_prefix:?} — nothing to destroy"
        );
        return Ok(());
    }
    eprintln!(
        "[zenfleet-vastai] destroying {} instances matching {label_prefix:?} (total ${:.3}/hr)",
        matched.len(),
        total_dph(&matched)
    );
    let mut failures = 0usize;
    for inst in &matched {
        if dry_run {
            eprintln!(
                "  DRY-RUN destroy {} ({:?})",
                inst.id,
                inst.label.as_deref().unwrap_or("")
            );
            continue;
        }
        let result = Command::new(vastai_bin)
            .args(["destroy", "instance", &inst.id.to_string()])
            // `vastai destroy instance <id>` prompts `y/n` on stdin —
            // pipe yes through so it's non-interactive. Using
            // Stdio::piped + write is overkill; `bash -c` with a here-
            // string keeps it one-shot.
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        let mut child = match result {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  WARN spawn destroy for {}: {e}", inst.id);
                failures += 1;
                continue;
            }
        };
        // Send "y\n" to confirm.
        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            let _ = stdin.write_all(b"y\n");
        }
        match child.wait_with_output() {
            Ok(out) if out.status.success() => {
                eprintln!(
                    "  destroyed {} ({:?})",
                    inst.id,
                    inst.label.as_deref().unwrap_or("")
                );
            }
            Ok(out) => {
                eprintln!(
                    "  WARN destroy {} exited {}: {}",
                    inst.id,
                    out.status,
                    String::from_utf8_lossy(&out.stderr)
                        .lines()
                        .next()
                        .unwrap_or("")
                );
                failures += 1;
            }
            Err(e) => {
                eprintln!("  WARN destroy {} wait error: {e}", inst.id);
                failures += 1;
            }
        }
    }
    eprintln!(
        "[zenfleet-vastai] done — {} destroyed, {failures} failures",
        matched.len() - failures
    );
    if failures > 0 {
        anyhow::bail!("{failures} destroy(s) failed; re-run zenfleet-vastai destroy to retry");
    }
    Ok(())
}

struct WatchArgs {
    vastai_bin: String,
    raw_input: Option<std::path::PathBuf>,
    label_prefix: String,
    target_sidecars: usize,
    r2_prefix: String,
    max_wall_min: u64,
    poll_secs: u64,
    r2_endpoint: Option<String>,
    s5cmd_profile: String,
    dry_run: bool,
}

fn cmd_watch(args: WatchArgs) -> anyhow::Result<()> {
    let start = Instant::now();
    let max_wall = Duration::from_secs(args.max_wall_min * 60);
    let poll = Duration::from_secs(args.poll_secs);

    eprintln!(
        "[zenfleet-vastai watch] label_prefix={:?} target={} r2_prefix={} max_wall={}min poll={}s",
        args.label_prefix, args.target_sidecars, args.r2_prefix, args.max_wall_min, args.poll_secs
    );

    loop {
        let elapsed = start.elapsed();
        let n_sidecars = count_sidecars(
            &args.r2_prefix,
            args.r2_endpoint.as_deref(),
            &args.s5cmd_profile,
        )
        .unwrap_or_else(|e| {
            eprintln!("[zenfleet-vastai watch] WARN sidecar count: {e}");
            0
        });

        // Always do a fleet status snapshot inside the loop — even if
        // the watcher exits without destroying, the operator gets a
        // last-known state in the log.
        let inst_result = fetch_instances(&args.vastai_bin, args.raw_input.as_deref());
        let (n_inst, dph) = match &inst_result {
            Ok(all) => {
                let matched = filter_by_label(all, &args.label_prefix);
                (matched.len(), total_dph(&matched))
            }
            Err(e) => {
                eprintln!("[zenfleet-vastai watch] WARN fetch instances: {e}");
                (0, 0.0)
            }
        };

        eprintln!(
            "[zenfleet-vastai watch] t+{:.1}min  sidecars: {}/{}  fleet: {} (${:.3}/hr)",
            elapsed.as_secs_f64() / 60.0,
            n_sidecars,
            args.target_sidecars,
            n_inst,
            dph
        );

        if n_sidecars >= args.target_sidecars {
            eprintln!("[zenfleet-vastai watch] target hit — destroying fleet");
            cmd_destroy(
                &args.vastai_bin,
                args.raw_input.as_deref(),
                &args.label_prefix,
                args.dry_run,
            )?;
            return Ok(());
        }
        if elapsed >= max_wall {
            eprintln!(
                "[zenfleet-vastai watch] wall cap {}min hit (sidecars={}/{}); destroying anyway",
                args.max_wall_min, n_sidecars, args.target_sidecars
            );
            cmd_destroy(
                &args.vastai_bin,
                args.raw_input.as_deref(),
                &args.label_prefix,
                args.dry_run,
            )?;
            return Ok(());
        }

        std::thread::sleep(poll);
    }
}

struct SelfDestroyArgs {
    error_log: std::path::PathBuf,
    r2_prefix: String,
    instance_id: Option<String>,
    api_key: Option<String>,
    r2_endpoint: Option<String>,
    s5cmd_profile: String,
    dry_run: bool,
    curl_bin: String,
    s5cmd_bin: String,
}

fn cmd_self_destroy(args: SelfDestroyArgs) -> anyhow::Result<()> {
    // Resolve instance ID + API key. Fail loudly if neither is set —
    // we never want to silently "succeed" without destroying the box.
    let instance_id = args
        .instance_id
        .clone()
        .or_else(|| std::env::var("CONTAINER_ID").ok())
        .ok_or_else(|| {
            anyhow::anyhow!("instance-id: pass --instance-id or set $CONTAINER_ID (vast.ai sets this in the container)")
        })?;
    let api_key = args
        .api_key
        .clone()
        .or_else(|| std::env::var("CONTAINER_API_KEY").ok())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "api-key: pass --api-key or set $CONTAINER_API_KEY (vast.ai sets this in the container)"
            )
        })?;

    eprintln!(
        "[zenfleet-vastai self-destroy] instance_id={instance_id} dry_run={}",
        args.dry_run
    );

    // Step 1: upload error log (best effort — don't fail destroy if
    // upload bombs). Use the standard R2 endpoint pattern other
    // workers use.
    let log_exists = args.error_log.exists();
    if log_exists {
        let endpoint = args.r2_endpoint.clone().or_else(|| {
            std::env::var("R2_ACCOUNT_ID")
                .ok()
                .map(|aid| format!("https://{aid}.r2.cloudflarestorage.com"))
        });
        // Normalize prefix to have trailing slash, then append <id>.log.
        let prefix = if args.r2_prefix.ends_with('/') {
            args.r2_prefix.clone()
        } else {
            format!("{}/", args.r2_prefix)
        };
        let target = format!("{prefix}{instance_id}.log");
        eprintln!(
            "  step 1/2: uploading {} -> {target}",
            args.error_log.display()
        );
        let mut cmd = Command::new(&args.s5cmd_bin);
        if let Some(ep) = endpoint.as_deref() {
            cmd.args(["--endpoint-url", ep]);
        }
        cmd.args([
            "--profile",
            &args.s5cmd_profile,
            "cp",
            args.error_log.to_str().expect("non-utf8 error-log path"),
            &target,
        ]);
        match cmd.output() {
            Ok(out) if out.status.success() => {
                eprintln!("  log uploaded: {target}");
            }
            Ok(out) => {
                eprintln!(
                    "  WARN log upload exited {}: {}",
                    out.status,
                    String::from_utf8_lossy(&out.stderr)
                );
            }
            Err(e) => {
                eprintln!("  WARN log upload spawn: {e}");
            }
        }
    } else {
        eprintln!(
            "  step 1/2: skipped (error-log {} does not exist)",
            args.error_log.display()
        );
    }

    // Step 2: destroy via vast.ai REST API. The instance management
    // endpoint accepts DELETE with the API key in the Authorization
    // header. See https://docs.vast.ai/api/v0/instances/.
    let url = format!("https://console.vast.ai/api/v0/instances/{instance_id}/");
    eprintln!("  step 2/2: DELETE {url}");
    if args.dry_run {
        eprintln!("  DRY-RUN — not calling DELETE; destroy skipped");
        return Ok(());
    }
    let out = Command::new(&args.curl_bin)
        .args([
            "-fsSL",
            "--max-time",
            "30",
            "-X",
            "DELETE",
            "-H",
            &format!("Authorization: Bearer {api_key}"),
            "-H",
            "Accept: application/json",
            &url,
        ])
        .output()
        .map_err(|e| anyhow::anyhow!("curl spawn: {e}"))?;
    if !out.status.success() {
        // curl returns nonzero on HTTP 4xx/5xx with -f. Don't bail —
        // the box is going down anyway; print the body for the
        // sidecar-error log so the operator sees the failure.
        anyhow::bail!(
            "curl DELETE exited {}: stdout={:?} stderr={:?}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    eprintln!(
        "  destroyed: {}",
        String::from_utf8_lossy(&out.stdout).trim()
    );
    Ok(())
}

/// Count objects under `prefix` in R2 by shelling out to `s5cmd ls`.
/// `s5cmd` is the same tool the workers use, so it's always installed
/// on the operator host.
fn count_sidecars(prefix: &str, endpoint: Option<&str>, profile: &str) -> anyhow::Result<usize> {
    let mut cmd = Command::new("s5cmd");
    if let Some(ep) = endpoint {
        cmd.args(["--endpoint-url", ep]);
    } else if let Ok(account_id) = std::env::var("R2_ACCOUNT_ID") {
        cmd.args([
            "--endpoint-url",
            &format!("https://{account_id}.r2.cloudflarestorage.com"),
        ]);
    }
    cmd.args(["--profile", profile, "ls", prefix]);
    let out = cmd.output()?;
    if !out.status.success() {
        anyhow::bail!(
            "s5cmd ls {prefix} exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    // `s5cmd ls` prints one line per object; blank lines are excluded.
    let count = stdout.lines().filter(|l| !l.trim().is_empty()).count();
    Ok(count)
}
