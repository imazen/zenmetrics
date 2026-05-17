#![forbid(unsafe_code)]

//! External-process metric plugin.
//!
//! Spawns a long-lived child process (typically a Python server with
//! PyTorch + a reference implementation loaded once) and exchanges
//! `(reference, distorted)` RGB8 frames + scalar scores over stdin/stdout.
//!
//! ## Protocol
//!
//! 1. **Handshake.** On spawn the server writes one line to stdout:
//!    `{"ready":true,"name":"<metric>"}`. zen-metrics blocks until it
//!    sees it before issuing any scoring requests.
//! 2. **Request (zen-metrics → server).** One ASCII header line followed by
//!    raw binary pixels:
//!    - `{"w":<u32>,"h":<u32>}\n`
//!    - `w * h * 3` bytes of reference RGB8 (row-major, tightly packed)
//!    - `w * h * 3` bytes of distorted RGB8
//! 3. **Response (server → zen-metrics).** One JSON line:
//!    - success: `{"score":<float>}\n` (optional extra keys ignored)
//!    - failure: `{"error":"<message>"}\n`
//!
//! ## Concurrency
//!
//! One subprocess per metric (singleton per zen-metrics process), guarded
//! by a `Mutex`. Rayon threads inside the sweep runner serialize through
//! the mutex; that's acceptable because the Python side does its own
//! batched GPU work anyway and we deliberately want one GPU-accelerated
//! metric server per worker. Throughput scaling comes from fanning out
//! multiple zen-metrics processes (one per CPU core), each with its own
//! Python server — the model `scripts/sweep/onstart_v3.sh` already uses
//! for the in-process metrics.
//!
//! ## Configuration
//!
//! The subprocess command line is read from an env var at first use:
//! - `ZEN_METRICS_EXTERNAL_CVVDP` for the cvvdp metric
//! - `ZEN_METRICS_EXTERNAL_IWSSIM` for the iwssim metric
//!
//! Values are split on ASCII whitespace; double-quoted runs are preserved
//! verbatim. Example:
//!     ZEN_METRICS_EXTERNAL_CVVDP="/opt/venvs/cvvdp/bin/python /opt/python/metric_server.py cvvdp"

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Mutex, OnceLock};

use crate::decode::Rgb8Image;

/// Long-lived child process holding stdin/stdout pipes.
pub struct ExternalServer {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl ExternalServer {
    fn spawn(cmd_line: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut parts = split_command(cmd_line);
        if parts.is_empty() {
            return Err("empty external metric command".into());
        }
        let program = parts.remove(0);
        let mut child = Command::new(&program)
            .args(&parts)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("failed to spawn `{program}`: {e}"))?;
        let stdin = child.stdin.take().ok_or("failed to capture child stdin")?;
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .ok_or("failed to capture child stdout")?,
        );
        let mut srv = ExternalServer {
            child,
            stdin,
            stdout,
        };
        let mut line = String::new();
        let n = srv.stdout.read_line(&mut line)?;
        if n == 0 {
            return Err("external metric server closed stdout before announcing ready".into());
        }
        if !line.contains("\"ready\"") {
            return Err(format!(
                "external metric server did not announce ready (got: {})",
                line.trim()
            )
            .into());
        }
        Ok(srv)
    }

    fn score(
        &mut self,
        reference: &Rgb8Image,
        distorted: &Rgb8Image,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err("external metric: dimension mismatch".into());
        }
        let expected_bytes = (reference.width as usize)
            .checked_mul(reference.height as usize)
            .and_then(|n| n.checked_mul(3))
            .ok_or("external metric: width*height*3 overflow")?;
        if reference.pixels.len() != expected_bytes || distorted.pixels.len() != expected_bytes {
            return Err(format!(
                "external metric: expected {expected_bytes} bytes per side, got ref={} dist={}",
                reference.pixels.len(),
                distorted.pixels.len()
            )
            .into());
        }
        let header = format!("{{\"w\":{},\"h\":{}}}\n", reference.width, reference.height);
        self.stdin.write_all(header.as_bytes())?;
        self.stdin.write_all(&reference.pixels)?;
        self.stdin.write_all(&distorted.pixels)?;
        self.stdin.flush()?;

        let mut line = String::new();
        let n = self.stdout.read_line(&mut line)?;
        if n == 0 {
            return Err("external metric server closed stdout mid-request".into());
        }
        parse_score(&line)
    }
}

impl Drop for ExternalServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn parse_score(line: &str) -> Result<f64, Box<dyn std::error::Error>> {
    if let Some(after) = line.find("\"error\"").map(|i| &line[i..])
        && let Some(quote_open) = after.find(':').and_then(|i| after[i..].find('"'))
    {
        let start = after.find(':').unwrap() + quote_open + 1;
        let rest = &after[start..];
        let end = rest.find('"').unwrap_or(rest.len());
        return Err(format!("external metric error: {}", &rest[..end]).into());
    }
    let key = "\"score\"";
    let idx = line
        .find(key)
        .ok_or_else(|| format!("external metric: unparseable response: {}", line.trim()))?;
    let after_colon = line[idx + key.len()..]
        .trim_start()
        .strip_prefix(':')
        .ok_or("external metric: missing ':' after score")?
        .trim_start();
    let end = after_colon
        .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(after_colon.len());
    after_colon[..end].parse::<f64>().map_err(|e| {
        format!(
            "external metric: bad score literal `{}`: {e}",
            &after_colon[..end]
        )
        .into()
    })
}

fn split_command(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_quote = false;
    for c in s.chars() {
        match c {
            '"' => in_quote = !in_quote,
            ws if ws.is_whitespace() && !in_quote => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

static CVVDP_SERVER: OnceLock<Mutex<Option<ExternalServer>>> = OnceLock::new();
static IWSSIM_SERVER: OnceLock<Mutex<Option<ExternalServer>>> = OnceLock::new();

fn ensure_server<'a>(
    env_var: &str,
    slot: &'a Mutex<Option<ExternalServer>>,
) -> Result<std::sync::MutexGuard<'a, Option<ExternalServer>>, Box<dyn std::error::Error>> {
    let mut guard = slot.lock().map_err(|_| "external server mutex poisoned")?;
    if guard.is_none() {
        let cmd = std::env::var(env_var).map_err(|_| {
            format!(
                "{env_var} env var not set — must point to the metric server launch command \
                 (e.g. `/opt/venvs/<metric>/bin/python /opt/python/metric_server.py <metric>`)"
            )
        })?;
        *guard = Some(ExternalServer::spawn(&cmd)?);
    }
    Ok(guard)
}

pub fn score_cvvdp(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    let slot = CVVDP_SERVER.get_or_init(|| Mutex::new(None));
    let mut guard = ensure_server("ZEN_METRICS_EXTERNAL_CVVDP", slot)?;
    guard.as_mut().unwrap().score(reference, distorted)
}

pub fn score_iwssim(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<f64, Box<dyn std::error::Error>> {
    let slot = IWSSIM_SERVER.get_or_init(|| Mutex::new(None));
    let mut guard = ensure_server("ZEN_METRICS_EXTERNAL_IWSSIM", slot)?;
    guard.as_mut().unwrap().score(reference, distorted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_score_line() {
        assert_eq!(parse_score("{\"score\":0.9123}\n").unwrap(), 0.9123);
        assert_eq!(parse_score("{\"score\": -1.25e-3}\n").unwrap(), -1.25e-3);
        assert_eq!(parse_score("{\"score\":7,\"extra\":1}\n").unwrap(), 7.0);
    }

    #[test]
    fn surfaces_error_field() {
        let err = parse_score("{\"error\":\"boom\"}\n").unwrap_err();
        assert!(err.to_string().contains("boom"));
    }

    #[test]
    fn split_command_handles_quotes() {
        assert_eq!(
            split_command(r#"python -m foo "arg with space" tail"#),
            vec!["python", "-m", "foo", "arg with space", "tail"]
        );
    }
}
