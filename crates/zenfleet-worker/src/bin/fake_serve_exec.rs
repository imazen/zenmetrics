#![forbid(unsafe_code)]
//! Test-support executor speaking BOTH halves of the `ZEN_EXEC` contract, with scriptable
//! failure shapes keyed on the job's `cell.codec`. Used by the [`zenfleet_worker`] warm-pool
//! tests (`CARGO_BIN_EXE_fake-serve-exec`); never deployed.
//!
//! One-shot mode (no args): read the DesiredJob JSON on stdin to EOF, echo it on stdout
//! (`/bin/cat` semantics — the A/B byte-identity partner).
//!
//! `--serve` mode: loop `[u32 LE len][job JSON]` → `[u8 status][u32 LE len][payload]` frames,
//! exactly like `zenmetrics jobexec --serve`. On start, appends `start <pid>` to the file named
//! by `$FAKE_SERVE_LOG` (how tests count process spawns).
//!
//! Behaviors by `cell.codec`:
//! - anything else → status 0, payload = the job JSON (echo)
//! - `job-error`    → status 1, payload `boom` (child stays alive)
//! - `classified-oom` → status 3, payload `{"class":"oom","detail":"vram exhausted"}`
//! - `die`          → `std::process::exit(9)` mid-job, NO response (worker sees a dead child)
//! - `grow`         → leak + touch ~96 MiB, then echo (drives the RSS-watermark recycle)

use std::io::{Read, Write};

fn read_exact_or_eof(r: &mut impl Read, buf: &mut [u8]) -> std::io::Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            Ok(0) => {
                return if filled == 0 {
                    Ok(false)
                } else {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "truncated frame",
                    ))
                };
            }
            Ok(n) => filled += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

fn codec_of(job_json: &[u8]) -> String {
    serde_json::from_slice::<serde_json::Value>(job_json)
        .ok()
        .and_then(|v| {
            v.get("cell")
                .and_then(|c| c.get("codec"))
                .and_then(|s| s.as_str())
                .map(str::to_string)
        })
        .unwrap_or_default()
}

fn main() {
    let serve = std::env::args().any(|a| a == "--serve");
    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();

    if !serve {
        // One-shot: echo the job JSON (cat semantics).
        let mut buf = Vec::new();
        stdin.read_to_end(&mut buf).expect("read stdin");
        stdout.write_all(&buf).expect("write stdout");
        return;
    }

    if let Ok(log) = std::env::var("FAKE_SERVE_LOG") {
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log)
            .expect("open FAKE_SERVE_LOG");
        writeln!(f, "start {}", std::process::id()).expect("log start");
    }

    // Held leaks for the `grow` behavior (touched so VmRSS actually materializes).
    let mut leaks: Vec<Vec<u8>> = Vec::new();

    let mut lenb = [0u8; 4];
    loop {
        match read_exact_or_eof(&mut stdin, &mut lenb) {
            Ok(false) => return, // clean EOF
            Ok(true) => {}
            Err(_) => std::process::exit(3),
        }
        let len = u32::from_le_bytes(lenb) as usize;
        let mut job = vec![0u8; len];
        if !read_exact_or_eof(&mut stdin, &mut job).unwrap_or(false) {
            std::process::exit(3);
        }
        let (status, payload): (u8, Vec<u8>) = match codec_of(&job).as_str() {
            "job-error" => (1, b"boom".to_vec()),
            "classified-oom" => (3, br#"{"class":"oom","detail":"vram exhausted"}"#.to_vec()),
            "die" => std::process::exit(9),
            "grow" => {
                let mut block = vec![0u8; 96 << 20];
                for i in (0..block.len()).step_by(4096) {
                    block[i] = 1; // touch every page so RSS is real
                }
                leaks.push(block);
                (0, job.clone())
            }
            _ => (0, job.clone()),
        };
        stdout.write_all(&[status]).expect("status");
        stdout
            .write_all(&(payload.len() as u32).to_le_bytes())
            .expect("len");
        stdout.write_all(&payload).expect("payload");
        stdout.flush().expect("flush");
    }
}
