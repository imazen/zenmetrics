//! Persistent distortion-generation worker for the sweep.
//!
//! When `--distort-cmd` is set, the sweep generates each cell's *distorted*
//! image by handing the reference to an external program instead of codec-
//! encoding it. The program is a **long-lived serve process** (spawned once per
//! `run_sweep`), so its interpreter/imports (numpy/scipy/torch for the KADIS
//! generator) are paid ONCE per box and amortized across every cell — keeping
//! generation well under the "< 30 % of scoring time" budget that makes
//! generate-discard worthwhile.
//!
//! It is image-major: one round-trip per source image carries the reference and
//! all of that image's cell knob-tuples, and returns one distorted variant per
//! cell. The reference is decoded once by the sweep (shared across the image's
//! cells) and once by the worker.
//!
//! ## Wire protocol (same length framing as `jobexec --serve`: `[u32 LE len][bytes]`)
//!
//! Request  = frame(JSON header) then frame(raw reference RGB8, `w*h*3` bytes):
//! ```json
//! {"width": W, "height": H,
//!  "cells": [{"q": 3, "knob_tuple_json": "{\"dist_type\":10}"}, ...]}
//! ```
//! Response = frame(JSON header) then frame(raw variant RGB8 bytes, concatenated):
//! ```json
//! {"variants": [{"ok": true}, {"ok": false, "error": "..."}, ...]}
//! ```
//! Each `ok` variant contributes `w*h*3` bytes to the trailing raw frame, in
//! cell order (failed variants contribute nothing). Variants must match the
//! reference dimensions (every KADIS distortion is dimension-preserving).

use std::error::Error;
use std::io::{BufReader, BufWriter, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use crate::decode::Rgb8Image;

/// A spawned, long-lived distortion-generation subprocess.
pub struct DistortWorker {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl DistortWorker {
    /// Spawn `cmd` through the platform shell so callers can pass a full command
    /// line (e.g. `python3 -m kadis_distort.serve`). The child's stderr is
    /// inherited so worker logs/tracebacks land in the sweep's stderr.
    pub fn spawn(cmd: &str) -> Result<Self, Box<dyn Error>> {
        #[cfg(windows)]
        let mut builder = {
            let mut c = Command::new("cmd");
            c.arg("/C").arg(cmd);
            c
        };
        #[cfg(not(windows))]
        let mut builder = {
            let mut c = Command::new("sh");
            c.arg("-c").arg(cmd);
            c
        };
        let mut child = builder
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("failed to spawn distort worker `{cmd}`: {e}"))?;
        let stdin = BufWriter::new(child.stdin.take().ok_or("distort worker: no stdin")?);
        let stdout = BufReader::new(child.stdout.take().ok_or("distort worker: no stdout")?);
        Ok(Self {
            child,
            stdin,
            stdout,
        })
    }

    /// Generate the distorted variant for each `(q, knob_tuple_json)` cell of
    /// `source`. Returns one entry per cell, `None` where the worker reported a
    /// generation failure for that cell. Errors (protocol/process failures) are
    /// fatal to the worker and propagate.
    pub fn generate(
        &mut self,
        source: &Rgb8Image,
        cells: &[(f64, String)],
    ) -> Result<Vec<Option<Rgb8Image>>, Box<dyn Error>> {
        let w = source.width;
        let h = source.height;
        let per = (w as usize) * (h as usize) * 3;
        if source.pixels.len() != per {
            return Err(format!(
                "distort worker: reference pixel buffer {} != w*h*3 {per}",
                source.pixels.len()
            )
            .into());
        }

        // --- request ---
        let cell_json: Vec<serde_json::Value> = cells
            .iter()
            .map(|(q, knob)| serde_json::json!({"q": q, "knob_tuple_json": knob}))
            .collect();
        let header = serde_json::json!({"width": w, "height": h, "cells": cell_json});
        write_frame(&mut self.stdin, &serde_json::to_vec(&header)?)?;
        write_frame(&mut self.stdin, &source.pixels)?;
        self.stdin.flush()?;

        // --- response header ---
        let resp_bytes = read_frame(&mut self.stdout)?;
        let resp: serde_json::Value = serde_json::from_slice(&resp_bytes)
            .map_err(|e| format!("distort worker: bad response JSON: {e}"))?;
        let variants = resp
            .get("variants")
            .and_then(|v| v.as_array())
            .ok_or("distort worker: response missing `variants` array")?;
        if variants.len() != cells.len() {
            return Err(format!(
                "distort worker: returned {} variants for {} cells",
                variants.len(),
                cells.len()
            )
            .into());
        }
        let oks: Vec<bool> = variants
            .iter()
            .map(|v| {
                v.get("ok")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            })
            .collect();
        for (v, ok) in variants.iter().zip(&oks) {
            if !*ok
                && let Some(err) = v.get("error").and_then(serde_json::Value::as_str)
            {
                eprintln!("[sweep] distort worker cell failed: {err}");
            }
        }

        // --- response raw pixels ---
        let raw = read_frame(&mut self.stdout)?;
        let n_ok = oks.iter().filter(|b| **b).count();
        if raw.len() != n_ok * per {
            return Err(format!(
                "distort worker: raw frame {} != n_ok({n_ok}) * w*h*3({per}) = {}",
                raw.len(),
                n_ok * per
            )
            .into());
        }
        let mut out = Vec::with_capacity(cells.len());
        let mut off = 0usize;
        for ok in &oks {
            if *ok {
                out.push(Some(Rgb8Image {
                    pixels: raw[off..off + per].to_vec(),
                    width: w,
                    height: h,
                }));
                off += per;
            } else {
                out.push(None);
            }
        }
        Ok(out)
    }
}

impl Drop for DistortWorker {
    fn drop(&mut self) {
        // Closing stdin signals EOF; the worker's serve loop exits cleanly.
        // Then reap so we don't leave a zombie.
        let _ = self.stdin.flush();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn write_frame<W: Write>(w: &mut W, data: &[u8]) -> std::io::Result<()> {
    let len = u32::try_from(data.len()).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "distort frame > 4 GiB")
    })?;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(data)
}

fn read_frame<R: Read>(r: &mut R) -> std::io::Result<Vec<u8>> {
    let mut lenb = [0u8; 4];
    r.read_exact(&mut lenb)?;
    let len = u32::from_le_bytes(lenb) as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// A minimal serve worker (the protocol's other end) that echoes the
    /// reference back as every cell's variant. Pure stdlib python3 — no
    /// dependency on the kadis_distort package — so it exercises the framing
    /// alone. Written to a temp file to avoid `-c` quoting.
    fn identity_worker_cmd() -> String {
        let script = "\
import sys, struct, json
def rd(n):
    b = b''
    while len(b) < n:
        c = sys.stdin.buffer.read(n - len(b))
        if not c:
            return None
        b += c
    return b
while True:
    lh = rd(4)
    if lh is None:
        break
    header = rd(struct.unpack('<I', lh)[0])
    ref = rd(struct.unpack('<I', rd(4))[0])
    n = len(json.loads(header)['cells'])
    resp = json.dumps({'variants': [{'ok': True} for _ in range(n)]}).encode()
    sys.stdout.buffer.write(struct.pack('<I', len(resp)) + resp)
    raw = ref * n
    sys.stdout.buffer.write(struct.pack('<I', len(raw)) + raw)
    sys.stdout.buffer.flush()
";
        let path =
            std::env::temp_dir().join(format!("kadis_identity_worker_{}.py", std::process::id()));
        std::fs::write(&path, script).unwrap();
        format!("python3 {}", path.display())
    }

    #[test]
    fn identity_worker_roundtrips_each_cell() {
        let mut worker = DistortWorker::spawn(&identity_worker_cmd()).expect("spawn worker");
        // 2x2 RGB8 reference.
        let src = Rgb8Image {
            pixels: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            width: 2,
            height: 2,
        };
        let cells = vec![
            (1.0, "{\"dist_type\":10}".to_string()),
            (2.0, "{\"dist_type\":11}".to_string()),
            (3.0, "{\"dist_type\":21}".to_string()),
        ];
        let out = worker.generate(&src, &cells).expect("generate");
        assert_eq!(out.len(), 3, "one variant per cell");
        for variant in &out {
            let img = variant
                .as_ref()
                .expect("identity worker returns ok for every cell");
            assert_eq!((img.width, img.height), (2, 2));
            assert_eq!(
                img.pixels, src.pixels,
                "identity worker echoes the reference"
            );
        }
    }
}
