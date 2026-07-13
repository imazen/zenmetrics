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
        let header = serde_json::json!({"width": w, "height": h, "cells": cells_json(cells)});
        let raws = self.generate_impl(&header, &source.pixels, per, cells.len())?;
        Ok(raws
            .into_iter()
            .map(|r| {
                r.map(|pixels| Rgb8Image {
                    pixels,
                    width: w,
                    height: h,
                })
            })
            .collect())
    }

    /// Protocol-v2 16-bit generation (2026-07-13, HDR): `rgb16` is tight
    /// interleaved RGB u16 **PQ code values**; the worker distorts in the PQ
    /// code-value domain (clamp, never mapmm) and each returned variant is the
    /// same shape. `ref_name` seeds the worker's per-cell RNG name-based
    /// (`kadis_distort.io.seed_for`) so fleet cells are byte-identical to the
    /// local HDR grid driver's. Cell `knob_tuple_json` must carry `dist_type`
    /// AND `level` (the HDR grid packs `q = dist_type*10 + level`, so q is not
    /// the level).
    pub fn generate_rgb16(
        &mut self,
        rgb16: &[u16],
        width: u32,
        height: u32,
        ref_name: &str,
        cells: &[(f64, String)],
    ) -> Result<Vec<Option<Vec<u16>>>, Box<dyn Error>> {
        let n = (width as usize) * (height as usize) * 3;
        if rgb16.len() != n {
            return Err(format!(
                "distort worker: reference sample buffer {} != w*h*3 {n}",
                rgb16.len()
            )
            .into());
        }
        let mut bytes = Vec::with_capacity(n * 2);
        for &v in rgb16 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let header = serde_json::json!({
            "width": width, "height": height, "bit_depth": 16,
            "ref_name": ref_name, "cells": cells_json(cells),
        });
        let raws = self.generate_impl(&header, &bytes, n * 2, cells.len())?;
        Ok(raws
            .into_iter()
            .map(|r| {
                r.map(|b| {
                    b.chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect()
                })
            })
            .collect())
    }

    /// Shared request/response core: send `header` + the raw reference frame,
    /// parse per-cell statuses, and split the trailing raw frame into
    /// `per`-byte variants (in cell order, failed cells contribute nothing).
    fn generate_impl(
        &mut self,
        header: &serde_json::Value,
        source_bytes: &[u8],
        per: usize,
        n_cells: usize,
    ) -> Result<Vec<Option<Vec<u8>>>, Box<dyn Error>> {
        // --- request ---
        write_frame(&mut self.stdin, &serde_json::to_vec(header)?)?;
        write_frame(&mut self.stdin, source_bytes)?;
        self.stdin.flush()?;

        // --- response header ---
        let resp_bytes = read_frame(&mut self.stdout)?;
        let resp: serde_json::Value = serde_json::from_slice(&resp_bytes)
            .map_err(|e| format!("distort worker: bad response JSON: {e}"))?;
        let variants = resp
            .get("variants")
            .and_then(|v| v.as_array())
            .ok_or("distort worker: response missing `variants` array")?;
        if variants.len() != n_cells {
            return Err(format!(
                "distort worker: returned {} variants for {} cells",
                variants.len(),
                n_cells
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
            if !*ok && let Some(err) = v.get("error").and_then(serde_json::Value::as_str) {
                eprintln!("[sweep] distort worker cell failed: {err}");
            }
        }

        // --- response raw pixels ---
        let raw = read_frame(&mut self.stdout)?;
        let n_ok = oks.iter().filter(|b| **b).count();
        if raw.len() != n_ok * per {
            return Err(format!(
                "distort worker: raw frame {} != n_ok({n_ok}) * per({per}) = {}",
                raw.len(),
                n_ok * per
            )
            .into());
        }
        let mut out = Vec::with_capacity(n_cells);
        let mut off = 0usize;
        for ok in &oks {
            if *ok {
                out.push(Some(raw[off..off + per].to_vec()));
                off += per;
            } else {
                out.push(None);
            }
        }
        Ok(out)
    }
}

fn cells_json(cells: &[(f64, String)]) -> Vec<serde_json::Value> {
    cells
        .iter()
        .map(|(q, knob)| serde_json::json!({"q": q, "knob_tuple_json": knob}))
        .collect()
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
    fn identity_worker_roundtrips_u16_cells() {
        // The identity worker echoes whatever reference bytes it received, so
        // it exercises the v2 framing (bit_depth 16 header + u16 LE frames)
        // without a kadis_distort dependency.
        let mut worker = DistortWorker::spawn(&identity_worker_cmd()).expect("spawn worker");
        let rgb16: Vec<u16> = (0u16..12).map(|v| v * 5000).collect(); // 2x2 RGB16
        let cells = vec![
            (102.0, "{\"dist_type\":10,\"level\":2}".to_string()),
            (213.0, "{\"dist_type\":21,\"level\":3}".to_string()),
        ];
        let out = worker
            .generate_rgb16(&rgb16, 2, 2, "ref_a.hdr.png", &cells)
            .expect("generate_rgb16");
        assert_eq!(out.len(), 2, "one variant per cell");
        for variant in &out {
            assert_eq!(
                variant.as_ref().expect("identity worker returns ok"),
                &rgb16,
                "identity worker echoes the u16 reference"
            );
        }
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
