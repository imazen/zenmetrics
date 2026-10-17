//! Dump DMM's trunk taps for every `dmm_goldens.tsv` row as `.npy`
//! (`{stem}_{size}_{ref|dist}_tap{0,1}.npy` under `<out_dir>`), for
//! `models/dmm/gen_netlib_goldens.py` — the netlib-`sgesdd` goldens that
//! `nn-parity-dmm` gates on must be scored on the same taps this engine
//! produces (the head amplifies tap noise ~100x into score units).
//! Also writes `<out_dir>/port.tsv` — this port's own score per row —
//! so the golden file can pin rows whose netlib draw diverges past the
//! 1e-3 JND gate (solver drift, see card.toml).
//!
//! ```text
//! ZENMETRICS_NN_ENGINE={rten|ort-cuda} \
//! cargo run --release -p zenmetrics-nn --features model-tests --example dmm_tap_dump -- \
//!     <model_dir> <ladder_dir> <out_dir>
//! ```

use std::io::Write as _;
use std::path::Path;

use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{EvalOpts, Image, Metric, Rgb8In};
use zenmetrics_nn::{Dmm, Engine};

fn load_rgb8(path: &Path) -> (Vec<u8>, usize, usize) {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let out = zenpng::decode(
        &data,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )
    .expect("png decode");
    let s = out.pixels.as_slice();
    let (w, h) = (s.width() as usize, s.rows() as usize);
    assert_eq!(
        s.descriptor().bytes_per_pixel(),
        3,
        "{}: expected RGB8",
        path.display()
    );
    let bytes = s.as_strided_bytes();
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        px.extend_from_slice(&bytes[y * s.stride()..y * s.stride() + w * 3]);
    }
    (px, w, h)
}

/// NPY v1.0, f32 little-endian, C-order `(c, h, w)`.
fn write_npy(path: &Path, c: usize, h: usize, w: usize, data: &[f32]) {
    let mut f = std::fs::File::create(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({c}, {h}, {w}), }}");
    // magic(6) + ver(2) + hlen(2) + header; header is padded so the whole
    // prefix is 64-aligned and ends with '\n'.
    let hlen = dict.len() + 1;
    let pad = (64 - (10 + hlen) % 64) % 64;
    let header = format!("{dict}{}\n", " ".repeat(pad));
    f.write_all(b"\x93NUMPY\x01\x00").unwrap();
    f.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    for &v in data {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (model_dir, ladder, out_dir) = (
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
    );
    let ladder_ns = ladder.with_file_name(format!(
        "{}-nonsquare",
        ladder.file_name().unwrap().to_str().unwrap()
    ));
    if let Some(parent) = out_dir.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    // Preserve prior tap/score artifacts. Use a fresh destination for a rerun.
    std::fs::create_dir(out_dir).expect("output directory must be new");
    let engine = match std::env::var("ZENMETRICS_NN_ENGINE").as_deref() {
        Err(_) | Ok("rten") => Engine::Rten,
        #[cfg(feature = "ort")]
        Ok("ort-cuda") => Engine::OrtCuda,
        Ok(other) => panic!("unknown or unbuilt engine {other}"),
    };
    let mut m = Dmm::from_dir(model_dir, engine).expect("load dmm");
    let goldens =
        std::fs::read_to_string(model_dir.join("dmm_goldens.tsv")).expect("dmm_goldens.tsv");
    let mut port = std::fs::File::create(out_dir.join("port.tsv")).expect("port.tsv");
    writeln!(port, "image\tsize\tdmm").unwrap();
    let mut checked = 0;
    for line in goldens.lines().skip(1) {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let (stem, size) = (f[0], f[1]);
        // Optional diagnostic subset, independent of the full parity gate.
        if args
            .get(4)
            .is_some_and(|rows| !rows.split(',').any(|row| row == format!("{stem}:{size}")))
        {
            continue;
        }
        checked += 1;
        let dir = if size.contains('x') {
            &ladder_ns
        } else {
            ladder
        };
        let mut pair = [Vec::new(), Vec::new()];
        let mut dims = (0usize, 0usize);
        for (s, side) in ["ref", "dist"].iter().enumerate() {
            let name = format!("{stem}_{size}_{side}");
            let (px, w, h) = load_rgb8(&dir.join(format!("{name}.png")));
            dims = (w, h);
            let img = Rgb8In {
                pixels: &px,
                width: w,
                height: h,
                stride: w * 3,
            };
            if !args.get(5).is_some_and(|mode| mode == "scores-only") {
                let taps = m
                    .taps_for_export(&img)
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                for (i, (c, th, tw, data)) in taps.iter().enumerate() {
                    write_npy(
                        &out_dir.join(format!("{name}_tap{i}.npy")),
                        *c,
                        *th,
                        *tw,
                        data,
                    );
                }
            }
            pair[s] = px;
        }
        // this port's own score on the same pair — the pin baseline for
        // rows whose netlib draw lands past the 1e-3 JND gate
        fn im(px: &[u8], w: usize, h: usize) -> Image<'_> {
            Image::from(
                PixelSlice::new(px, w as u32, h as u32, w * 3, PixelDescriptor::RGB8_SRGB).unwrap(),
            )
        }
        let s = m
            .evaluate(
                im(&pair[0], dims.0, dims.1),
                im(&pair[1], dims.0, dims.1),
                EvalOpts::new(),
            )
            .expect("evaluate")
            .scores()
            .find(|(k, _)| k == "dmm")
            .unwrap()
            .1;
        writeln!(port, "{stem}\t{size}\t{s:.17e}").unwrap();
        port.flush().unwrap();
        eprintln!("{stem} {size}: {s:.17e}");
    }
    assert!(checked > 0, "no golden rows selected");
}
