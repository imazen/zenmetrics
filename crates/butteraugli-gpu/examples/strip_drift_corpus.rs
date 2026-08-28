//! Corpus-scale strip-vs-whole drift measurement for the MULTIRES
//! butteraugli walker (zenmetrics#47 item 4).
//!
//! The question this answers: is the `new_multires_strip` vs
//! `new_multires` relative error **bounded** at ~1e-4, or does it
//! GROW with distortion magnitude / resolution / content class?
//! `benchmarks/butter_strip_halo_2026-05-31.md` established the
//! ~1e-4 figure on ONE synthetic HF checkerboard at 512²/1024²;
//! this harness re-measures it on a real corpus × a JPEG quality
//! ladder that spans tiny and huge butteraugli deltas.
//!
//! Usage (env-driven so it stays a plain `cargo run --example`):
//!
//! ```text
//! DRIFT_MANIFEST=/path/to/manifest.tsv \
//! DRIFT_OUT=/path/to/out.csv \
//! DRIFT_QUALITIES=95,85,70,50,30,15 \
//! DRIFT_BODIES=auto,128,256 \
//! cargo run --release -p butteraugli-gpu \
//!   --no-default-features --features wgpu,cubecl-types \
//!   --example strip_drift_corpus
//! ```
//!
//! `DRIFT_MANIFEST` is a headerless TSV of `class<TAB>path` rows.
//! `DRIFT_SCALES` (optional, e.g. `256,512,1024,2048`) additionally
//! runs each image resampled to that long-edge size (Lanczos3), which
//! isolates the resolution axis from the content axis.
//!
//! Output CSV columns are documented in `header()` below.

use butteraugli_gpu::{
    Butteraugli, ButteraugliOpaque, ButteraugliParams, MemoryMode, ResolvedMode,
};
use cubecl::Runtime;
use image::codecs::jpeg::JpegEncoder;
use std::io::Write;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
type Backend = cubecl::cpu::CpuRuntime;

#[cfg(feature = "cuda")]
const BACKEND_NAME: &str = "cubecl-cuda";
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const BACKEND_NAME: &str = "cubecl-wgpu";
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
const BACKEND_NAME: &str = "cubecl-cpu";

#[cfg(feature = "cuda")]
const BACKEND_E: butteraugli_gpu::Backend = butteraugli_gpu::Backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const BACKEND_E: butteraugli_gpu::Backend = butteraugli_gpu::Backend::Wgpu;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
const BACKEND_E: butteraugli_gpu::Backend = butteraugli_gpu::Backend::CubeclCpu;

fn header() -> &'static str {
    "class,image,width,height,n_pixels,megapixels,scale_label,quality,\
     jpeg_bytes,body_label,body_h,n_strips,whole_score,strip_score,\
     abs_score,rel_score,whole_p3,strip_p3,abs_p3,rel_p3,\
     singleres_whole_score,rel_singleres_control,\
     opaque_full,opaque_auto,rel_opaque,\
     cpu_whole_score,cpu_strip_score,rel_cpu_strip_vs_cpu_whole,rel_gpu_whole_vs_cpu_whole"
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn rel(want: f64, got: f64) -> f64 {
    (got - want).abs() / want.abs().max(1e-12)
}

/// Encode `rgb` as JPEG at `quality`, decode it back, return
/// `(decoded_rgb8, encoded_byte_len)`.
fn jpeg_roundtrip(rgb: &[u8], w: u32, h: u32, quality: u8) -> (Vec<u8>, usize) {
    let mut encoded: Vec<u8> = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut encoded, quality);
        enc.encode(rgb, w, h, image::ExtendedColorType::Rgb8)
            .expect("jpeg encode");
    }
    let decoded = image::load_from_memory_with_format(&encoded, image::ImageFormat::Jpeg)
        .expect("jpeg decode")
        .to_rgb8();
    let n = encoded.len();
    (decoded.into_raw(), n)
}

fn load_rgb8(path: &str) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::ImageReader::open(path)
        .ok()?
        .decode()
        .ok()?
        .to_rgb8();
    let (w, h) = (img.width(), img.height());
    Some((img.into_raw(), w, h))
}

/// Lanczos3 downscale to `long_edge` on the longer side (never upscales
/// — synthetic upsampling has no high-frequency content and would
/// bias any size-dependence conclusion).
fn resize_long_edge(rgb: &[u8], w: u32, h: u32, long_edge: u32) -> Option<(Vec<u8>, u32, u32)> {
    let cur_long = w.max(h);
    if long_edge >= cur_long {
        return None;
    }
    let scale = long_edge as f64 / cur_long as f64;
    let nw = ((w as f64 * scale).round() as u32).max(1);
    let nh = ((h as f64 * scale).round() as u32).max(1);
    let img = image::RgbImage::from_raw(w, h, rgb.to_vec())?;
    let out = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Lanczos3);
    Some((out.into_raw(), nw, nh))
}

/// Flat interleaved RGB8 → the CPU crate's `ImgVec<RGB8>` input shape.
fn to_imgvec(buf: &[u8], w: u32, h: u32) -> imgref::ImgVec<rgb::RGB8> {
    let px: Vec<rgb::RGB8> = buf
        .chunks_exact(3)
        .map(|c| rgb::RGB8 {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect();
    imgref::ImgVec::new(px, w as usize, h as usize)
}

struct Cell {
    class: String,
    name: String,
    scale_label: String,
    rgb: Vec<u8>,
    w: u32,
    h: u32,
}

fn main() {
    let manifest_path = env_or("DRIFT_MANIFEST", "");
    assert!(!manifest_path.is_empty(), "set DRIFT_MANIFEST");
    let out_path = env_or("DRIFT_OUT", "");
    assert!(!out_path.is_empty(), "set DRIFT_OUT");

    let qualities: Vec<u8> = env_or("DRIFT_QUALITIES", "95,85,70,50,30,15")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let body_specs: Vec<String> = env_or("DRIFT_BODIES", "auto,128,256")
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let scales: Vec<u32> = env_or("DRIFT_SCALES", "")
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let max_mp: f64 = env_or("DRIFT_MAX_MP", "8.0").parse().unwrap_or(8.0);
    let cpu_leg = env_or("DRIFT_CPU", "0") == "1";
    // The orchestrator's CPU adapter walks butter at 256 body rows
    // (`cpu_adapter.rs`: "256 rows for ssim2/butter/zensim").
    let cpu_strip_rows: u32 = env_or("DRIFT_CPU_STRIP_ROWS", "256").parse().unwrap_or(256);
    let cpu_params = butteraugli::ButteraugliParams::default();

    let manifest = std::fs::read_to_string(&manifest_path).expect("read manifest");
    let mut out = std::fs::File::create(&out_path).expect("create out csv");
    writeln!(out, "{}", header()).unwrap();

    let client = Backend::client(&Default::default());
    let cap = butteraugli_gpu::vram_cap_bytes();
    let params = ButteraugliParams::default();

    eprintln!("# backend={BACKEND_NAME} vram_cap_bytes={cap}");
    eprintln!("# qualities={qualities:?} bodies={body_specs:?} scales={scales:?}");

    let mut rows = 0usize;
    for line in manifest.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split('\t');
        let class = parts.next().unwrap_or("unknown").to_string();
        let path = match parts.next() {
            Some(p) => p.to_string(),
            None => continue,
        };
        let Some((src_rgb, src_w, src_h)) = load_rgb8(&path) else {
            eprintln!("!! skip (decode failed): {path}");
            continue;
        };
        let name = std::path::Path::new(&path)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| path.clone());

        // Build the (native + optional resampled) cells for this image.
        let mut cells: Vec<Cell> = Vec::new();
        if (src_w as f64 * src_h as f64) / 1e6 <= max_mp {
            cells.push(Cell {
                class: class.clone(),
                name: name.clone(),
                scale_label: "native".to_string(),
                rgb: src_rgb.clone(),
                w: src_w,
                h: src_h,
            });
        } else {
            eprintln!("!! skip native (>{max_mp} MP): {name} {src_w}x{src_h}");
        }
        for &s in &scales {
            if let Some((r, w, h)) = resize_long_edge(&src_rgb, src_w, src_h, s) {
                cells.push(Cell {
                    class: class.clone(),
                    name: name.clone(),
                    scale_label: format!("le{s}"),
                    rgb: r,
                    w,
                    h,
                });
            }
        }

        for cell in &cells {
            let (w, h) = (cell.w, cell.h);
            let n_pixels = (w as u64) * (h as u64);
            let mp = n_pixels as f64 / 1e6;

            // Resolve the bodies for this geometry once.
            let mut bodies: Vec<(String, u32)> = Vec::new();
            for spec in &body_specs {
                if spec == "auto" {
                    match butteraugli_gpu::memory_mode::resolve_auto(w, h, cap) {
                        Ok(ResolvedMode::Strip { h_body }) => {
                            bodies.push(("auto".to_string(), h_body))
                        }
                        // Auto would run whole-image here, so there is no
                        // strip drift to measure on the production path.
                        Ok(ResolvedMode::Full) => {
                            eprintln!("   auto=Full at {w}x{h} — no strip cell emitted")
                        }
                        Err(e) => eprintln!("   auto resolve failed at {w}x{h}: {e}"),
                    }
                } else if let Ok(b) = spec.parse::<u32>() {
                    // A body >= image height degenerates to one strip,
                    // which is not an interesting drift cell but is a
                    // useful control; keep it only if it actually
                    // produces >= 2 strips OR it is the smallest body.
                    bodies.push((format!("b{b}"), b.min(h)));
                }
            }
            if bodies.is_empty() {
                continue;
            }

            // One whole-image instance + one instance per body, reused
            // across the whole quality ladder (dims are constant).
            let mut whole = Butteraugli::<Backend>::new_multires(client.clone(), w, h);
            // NEGATIVE CONTROL: the SINGLE-resolution whole-image score.
            // This is what `ButteraugliOpaque`'s Strip arm used to
            // compute before the multires-strip fix; it diverges from
            // multires by ~8-30 %. Emitting it per row proves the
            // harness is sensitive enough to see a real divergence —
            // a rel_score column that is all zeros is only meaningful
            // next to a control column that is not.
            let mut single = Butteraugli::<Backend>::new(client.clone(), w, h);
            // PRODUCTION SURFACE: `ButteraugliOpaque` is what the
            // orchestrator constructs. `MemoryMode::Full` is what the
            // legacy CLI path effectively runs (`new_multires`);
            // `MemoryMode::Auto` is what the orchestrator resolves
            // (strip-preferred). `rel_opaque` is therefore the exact
            // legacy-vs-orchestrator score delta for butteraugli.
            let mut op_full =
                ButteraugliOpaque::new_with_memory_mode(BACKEND_E, w, h, params, MemoryMode::Full)
                    .ok();
            let mut op_auto =
                ButteraugliOpaque::new_with_memory_mode(BACKEND_E, w, h, params, MemoryMode::Auto)
                    .ok();
            let mut strips: Vec<(String, u32, Butteraugli<Backend>)> = bodies
                .iter()
                .map(|(label, b)| {
                    (
                        label.clone(),
                        *b,
                        Butteraugli::<Backend>::new_multires_strip(client.clone(), w, h, *b),
                    )
                })
                .collect();

            for &q in &qualities {
                let (dist, jbytes) = jpeg_roundtrip(&cell.rgb, w, h, q);
                let wr = match whole.compute_with_options(&cell.rgb, &dist, &params) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("!! whole failed {} {w}x{h} q{q}: {e}", cell.name);
                        continue;
                    }
                };
                let single_score = single
                    .compute_with_options(&cell.rgb, &dist, &params)
                    .map(|r| r.score as f64)
                    .unwrap_or(f64::NAN);
                let opaque_full = op_full
                    .as_mut()
                    .and_then(|o| o.compute_srgb_u8(&cell.rgb, &dist).ok())
                    .map(|s| s.value)
                    .unwrap_or(f64::NAN);
                let opaque_auto = op_auto
                    .as_mut()
                    .and_then(|o| o.compute_srgb_u8(&cell.rgb, &dist).ok())
                    .map(|s| s.value)
                    .unwrap_or(f64::NAN);
                // OPTIONAL CPU LEG (`DRIFT_CPU=1`). Two extra facts the
                // orchestrator decision needs but the GPU-only columns
                // cannot supply:
                //   * `cpu_whole` vs `cpu_strip` — the CPU crate's own
                //     whole-vs-strip drift (the orchestrator's CPU
                //     adapter runs `butteraugli_strip`).
                //   * `gpu_whole` vs `cpu_whole` — the CPU↔GPU backend
                //     envelope, which the orchestrator's chooser can
                //     cross independently of any strip decision.
                // CPU butteraugli is ~seconds/MP, so this is off by
                // default and meant for a small ladder subset.
                let (cpu_whole, cpu_strip) = if cpu_leg {
                    let r = to_imgvec(&cell.rgb, w, h);
                    let d = to_imgvec(&dist, w, h);
                    let whole_cpu = butteraugli::butteraugli(r.as_ref(), d.as_ref(), &cpu_params)
                        .map(|x| x.score)
                        .unwrap_or(f64::NAN);
                    let strip_cpu = butteraugli::butteraugli_strip(
                        r.as_ref(),
                        d.as_ref(),
                        &cpu_params,
                        cpu_strip_rows,
                    )
                    .map(|x| x.score)
                    .unwrap_or(f64::NAN);
                    (whole_cpu, strip_cpu)
                } else {
                    (f64::NAN, f64::NAN)
                };
                for (label, body, inst) in strips.iter_mut() {
                    let sr = match inst.compute_strip_with_options(&cell.rgb, &dist, &params) {
                        Ok(r) => r,
                        Err(e) => {
                            eprintln!("!! strip failed {} {w}x{h} q{q} {label}: {e}", cell.name);
                            continue;
                        }
                    };
                    // The constructor rounds body_h down to even and
                    // clamps to image height; mirror that here so
                    // n_strips is the real dispatch count.
                    let body_eff = ((*body / 2).max(1) * 2).min(h);
                    let n_strips = h.div_ceil(body_eff);
                    let (ws, ss) = (wr.score as f64, sr.score as f64);
                    let (wp, sp) = (wr.pnorm_3 as f64, sr.pnorm_3 as f64);
                    writeln!(
                        out,
                        "{},{},{},{},{},{:.6},{},{},{},{},{},{},{:.9},{:.9},{:.3e},{:.3e},{:.9},{:.9},{:.3e},{:.3e},{:.9},{:.3e},{:.9},{:.9},{:.3e},{:.9},{:.9},{:.3e},{:.3e}",
                        cell.class,
                        cell.name,
                        w,
                        h,
                        n_pixels,
                        mp,
                        cell.scale_label,
                        q,
                        jbytes,
                        label,
                        body_eff,
                        n_strips,
                        ws,
                        ss,
                        (ss - ws).abs(),
                        rel(ws, ss),
                        wp,
                        sp,
                        (sp - wp).abs(),
                        rel(wp, sp),
                        single_score,
                        rel(ws, single_score),
                        opaque_full,
                        opaque_auto,
                        rel(opaque_full, opaque_auto),
                        cpu_whole,
                        cpu_strip,
                        rel(cpu_whole, cpu_strip),
                        rel(cpu_whole, ws),
                    )
                    .unwrap();
                    rows += 1;
                }
                out.flush().unwrap();
            }
            eprintln!(
                "ok {} [{}] {w}x{h} ({mp:.2} MP) bodies={:?} rows={rows}",
                cell.name,
                cell.scale_label,
                bodies
                    .iter()
                    .map(|(l, b)| (l.as_str(), *b))
                    .collect::<Vec<_>>()
            );
        }
    }

    eprintln!("# wrote {rows} rows to {out_path}");
}
