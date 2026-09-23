//! Deterministic synthetic *video* corpus for the cvvdp video-path
//! conformance gate.
//!
//! A *video situation* is one `(ref_frames, dist_frames)` pair plus a
//! label, the temporal content/distortion class it exercises, and the
//! frame rate. Every situation is produced by pure, PRNG-free
//! deterministic arithmetic, so the SAME bytes are scored by the
//! Rust impl AND the pycvvdp reference: the golden builder
//! (`scripts/cvvdp_goldens/build_video_goldens.py`) scores the exact
//! PNG frame sequences `emit_video_situations` writes to
//! `/scratch/cvvdpvideo/`.
//!
//! Classes (per the video-port work order):
//!
//! - **Static** — identical frames in ref; dist carries a static
//!   spatial distortion only. Exercises the sustained channels with
//!   zero transient energy.
//! - **Global flicker** — whole-frame luminance oscillation in ref;
//!   dist flickers with a different amplitude/phase. Drives the
//!   transient channel hard.
//! - **Motion** — a scrolling edge / texture; dist scrolls at a
//!   different speed (motion judder).
//! - **Temporal noise** — iid per-frame noise in dist on a static
//!   ref.
//! - **Blocky codec artefact** — dist is 8×8-block-quantized with a
//!   per-frame per-block DC wobble.
//! - **Chroma-only flicker** — chroma oscillates in dist while luma
//!   stays static.
//! - **Brief flash** — a single-frame luminance spike in dist
//!   (maximally transient stimulus).
//! - **Identical** — `ref == dist` byte-for-byte; JOD must be 10.
//!
//! Sizes are deliberately small (64×64, 96×80, and one odd 73×85)
//! and frame counts straddle the temporal filter length both ways
//! (N = 5 < fl(30) = 9 and N = 24 > fl(60) = 17). fps ∈ {24, 30, 60}.

/// One conformance video situation: a labelled
/// `(ref_frames, dist_frames)` clip.
#[derive(Clone)]
pub struct VideoSituation {
    /// Stable identifier — used as the manifest key.
    pub name: &'static str,
    /// Content/distortion class for grouping in reports.
    pub class: VideoClass,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame rate in Hz.
    pub fps: f32,
    /// Reference RGB8 frames, each `width*height*3` bytes row-major.
    pub ref_frames: Vec<Vec<u8>>,
    /// Distorted RGB8 frames, same layout; `len == ref_frames.len()`.
    pub dist_frames: Vec<Vec<u8>>,
}

/// Grouping of video situations for reporting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoClass {
    Static,
    GlobalFlicker,
    Motion,
    TemporalNoise,
    BlockyCodec,
    ChromaFlicker,
    BriefFlash,
    Identical,
}

impl VideoClass {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            VideoClass::Static => "static",
            VideoClass::GlobalFlicker => "global_flicker",
            VideoClass::Motion => "motion",
            VideoClass::TemporalNoise => "temporal_noise",
            VideoClass::BlockyCodec => "blocky_codec",
            VideoClass::ChromaFlicker => "chroma_flicker",
            VideoClass::BriefFlash => "brief_flash",
            VideoClass::Identical => "identical",
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic frame content (PRNG-free).
// ---------------------------------------------------------------------------

/// Mid-range textured base frame: a luminance ramp plus a
/// medium-frequency pattern and a couple of moderately saturated
/// regions — enough spatial content that the spatial pyramid + CSF
/// see non-trivial input, without per-pixel chaos that would make a
/// single-pel scroll unreadable.
fn base_px(x: usize, y: usize, w: usize, h: usize) -> [u8; 3] {
    // Diagonal luma ramp ~[80, 200] + woven texture.
    let ramp = 80 + ((x * 90 / w.max(1)) as i32) + ((y * 30 / h.max(1)) as i32);
    let weave = (((x * 7 + y * 13) % 31) as i32) - 15;
    let luma = (ramp + weave / 3).clamp(20, 235);
    // Mild spatial chroma variation (mostly desaturated).
    let cr = ((x * 5 + y * 3) % 23) as i32 - 11;
    let cb = ((x * 3 + y * 5) % 19) as i32 - 9;
    [
        (luma + cr).clamp(0, 255) as u8,
        luma.clamp(0, 255) as u8,
        (luma + cb).clamp(0, 255) as u8,
    ]
}

fn base_frame(w: usize, h: usize) -> Vec<u8> {
    let mut b = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            b[i..i + 3].copy_from_slice(&base_px(x, y, w, h));
        }
    }
    b
}

/// Deterministic per-frame iid noise field via a fixed-seed
/// SplitMix64-ish LCG (same construction as `situations::noise_dist`).
/// `amp` is the peak ± perturbation in code values.
fn frame_noise(w: usize, h: usize, amp: i32, seed: u64) -> Vec<i16> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    (0..w * h * 3)
        .map(|_| (next() % ((2 * amp + 1) as u64)) as i16 - amp as i16)
        .collect()
}

/// Add a signed field to an RGB frame, clamping to [0, 255].
fn add_field(frame: &[u8], field: &[i16]) -> Vec<u8> {
    frame
        .iter()
        .zip(field.iter())
        .map(|(&v, &d)| (i32::from(v) + i32::from(d)).clamp(0, 255) as u8)
        .collect()
}

/// Uniform per-frame gain about mid-gray: `v' = 128 + (v-128)*gain`,
/// clamped. `gain` near 1.0.
fn gain_frame(frame: &[u8], gain: f32) -> Vec<u8> {
    frame
        .iter()
        .map(|&v| {
            (128.0 + (f32::from(v) - 128.0) * gain)
                .round()
                .clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Same gain but only on the R and B channels (chroma flicker keeps
/// the luma-ish G channel untouched — approximate chroma-only for a
/// synthetic fixture).
fn chroma_gain_frame(frame: &[u8], gain: f32) -> Vec<u8> {
    frame
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|p| {
            let adj = |v: u8| {
                (128.0 + (f32::from(v) - 128.0) * gain)
                    .round()
                    .clamp(0.0, 255.0) as u8
            };
            [adj(p[0]), p[1], adj(p[2])]
        })
        .collect()
}

/// Horizontally scrolled copy of `frame` (wraps), `dx` pixels.
fn scroll_x(frame: &[u8], w: usize, h: usize, dx: i32) -> Vec<u8> {
    let mut out = vec![0u8; frame.len()];
    for y in 0..h {
        for x in 0..w {
            let sx = ((x as i32 - dx).rem_euclid(w as i32)) as usize;
            let i = (y * w + x) * 3;
            let s = (y * w + sx) * 3;
            out[i..i + 3].copy_from_slice(&frame[s..s + 3]);
        }
    }
    out
}

/// A vertical-edge base frame: smooth on both sides with a sharp
/// transition at `x == edge`.
fn edge_frame(w: usize, h: usize, edge: i32) -> Vec<u8> {
    let mut b = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let v = if (x as i32) < edge { 60u8 } else { 190u8 };
            // slight vertical structure so the frame isn't degenerate
            let vv = (i32::from(v) + (((y * 3) % 11) as i32) - 5).clamp(0, 255) as u8;
            let i = (y * w + x) * 3;
            b[i] = vv;
            b[i + 1] = vv;
            b[i + 2] = vv;
        }
    }
    b
}

/// 8×8 block quantization + per-frame per-block DC wobble — the
/// "blocky codec artefact" class. Each 8×8 block collapses to its
/// mean plus a deterministic frame-varying offset.
fn blocky_frame(frame: &[u8], w: usize, h: usize, f: usize) -> Vec<u8> {
    let mut out = vec![0u8; frame.len()];
    for by in (0..h).step_by(8) {
        for bx in (0..w).step_by(8) {
            let mut acc = [0u32; 3];
            let mut cnt = 0u32;
            for y in by..(by + 8).min(h) {
                for x in bx..(bx + 8).min(w) {
                    let i = (y * w + x) * 3;
                    acc[0] += u32::from(frame[i]);
                    acc[1] += u32::from(frame[i + 1]);
                    acc[2] += u32::from(frame[i + 2]);
                    cnt += 1;
                }
            }
            // Deterministic ±9 wobble per (block, frame).
            let wob = ((bx * 31 + by * 17 + f * 13) % 19) as i32 - 9;
            for y in by..(by + 8).min(h) {
                for x in bx..(bx + 8).min(w) {
                    let i = (y * w + x) * 3;
                    for c in 0..3 {
                        out[i + c] = ((acc[c] / cnt) as i32 + wob).clamp(0, 255) as u8;
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The video situation registry.
// ---------------------------------------------------------------------------

/// Build the full conformance video corpus. All situations are
/// synthetic and always present — no corpus dependence.
#[must_use]
pub fn all_video_situations() -> Vec<VideoSituation> {
    let mut out = Vec::new();

    let mk = |name: &'static str,
              class: VideoClass,
              w: u32,
              h: u32,
              fps: f32,
              ref_frames: Vec<Vec<u8>>,
              dist_frames: Vec<Vec<u8>>|
     -> VideoSituation {
        let n = (w * h * 3) as usize;
        assert_eq!(ref_frames.len(), dist_frames.len(), "{name} frame count");
        for (k, f) in ref_frames.iter().enumerate() {
            assert_eq!(f.len(), n, "{name} ref[{k}] len");
        }
        for (k, f) in dist_frames.iter().enumerate() {
            assert_eq!(f.len(), n, "{name} dist[{k}] len");
        }
        VideoSituation {
            name,
            class,
            width: w,
            height: h,
            fps,
            ref_frames,
            dist_frames,
        }
    };

    // --- Static: identical frames; dist = static noise field. ---
    {
        let (w, h) = (64usize, 64usize);
        let base = base_frame(w, h);
        let n = 12;
        let ref_frames = vec![base.clone(); n];
        let dfield = frame_noise(w, h, 14, 0xD157);
        let dist = add_field(&base, &dfield);
        let dist_frames = vec![dist; n];
        out.push(mk(
            "vid_static_texture_64",
            VideoClass::Static,
            64,
            64,
            30.0,
            ref_frames,
            dist_frames,
        ));
    }

    // --- Global flicker at 24 fps (filter len 7) and 60 fps (17). ---
    {
        let (w, h) = (64usize, 64usize);
        let base = base_frame(w, h);
        for (fps, n, name) in [
            (24.0f32, 16usize, "vid_flicker_24"),
            (60.0, 24, "vid_flicker_60"),
        ] {
            let mut ref_frames = Vec::with_capacity(n);
            let mut dist_frames = Vec::with_capacity(n);
            for f in 0..n {
                // ~3 Hz oscillation relative to fps.
                let phase = f as f32 * 3.0 / fps * core::f32::consts::TAU;
                let g_ref = 1.0 + 0.14 * phase.sin();
                let g_dist = 1.0 + 0.10 * (phase + 0.6).sin();
                ref_frames.push(gain_frame(&base, g_ref));
                dist_frames.push(gain_frame(&base, g_dist));
            }
            out.push(mk(
                name,
                VideoClass::GlobalFlicker,
                64,
                64,
                fps,
                ref_frames,
                dist_frames,
            ));
        }
    }

    // --- Motion: scrolling edge at 30 fps (dist lags 1px) and
    //     scrolling texture at 60 fps (dist judders at half rate). ---
    {
        let (w, h) = (96usize, 80usize);
        let n = 12;
        let mut ref_frames = Vec::with_capacity(n);
        let mut dist_frames = Vec::with_capacity(n);
        for f in 0..n {
            ref_frames.push(edge_frame(w, h, 8 + 2 * f as i32));
            dist_frames.push(edge_frame(w, h, 8 + f as i32));
        }
        out.push(mk(
            "vid_edge_scroll_30",
            VideoClass::Motion,
            96,
            80,
            30.0,
            ref_frames,
            dist_frames,
        ));

        let base = base_frame(w, h);
        let n = 20;
        let mut ref_frames = Vec::with_capacity(n);
        let mut dist_frames = Vec::with_capacity(n);
        for f in 0..n {
            ref_frames.push(scroll_x(&base, w, h, f as i32));
            // Half-rate scroll — motion judder.
            dist_frames.push(scroll_x(&base, w, h, (f / 2) as i32));
        }
        out.push(mk(
            "vid_scroll_texture_60",
            VideoClass::Motion,
            96,
            80,
            60.0,
            ref_frames,
            dist_frames,
        ));
    }

    // --- Temporal noise: static ref, per-frame iid noise in dist. ---
    {
        let (w, h) = (64usize, 64usize);
        let base = base_frame(w, h);
        let n = 10;
        let ref_frames = vec![base.clone(); n];
        let dist_frames = (0..n)
            .map(|f| add_field(&base, &frame_noise(w, h, 10, 0x5EED + f as u64)))
            .collect();
        out.push(mk(
            "vid_temporal_noise_30",
            VideoClass::TemporalNoise,
            64,
            64,
            30.0,
            ref_frames,
            dist_frames,
        ));
    }

    // --- Blocky codec artefact on a slowly scrolling texture. ---
    {
        let (w, h) = (96usize, 80usize);
        let base = base_frame(w, h);
        let n = 14;
        let mut ref_frames = Vec::with_capacity(n);
        let mut dist_frames = Vec::with_capacity(n);
        for f in 0..n {
            let src = scroll_x(&base, w, h, f as i32 / 2);
            ref_frames.push(src.clone());
            dist_frames.push(blocky_frame(&src, w, h, f));
        }
        out.push(mk(
            "vid_blocky_24",
            VideoClass::BlockyCodec,
            96,
            80,
            24.0,
            ref_frames,
            dist_frames,
        ));
    }

    // --- Chroma-only flicker: luma static, chroma oscillates ~4 Hz. ---
    {
        let (w, h) = (64usize, 64usize);
        let base = base_frame(w, h);
        let n = 16;
        let fps = 30.0f32;
        let ref_frames = vec![base.clone(); n];
        let mut dist_frames = Vec::with_capacity(n);
        for f in 0..n {
            let phase = f as f32 * 4.0 / fps * core::f32::consts::TAU;
            dist_frames.push(chroma_gain_frame(&base, 1.0 + 0.18 * phase.sin()));
        }
        out.push(mk(
            "vid_chroma_flicker_30",
            VideoClass::ChromaFlicker,
            64,
            64,
            30.0,
            ref_frames,
            dist_frames,
        ));
    }

    // --- Brief flash: single-frame +30 luminance spike in dist. ---
    {
        let (w, h) = (64usize, 64usize);
        let base = base_frame(w, h);
        let n = 20;
        let ref_frames = vec![base.clone(); n];
        let mut dist_frames = vec![base.clone(); n];
        dist_frames[10] = gain_frame(&base, 1.35);
        out.push(mk(
            "vid_brief_flash_60",
            VideoClass::BriefFlash,
            64,
            64,
            60.0,
            ref_frames,
            dist_frames,
        ));
    }

    // --- Short clip at odd size below the 30 fps filter length (9). ---
    {
        let (w, h) = (73usize, 85usize);
        let base = base_frame(w, h);
        let n = 5;
        let ref_frames = (0..n).map(|f| scroll_x(&base, w, h, f as i32)).collect();
        let mut dist_frames = Vec::with_capacity(n);
        for f in 0..n {
            let src = scroll_x(&base, w, h, f as i32);
            dist_frames.push(add_field(&src, &frame_noise(w, h, 8, 0x0DD + f as u64)));
        }
        out.push(mk(
            "vid_short_clip_odd",
            VideoClass::TemporalNoise,
            73,
            85,
            30.0,
            ref_frames,
            dist_frames,
        ));
    }

    // --- Identical ref/dist: sanity cell, JOD must be 10. ---
    {
        let (w, h) = (64usize, 64usize);
        let base = base_frame(w, h);
        let n = 8;
        let frames = (0..n)
            .map(|f| gain_frame(&base, 1.0 + 0.1 * (f as f32 * 0.7).sin()))
            .collect::<Vec<_>>();
        out.push(mk(
            "vid_identical_30",
            VideoClass::Identical,
            64,
            64,
            30.0,
            frames.clone(),
            frames,
        ));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn video_situations_are_well_formed() {
        for s in all_video_situations() {
            assert!(s.width >= 8 && s.height >= 8, "{} too small", s.name);
            assert_eq!(s.ref_frames.len(), s.dist_frames.len(), "{}", s.name);
            assert!(s.ref_frames.len() >= 2, "{} needs >= 2 frames", s.name);
            let n = (s.width * s.height * 3) as usize;
            for f in &s.ref_frames {
                assert_eq!(f.len(), n, "{} ref len", s.name);
            }
            for f in &s.dist_frames {
                assert_eq!(f.len(), n, "{} dist len", s.name);
            }
        }
    }

    #[test]
    fn video_situation_names_unique() {
        let mut names: Vec<&str> = all_video_situations().iter().map(|s| s.name).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "duplicate video situation name");
    }

    #[test]
    fn video_corpus_covers_required_cells() {
        let sits = all_video_situations();
        // fps coverage {24, 30, 60}
        for fps in [24.0f32, 30.0, 60.0] {
            assert!(
                sits.iter().any(|s| s.fps == fps),
                "no situation at fps {fps}"
            );
        }
        // Frame counts both below and above every filter length used:
        // fl(24)=7, fl(30)=9, fl(60)=17 — so N=5 is below all, N=24
        // is above all.
        assert!(sits.iter().any(|s| s.ref_frames.len() < 7));
        assert!(sits.iter().any(|s| s.ref_frames.len() > 17));
        // Odd dimension present.
        assert!(sits.iter().any(|s| s.width % 2 == 1 || s.height % 2 == 1));
        // Every class present.
        for c in [
            VideoClass::Static,
            VideoClass::GlobalFlicker,
            VideoClass::Motion,
            VideoClass::TemporalNoise,
            VideoClass::BlockyCodec,
            VideoClass::ChromaFlicker,
            VideoClass::BriefFlash,
            VideoClass::Identical,
        ] {
            assert!(sits.iter().any(|s| s.class == c), "missing class {c:?}");
        }
    }
}
