//! Contract checks every [`Metric`] implementation should pass.
//!
//! Call [`assert_conforms`] from each metric crate's tests with a
//! representative non-identical pair. It checks:
//!
//! - identity: column names non-empty, sanitized and unique; outputs
//!   declared;
//! - `accepts` holds for the pair it is given;
//! - `score()` is the first of `scores()`, whose keys are declared outputs
//!   in declared order of the headline;
//! - requesting a map never changes a score (bit for bit);
//! - map sizes match their geometry, and geometry describes the input;
//! - a subsampled request serves the subsampled geometry and
//!   `map_score` still folds the native map;
//! - a warm reference scores bit-identically to cold evaluation;
//! - a stateless sequence returns one result per push equal to
//!   `evaluate`;
//! - a stopped pulse yields [`ErrorKind::Stopped`].

use alloc::borrow::Cow;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use how_far::{StopOnly, StopReason};

use crate::error::ErrorKind;
use crate::eval::Evaluation;
use crate::identity::is_sanitized_column;
use crate::input::Image;
use crate::map::{MapFold, MapRequest};
use crate::metric::{AlwaysStop, EvalOpts, Metric, SequenceInfo, Temporal};

/// One failed check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    /// Which check.
    pub check: &'static str,
    /// What went wrong.
    pub detail: String,
}

fn bits_eq(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn scores_of(e: &dyn Evaluation) -> Vec<(Cow<'static, str>, f64)> {
    e.scores().collect()
}

fn same_scores(a: &[(Cow<'static, str>, f64)], b: &[(Cow<'static, str>, f64)]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|((ka, va), (kb, vb))| ka == kb && bits_eq(*va, *vb))
}

/// Run every check; an empty result means the metric conforms.
pub fn check(metric: &mut dyn Metric, r: &Image<'_>, d: &Image<'_>) -> Vec<Violation> {
    let mut out = Vec::new();
    let mut v = |check: &'static str, detail: String| out.push(Violation { check, detail });

    // Identity and declared outputs.
    let id = metric.id();
    let outputs = metric.outputs();
    if outputs.is_empty() {
        v("outputs", "no outputs declared".into());
    }
    let cols = metric.columns();
    for (i, c) in cols.iter().enumerate() {
        if !is_sanitized_column(c) {
            v("columns", format!("column {c:?} is not [a-z0-9_]+"));
        }
        if cols[..i].contains(c) {
            v("columns", format!("duplicate column {c:?}"));
        }
    }
    for s in outputs.iter() {
        if !is_sanitized_column(&s.key) {
            v(
                "outputs",
                format!("output key {:?} is not [a-z0-9_]+", s.key),
            );
        }
    }
    if id.family().is_empty() {
        v("id", "empty family".into());
    }
    if metric.score_kind() != outputs.first().map_or(metric.score_kind(), |s| s.kind) {
        v(
            "score_kind",
            "score_kind() differs from the first output's kind".into(),
        );
    }

    // Formats.
    for (name, img) in [("reference", r), ("distorted", d)] {
        if !metric.accepts(&img.format()) {
            v(
                "accepts",
                format!("{name} format {:?} not accepted", img.format()),
            );
        }
    }

    // Score-only evaluation.
    let cold = match metric.evaluate(r.clone(), d.clone(), EvalOpts::SCORE_ONLY) {
        Ok(e) => e,
        Err(e) => {
            v("evaluate", format!("score-only evaluate failed: {e}"));
            return out;
        }
    };
    let cold_scores = scores_of(&*cold);
    match cold_scores.first() {
        None => v("scores", "scores() is empty".into()),
        Some((k, val)) => {
            if !bits_eq(*val, cold.score()) {
                v(
                    "scores",
                    format!("score() {} != first scores() value {val}", cold.score()),
                );
            }
            if outputs.first().is_some_and(|s| s.key != *k) {
                v(
                    "scores",
                    format!("headline key {k:?} is not the first declared output"),
                );
            }
        }
    }
    for (k, _) in &cold_scores {
        if !outputs.iter().any(|s| s.key == *k) {
            v("scores", format!("undeclared output key {k:?}"));
        }
    }
    if cold.map().is_some() {
        v("map", "map produced without a map request".into());
    }

    // Native maps.
    match metric.evaluate(r.clone(), d.clone(), EvalOpts::SCORE_ONLY.with_native_map()) {
        Err(e) if e.error().kind() == ErrorKind::NoMap => {}
        Err(e) => v("map", format!("native-map evaluate failed: {e}")),
        Ok(native) => {
            if !same_scores(&cold_scores, &scores_of(&*native)) {
                v("map", "requesting a map changed the scores".into());
            }
            let maps = native.maps();
            if maps.is_empty() {
                v("map", "map requested, Ok returned, no map".into());
            }
            for m in &maps {
                let g = m.geometry;
                if (g.input_width, g.input_height) != (r.width(), r.height()) {
                    v(
                        "geometry",
                        format!(
                            "map {}: geometry input {}x{} != image",
                            m.name, g.input_width, g.input_height
                        ),
                    );
                }
                let sl = m.pixels.as_slice();
                if (sl.width(), sl.rows()) != (g.cells_x, g.cells_y) {
                    v(
                        "geometry",
                        format!(
                            "map {}: {}x{} values, geometry {}x{}",
                            m.name,
                            sl.width(),
                            sl.rows(),
                            g.cells_x,
                            g.cells_y
                        ),
                    );
                }
                if m.served != MapRequest::Native {
                    v(
                        "map",
                        format!("map {}: served {:?} for a native request", m.name, m.served),
                    );
                }
            }
            let native_mean = native.map_score(MapFold::Mean);

            // Subsampled maps.
            match metric.evaluate(
                r.clone(),
                d.clone(),
                EvalOpts::SCORE_ONLY.with_map(MapRequest::subsampled(2)),
            ) {
                Err(e) => v("subsampled", format!("subsampled evaluate failed: {e}")),
                Ok(sub) => {
                    if !same_scores(&cold_scores, &scores_of(&*sub)) {
                        v(
                            "subsampled",
                            "subsampled map request changed the scores".into(),
                        );
                    }
                    for (s, n) in sub.maps().iter().zip(&maps) {
                        if s.geometry != n.geometry.subsampled(2) {
                            v(
                                "subsampled",
                                format!(
                                    "map {}: geometry {:?} != native.subsampled(2)",
                                    s.name, s.geometry
                                ),
                            );
                        }
                        let sl = s.pixels.as_slice();
                        if (sl.width(), sl.rows()) != (s.geometry.cells_x, s.geometry.cells_y) {
                            v(
                                "subsampled",
                                format!("map {}: size disagrees with geometry", s.name),
                            );
                        }
                    }
                    let sub_mean = sub.map_score(MapFold::Mean);
                    match (&native_mean, &sub_mean) {
                        (Ok(a), Ok(b)) if bits_eq(*a, *b) => {}
                        (a, b) => v(
                            "subsampled",
                            format!("map_score(Mean) native {a:?} vs subsampled {b:?}"),
                        ),
                    }
                }
            }
        }
    }

    // Warm reference.
    match metric.warm_reference(r.clone(), EvalOpts::SCORE_ONLY) {
        Err(e) => v("warm", format!("warm_reference failed: {e}")),
        Ok(mut warm) => {
            for round in 0..2 {
                match warm.evaluate(d.clone(), EvalOpts::SCORE_ONLY) {
                    Ok(e) if same_scores(&cold_scores, &scores_of(&*e)) => {}
                    Ok(_) => v(
                        "warm",
                        format!("warm candidate {round} differs from cold evaluate"),
                    ),
                    Err(e) => v("warm", format!("warm candidate {round} failed: {e}")),
                }
            }
        }
    }

    // Stateless sequences.
    if metric.temporal() == Temporal::Stateless {
        match metric.sequence(SequenceInfo::new()) {
            Err(e) => v("sequence", format!("sequence failed: {e}")),
            Ok(mut seq) => match seq.push(r.clone(), d.clone(), EvalOpts::SCORE_ONLY) {
                Err(e) => v("sequence", format!("push failed: {e}")),
                Ok(frames) => {
                    if frames.len() != 1 || frames[0].index != 0 {
                        v("sequence", format!("stateless push returned {frames:?}"));
                    } else if !same_scores(&cold_scores, &scores_of(&*frames[0].evaluation)) {
                        v("sequence", "sequence frame differs from evaluate".into());
                    }
                    if seq.frames_pushed() != 1 {
                        v("sequence", format!("frames_pushed {}", seq.frames_pushed()));
                    }
                    match seq.finish(EvalOpts::SCORE_ONLY) {
                        Ok(s) if s.frame_count == 1 && s.frames.is_empty() => {}
                        Ok(s) => v(
                            "sequence",
                            format!(
                                "finish: {} frames held, count {}",
                                s.frames.len(),
                                s.frame_count
                            ),
                        ),
                        Err(e) => v("sequence", format!("finish failed: {e}")),
                    }
                }
            },
        }
    }

    // Cancellation.
    let stopped = StopOnly::new(AlwaysStop(StopReason::Cancelled));
    match metric.evaluate(
        r.clone(),
        d.clone(),
        EvalOpts::SCORE_ONLY.with_pulse(&stopped),
    ) {
        Err(e) if e.error().kind() == ErrorKind::Stopped(StopReason::Cancelled) => {}
        Err(e) => v(
            "stop",
            format!(
                "stopped pulse gave {:?}, not Stopped(Cancelled)",
                e.error().kind()
            ),
        ),
        Ok(_) => v("stop", "stopped pulse still produced a result".into()),
    }

    out
}

/// `(key, value)` scores collected from one evaluation.
pub type NamedScores = Vec<(Cow<'static, str>, f64)>;

/// What [`check_sequence`] observed.
#[derive(Debug, Default)]
pub struct SequenceReport {
    /// Failed checks; empty when the session conforms.
    pub violations: Vec<Violation>,
    /// Every frame result in arrival order: `(index, scores)`.
    pub frames: Vec<(u64, NamedScores)>,
    /// The pooled result's scores, when the metric pools.
    pub pooled: Option<NamedScores>,
}

/// Drive a full sequence over `pairs` and check the [`crate::Sequence`]
/// contract: every pushed frame's result arrives exactly once, in index
/// order, no later than [`Temporal::Stateful`]'s `latency` pushes after
/// its own (immediately for [`Temporal::Stateless`]); a
/// [`Temporal::Pooled`] metric returns no frame results and a pooled one;
/// `frames_pushed` and the summary's `frame_count` agree; each result's
/// `score()` is its first `scores()` value. `info`'s frame count is set to
/// `pairs.len()`.
pub fn check_sequence(
    metric: &mut dyn Metric,
    info: SequenceInfo,
    pairs: &[(Image<'_>, Image<'_>)],
) -> SequenceReport {
    let mut rep = SequenceReport::default();
    let temporal = metric.temporal();
    let latency = match temporal {
        Temporal::Stateless => 0,
        Temporal::Stateful { latency, .. } => u64::from(latency),
        Temporal::Pooled => 0,
    };
    let per_frame = temporal != Temporal::Pooled;
    let mut seq = match metric.sequence(info.with_frame_count(pairs.len() as u64)) {
        Ok(s) => s,
        Err(e) => {
            rep.violations.push(Violation {
                check: "sequence",
                detail: format!("sequence failed: {e}"),
            });
            return rep;
        }
    };
    let accept = |rep: &mut SequenceReport,
                  pushed: u64,
                  frames: Vec<crate::metric::FrameEvaluation>,
                  finishing: bool| {
        for f in frames {
            let expected = rep.frames.len() as u64;
            if f.index != expected {
                rep.violations.push(Violation {
                    check: "sequence",
                    detail: format!("got frame {} where {expected} was due", f.index),
                });
            }
            if !finishing && pushed.saturating_sub(1).saturating_sub(f.index) > latency {
                rep.violations.push(Violation {
                    check: "sequence",
                    detail: format!(
                        "frame {} arrived after push {} (latency {latency})",
                        f.index,
                        pushed - 1
                    ),
                });
            }
            let s = scores_of(&*f.evaluation);
            if s.first()
                .is_none_or(|(_, v)| !bits_eq(*v, f.evaluation.score()))
            {
                rep.violations.push(Violation {
                    check: "sequence",
                    detail: format!("frame {}: score() is not the first scores() value", f.index),
                });
            }
            rep.frames.push((f.index, s));
        }
    };
    for (i, (r, d)) in pairs.iter().enumerate() {
        match seq.push(r.clone(), d.clone(), EvalOpts::SCORE_ONLY) {
            Ok(frames) => accept(&mut rep, i as u64 + 1, frames, false),
            Err(e) => {
                rep.violations.push(Violation {
                    check: "sequence",
                    detail: format!("push {i} failed: {e}"),
                });
                return rep;
            }
        }
        if seq.frames_pushed() != i as u64 + 1 {
            rep.violations.push(Violation {
                check: "sequence",
                detail: format!(
                    "frames_pushed {} after {} pushes",
                    seq.frames_pushed(),
                    i + 1
                ),
            });
        }
    }
    match seq.finish(EvalOpts::SCORE_ONLY) {
        Ok(summary) => {
            if summary.frame_count != pairs.len() as u64 {
                rep.violations.push(Violation {
                    check: "sequence",
                    detail: format!(
                        "frame_count {} for {} pushes",
                        summary.frame_count,
                        pairs.len()
                    ),
                });
            }
            accept(&mut rep, pairs.len() as u64, summary.frames, true);
            rep.pooled = summary.pooled.as_deref().map(scores_of);
        }
        Err(e) => rep.violations.push(Violation {
            check: "sequence",
            detail: format!("finish failed: {e}"),
        }),
    }
    if !per_frame {
        if !rep.frames.is_empty() {
            rep.violations.push(Violation {
                check: "sequence",
                detail: format!("pooled metric returned {} frame results", rep.frames.len()),
            });
        }
        if rep.pooled.is_none() {
            rep.violations.push(Violation {
                check: "sequence",
                detail: "pooled metric returned no pooled result".into(),
            });
        }
    } else if rep.frames.len() != pairs.len() {
        rep.violations.push(Violation {
            check: "sequence",
            detail: format!("{} results for {} frames", rep.frames.len(), pairs.len()),
        });
    }
    rep
}

/// Panic with every violation when `metric` does not conform.
///
/// # Panics
/// As described.
#[track_caller]
pub fn assert_conforms(metric: &mut dyn Metric, r: &Image<'_>, d: &Image<'_>) {
    let violations = check(metric, r, d);
    if !violations.is_empty() {
        let mut msg = format!("{} does not conform:", metric.id());
        for x in &violations {
            msg.push_str(&format!("\n  [{}] {}", x.check, x.detail));
        }
        panic!("{msg}");
    }
}

/// Deterministic textured sRGB8 test content (`w·h·3` bytes): gradients,
/// a checker of hard edges and LCG noise, so filters see structure at
/// every scale.
pub fn textured_rgb8(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3usize {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (s >> 27) as usize;
                let grad = (x * 255 / w.max(1) + y * 128 / h.max(1) + c * 40) % 200;
                let edge = if (x / 7 + y / 5) % 2 == 0 { 45 } else { 0 };
                out.push((grad + edge + noise).min(255) as u8);
            }
        }
    }
    out
}

/// A distorted copy of [`textured_rgb8`] content: quantized, with a
/// bright block, so every metric sees a non-identical pair.
pub fn distort_rgb8(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out: Vec<u8> = src.iter().map(|&v| (v / 12) * 12).collect();
    for y in h / 4..h / 2 {
        for x in w / 4..w / 2 {
            for c in 0..3 {
                out[(y * w + x) * 3 + c] = 250;
            }
        }
    }
    out
}

/// Wrap packed sRGB8 bytes as an [`Image`].
///
/// # Panics
/// When `data` is shorter than `w·h·3`.
pub fn rgb8_image(data: &[u8], w: usize, h: usize) -> Image<'_> {
    zenpixels::PixelSlice::new(
        data,
        w as u32,
        h as u32,
        w * 3,
        zenpixels::PixelDescriptor::RGB8_SRGB,
    )
    .expect("packed rgb8 test image")
    .into()
}

/// [`assert_conforms`] on a [`textured_rgb8`] / [`distort_rgb8`] pair of
/// size `w × h`.
///
/// # Panics
/// As [`assert_conforms`].
#[track_caller]
pub fn assert_conforms_rgb8(metric: &mut dyn Metric, w: usize, h: usize) {
    let a = textured_rgb8(w, h, 1);
    let b = distort_rgb8(&a, w, h);
    assert_conforms(metric, &rgb8_image(&a, w, h), &rgb8_image(&b, w, h));
}

/// A boxed-result helper for metric tests: the scores of one evaluation.
pub fn scores(e: &dyn Evaluation) -> Vec<(Cow<'static, str>, f64)> {
    scores_of(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MetricResult;
    use crate::eval::{ScalarEval, ScoreKind, ScoreSpec};
    use crate::identity::{Implementation, MetricId};
    use crate::input::{InputFormat, is_rgb8_srgb, rgb8_pair};
    use crate::metric::EvalOpts;
    use alloc::boxed::Box;
    use alloc::vec;
    use zenpixels::{PixelDescriptor, PixelSlice};

    /// Mean absolute byte difference; no map.
    struct Mad8 {
        forget_checkpoint: bool,
    }

    impl Metric for Mad8 {
        fn id(&self) -> MetricId {
            MetricId::new("mad8", Implementation::new("test", "0", 1))
        }
        fn outputs(&self) -> Cow<'static, [crate::eval::ScoreSpec]> {
            const O: &[ScoreSpec] = &[ScoreSpec::new("mad", ScoreKind::Difference)];
            Cow::Borrowed(O)
        }
        fn accepts(&self, f: &InputFormat) -> bool {
            is_rgb8_srgb(f)
        }
        fn evaluate(
            &mut self,
            r: Image<'_>,
            d: Image<'_>,
            opts: EvalOpts<'_>,
        ) -> MetricResult<Box<dyn Evaluation>> {
            if !self.forget_checkpoint {
                opts.checkpoint()?;
            }
            opts.refuse_map("mad8")?;
            let (a, b) = rgb8_pair(&r, &d)?;
            let s: f64 = a
                .pixels
                .iter()
                .zip(b.pixels)
                .map(|(x, y)| f64::from(x.abs_diff(*y)))
                .sum();
            Ok(Box::new(ScalarEval::single(
                "mad",
                s / a.pixels.len() as f64,
            )))
        }
    }

    #[test]
    fn stateless_sequence_conforms() {
        let imgs: Vec<_> = (0..3)
            .map(|k| (textured_rgb8(8, 6, k), textured_rgb8(8, 6, k + 9)))
            .collect();
        let pairs: Vec<_> = imgs
            .iter()
            .map(|(a, b)| (rgb8_image(a, 8, 6), rgb8_image(b, 8, 6)))
            .collect();
        let rep = check_sequence(
            &mut Mad8 {
                forget_checkpoint: false,
            },
            SequenceInfo::new(),
            &pairs,
        );
        assert!(rep.violations.is_empty(), "{:?}", rep.violations);
        assert_eq!(
            rep.frames.iter().map(|f| f.0).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert!(rep.pooled.is_none());
    }

    #[test]
    fn simple_metric_conforms_and_missing_checkpoint_is_caught() {
        let (a, b) = (vec![10u8; 12], vec![12u8; 12]);
        let r = Image::from(PixelSlice::new(&a, 2, 2, 6, PixelDescriptor::RGB8_SRGB).unwrap());
        let d = Image::from(PixelSlice::new(&b, 2, 2, 6, PixelDescriptor::RGB8_SRGB).unwrap());
        assert_conforms(
            &mut Mad8 {
                forget_checkpoint: false,
            },
            &r,
            &d,
        );
        let v = check(
            &mut Mad8 {
                forget_checkpoint: true,
            },
            &r,
            &d,
        );
        assert_eq!(v.len(), 1, "{v:?}");
        assert_eq!(v[0].check, "stop");
    }
}
