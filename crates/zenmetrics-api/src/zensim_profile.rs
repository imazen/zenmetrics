//! Runtime selection of the **zensim scoring profile** — a named built-in
//! (`zensim-b`, `zensim-c`, …) or an arbitrary ZNPR bake loaded from disk.
//!
//! ## Why this module exists
//!
//! Every zensim construction site in this crate used to hard-code
//! `zensim::ZensimProfile::latest_preview()`, so the fleet metric path
//! (`zenmetrics jobexec`, `zenmetrics score-pairs`, the sweep workers) could
//! only ever score with the shipped default. A candidate bake could not be
//! scored through the fleet at all — which is where any large re-scoring has
//! to happen. This module is the missing selector.
//!
//! ## Two ways in, both additive
//!
//! 1. **Explicit** — resolve a spec to a [`zensim::ZensimProfile`] and pass it
//!    through the existing parameter plumbing
//!    ([`zensim_gpu::ZensimParams::with_profile`](crate::zensim::ZensimParams::with_profile)).
//! 2. **Process default** — [`set_default`] installs a process-wide override
//!    that every construction site consults through [`default_profile`]. This
//!    is what a CLI flag drives, and it is the only route available on the
//!    parameter-less paths (`Metric::new_cpu_hdr`, the CPU-only
//!    `MetricParams::Zensim(())` placeholder).
//!
//! **Unset, [`default_profile`] returns `zensim::ZensimProfile::latest_preview()`
//! — byte-identical to the previous hard-coded literal.**
//!
//! ## The `fn() -> &'static [u8]` friction (read before extending)
//!
//! `zensim::profile::ProfileParams::builder().mlp(..)` takes a bare
//! `fn() -> &'static [u8]`, **not** a closure, so a bake chosen at runtime
//! cannot simply be captured — the bytes must already live at `'static`
//! behind a plain function item. This module therefore parks each loaded bake
//! in one of [`MAX_RUNTIME_BAKES`] process-global slots and hands the builder
//! the matching monomorphized accessor. That caps a single process at
//! [`MAX_RUNTIME_BAKES`] *distinct* runtime bakes (repeat resolutions of the
//! same path reuse their slot); the cap is a hard error, never a silent
//! wrap-around.
//!
//! ## Spline-carrying bakes (the 0.000000 trap)
//!
//! A bake that carries its own `zentrain.output_calibration_spline` MUST be
//! configured `skip_score_mapping = true` **and** `extrapolate_score = true`.
//! Omit `skip_score_mapping` and zensim applies the legacy `100 − A·d^B`
//! mapping *on top of* the bake's calibration spline, and every score from
//! q10 to q100 comes back exactly `0.000000` with no error at all.
//! [`BakeProfileConfig::default`] therefore mirrors zensim's own shipped
//! `PROFILE_B` literal (both flags set, extended + IW features on, 372-wide
//! caller input). Override only with a reason.

use std::path::Path;
use std::sync::{Mutex, OnceLock, RwLock};

use crate::{Error, Result};

/// How many **distinct** runtime bakes one process can register.
///
/// See the module docs: `ProfileParams::builder().mlp()` takes a bare
/// `fn() -> &'static [u8]`, so each runtime bake needs its own statically
/// monomorphized accessor. Resolving the same path (with the same
/// [`BakeProfileConfig`]) repeatedly reuses one slot, so this is a cap on
/// distinct bakes, not on calls.
pub const MAX_RUNTIME_BAKES: usize = 8;

/// Leaked bake bytes, one slot per distinct runtime bake.
static BAKE_SLOTS: [OnceLock<&'static [u8]>; MAX_RUNTIME_BAKES] =
    [const { OnceLock::new() }; MAX_RUNTIME_BAKES];

/// The `fn() -> &'static [u8]` the builder wants, one monomorphization per
/// slot. Unfilled slots yield an empty slice rather than panicking: a profile
/// is only ever handed out *after* its slot is filled (see [`register_bake`]),
/// so an empty read is unreachable, and zensim rejects a zero-length bake
/// loudly if that invariant ever breaks.
fn slot_bytes<const N: usize>() -> &'static [u8] {
    BAKE_SLOTS[N].get().copied().unwrap_or(&[])
}

static SLOT_FNS: [fn() -> &'static [u8]; MAX_RUNTIME_BAKES] = [
    slot_bytes::<0>,
    slot_bytes::<1>,
    slot_bytes::<2>,
    slot_bytes::<3>,
    slot_bytes::<4>,
    slot_bytes::<5>,
    slot_bytes::<6>,
    slot_bytes::<7>,
];

/// `(dedup key, profile)` for every bake registered in this process.
static REGISTERED: Mutex<Vec<(String, ::zensim::ZensimProfile)>> = Mutex::new(Vec::new());

/// Process-wide profile override. `None` ⇒ `latest_preview()`.
static DEFAULT_OVERRIDE: RwLock<Option<::zensim::ZensimProfile>> = RwLock::new(None);

fn err(message: String) -> Error {
    Error::Metric {
        kind: "zensim",
        message,
    }
}

/// Runtime dispositions applied to a bake loaded from disk.
///
/// [`Default`] mirrors zensim's shipped `PROFILE_B` literal, which is the
/// correct shape for any bake carrying an output-calibration spline — see the
/// module docs for what happens when `skip_score_mapping` is left off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct BakeProfileConfig {
    /// Compute the extended (228 → 300) masked-feature block.
    pub extended_features: bool,
    /// Compute the IW-pool (300 → 372) block. Implies `extended_features`.
    pub compute_iw_features: bool,
    /// Return the bake's own (spline-calibrated) output instead of applying
    /// the legacy `100 − A·d^B` mapping. **Required for a spline bake.**
    pub skip_score_mapping: bool,
    /// Return the spline-extrapolated score unclamped, so worse-than-worst
    /// inputs land below 0 instead of tying at 0. **Required for a spline
    /// bake** (and the negative-tail product contract).
    pub extrapolate_score: bool,
    /// Logistic soft-clamp instead of a hard `clamp(0, 100)`. Ignored when
    /// `extrapolate_score` is set.
    pub soft_clamp_score: bool,
}

impl Default for BakeProfileConfig {
    fn default() -> Self {
        // Mirrors zensim's `static PROFILE_B` exactly.
        Self {
            extended_features: true,
            compute_iw_features: true,
            skip_score_mapping: true,
            extrapolate_score: true,
            soft_clamp_score: false,
        }
    }
}

impl BakeProfileConfig {
    fn key_tag(&self) -> String {
        format!(
            "ext={},iw={},skipmap={},extrap={},soft={}",
            self.extended_features,
            self.compute_iw_features,
            self.skip_score_mapping,
            self.extrapolate_score,
            self.soft_clamp_score,
        )
    }
}

/// Every built-in profile name [`builtin`] accepts, in the canonical
/// `zensim-*` spelling. The short form (`"b"`, `"c-hdr"`, …) and any ASCII
/// case are accepted too.
///
/// The list is build-dependent: `zensim-a` needs zensim's
/// `deprecated-profiles` feature and `zensim-c` / `zensim-c-hdr` need
/// `candidate-profiles` (both are on by zensim default).
pub fn builtin_names() -> Vec<&'static str> {
    let mut v = vec![
        "zensim-preview-v0.1",
        "zensim-preview-v0.2",
        "zensim-b",
        "zensim-b-hdr",
    ];
    if builtin("zensim-a").is_some() {
        v.push("zensim-a");
    }
    if builtin("zensim-c").is_some() {
        v.push("zensim-c");
        v.push("zensim-c-hdr");
    }
    v.sort_unstable();
    v
}

/// Resolve a **built-in** profile by name. Accepts the canonical
/// `zensim-b` spelling and the short `b` form, any ASCII case.
/// Returns `None` for an unknown name (or one whose zensim Cargo feature is
/// off in this build).
pub fn builtin(name: &str) -> Option<::zensim::ZensimProfile> {
    use ::zensim::ZensimProfile as P;
    let n = name.trim().to_ascii_lowercase();
    let n = n.strip_prefix("zensim-").unwrap_or(&n);
    match n {
        "preview-v0.1" | "preview-v0_1" => Some(P::PreviewV0_1),
        "preview-v0.2" | "preview-v0_2" => Some(P::PreviewV0_2),
        "b" => Some(P::B),
        "b-hdr" | "bhdr" => Some(P::BHdr),
        "latest" | "latest-preview" | "codec-target" | "default" => Some(P::latest_preview()),
        #[cfg(feature = "zensim-deprecated-profiles")]
        "a" => {
            #[allow(deprecated)]
            Some(P::A)
        }
        #[cfg(feature = "zensim-candidate-profiles")]
        "c" => Some(P::C),
        #[cfg(feature = "zensim-candidate-profiles")]
        "c-hdr" | "chdr" => Some(P::CHdr),
        _ => None,
    }
}

/// Load a ZNPR bake from `path` and wrap it as a
/// `zensim::ZensimProfile::Custom` using [`BakeProfileConfig::default`]
/// (the spline-bake shape — see the module docs).
pub fn from_bake_path(path: impl AsRef<Path>) -> Result<::zensim::ZensimProfile> {
    from_bake_path_with(path, BakeProfileConfig::default())
}

/// [`from_bake_path`] with explicit dispositions.
///
/// Resolving the same `(path, cfg)` again returns the *same* profile (same
/// leaked `ProfileParams` pointer), so `ZensimProfile` equality holds across
/// calls and no slot is wasted.
pub fn from_bake_path_with(
    path: impl AsRef<Path>,
    cfg: BakeProfileConfig,
) -> Result<::zensim::ZensimProfile> {
    let path = path.as_ref();
    let canonical = std::fs::canonicalize(path)
        .map_err(|e| err(format!("zensim bake {}: {e}", path.display())))?;
    let key = format!("{}|{}", canonical.display(), cfg.key_tag());

    let mut reg = REGISTERED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some((_, p)) = reg.iter().find(|(k, _)| *k == key) {
        return Ok(*p);
    }

    let bytes = std::fs::read(&canonical)
        .map_err(|e| err(format!("zensim bake {}: {e}", canonical.display())))?;
    if bytes.is_empty() {
        return Err(err(format!(
            "zensim bake {} is empty (0 bytes)",
            canonical.display()
        )));
    }

    let slot = reg.len();
    if slot >= MAX_RUNTIME_BAKES {
        return Err(err(format!(
            "cannot register a {}th runtime zensim bake: this process caps at \
             MAX_RUNTIME_BAKES = {MAX_RUNTIME_BAKES} distinct bakes because \
             zensim's ProfileParams builder takes a bare fn() -> &'static [u8] \
             (see zenmetrics_api::zensim_profile module docs)",
            slot + 1
        )));
    }

    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    // Slots are handed out strictly in order and each is written exactly once,
    // so `set` cannot already be filled.
    BAKE_SLOTS[slot]
        .set(leaked)
        .map_err(|_| err(format!("zensim bake slot {slot} was already filled")))?;

    let params = ::zensim::profile::ProfileParams::builder()
        .mlp(SLOT_FNS[slot])
        .extended_features(cfg.extended_features)
        .compute_iw_features(cfg.compute_iw_features)
        .skip_score_mapping(cfg.skip_score_mapping)
        .extrapolate_score(cfg.extrapolate_score)
        .soft_clamp_score(cfg.soft_clamp_score)
        .build();

    let name: &'static str = Box::leak(
        format!(
            "zensim-bake:{}",
            canonical
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| canonical.display().to_string())
        )
        .into_boxed_str(),
    );

    let profile = ::zensim::ZensimProfile::Custom {
        params: Box::leak(Box::new(params)),
        name,
    };
    reg.push((key, profile));
    Ok(profile)
}

/// Resolve a profile **spec**: either a built-in name ([`builtin`]) or a
/// filesystem path to a ZNPR bake ([`from_bake_path`]).
///
/// A spec that is neither a known built-in nor an existing file is an error
/// naming both possibilities — it never silently falls back to the default.
pub fn resolve(spec: &str) -> Result<::zensim::ZensimProfile> {
    if let Some(p) = builtin(spec) {
        return Ok(p);
    }
    let path = Path::new(spec);
    if path.is_file() {
        return from_bake_path(path);
    }
    Err(err(format!(
        "unknown zensim profile spec {spec:?}: not a built-in profile ({}) \
         and not a readable bake file",
        builtin_names().join(", ")
    )))
}

/// Install a process-wide profile override. Every zensim construction site in
/// this crate that has no explicit profile consults it via
/// [`default_profile`]. Intended to be called **once, early** (a CLI flag);
/// calling it after a scorer is built does not retro-fit that scorer.
pub fn set_default(profile: ::zensim::ZensimProfile) {
    *DEFAULT_OVERRIDE
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(profile);
}

/// [`resolve`] + [`set_default`]; returns the installed profile.
pub fn set_default_from_spec(spec: &str) -> Result<::zensim::ZensimProfile> {
    let p = resolve(spec)?;
    set_default(p);
    Ok(p)
}

/// Drop any override installed by [`set_default`], restoring
/// `latest_preview()`.
pub fn clear_default() {
    *DEFAULT_OVERRIDE
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
}

/// Whether a [`set_default`] override is currently installed.
pub fn has_default_override() -> bool {
    override_profile().is_some()
}

/// The installed override, if any.
///
/// **Precedence note.** Construction sites consult this *before* any
/// profile carried in `MetricParams`, because `MetricParams::default_for(
/// MetricKind::Zensim)` unconditionally stamps `Some(latest())` — so if
/// params won, an operator's `--zensim-bake` would be silently discarded on
/// every default-params call, which is the exact D14 failure this module
/// exists to remove. An override is only ever installed by an explicit
/// operator action, so it is the more specific instruction of the two.
pub fn override_profile() -> Option<::zensim::ZensimProfile> {
    *DEFAULT_OVERRIDE
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The profile every parameter-less zensim construction site in this crate
/// uses. Returns `zensim::ZensimProfile::latest_preview()` unless
/// [`set_default`] installed an override — so the default build is
/// byte-identical to the previous hard-coded literal.
pub fn default_profile() -> ::zensim::ZensimProfile {
    override_profile().unwrap_or_else(::zensim::ZensimProfile::latest_preview)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_accepts_canonical_and_short_and_case() {
        for spec in ["zensim-b", "b", "B", "  ZenSim-B "] {
            assert_eq!(
                builtin(spec).map(|p| p.name()),
                Some("zensim-b"),
                "spec {spec:?}"
            );
        }
        assert!(builtin("not-a-profile").is_none());
    }

    #[test]
    fn resolve_rejects_unknown_spec_instead_of_defaulting() {
        let e = resolve("definitely-not-a-profile-or-a-file").unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("unknown zensim profile spec"), "{msg}");
    }

    #[test]
    fn missing_bake_file_is_an_error_not_a_default() {
        let e = from_bake_path("/nonexistent/zensim/bake/path.bin").unwrap_err();
        assert!(e.to_string().contains("zensim bake"), "{e}");
    }
}
