//! Consolidated integration-test entry point.
//!
//! Every former `tests/<name>.rs` is a submodule here, compiled into one
//! `it` test binary instead of N separate binaries (one link step, not N).
//! Per-test gating that used to live in `[[test]] required-features` is now
//! a `#[cfg(...)]` on each `mod` line. Select a former target with a module
//! filter: `cargo test --test it <name>::`.

#[cfg(all(feature = "all-metrics", feature = "cpu-metrics"))]
mod backend_matrix;
mod backend_resolve;
mod cached_ref_parity;
mod cancel;
#[cfg(all(feature = "cubecl-types", feature = "cuda"))]
mod compute_handles;
#[cfg(all(feature = "butter", feature = "cuda", feature = "zensim"))]
mod compute_multi;
#[cfg(all(
    feature = "butter",
    feature = "cpu-butter",
    feature = "cuda",
    feature = "pixels"
))]
mod cpu_butter_linear;
#[cfg(all(
    feature = "cpu-cvvdp",
    feature = "cuda",
    feature = "cvvdp",
    feature = "hdr"
))]
mod cpu_cvvdp_linear;
#[cfg(all(
    feature = "cpu-butter",
    feature = "cpu-cvvdp",
    feature = "cpu-dssim",
    feature = "cpu-iwssim",
    feature = "cpu-ssim2",
    feature = "cpu-zensim"
))]
mod cpu_dispatch;
mod cpu_hdrvdp3_pu;
mod cpu_hdrvdp_pu;
#[cfg(all(feature = "cpu-ssim2", feature = "hdr", feature = "ssim2"))]
mod cpu_ssim2_pu;
#[cfg(all(feature = "cpu-zensim", feature = "hdr", feature = "zensim"))]
mod cpu_zensim_pu;
#[cfg(all(feature = "cuda", feature = "cvvdp"))]
mod cvvdp_display;
mod dispatch;
#[cfg(all(
    any(feature = "cuda", feature = "wgpu"),
    feature = "hdr",
    feature = "zensim"
))]
mod gpu_zensim_pu;
#[cfg(all(
    feature = "butter",
    feature = "cuda",
    feature = "hdr",
    feature = "ssim2",
    feature = "zensim"
))]
mod hdr_scorer;
#[cfg(all(
    feature = "cpu-butter",
    feature = "cpu-ssim2",
    feature = "cpu-zensim",
    feature = "hdr"
))]
mod hdr_scorer_cpu;
mod metric_base_hdr;
mod pixels_smoke;
mod score_pair;
mod session_alloc_flat;
#[cfg(all(feature = "cuda", feature = "cvvdp"))]
mod session_cap;
mod session_owned;
mod session_owned_cap;
#[cfg(all(feature = "cuda", feature = "cvvdp"))]
mod session_parity;
mod session_reclaim_non_cvvdp;
#[cfg(all(feature = "cuda", feature = "cvvdp"))]
mod session_vram_isolation;
#[cfg(all(
    feature = "butter",
    feature = "cuda",
    feature = "hdr",
    feature = "pixels",
    feature = "ssim2"
))]
mod unified_pixels;

/// Test parameters for `kind`. cvvdp has no default display in the umbrella
/// (`MetricParams::try_default_for` refuses it), so the suite names the
/// `standard_4k` preset explicitly — the display every parity golden was
/// captured at.
#[allow(dead_code)]
pub(crate) fn try_params_for(
    kind: zenmetrics_api::MetricKind,
) -> zenmetrics_api::Result<zenmetrics_api::MetricParams> {
    #[cfg(feature = "cvvdp")]
    if kind == zenmetrics_api::MetricKind::Cvvdp {
        return Ok(zenmetrics_api::MetricParams::cvvdp(
            zenmetrics_api::cvvdp::params::DisplayPreset::Standard4k,
        ));
    }
    zenmetrics_api::MetricParams::try_default_for(kind)
}

/// [`try_params_for`], panicking on a disabled metric (the old
/// `MetricParams::default_for` contract).
#[allow(dead_code)]
pub(crate) fn params_for(kind: zenmetrics_api::MetricKind) -> zenmetrics_api::MetricParams {
    try_params_for(kind).unwrap_or_else(|e| panic!("{e}"))
}

/// Env var that marks a re-exec'd child copy of this test binary and names
/// the case it should run. Unset in the shared test process.
pub(crate) const CHILD_CASE_VAR: &str = "ZM_IT_CHILD";

/// The case this process was spawned to run, or `None` in the shared test
/// process (the parent, which spawns children via [`run_in_child`]).
pub(crate) fn child_case() -> Option<String> {
    std::env::var(CHILD_CASE_VAR).ok()
}

/// State of `ZENMETRICS_FORCE_NO_GPU` a child is spawned with.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ForceNoGpu {
    /// Variable removed from the child's environment.
    Unset,
    /// Variable set to `"1"` in the child's environment.
    Set,
}

/// Runs the test `test_path` (its libtest name inside this binary, e.g.
/// `backend_matrix::foo`) in a CHILD copy of this test binary, with
/// `ZM_IT_CHILD=<case>` and `ZENMETRICS_FORCE_NO_GPU` set or cleared at spawn.
///
/// `ZENMETRICS_FORCE_NO_GPU` is read process-globally by `Backend::Auto`
/// resolution, and the tests in this binary run concurrently. A test that
/// needs a specific state of it therefore never mutates its own environment;
/// it spawns a child that owns the state from its first instruction. The
/// shared process's environment is never written, so no reader needs a lock
/// and no `unsafe` env call exists.
///
/// Panics unless the child exited successfully AND actually ran exactly the
/// requested test (`--exact` on a name that matches nothing exits 0 with
/// "running 0 tests"; that must not pass vacuously). The child's stdout and
/// stderr are included in the panic message.
#[allow(dead_code)]
pub(crate) fn run_in_child(test_path: &str, case: &str, force_no_gpu: ForceNoGpu) {
    let exe = std::env::current_exe().expect("current_exe of the test binary");
    let mut cmd = std::process::Command::new(exe);
    cmd.args([test_path, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD_CASE_VAR, case);
    match force_no_gpu {
        ForceNoGpu::Unset => cmd.env_remove("ZENMETRICS_FORCE_NO_GPU"),
        ForceNoGpu::Set => cmd.env("ZENMETRICS_FORCE_NO_GPU", "1"),
    };
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("spawning child for {test_path} [{case}]: {e}"));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "child {test_path} [{case}] failed ({}):\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
        out.status
    );
    assert!(
        stdout.contains("running 1 test\n") && stdout.contains("1 passed"),
        "child {test_path} [{case}] did not run exactly the requested test:\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
}
