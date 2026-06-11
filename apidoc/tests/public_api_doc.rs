//! Public-API surface snapshots for the PARENT workspace (docs/public-api/).
//! Shared implementation + format docs: the `zenutils-apidoc` crate.
//!
//! Every workspace member is `publish = false`, so auto-discovery would find
//! nothing — the explicit list below mirrors the deliberate selection of the
//! pre-runner snapshot test: the umbrella API crate, the two CPU metric
//! crates, and the six GPU metric crates. Fleet/jobdash infra, corpus/
//! test-only crates, codegen tools, and pure-bin crates are not snapshotted.
#[test]
fn public_api_surface_docs_are_current() {
    zenutils_apidoc::ApiDoc::new()
        .workspace_dir("..")
        .crates([
            "zenmetrics-api",
            "iwssim",
            "cvvdp",
            "butteraugli-gpu",
            "cvvdp-gpu",
            "dssim-gpu",
            "iwssim-gpu",
            "ssim2-gpu",
            "zensim-gpu",
        ])
        .run();
}
