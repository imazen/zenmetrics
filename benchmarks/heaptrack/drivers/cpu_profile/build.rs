fn main() {
    println!("cargo::rustc-check-cfg=cfg(iqa_compete_cached)");
    println!("cargo::rustc-check-cfg=cfg(iqa_compete_ingress)");
    if std::env::var_os("CARGO_FEATURE_COMPETE").is_some() {
        // Reuse the exact private CLI ingress in this benchmark without adding
        // a shipped public API or maintaining a second conversion formula.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../crates/zenmetrics-cli/src/metrics/mod.rs");
        println!("cargo::rerun-if-changed={}", path.display());
        let source = std::fs::read_to_string(path).expect("CLI ingress source");
        let start = source
            .find("fn studio601_yuv444(")
            .expect("CLI YUV444 helper");
        let body = start + source[start..].find('{').expect("helper body");
        let mut depth = 0;
        let mut end = None;
        for (offset, ch) in source[body..].char_indices() {
            if ch == '{' {
                depth += 1;
            }
            if ch == '}' {
                depth -= 1;
                if depth == 0 {
                    end = Some(body + offset + 1);
                    break;
                }
            }
        }
        let function = &source[start..end.expect("balanced CLI helper body")];
        let generated = format!("use zenmetrics_cli::decode::Rgb8Image;\npub(super) {function}\n");
        std::fs::write(
            std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap())
                .join("iqa_ingress444.rs"),
            generated,
        )
        .expect("write benchmark ingress");
    }
}
