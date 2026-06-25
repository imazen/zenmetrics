// Capture the git commit SHA of each path-dependency codec at build time so every sweep can stamp its
// output with the exact codec versions that produced the encoded blobs. This is the guardrail against
// working off stale/bad codec data (e.g. a docker image baked before a codec memory fix landed).
//
// Each SHA is read from an env override first (ZEN_CODEC_<NAME>_COMMIT — set as a docker `--build-arg` in
// CI where the sibling `.git` dirs aren't copied into the build context), else from `git -C <path>`.
// Exposed to the crate via `option_env!("ZEN_CODEC_<NAME>_COMMIT")`.
use std::process::Command;

fn git_short(rel: &str) -> String {
    let out = Command::new("git")
        .args(["-C", rel, "rev-parse", "--short=12", "HEAD"])
        .output();
    match out {
        Ok(o) if o.status.success() => {
            let sha = String::from_utf8_lossy(&o.stdout).trim().to_string();
            let dirty = Command::new("git")
                .args(["-C", rel, "status", "--porcelain", "--untracked-files=no"])
                .output()
                .map(|s| !s.stdout.is_empty())
                .unwrap_or(false);
            if sha.is_empty() {
                "unknown".to_string()
            } else if dirty {
                format!("{sha}-dirty")
            } else {
                sha
            }
        }
        _ => "unknown".to_string(),
    }
}

fn main() {
    // Paths relative to crates/zenmetrics-cli/ → ../../../ is ~/work/zen (the codec sibling repos).
    let codecs = [
        ("JXL_ENCODER", "../../../jxl-encoder"),
        ("ZENJXL", "../../../zenjxl"),
        ("ZENAVIF", "../../../zenavif"),
        ("ZENRAV1E", "../../../zenrav1e"),
        ("ZENJPEG", "../../../zenjpeg"),
        ("ZENWEBP", "../../../zenwebp"),
        ("BUTTERAUGLI", "../../../../butteraugli"),
    ];
    for (name, rel) in codecs {
        let key = format!("ZEN_CODEC_{name}_COMMIT");
        let val = std::env::var(&key).unwrap_or_else(|_| git_short(rel));
        println!("cargo:rustc-env={key}={val}");
        println!("cargo:rerun-if-changed={rel}/.git/HEAD");
        println!("cargo:rerun-if-env-changed={key}");
    }
    let zm = std::env::var("ZEN_METRICS_COMMIT").unwrap_or_else(|_| git_short("."));
    println!("cargo:rustc-env=ZEN_METRICS_COMMIT={zm}");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-env-changed=ZEN_METRICS_COMMIT");
}
