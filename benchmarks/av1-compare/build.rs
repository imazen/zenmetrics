use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn run(cmd: &mut Command) {
    let status = cmd.status().expect("start C build command");
    assert!(status.success(), "C build command failed: {cmd:?}");
}
fn revision(path: &Path) -> String {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(path)
        .output()
        .unwrap();
    assert!(out.status.success(), "cannot identify source revision");
    let dirty = Command::new("git")
        .args(["diff", "--quiet", "HEAD", "--"])
        .current_dir(path)
        .status()
        .unwrap();
    format!(
        "{}{}",
        String::from_utf8(out.stdout).unwrap().trim(),
        if dirty.success() { "" } else { "+dirty" }
    )
}
fn build(source: &Path, output: &Path, target: &str, opts: &[&str]) {
    assert!(
        source.join("CMakeLists.txt").is_file(),
        "missing pinned C source: {}",
        source.display()
    );
    let compiler = cc::Build::new().get_compiler();
    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(source)
        .arg("-B")
        .arg(output)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg(format!("-DCMAKE_C_COMPILER={}", compiler.path().display()))
        .arg(format!("-DCMAKE_OUTPUT_DIRECTORY={}", output.display()))
        .arg("-DCMAKE_C_FLAGS=-ffp-contract=off")
        .args(opts);
    run(&mut configure);
    run(Command::new("cmake")
        .arg("--build")
        .arg(output)
        .arg("--target")
        .arg(target)
        .arg("--parallel")
        .arg(env::var("NUM_JOBS").unwrap_or("1".into())));
}
fn main() {
    // These libraries belong to this target's OUT_DIR. Never reuse host archives
    // or mutate a parity oracle's build flags/cache for a performance run.
    let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let siblings = root.join("../../..").canonicalize().unwrap();
    let aom = siblings.join("zenav1-aom/upstream");
    let svt = siblings.join("zenav1-svt/reference/svt-av1");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    build(
        &aom,
        &out.join("aom"),
        "aom",
        &[
            "-DENABLE_TESTS=OFF",
            "-DENABLE_EXAMPLES=OFF",
            "-DENABLE_TOOLS=OFF",
            "-DCONFIG_MULTITHREAD=1",
            "-DCONFIG_AV1_ENCODER=1",
            "-DCONFIG_AV1_DECODER=1",
        ],
    );
    build(
        &svt,
        &out.join("svt"),
        "SvtAv1Enc",
        &[
            "-DBUILD_APPS=OFF",
            "-DBUILD_TESTING=OFF",
            "-DSVT_HDR_MODE=OFF",
            "-DNATIVE=OFF",
            "-DSVT_AV1_LTO=OFF",
        ],
    );
    cc::Build::new()
        .file("src/c_api.c")
        .include(&aom)
        .include(out.join("aom"))
        .include(svt.join("Source/API"))
        .flag_if_supported("-ffp-contract=off")
        .warnings(true)
        .compile("zenmetrics_av1_c");
    println!(
        "cargo:rustc-link-search=native={}",
        out.join("aom").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        out.join("svt").display()
    );
    println!("cargo:rustc-link-lib=static=aom");
    println!("cargo:rustc-link-lib=static=SvtAv1Enc");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-link-lib=pthread");
    for (name, path) in [
        ("LIBAOM", aom),
        ("C_SVT", svt),
        ("ZENAV1_SVT", siblings.join("zenav1-svt")),
        ("ZENAV1_AOM", siblings.join("zenav1-aom")),
        ("ZENRAV1E", siblings.join("zenrav1e")),
    ] {
        println!("cargo:rustc-env={name}_REV={}", revision(&path));
        // Track code edits without recursively scanning sibling target/ caches.
        // A dirty label alone is not a cache invalidator.
        let inputs: &[&str] = match name {
            "ZENAV1_SVT" => &["rust/Cargo.toml", "rust/crates", "rust/svtav1"],
            "ZENAV1_AOM" => &["Cargo.toml", "crates"],
            "ZENRAV1E" => &["Cargo.toml", "build.rs", "src"],
            _ => &["."],
        };
        for input in inputs {
            println!("cargo:rerun-if-changed={}", path.join(input).display());
        }
    }
    println!("cargo:rerun-if-changed=src/c_api.c");
    println!("cargo:rerun-if-env-changed=CC");
}
