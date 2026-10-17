//! Portable CPU coordinator. All metric timing remains inside the zenbench worker.
#![forbid(unsafe_code)]
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    env,
    error::Error,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};
#[path = "../optional_dvifm.rs"]
mod optional_dvifm;
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn version(program: &str) -> String {
    Command::new(program)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unavailable".into())
}
fn commit() -> Result<String> {
    for (program, args) in [
        (
            "jj",
            vec![
                "log",
                "-r",
                "latest(ancestors(@) & ~empty())",
                "--no-graph",
                "-T",
                "commit_id",
            ],
        ),
        ("git", vec!["rev-parse", "HEAD"]),
    ] {
        if let Ok(o) = Command::new(program).args(args).output()
            && o.status.success()
        {
            return Ok(String::from_utf8(o.stdout)?.trim().into());
        }
    }
    Err("Cannot resolve checkout commit using jj or git".into())
}
fn dimensions(name: &str) -> Result<(usize, usize)> {
    Ok(match name {
        "smoke" => (64, 64),
        "cid22" => (512, 512),
        "aic-small" => (840, 944),
        "aic-medium" => (1680, 1888),
        "aic-large" => (3355, 2516),
        "2k" => (2048, 1080),
        "4k" => (3840, 2160),
        "8k" => (7680, 4320),
        _ => return Err(format!("Unknown size {name}").into()),
    })
}
fn required(name: &str) -> usize {
    match name {
        "8k" => 1,
        "aic-large" | "2k" | "4k" => 3,
        _ => 5,
    }
}
fn worker() -> Result<PathBuf> {
    let mut p = env::current_exe()?;
    p.set_file_name(format!("iqa-compete{}", env::consts::EXE_SUFFIX));
    if !p.is_file() {
        return Err("Build iqa-compete beside bench-iqa (use just bench-iqa)".into());
    }
    Ok(p)
}
fn wait(mut command: Command, log: &Path) -> Result<bool> {
    let file = OpenOptions::new().write(true).create_new(true).open(log)?;
    let mut child = command
        .stdout(Stdio::from(file.try_clone()?))
        .stderr(Stdio::from(file))
        .spawn()?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status.success());
        }
        if start.elapsed() > Duration::from_secs(1120) {
            child.kill()?;
            child.wait()?;
            return Err(format!("Cell exceeded 1100 s: {}", log.display()).into());
        }
        thread::sleep(Duration::from_millis(20));
    }
}
fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.first().is_some_and(|s| s == "--child") {
        let count = &args[1];
        let mut child = Command::new(worker()?);
        child.arg(&args[2]);
        for key in [
            "RAYON_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "RTEN_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ] {
            child.env(key, count);
        }
        child
            .env("ZENBENCH_CPU_POLICY", "unrestricted")
            .env("ZENBENCH_NO_SAVE", "1");
        let mut process = child.spawn()?;
        let started = Instant::now();
        loop {
            if let Some(status) = process.try_wait()? {
                std::process::exit(status.code().unwrap_or(1));
            }
            if started.elapsed() > Duration::from_secs(1100) {
                process.kill()?;
                process.wait()?;
                return Err("Metric worker exceeded 1100 s".into());
            }
            thread::sleep(Duration::from_millis(20));
        }
    }
    if args.iter().any(|s| s == "--help") {
        println!(
            "just bench-iqa [sizes] [metrics] --host LABEL (--fixture synthetic-v1 | --corpus DIR) --skip-native [--output DIR] [--with-dvifmish]\nCPU only; GPU/NN/Python use just bench-iqa-advanced explicitly. Native comparator setup currently requires the advanced path."
        );
        return Ok(());
    }
    let sizes = args
        .first()
        .filter(|s| !s.starts_with('-'))
        .map(String::as_str)
        .unwrap_or("cid22");
    let metrics = args
        .get(1)
        .filter(|s| !s.starts_with('-'))
        .map(String::as_str)
        .unwrap_or("all");
    let mut host = None;
    let mut fixture = None;
    let mut corpus = None;
    let mut output = PathBuf::from("target/bench-iqa");
    let mut skip_native = false;
    let mut with_dvifmish = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--skip-native" => skip_native = true,
            "--with-dvifmish" => with_dvifmish = true,
            "--host" | "--fixture" | "--corpus" | "--output" => {
                let flag = &args[i];
                i += 1;
                let value = args.get(i).ok_or("Missing option value")?;
                match flag.as_str() {
                    "--host" => host = Some(value.clone()),
                    "--fixture" => fixture = Some(value.clone()),
                    "--corpus" => corpus = Some(PathBuf::from(value)),
                    _ => output = PathBuf::from(value),
                }
            }
            other => return Err(format!("Unknown option {other}; use --help").into()),
        }
        i += 1;
    }
    let host = host.ok_or("--host LABEL is required")?;
    if !host
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    {
        return Err("Host label must use lowercase ASCII letters, digits and hyphens".into());
    }
    if !skip_native {
        return Err("Native comparators are not silently omitted: pass --skip-native for Rust-only CPU, or use bench-iqa-advanced for pinned native/Python competitors. See docs/BENCHMARKING.md".into());
    }
    eprintln!("Explicit --skip-native: native competitors omitted; Rust CPU metrics only.");
    if fixture.as_deref() != Some("synthetic-v1") && corpus.is_none() {
        return Err(
            "Supply --fixture synthetic-v1 (smoke, not photo performance) or --corpus AIC2026_DIR"
                .into(),
        );
    }
    if fixture.is_some() && corpus.is_some() {
        return Err("Choose exactly one input source".into());
    }
    let bin = worker()?;
    let source = commit()?;
    let bytes = fs::read(&bin)?;
    let inventory: Vec<Value> =
        serde_json::from_slice(&Command::new(&bin).arg("inventory").output()?.stdout)?;
    let compiled: BTreeSet<String> = inventory
        .iter()
        .filter(|v| v["compiled"] == true && v["gpu"] == false)
        .map(|v| v["metric_name"].as_str().unwrap().into())
        .collect();
    let selected: BTreeSet<String> = if metrics == "all" {
        compiled.clone()
    } else {
        metrics.split(',').map(String::from).collect()
    };
    if !selected.is_subset(&compiled) {
        return Err(format!(
            "Uncompiled CPU metric(s): {:?}; NN/GPU require explicit advanced features",
            selected.difference(&compiled).collect::<Vec<_>>()
        )
        .into());
    }
    let all = thread::available_parallelism()?.get();
    let threads: BTreeSet<usize> = [1, all].into_iter().collect();
    let repo = env::current_dir()?;
    let optional_fingerprint = if with_dvifmish {
        Some(optional_dvifm::fingerprint(&repo)?)
    } else {
        None
    };
    let identity = json!({"commit":source,"binary_sha256":sha(&bytes),"host":host,"os":env::consts::OS,"arch":env::consts::ARCH,"fixture":fixture,"corpus":corpus,"threads":threads,"protocol":"coverage-5-3-1-v1","with_dvifmish":with_dvifmish,"optional_fingerprint":optional_fingerprint});
    let content = sha(&serde_json::to_vec(&identity)?);
    let root = output.join(&content);
    fs::create_dir_all(&root)?;
    let inputs = root.join("inputs");
    fs::create_dir_all(&inputs)?;
    if let Some(dir) = &corpus {
        let manifest: Value = serde_json::from_str(include_str!(
            "../../../../../iqa_compete_2026-10-16/inputs.meta.json"
        ))?;
        for (name, hash) in manifest["source"]["encoded_sha256"]
            .as_object()
            .ok_or("Input manifest")?
        {
            if sha(&fs::read(dir.join(name))?) != hash.as_str().ok_or("Hash")? {
                return Err(format!("AIC checksum differs: {name}").into());
            }
        }
        let prepared = inputs.join("aic");
        if !prepared.exists() {
            let status = Command::new(&bin)
                .arg("prepare")
                .arg(dir.join("S67_Ref_00.png"))
                .arg(dir.join("S67_CCM_15.png"))
                .arg(&prepared)
                .status()?;
            if !status.success() {
                return Err("AIC preparation failed".into());
            }
        }
        for pair in manifest["derivatives"].as_array().ok_or("Derivatives")? {
            for role in ["reference", "distorted"] {
                let file = prepared.join(pair[role].as_str().ok_or("Input path")?);
                if sha(&fs::read(file)?) != pair["sha256"][role].as_str().ok_or("Input hash")? {
                    return Err("AIC derivative checksum differs".into());
                }
            }
        }
    }
    let raw = root.join("rows.jsonl");
    let mut done = BTreeSet::new();
    let mut rows = Vec::new();
    if raw.exists() {
        for line in fs::read_to_string(&raw)?.lines() {
            let row: Value = serde_json::from_str(line)?;
            if row["status"] == "ok" || row["status"] == "unsupported" {
                done.insert(format!(
                    "{}:{}:{}:{}",
                    row["metric_name"].as_str().ok_or("metric")?,
                    row["size_name"].as_str().ok_or("size")?,
                    row["threads"].as_u64().ok_or("threads")?,
                    row["mode"].as_str().ok_or("mode")?
                ));
            }
            rows.push(row);
        }
    }
    let before = rows.len();
    let mut skipped = 0;
    for size in sizes.split(',') {
        let (w, h) = dimensions(size)?;
        let (r, d) = if corpus.is_some() {
            (
                inputs.join("aic").join(format!("{size}-reference.rgb")),
                inputs.join("aic").join(format!("{size}-distorted.rgb")),
            )
        } else {
            let r = inputs.join(format!("{size}-reference.rgb"));
            let d = inputs.join(format!("{size}-distorted.rgb"));
            let mut a = Vec::with_capacity(w * h * 3);
            let mut b = Vec::with_capacity(w * h * 3);
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        let value = ((x * 17 + y * 29 + c * 71 + (x * y) % 251) % 256) as u8;
                        a.push(value);
                        b.push(value.saturating_add(if (x + y + c) % 7 == 0 { 9 } else { 0 }));
                    }
                }
            }
            if !r.exists() {
                fs::write(&r, &a)?;
                fs::write(&d, &b)?;
            }
            if fs::read(&r)? != a || fs::read(&d)? != b {
                return Err("Synthetic fixture checksum differs".into());
            }
            (r, d)
        };
        let hashes = json!({"reference":sha(&fs::read(&r)?),"distorted":sha(&fs::read(&d)?)});
        let pair_sha = sha(&serde_json::to_vec(&hashes)?);
        for count in &threads {
            for metric in &selected {
                if ["score", "score+map"]
                    .iter()
                    .all(|mode| done.contains(&format!("{metric}:{size}:{count}:{mode}")))
                {
                    skipped += 2;
                    continue;
                }
                let config = json!({"host":host,"os":env::consts::OS,"arch":env::consts::ARCH,"cpu_model":env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_|env::consts::ARCH.into()),"gpu":null,"device":"cpu","impl":"zenmetrics","impl_version":source,"impl_binary_sha":identity["binary_sha256"],"engine":"native","metric_family":metric,"metric_name":metric,"size_name":size,"width":w,"height":h,"threads":count,"reference":fs::canonicalize(&r)?,"distorted":fs::canonicalize(&d)?,"pair_sha256":pair_sha,"input_sha256":hashes,"coverage_first":true,"required_samples":required(size),"sample_target_ns":1000000,"chunk_budget_ms":850000,"locked_process_accounting":true,"notes":"Portable CPU smoke; no OS affinity pin or contention qualification claimed"});
                let attempt = root.join(format!("{metric}-{size}-{count}-{}.json", rows.len()));
                fs::write(&attempt, serde_json::to_vec_pretty(&config)?)?;
                let log = attempt.with_extension("log");
                let configured = env::var_os("IQA_RUN_HEAVY")
                    .map(PathBuf::from)
                    .or_else(|| {
                        env::var_os("HOME")
                            .map(|h| PathBuf::from(h).join("work/claudehints/scripts/run-heavy"))
                    })
                    .filter(|p| p.is_file());
                let mut cmd = if let Some(wrapper) = configured {
                    let mut c = Command::new(wrapper);
                    c.args([
                        "--mem",
                        if cfg!(target_os = "macos") {
                            "12G"
                        } else {
                            "16G"
                        },
                        "--jobs",
                        &all.saturating_sub(4).max(1).to_string(),
                        "--",
                    ])
                    .arg(env::current_exe()?);
                    c
                } else {
                    Command::new(env::current_exe()?)
                };
                cmd.arg("--child").arg(count.to_string()).arg(&attempt);
                if !wait(cmd, &log)? {
                    return Err(format!("Worker failed: {}", log.display()).into());
                }
                let mut produced = 0;
                for line in fs::read_to_string(&log)?.lines() {
                    if let Some(text) = line.strip_prefix("IQA_ROW ") {
                        let row: Value = serde_json::from_str(text)?;
                        if row["status"] == "ok"
                            && row["n"].as_u64().unwrap_or(0) < required(size) as u64
                        {
                            return Err("Worker returned fewer than the required samples".into());
                        }
                        let mut file = OpenOptions::new().append(true).create(true).open(&raw)?;
                        writeln!(file, "{}", serde_json::to_string(&row)?)?;
                        rows.push(row);
                        produced += 1;
                    }
                }
                if produced != 2 {
                    return Err("Worker must report both score and score+map".into());
                }
            }
        }
    }
    let optional = if with_dvifmish {
        optional_dvifm::run(
            &repo,
            &inputs.join("aic"),
            &root.join("private-dvifm"),
            sizes,
            corpus.is_some(),
        )?
    } else {
        Vec::new()
    };
    let meta = json!({"identity":identity,"build_commit":source,"command":env::args().collect::<Vec<_>>(),"host":host,"hardware":{"os":env::consts::OS,"arch":env::consts::ARCH,"threads":all},"toolchains":{"rustc":version("rustc"),"cargo":version("cargo"),"cmake":version("cmake")},"competitors":{"native":"explicitly skipped","python":"not enabled"},"resource_policy":"Configured/local run-heavy per cell when present; otherwise plain portable child timeout, 1100 s; no plain-path memory cap claimed"});
    fs::write(root.join("run.meta"), serde_json::to_vec_pretty(&meta)?)?;
    let mut private_meta = meta.clone();
    private_meta["optional_dvifm"] = json!(optional);
    fs::write(
        root.join("run.meta"),
        serde_json::to_vec_pretty(&private_meta)?,
    )?;
    // Dashboard projection deliberately excludes raw paths and sample arrays.
    let fields = [
        "host",
        "os",
        "arch",
        "cpu_model",
        "gpu",
        "device",
        "impl",
        "impl_version",
        "impl_binary_sha",
        "engine",
        "metric_family",
        "metric_name",
        "size_name",
        "width",
        "height",
        "threads",
        "mode",
        "pair_sha256",
        "n",
        "score",
        "status",
        "median_ns",
        "p10_ns",
        "p90_ns",
        "mean_ns",
        "coverage_first",
        "required_samples",
        "notes",
        "input_sha256",
    ];
    let dashboard = PathBuf::from("benchmarks/iqa_compete_2026-10-16/repro-results").join(&content);
    fs::create_dir_all(&dashboard)?;
    for (index, chunk) in rows.chunks(8).enumerate() {
        let projected: Vec<Value> = chunk
            .iter()
            .map(|r| {
                Value::Object(
                    fields
                        .iter()
                        .filter_map(|k| r.get(*k).map(|v| ((*k).to_string(), v.clone())))
                        .collect(),
                )
            })
            .collect();
        fs::write(
            dashboard.join(format!("cells-{index:04}.json")),
            serde_json::to_vec_pretty(&projected)?,
        )?;
    }
    fs::write(
        dashboard.join("run.meta"),
        serde_json::to_vec_pretty(&meta)?,
    )?;
    println!(
        "RESULTS {} rows={} added={} skipped={skipped}",
        root.display(),
        rows.len(),
        rows.len() - before
    );
    Ok(())
}
