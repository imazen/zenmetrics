//! Optional external benchmark adapter. Results remain outside dashboard rows.
use serde_json::{Value, json};
use std::{env, path::Path, process::Command};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn inventory(checkout: &Path, cpp: Option<&Path>) -> Vec<Value> {
    let rust = checkout
        .join("target/release/examples/bench_pair")
        .is_file()
        && checkout.join("tools/bench_compare.py").is_file();
    let mut rows = Vec::new();
    for variant in ["min-v0.2", "2d-v0.2", "lap-v0.2", "lap-v0.3"] {
        rows.push(json!({"implementation":format!("dvifmish-port {variant}"),"status":if rust {"available"} else {"unavailable"},"reason":if rust {"built external worker"} else {"checkout or built worker absent"}}));
        for build in ["default", "strict"] {
            let present = cpp.is_some_and(|p| p.join(format!("{variant}-{build}")).is_file());
            rows.push(json!({"implementation":format!("dvifm-cpp {variant} {build}"),"status":if present {"available"} else {"unavailable"},"reason":if present {"external build found"} else {"DVIFM_CPP_DIR build absent"}}));
        }
    }
    for model in ["luma", "ycbcr3", "xyb3", "xyby"] {
        rows.push(json!({"implementation":format!("dvifmish {model}"),"status":if rust {"available"} else {"unavailable"},"reason":if rust {"external preset selection"} else {"checkout or built worker absent"}}));
    }
    rows
}

pub(super) fn fingerprint(repo: &Path) -> Result<String> {
    let checkout = env::var_os("DVIFMISH_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| repo.join("../dvifmish"));
    let cpp = env::var_os("DVIFM_CPP_DIR").map(std::path::PathBuf::from);
    let mut paths = vec![
        checkout.join("target/release/examples/bench_pair"),
        checkout.join("tools/bench_compare.py"),
        checkout.join("tools/bench-inputs.json"),
    ];
    if let Some(cpp) = &cpp {
        for variant in ["min-v0.2", "2d-v0.2", "lap-v0.2", "lap-v0.3"] {
            for build in ["default", "strict"] {
                paths.push(cpp.join(format!("{variant}-{build}")));
            }
        }
        paths.push(cpp.join("build.meta"));
    }
    let hashes: Vec<_> = paths
        .iter()
        .map(|p| {
            if p.is_file() {
                std::fs::read(p).map(|b| super::sha(&b))
            } else {
                Ok("unavailable".into())
            }
        })
        .collect::<std::io::Result<_>>()?;
    Ok(super::sha(&serde_json::to_vec(&hashes)?))
}

pub(super) fn run(
    repo: &Path,
    inputs: &Path,
    output: &Path,
    sizes: &str,
    canonical: bool,
) -> Result<Vec<Value>> {
    let checkout = env::var_os("DVIFMISH_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| repo.join("../dvifmish"));
    let cpp = env::var_os("DVIFM_CPP_DIR").map(std::path::PathBuf::from);
    let mut rows = inventory(&checkout, cpp.as_deref());
    if !canonical {
        for row in &mut rows {
            if row["status"] == "available" {
                row["status"] = json!("unavailable");
                row["reason"] = json!("external panel requires canonical AIC input manifest");
            }
        }
    } else if rows.iter().any(|r| {
        r["implementation"]
            .as_str()
            .is_some_and(|s| s.starts_with("dvifmish-port"))
            && r["status"] == "available"
    }) {
        // The external checkout owns implementation details and private output.
        // Never add these files to the dashboard projection.
        let wrapper = env::var_os("IQA_RUN_HEAVY")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                env::var_os("HOME")
                    .map(|p| std::path::PathBuf::from(p).join("work/claudehints/scripts/run-heavy"))
            })
            .filter(|p| p.is_file());
        let mut command = if let Some(wrapper) = wrapper {
            let mut c = Command::new(wrapper);
            c.args([
                "--mem",
                if cfg!(target_os = "macos") {
                    "12G"
                } else {
                    "16G"
                },
                "--",
                "python3",
            ]);
            c
        } else {
            Command::new("python3")
        };
        command
            .arg(checkout.join("tools/bench_compare.py"))
            .arg("--inputs")
            .arg(inputs)
            .arg("--output")
            .arg(output)
            .arg("--sizes")
            .arg(sizes);
        if let Some(cpp) = cpp {
            command.arg("--cpp-dir").arg(cpp);
        }
        let finished = output.join("run.meta");
        let complete = finished.is_file()
            && serde_json::from_slice::<Value>(&std::fs::read(&finished)?)?["finished_utc"]
                .is_string();
        let status = if complete {
            None
        } else {
            Some(command.status()?)
        };
        if status.is_some_and(|s| !s.success()) {
            return Err("Available external DVIFM benchmark failed; inspect private output".into());
        }
        for row in &mut rows {
            if row["status"] == "available" {
                row["status"] = json!("completed");
            }
        }
    } else {
        for row in &mut rows {
            if row["status"] == "available" {
                row["status"] = json!("unavailable");
                row["reason"] = json!("external driver absent; C++ panel needs checkout driver");
            }
        }
    }
    for row in &rows {
        eprintln!(
            "OPTIONAL {} {} ({})",
            row["implementation"], row["status"], row["reason"]
        );
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_checkout_and_cpp_are_individual_unavailable_rows() {
        let missing = Path::new("/nonexistent-optional-dvifm-benchmark");
        let rows = inventory(missing, None);
        assert_eq!(rows.len(), 16);
        assert!(rows.iter().all(|r| r["status"] == "unavailable"));
        let names: std::collections::BTreeSet<_> = rows
            .iter()
            .map(|r| r["implementation"].as_str().unwrap())
            .collect();
        assert_eq!(names.len(), rows.len());
    }
    #[test]
    fn partial_cpp_inventory_preserves_each_missing_build() {
        let root = std::path::PathBuf::from(std::env::var_os("HOME").unwrap())
            .join("tmp")
            .join(format!("optional-dvifm-test-{}", std::process::id()));
        assert!(!root.exists());
        std::fs::create_dir_all(root.join("checkout/target/release/examples")).unwrap();
        std::fs::create_dir_all(root.join("checkout/tools")).unwrap();
        std::fs::create_dir_all(root.join("cpp")).unwrap();
        for path in [
            "checkout/target/release/examples/bench_pair",
            "checkout/tools/bench_compare.py",
            "cpp/min-v0.2-default",
        ] {
            std::fs::write(root.join(path), b"test fixture").unwrap();
        }
        let rows = inventory(&root.join("checkout"), Some(&root.join("cpp")));
        assert_eq!(
            rows.iter().filter(|r| r["status"] == "available").count(),
            9
        );
        assert_eq!(
            rows.iter().filter(|r| r["status"] == "unavailable").count(),
            7
        );
        assert!(
            rows.iter()
                .any(|r| r["implementation"] == "dvifm-cpp min-v0.2 default"
                    && r["status"] == "available")
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
