//! Scheduling policy only; explicit names and scoring behavior are unchanged.
use std::collections::BTreeSet;

pub(super) fn select(request: &str, compiled: &BTreeSet<String>) -> BTreeSet<String> {
    if request != "all" {
        return request.split(',').map(String::from).collect();
    }
    let mut selected = compiled.clone();
    for name in ["pieapp", "pieapp-pyiqa"] {
        if selected.remove(name) {
            eprintln!("BENCHMARK OMITTED (slow; explicit selection required): {name}");
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inventory() -> BTreeSet<String> {
        ["psnr", "pieapp", "pieapp-pyiqa", "dists"]
            .map(String::from)
            .into_iter()
            .collect()
    }

    #[test]
    fn default_group_omits_both_pieapp_routes() {
        assert_eq!(
            select("all", &inventory()),
            ["dists", "psnr"].map(String::from).into_iter().collect()
        );
    }

    #[test]
    fn explicit_names_preserve_both_routes_and_unknown_validation() {
        for request in [
            "pieapp",
            "pieapp-pyiqa",
            "psnr,pieapp,pieapp-pyiqa",
            "unknown",
        ] {
            assert_eq!(
                select(request, &inventory()),
                request.split(',').map(String::from).collect()
            );
        }
    }
}
