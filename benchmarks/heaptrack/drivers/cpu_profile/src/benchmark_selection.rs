//! Scheduling policy only; explicit names and scoring behavior are unchanged.
use std::collections::BTreeSet;

const CHANGED_ORDER: &str =
    include_str!("../../../../rebenchmark_since_f7668c7/order_current_local41_2026-10-20.tsv");

fn changed_order() -> Vec<String> {
    CHANGED_ORDER
        .lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1) // TSV header
        .map(|line| line.split('\t').next().unwrap().to_owned())
        .collect()
}

/// Retain the measured priority order instead of the membership set's alphabetic order.
pub(super) fn schedule(request: &str, selected: &BTreeSet<String>) -> Vec<String> {
    if request == "changed" {
        changed_order()
    } else {
        selected.iter().cloned().collect()
    }
}

pub(super) fn select(request: &str, compiled: &BTreeSet<String>) -> BTreeSet<String> {
    if request == "changed" {
        return changed_order().into_iter().collect();
    }
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

    #[test]
    fn changed_schedule_contains_only_the_audited_scope_in_timing_order() {
        let names = include_str!("../../../../rebenchmark_since_f7668c7/existing_current41.txt")
            .lines()
            .chain(include_str!("../../../../rebenchmark_since_f7668c7/new.txt").lines())
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        let selected = select("changed", &names);
        assert_eq!(selected, names);
        let ordered = schedule("changed", &selected);
        assert_eq!(ordered.len(), names.len());
        assert_eq!(ordered.iter().cloned().collect::<BTreeSet<_>>(), names);
        let times: Vec<f64> = CHANGED_ORDER
            .lines()
            .filter(|line| !line.starts_with('#'))
            .skip(1)
            .map(|line| line.split('\t').nth(1).unwrap().parse().unwrap())
            .collect();
        assert!(times.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(ordered.first().unwrap(), "psnr");
        assert_eq!(ordered.last().unwrap(), "hdrvdp3-aic2026");
        assert_ne!(ordered, selected.into_iter().collect::<Vec<_>>());
    }
}
