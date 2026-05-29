//! Notifications (goal D) — pure event *detection* + *formatting*. Conditions are derived from the
//! same ledger-backed views the dashboard shows, so notifications can't disagree with the UI. The
//! formatted payload (text + deep link) is posted to the user's webhook by a thin sender added with
//! the webhook URL; detection/formatting is fully testable offline here.

use serde::{Deserialize, Serialize};

use crate::views::{CostView, KindProgress};

/// A fire-worthy condition. Carries enough context for the message + a deep link.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum NotifyEvent {
    /// Every kind has drained (no in-flight, all terminal).
    RunComplete { done: usize, poison: usize },
    BudgetCrossed { spent_usd: f64, cap_usd: f64 },
    FleetStalled { stalled_workers: usize },
    PoisonSpike { kind: String, poison: usize },
    KindDrained { kind: String },
}

/// What gets posted to the webhook.
#[derive(Serialize, Debug, PartialEq, Eq)]
pub struct NotifyPayload {
    pub text: String,
    pub link: String,
}

/// Render an event to a human message + deep link back to the dashboard.
pub fn format_event(ev: &NotifyEvent, base_url: &str) -> NotifyPayload {
    let base = base_url.trim_end_matches('/');
    let (text, frag) = match ev {
        NotifyEvent::RunComplete { done, poison } => {
            (format!("run complete: {done} done, {poison} poison"), "#progress")
        }
        NotifyEvent::BudgetCrossed { spent_usd, cap_usd } => (
            format!("budget crossed: ${spent_usd:.2} >= cap ${cap_usd:.2} - paid tiers tearing down"),
            "#cost",
        ),
        NotifyEvent::FleetStalled { stalled_workers } => {
            (format!("fleet stalled: {stalled_workers} worker(s) not progressing"), "#progress")
        }
        NotifyEvent::PoisonSpike { kind, poison } => {
            (format!("poison spike: {poison} poisoned in {kind}"), "#failures")
        }
        NotifyEvent::KindDrained { kind } => (format!("{kind} fully drained"), "#progress"),
    };
    NotifyPayload { text, link: format!("{base}/{frag}") }
}

/// Detect currently-true conditions from the ledger-backed views. Caller is responsible for
/// de-duplicating against already-sent events (so a standing condition fires once, not every poll).
pub fn detect(progress: &[KindProgress], cost: &CostView, cap_usd: f64, poison_threshold: usize) -> Vec<NotifyEvent> {
    let mut events = Vec::new();

    // Budget (goal F's stop-spend trigger).
    if cap_usd > 0.0 && cost.total_spent_usd >= cap_usd {
        events.push(NotifyEvent::BudgetCrossed { spent_usd: cost.total_spent_usd, cap_usd });
    }

    let mut all_drained = !progress.is_empty();
    let mut total_done = 0usize;
    let mut total_poison = 0usize;
    for k in progress {
        total_done += k.done;
        total_poison += k.poison;
        let drained = k.total > 0 && k.in_flight == 0 && k.done + k.poison == k.total;
        if drained {
            events.push(NotifyEvent::KindDrained { kind: k.kind.clone() });
        } else {
            all_drained = false;
        }
        if k.poison >= poison_threshold && poison_threshold > 0 {
            events.push(NotifyEvent::PoisonSpike { kind: k.kind.clone(), poison: k.poison });
        }
    }
    if all_drained {
        events.push(NotifyEvent::RunComplete { done: total_done, poison: total_poison });
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kp(kind: &str, total: usize, done: usize, poison: usize, in_flight: usize) -> KindProgress {
        KindProgress { kind: kind.into(), total, done, failed: total - done - poison - in_flight, poison, in_flight }
    }

    fn cost(spent: f64) -> CostView {
        CostView { total_spent_usd: spent, burn_usd_per_hr: 0.0, jobs_done: 0, cost_per_1000_jobs: None, per_tier: vec![] }
    }

    #[test]
    fn budget_crossed_fires_at_cap() {
        let evs = detect(&[], &cost(5.0), 5.0, 10);
        assert!(matches!(evs.as_slice(), [NotifyEvent::BudgetCrossed { .. }]));
        assert!(detect(&[], &cost(4.9), 5.0, 10).is_empty());
    }

    #[test]
    fn kind_drained_and_run_complete() {
        // one kind, fully terminal (3 done + 1 poison = total 4, none in flight)
        let evs = detect(&[kp("metric:cvvdp", 4, 3, 1, 0)], &cost(0.0), 0.0, 99);
        assert!(evs.contains(&NotifyEvent::KindDrained { kind: "metric:cvvdp".into() }));
        assert!(evs.contains(&NotifyEvent::RunComplete { done: 3, poison: 1 }));
    }

    #[test]
    fn in_flight_blocks_run_complete() {
        let evs = detect(&[kp("metric:cvvdp", 4, 2, 0, 2)], &cost(0.0), 0.0, 99);
        assert!(!evs.iter().any(|e| matches!(e, NotifyEvent::RunComplete { .. })));
        assert!(!evs.iter().any(|e| matches!(e, NotifyEvent::KindDrained { .. })));
    }

    #[test]
    fn poison_spike_threshold() {
        let evs = detect(&[kp("metric:cvvdp", 10, 3, 5, 2)], &cost(0.0), 0.0, 5);
        assert!(evs.contains(&NotifyEvent::PoisonSpike { kind: "metric:cvvdp".into(), poison: 5 }));
    }

    #[test]
    fn format_has_deep_link() {
        let p = format_event(&NotifyEvent::KindDrained { kind: "metric:cvvdp".into() }, "https://dash.up.railway.app/");
        assert!(p.text.contains("metric:cvvdp"));
        assert_eq!(p.link, "https://dash.up.railway.app/#progress");
    }
}
