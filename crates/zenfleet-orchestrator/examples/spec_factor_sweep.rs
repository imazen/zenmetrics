use zenfleet_orchestrator::{SpeculativeConfig, SpeculativeState};

fn main() {
    let fast: Vec<f64> = vec![
        140.0, 150.0, 130.0, 160.0, 145.0, 155.0, 135.0, 165.0,
        185.0, 192.0, 175.0, 198.0, 188.0, 196.0, 180.0, 200.0,
        220.0, 230.0, 215.0, 240.0, 225.0, 235.0, 218.0, 245.0,
        255.0, 268.0, 272.0, 280.0,
    ];
    let stragglers: Vec<f64> = vec![380.0, 395.0];
    let baseline = 388.0_f64;
    println!("factor  n_disp  t_first  sim_t_done  reduction");
    for factor in [1.5_f64, 1.4, 1.3, 1.25, 1.2, 1.15, 1.1] {
        let mut state = SpeculativeState::new();
        let cfg = SpeculativeConfig {
            enabled: true,
            straggler_factor: factor,
            min_completed_for_stats: 3,
            speculation_cap_per_chunk: 1,
        };
        let ids: Vec<String> = (0..30).map(|i| format!("c{:02}", i)).collect();
        for cid in &ids { state.record_dispatched(cid, 0.0); }
        let mut sched: Vec<(String, f64)> = Vec::new();
        for (i, t) in fast.iter().enumerate() { sched.push((format!("c{:02}", i), *t)); }
        for (i, t) in stragglers.iter().enumerate() { sched.push((format!("c{:02}", 28 + i), *t)); }
        let mut spec: Vec<(String, f64)> = Vec::new();
        let mut t = 0.0_f64;
        while t <= 450.0 {
            for (cid, td) in &sched { if *td <= t + 5.0 { state.record_completed(cid, *td); } }
            for cid in &ids {
                if let Some(_e) = state.decide_speculative(cid, t, &cfg) {
                    spec.push((cid.clone(), t));
                    state.record_speculative_dispatched(cid);
                }
            }
            t += 10.0;
        }
        let pickup = 60.0_f64;
        let mut sim = 0.0_f64;
        for (cid, t_orig) in &sched {
            let st = spec.iter().find(|(c, _)| c == cid).map(|(_, t)| *t + pickup);
            let eff = match st { Some(s) => t_orig.min(s), None => *t_orig };
            if eff > sim { sim = eff; }
        }
        let r = ((baseline - sim) / baseline) * 100.0;
        let first = spec.first().map(|(_, t)| *t).unwrap_or(-1.0);
        println!("{:.2}    {:>3}     {:>6.1}  {:>10.1}  {:+.1}%", factor, spec.len(), first, sim, r);
    }
}
