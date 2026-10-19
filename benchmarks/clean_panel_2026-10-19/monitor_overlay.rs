//! Isolated diagnostic instrumentation only; never a production timing cohort.
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, io::Write, sync::{Mutex, OnceLock, atomic::{AtomicU64, Ordering}}, time::Instant};
static CLOCK: OnceLock<Instant> = OnceLock::new();
static NEXT: AtomicU64 = AtomicU64::new(1);
static OUTPUT: Mutex<()> = Mutex::new(());
pub(crate) fn now() -> u64 { CLOCK.get_or_init(Instant::now).elapsed().as_nanos().try_into().unwrap() }
fn proc_row(pid: u32) -> Option<Value> {
    let raw=fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let f:Vec<_>=raw[raw.rfind(')')?+2..].split_whitespace().collect();
    Some(json!({"pid":pid,"start_ticks":f[19].parse::<u64>().ok()?,"pgid":f[2].parse::<u32>().ok()?,"sid":f[3].parse::<u32>().ok()?,"ticks":f[11].parse::<u64>().ok()?+f[12].parse::<u64>().ok()?,"cgroup":fs::read_to_string(format!("/proc/{pid}/cgroup")).ok()?}))
}
pub(crate) fn before() -> (u64,BTreeMap<u32,Value>) {
    let t=now();let mut rows=BTreeMap::new();
    for entry in fs::read_dir("/proc").unwrap().flatten() {
        if let Ok(pid)=entry.file_name().to_string_lossy().parse::<u32>() { if let Some(row)=proc_row(pid) { rows.insert(pid,row); } }
    }
    (t,rows)
}
pub(crate) fn emit(kind: &str,value: &Value) {
    let _guard=OUTPUT.lock().unwrap();
    let path=std::env::var("CLEAN_PANEL_TRACE").expect("diagnostic trace required");
    let mut f=fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
    writeln!(f,"{}",json!({"kind":kind,"value":value})).unwrap();f.flush().unwrap();
}
pub(crate) fn snapshot(before: (u64,BTreeMap<u32,Value>),sys: &sysinfo::System,owned: &[sysinfo::Pid],tasks: &[sysinfo::Pid],count: usize) -> Value {
    let t=now();let mut rows=Vec::new();let mut errors=Vec::new();
    for p in sys.processes().values() {
        let own=owned.contains(&p.pid());let counted=!own && !tasks.contains(&p.pid()) && p.cpu_usage()>10.0;
        if !counted && !own {continue;}
        let pid=p.pid().as_u32();
        match (before.1.get(&pid),proc_row(pid)) {
            (Some(a),Some(mut b)) if a["start_ticks"]==b["start_ticks"] => {
                b["cpu_ticks_before"]=a["ticks"].clone();b["cpu_ticks_after"]=b["ticks"].clone();
                b["interval_ns"]=json!(t-before.0);b["counted"]=json!(counted);b["owned"]=json!(own);
                b["cpu_pct"]=json!(p.cpu_usage());b["ownership_basis"]=json!(if own {"monitor registered owner"} else {"monitor unregistered non-task >10%"});rows.push(b);
            },
            _=>errors.push(pid),
        }
    }
    rows.sort_by_key(|p|p["pid"].as_u64());
    let value=json!({"snapshot_id":format!("s{}",NEXT.fetch_add(1,Ordering::SeqCst)),"sample_monotonic_ns":t,"counter_interval_start_ns":before.0,"processes":rows,"heavy_process_count":count,"identity_errors":errors,"counter_scope":"proc interval brackets refresh; not sysinfo internal CPU denominator"});
    emit("snapshot",&value);value
}
pub(crate) fn check(snapshot: &Value) {
    use sha2::{Digest,Sha256};
    let t=now();let bytes=serde_json::to_vec(snapshot).unwrap();
    let ids:Vec<_>=snapshot["processes"].as_array().unwrap().iter().filter(|p|p["counted"]==true).map(|p|json!([p["pid"],p["start_ticks"]])).collect();
    emit("check",&json!({"check_id":format!("c{}",NEXT.fetch_add(1,Ordering::SeqCst)),"snapshot_id":snapshot["snapshot_id"],"snapshot_sha256":Sha256::digest(bytes).iter().map(|b|format!("{b:02x}")).collect::<String>(),"checked_monotonic_ns":t,"observation_age_ns":t-snapshot["sample_monotonic_ns"].as_u64().unwrap(),"heavy_process_count":snapshot["heavy_process_count"],"counted_identities":ids}));
}

#[cfg(test)]
mod controls {
    #[test]
    fn owned_nonmetric_monitor() {
        use std::io::BufRead;
        let _=super::now();
        println!("MONITOR_READY");std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let mut input=String::new();std::io::stdin().lock().read_line(&mut input).unwrap();assert_eq!(input.trim(),"GO");
        // Actual cached monitor and ResourceGate path, no benchmark/scorer.
        let mut gate=crate::gate::ResourceGate::new(crate::gate::GateConfig::default());
        gate.start_monitoring();
        for _ in 0..40 {gate.check_and_record_status();std::thread::sleep(std::time::Duration::from_millis(5));}
        drop(gate);
    }
}
