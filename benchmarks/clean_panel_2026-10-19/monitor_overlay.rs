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
thread_local! {
    static READ: std::cell::RefCell<Option<Value>> = const { std::cell::RefCell::new(None) };
    static ENTRY: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}
pub(crate) fn stamp(time: Instant) -> u64 { time.duration_since(*CLOCK.get_or_init(Instant::now)).as_nanos().try_into().unwrap() }
pub(crate) fn publication(snapshot: &Value,time: Instant) {
    emit("publication",&json!({"snapshot_id":snapshot["snapshot_id"],"publication_monotonic_ns":stamp(time)}));
}
pub(crate) fn begin() { ENTRY.with(|t|t.set(now()));READ.with(|r|*r.borrow_mut()=None); }
pub(crate) fn read(snapshot: Option<&Value>,publication: Option<Instant>,observed: Instant,age: Option<std::time::Duration>,status: &str) {
    READ.with(|r|*r.borrow_mut()=Some(json!({"snapshot_id":snapshot.map(|s|s["snapshot_id"].clone()),"publication_monotonic_ns":publication.map(stamp),"observed_monotonic_ns":stamp(observed),"age_ns":age.map(|a|a.as_nanos().min(u64::MAX as u128) as u64),"status":status})));
}
pub(crate) fn check(snapshot: &Value) -> String {
    use sha2::{Digest,Sha256};
    let t=now();let bytes=serde_json::to_vec(snapshot).unwrap();
    let ids:Vec<_>=snapshot["processes"].as_array().unwrap().iter().filter(|p|p["counted"]==true).map(|p|json!([p["pid"],p["start_ticks"]])).collect();
    let id=format!("c{}",NEXT.fetch_add(1,Ordering::SeqCst));
    emit("check",&json!({"check_id":id,"snapshot_id":snapshot["snapshot_id"],"snapshot_sha256":Sha256::digest(bytes).iter().map(|b|format!("{b:02x}")).collect::<String>(),"checked_monotonic_ns":t,"observation_age_ns":t-snapshot["sample_monotonic_ns"].as_u64().unwrap(),"heavy_process_count":snapshot["heavy_process_count"],"counted_identities":ids,"age_scope":"hook minus refresh sample; not gate cached age"}));
    id
}
pub(crate) fn gate(state: Option<&crate::platform::SystemState>,age: Option<u64>,clean: Option<bool>,reason: Option<String>) {
    let check_id=state.map(|s|check(&s.attribution));
    let fields=state.map(|s|json!({"cpu_load":s.cpu_load,"available_ram_bytes":s.available_ram_bytes,"total_ram_bytes":s.total_ram_bytes,"cpu_temp_c":s.cpu_temp_c,"heavy_process_count":s.heavy_process_count}));
    emit("gate",&json!({"gate_id":format!("g{}",NEXT.fetch_add(1,Ordering::SeqCst)),"entry_monotonic_ns":ENTRY.with(|t|t.get()),"hook_monotonic_ns":now(),"read":READ.with(|r|r.borrow().clone()),"gate_observation_age_ns":age,"status":if clean.is_some(){"decision"}else{"unknown"},"clean":clean,"reason":reason,"state":fields,"check_id":check_id}));
}

#[cfg(test)]
mod controls {
    #[test]
    fn owned_nonmetric_monitor() {
        use std::io::BufRead;
        let _=super::now();
        println!("MONITOR_READY");std::io::Write::flush(&mut std::io::stdout()).unwrap();
        let mut input=String::new();std::io::stdin().lock().read_line(&mut input).unwrap();assert_eq!(input.trim(),"GO");
        let mut gate=crate::gate::ResourceGate::new(crate::gate::GateConfig::default());
        gate.start_monitoring();
        for _ in 0..40 {assert!(gate.check_and_record_status().is_some());std::thread::sleep(std::time::Duration::from_millis(5));}
        // Stop this owned sampler; retain its final real publication and wait
        // for normal stale-age logic. No timestamp rewriting or forced refresh.
        gate.control_stop_sampler();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        assert!(gate.check_and_record_status().is_none());
        // Explicit test-only missing-cache injection exercises the real branch.
        gate.control_clear_cache();
        assert!(gate.check_and_record_status().is_none());
        drop(gate);
    }
}
