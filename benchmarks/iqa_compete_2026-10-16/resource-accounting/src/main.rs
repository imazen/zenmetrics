//! Diagnostic observations only: no benchmark gates or timing qualification.
#![forbid(unsafe_code)]
use serde_json::json;
use std::{
    hint::black_box,
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime},
};
use sysinfo::{ProcessesToUpdate, System};

fn work(stop: &AtomicBool, until: Option<Instant>) {
    let mut x = 0.1234567_f64;
    while !stop.load(Ordering::Relaxed) && until.is_none_or(|end| Instant::now() < end) {
        for _ in 0..1000 {
            x = (black_box(x) + 0.00001).sin().abs();
        }
    }
    black_box(x);
}
fn native_time(pid: u32) -> String {
    let out = Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "time="])
        .output()
        .expect("native ps");
    assert!(out.status.success(), "native ps failed");
    String::from_utf8(out.stdout)
        .expect("ps UTF8")
        .trim()
        .to_owned()
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() == 4,
        "usage: iqa-resource-accounting idle|stable|churn WORKERS SAMPLES"
    );
    let mode = args[1].as_str();
    assert!(["idle", "stable", "churn"].contains(&mode));
    let workers: usize = args[2].parse().expect("workers");
    let samples: usize = args[3].parse().expect("samples");
    assert!((1..=8).contains(&workers) && (1..=300).contains(&samples));
    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();
    if mode == "stable" {
        for _ in 0..workers {
            let stop = stop.clone();
            handles.push(thread::spawn(move || work(&stop, None)));
        }
    } else if mode == "churn" {
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                thread::scope(|scope| {
                    for _ in 0..workers {
                        let stop = &stop;
                        scope.spawn(move || {
                            work(stop, Some(Instant::now() + Duration::from_millis(20)))
                        });
                    }
                });
            }
        }));
    }
    let pid = std::process::id();
    let owner = sysinfo::Pid::from_u32(pid);
    let mut sys = System::new_all();
    let start = Instant::now();
    let mut previous_owner = sys.process(owner).expect("owner").accumulated_cpu_time();
    let mut previous_wall = Instant::now();
    let mut previous_native = native_time(pid);
    println!(
        "{}",
        json!({"schema":1,"mode":mode,"workers":workers,"samples":samples,"owner":pid,"os":std::env::consts::OS,"sysinfo":"0.36.1","interval_ms":200,"qualification":false,"native_time":"ps cumulative user+system CPU, centisecond display; independent bounded read adjacent to sysinfo refresh","order":"refresh_cpu_all; refresh_memory; refresh_processes(All,true), matching zenbench ac675d3"})
    );
    for index in 0..samples {
        thread::sleep(Duration::from_millis(200));
        let refresh = Instant::now();
        sys.refresh_cpu_all();
        sys.refresh_memory();
        sys.refresh_processes(ProcessesToUpdate::All, true);
        let now = Instant::now();
        let p = sys.process(owner).expect("owner alive");
        let accumulated = p.accumulated_cpu_time();
        let wall_ms = now.duration_since(previous_wall).as_secs_f64() * 1000.0;
        let cores = sys.cpus().len();
        let global = sys
            .cpus()
            .iter()
            .map(|c| f64::from(c.cpu_usage()))
            .sum::<f64>()
            / cores as f64
            / 100.0;
        let own = f64::from(p.cpu_usage()) / cores as f64 / 100.0;
        let native = native_time(pid);
        let processes:Vec<_>=sys.processes().values().filter(|p|p.cpu_usage()>0.0 || p.pid()==owner).map(|p|json!({"pid":p.pid().as_u32(),"parent":p.parent().map(|p|p.as_u32()),"cpu_pct":p.cpu_usage(),"accumulated_cpu_ms":p.accumulated_cpu_time(),"thread_kind":format!("{:?}",p.thread_kind())})).collect();
        println!(
            "{}",
            json!({"sample":index,"unix_ms":SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis(),"elapsed_ms":start.elapsed().as_secs_f64()*1000.0,"refresh_ms":refresh.elapsed().as_secs_f64()*1000.0,"interval_wall_ms":wall_ms,"cores":cores,"global_fraction":global,"per_cpu_pct":sys.cpus().iter().map(|c|c.cpu_usage()).collect::<Vec<_>>(),"owner_pct":p.cpu_usage(),"owner_accumulated_cpu_ms":accumulated,"owner_delta_cpu_ms":accumulated.saturating_sub(previous_owner),"owner_delta_fraction":accumulated.saturating_sub(previous_owner) as f64/wall_ms/cores as f64,"zenbench_foreign_fraction":(global-own).max(0.0),"native_ps_cpu_before":previous_native,"native_ps_cpu_after":native,"processes":processes})
        );
        previous_owner = accumulated;
        previous_wall = now;
        previous_native = native;
    }
    stop.store(true, Ordering::Relaxed);
    for handle in handles {
        handle.join().expect("payload");
    }
}
