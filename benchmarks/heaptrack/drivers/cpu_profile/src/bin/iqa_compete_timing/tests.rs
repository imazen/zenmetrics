#[path = "mod.rs"]
mod timing;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;
use timing::cold_call;

#[test]
fn cold_includes_constructor_and_calls_once() {
    let constructed = AtomicUsize::new(0);
    let calls = Arc::new(AtomicUsize::new(0));
    let observed_calls = calls.clone();
    let constructor_delay = Duration::from_millis(10);
    let (value, elapsed_ns) = cold_call(|| {
        constructed.fetch_add(1, Ordering::SeqCst);
        std::thread::sleep(constructor_delay);
        Ok(Box::new(move || {
            observed_calls.fetch_add(1, Ordering::SeqCst);
            Ok(7.0)
        }))
    });
    assert_eq!(value.unwrap().to_bits(), 7.0_f64.to_bits());
    assert_eq!(constructed.load(Ordering::SeqCst), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(elapsed_ns >= constructor_delay.as_nanos());
}

#[test]
fn cold_constructor_and_call_refusals_remain_errors() {
    assert_eq!(
        cold_call(|| Err::<fn() -> Result<f64, String>, _>("constructor refusal".to_owned()))
            .0
            .unwrap_err(),
        "constructor refusal"
    );
    assert_eq!(
        cold_call(|| Ok(Box::new(|| Err("call refusal".to_owned()))))
            .0
            .unwrap_err(),
        "call refusal"
    );
}
