use std::time::Instant;

// Keep construction inside the cold clock: creating a Call may allocate scorer
// scratch, initialize a runtime or prepare a model before its first evaluation.
pub(super) fn cold_call<C: FnMut() -> Result<f64, String>>(
    make: impl FnOnce() -> Result<C, String>,
) -> (Result<f64, String>, u128) {
    let start = Instant::now();
    let mut call = make();
    let result = match &mut call {
        Ok(call) => call(),
        Err(error) => Err(error.clone()),
    };
    let elapsed_ns = start.elapsed().as_nanos();
    (result, elapsed_ns)
}
