// Times tract-onnx on fixed-shape ONNX models, single-thread and multithread-mm executor.
use std::time::Instant;
use tract_onnx::prelude::*;
fn main() -> TractResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let (path, hw, threads, iters): (&str, usize, usize, usize) =
        (&args[1], args[2].parse()?, args[3].parse()?, args[4].parse()?);
    if threads > 1 {
        tract_linalg::multithread::set_default_executor(tract_linalg::multithread::Executor::multithread(threads));
    }
    let model = tract_onnx::onnx()
        .model_for_path(path)?
        .with_input_fact(0, f32::fact([1, 3, hw, hw]).into())?
        .into_optimized()?
        .into_runnable()?;
    let x: Tensor = tract_ndarray::Array4::<f32>::from_elem((1, 3, hw, hw), 0.5).into();
    let mut times = vec![];
    for i in 0..iters + 2 {
        let t = Instant::now();
        let _ = model.run(tvec!(x.clone().into()))?;
        if i >= 2 { times.push(t.elapsed().as_secs_f64() * 1e3); }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{path}\tthreads={threads}\tmin={:.1}ms\tmed={:.1}ms", times[0], times[times.len() / 2]);
    Ok(())
}
