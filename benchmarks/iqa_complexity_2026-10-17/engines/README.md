# NN metrics across engines, i265, 1024×1024 (2026-10-17)

Same models, same input pair, same thread budget (8 threads on P-cores 0-7); only the engine
differs. rten (pure-Rust CPU) is `../measured/i265_nn_1024.tsv` (8-thread rows). ort-cuda and
ort-webgpu run on the RTX 4080 (driver 595.99.02) through ONNX Runtime 1.31.0.

- Method: `scripts/complexity/perf_deltas.py`, per call = (T(1+K) − T(1)) / K with K chosen so the
  delta covers ≥ 3 s, 3 repeats. A first pass with K = 2 gave negative GPU deltas (session setup
  variance exceeds one call) and is not used.
- `task_clock_ms_per_call` is host CPU time, `wall_ms_per_call` is elapsed time. Where CPU time
  is far above wall time on a GPU engine (DMM, WD, PieAPP, WaDIQaM-FR, PaQ-2-PiQ), the model runs
  part of its work on the CPU (SVD, patch loops or resize outside the graph).
- WebGPU: A-FINE, ARNIQA, CKDN, DeepDC and MUSIQ contain operators the WebGPU execution provider
  does not support; CPU fallback is disabled on purpose, so those rows are `error`.
- DreamSim's rows come from a model directory holding its external-data files (same ONNX sha256
  as the frozen set); the frozen directory's symlinks omit them.
- Scores agree with rten to ≤ 1.5e-5 on every model except DMM (7.8e-4, its SVD).
- Models: frozen ONNX set `models-df35ce6e87bbf1af`; consent list
  `ckdn,afine,dreamsim,liqe,hyperiqa,paq2piq,pieapp` (timing only).
- Binary: complexity-reps with `zenmetrics-cli/nn-webgpu,zenmetrics-cli/nr`, sha256
  6fd495cd38ba2f3b861155d8e2e40b4f92006ffefa103fec61bef90c3d4ecaeb. Command: `run-engines.sh`.
- run-heavy: done rc=0 1061s | peak-RSS 2.15GiB | min-avail 16155MiB | peak-load 9.85
  (2026-10-17T21:30:30Z–21:48:11Z).
