# nn host-side optimization — PieAPP/WD (2026-10-17)

Host-side execution time of the PieAPP and WD neural metrics at 1024×1024 on
i265 (Core Ultra 7 265K, Arrow Lake: 8 P-cores + 12 E-cores), 8 threads,
CPUs 0-7. Goal: reduce host CPU per call without changing scores, public
APIs, ONNX graphs or goldens.

## Method

- Binary: `benchmarks/heaptrack/drivers/cpu_profile` `complexity-reps` bin,
  release + debuginfo, `zenmetrics-cli/nn-ort` feature.
- `scripts/complexity/perf_deltas.py` — per-call deltas `(T(1+K)-T(1))/K`,
  3 repeats, `--threads 8 --cpu 0-7`; `pieapp` rten uses `--extra 1`
  (~98 s/call). Median of repeats reported.
- Engines: `ZENMETRICS_NN_ENGINE=rten` (`RTEN_NUM_THREADS=8`) and
  `ort-cuda` (ORT 1.31.0, TF32 off, exhaustive cuDNN search,
  deterministic compute). `ZENMETRICS_NN_ACCEPT_WEIGHTS=pieapp`.
- Models: `pieapp.onnx` review revision 3 (sha256
  `e3905d36…8653`), `wd_trunk.onnx`; goldens + ladder under
  `~/tmp/cx-nnhost` and `~/tmp/nn-phase0/ladder`.
- Profiles: `perf record -F 249 -g`, taskset 0-7.

## Baseline (pre-change)

| metric | engine | task-clock ms/call | wall ms/call |
|---|---|---|---|
| pieapp | ort-cuda | 3833 | 3807 |
| pieapp | rten | 571299 | 98524 |
| wd | ort-cuda | 4935 | 1842 |
| wd | rten | 14399 | 2554 |

## Hotspots (perf record, pre-change)

- pieapp ort-cuda: 91.6% of samples at one libonnxruntime PC decoding to
  `tpause` — ORT intra-op pool spin-wait (`session.intra_op.allow_spinning`
  defaults on). Host CPU ≈ wall; patch extraction <0.2%.
- wd ort-cuda: 93.8% `tpause`; Rust head below noise.
- wd rten: `wd::lowpass` 17.4%, head closure 3.9%, decimate 1.4%, memset
  2.1%; rten FmaKernel GEMM ~53% over 8 workers. Host head ≈ 25%.
- pieapp rten: ~85% inside rten GEMM (FmaKernel + im2col); compute-bound.

## Attempts

See `attempts.tsv`. One change per row set; retained only when scores are
bit-identical and parity/clippy/fmt gates pass.
