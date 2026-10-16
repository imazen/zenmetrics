# nn-opt3 — CPU host-path speed, MUSIQ / HyperIQA / PaQ-2-PiQ (i265, 2026-10-16)

Lane `oc-perf` off `nn-lpips-vgg@origin` (PR #81). Scope: Rust-side host work
inside `crates/zenmetrics-nn/src/{musiq,hyperiqa,paq2piq}.rs` (preprocess,
resize, crops, patch extraction, RoIPool). The ONNX engine run is out of scope.
Every score must stay bit-identical.

Harness: `crates/zenmetrics-nn/examples/nn_bench.rs`
(`ZENMETRICS_NN_BENCH_MODES=compute,warm`), rten engine, ladder image `1415`,
models `~/tmp/nr-learned-scratch/models`, sizes 256/1024/2048, `RTEN_NUM_THREADS`
1 and 16. `compute` evaluates both ladder images (two preps + two runs); `warm`
caches the reference taps and scores one (one prep + one run). All three models
are `NoReference`, so their `warm` reference caching is a single cached run and
`compute ≈ 2× warm`.

## Host share (baseline) — before any change

Raw stage times: `host_stages_base.tsv`; share of one `warm` evaluation at 16T:
`host_stage_share.tsv`. Measured with an instrumented throwaway build
(`ZENMETRICS_NN_HOST_TIMING=1`), e.g.:

```text
RTEN_NUM_THREADS=16 ZENMETRICS_NN_HOST_TIMING=1 ZENMETRICS_NN_MODEL=<m> \
  ZENMETRICS_NN_BENCH_MODES=compute nn_bench <models> <ladder> /dev/null 1415 2048 0.1
```

16T, `ms` and share of one `warm` evaluation:

| model | size | stage | ms | share |
|---|---|---:|---:|---:|
| musiq | 256 | **prep total** | 2.37 | 3.64% |
| musiq | 1024 | **prep total** | 15.21 | 4.19% |
| musiq | 2048 | **prep total** | 47.59 | 2.26% |
| musiq | 1024 | patchfull (largest single pass) | 5.49 | 1.51% |
| musiq | 1024 | normalise | 4.09 | 1.13% |
| hyperiqa | 256/1024/2048 | prep total | 7.5 / 7.4 / 8.1 | 0.75 / 0.74 / 0.81% |
| paq2piq | 256 | **RoIPool** | 1.60 | 8.98% |
| paq2piq | 1024 | **RoIPool** | 4.34 | 3.78% |
| paq2piq | 2048 | RoIPool | 7.38 | 1.78% |
| paq2piq | 256 | body (engine) | 11.44 | 64.4% |

At 1T every host stage is ≤2.8% of `warm` (engine-bound); the table above is the
relevant case. HyperIQA's whole preprocessing is <0.9% at every size (it always
resizes to a fixed 384×512, then 25 crops of 224²). MUSIQ's preprocessing sums to
3.6–4.2% at 256/1024 but **no single pass inside it reaches 3%**: the largest are
the full-resolution patch extract (5.49 ms, 1.51% at 1024; 25.77 ms, 1.22% at
2048) and the serial normalise (14.13 ms, 0.67% at 2048). The only host *stage*
above 3% is **PaQ-2-PiQ's host RoIPool** (8.98% at 256, 3.78% at 1024).

## Change: `roi_pool` runs the ROIs on scoped worker threads

`crates/zenmetrics-nn/src/paq2piq.rs`. The 401 ROIs (whole image + 20×20 block
grid) are independent: each reads only `feats` and writes only its own `C·4`
slice of the pooled output. The bins for one ROI are factored into `roi_bins`,
the total max-accumulate count (`cells·C`) is summed once for the thread
estimate, and the ROIs run through the existing `par::for_each_row` helper.
Each output element is still its max over the same cells in the same order, so
the result is bit-identical; `threads_for` caps at `workers()`, so
`RTEN_NUM_THREADS=1` stays sequential and starts no thread. A new unit test
(`roi_pool_thread_bit_identical`) asserts sequential/parallel bit identity and
zero helper launches under a one-worker cap.

## A/B (interleaved base, new, base, new; median/min ms)

Files `ab_{a,b}_{base,new}_t{1,16}.tsv`. Round-a/round-b medians, PaQ-2-PiQ:

| mode | size | base a | new a | Δ a | base b | new b | Δ b |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16T compute | 256 | 35.68 | 35.32 | −1.0% | 35.21 | 35.34 | +0.4% |
| 16T compute | 1024 | 224.92 | 217.38 | −3.4% | 233.23 | 223.56 | −4.1% |
| 16T compute | 2048 | 817.81 | 809.75 | −1.0% | 820.42 | 817.20 | −0.4% |
| 16T warm | 256 | 17.50 | 17.29 | −1.2% | 18.03 | 17.31 | −4.0% |
| 16T warm | 1024 | 113.92 | 111.99 | −1.7% | 115.69 | 112.69 | −2.6% |
| 16T warm | 2048 | 414.35 | 409.45 | −1.2% | 416.45 | 405.02 | −2.7% |
| 1T compute | 256 | 109.91 | 114.06 | +3.8% | 114.20 | 109.97 | −3.7% |
| 1T compute | 1024 | 1324.53 | 1318.85 | −0.4% | 1318.03 | 1320.73 | +0.2% |
| 1T compute | 2048 | 5294.02 | 5215.03 | −1.5% | 5224.27 | 5300.29 | +1.5% |
| 1T warm | 256 | 55.21 | 58.35 | +5.7% | 59.21 | 54.61 | −7.8% |
| 1T warm | 1024 | 662.49 | 662.65 | +0.0% | 663.38 | 661.53 | −0.3% |
| 1T warm | 2048 | 2644.46 | 2619.03 | −1.0% | 2642.67 | 2645.63 | +0.1% |

At 16T the win is consistent in sign at 1024/2048 and mostly so at 256; the
1024 compute rows clear the 3% bar (−3.4%, −4.1%). At 1T the medians swing both
directions (engine-bound, ±3–8% noise on this box) while the **minimum** times
are flat: e.g. 1T 256 warm base min 54.71/55.40 vs new 54.76/54.38; the
sequential path is unchanged apart from the trivial `roi_bins` recompute for the
work estimate. At 256 the stage is 9% of `warm` but the thread estimate
(`cells·C = 742k`) only yields 2 workers, so the measured total gain there is
1–4%.

### Score identity

All three models, sizes 256/1024/2048, both thread counts, compute+warm, both
rounds: the full-precision f64 `score` column is byte-identical base-vs-new
(18 rows/file, all identical; diff of the `metric,mode,size,score` projection
empty). The new unit test also checks the pool itself.

## Gates

- `just snap-cargo @ test -p zenmetrics-nn` — 56 passed; release likewise
  (`~/tmp/oc-perf-scratch/gate_test_{debug,release}.log`).
- CI lint: `just snap-cargo @ clippy --workspace --no-default-features
  --features wgpu,all-metrics -- -D warnings …` — clean
  (`~/tmp/oc-perf-scratch/gate_clippy.log`).
- `just nn-parity … rten model-tests "musiq,hyperiqa,paq2piq"` — worst |d|
  musiq 1.20e-4 JND, hyperiqa 3.42e-6, paq2piq 2.51e-5 (gate 1e-3)
  (`~/tmp/oc-perf-scratch/gate_parity.log`).

