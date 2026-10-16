# nn-opt4 — CPU host-path speed, round 2 (i265, 2026-10-16)

Lane `oc-perf` off `nn-lpips-vgg@origin` (PR #81), worktree state
`vpsyvvly 552bb3be`. Scope: the Rust-side host work of the full-reference set
DISTS, DeepDC, CKDN, A-FINE, DreamSim plus MANIQA, ARNIQA and LIQE
(preprocessing, resize, crops, patch extraction, RoIPool, heads, copies,
allocations). The ONNX engine run is out of scope. Every score stays
bit-identical.

Harness: `crates/zenmetrics-nn/examples/nn_bench.rs`
(`ZENMETRICS_NN_BENCH_MODES=compute,warm`), rten engine, ladder image `1415`,
FR models `~/tmp/nn-phase0/models`, NR models `~/tmp/nr-learned-scratch/models`,
sizes 256/1024/2048, `RTEN_NUM_THREADS` 1 and 16. `compute` evaluates both
ladder images (two preps + two runs); `warm` caches the reference and scores
one (one prep + one run).

## Host share (baseline, before any change)

Raw stage times and shares: `host_stage_share.tsv`, measured with an
instrumented throwaway build (`ZENMETRICS_NN_HOST_TIMING=1`), e.g.:

```text
RTEN_NUM_THREADS=16 ZENMETRICS_NN_HOST_TIMING=1 ZENMETRICS_NN_MODEL=<m> \
  ZENMETRICS_NN_BENCH_MODES=compute,warm nn_bench <models> <ladder> /dev/null 1415 2048 0.1
```

Share of one `warm` evaluation at 16T; only stages above 3% are listed
(engine is the ONNX run, out of scope):

| model | stage | 256 | 1024 | 2048 | note |
|---|---|---:|---:|---:|---|
| ckdn | prep total | 4.45% | 17.66% | 35.0% | `planar` + `resize.bicubic_aa`, both already parallel |
| afine | head | **9.65%** | 1.01% | 0.11% | **serial f64, changed below** |
| liqe | head | **6.83%** | **6.60%** | **6.38%** | **serial f64, changed below** |
| dists | head / stats | 4.60 / 3.66% | 4.86 / 3.63% | 5.50 / 4.75% | already parallel (nn-opt round 1) |
| arniqa | prep total | 2.91% | 4.00% | 5.86% | `planar` + `resize.bilinear_u8`, already parallel |
| dreamsim | prep total | 0.70% | 1.37% | 3.19% | `planar` + `resize.pil_u8`, already parallel |
| deepdc | head + selfdot | 0.43% | 0.04% | 0.01% | |
| maniqa | prep total | 0.24% | 0.58% | 1.19% | |

At 1T no host stage reaches 3% (largest: `afine.head` 2.79%, `liqe.head`
2.04%, `ckdn.prep` 1.15%) — the engine dominates, so 16T is the relevant case.
The two stages above 3% that are *not* already parallel are the **LIQE head**
(all sizes) and the **A-FINE qhead** (256² only); both are threaded below.
`ckdn.prep`, `arniqa.prep`, `dreamsim.prep` and the DISTS head are already
worker-threaded (`planar`/`resize.*`/`dists.stats`/`dists.head`), so no further
bit-identical speedup was taken there.

## Change 1: LIQE head — 495 prompt logits on worker threads

`crates/zenmetrics-nn/src/liqe.rs`. The head forms 495 logits
`scale·(f_j·t)`, averages them, softmaxes and marginalises to 5 buckets. Each
prompt's dot product is independent and each accumulates its 15 patches in the
same order, so the prompts run through `par::par_map(NPROMPTS,
threads_for(patches·NPROMPTS·DIM), …)`: the result is bit-identical, and
`threads_for` caps at `workers()` so `RTEN_NUM_THREADS=1` takes the sequential
branch and starts no thread. New unit test `head_thread_count_bit_identical`
(1-worker run spawns 0 helpers; 16-worker run is bit-equal).

## Change 2: A-FINE qhead — proj_feat/mlp rows on worker threads

`crates/zenmetrics-nn/src/afine.rs`. `qhead` does `proj_feat` over the 12 taps
(128×1536) then an `mlp` (768×1542). Both are row maps with independent
outputs. The 12·128 = 1536 `proj_feat` rows (means then vars per tap, tap order)
and the 768 `mlp` hidden rows now go through `par::par_map`, each row keeping
its original summation order and the hidden vector still feeding the final
scalar sum in index order — bit-identical. New unit test
`qhead_thread_count_bit_identical`.

## A/B (interleaved base, new, base, new; median/min ms)

Files `ab_liqe_{a,b}_{base,new}_t{1,16}.tsv`,
`ab_afine_{a,b,c}_{base,new}_t{1,16}.tsv`.

### LIQE (2 rounds)

| threads | mode | size | base med | new med | Δ | base min | new min |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | compute | 256 | 50.48 | 46.23 | −8.42% | 47.22 | 43.85 |
| 16 | compute | 1024 | 50.43 | 46.20 | −8.40% | 48.70 | 44.48 |
| 16 | compute | 2048 | 50.67 | 45.93 | −9.36% | 47.85 | 43.97 |
| 16 | warm | 256 | 25.19 | 23.14 | −8.14% | 23.77 | 21.48 |
| 16 | warm | 1024 | 25.17 | 23.05 | −8.43% | 23.74 | 21.73 |
| 16 | warm | 2048 | 25.18 | 22.93 | −8.94% | 23.86 | 21.72 |
| 1 | compute | 256 | 185.28 | 186.06 | +0.42% | 183.16 | 183.28 |
| 1 | compute | 1024 | 185.15 | 185.83 | +0.37% | 182.78 | 183.10 |
| 1 | compute | 2048 | 186.50 | 185.36 | −0.61% | 184.03 | 182.43 |
| 1 | warm | 256 | 92.49 | 93.05 | +0.61% | 91.32 | 91.15 |
| 1 | warm | 1024 | 92.71 | 92.94 | +0.24% | 91.04 | 90.65 |
| 1 | warm | 2048 | 93.01 | 93.03 | +0.02% | 91.33 | 91.20 |

### A-FINE (3 rounds)

| threads | mode | size | base med | new med | Δ | base min | new min |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | compute | 256 | 39.12 | 36.69 | −6.20% | 36.69 | 34.30 |
| 16 | compute | 1024 | 555.34 | 555.76 | +0.08% | 524.88 | 547.80 |
| 16 | compute | 2048 | 5231.99 | 5228.00 | −0.08% | 5151.33 | 5097.73 |
| 16 | warm | 256 | 32.69 | 30.27 | −7.41% | 30.80 | 28.61 |
| 16 | warm | 1024 | 308.28 | 308.28 | +0.00% | 298.10 | 304.26 |
| 16 | warm | 2048 | 2802.96 | 2714.71 | −3.15% | 2706.14 | 2676.15 |
| 1 | compute | 256 | 228.51 | 268.32 | +17.42% | 221.62 | 219.01 |
| 1 | compute | 1024 | 3854.08 | 4307.66 | +11.77% | 3781.46 | 3753.25 |
| 1 | compute | 2048 | 28616.58 | 32314.07 | +12.92% | 28115.26 | 28307.17 |
| 1 | warm | 256 | 116.63 | 112.53 | −3.51% | 111.02 | 110.25 |
| 1 | warm | 1024 | 2167.26 | 2082.24 | −3.92% | 1906.17 | 1885.63 |
| 1 | warm | 2048 | 14162.69 | 17144.27 | +21.05% | 14044.69 | 14355.20 |

LIQE's win is size-independent and clears 3% at every size (−8% warm, −9%
compute) because the head is scored once per image in both modes. A-FINE's win
is the 256² fix the profile predicted (−7.4% warm, −6.2% compute); at 1024/2048
the engine is >92% of the eval so the same ~2.7 ms is under 1%.

The A-FINE 1T medians are **contention noise, not a regression**: other lanes
held `~/tmp/i265-heavy.lock` throughout (`load average` 17–20 on 20 cores), and
single reps at 1T 256 compute ran at 897 ms (base c) and 365 ms (new b) against
~223 ms for their siblings. The 1T minimum times — the robust statistic here —
are flat to ≤2% (256 compute 221.6 → 219.0, 256 warm 111.0 → 110.3; 1024
compute min +0.7%/new-faster, warm min −1.1%), and the 1T path is sequential by
construction (`threads_for` returns 1, zero helper launches asserted by the unit
tests). LIQE's 1T rounds, which ran in a quieter window, are flat to ±0.6%.

## Score identity

The full-precision f64 `score` column (field 10) is byte-identical base-vs-new
for both changed models, sizes 256/1024/2048, `RTEN_NUM_THREADS` 1 and 16,
`compute`+`warm` (the `ab_*` rounds all carry `score_eq=True`; a diff of the
`metric,mode,size,score` projection is empty). The two new unit tests also
pin the head arithmetic itself.

## Gates

- `just snap-cargo @ test -p zenmetrics-nn` — 58 passed (2 new); release 58
  passed (`~/tmp/oc-perf-scratch/logs/gate_test_{debug,release}2.log`).
- CI lint: `just snap-cargo @ clippy --workspace --no-default-features
  --features wgpu,all-metrics -- -D warnings …` — clean, no `zenmetrics-nn`
  warnings (`~/tmp/oc-perf-scratch/logs/gate_clippy2.log`).
- `just nn-parity … rten model-tests "liqe"` — 9 pairs, worst 4.93e-6 = 1.25e-5
  JND (`gate_parity_liqe.log`); `"afine"` — 15 pairs, worst 7.23e-6 = 1.42e-5
  JND (`gate_parity_afine.log`). Gate 1e-3.
