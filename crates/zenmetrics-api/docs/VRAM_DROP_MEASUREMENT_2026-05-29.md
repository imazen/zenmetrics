# VRAM drop/reclaim measurement — 2026-05-29

Task #150 (`claude-vram-drop`). Measured proof that the STEP 2 fix
(`reclaim_pooled_vram` + orchestrator swap cleanup) does what the
[audit](./VRAM_DROP_AUDIT_2026-05-29.md) predicted: dropping a metric
alone leaves its VRAM in cubecl's pool (the #147 plateau), and the new
reclaim returns it to the driver — at metric drop (user) and at the
orchestrator's metric-signature swap (peak ≈ MAX not SUM).

Data: `benchmarks/vram_cleanup_2026-05-29.tsv` + `.meta`. Host: RTX 5070
(12 GiB, WSL2), driver 596.21, CudaRuntime, release. Probe: `nvidia-smi
memory.used` (global; per-PID hidden under WSL2), min-of-window per step.
A concurrent `claude-setref-all` GPU process ran during the window — the
qualitative findings (GiB-scale steps, OOM-vs-no-OOM) are robust to its
sub-GiB transients; absolute baselines drift ±300 MiB.

## 1. Drop alone does NOT free; reclaim does (the headline)

`vram_drop_reclaim --scenario drop_reclaim --size 4096` (16 MP butter):

| step                       | used MiB | delta | meaning |
|----------------------------|---------:|------:|---------|
| baseline                   |     3097 |     0 | |
| after construct + score    |     8097 | +5000 | working set resident |
| **after drop**             |   **8097** | **+5000** | **pool plateau — drop freed NOTHING to driver** |
| after `reclaim_pooled_vram`|     3254 |  +157 | **returned 4843 MiB to driver** |

This is the #147 finding made actionable: `drop(metric)` returns handles
to cubecl's pool but the device pages stay resident (+5000 MiB
unchanged); `reclaim_pooled_vram(Backend::Cuda)` hands them back
(post-reclaim +157 MiB ≈ baseline; the residual is leftover pool
pages / PTX cache / CUDA context). `Metric::release(backend)` bundles
drop + reclaim.

## 2. Orchestrator swap cleanup → peak ≈ MAX(single), not SUM

`vram_swap_peak` runs a real `Orchestrator::run_all` chunk
`[cvvdp, cvvdp, ssim2, ssim2, butter, butter]` (each metric twice, so a
same-signature warm reuse sits between the swaps), once with the swap
reclaim ON and once with `ZENMETRICS_NO_SWAP_VRAM_CLEANUP=1`, each in a
cold child subprocess.

**4 MP (2048², all tasks fit):**

| variant     | peak Δ MiB | ok | swap_reclaims |
|-------------|-----------:|---:|--------------:|
| reclaim     |   **+2100** | 6/6 | 2 |
| no_reclaim  |   **+2829** | 6/6 | 0 |

Reclaim cut peak by **729 MiB (1.35× lower)**. `+2100` ≈ the single
largest metric (ssim2 @ 4 MP); `+2829` is the accumulated pool. The 2
`swap_reclaims` correspond to exactly the 2 signature changes
(cvvdp→ssim2, ssim2→butter) — reclaim did NOT fire between the
same-signature pairs.

**16 MP (4096², near the 12 GiB ceiling) — correctness, not just peak:**

| variant     | peak Δ MiB | ok | swap_reclaims |
|-------------|-----------:|---:|--------------:|
| reclaim     |     +7839 | **6/6** | 2 |
| no_reclaim  |     +8408 | **4/6** | 0 |

Without the swap reclaim, **2 of 6 tasks OOM'd** (butter could not
allocate over ssim2's still-pooled pages); with it, all 6 succeeded. At
the high end the fix is correctness-relevant, not merely peak-shaving.
(Absolute peaks here carry the most concurrent-process noise; the
6/6-vs-4/6 outcome is the clean signal.)

## 3. Warm per-call path NOT regressed

`vram_drop_reclaim --scenario warm_loop --size 1024` (1 MP cvvdp,
WARM_N=40, one signature, 40 warm-ref scores):

- warm per-call wall: **p50 = 3.37 ms** (p10 2.89 / p90 8.23)
- VRAM floor **385 MiB** — matches the documented warm_ref footprint in
  `crates/cvvdp-gpu/benchmarks/gpu_vram_sweep_2026-05-28.tsv` (385 MiB)
  exactly
- VRAM span across the 40 scores: **9 MiB** (nvidia-smi quantization) —
  flat, i.e. no per-call growth and **no reclaim churn**

The swap reclaim fires only on signature change / drop, never between
same-signature scores (`swap_vram_reclaim_count()` stays 0 across the
warm loop), so the #145 warm path is untouched. Confirmed both by the
flat VRAM floor and by the per-call timing matching the documented
warm-ref steady state.

## Reproduce

```sh
cargo build -p zenmetrics-api --release --features cuda,all-metrics,pixels \
  --example vram_drop_reclaim
./target/release/examples/vram_drop_reclaim --size 4096 --scenario drop_reclaim \
  --settle-ms 100 --reads 6
WARM_N=40 ./target/release/examples/vram_drop_reclaim --size 1024 \
  --scenario warm_loop

cargo build -p zenmetrics-orchestrator --release --features cuda \
  --example vram_swap_peak
./target/release/examples/vram_swap_peak --size 2048   # 4 MP, all fit
./target/release/examples/vram_swap_peak --size 4096   # 16 MP, OOM-vs-no-OOM
```
