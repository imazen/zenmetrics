# cvvdp-gpu pool determinism — 2026-09-26

Lane `cvvdpdet`. Code: pool change `0a010398` + grid fold `9a8326fd` (jj changes `oortkqmt`,
`zmlvwrrt`), measured on base master `9e918021` and then rebased onto `2f5ff8fb` (the 8 upstream
commits touch no Cargo manifest, lock, pin, cvvdp or zenmetrics-api file); "old" = master `9e918021`. Host: RTX 2080 8 GB, driver
580.178.04; wgpu 29.0.4 on Vulkan, CUDA via the system NVRTC 12.4. Everything built at the CI
sibling pins (`LOCK_SNAP_DIR=… scripts/ci/lock.sh --check` PASS) with `--locked`. Raw TSVs, logs
and suites: `/mnt/v/output/zenmetrics/cvvdp_pool_determinism_2026-09-26/raw.tar.gz`
(sha256 `b9b9c85b…6fdd29`).

## What changed

The spatial pool summed with `Atomic<f32>::fetch_add`, so its order was the scheduler's. It is
now two fixed-order passes (`crates/cvvdp-gpu/src/kernels/pool.rs`): `pool_rows_3ch_kernel`, one
256-lane workgroup per band row (lane t sums x = t, t+256, … in order, then a fixed 256→1
tree), and `pool_rows_finalize_kernel`, the same shape over each (level, channel)'s rows.

**Invariant.** A band partial depends only on that band's D values, width and height. Strips cut
on row boundaries, so Full, Mode E and Mode B pool identical D planes to identical bits
regardless of strip geometry, dispatch count or scheduling. The only fixed constant is the
256-lane tree width (part of the definition; both kernels must launch with it). Widths stay f32.

## Gate 1 — determinism (probe `crates/cvvdp-gpu/examples/pool_determinism.rs`)

3 processes × 50 reps per cell; pairs `noise` (the zenmetrics-api `cancel.rs` pair, JOD ≈ 2) and
`mild` (JOD ≈ 10); Mode E/B at `h_body = 128`. `wgpu1strip` = Mode B with `h_body = 1024` (one
strip), because multi-strip Mode B panics on wgpu before and after this change (see CLAUDE.md
Known Bugs). Full mode also hashes the per-band score vector of each rep.

| backend | pair | size | mode | old JOD distinct (per proc max / all) | old band-score vectors distinct (all 150) | new JOD distinct | new band-score distinct | old JOD band [min, max] | new JOD | new − old median | inside |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cuda | mild | 1000x750 | Full | 1 / 1 | 150 | 1 | 1 | [9.9616423, 9.9616423] | 9.9616423 | +0.00e+00 | yes |
| cuda | mild | 1000x750 | Mode B | 1 / 1 | - | 1 | - | [9.9616461, 9.9616461] | 9.9616461 | +0.00e+00 | yes |
| cuda | mild | 1000x750 | Mode E | 1 / 1 | - | 1 | - | [9.9616423, 9.9616423] | 9.9616423 | +0.00e+00 | yes |
| cuda | mild | 1024x1024 | Full | 1 / 1 | 150 | 1 | 1 | [9.9564457, 9.9564457] | 9.9564457 | +0.00e+00 | yes |
| cuda | mild | 1024x1024 | Mode B | 1 / 1 | - | 1 | - | [9.9564457, 9.9564457] | 9.9564457 | +0.00e+00 | yes |
| cuda | mild | 1024x1024 | Mode E | 1 / 1 | - | 1 | - | [9.9564457, 9.9564457] | 9.9564457 | +0.00e+00 | yes |
| cuda | mild | 256x256 | Full | 1 / 1 | 150 | 1 | 1 | [9.9491653, 9.9491653] | 9.9491653 | +0.00e+00 | yes |
| cuda | mild | 256x256 | Mode B | 1 / 1 | - | 1 | - | [9.9491634, 9.9491634] | 9.9491634 | +0.00e+00 | yes |
| cuda | mild | 256x256 | Mode E | 1 / 1 | - | 1 | - | [9.9491653, 9.9491653] | 9.9491653 | +0.00e+00 | yes |
| cuda | noise | 1000x750 | Full | 2 / 2 | 150 | 1 | 1 | [5.0083075, 5.0083079] | 5.0083036 | -4.29e-06 | **no** |
| cuda | noise | 1000x750 | Mode B | 10 / 13 | - | 1 | - | [5.0115566, 5.0115638] | 5.0115414 | -1.86e-05 | **no** |
| cuda | noise | 1000x750 | Mode E | 11 / 13 | - | 1 | - | [5.0083289, 5.0083385] | 5.0083036 | -2.96e-05 | **no** |
| cuda | noise | 1024x1024 | Full | 3 / 3 | 150 | 1 | 1 | [1.8762579, 1.8762598] | 1.8762474 | -1.05e-05 | **no** |
| cuda | noise | 1024x1024 | Mode B | 4 / 4 | - | 1 | - | [1.8762474, 1.8762512] | 1.8762474 | -1.91e-06 | yes |
| cuda | noise | 1024x1024 | Mode E | 6 / 6 | - | 1 | - | [1.8762465, 1.8762522] | 1.8762474 | -1.91e-06 | yes |
| cuda | noise | 256x256 | Full | 2 / 2 | 149 | 1 | 1 | [1.9945707, 1.9945717] | 1.9945707 | +0.00e+00 | yes |
| cuda | noise | 256x256 | Mode B | 2 / 2 | - | 1 | - | [1.9945784, 1.9945803] | 1.9945707 | -7.63e-06 | **no** |
| cuda | noise | 256x256 | Mode E | 3 / 3 | - | 1 | - | [1.9945784, 1.9945812] | 1.9945707 | -9.54e-06 | **no** |
| wgpu | mild | 1000x750 | Full | 1 / 1 | 150 | 1 | 1 | [9.9616423, 9.9616423] | 9.9616423 | +0.00e+00 | yes |
| wgpu | mild | 1000x750 | Mode E | 1 / 1 | - | 1 | - | [9.9901705, 9.9901705] | 9.9901667 | -3.81e-06 | **no** |
| wgpu | mild | 1024x1024 | Full | 1 / 1 | 150 | 1 | 1 | [9.9564457, 9.9564457] | 9.9564457 | +0.00e+00 | yes |
| wgpu | mild | 1024x1024 | Mode E | 1 / 1 | - | 1 | - | [9.9564457, 9.9564457] | 9.9564457 | +0.00e+00 | yes |
| wgpu | mild | 256x256 | Full | 1 / 1 | 150 | 1 | 1 | [9.9491653, 9.9491653] | 9.9491653 | +0.00e+00 | yes |
| wgpu | mild | 256x256 | Mode E | 1 / 1 | - | 1 | - | [9.9491653, 9.9491653] | 9.9491653 | +0.00e+00 | yes |
| wgpu | noise | 1000x750 | Full | 2 / 2 | 150 | 1 | 1 | [5.0083089, 5.0083094] | 5.0083051 | -3.81e-06 | **no** |
| wgpu | noise | 1000x750 | Mode E | 12 / 12 | - | 1 | - | [4.6037216, 4.6037292] | 4.6037521 | +2.77e-05 | **no** |
| wgpu | noise | 1024x1024 | Full | 2 / 2 | 150 | 1 | 1 | [1.8762598, 1.8762608] | 1.8762503 | -1.05e-05 | **no** |
| wgpu | noise | 1024x1024 | Mode E | 6 / 6 | - | 1 | - | [1.8762493, 1.8762541] | 1.8762503 | -9.54e-07 | yes |
| wgpu | noise | 256x256 | Full | 2 / 2 | 148 | 1 | 1 | [1.9945726, 1.9945736] | 1.9945726 | -9.54e-07 | yes |
| wgpu | noise | 256x256 | Mode E | 4 / 4 | - | 1 | - | [1.9945803, 1.9945831] | 1.9945726 | -8.58e-06 | **no** |
| wgpu1strip | mild | 1000x750 | Mode B | 1 / 1 | - | 1 | - | [9.9616423, 9.9616423] | 9.9616423 | +0.00e+00 | yes |
| wgpu1strip | mild | 1024x1024 | Mode B | 1 / 1 | - | 1 | - | [9.9564457, 9.9564457] | 9.9564457 | +0.00e+00 | yes |
| wgpu1strip | mild | 256x256 | Mode B | 1 / 1 | - | 1 | - | [9.9491653, 9.9491653] | 9.9491653 | +0.00e+00 | yes |
| wgpu1strip | noise | 1000x750 | Mode B | 18 / 18 | - | 1 | - | [5.0083222, 5.0083332] | 5.0083046 | -2.24e-05 | **no** |
| wgpu1strip | noise | 1024x1024 | Mode B | 7 / 7 | - | 1 | - | [1.8762503, 1.8762569] | 1.8762503 | -3.81e-06 | yes |
| wgpu1strip | noise | 256x256 | Mode B | 6 / 6 | - | 1 | - | [1.9926901, 1.9926958] | 1.9926834 | -9.54e-06 | **no** |

Old kernel: the band-score vector differs on **every** call (150 distinct of 150 in every Full
cell); on the mild pair the JOD's f32 rounding at ≈ 10 hides it. New kernel: one JOD and one
band-score vector per cell, identical across the 3 processes. Rerun on the final code (with the
grid fold): all 9 TSVs byte-identical to the pool-only run (`cmp`).

Note the negative-control trap hit on the way: a `CARGO_TARGET_DIR` copied from the new tree made
the first "old" builds link the NEW library (git-archive sources are older than the copied
artifacts, and fingerprints are path-relative). Old binaries were rebuilt after `touch`-ing the
old tree and verified by content (`strings … | grep -c pool_rows_3ch_kernel` = 0).

## Scores moved: outside the old run-to-run band in 13 of 36 cells

New − old median: 12 cells below the old band, 1 above (the wgpu Mode E 1000×750 cell, which is
itself wrong — see Known Bugs), max |shift| **2.96e-5 JOD** (CUDA noise 1000×750 Mode E). CUDA at
4096², noise, Full: old 1.843221 vs new 1.843298 (7.7e-5). The move is toward the exact sum.
`benches/pool_reduction.rs`, every (level, channel) partial vs an f64 host sum of the same
contributions:

| backend | size | old: distinct partial vectors in 20 runs | old max \|rel err\| | old mean rel err | new max \|rel err\| | new mean rel err |
|---|---|---|---|---|---|---|
| wgpu | 256² | 20 | 1.293e-6 | +1.063e-7 | 1.822e-7 | +8.057e-8 |
| wgpu | 1024² | 20 | 2.037e-6 | +5.345e-8 | 1.616e-7 | +8.062e-8 |
| cuda | 256² | 20 | 1.476e-6 | +2.616e-8 | 1.051e-7 | +2.597e-8 |
| cuda | 1024² | 20 | 2.037e-6 | −2.945e-8 | 9.960e-8 | +3.251e-8 |
| cuda | 4096² | 20 | 3.411e-6 | −1.919e-7 | 8.520e-8 | +9.830e-9 |

(Uniform random D in [−2, 2]; the same f32 per-pixel contributions feed both pools. New repeat
runs bit-identical in every row.)

Goldens and CPU parity did not move at their printed precision: shadow JOD vs the pycvvdp
manifest (q = 1…90), the 73×91 / 256² noise / blur / chroma goldens and host-scalar parity print
identical values old vs new on wgpu (4–6 decimals).

## Gate 2 — zenmetrics-api `cancel::` (features `all-metrics,cpu-metrics,wgpu,pixels,encoded`)

- `cancel::gpu_one_shot_pre_cancelled_stop_is_error_cancelled_for_every_metric --exact`, 16 runs
  each: old 3 failed (`Cvvdp left: 4611661575768506368 right: 4611661580063473664`, i.e.
  1.9945726 vs 1.9945736), new 0 failed.
- Full `it` suite, 10 consecutive runs: 10 / 10 green, 43 passed each — on the pool-only binary
  and again on the final binary (sha256 `339f950a…`).

## Gate 3 — cvvdp-gpu `it` suite (`--release`, `parity-goldens`, one process per module)

| build | modules | failing tests |
|---|---|---|
| new, cuda | 68 | 0 |
| new, wgpu | 67 | 8 — all multi-strip Mode B (`mode_b_walker_parity` ×4, `strip_mode_b_csf_halo_parity` ×4): unaligned `offset_start` binding panic |
| old, wgpu | 67 | the same 8, plus `pipeline_color::compute_dkl_jod{,_with_warm_ref}_matches_pycvvdp_at_12mp_synth` (GPU 10.0000 vs golden 9.4580) |

Lib clippy, doctests (17 passed, 16 ignored) and CI's workspace clippy pass.

## Grid fold (`9a8326fd`) — wgpu returned JOD 10 from 2048×2048

Noise pair, Full, wgpu: old 2000×2000 = 4.178282, **2048×2048 = 10.000000, 3000×2000 =
10.000000**, 4096² = `ReadbackFailed`. After: 4.178301 / 1.854582 / 3.804538 / 1.843300, CUDA
4.178299 / 1.854579 / 3.804536 / 1.843298; 0 dispatch-limit errors.

## Perf — CONTENDED (load 20–50 on 32 threads throughout)

`benches/pool_reduction.rs`, zenbench interleaved, pool stage only (the rest of the JOD pipeline
is unchanged), including the partials readback:

| backend | size | old_atomic | new_fixed_order | 95% CI new vs old |
|---|---|---|---|---|
| wgpu | 256² | 1.4 ± 0.2 ms | 1.4 ± 0.2 ms | −7.8% … +3.1% |
| wgpu | 1024² | 3.0 ± 0.1 ms | 3.3 ± 0.3 ms | +1.0% … +11.3% |
| cuda | 256² | 211.6 ± 38.0 µs | 258.4 ± 45.6 µs | −54.9% … +98.6% (CV 53–79%) |
| cuda | 1024² | 601.6 ± 297.3 µs | 526.5 ± 316.2 µs | −78.2% … +48.5% (CV 60–61%) |
| cuda | 4096² | 5.2 ± 0.1 ms | 6.8 ± 3.0 ms | −75.2% … +131.3% (CV 72–84%) |

Only 10 rounds per group survived the noise filter; the CUDA rows are inconclusive. wgpu 4096²
cannot run the old side (its LDS dispatch asks for 65 536 workgroups in x).

## Recompute

```
# determinism probe (per process; 3 processes for the table)
DET_REPS=50 DET_MODES=full,strip DET_HBODY=128 DET_OUT=new_wgpu_p1.tsv \
  cargo run --release -p cvvdp-gpu --no-default-features --features wgpu,cubecl-types --example pool_determinism
DET_REPS=50 DET_MODES=pair DET_HBODY=1024 DET_OUT=new_wgpu1strip_p1.tsv  cargo run … (same)
DET_REPS=50 DET_HBODY=128 DET_OUT=new_cuda_p1.tsv \
  cargo run --release -p cvvdp-gpu --no-default-features --features cuda,cubecl-types --example pool_determinism
python3 benchmarks/cvvdp_pool_determinism_2026-09-26.analyze.py <dir with {old,new}_*_p*.tsv>
# accuracy + perf
cargo bench -p cvvdp-gpu --no-default-features --features wgpu,cubecl-types --bench pool_reduction   # POOL_BENCH_SIZES=256,1024
cargo bench -p cvvdp-gpu --no-default-features --features cuda,cubecl-types --bench pool_reduction
# grid fold
DET_PAIRS=noise DET_SIZES=2000x2000,2048x2048,3000x2000,4096x4096 DET_MODES=full DET_REPS=3 cargo run … --example pool_determinism
```
