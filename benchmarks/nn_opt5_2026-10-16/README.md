# nn-opt5 — WD host statistics head (i265, 2026-10-16)

Lane `oc-perf`, Round 3, off `nn-lpips-vgg@origin` (WD merge `51d804f3`). Scope: the
Rust-side statistics head of `crates/zenmetrics-nn/src/wd.rs` (`head`, `channel`,
`weights`, `lowpass`, `decimate`). The ONNX trunk run is out of scope. Every score
stays byte-identical.

Harness: `crates/zenmetrics-nn/examples/wd_score.rs` (`machine wd_score MODEL_DIR
PAIRS_TSV OUT_TSV rten`), model dir `~/tmp/wd-port/models`, ladder
`~/tmp/nn-phase0/ladder` (`1415_{256,1024,2048}_ref/dist.png`), 3000×3000 area-cap
pair `~/tmp/wd-cap/pairs.tsv`, sizes 256/1024/2048, `RTEN_NUM_THREADS` 1 and 16.
The example runs one `warm_reference` (trunk) plus one `evaluate` (trunk + head)
and retries `evaluate` until two runs agree bitwise, so scores are reproducible.

## Host share of one evaluation

Measured with a throwaway instrumented build (`ZENMETRICS_NN_HOST_TIMING=1`; the
per-stage wrapper was removed before committing), raw totals in
`host_stage_share.tsv`. `wd.head` and `wd.trunk` are wall times (each /2 heads
and /3 trunk calls per process); `wd.channel`/`wd.lowpass`/`wd.decimate`/
`wd.weights` are nested inside `head` and `channel` runs on up to 4 workers, so
their *summed* wall can exceed the `head` wall (it is parallel work, not serial).

Command (e.g. 2048, 16T):

```text
RTEN_NUM_THREADS=16 ZENMETRICS_NN_HOST_TIMING=1 wd_score_instr \
  ~/tmp/wd-port/models ~/tmp/oc-perf-scratch/wd/pairs_2048.tsv /dev/null rten
```

Logs: `~/tmp/oc-perf-scratch/logs/wd_prof_{256,1024,2048}_t{1,16}.log`.

Per-head / per-trunk wall (`ms`) and the host head's share of `head+trunk`:

| size | threads | head ms | trunk ms | head share |
|---|---:|---:|---:|---:|
| 256 | 16 | 76.95 | 45.66 | 62.8% |
| 1024 | 16 | 1116.1 | 539.0 | 67.4% |
| 2048 | 16 | 4704.4 | 2071.5 | 69.4% |
| 256 | 1 | 202.7 | 296.8 | 40.6% |

The host head is 63–69% of the head+trunk wall at 16T; `channel` (the per-channel
statistics loop, parallelised over at most 4 channels by
`par::threads_for(c*plane).min(4)`) dominates it, and `lowpass` is the largest
single pass inside `channel`. Only the 1T/1024 and 1T/2048 profile rows are
missing (`wd_prof_1024_t1.log`/`wd_prof_2048_t1.log` are empty): those runs kept
timing out waiting on the shared i265 `flock` under heavy cross-lane contention.

## Change: `channel()` borrows its input planes instead of copying them

`crates/zenmetrics-nn/src/wd.rs`, `channel()`. The old code opened with
`let (mut a, mut b) = (a.to_vec(), b.to_vec());`, copying both full-resolution
input planes (review item: "at level 0 `channel()` copies both input planes
(`to_vec`)"). They are now `Cow::Borrowed` until the first `decimate` rebinds them
to owned quarter-resolution planes; every value fed to `lowpass`/`square` and every
accumulation is unchanged, so all five scores are bit-identical. This removes two
full-resolution `memcpy` passes and two full-resolution planes of live scratch per
channel (out of ~9 plane-equivalents at level 0). No new threads, no new allocation
per row; `threads_for`/`workers()` semantics are untouched.

### Variants tested and rejected

The other review items (build the two squared planes in one pass; keep one sigma
map per level scaled by `k+1`; compute the weights inline instead of building 25
maps) were implemented and measured. Individually and together they give larger
wins at ≥1024 (up to −8.5% at 1024, −12% at the 3000² cap at 16T) but each costs
~3.4% on **256 at 16T** (`+10 ms` on 290 ms, 12-rep interleaved, identical in
`ab.tsv` as `excluded_otffuse`). Since "small images must not get slower" is a hard
gate, only the copy elimination is committed. Both rejected variants were verified
bit-identical (golden test + inline-scaling and fused-squaring unit tests, all
passing) — they are excluded purely on the small-image timing, not on correctness.

## A/B (interleaved base, commit; median/min ms over the reps in `ab.tsv`)

`ab.tsv` (size, threads, bin, rep, ms, rss). base = `51d804f3`, commit = this lane.
16T rows: 4 reps except 256 (16 reps); 1T: 256 12 reps, 1024/2048/cap 2 reps.

| threads | size | base min/med | commit min/med | Δ med |
|---|---:|---:|---:|---:|
| 16 | 256 | 280 / 290 | 280 / 290 | +0.0% |
| 16 | 1024 | 3880 / 3890 | 3830 / 3855 | −0.9% |
| 16 | 2048 | 15560 / 15585 | 14500 / 14570 | −6.5% |
| 16 | cap 3000² | 34310 / 34370 | 32890 / 33040 | −3.9% |
| 1 | 256 | 1280 / 1300 | 1280 / 1300 | +0.0% |
| 1 | 1024 | 20110 / 20125 | 20060 / 20090 | −0.2% |
| 1 | 2048 | 84500 / 84695 | 82830 / 83065 | −1.9% |
| 1 | cap 3000² | 185130 / 185810 | 181740 / 182145 | −2.0% |

No case regresses; 2048 and the 3000² cap clear 3% at 16T. Small sizes are flat
(256 identical min and median at both thread counts).

### Peak memory (3000² cap pair)

`cap_memory.tsv`; `/usr/bin/time -f "%e %M"`. At 16T peak RSS drops from
12.34–12.36 GiB (base) to 12.18–12.19 GiB (commit), −1.2% / ~150 MiB, consistent
over 4 interleaved reps. At 1T both are ~11.5 GiB and unchanged (that run is
trunk-bound, so the head scratch is a smaller fraction). The 3000² pair is the
`~/tmp/wd-cap/pairs.tsv` cap used by `benchmarks/wd_port_2026-10-16/cap_memory.tsv`.

### Score identity

`wd_score` full-precision `wd_s1..wd_s5` (`{:.17e}`): byte-identical base vs
commit at 256/1024/2048/cap × 1T/16T (8/8 `cmp` clean). The `wd::tests` unit
tests, including `statistics_match_independent_upstream_golden`, pass.

## Gates

- `just snap-cargo @ test -p zenmetrics-nn` — 60 passed; release likewise
  (`~/tmp/oc-perf-scratch/logs/gate_test.log`).
- CI lint: `just snap-cargo @ clippy --workspace --no-default-features
  --features wgpu,all-metrics -- -D warnings …` — exit 0, 0 errors
  (`~/tmp/oc-perf-scratch/logs/gate_clippy.log`).
- `just nn-wd-parity ~/tmp/wd-port/models ~/tmp/nn-phase0/ladder rten` — 25 pairs,
  worst 1.87e-6 JND (`~/tmp/oc-perf-scratch/logs/gate_wd_parity_rten.log`).
- `just nn-wd-parity … ort-cuda` (`. ~/tmp/ort-env.sh`) — worst 3.44e-6 JND
  (`~/tmp/oc-perf-scratch/logs/gate_wd_parity_ort.log`).
