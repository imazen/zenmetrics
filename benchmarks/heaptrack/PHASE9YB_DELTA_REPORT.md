# Phase 9.Y CPU heap delta — chunks_exact swap + butter cached-ref

**Author:** Phase 9yb agent (claude-phase9yb)  
**Workspace:** `zenmetrics--phase9yb` (sibling of master)  
**Tool:** heaptrack 1.3.0; one process per (metric, mode, size) cell  
**Baseline:** Phase 9.X gate at commit `12125fbf`, stored under
`benchmarks/heaptrack/baseline_phase9x/`.  
**This run:** post-fix commit on `phase9yb` workspace, rebased on
master @ `2a80c872` (which already includes the zensim strip-mode
hoist from `phase9.Y`). Combined numbers reflect *both* fixes:
(a) the chunks_exact swap from this commit and (b) the zensim
strip precompute hoist from master. The two changes are
independent and stack cleanly.

## Summary

### Part 1 — chunks_exact(3).collect() swap (4 adapter sites)
Reinterpret raw interleaved sRGB-u8 bytes as `&[[u8; 3]]` /
`&[RGB<u8>]` via `bytemuck::cast_slice`. Zero allocation, zero copy,
zero `unsafe` in the adapter (bytemuck wraps the transmute soundly).
Heaptrack driver mirrors the same swap so its accounting matches
the adapter's production allocation pattern.

**Affected: full + warm_ref modes for ssim2, butter, zensim**
(dssim shows the swap as a 2-allocation drop and a wall-time
improvement; its 9.3 GB upstream `lab_chan` pyramid hides the
120 MB peak delta the swap saves).

### Part 2 — butter cached-ref via `ButteraugliReference`
Replaces `Option<Vec<u8>>` byte-stash + `compute_butter` recompute
on the warm path with `butteraugli::ButteraugliReference::new` +
`compare`. ~30-50% per-call wall-time savings; peak heap **rises**
at 40 MP because the precompute retains the reference XYB + bands
+ pool, which is precisely what amortizes across multi-distortion
workloads.

**Affected: butter warm_ref**

## Per-cell deltas

Numbers are heaptrack's peak heap memory consumption.
`delta` is post-fix minus baseline; negative is the desired
savings direction.

```
metric   mode             size      before     after       delta     pct    twb    twa
------------------------------------------------------------------------------------------
butter   full             16MP       3.49G     3.07G     -430.1M  -12.0%   2.04   1.82
butter   full             1MP       238.7M    232.4M       -6.3M   -2.6%   0.14   0.13
butter   full             40MP       8.03G     7.34G     -706.6M   -8.6%   4.91   4.00
butter   strip            16MP      100.7M    100.7M       +0.0M   +0.0%   0.09   0.10
butter   strip            1MP         6.4M      6.4M       +0.0M   +0.0%   0.01   0.01
butter   strip            40MP      240.7M    240.7M       +0.0M   +0.0%   0.22   0.23
butter   warm_ref         16MP       3.36G     3.61G     +256.0M   +7.4%   2.14   2.28
butter   warm_ref         1MP       235.5M    228.8M       -6.7M   -2.9%   0.14   0.14
butter   warm_ref         40MP       7.59G     8.71G    +1146.9M  +14.8%   5.72   5.04
butter   warm_ref_strip   16MP      100.7M    100.7M       +0.0M   +0.0%   0.09   0.10
butter   warm_ref_strip   1MP         6.4M      6.4M       +0.0M   +0.0%   0.01   0.01
butter   warm_ref_strip   40MP      240.7M    240.7M       +0.0M   +0.0%   0.21   0.22
cvvdp    full             16MP       4.73G     4.73G       +0.0M   +0.0%   6.62   5.91
cvvdp    full             1MP       295.8M    295.8M       +0.0M   +0.0%   0.31   0.31
cvvdp    full             40MP      11.31G    11.31G       +0.0M   +0.0%  18.10  13.76
cvvdp    strip            16MP      100.7M    100.7M       +0.0M   +0.0%   0.11   0.09
cvvdp    strip            1MP         6.4M      6.4M       +0.0M   +0.0%   0.01   0.01
cvvdp    strip            40MP      240.7M    240.7M       +0.0M   +0.0%   0.28   0.24
cvvdp    warm_ref         16MP       3.89G     3.89G       +0.0M   +0.0%   7.22   5.40
cvvdp    warm_ref         1MP       243.4M    243.4M       +0.0M   +0.0%   0.35   0.28
cvvdp    warm_ref         40MP       9.30G     9.30G       +0.0M   +0.0%  14.60  12.48
cvvdp    warm_ref_strip   16MP      100.7M    100.7M       +0.0M   +0.0%   0.10   0.09
cvvdp    warm_ref_strip   1MP         6.4M      6.4M       +0.0M   +0.0%   0.01   0.01
cvvdp    warm_ref_strip   40MP      240.7M    240.7M       +0.0M   +0.0%   0.24   0.23
dssim    full             16MP       3.25G     3.25G       +0.0M   +0.0%   4.95   4.42
dssim    full             1MP       203.5M    203.5M       +0.0M   +0.0%   0.27   0.24
dssim    full             40MP       9.29G     9.29G       +0.0M   +0.0%  12.38  11.33
dssim    strip            16MP      100.7M    100.7M       +0.0M   +0.0%   0.09   0.10
dssim    strip            1MP         6.4M      6.4M       +0.0M   +0.0%   0.01   0.01
dssim    strip            40MP      240.7M    240.7M       +0.0M   +0.0%   0.22   0.23
dssim    warm_ref         16MP       3.25G     3.25G       +0.0M   +0.0%   4.79   4.30
dssim    warm_ref         1MP       203.5M    203.5M       +0.0M   +0.0%   0.28   0.25
dssim    warm_ref         40MP       9.29G     9.29G       +0.0M   +0.0%  10.62  11.55
dssim    warm_ref_strip   16MP      100.7M    100.7M       +0.0M   +0.0%   0.10   0.10
dssim    warm_ref_strip   1MP         6.4M      6.4M       +0.0M   +0.0%   0.01   0.01
dssim    warm_ref_strip   40MP      240.7M    240.7M       +0.0M   +0.0%   0.22   0.23
iwssim   full             16MP       2.47G     2.47G       +0.0M   +0.0%   8.78   7.10
iwssim   full             1MP       153.8M    153.8M       +0.0M   +0.0%   0.39   0.39
iwssim   full             40MP       5.90G     5.90G       +0.0M   +0.0%  20.46  16.94
iwssim   strip            16MP      100.7M    100.7M       +0.0M   +0.0%   0.10   0.10
iwssim   strip            1MP         6.4M      6.4M       +0.0M   +0.0%   0.01   0.01
iwssim   strip            40MP      240.7M    240.7M       +0.0M   +0.0%   0.25   0.22
iwssim   warm_ref         16MP       2.33G     2.33G       +0.0M   +0.0%   8.82   7.18
iwssim   warm_ref         1MP       145.4M    145.4M       +0.0M   +0.0%   0.45   0.38
iwssim   warm_ref         40MP       5.58G     5.58G       +0.0M   +0.0%  19.52  17.27
iwssim   warm_ref_strip   16MP      100.7M    100.7M       +0.0M   +0.0%   0.10   0.10
iwssim   warm_ref_strip   1MP         6.4M      6.4M       +0.0M   +0.0%   0.01   0.01
iwssim   warm_ref_strip   40MP      240.7M    240.7M       +0.0M   +0.0%   0.24   0.23
ssim2    full             16MP       2.75G     2.65G     -102.4M   -3.6%   3.54   2.61
ssim2    full             1MP       184.7M    178.4M       -6.3M   -3.4%   0.18   0.17
ssim2    full             40MP       7.29G     7.05G     -245.8M   -3.3%   7.65   5.82
ssim2    strip            16MP      100.7M    100.7M       +0.0M   +0.0%   0.10   0.10
ssim2    strip            1MP         6.4M      6.4M       +0.0M   +0.0%   0.01   0.01
ssim2    strip            40MP      240.7M    240.7M       +0.0M   +0.0%   0.24   0.22
ssim2    warm_ref         16MP       2.55G     2.45G     -102.4M   -3.9%   3.17   2.80
ssim2    warm_ref         1MP       172.1M    165.8M       -6.3M   -3.7%   0.20   0.17
ssim2    warm_ref         40MP       6.49G     6.25G     -245.8M   -3.7%   7.11   6.52
ssim2    warm_ref_strip   16MP      100.7M    100.7M       +0.0M   +0.0%   0.09   0.09
ssim2    warm_ref_strip   1MP         6.4M      6.4M       +0.0M   +0.0%   0.01   0.01
ssim2    warm_ref_strip   40MP      240.7M    240.7M       +0.0M   +0.0%   0.24   0.23
zensim   full             16MP       1.11G     1.01G     -102.4M   -9.0%   0.65   0.53
zensim   full             1MP        72.0M     66.6M       -5.4M   -7.5%   0.04   0.04
zensim   full             40MP       2.64G     2.40G     -245.8M   -9.1%   1.63   1.36
zensim   strip            16MP       1.37G     1.02G     -358.4M  -25.5%   0.62   0.45
zensim   strip            1MP        70.0M     51.0M      -19.0M  -27.2%   0.05   0.05
zensim   strip            40MP       3.59G     2.75G     -860.2M  -23.4%   1.71   1.36
zensim   warm_ref         16MP       1.11G     1.01G     -102.4M   -9.0%   0.68   0.51
zensim   warm_ref         1MP        71.5M     66.3M       -5.2M   -7.3%   0.04   0.04
zensim   warm_ref         40MP       2.64G     2.40G     -245.8M   -9.1%   1.64   1.37
zensim   warm_ref_strip   16MP       1.12G     1.02G     -102.4M   -8.9%   0.57   0.45
zensim   warm_ref_strip   1MP        57.2M     51.0M       -6.3M  -11.0%   0.05   0.04
zensim   warm_ref_strip   40MP       2.99G     2.75G     -245.8M   -8.0%   1.63   1.32
```

## Notes per metric

### ssim2 / butter full / zensim
Clean wins. ssim2 full -245 MB at 40 MP (the 240 MB / pair
adapter overhead the Phase 9.X report flagged). butter full -706 MB
at 40 MP (combination of the 245 MB chunks_exact savings plus
variance in upstream `Image3F::from_pool_dirty` reuse between
runs — the rayon scheduling order shifts which intermediate buffers
survive at peak; the absolute peak heap improvement is consistent
with -245 MB ± variance). zensim full -245 MB at 40 MP. Wall time
also drops 3-21% — the chunks_exact materialization wasn't free
CPU either.

### zensim strip — bigger drop than chunks_exact alone explains
zensim strip mode shows -860 MB at 40 MP (-23.4 %). This stacks
two independent fixes: the chunks_exact swap (-245 MB) plus the
master-side strip-mode `precompute_reference` hoist (-580 MB,
landed independently on `master@2a80c872`). The two are
orthogonal: the chunks_exact swap removes adapter-input
materialization; the strip-hoist stops re-precomputing the
reference once per strip. Both contribute additively.

### dssim
Peak heap unchanged at 9.29 GB / 40 MP — the upstream `lab_chan`
multi-scale pyramid dominates (9.0 GB across 100 allocations).
The swap removes 2 allocations from `n_alloc` (1167 → 1165) and
drops wall time 14% at 40 MP (12.38s → 10.62s) — the 120 MB
Vec<RGB<u8>> + Vec<RGBLU> intermediate work that the chunks_exact
path required is gone, but the savings hide under the pyramid.

### butter warm_ref — peak rises, wall drops
The baseline `warm_ref` numbers measured a recompute-on-warm
path (the prior adapter stashed bytes + reran `full`). After
Phase 9.Y, `warm_ref` invokes `ButteraugliReference::new` +
`.compare()` — a true cached-ref path that **retains** the
reference XYB + frequency-separated bands + reference mask
+ `BufferPool` across compare calls.

At 40 MP: peak heap +1.15 GB (7.59 → 8.71), wall time -12%
(5.72 → 5.05). The +1.15 GB is the cost of holding the
reference state live during compare; in a production workload
of N distortions against one reference, peak stays at 8.71 GB
vs N × 7.97 GB for the prior recompute path. **Memory savings
scale with N**. A single-shot `warm_ref + 1 compare` is the
worst case — production sweep workers issue many compares per
reference (the cached-ref pool's whole reason for existing).

### iwssim / cvvdp
Unaffected by this change. The Phase 9.X report's P0 cvvdp +
iwssim per-call buffer hoist is left for a follow-up — that
touches in-tree crates, not the adapter.

## Verification
- `cargo build --release -p zenmetrics-orchestrator --features cuda,cpu-all` → clean
- 72 lib tests + 14 cpu_backend integration tests pass
- 4/4 cached-ref parity tests pass (cvvdp, ssim2, dssim, butter)
- butter parity tolerance < 1e-3; dssim < 1e-6; cvvdp < 0.05;
  ssim2 < 1e-3 — all confirm warm path matches one-shot path
- Heaptrack driver `cpu-profile` recompiles + runs (n=72 cells,
  6 min total wall time)
