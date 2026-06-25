# JXL lossy per-effort encode time + memory — estimate validation (2026-06-25)

Validates the per-cell **time and memory estimates** the JXL lossy knob-space
ablation program ([`docs/JXL_LOSSY_KNOBSPACE_ABLATION_PROGRAM.md`](../docs/JXL_LOSSY_KNOBSPACE_ABLATION_PROGRAM.md))
and the fleet sizing rely on, across the full `e1..=e9` ladder `lossy_dense`
sweeps. Measured against `jxl_encoder::heuristics::estimate_encode_threaded`
(the model `JxlEncoderConfig::estimate_encode_resources` delegates to).

## Method (one encode per process — clean VmHWM)

- Tool: zenjxl `examples/mem_probe_encode` (commit `77fd5888`), which now logs
  measured `encode_ms` alongside `VmHWM` peak RSS, with an `est` mode printing
  the model's predicted typ/max peak + `time_ms` for the same cell.
- One encode per process → `VmHWM` is the cell's peak RSS (the sanctioned
  `time -v`-equivalent high-water; not allocator-traced, not extrapolated).
- Grid: photo + screenshot × {256², 512², 1024²} × `e1..=e9` × q{30,90},
  `threads=1`, `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=131072`. 108 cells.
- Data: [`jxl_lossy_memval_2026-06-25.tsv`](jxl_lossy_memval_2026-06-25.tsv)
  (`marg_kb` = `VmHWM − pre_RSS` = encode working set; `in_band` = measured vs
  the model's [min,max]).

```
# zenjxl @ 77fd5888, --features __expert
mem_probe_encode <rgb8.bin> <w> <h> lossy <e 1..9> <q> 1        # measured
mem_probe_encode <rgb8.bin> <w> <h> lossy <e 1..9> <q> 1 est    # model
```

## Findings

1. **The estimate is conservative everywhere — 0 / 108 cells exceed the model's
   max.** Fleet concurrency sized to the estimate (`box_RAM ÷ per-cell-GB`)
   cannot OOM on the lossy path.
2. **Memory is effort-banded**, not flat: measured marginal ≈ **50 B/px at
   e≤4 → 55 at e5 → ~91 B/px at e≥6** (photo 1024²). The model carries a flat
   133 B/px (143 at e≥8) — conservative by ~1.5–2.7×.
3. **Worst measured peak = 122 MB at 1 MP** (photo e8). The doc's 0.20 GB/cell
   lossy premise (measured at 3.15 MP) holds with margin.
4. **Tiny sizes are the most over-predicted** (256²: ~4.5 MB measured vs ~56 MB
   est typ → `in_band=LO`): the model's fixed-overhead intercept dominates when
   per-pixel work is small — the §1 sweep-discipline intercept effect, made
   explicit here.
5. **Per-effort time ramp** (photo 1024² q90): e1–e4 ≈ 63–81 ms, e5 ≈ 207,
   e6 ≈ 343, e7 ≈ 408, e8 ≈ 578, **e9 ≈ 1056 ms**. The model's `time_ms`
   tracks and is also conservative (e9 est 1505 ms). This is the cost half of
   the per-effort (quality, time, memory) trade the picker optimizes — and the
   evidence behind "a low effort + the right knob can mimic a higher effort".

## So what

- `encode_ms` is already persisted per cell in the sweep omni TSV; peak RSS is
  cleanly per-cell only in one-encode-per-process mode (the job system). This
  probe is the per-effort memory validator; wiring per-cell peak RSS into every
  omni row (monolithic sweep) is a possible follow-up, but the estimate's proven
  conservatism means it is not blocking the first fleet run.
