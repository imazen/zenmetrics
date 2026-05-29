# Superseded CPU heaptrack data — archived 2026-05-28

These files were the live CPU heaptrack record until 2026-05-28. They are
**superseded** by the canonical refresh at
[`benchmarks/cpu_metrics_full_table_2026-05-28.tsv`](../../cpu_metrics_full_table_2026-05-28.tsv)
(+ [`docs/CPU_BENCHMARK_TABLE_2026-05-28.md`](../../../docs/CPU_BENCHMARK_TABLE_2026-05-28.md)),
produced by task #139 with zenbench wall times + heaptrack process-peak,
every mode × cold/warm.

## Why they were retired

1. **30 fabricated strip rows DELETED** from `stats_pre-refresh.tsv`. The
   original `stats.tsv` laundered 30 `strip` / `warm_ref_strip` rows for
   butter/cvvdp/dssim/ssim2/iwssim that were **not measurements** — they
   were the `cpu_profile` driver's input-buffer pass-through
   (`top1_caller = cpu_profile::main`, a flat 100.74 MiB @ 16 MP / 240.65 MiB
   @ 40 MP at every size, ~0.01–0.23 s). They appeared in `stats.tsv` with
   no GAP marker, so a reader would mistake them for real strip results.
   The companion `summary_*.tsv` files honestly flagged them `outcome=GAP`;
   `stats.tsv` did not. The 30 fakes were sub-second stubs (not protected by
   the ">60s data" preservation rule) and are removed outright.

2. **The real `full` / `warm_ref` rows are stale** (preserved here as honest
   historical data, not deleted):
   - `butter` warm_ref shows 8.71 GiB @ 40 MP — the **0.9.3** baseline.
     butteraugli **0.9.4** (#135) fixed it to 7.83 GiB.
   - `cvvdp` strip was GAP here — the **Path A** walker (#127) now delivers a
     real strip (1.58 GiB @ 16 MP / 3.32 GiB @ 40 MP). The canonical table
     documents that Path A strips only the pool stage so peak == full at
     small sizes.
   - `zensim` strip rows here ARE real (the only metric that was genuinely
     measured in strip mode in this run).

3. The `summary_*.tsv` files reference heaptrack `.zst` paths under a deleted
   sibling workspace (`zenmetrics--phase9x`) and describe the superseded run.
   Retained as provenance.

## Canonical source going forward

Use `benchmarks/cpu_metrics_full_table_2026-05-28.tsv`. Do not cite these
archived files as current.
