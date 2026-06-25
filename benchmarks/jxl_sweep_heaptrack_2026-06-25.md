# jxl lossy sweep — heaptrack: where the allocations come from (2026-06-25)

Profiled: `zenmetrics-before sweep --codec zenjxl --plan lossy_dense --plan-budget 60
--sources <1.05MP photo> --q-grid 5,10,..,95 --metric zensim --jobs 1` (parent binary, prod path).
heaptrack 1.3.0. Answers user Q 'where did the allocs come from?'

## Totals
total runtime: 15.17s.
temporary memory allocations: 2485778 (163839/s)
peak heap memory consumption: 2.72G
peak RSS (including heaptrack overhead): 3.29G
total memory leaked: 3.62M

## Top PEAK MEMORY CONSUMERS (all jxl_encoder; zensim metric has ZERO frames)
  739.64M  butteraugli::image::BufferPool::take  [butteraugli::image]
  503.32M  _ alloc..vec..Vec T  as alloc..vec..spec_from_iter_nested..SpecFromIte  [jxl_encoder::vardct::bitstream::_]
  352.32M  jxl_encoder::api::srgb_u8_to_linear_f32  [jxl_encoder::api::srgb_u]
  349.83M  _ T as alloc..vec..spec_from_elem..SpecFromElem ::from_elem  [jxl_encoder::vardct::transform]
  126.99M  jxl_encoder::vardct::epf::compute_epf_sharpness  [jxl_encoder::vardct::epf::compute_epf_sharpness::h]
  101.19M  jxl_encoder::vardct::perceptual_loop::_ impl jxl_encoder..vardct..enco  [jxl_encoder::vardct::perceptual_loop::_]
  100.66M  jxl_encoder::budget::try_alloc_vec_f32_dirty_permanent  [jxl_encoder::budget::try_alloc_vec_f]
  83.89M  jxl_encoder::vardct::bitstream::_ impl jxl_encoder..vardct..encoder..V  [jxl_encoder::vardct::bitstream::_]
  75.50M  jxl_encoder::vardct::reconstruct::reconstruct_xyb  [jxl_encoder::vardct::reconstruct::reconstruct_xyb::h]
  75.50M  jxl_encoder::vardct::epf::epf_step0  [jxl_encoder::vardct::epf::epf_step]

## Peak RSS vs cells (jobs 1, /usr/bin/time -v) — peak = heaviest stratum sampled, NOT accumulation
  q-vals=1  cells=35  1997 MB     | plan-budget=1 q=50  cells=4   443 MB
  q-vals=3  cells=60  3037 MB     | plan-budget=5 q=50  cells=5   536 MB
  q-vals=5  cells=60  3426 MB
  q-vals=10 cells=54  4494 MB     (fewer cells, MORE RSS than 60-cell/3q → strata richness, not count)

## Verdict
Allocations = jxl lossy ENCODER per-encode working set (dominant: internal butteraugli
quant-refinement BufferPool 739M). NOT the zensim metric. NOT a leak (3.62M). NOT glibc
high-water accumulation (peak heap 2.72G ≈ RSS 3.06G). Mechanism: per-encode cost (stratum-
dependent ~10x spread) × cell concurrency. Job system bounds it (1 encode/process).
