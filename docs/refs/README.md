# refs/

Reference material and design notes for zenmetrics GPU work.

Everything here is **text**: design docs, perf analyses, benchmark
data, and external code references. Per the >30KB binary commit rule
in CLAUDE.md, the following sit on the filesystem only (not in git):

- `IWAITSPIE_Kanetaka.pdf` — Kanetaka et al. IWAIT 2026 paper.
  Citation: doi:[10.1117/12.3100969](https://doi.org/10.1117/12.3100969).
  IWAIT 2026 Best Paper Award.
- `IWSSIM.pdf` — Wang & Li 2011 IWSSIM reference paper.
- `iwssim_iwpsnr.zip`, `python_iwssim.zip`, `matlab/`, `python/` —
  reference implementations downloaded from author sites.

When working in this repo, look for those at
`~/work/zen/zenmetrics-refs/` if a doc references them locally. They
should NOT be committed (binary blobs, redownloadable).

## Files in this directory

- `cubecl-wishlist-2026-05-17.md` — prioritized list of cubecl
  improvements with cross-crate ROI estimates, derived from this
  session's perf work + gotchas hit while porting 6 GPU crates.
- `kanetaka-iwait-2026-paper-notes.md` — full reading + analysis of
  the IWAIT 2026 SSIMULACRA2 paper, with corrected speedup figures
  (the SPIE abstract says ×82.4; paper conclusion says ×44.2) and
  per-technique decomposition (FIR alone ×12, +skip ×1.45, +PreR ×1.94).
- `vship-acceleration-analysis-2026-05-16.md` — architecture
  comparison of vship (the published GPU SSIMU2 reference) against
  our zenmetrics-gpu crates, with prioritized porting plan + measured
  results.
- `vship-bench-2026-05-16/` — minimal C harness + nvcc build
  instructions + per-size results CSV for benching vship CVVDP on
  RTX 5070. Reproducible from clean.

## Cross-references

- `docs/CUBECL_GOTCHAS.md` — catalog of cubecl 0.10 pitfalls (G1.x — G7.x)
- `docs/CUBECL_PORTING_GUIDE.md` — performance checklist + porting strategy
