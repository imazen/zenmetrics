# vram_leak_check 2026-05-29 — per-cycle raw data (block storage)

The in-repo `vram_leak_check_2026-05-29.tsv` is the compact **per-cell
summary** (22 rows; floor / range / env-delta / mono% / verdict). The full
**per-cycle** series (1680 rows, ~85 KB — over the 30 KB in-repo limit)
lives in block storage:

- `/mnt/v/zen/butteraugli-gpu/vram-leak-2026-05-29/vram_leak_check_2026-05-29_per_cycle.tsv`
  - sha256: `13fdac7f45d51f42e42697abf179153c17c91aa93912b6799de78449a88ccbc1`
  - columns: `check  mode  size_mp  w  h  cycle  vram_used_mib  vram_delta_mib`
  - = clean 16 MP rows (grid2) + clean 1 MP rows (strict-quiet re-run).

Forensic raw passes (also in that dir):
- `grid2_raw_2026-05-29.tsv` — the full grid2 pass (16 MP clean, 1 MP
  partly zensim-contaminated).
- `clean_1mp_rerun_2026-05-29.tsv` — the strict-quiet 1 MP re-run.
- `pass1_singlesample_contaminated_2026-05-29.tsv` — the first
  single-sample pass (kept to show the contamination signature, e.g. a
  −1200 MiB delta).

Provenance: `vram_leak_check_2026-05-29.meta`. Method + analysis:
`../docs/VRAM_LEAK_CHECK_2026-05-29.md`. Regenerate via
`examples/vram_leak_check.rs` (see the doc's Method section).

Not mirrored to Tower (transient measurement data, fully reproducible from
the committed driver; the per-cell summary + verdict are the load-bearing
artifacts and are in-repo).
