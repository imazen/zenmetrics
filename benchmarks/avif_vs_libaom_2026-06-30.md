# AVIF (zenrav1e) vs latest libaom-at-slow — RD comparison (2026-06-30)

**Question.** Our canonical picker data shows JXL winning the ssim2→bpp Pareto at HQ by −39–63% vs
best-other (`scripts/picker/hq_pareto.py`) — *more* than Cloudinary/libjxl's published ~−28% vs
libavif. Is our AVIF encoder (zenrav1e) weaker than the reference (libaom-at-slow), inflating JXL's
relative margin?

**Answer: yes, partly.** Our AVIF is a genuinely weaker AV1 encoder than the latest libaom-slow.

## Result — paired per-image bpp, our-AVIF **larger** than libaom-slow (cpu-used=2, 28 images)

| ssim2 | PHOTOS median (n=16–18) | ALL 28 median | synthetic plots (fam7) |
|------:|------------------------:|--------------:|-----------------------:|
| 82 | **+16.3%** | +22.8% | +130.7% |
| 85 | **+11.8%** | +15.9% | +248.9% |
| 88 | **+7.8%** | +21.8% | +413.9% |
| 90 | **+18.5%** | +23.8% | +468.8% |

- BD-rate (integrated) **+25% median** across all 28; **28/28 images need more bits**.
- **Photos ~8–18%** is the honest headline. The extreme +130–470% is entirely **synthetic
  screen-content** (family `7000-lilith-plots`) where AVIF is weak regardless (content mismatch,
  over-represented at ~1MP) — not pure encoder weakness.
- **Not a speed cap:** cavif/zenrav1e **speed1 == speed2 byte-identical**; the gap holds at cpu-used=0
  and *widens* as libaom is given more time. At **matched speed** (both ≈0.088 Mpx/s) our AVIF still
  needs ~10–18% more bits. Running our sweep slower would not close it — it is intrinsic RD.

## Implication for the JXL HQ finding

The measured JXL HQ win (−39–63% vs best-other) is **partly inflated** by our weak AVIF. Correcting for
a strong (libaom) AVIF, JXL's **photo** margin at HQ is **~15–25%** — consistent with Cloudinary's
−28%. **JXL still genuinely wins at HQ** (it's not an encoder mirage), but:
- Using our AVIF as "best-other" **overstates** JXL's advantage on photographic content.
- A picker trained on our-AVIF data **over-picks JXL** vs what a libaom-quality AVIF would yield.
- zenrav1e has a real **~10–18% RD improvement target** vs the libaom reference on photos.

See the correction boxes in `docs/METRIC_CODEC_BIAS_2026-06-30.md` +
`docs/BUTTERAUGLI_JXL_BIAS_2026-06-30.md`.

## Provenance

- **libaom:** git HEAD `632172a468f5e91c5b40daaa0a91f4a291c63af4` (aomedia googlesource), aomenc 3.14.1,
  cmake Release. Settings `--cpu-used={2,1,0} --end-usage=q --cq-level=<sweep> --passes=1`, default tune,
  color-exact BT.601 full-range / identity-RGB, formats {420-YCbCr, 444-YCbCr, 444-RGB}, 8-bit.
- **our-AVIF:** extracted from the canonical `zenavif_lossy` parquet — zenavif `a5697e0a8b0d` /
  zenrav1e `22a58d58db1d`, `modes_full` best-of frontier over speeds {2,4,6,8} × 3 formats × 8/10-bit;
  cross-checked vs a fresh `cavif` encode (bpp within 1%).
- **Scorer:** `fast-ssim2-cli v0.6.0` / core 0.8.2 (HEAD `585006c`) — the **same** crate (path dep) that
  produced the canonical `score_ssim2` column. No scorer mixing.
- **Sample:** 28 images from `clean-picker-corpus-2026-06-26`, families 1/2/3/5/6/7/9, mostly ~1MP
  (corpus max).
- **Confound found+fixed:** ffmpeg RGB↔YUV conversion added ~MAE 0.07 (capped ssim2 ~66); replaced with
  a color-exact converter so the only roundtrip error is AV1 compression.

## Data

Full analysis, RD graph (`rd_curves.png`), and CSVs live at
`/mnt/v/output/avif-vs-libaom-2026-06-30/` (mirrored to `/mnt/tower/output/avif-vs-libaom-2026-06-30/`,
49 files). Raw encode tables: `aom_results.tsv` (840 cpu2 encodes) + `aom_cpu1.tsv` (360) +
`aom_cpu0.tsv` (36). Harness: `color.py` / `aom_cell.sh` / `run_sweep.sh` / `analyze.py`. All encodes
ran under `~/work/zen/scripts/run-heavy` (peak RSS 0.16 GiB, box responsive throughout).
