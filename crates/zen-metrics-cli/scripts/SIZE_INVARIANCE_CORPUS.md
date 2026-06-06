# Size-invariance validation corpus

Pre-encoded corpus for `zen-metrics size-invariance` (the metric size-invariance
property tester). Tests that a fixed (ref,dist) pair scores consistently as it
is downsampled — across multiple metrics (zensim / ssim2 / butteraugli) and pad
strategies (raw / mirror / tile).

- **Generator:** `gen_size_invariance_corpus.py` (PIL + libjxl `cjxl`/`djxl`).
- **Contents:** 3 inputs (frymire, per-size gradient, canon_5d real photo) ×
  144 size configs (step-1 1→32, step-3 33→63, step-7 64→256; each square + 1:3)
  × 10 distortions (JPEG q90/q80/q60/q40/q25, JXL d0/d1/d2/d3/d6). 4752 PNGs.
- **R2:** `s3://zentrain/size-invariance-corpus/size_invariance_corpus_2026-06-06.zip`
  (148 MB; endpoint `https://338ad3b06716695d6e2c81c864e387d8.r2.cloudflarestorage.com`).
- **Local:** `/mnt/v/zen/size-invariance-corpus/corpus`.

## Run
```
zen-metrics size-invariance               # downsample-pair gate (default)
zen-metrics size-invariance --mode encode-per-size   # error-free small-image scoring
```
Key result (2026-06-06): raw errors on sub-64px cells for zensim/ssim2/butteraugli;
mirror & tile score every size to 1px. Mirror chosen for production (zensim
`reflect_pad_to_min`, landed zensim main 2ff8c882): bake-/metric-agnostic, beats
tile on textured invariance, solid-colour Δ≈0 at every size.
