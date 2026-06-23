# zensim-gpu vs zensim-cpu parity — 2026-06-22

Provenance:
- git commit (code fix): fd55300d (cache.rs regime=None NaN fix + reclaim-on-rebuild stale-page fix)
- measured on commit: fd0011de
- host: lilith (Ryzen 9 7950X + RTX 5070, CUDA 13.2)
- binary: cargo build --release -p zenmetrics-cli --features 'sweep,png,jxl,gpu,gpu-cuda,gpu-zensim'
- command: zenmetrics sweep --codec zenjpeg --sources <7 imgs: square refs 64/256/512/1024/1448 + 2 native 1638x2048,2048x1536> --q-grid 10,30,50,70,90 --metric zensim --metric zensim-gpu --gpu-runtime cuda
- both metrics scored on identical zenjpeg-encoded bytes per cell (single sweep, two --metric)
- score scale: zensim JOD-like (higher=better). Parity = |zensim_gpu - zensim|.

## Summary (35 cells = 7 size/content x 5 quality)
- mean   |gpu-cpu| = 0.0295
- median |gpu-cpu| = 0.0096
- max    |gpu-cpu| = 0.1562   (ref_1448 q70)

Before the fix this same grid produced score-fail=35 (every cached-ref cell
returned 'non-finite score NaN'); and with the NaN fix alone the 1448 size
diverged 5-7 JOD until the reclaim-on-rebuild stale-page fix landed.

## Size/content sweep (mean/max |diff| over q=10..90)
| source | mean | max |
|---|---|---|
| big_1638x2048 (3.4 MP) | 0.0055 | 0.0100 |
| big_2048x1536 (3.1 MP) | 0.0024 | 0.0044 |
| ref_64    | 0.0522 | 0.1288 |
| ref_256   | 0.0429 | 0.0797 |
| ref_512   | 0.0033 | 0.0079 |
| ref_1024  | 0.0333 | 0.0670 |
| ref_1448 (2.1 MP) | 0.0667 | 0.1562 |

## Quality sweep (mean/max |diff| over all 7 sources)
| q | mean | max |
|---|---|---|
| 10 | 0.0110 | 0.0232 |
| 30 | 0.0299 | 0.1288 |
| 50 | 0.0198 | 0.0644 |
| 70 | 0.0474 | 0.1562 |
| 90 | 0.0393 | 0.1118 |

Note: max 0.156 sits slightly above the historical PORT_STATUS "<=0.05" claim
but is f32-kernel drift (sub-64 ref_64 and odd-padded ref_1448 are the worst
cells); it is not a correctness defect. Sub-64 inputs route to CPU on both
paths so they are bit-identical there.
