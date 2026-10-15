# Encoder loop player, 2026-10-15

The viewer's "Encoder loop player" panel steps through jxl-encoder's
perceptual quantization loop: play, pause, step back and forward, or scrub
through every scored iteration of every seed. It reads a `.ltrace` file that
jxl-encoder writes with its trace API (imazen/jxl-encoder PR #129,
`EncodeRequest::with_loop_trace`):

```
cargo run --release --example loop_trace_dump -- in.png out.ltrace [distance] [effort] [iterations]
```

Layers, all at the image's resolution:

| Layer | Shows |
| --- | --- |
| Reconstruction | the image this iteration scored (sRGB8) |
| Metric diffmap | the loop metric's per-pixel diffmap, one square-root scale across all frames (99.5th percentile) |
| Block distance vs target | each 8×8 block's distance (16-norm of its diffmap times the tile constant); the middle of the palette is the target, and blocks above it get finer quantization next |
| Quant field | the float quant field this iteration encoded with, one scale across frames |
| Quant field change | the loop's adjustment for the next iteration, `log2(next/current)`: red finer, blue coarser |
| Source | the input |

The info line gives the seed, iteration, score against the target, how many
blocks are above it, how many the loop made finer or coarser, and the final
encoded size.

Format (`JXLTRACE1`, little-endian): magic `JXLTRACE1\n`, a `u32` header
length, the JSON header (`width`, `height`, `distance`, `effort`, `iters`,
`bytes`, `xsize_blocks`, `ysize_blocks`, `frames: [{seed, iter, metric,
score, target, has_next}]`), the source as packed sRGB8, then per frame the
`f32` quant field, next quant field (when `has_next`), tile distances and
diffmap, and the reconstruction as packed sRGB8.

Why a file, not a live encode: zenmetrics pins jxl-encoder 248 commits
behind its `main`, and its CLI sweeps encode through that pin. Building the
trace API into the viewer's WASM would move that pin and change encoder
output for the benchmark sweeps. A live in-browser encode needs a separate
WASM module built on jxl-encoder's own pinned sibling set.

Validation: `check-pages.py` loads a synthetic two-frame trace in headless
Chromium, checks the info line (blocks above target, finer/coarser counts,
the score-only final pass) and exact pixels of the reconstruction and
quant-field-change layers, and refuses a malformed file. A real trace
(frymire 1118×1105, d1 e7, four butteraugli iterations: five frames,
48 MB) loads and steps through all frames.
