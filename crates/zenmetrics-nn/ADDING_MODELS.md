# Adding a model to zenmetrics-nn

Which models are allowed, and from which source, is decided in
[`docs/PYIQA_UPSTREAM_LICENSES_2026-10-15.md`](../../docs/PYIQA_UPSTREAM_LICENSES_2026-10-15.md)
and [`docs/PYIQA_PORT_PLAN.md`](../../docs/PYIQA_PORT_PLAN.md). Port from the original
upstream package, never from pyiqa's files. For NC or unlicensed upstreams, write the
code from the paper.

`lpips` and `dists` are the templates: LPIPS has a per-tap head over a network with
several variants; DISTS has an extra input tap and per-channel statistics.

## 1. `models/<name>/`

- `card.toml`: kind, direction, columns, paper, upstream repo and package version, code
  source (`upstream-port` or `paper`) and licence, `weights.licence_class`
  (`permissive` / `unknown` / `nc`) with the weight URLs, input normalisation, resize
  rule, min size, parity oracle, tolerance and the measured parity.
- `export.py`, run with a venv that has the upstream package. It writes into one output
  directory, all files prefixed with the model's column name `<col>`:
  - `<col>_trunk.onnx`: one input `x[1,3,H,W]`, one output per tap, dynamic H/W,
    opset 17, `dynamo=False`. Everything before the taps (normalisation, scaling)
    goes inside the graph.
  - `<col>_full.onnx`: inputs `r`, `d`, one scalar output named after the column: the
    upstream forward as is, for whole-graph engines (ort-cuda).
  - head weights as raw little-endian f32 (`<col>_lin.f32`, `dists_alpha.f32`, …).
    When upstream derives weights at run time (DISTS divides α, β by their sum),
    export the derived f32 values so Rust doesn't redo the arithmetic differently.
  - `<col>.meta`: `channels c0,c1,...` (tap channel counts) and `min_size n`
    (probe it: smallest square input whose every tap is non-empty).
  - `<col>_goldens.tsv`: `image, size, <score>, ...` from the upstream package on CPU
    in fp32, over the ladder (`benchmarks/nn_lpips_phase0_2026-10-15/scripts/ladder.py`).
  - `<col>_export.meta`: torch and package versions, host, sha256 of every file.

## 2. Rust

- `src/<name>.rs`: a public struct implementing `za::Metric`, constructed with
  `from_dir(dir, [variant,] Engine)`. Use `common::{Backend, read_meta, read_f32}`:
  `Backend::Trunk` (rten: taps come back, head here) and `Backend::Pair` (ort-cuda: the
  whole graph, score only).
- Preprocess with `preprocess.rs` (add a function when the model's input scaling is new),
  matching upstream's float operation order exactly.
- The head reproduces upstream's formula. Accumulate spatial sums in f64; where
  upstream's own f32 rounding decides a visible result (DISTS's cross-tap total), do
  that step in f32 like upstream, and say why in a comment.
- `warm_reference` caches whatever the head needs from the reference (normalised taps,
  per-channel statistics); `reuses_reference_work()` is true on trunk engines.
- Register the model in `tests/parity.rs` (`models()`), `examples/nn_bench.rs`, the
  CLI (`crates/zenmetrics-cli/src/metrics/{mod,nn}.rs`: enum variant, name, column,
  dispatch), and the README table.

## 3. Gates

```sh
just nn-export-<name> <python> <ladder-dir> <model-dir>
just nn-parity <model-dir> <ladder-dir> rten model-tests <col>
ORT_DYLIB_PATH=... just nn-parity <model-dir> <ladder-dir> ort-cuda model-tests,ort <col>
```

- JND scale: measure one JND for each new score column on KonJND-1k with the upstream
  package (pooled median score at the PJND threshold; method and scripts in
  `benchmarks/nn_konjnd_scale_2026-10-15.md`) and add it to `models/jnd_scale.tsv`.
  The parity test fails on a column without an entry.
- Parity: every golden pair within `TOL_JND` (1e-3 of the column's JND) on both engines;
  record the worst |Δ| (absolute and in JND) in the card and README.
- Warm reference: bit-identical to the cold score; identical pair scores 0 where
  upstream does.
- `cargo clippy -p zenmetrics-nn --all-targets --features model-tests,ort` clean, and
  `rustfmt`. Run cargo only through `just snap-cargo @ ...` (never plain cargo in a dev
  tree: it rewrites `Cargo.lock`).
- On i265 the GPU flips a bit every few minutes: rerun any GPU mismatch before
  believing it; a real one reproduces.
