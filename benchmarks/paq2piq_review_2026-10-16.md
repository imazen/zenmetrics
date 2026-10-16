# PaQ-2-PiQ review — 2026-10-16

Base: `nr-learned` at `3ad697a7`. Input/consent/preprocessing fixes are
`59816545`; CLI routing and sweep matching are `62593d38`; reproducible
export is `ad0e37f2`; small goldens and gate driver are `fff5e6d1`.
No public Rust API, checkpoint, model graph or parity tolerance changed.

The native-resolution input bound is now **4096 pixels per axis**, checked
before allocation. Oversized input returns `Error::TooLarge`; tests cover
16,384² and oversized individual axes without allocating an image buffer.
Consent is checked before file access. RoIPool uses `f32::round`; preprocessing
uses the existing `rgb8_to_planar_01`.

The sweep range match and default NR route now include PaQ-2-PiQ. The sweep
build also exposed missing CKDN, DreamSim and A-FINE arms in the base. CKDN
uses its sigmoid range; DreamSim uses its cosine-distance range; the unbounded
signed A-FINE score has no characterised mean/identity bounds.

## Measurements and checks

- Debug/release NN tests, pinned workspace formatting and CI Lint clippy pass.
- Both CLI checks pass: `--features sweep,png,gpu,gpu-cuda` and
  `--features sweep,orchestrator,nn,nr`.
- Routing regression test asserts PaQ-2-PiQ bypasses the FR orchestrator.
- Original-model RTen parity passes: PaQ-2-PiQ 27 pairs, worst 3.087e-5 score
  units = 2.51e-5 JND; MUSIQ 24 pairs, worst 2.366e-4 = 1.20e-4 JND;
  HyperIQA 27 pairs, worst 1.526e-5 = 3.42e-6 JND. This includes each
  model's non-square entries resolved through the sibling directory.
- Fresh export writes both `paq2piq.meta` and `paq2piq_export.meta` and copies
  supplied goldens. Self-check error 3.815e-6; ONNX CPU parity over all 29
  pairs has worst 3.087e-5. Its graph hashes exactly match the original export.
- Rust loads that fresh directory and passes all 29 pairs, including 32² and
  33×47; worst remains 3.087e-5 = 2.51e-5 JND. Warm/cold controls pass.
- Real CLI `score --metric paq2piq` with the default scheduler enabled,
  `ZENMETRICS_USE_LEGACY_SCHEDULER=0`, and no legacy flag returns
  `paq2piq=60.86320495605469`, `paq2piq_ref=60.96868896484375` on 32².
  The smoke recipe asserts both columns against the new upstream goldens at
  the unchanged 1e-3 JND gate and asserts the scheduler-enabled diagnostic.

The small pairs are top-left crops of the existing 1415 64² pair, generated
with `models/paq2piq/small_inputs.py`; no resize or interpolation. Goldens come
from the authors' model at commit
`48c91e844e9f7a768f6ddcfc744dddfdd1160fea`, using the pinned checkpoint SHA-256
`5ece3dcd18e7bfc619ef6eee0c6e2c3b94350a623d62409f5cac95a2979b5b32`.
The oracle's explicit ROI construction is equivalent to upstream
`input_block_rois(include_image=True)`; it is not a verbatim copy.

## Reproduction

Run heavy commands serialized under `flock ~/tmp/i265-heavy.lock` and
`run-heavy --mem 16G --jobs 16`, with disk-backed `TMPDIR` and
`LOCK_SNAP_DIR=~/tmp/zenmetrics-lock-snap-aicvariants`.

```
just nn-review-paq2piq-gates
just nn-paq2piq-small-inputs <python> <square-ladder> <nonsquare-ladder>
just nn-goldens-paq2piq <python> <upstream> <checkpoint> <out-tsv> <input-dir>
just nn-export-paq2piq <python> <upstream> <checkpoint> <new-out-dir> <goldens-tsv> <square-ladder> <nonsquare-ladder>
just nn-parity <new-out-dir> <square-ladder> rten model-tests paq2piq
just snap-cargo @ build -p zenmetrics-cli --features sweep,orchestrator,nn,nr --bin zenmetrics
just nn-paq2piq-cli-smoke <binary> <new-out-dir> <nonsquare-ladder>
```

The Python scripts also accept upstream/checkpoint paths through
`PAQ2PIQ_UPSTREAM` and `PAQ2PIQ_CHECKPOINT`. The TSV oracle emits a header;
the exporter checks that header and refuses empty goldens.

Observed on i265: original models `~/tmp/nr-learned-scratch/models`, square
ladder `~/tmp/nn-phase0/ladder`, non-square and new small images
`~/tmp/nn-phase0/ladder-nonsquare`. These are external eval data; no images or
weights are committed. Full logs and the new export are retained under
`~/tmp/aic-variants-scratch/round11-*`. The first gate attempt failed at the
additional missing sweep match arms; the corrected full gate run passes.

Wrapper receipts:

```
baseline fmt/clippy: rc=0 24s | peak-RSS 0.83GiB | min-avail 21388MiB | peak-load 4.19
small oracle: rc=0 2s | peak-RSS 0.85GiB | min-avail 24171MiB | peak-load 0.98
full final gates: rc=0 157s | peak-RSS 3.24GiB | min-avail 18698MiB | peak-load 11.63
fresh export: rc=0 21s | peak-RSS 3.02GiB | min-avail 21264MiB | peak-load 15.87
fresh 29-pair Rust parity: rc=0 16s | peak-RSS 2.20GiB | min-avail 21594MiB | peak-load 13.43
CLI build: rc=0 8s | peak-RSS 3.69GiB | min-avail 23323MiB | peak-load 10.99
CLI smoke: rc=0 1s | peak-RSS 0.10GiB | min-avail 24157MiB | peak-load 5.90
```
