# zenresize migration receipt — 2026-10-16

Commit: `nn-zenresize` working copy (see `jj log` for the commit; pushed via
`scripts/safe_push.sh`). zenresize pinned to **unmerged PR #18** head
`4f5287e65f28b24815ffe49effa2dc40284da471` (`resample-conventions`) —
`crates/zenmetrics-nn/Cargo.toml` carries the pin comment; move to the
released version at landing.

## What changed

`crates/zenmetrics-nn/src/resize.rs` keeps the planar `[c, h, w]` signatures;
five kernels now delegate to zenresize (planar↔interleaved shuffles are pure
reorders). `src/tfresize.rs` is deleted. Call sites unchanged: dreamsim
(bicubic_pil_u8), ckdn (bicubic_aa + nearest), afine (bicubic ×2: image and
c=1 positional embedding), musiq (gaussian_tf_f32).

## Bit-identity sweep (old local port vs zenresize path)

Harness: legacy impls extracted from the parent commit into a
`#[cfg(test)]`-only module, run over **83 images** — 13 synthetic patterns
plus all 70 RGB8 PNGs in `~/tmp/nn-phase0/ladder{,-nonsquare}` — at every
caller geometry: dreamsim →224²; arniqa →(h/2,w/2); hyperiqa →512×384;
maniqa →224²; ckdn →short-side-329 keep-aspect; afine →next multiple of 32
per axis (upscale) **and** `[1,50,768]→[1,n_tok,768]` pos-emb (13 shapes);
musiq →longer side {224,384} keep-aspect.

| function | zenresize convention | result |
|---|---|---|
| `resize_bicubic_pil_u8` (dreamsim) | `PillowCubicResizer` | **bit-identical** — 0 diffs on all 83 |
| `resize_bilinear_pil_u8` (arniqa, hyperiqa) | — | **kept** — no Pillow bilinear path at the pin (gap) |
| `resize_bicubic_cv2_u8` (maniqa) | — | **kept** — OpenCV CV_8U cubic not in zenresize (gap) |
| `resize_bicubic_aa` (ckdn) | `cubic_bc(0,0.5)`+AA+`Truncate` | not identical; worst abs Δ **2.75e-4** (0..255 input) |
| `resize_nearest` (ckdn) | `nearest()`+`Asymmetric` | **bit-identical** — 0 diffs on all 83 |
| `resize_bicubic` (afine) | `cubic_bc(0,0.75)`+`Clamp` | not identical; worst abs Δ **1.22e-4** image, **3.58e-7** pos-emb |
| `resize_gaussian_tf_f32` (musiq) | `gaussian_tf()`+no AA | not identical; worst abs Δ **1.22e-4** |

## Model-level parity after migration (gate: ≤1e-3 JND AND ≤2× recorded worst)

Command: `just nn-parity` = `cargo test --release -p zenmetrics-nn
--features model-tests[,ort] --test parity` in the lock snapshot;
`ZENMETRICS_NN_LADDER=~/tmp/nn-phase0/ladder`, model dirs
`~/tmp/nn-phase0/models` + `~/tmp/nr-learned-scratch/models`;
ort-cuda via `. ~/tmp/ort-env.sh` (cuDNN 9.19).

| affected model | engine | recorded worst | post-migration worst | verdict |
|---|---|---|---|---|
| ckdn (bicubic_aa + nearest) | rten | 5.65e-5 JND (2.981e-7) | 5.65e-5 JND (2.981e-7) | unchanged |
| ckdn | ort-cuda | 5.1e-5 JND (2.683e-7) | 6.22e-5 JND (3.279e-7) | 1.22× — pass |
| afine (bicubic) | rten | 1.42e-5 JND (7.228e-6) | 1.16e-5 JND (5.878e-6) | improved |
| afine | ort-cuda | 1.94e-5 JND (9.865e-6) | 1.94e-5 JND (9.865e-6) | unchanged |
| dreamsim (pil_u8) | rten | 1.17e-5 JND (2.064e-7) | 1.17e-5 JND (2.064e-7) | unchanged (bit-exact) |
| dreamsim | ort-cuda | 2.03e-5 JND (3.577e-7) | 2.03e-5 JND (3.577e-7) | unchanged (bit-exact) |
| musiq (gaussian_tf) | rten | 1.20e-4 JND (2.37e-4) | 1.07e-4 JND (2.099e-4) | improved |
| musiq | ort-cuda | 6.98e-5 JND (1.37e-4) | 9.11e-5 JND (1.793e-4) | 1.31× — pass |

All five migrated functions pass the gate; the two retained ports are
documented gaps (module doc + this file + each model's card.toml).

## Compatibility gaps at the pinned rev

1. **Pillow bilinear u8** — `PillowCubicResizer` is bicubic-only; PR #18's
   "Not included" list has no Pillow bilinear. `resize_bilinear_pil_u8`
   (arniqa h/2 view, hyperiqa 512×384) stays on the local fixed-point port
   (LICENSE-pillow conventions, `pil_1d` tent kernel).
2. **OpenCV CV_8U bicubic** — PR #18 explicitly excludes the OpenCV
   fixed-point u8 cubic/linear paths (independent implementations left ±1
   diffs). `resize_bicubic_cv2_u8` (maniqa 224²) stays on the local port.
3. Torch-AA-bicubic at out-width 1 (PR-documented gap) is unreachable here —
   ckdn's smallest output side is 329.
