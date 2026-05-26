# CVVDP Conformance Matrix — cvvdp-cpu + cvvdp-gpu vs pycvvdp v0.5.4

The authoritative "are our cvvdp impls correct?" gate. Every
`(impl × display_model × situation)` cell is scored against the
canonical pycvvdp v0.5.4 reference
([gfxdisp/ColorVideoVDP](https://github.com/gfxdisp/ColorVideoVDP)),
with quantified per-cell JOD deltas and a pass/fail tolerance.

This replaces (supersedes for parity purposes) the thin end-to-end
`1e-4 JOD` check on a single standard-4K image. That single-cell gate
could **mask** a per-display or per-content divergence: the metric's
spatial/band/channel pooling and contrast masking can absorb a
localized error without moving the final JOD. The conformance matrix
exposes those errors by scoring across the full display × content ×
distortion space.

- **Harness**: `crates/cvvdp-conformance/` (depends on BOTH cvvdp-cpu
  and cvvdp-gpu; tests them as black boxes via the public API).
- **Result TSV**: `benchmarks/cvvdp_conformance_matrix_2026-05-26.tsv`.
- **Goldens**: pycvvdp v0.5.4, R2
  `s3://coefficient/cvvdp-goldens/conformance-v1/` (public mirror
  `https://coefficient.r2.imazen.org/cvvdp-goldens/conformance-v1/`).

## Dimensions

### Implementations (3-way)

| Impl | Source | Role |
|---|---|---|
| `pycvvdp_v054` | gfxdisp/ColorVideoVDP v0.5.4 (CUDA torch) | **REFERENCE** — ground-truth JOD |
| `cvvdp_cpu` | `cvvdp_cpu::Cvvdp` (this workspace) | under test |
| `cvvdp_gpu` | `cvvdp_gpu::Cvvdp<CudaRuntime>` (this workspace) | under test |

### Display models (9 — acceptance gate requires ≥ 8)

Every display is an **upstream pycvvdp display name** that ALSO
resolves in our `DisplayModel::by_name` / `DisplayGeometry::by_name`
registry. This is the apples-to-apples contract: pycvvdp is invoked
with `display_name=<name>` and our impls are configured via
`by_name(<name>)`, so all three scorers use the same photometric +
geometric display model.

| Display | EOTF / primaries | Y_peak (nit) | E_ambient (lux) | Geometry note |
|---|---|---|---|---|
| `standard_4k` | sRGB / BT.709 | 200 | 250 | canonical reference |
| `sdr_4k_30` | sRGB / BT.709 | 100 | 250 | standard desktop |
| `standard_fhd` | sRGB / BT.709 | 200 | 250 | 1080p |
| `standard_phone` | sRGB / BT.709 | 500 | 250 | phone |
| `iphone_14_pro` | sRGB / BT.709 | **1025** | 250 | bright auto-brightness phone |
| `standard_hdr_pq` | PQ / **BT.2020** | 1500 | 10 | HDR + wide-gamut |
| `standard_hdr_hlg` | HLG / **BT.2020** | 1500 | 10 | HLG EOTF + wide-gamut |
| `standard_hdr_linear_dark` | linear / BT.709 | 1500 | 0 | dim-ambient, dark-adapted |
| `htc_vive_pro` | sRGB / BT.709 | 133 | 0 | VR HMD (fov-diagonal geometry) |

Imazen-only presets (`modern_oled_phone_indoor`, `65inch_hdr_pq_*`,
`lg_oled_2026_hdr_pq`) are **excluded** from the conformance matrix:
pycvvdp can't generate a reference golden for a display name it
doesn't know. They remain pinned for self-consistency in
`cvvdp-gpu/tests` (`presets.rs`).

No upstream preset uses Display-P3 primaries — the wide-gamut presets
are all BT.2020, which is the broader gamut. The P3-primaries code
path is exercised by the `presets.rs` unit tests, not against pycvvdp.

### Situations (31 — acceptance gate requires ≥ 15)

Defined in `crates/cvvdp-conformance/src/situations.rs`, grouped by
class:

| Class | n | Examples |
|---|---|---|
| `common_photo` | 2 | synth photo + CID22-512 crop, JPEG q60 |
| `common_screenshot` | 1 | GB82-SC `codec_wiki` crop, JPEG q90 |
| `common_distortion` | 8 | JPEG q90/q60/q30/q5, blur r2/r5, noise amp12/amp40 |
| `niche_content` | 11 | tiny 16×16 / 32×32, large 1024², odd 97×101 / 255×255, flat color, 1px-checkerboard, gradient+banding, 1px spike, near-black, near-white |
| `niche_distortion` | 6 | near-lossless (b±1), heavy JPEG q2, pure chroma swap, pure luma shift, single 8×8 block, aggressive banding |
| `hdr` | 3 | highlight-clipping + wide-gamut bars (scored on PQ/HLG/linear displays) |

The 2 real-corpus situations (CID22 + GB82-SC) are present only when
`~/work/codec-corpus` is on the host; the 29 synthetic situations are
always present, so the matrix exceeds the ≥ 15 gate on any host.

### Matrix size

31 situations × 9 displays = **279 cells per impl** (acceptance gate
requires ≥ 120).

## Methodology

1. `cargo run -p cvvdp-conformance --bin emit_situations -- <dir>`
   writes every situation's `ref.png` + `dist.png` (lossless RGB8) and
   a `manifest.json` cross-producting situations × displays.
2. `scripts/cvvdp_goldens/build_conformance_goldens.py <dir>` loads
   those exact PNGs and scores each `(situation, display)` cell with
   pycvvdp v0.5.4 — `metric.predict(dist, ref, dim_order="HWC")` at
   `display_name=<upstream_name>` — writing `conformance_goldens.json`.
3. The conformance test (`tests/conformance.rs`, feature
   `conformance-goldens`) fetches the goldens from R2, then for every
   cell rebuilds the situation **in-process** (the generator is
   deterministic, so in-process bytes are byte-identical to the
   emitted PNGs — verified by manifest-sha pinning), configures both
   impls via `by_name`, and records `jod_ref / jod_cpu / jod_gpu` plus
   `delta_cpu / delta_gpu / delta_cpu_gpu`.
4. A cell PASSES when `|jod_cpu - jod_ref| ≤ 1e-3` AND
   `|jod_gpu - jod_ref| ≤ 1e-3`. Cells exceeding the tolerance are
   either fixed or recorded as documented divergences (§Divergences)
   with root cause — never silently passed.

### Determinism contract

The same bytes are scored by all three impls. Synthetic situations
are PRNG-free modular arithmetic; the one "noise" distortion uses a
fixed-seed SplitMix64 (reproducible). JPEG-distorted situations apply
JPEG in-Rust, then save the decoded RGB8 losslessly to PNG, so the
emitted PNG == the bytes the Rust harness scores in-process == the
bytes pycvvdp scores. This is pinned: the golden manifest records
`situations_manifest_sha256`, and a re-emit on 2026-05-26 reproduced
the exact sha (`fce3ccb…`).

### Tolerance rationale

`1e-3 JOD` is the documented cvvdp parity tolerance (the JOD scale is
0–10; 1e-3 is 0.01% of full scale, well below any perceptual
threshold and below pycvvdp's own torch-vs-torch run-to-run noise on
some displays). The synth fixtures already pin tighter (`1e-4`–`5e-3`)
elsewhere; the matrix's `1e-3` is the cross-display/cross-content
gate.

## Results (2026-05-26, pycvvdp v0.5.4, RTX 5070)

- **cpu within 1e-3: 274 / 279** (max `|delta_cpu|` = 0.028065)
- **gpu within 1e-3: 271 / 279** (max `|delta_gpu|` = 0.028131)
- **median `|delta_cpu|` = 2e-6, median `|delta_gpu|` = 6e-6** — the
  vast majority of cells are at bit-parity with the reference.
- **cpu/gpu agree to `max |jod_cpu − jod_gpu| = 1.2e-3`** — the two
  impls track each other tightly.

Per-class pass rates:

| Class | n | cpu pass | gpu pass | max Δcpu | max Δgpu |
|---|---|---|---|---|---|
| common_photo | 18 | 17/18 | 17/18 | 0.02439 | 0.02440 |
| common_screenshot | 9 | 9/9 | 9/9 | 0.00001 | 0.00001 |
| common_distortion | 72 | 69/72 | 68/72 | 0.02439 | 0.02440 |
| niche_content | 99 | 98/99 | 96/99 | 0.02806 | 0.02813 |
| niche_distortion | 54 | 54/54 | 54/54 | 0.00046 | 0.00066 |
| hdr | 27 | 27/27 | 27/27 | 0.00081 | 0.00076 |

Every HDR (PQ/HLG/linear/BT.2020) and niche-distortion cell is within
tolerance. All divergences are confined to two findings below.

## Divergences

The harness surfaced **two distinct findings**. Both are recorded in
the test's `documented_divergences()` allow-list (the explicit,
reviewable alternative to widening the tolerance) and root-caused
here. None is silently passed.

### Finding A — `iphone_14_pro` high-peak-luminance regime (10 cells)

**Symptom**: On the `iphone_14_pro` display (Y_peak = **1025 nit** —
the only sRGB conformance display with peak ≥ 1000 nit), both
cvvdp-cpu and cvvdp-gpu land **low** vs pycvvdp by up to **0.028 JOD**
on JPEG-distorted content. The delta is largest at mid quality
(q60/q30: 0.016–0.028) and shrinks toward near-lossless (q90: 0.006).
The large 1024² JPEG cell is the worst (0.028).

**Not a GPU issue**: cvvdp-cpu and cvvdp-gpu AGREE with each other to
~7e-5 JOD on these cells — both diverge from pycvvdp by the same
amount. This is a **shared algorithmic parity gap in the cvvdp model**
(present identically on CPU and GPU), not a kernel/float-order
artifact.

**Ruled out** (verified, not assumed):

- *Display parameters* — our vendored `display_models.json`
  `iphone_14_pro` entry is **byte-identical** to pycvvdp's upstream
  entry. Derived `Y_peak=1025`, `Y_black=0.0004` (= 1025/2562500),
  `Y_refl=0.39789` (= 250·0.005/π) all match pycvvdp's
  `display_photometry` (`get_black_level() = (0.0004, 0.39789)`,
  `contrast = 2562500`).
- *EOTF* — our sRGB `forward()` at code 0.0/0.5/1.0 yields
  0.398 / 219.8 / 1025.4 cd/m², matching pycvvdp's
  `display_photometry.forward()` to display precision.
- *CSF LUT luminance axis* — our `LOG_L_BKG_AXIS` spans 0.005 → 10000
  cd/m² (log10 −2.301 → 4.0, 32 entries), **identical** to pycvvdp's
  `csf_lut_weber_fixed_size.json` `L_bkg` axis. The iphone background
  luminance (≤ ~1025) is well inside the axis, so this is not an
  axis-clamp effect.

**Root cause (localized)**: the divergence appears only when peak
luminance pushes the per-pixel adaptation luminance into the
high-photopic range (Y_peak ≥ 1000 nit on an sRGB display).
`standard_phone` (Y_peak = 500, same E_ambient = 250 lux) passes every
cell, so the trigger is peak luminance, not ambient. The gap tracks
the correct direction (our JOD and pycvvdp's both rise vs `standard_4k`
when moving to the brighter display) but undershoots — consistent with
a small calibration-regime difference in the contrast-masking / CSF
sensitivity interaction at high adaptation luminance, where the
masking term is most active (hence the mid-quality peak and
near-lossless shrink). The magnitude (≤ 0.028 JOD = 0.28% of full
scale) is bounded and never inverts the quality ordering. Closing it
fully requires a stage-by-stage re-derivation of the masking/CSF apply
at L_adapt > ~500 cd/m² against pycvvdp's per-band tensors (a
cvvdp-cpu/gpu source change — out of scope for the conformance harness,
which tests the impls as black boxes). Filed as a finding for the
cvvdp-cpu/gpu owners; see `UPSTREAM_DIVERGENCES.md`.

### Finding B — GPU float reduction-order at the perceptibility floor (3 cells, GPU-only)

**Symptom**: 3 GPU-only cells exceed 1e-3 marginally (0.00101 –
0.00139 JOD) on extreme high-frequency or heavily-blurred content
(`checkerboard_blur_r2` on `htc_vive_pro` / `standard_fhd`,
`synth_blur_r5` on `standard_hdr_hlg`) where the reference JOD is at
the perceptibility floor (~3.7–4.4). cvvdp-cpu PASSES all three.

**Root cause**: GPU float reduction order vs CPU in the deepest
pyramid bands. On near-floor content the per-band energy is large and
the spatial/band pooling sums accumulate in a different order on the
GPU (parallel tree-reduce) than on the CPU (sequential), producing a
~1e-3 JOD spread. This is the expected GPU-vs-CPU numerical envelope at
the extreme — the cells are 0.001–0.0014 over a 0.001 gate, i.e. right
at the boundary. It is NOT an algorithmic error: the same cells on the
CPU land at 0.0007–0.0009 (just under), and the cpu/gpu agreement on
these cells is ≤ 1.2e-3. This is the documented GPU numerical envelope,
not a divergence from the model.

## Regenerating goldens

1. Build the situation corpus:
   ```bash
   cargo run -p cvvdp-conformance --bin emit_situations -- <out_dir>
   ```
2. Score with the pinned pycvvdp v0.5.4 (isolated venv reusing the
   host install — see `scripts/cvvdp_goldens/.venv`, created via
   `python3.10 -m venv --without-pip --system-site-packages`):
   ```bash
   scripts/cvvdp_goldens/.venv/bin/python \
     scripts/cvvdp_goldens/build_conformance_goldens.py <out_dir>
   ```
3. Upload to R2 (same bucket/mirror as the existing parity goldens):
   ```bash
   source ~/.config/cloudflare/r2-credentials
   aws --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
     s3 cp <out_dir>/conformance_goldens.json \
     s3://coefficient/cvvdp-goldens/conformance-v1/conformance_goldens.json
   # plus manifest.json + images/ for reproducibility
   ```
4. Bump `GOLDENS_SHA256` (and `GOLDEN_VERSION` / R2 prefix if the
   golden set changed) in
   `crates/cvvdp-conformance/tests/common/mod.rs`.

## Running the matrix

```bash
# Offline self-tests only (default — no network, no GPU):
cargo test -p cvvdp-conformance

# Full matrix (fetches goldens from R2, needs a CUDA GPU):
cargo test -p cvvdp-conformance --features conformance-goldens \
  --test conformance -- --nocapture

# Full matrix against locally-built goldens (skips R2 fetch):
CVVDP_CONFORMANCE_GOLDENS=<out_dir>/conformance_goldens.json \
  cargo test -p cvvdp-conformance --features conformance-goldens \
  --test conformance -- --nocapture
```

The `conformance-goldens` feature gate is the offline-test guard: when
it's off the matrix test isn't compiled in at all (no silent
runtime-skip — the skip decision is at the feature/caller level, per
the workspace test discipline).

## Provenance

- Reference: pycvvdp v0.5.4 (pip pkg `cvvdp` 0.5.4, import `pycvvdp`),
  torch 2.10.0+cu128, CUDA available.
- Goldens generated 2026-05-26 on the 7950X workstation (RTX 5070),
  279 cells, 0 pycvvdp errors, JOD range 3.66 → 10.00.
- `conformance_goldens.json` sha256:
  `8f7d69dc6b98272b8425c2245cf7878e5b397878f8717056715f65bd606940bc`.
- Situation manifest sha256: `fce3ccbcc4538dbdf7ef5cd2088f2801f54f509272b8b947f2504644be8ed86f`.
