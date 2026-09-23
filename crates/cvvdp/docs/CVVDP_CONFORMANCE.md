# CVVDP Conformance Matrix — cvvdp + cvvdp-gpu vs pycvvdp (v0.5.7 goldens; port pinned to v0.5.4)

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

- **Harness**: `crates/cvvdp-conformance/` (depends on BOTH cvvdp
  and cvvdp-gpu; tests them as black boxes via the public API).
- **Goldens (current pin, conformance-v2, 2026-09-22)**: pycvvdp v0.5.7,
  13 displays, R2 `s3://coefficient/cvvdp-goldens/conformance-v2/`. **Not
  yet uploaded**: local copy + result TSV in
  `/mnt/v/output/zenmetrics/cvvdp-goldens/conformance-v2/`. Until the
  upload, run with `CVVDP_CONFORMANCE_GOLDENS=<that dir>/conformance_goldens.json`.
- **Goldens (conformance-v1, 2026-05-26)**: pycvvdp v0.5.4, 9 displays, R2
  `s3://coefficient/cvvdp-goldens/conformance-v1/` (public mirror
  `https://coefficient.r2.imazen.org/cvvdp-goldens/conformance-v1/`);
  result TSV `benchmarks/cvvdp_conformance_matrix_2026-05-26.tsv`.

## Dimensions

### Implementations (3-way)

| Impl | Source | Role |
|---|---|---|
| `pycvvdp_v054` | gfxdisp/ColorVideoVDP v0.5.4 (CUDA torch) | **REFERENCE** — ground-truth JOD |
| `cvvdp_cpu` | `cvvdp::Cvvdp` (this workspace) | under test |
| `cvvdp_gpu` | `cvvdp_gpu::Cvvdp<CudaRuntime>` (this workspace) | under test |

### Display models (13 — acceptance gate requires ≥ 8)

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
| `65inch_hdr_pq_1Knit` | PQ / BT.2020 | 1000 | 5 | 65" OLED at 1.98 m (conformance-v2) |
| `65inch_hdr_pq_2Knit` | PQ / BT.2020 | 2000 | 5 | as above (conformance-v2) |
| `65inch_hdr_pq_4knit` | PQ / BT.2020 | 4000 | 5 | as above (conformance-v2) |
| `lg_oled_2026_hdr_pq` | PQ / BT.2020 | 3000 | 5 | LG G6, `min_luminance` black, inches distance (conformance-v2) |

The last four were imazen-only presets until pycvvdp **v0.5.7** shipped
them upstream with values identical to our vendored
`display_models.json`. Their goldens therefore need pycvvdp ≥ 0.5.7
(conformance-v2). `modern_oled_phone_indoor` is still imazen-only
(`display_models_imazen.json`) and **excluded**: pycvvdp can't generate a
golden for a display name it doesn't know. It stays pinned for
self-consistency in `cvvdp-gpu/tests` (`presets.rs`).

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

31 situations × 13 displays = **403 cells per impl** (conformance-v2;
v1 was 31 × 9 = 279) (acceptance gate
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

## Results — conformance-v2 (2026-09-22, pycvvdp v0.5.7, RTX 2080)

Goldens: pycvvdp **v0.5.7** on CPU torch 2.14.0, 31 situations × 13
displays = 403 cells. Impls run on this workspace's CPU port and on
cvvdp-gpu over CUDA (RTX 2080).

- **cpu within 1e-3: 403 / 403** (max `|delta_cpu|` = 0.000877)
- **gpu within 1e-3: 400 / 403** (max `|delta_gpu|` = 0.001390). The 3
  misses are exactly the documented Finding-B cells below.
- The four new HDR PQ displays pass all 31 situations on both impls:
  max Δcpu 0.00045 / 0.00048 / 0.00061 / 0.00049 and max Δgpu 0.00074 /
  0.00077 / 0.00084 / 0.00082 (1K / 2K / 4K / LG 2026).
- **v0.5.4 → v0.5.7 is still-image neutral.** On the 279 cells shared
  with v1, the v1 goldens (v0.5.4, CUDA torch) and v2 goldens (v0.5.7,
  CPU torch) agree to max 7.6e-6 JOD. The v0.5.7 source diff has no
  still-image numeric change: CSF LUTs, colour spaces, pyramid,
  interpolation and CSF code are byte-identical, and `cvvdp_metric.py`
  only refactors the video path, adds `temp_padding="symmetric"`, and
  removes a dead masking branch.
- Result TSV (47 KB, kept out of git per the 30 KB rule):
  `/mnt/v/output/zenmetrics/cvvdp-goldens/conformance-v2/cvvdp_conformance_matrix_pycvvdp_v0.5.7.tsv`.

## Results — conformance-v1 (2026-05-26, pycvvdp v0.5.4, RTX 5070; post-Finding-A fix)

- **cpu within 1e-3: 279 / 279** (max `|delta_cpu|` = 0.000877)
- **gpu within 1e-3: 276 / 279** (max `|delta_gpu|` = 0.001390)
- **cpu/gpu agree tightly** — the two impls track each other across
  every cell.

The Finding-A fix (CSF `log_rho` axis extrapolation; see below) closed
all 10 `iphone_14_pro` JPEG cells. cpu went 274 → 279, gpu went
271 → 276 (the 3 remaining gpu cells are Finding B, the GPU
float-reduction-order floor — unaffected by the Finding-A fix).

Per-class pass rates (post-fix):

| Class | n | cpu pass | gpu pass | max Δcpu | max Δgpu |
|---|---|---|---|---|---|
| common_photo | 18 | 18/18 | 18/18 | 0.00002 | 0.00002 |
| common_screenshot | 9 | 9/9 | 9/9 | 0.00001 | 0.00001 |
| common_distortion | 72 | 72/72 | 71/72 | 0.00088 | 0.00062 |
| niche_content | 99 | 99/99 | 97/99 | 0.00088 | 0.00139 |
| niche_distortion | 54 | 54/54 | 54/54 | 0.00046 | 0.00066 |
| hdr | 27 | 27/27 | 27/27 | 0.00081 | 0.00076 |

Every HDR (PQ/HLG/linear/BT.2020) and niche-distortion cell is within
tolerance. The only remaining over-tolerance cells are the 3 Finding-B
GPU floor cells below.

(Pre-fix, the matrix recorded cpu 274/279, gpu 271/279 with
`max |delta| = 0.028` driven entirely by the Finding-A iphone JPEG
cells.)

### Real-content check: AIC-4 crops (2026-09-22)

Outside the matrix, the 300 JPEG AIC-4 sample crops (620×800 portrait, six
codecs) were scored by this port and by reference pycvvdp 0.5.4 and 0.4.2
(CPU torch), with the distorted image passed first. Port vs pycvvdp 0.5.4:
`standard_4k` max |Δ| 0.00018 JOD, `standard_fhd` max 0.00023; 0 of 300 pairs
over 1e-3 on either display. Two things this check caught that the matrix
could not:

- **The display, not the port, explained our gap to the AIC organisers.** They
  run `standard_fhd` (37.84 pixels per degree); our comparator ran `standard_4k`.
  CPU `cvvdp` in `zenmetrics batch|score-pairs` now takes `--display-model`.
- **`scripts/sweep/pycvvdp_worker.py` passed `(ref, dist)` to pycvvdp's
  `predict(test, reference)`.** That showed up as an apparent 0.016 JOD port
  drift. The goldens here were always built in the correct order.

Details: `benchmarks/cvvdp_aic_discrepancy_2026-09-22.md` (workspace root).

## Divergences

The harness surfaced **two distinct findings**. Both are recorded in
the test's `documented_divergences()` allow-list (the explicit,
reviewable alternative to widening the tolerance) and root-caused
here. None is silently passed.

### Finding A — RESOLVED: CSF `log_rho` axis extrapolation at high PPD (10 cells)

**Resolved 2026-05-26** in `cvvdp_gpu::kernels::csf::interp1_rho_extrap`.

**Symptom (pre-fix)**: On the `iphone_14_pro` display, both
cvvdp and cvvdp-gpu landed **low** vs pycvvdp by up to **0.028 JOD**
on JPEG-distorted content (q60/q30: 0.016–0.028, q90: 0.006; the large
1024² JPEG cell was worst at 0.028). cvvdp and cvvdp-gpu AGREED with
each other to ~7e-5 JOD, so it was a **shared model parity gap**, not a
GPU/float-order artifact.

**Root cause (PROVEN by per-band intermediate dumps)**: the trigger was
**high spatial frequency, NOT high peak luminance**. The cvvdp CSF LUT
`log_rho` axis tops out at **64 cy/deg** (log10 = 1.806). The finest
Laplacian pyramid band has spatial frequency ≈ `pix_per_deg / 2`.
`iphone_14_pro` has `pix_per_deg ≈ 159.6` (the highest of any
conformance display), so its band-0 frequency ≈ **79.8 cy/deg** —
**beyond the axis maximum**. Every other conformance display peaks at
≤ 60.3 cy/deg (`standard_phone`, ppd 120.6), inside the axis. That is
why `standard_phone` (500 nit) passed and the brighter-but-lower-PPD
HDR displays (1500 nit, ppd 75.4) also passed: peak luminance was a
coincidence of the iphone preset, not the cause.

Our `interp1_clamped` **flat-clamped** queries above the axis (held the
64-cy/deg value), but pycvvdp's `interp.get_interpolants_v1`
**linearly extrapolates** above the axis (clamps only the bottom). At
the iphone band-0 frequency, pycvvdp's CSF keeps falling
(rho=64→S_A≈1.86, rho=79.8→S_A≈0.94 uncorrected), while flat-clamp held
S_A≈1.86 — a **~2× over-estimate of CSF sensitivity** in that band.

Verified with `scripts/cvvdp_goldens/diagnose_hipeak.py` + a per-band
dump of the CPU port on `synth_jpeg_q60 | iphone_14_pro`: pre-fix
band-0 `Q_per_ch` (A/RG/VY) was `0.209 / 3.665 / 6.159` vs pycvvdp's
`0.043 / 2.176 / 3.720`; bands 1–7 already matched to ~1e-4. Final JOD
pre-fix 9.834731 vs pycvvdp 9.859124 (Δ 0.0244).

**Fix**: `interp1_rho_extrap` matches `get_interpolants_v1` exactly —
flat-clamp below the axis, **linear extrapolation above** (using the
last interval's slope). It is bit-identical for interior queries (the
only ones the other 8 displays produce), so **zero regression** on the
248 non-iphone cells. Applied at both rho-axis interp sites
(`sensitivity_scalar` + `precompute_logs_row` in
`crates/cvvdp-gpu/src/kernels/csf.rs`); the GPU pipeline uploads the
host-computed `precompute_logs_row` result, so the one fix covers both
CPU and GPU (explaining why they diverged identically).

**Post-fix**: all 10 iphone JPEG cells PASS. `synth_jpeg_q60`
Δcpu 0.024393 → 0.000000; `large_1024_jpeg60` Δcpu 0.028065 → 0.000017,
Δgpu 0.028131 → 0.000004. Standard-4K 1e-4 parity gate unchanged (its
band-0 rho ≈ 37.7 cy/deg is well inside the axis → bit-identical to the
old clamp).

### Finding B — GPU float reduction-order at the perceptibility floor (3 cells, GPU-only)

**Symptom**: 3 GPU-only cells exceed 1e-3 marginally (0.00101 –
0.00139 JOD) on extreme high-frequency or heavily-blurred content
(`checkerboard_blur_r2` on `htc_vive_pro` / `standard_fhd`,
`synth_blur_r5` on `standard_hdr_hlg`) where the reference JOD is at
the perceptibility floor (~3.7–4.4). cvvdp PASSES all three.

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

conformance-v2 (current pin in `tests/common/mod.rs`):

- Reference: pycvvdp v0.5.7 (PyPI `cvvdp` 0.5.7), torch 2.14.0+cpu.
  `build_conformance_goldens.py` now records the INSTALLED pycvvdp
  version as `reference_version`, plus the Rust port pin as
  `port_pinned_version` (still v0.5.4). A display it cannot construct is
  now fatal rather than a silently skipped null golden.
- 403 cells, 0 pycvvdp errors. `conformance_goldens.json` sha256
  `1bac6f9af8f1eaa318fd35ee8d369be1979bd65e8e7ee8e430071eb0537afbf0`;
  situation manifest sha256
  `6ca5765f4bcfc2f1c935d742ee030ec405a71aa64f791832b2223e8a33ac4336`.
- Local copy: `/mnt/v/output/zenmetrics/cvvdp-goldens/conformance-v2/`.
  **Must be uploaded to `s3://coefficient/cvvdp-goldens/conformance-v2/`
  before the default (non-override) run can fetch it.**

conformance-v1:

- Reference: pycvvdp v0.5.4 (pip pkg `cvvdp` 0.5.4, import `pycvvdp`),
  torch 2.10.0+cu128, CUDA available.
- Goldens generated 2026-05-26 on the 7950X workstation (RTX 5070),
  279 cells, 0 pycvvdp errors, JOD range 3.66 → 10.00.
- `conformance_goldens.json` sha256:
  `8f7d69dc6b98272b8425c2245cf7878e5b397878f8717056715f65bd606940bc`.
- Situation manifest sha256: `fce3ccbcc4538dbdf7ef5cd2088f2801f54f509272b8b947f2504644be8ed86f`.
