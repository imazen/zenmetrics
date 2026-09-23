# cvvdp video path — design + parity notes

Port of pycvvdp **v0.5.7**'s video scoring path (`cvvdp_metric.py`)
into the pure-Rust CPU crate. The still-image path
(`host_scalar::predict_jod_still_3ch`, `Cvvdp::score`) is untouched and
bit-identical; video support is additive.

## What is ported, and from where

| pycvvdp v0.5.7 source | Rust port |
|---|---|
| `cvvdp_metric.py::get_temporal_filters` | `kernels::temporal::{temporal_filters, temporal_filter_len}` |
| `read_block_of_frames` sliding window + `temp_padding="replicate"` | `video::VideoScorer` ring window + causal FIR |
| `process_block_of_frames` per-band CSF/masking/pool loop | `video` per-frame band loop over `kernels::{csf,masking,pool,pyramid}` |
| `csf.py::castleCSF.sensitivity` (`omega=5` → `o5_c1` LUT) | `kernels::csf::sensitivity_corrected_scalar_o5` + `LOG_S_O5_C1` |
| `apply_masking_model` "mult-mutual", 4 channels | `kernels::masking::mult_mutual_band_4ch` |
| `do_pooling_and_jods` (`is_image=false`) | `kernels::pool::do_pooling_and_jod_video_4ch` |
| `lpyr_dec.py::weber_contrast_pyr.decompose` (`weber_g1`) | `kernels::pyramid::weber_contrast_pyr_dec_scalar` (shared with stills) |

### Temporal filtering

`N = 2*ceil(0.125*fps) + 1` odd-length FIR per channel: three sustained
low-pass kernels (A, RG, VY) and one transient achromatic band-pass
centred on 5 Hz. Taps are computed in closed form (the `irfft` +
`fftshift` reduces to a cosine sum for odd N) in f64 and cast to f32;
verified ≤ 7e-8 absolute against torch's taps for fps 24/30/60.

The filter is applied **causally** exactly as upstream:
`out[t] = Σ_j taps[j] · frame[t − j]` with `frame[k<0] = frame[0]`
(`temp_padding="replicate"` — the default and only ported padding).
The symmetric kernel therefore carries an `(N−1)/2`-frame group delay
vs its centre tap — matching pycvvdp, which likewise emits frame `t`
from the window ending at `t`. The transient channel filters the same
sustained-A frame buffer with the `o5` taps (`sw_ch = 0 if cc==3`).

Because the FIR is causal, output frame `t` depends only on frames
`≤ t`, so `VideoScorer` is genuinely streaming: it holds a ring window
of at most `N` frames per side (`3` planes each — the transient channel
reuses plane 0) plus the running `Q_per_ch` table
(`N_frames × n_bands × 4` f32 — a few KB). Per-frame memory is bounded
by the filter length, not the clip length.

### Per-frame processing

Each emitted frame carries 4 filtered planes per side
(sustained A/RG/VY + transient A). For each side and channel a
`weber_g1` Weber-contrast pyramid is built; the `l_bkg` plane is always
the **same side's sustained-A** filtered plane — which reproduces
upstream's interleaved `[test-Y, ref-Y, test-RG, ref-RG, test-VY,
ref-VY, test-T, ref-T]` tensor divided per-side by `L_bkg[0]`/`L_bkg[1]`
(test ÷ test sustained, ref ÷ ref sustained).

Per band `k` (`n_bands` total, last = baseband):

- `rho = freqs[k]`, baseband `rho = 0.1` (`CSF_BASEBAND_RHO`).
- `logL_bkg` = the **reference** sustained-A pyramid's `log_l_bkg[k]`
  (upstream `logL_bkg[...,1:2]` — channel index 1 = ref sustained Y).
- `S[c]` = sustained `sensitivity_corrected_scalar(rho, logL, c)` for
  c<3, `sensitivity_corrected_scalar_o5(rho, logL)` for the transient
  channel (`omega = [0, 5]`, `cch = 0`).
- `T_p[c] = band_mul · contrast_test[c] · S[c] · CH_GAIN_4[c]`,
  `R_p[c]` likewise; `band_mul` = 1.0 at band 0 and baseband, 2.0
  otherwise (`lpyr_dec.get_band`).
- Baseband: `D[c] = |T − R| · S[c]` (direct absolute difference).
- Other bands: `D = mult_mutual_band_4ch(T_p, R_p)` — v0.5.7's 4×4
  `2**xcm_weights` matrix, `mask_q[0..4]`, `mask_p`, phase-uncertainty
  σ=3 blur (skipped for bands ≤ `PU_PADSIZE`), soft `clamp_diffs`.
- `Q_per_ch[t][k][c] = lp_norm_mean(D[c], β=2)` spatial pooling.

### Pooling (`do_pooling_and_jod_video_4ch`)

`Q_sc[t][c] = lp_norm_sum_k(Q·per_ch_w·per_sband_w, β_sch=4)` →
`Q_tc[t] = lp_norm_sum_c(Q_sc, β_tch=4)` →
`Q = lp_norm_mean_t(Q_tc, β_t=2)` → `met2jod`.
`per_ch_w = [1, ch_chrom_w=1, 1, ch_trans_w=0.8081]`;
`per_sband_w` applies `baseband_weight[4]` at the last band only.
`t_int = 1.0` for video — **no `image_int` multiplier** (that's the
`is_image` path only).

### Single-frame rule

pycvvdp routes `N_frames == 1` through the still-image path
(`is_image`, `temp_ch = 1`, `image_int` applied). `VideoScorer` does
the same: until a **second** frame is pushed, the first frame's raw
sRGB-8 bytes are retained (one frame of bytes — bounded) and the video
pipeline is not run at all. `finish()` on a 1-frame clip calls
`predict_jod_still_3ch` — bit-identical to `Cvvdp::score` on the same
pair. The stored bytes are dropped when frame 2 arrives.

## Public API

```rust
use cvvdp::{CvvdpParams, DisplayGeometry, VideoScorer};

let mut v = VideoScorer::new(
    1920, 1080, 30.0,
    CvvdpParams::default(),
    DisplayGeometry::STANDARD_FHD,
)?;
for (r, d) in ref_frames.iter().zip(dist_frames.iter()) {
    v.push_frame(r, d)?;   // &[u8] sRGB-8, w*h*3 each
}
let jod = v.finish()?;
```

`Cvvdp::video(...)` is an equivalent constructor. A whole-clip
convenience `score_video(ref_frames, dist_frames, w, h, fps, params,
geometry)` wraps the same push/finish loop — streaming and whole-clip
are the same code path, so they are bit-identical by construction.

### pycvvdp API mapping

The crate surface covers the byte-slice analogs of the official
pycvvdp entry points:

| pycvvdp | cvvdp |
|---|---|
| `predict(test, ref, dim_order, fps)` → `(Q_jod, stats)` | `score_video_with_stats(...)` → `VideoStats` |
| `predict` JOD only | `score_video(...)` / `VideoScorer::finish()` |
| `loss(test, ref, ...)` → `10 − JOD` | `VideoStats::loss()` |
| `predict_video_source(vid_source)` streaming | `VideoScorer::push_frame` × N → `finish()` |
| `dim_order="…HWC"` / `"…CHW"` | `FrameLayout::Interleaved` / `Planar` via `VideoScorer::with_layout`, `Cvvdp::video_with_layout`, `score_video_with_stats` |
| `stats['Q_per_ch']` `[F,C,B]` | `VideoStats::q_per_ch` `[frame][band][ch]` (ch = A, RG, VY, transient) |
| `stats['rho_band']` | `VideoStats::rho_band` / `VideoScorer::band_frequencies()` |
| `stats['frames_per_second'/'width'/'height'/'N_frames']` | same-named `VideoStats` fields |
| `temp_padding="replicate"` / `"symmetric"` | `TempPadding::{Replicate,Symmetric}` via `VideoScorer::with_layout_and_padding` / `score_video_with_stats` |

`VideoStats::q_per_ch` rows hold the spatially-pooled per-band masked
differences before temporal/channel pooling — the same quantity
pycvvdp stores in `stats['Q_per_ch']` (transient channel slot is
`f32::NAN` for a one-frame clip, matching upstream image mode which
has no transient channel).

### `temp_padding="symmetric"`

Upstream's only other padding mode (`"valid"` is listed in its
docstring but raises `RuntimeError`). Frame `t`'s filtered output is
`Σ_j F[j]·frame[t−j]`; symmetric maps `frame[-k] → frame[k]` with a
ping-pong wrap for clips shorter than the filter
(`_get_symmetric_frame_index`). Because output `t` needs input frames
up to index `fl−1−t`, `VideoScorer` defers the first `fl−1` outputs
until enough lookahead frames have been pushed (`push_frame` emits
zero or several rows per call; the ring stays bounded by `fl`
frames). Clips with `N < fl` emit everything at `finish()`. Parity:
44 cells, max |Δ| = 3e-6 JOD, including the `vid_short_clip_odd`
(5 < fl=9) ping-pong fixture
(`scripts/cvvdp_goldens/video_goldens_symmetric.json`).

Deliberately not exposed (see "Not ported"): file/codec video
sources (`video_source_file*`, YUV readers), GPU paths, heatmap
outputs, foveation, `temp_resample`, alternate `temp_filter`
branches, ML/PSNR metrics.

### CLI

`zenmetrics score-video` (zenmetrics-cli, `cpu-cvvdp` feature — on by
default) scores two directories of frames:

```bash
zenmetrics score-video \
    --reference-dir ref/ --distorted-dir dist/ --fps 30 \
    [--display-model standard_4k] [--temp-padding replicate|symmetric] \
    [--output plain|tsv|json] [--stats]
```

Frames pair up by lexicographic filename order (zero-pad frame
numbers). Decode streams frame-by-frame into the scorer — memory
stays bounded by the temporal window, never the whole clip.

New `Error` variants: `InvalidFps`, `NoFrames`, `AlreadyFinished` is
avoided by `finish(self)` consuming the scorer. (`Error` gains two
variants — additive for callers that already match non-exhaustively;
the enum is `#[derive]`d plain, so strictly this is a minor-version
addition on a 0.x crate.)

## Conformance goldens & rounding policy

`scripts/cvvdp_goldens/video_goldens.json` (committed, ≈26 KB) is built
by `scripts/cvvdp_goldens/build_video_goldens.py` against the PNG
frames emitted to `/scratch/cvvdpvideo/` by
`cvvdp-conformance::emit_video_situations`. 44 cells = 11 situations ×
4 displays (`standard_4k`, `standard_fhd`, `standard_phone`,
`standard_hdr_pq`).

Rounding (to stay under the 30 KB commit cap):

- **JOD goldens: `round(x, 6)`** — quantization error ≤ 5e-7 JOD,
  which is 2000× below the 1e-3 acceptance tolerance and cannot
  materially consume it.
- **Temporal taps + omega: `round(x, 10)`** — the V1 unit gate is
  1e-6/tap; 10-decimal storage adds ≤ 5e-11.
- **`Q_per_ch` stage dumps: `round(x, 6)`** — diagnostics only; the
  JOD gate is what counts.

## Not ported (follow-ups)

- `temp_padding="valid"` — upstream lists it in the docstring but the
  implementation raises `RuntimeError`; `replicate` (default) and
  `symmetric` are both ported.
- `temp_filter` alternates `hp_trans`/`grad_trans` — only the default
  Gaussian band-pass branch.
- `temp_resample` (non-native fps resampling of `Q_per_ch`).
- Heatmap/diffmap output for video, foveation, `dump_channels`.
- GPU (cvvdp-gpu) video, CLI integration, real codec-decoded input.
- `masking_model` variants other than `mult-mutual`.

## Measured parity

Release-mode `cargo test -p cvvdp-conformance --test video_parity`
(2026-09-23, this tree, SIMD path):

- **End-to-end JOD, all 44 cells** (11 situations × 4 displays):
  max |Δ| = **0.000003**, mean |Δ| = 0.000001 — ~300× inside the
  1e-3 gate. Per-display max |Δ|: `standard_4k` 1e-6, `standard_fhd`
  2e-6, `standard_hdr_pq` 3e-6, `standard_phone` 1e-6.
- **`Q_per_ch` stage dumps** (diagnostic gate):
  `vid_flicker_24|standard_4k` max |Δ| 5.2e-5 / mean 1e-6 (n=384);
  `vid_temporal_noise_30|standard_4k` max |Δ| 9.6e-5 / mean 4e-6
  (n=240).
- **Temporal taps** (V1 unit gate, vs dumped torch taps at fps
  24/30/60): max |Δ| ≤ 7e-8, under the 1e-6 requirement.
- **4-channel masking** (V2 unit pin vs `apply_masking_model`):
  bit-exact f32 on the 8×8 synthetic band; the SIMD
  `mult_mutual_band_4ch_into` path is within 1e-3 relative of the
  scalar pin across sizes 4×4–64×64.

Residual ~1e-6 JOD deltas are f32 accumulation-order noise
(pycvvdp computes in torch f32 with its own reduction order); they are
three orders of magnitude below the acceptance tolerance.

## Performance

`VideoScorer` allocates all per-frame scratch once in `new()`
(`VideoScratch`: filtered planes, `WeberPyramidCache`s, output
pyramids, sensitivity maps, masking intermediates) and reuses it for
every emitted frame — the only per-frame heap traffic left is the
small `Q_per_ch` row (n_levels × 4 floats). Compute runs on the same
SIMD kernels as the still path (`weber_contrast_pyr_into`,
`safe_pow_with_offset_into`, `compute_sensitivities_into`) plus a set
of magetypes/archmage helpers written for the video hot loops:
`vaxpy_into` (temporal FIR accumulation), `vmul2_scale2_into`
(`band_mul·x·s·gain` in the scalar op order), `vabs_diff_into` /
`vabs_diff_mul_into` / `vmin_abs_into` / `vscale_into` (masking),
`vxcm_pool_clamp_4ch_into` (fused 4×4 cross-channel pool + soft
clamp), and `vlp_norm_mean_p2` (`safe_pow_lp` p=2 spatial pooling —
specialized to `BETA_SPATIAL == 2.0`, guarded by a debug assert).
Under the default `parallel` feature the 8 pyramid builds
(4 channels × 2 sides) run on rayon's pool; each owns a disjoint
scratch slot, so results are deterministic regardless of scheduling.

Measured with `cargo run -p cvvdp --release --example video_sweep`
(2026-09-23, this box; scalar = pre-SIMD port, same gates passing):

| size × frames | scalar | SIMD + rayon |
|---|---|---|
| 256×256 ×12 | ~67 ms/frame | 6.90 ms/frame |
| 256×256 ×24 | ~67 ms/frame | 6.88 ms/frame |
| 512×512 ×12 | 279 ms/frame | 31.04 ms/frame |
| 512×512 ×24 | ~279 ms/frame | 29.41 ms/frame |
| 1280×720 ×12 | 986 ms/frame | 134.73 ms/frame |
| 1280×720 ×24 | ~986 ms/frame | 124.94 ms/frame |
| 1920×1080 ×12 | 2237 ms/frame | 331.28 ms/frame |
| 1920×1080 ×24 | ~2237 ms/frame | 324.60 ms/frame |

~6.9× at 1080p vs the scalar port. Note `video_sweep` builds with
`parallel`; a `--no-default-features` build takes the sequential
fallback.

### vs fast-ssim2 per frame

The natural still-metric baseline for video scoring is fast-ssim2
(SSIMULACRA2) applied to each frame pair and averaged. Measured with
`cargo run -p cvvdp --release --example video_vs_ssim2 -- <cvvdp|ssim2>
<W> <H> <N>` (2026-09-23, this box, release, no
`-C target-cpu=native`; same deterministic clip both paths; frames
synthesized lazily inside the timed loop so peak RSS reflects the
metric's own working set; committed data:
[`benchmarks/video_vs_ssim2_2026-09-23.tsv`](../benchmarks/video_vs_ssim2_2026-09-23.tsv)).
`ms/frame` is `(wall − gen)/24`; gen is the shared frame-synthesis
cost measured by the `gen` mode of the same binary.

**24-frame clip, 1 thread (`RAYON_NUM_THREADS=1`):**

| size | cvvdp ms/frame | ssim2 ms/frame | cvvdp user+sys ms/f | ssim2 user+sys ms/f | cvvdp peak RSS | ssim2 peak RSS |
|---|---|---|---|---|---|---|
| 512² | 48.1 | 32.0 | 50.8 | 33.8 | 166 MB | 42 MB |
| 1280×720 | 185.1 | 118.5 | 194.2 | 128.8 | 573 MB | 144 MB |
| 1920×1080 | 432.4 | 269.3 | 452.1 | 290.0 | 1285 MB | 319 MB |

**24-frame clip, 8 threads (`RAYON_NUM_THREADS=8`):**

| size | cvvdp ms/frame | ssim2 ms/frame | cvvdp user+sys ms/f | ssim2 user+sys ms/f | cvvdp peak RSS | ssim2 peak RSS |
|---|---|---|---|---|---|---|
| 512² | 37.0 | 30.8 | 90.8 | 33.3 | 165 MB | 42 MB |
| 1280×720 | 130.3 | 119.3 | 408.3 | 139.2 | 572 MB | 144 MB |
| 1920×1080 | 324.9 | 273.1 | 1083.8 | 309.6 | 1284 MB | 319 MB |

Honest reading:

- cvvdp video costs **~1.2×–1.6× ssim2-per-frame wall** depending on
  threads and size — reasonable for what it computes (4 temporal
  channels, 2 pyramid decomps per channel per frame, 4-channel
  masking + pooling) but it is *not* cheap: this is a ~100–300 MB and
  ~30–430 ms/frame metric, not a ~40 MB / ~30 ms/frame one.
- **Peak RSS ≈ 4× ssim2's** at every size. The streaming bound holds
  (input frames are not retained — RSS is flat in `n_frames`), but
  the bound is the *temporal window*: at 1080p/30 fps the filter is 9
  taps, so the ring keeps 18 DKL frame sets (~24 MB each) plus 8
  pyramid caches and scratch. Bounded ≠ small.
- **Thread scaling is real but shallow for cvvdp**: 1t→8t buys
  ~1.24–1.32× (CPU/wall ≈ 2.6–3.2 effective threads) — only the
  8-way pyramid `rayon::scope` parallelizes; FIR, masking and pooling
  stay serial. ssim2-per-frame shows *no measurable scaling* at these
  sizes (user+sys ≈ wall at 8t; its `rayon` feature parallelizes
  only the gaussian-blur row pass, a negligible fraction), so
  cvvdp's relative gap narrows with threads.
- Scores are not comparable units (JOD 0–10 vs SSIMULACRA2's
  unbounded scale); the ssim2 arm exists to price the "just score
  frames" alternative, not to compare quality.
