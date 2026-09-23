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

- `temp_padding="symmetric"`/`"valid"` — only `replicate` (the default).
- `temp_filter` alternates `hp_trans`/`grad_trans` — only the default
  Gaussian band-pass branch.
- `temp_resample` (non-native fps resampling of `Q_per_ch`).
- Heatmap/diffmap output for video, foveation, `dump_channels`.
- GPU (cvvdp-gpu) video, CLI integration, real codec-decoded input.
- `masking_model` variants other than `mult-mutual`.

## Measured parity

Release-mode `cargo test -p cvvdp-conformance --test video_parity`
(2026-09-23, this tree):

- **End-to-end JOD, all 44 cells** (11 situations × 4 displays):
  max |Δ| = **0.000002**, mean |Δ| = 0.000001 — ~500× inside the
  1e-3 gate. Per-display max |Δ|: `standard_4k` 1e-6, `standard_fhd`
  1e-6, `standard_hdr_pq` 2e-6, `standard_phone` 1e-6.
- **`Q_per_ch` stage dumps** (diagnostic gate):
  `vid_flicker_24|standard_4k` max |Δ| 5.1e-5 / mean 1e-6 (n=384);
  `vid_temporal_noise_30|standard_4k` max |Δ| 9.0e-5 / mean 4e-6
  (n=240).
- **Temporal taps** (V1 unit gate, vs dumped torch taps at fps
  24/30/60): max |Δ| ≤ 7e-8, under the 1e-6 requirement.
- **4-channel masking** (V2 unit pin vs `apply_masking_model`):
  bit-exact f32 on the 8×8 synthetic band.

Residual ~1e-6 JOD deltas are f32 accumulation-order noise
(pycvvdp computes in torch f32 with its own reduction order); they are
three orders of magnitude below the acceptance tolerance.
