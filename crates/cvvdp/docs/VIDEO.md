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
by the filter length, not the clip length. With
`VideoScorerOptions::low_memory` the ring stores the raw source
samples (u8/u16/f32 — whichever `push_frame*` variant fed the scorer)
instead of f32 DKL planes (4× smaller for u8 — ~450 MB → ~113 MB per
side at 1080p/30 fps) and re-converts each slot at emit; the
conversion is deterministic, so scores are bit-identical.

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
source samples are retained (one frame — bounded) and the video
pipeline is not run at all. `finish()` on a 1-frame clip calls the
still path for the pushed sample type — bit-identical to
`Cvvdp::score`/`score_u16`/`score_f32` on the same pair. The stored
samples are dropped when frame 2 arrives.

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

### Input sample types — `u8` / `u16` / `f32`

All three `push_frame*` variants feed the same display-encoded
contract that pycvvdp's `video_source_array._get_frame` applies:

| sample | normalization | pycvvdp analog |
|---|---|---|
| `&[u8]` | `v / 255` | `uint8` clip |
| `&[u16]` | `v / 65535` | `uint16` clip |
| `&[f32]` | used as-is | `float32`/`float16` clip |

The normalized value is **display-encoded** under the configured
`DisplayModel`'s EOTF — `[0,1]` for sRGB, PQ, HLG, gamma and BT.1886
display models, absolute cd/m² for `Eotf::Linear`. So a real 10-bit
HDR10/PQ master (e.g. BT.2020 + SMPTE 2084 code values unpacked to
`u16`) scored under `standard_hdr_pq` gets genuine nit-domain
processing — the u16 path is not an 8-bit upscale.

```rust
v.push_frame_u16(&ref16, &dist16)?;  // display-encoded, /65535
v.push_frame_f32(&ref32, &dist32)?;  // display-encoded [0,1]
```

`score_video_u16`/`score_video_f32` (+ `_with_stats`) are the
whole-clip equivalents; `Cvvdp::score_u16`/`score_f32` are the still
counterparts. Both layouts (`Interleaved` RGBRGB…, `Planar` R…G…B…)
are supported for all three types. Mixing sample types within one
scorer is rejected with `Error::MixedSampleTypes` — the temporal ring
is typed by the first push.

u16/f32 conversion evaluates `Eotf::forward` analytically per sample
(the u8 path keeps its LUT for speed; both land on the same curve).
Parity vs pycvvdp v0.5.7 (`scripts/cvvdp_goldens/video_goldens_u16.json`):
16 video cells max |Δ| = **2e-6 JOD**, 16 still cells max |Δ| =
**1.2e-5** — including `vid16_hdr10_sky_real`, a committed RGB16-PNG
crop of real HDR10 content (JonaNorman/HDRSample `hdr-pq-sky`,
BT.2020/PQ, ~1 000 nits peak in crop, 100 % low-bit usage; distorted =
8-bit roundtrip banding).

### pycvvdp API mapping

The crate surface covers the byte-slice analogs of the official
pycvvdp entry points:

| pycvvdp | cvvdp |
|---|---|
| `predict(test, ref, dim_order, fps)` → `(Q_jod, stats)` | `score_video_with_stats(...)` → `VideoStats` |
| `predict` JOD only | `score_video(...)` / `VideoScorer::finish()` |
| `loss(test, ref, ...)` → `10 − JOD` | `VideoStats::loss()` |
| `predict_video_source(vid_source)` streaming | `VideoScorer::push_frame` × N → `finish()` |
| `video_source_array` `uint8`/`uint16`/`float` dtypes | `push_frame`/`push_frame_u16`/`push_frame_f32`, `score_video*` variants; `Cvvdp::score`/`score_u16`/`score_f32` for stills |
| `dim_order="…HWC"` / `"…CHW"` | `FrameLayout::Interleaved` / `Planar` via `VideoScorer::with_layout`, `Cvvdp::video_with_layout`, `score_video_with_stats` |
| `stats['Q_per_ch']` `[F,C,B]` | `VideoStats::q_per_ch` `[frame][band][ch]` (ch = A, RG, VY, transient) |
| `stats['rho_band']` | `VideoStats::rho_band` / `VideoScorer::band_frequencies()` |
| `stats['frames_per_second'/'width'/'height'/'N_frames']` | same-named `VideoStats` fields |
| `temp_padding="replicate"` / `"symmetric"` | `TempPadding::{Replicate,Symmetric}` via `VideoScorer::with_layout_and_padding` / `score_video_with_stats` |
| — (no upstream analog; memory knob) | `VideoScorerOptions::low_memory` via `VideoScorer::with_options` / `Cvvdp::video_with_options` — source-sample ring window, bit-identical scores |

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
    [--output plain|tsv|json] [--stats] [--low-memory]
```

Frames pair up by lexicographic filename order (zero-pad frame
numbers). Decode streams frame-by-frame into the scorer — memory
stays bounded by the temporal window, never the whole clip.
`--low-memory` switches the ring to the u8-window mode described
above (scores unchanged).

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
small `Q_per_ch` row (n_levels × 4 floats). Two structural
deduplications keep the pyramid stage cheap: all four channels share
one `gauss_l` background pyramid per side (every channel's `l_bkg`
source is the same sustained-achromatic plane), and only the
reference achromatic pyramid materializes `log_l_bkg` planes — the
other seven run a `vweber_band_nolog_into` variant that skips the
`log10` pass entirely. For channel 0 the image pyramid *is* the
shared background pyramid, so its per-level expands run once.
Compute runs on the same SIMD kernels as the still path
(`safe_pow_with_offset_into`, `compute_sensitivities_into`) plus a
set of magetypes/archmage helpers written for the video hot loops:
`vfir_into`/`vfir2_into` (fused N-tap temporal FIR — one pass over
all window-slot planes per output channel; `vfir2` shares the
plane-0 loads between the sustained and transient achromatic
channels; `low_memory` keeps the per-tap `vscale2`/`vaxpy2` chain
since its source planes materialize one slot at a time),
`vmul2_scale2_pair_into` (paired test/reference CSF scaling),
`vabs_diff_mul_lp2_sum` + `lp2_finish` (fused baseband |t−r|·s +
p=2 norm in one pass, split into raw partial + normalize for banded
reduction), `vweber_band_nolog_into` (log-free Weber band), and
`vxcm_pool_clamp_4ch_sqsum_partial` + `xcm4_finish` (fused 4×4
cross-channel pool + soft clamp + p=2 accumulation inside
`mult_mutual_band_4ch_into`, which returns the four pooled values
directly and never materializes clamped-difference planes —
`s_map`/`d` aliases are served by the `m_mm`/`t_p` scratch).

Under the default `parallel` feature nearly every stage runs in
deterministic bands (`src/par.rs`): band boundaries are a pure
function of plane length — never of `RAYON_NUM_THREADS` — so
single- and multi-threaded runs execute the identical partition.
Elementwise maps are bit-identical by construction; the two
reductions fold per-band partials in fixed band order. The σ=3
masking blur bands both separable passes by rows (the vertical
pass reads its halo from the full horizontal-pass plane by
absolute row index), the DKL converts band the same way, the two
shared `gauss_l` builds run under `rayon::join`, and the 8 band
stages (4 channels × 2 sides) run on rayon's pool over disjoint
scratch slots. Planes below `PAR_MIN_SAMPLES = 1<<17` stay serial —
dispatch overhead dominates below ~128k samples.

Measured with `cargo run -p cvvdp --release --example video_sweep`
(2026-09-24, this box; scalar = pre-SIMD port, same gates passing):

| size × frames | scalar | SIMD + rayon |
|---|---|---|
| 256×256 ×12 | ~67 ms/frame | 6.03 ms/frame |
| 256×256 ×24 | ~67 ms/frame | 5.18 ms/frame |
| 512×512 ×12 | 279 ms/frame | 23.22 ms/frame |
| 512×512 ×24 | ~279 ms/frame | 20.93 ms/frame |
| 1280×720 ×12 | 986 ms/frame | 69.51 ms/frame |
| 1280×720 ×24 | ~986 ms/frame | 67.58 ms/frame |
| 1920×1080 ×12 | 2237 ms/frame | 162.64 ms/frame |
| 1920×1080 ×24 | ~2237 ms/frame | 154.28 ms/frame |

~14.5× at 1080p vs the scalar port (committed data:
[`benchmarks/cvvdp_cpu_video_sweep_2026-09-24.tsv`](../benchmarks/cvvdp_cpu_video_sweep_2026-09-24.tsv);
pre-banding 2026-09-23 numbers — 231 ms/f at 1080p — in
[`..._2026-09-23.tsv`](../benchmarks/cvvdp_cpu_video_sweep_2026-09-23.tsv)).
Note `video_sweep` builds with `parallel`; a `--no-default-features`
build takes the sequential fallback. The 256² rows sit under
`PAR_MIN_SAMPLES` — unchanged by design.

### vs fast-ssim2 per frame

The natural still-metric baseline for video scoring is fast-ssim2
(SSIMULACRA2) applied to each frame pair and averaged. Measured with
`cargo run -p cvvdp --release --example video_vs_ssim2 -- <cvvdp|ssim2>
<W> <H> <N>` (2026-09-24, this box, release, no
`-C target-cpu=native`; fast-ssim2 is the sibling checkout at
v0.9.0-15-gf011259 via a direct path dev-dep; same deterministic clip
both paths; frames synthesized lazily inside the timed loop so peak
RSS reflects the metric's own working set; committed data:
[`benchmarks/video_vs_ssim2_par_2026-09-24.tsv`](../benchmarks/video_vs_ssim2_par_2026-09-24.tsv)).
`ms/frame` is `(wall − gen)/24`; gen is the shared frame-synthesis
cost measured by the `gen` mode of the same binary.

**24-frame clip, 1 thread (`RAYON_NUM_THREADS=1`):**

| size | cvvdp ms/frame | ssim2 ms/frame | cvvdp user+sys ms/f | ssim2 user+sys ms/f | cvvdp peak RSS | ssim2 peak RSS |
|---|---|---|---|---|---|---|
| 512² | 26.6 | 19.4 | 26.7 | 19.6 | 118 MB | 30 MB |
| 1280×720 | 94.9 | 81.4 | 95.8 | 81.7 | 405 MB | 96 MB |
| 1920×1080 | 213.0 | 186.6 | 212.5 | 186.7 | 908 MB | 213 MB |

**24-frame clip, 8 threads (`RAYON_NUM_THREADS=8`):**

| size | cvvdp ms/frame | ssim2 ms/frame | cvvdp user+sys ms/f | ssim2 user+sys ms/f | cvvdp peak RSS | ssim2 peak RSS |
|---|---|---|---|---|---|---|
| 512² | 20.6 | 15.1 | 68.3 | 28.3 | 123 MB | 30 MB |
| 1280×720 | 56.3 | 63.7 | 298.8 | 110.8 | 421 MB | 96 MB |
| 1920×1080 | 135.4 | 144.2 | 763.8 | 290.8 | 932 MB | 214 MB |

Honest reading:

- **8t wall is at parity everywhere ≥720p** — 56.3/135.4 vs ssim2's
  63.7/144.2 ms/frame (ssim2 run-to-run spread at 1080p is
  ~131–144). ssim2 still leads at 512² (15.1 vs 20.6) where
  cvvdp's four-channel constant dominates small planes. This was
  1.7–1.9× behind before the parallel+traffic passes.
- **Serial CPU is within ~1.14–1.17× of ssim2** (212.5 vs 186.7
  ms/f at 1080p; 95.8 vs 81.7 at 720p; 1.36× at 512²). cvvdp still
  computes strictly more per frame — 4 temporal channels, 8
  pyramid decomps, 4-channel masking + pooling — the gap is
  inherent work, not overhead.
- **8t user+sys remains ~2.6× ssim2's** (763.8 vs 290.8 ms/f at
  1080p). The parallel stages are bandwidth-bound: contention
  stalls inflate per-frame CPU even when wall time matches.
  Reducing plane traffic is what moves this number, not more
  threads.
- **Peak RSS is ~4.4× ssim2's** (932 vs 214 MB at 1080p). The
  streaming bound holds (input frames are not retained — RSS is
  flat in `n_frames`), but the bound is the *temporal window*: at
  1080p/30 fps the filter is 9 taps, so the ring keeps 18 DKL frame
  sets (~24 MB each, ~450 MB total) plus pyramid caches and
  scratch. Bounded ≠ small.
- **`low_memory` cuts 1080p RSS ~36 % (600 vs 932 MB, ~2.8× of
  ssim2)** and is JOD-bit-identical — but it is now purely a
  memory knob: the emit path re-converts each source frame up to
  `2×fl` times during its ring lifetime (per-tap `to_dkl` +
  accumulate), and that conversion work cannot be amortised
  without storing the planes — which is what the default mode is.
  At 1080p it costs +29 % wall at 8t / +46 % at 1t. Use it when
  the ~600 MB bound matters more than time.
- What landed after the first parallel pass (all bit-identical —
  TSV `score` column unchanged at 1e-6): FIR→band-0 writes killed
  the `filt` staging planes and their copies (~128 MB/frame of
  traffic); shared sustained-A expands cut 24 redundant
  `gausspyr_expand` calls per side to 6
  (`weber_bands_from_gauss_lexp`); the four CSF sensitivity maps
  share one pass over `log_l_bkg` (index math + load amortised);
  masking's `|T−R|` + `safe_pow` is one fused pass. A u8-source
  ring + fused u8→DKL FIR was built, measured, and reverted: the
  per-emit reconversion cost ~9× per frame and regressed both wall
  and CPU — the `low_memory` structure only pays when its ring is
  also the *storage* win, not when conversion must be repeated.
- cvvdp's `parallel` feature gates all of this — band helpers run
  the same decomposition sequentially when it's off (`--no-default-
  features --features std` builds clean). Planes below
  `PAR_MIN_SAMPLES = 1<<17` stay serial regardless of pool size —
  dispatch overhead dominates below ~128k samples (ssim2's Tuning
  measured the same cliff at 1<<18).
- Scores are not comparable units (JOD 0–10 vs SSIMULACRA2's
  unbounded scale); the ssim2 arm exists to price the "just score
  frames" alternative, not to compare quality.
