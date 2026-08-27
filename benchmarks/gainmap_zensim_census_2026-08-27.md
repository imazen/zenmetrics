# jpeg-gainmap (Ultra HDR) zensim HDR target loop — 27-cell census (2026-08-27)

The LAST open census of the 7-encoder family (zensim GOAL criterion 4; family
records: jxl / zenwebp / zenavif+zenrav1e / zenjpeg / zenav1-svt in their repos'
`benchmarks/zensim*census*2026-08*`; zenav1-aom ruled premature). Registered
BEFORE any cell ran; design + gates frozen at registration.

## Design (FROZEN)

- **Encoder = THE fleet arm, in-process**: `zenmetrics_cli::sweep::hdr::
  encode_hdr(HdrCodec::JpegGainmap, ...)` — same crate, same commit as the
  fleet executor → drift-free BY CONSTRUCTION (stronger than the family's
  shelled-encoder pattern: the census measures exactly the code the fleet
  runs). Knobs = arm defaults (empty knob map → `gm_quality=85`, `gm_scale=4`
  per `encode_gainmap_hdr`). Dial = **base JPEG q**, integer 1..=100.
- **Judge = shelled sibling** `zenmetrics score --metric zensim --hdr
  --reference R --distorted D.jpg` (the fleet-proven HDR route: Ultra HDR
  decode via `ultrahdr_core::apply_gainmap`, PU-rescale, measured peak;
  parse `zensim=`). Same tree; bin path is an explicit arg. Probe verified
  pre-registration on the hdrgrid first-cell gate pair (`zensim=10.306555`).
- **Search = blind-midpoint bracketed bisection** (the family baseline shape;
  svt precedent): bracket [1,100], seed q=50, monotone NON-DECREASING score
  in q assumed (s<t → lo=q+1, else hi=q-1; next = ⌊(lo+hi)/2⌋),
  **best-not-last** (report the trial with min |s−t|), budget k encodes.
- **Instrument = the FROZEN 27-cell HDR instrument**: zensim
  `benchmarks/hdr_instrument_refs_2026-08-27.tsv` (9 scenes × large/mid/small),
  refs dir `/mnt/v/output/imazen-26-png-v2`-derived
  `/mnt/v/output/imazen-26-hdr-grid-2026-06-14/` (the campaign HDR corpus, an
  EVAL corpus — never training), targets t ∈ {70,80,88} (BHdr scale),
  k ∈ {2,3}.
- **Artifacts persisted**: per-k TSV (scene, tier, t, q_best, encodes_used,
  score_best, |err|, bytes_best, secs) + run logs + each cell's BEST Ultra HDR
  JPEG → `/mnt/v/output/zenmetrics/gainmap-census-2026-08-27/`.

## Gates (FROZEN)

- **G-V validity**: 27/27 cells encode + judge cleanly at BOTH k; harness
  errors loud (judge rc≠0 / unparsable output / encode Err all abort the cell
  with context, never a silent skip).
- **First-cell gate** (1 scene × t80 × k3) before the 54-cell run.
- This is a **CENSUS** (baseline measurement), not an arm: no PASS/FAIL bar on
  |err|. Numbers land in the family table beside the 5 sibling censuses;
  seed-arm value propositions for gainmap quantify against THIS baseline and
  register separately.

## Results

**G-V PASS**: 27/27 cells encoded + judged cleanly at both k (harness
`crates/zenmetrics-cli/examples/gainmap_zensim_census.rs`; cells + best JPGs +
logs at `/mnt/v/output/zenmetrics/gainmap-census-2026-08-27/`).

| k | median \|err\| | ±2 hits | t70 | t80 | t88 | large/mid/small |
|---|---|---|---|---|---|---|
| 2 | 71.519 | 0/27 | 58.15 | 68.15 | 76.15 | 74.10 / 68.15 / 71.52 |
| 3 | 70.426 | 0/27 | 57.83 | 67.83 | 75.83 | 74.10 / 67.39 / 71.08 |

**Structural verdict: the arm is CEILING-BOUND, not seed-bound.** A forced
q→100 climb (t=999, k=7) measured every scene's ceiling under today's judge:
**−4.1 .. 27.5** (median ≈ 13; two scenes NEGATIVE — tulips −4.1, cake −1.1).
Every instrument target (70/80/88) lies far above every scene's ceiling, so
0/27 hits is a property of the ENCODE ARM (single-channel luminance gain map
at gm_scale=4 + filmic-tonemapped base), not of the blind-midpoint search.
No seed arm can help this line; the levers are arm-side (multi-channel gain
map, gm_scale=1, gm_quality>85, alternate base tonemap) and register as arm
waves, not seed waves. Ceiling holds under BOTH judge eras (old-ledger
ceilings on the same 9 renditions: 12.9–54.8 — still all < t70).

Search validity note: the k7 probe shows score rises to a PLATEAU with
±1–3-point noise near the top (best-not-last picked q75–q97 over q100 on 5/9
scenes); monotone bisection stays valid below the plateau and best-not-last
absorbs the plateau noise.

## Judge-era finding (bigger than the census — DATA-criterion hazard)

Cross-checking the harness against fleet data exposed **judge drift**: the
fleet blob `9145c4…` (lighthouse q100, 121,014 B) scores **43.569 in the
hdrgrid ledger** but **25.126 under today's judge** — identical bytes,
Δ≈18. Cause: `score --hdr --metric zensim` computes zensim via the sibling
path dep (`zensim = { path = "../zensim/zensim" }`, profile
`codec_target()`), so every executor image bakes the zensim tree of its
build date. The hdrgrid `zensim_score` column is **era-MIXED**:
wsl-c1/c2 rows 08-07..08-08 (old era, the bulk) sit beside the
08-26..08-27 corruption re-drain rows (image `exec-zensim944hdr-9dffa5ca`
era) in the same run + harvest (`harvest-2026-08-26/scores.parquet`,
built 08-26 06:48). Consumers joining `zensim_score` across cells mix
incomparable scales. ssim2/butteraugli/cvvdp are model-free and unaffected;
features are regime-pinned separately. Remedy recorded in the zensim plan
(re-score the pre-08-26 zensim rows under the pinned current image, or
era-stamp + filter); the HDR-944 breadth leg must not train on the mixed
column. THIS census is single-era by construction (one binary, one tree,
today) and family-comparable (the svt census ran the same-era judge
yesterday).

