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


## ARM WAVE 1 (registered 2026-08-27, pre-run): ceiling levers

The census's structural verdict makes seed arms pointless and points every
lever at the ARM. This wave measures the CEILING (not target-hitting) per
config on the same 9 instrument scenes, q=100 fixed (1 encode/cell via the
harness's `GMC_FIXED_Q` mode; knobs via `GMC_KNOBS` JSON):

| arm | gm_scale | gm_quality | gm_multi | hypothesis |
|---|---|---|---|---|
| C0 | 4 | 85 | false | fleet default (measured: −4.1..27.5) |
| C1 | 1 | 85 | false | map resolution is the binding loss |
| C2 | 4 | 100 | false | map quantization is the binding loss |
| C3 | 1 | 100 | false | both |
| C4 | 1 | 100 | true | LUMINANCE-ONLY gain is the binding loss (colored highlights — tulips/cake score NEGATIVE at C0) |

`gm_multi` is a NEW fleet-arm knob (this commit: `GAINMAP_HDR_KNOBS` +
`gainmap_hdr_config(multi_channel)`, knob-guard + roundtrip test); enabling
it exposed and fixed an encode-half bug at the owner (ultrahdr `971ad8d4`:
multi-channel maps fed w*h*3 bytes into a Gray8 encoder — now YCbCr 4:4:4 +
regression test, the #27 decode-half counterpart).

Gates: measurement wave (no PASS bar). Decision output = does ANY config
cross t70 on ANY scene; lever ranking (median ceiling delta per lever).
Judge + refs identical to the census.

### Results

| scene (abbr) | C0 def | C1 s1 | C2 q100 | C3 s1q100 | C4 +multi |
|---|---|---|---|---|---|
| tulips (neg @C0) | −5.44 | −0.67 | −5.03 | −0.61 | −3.62 |
| storage-room | 13.65 | 15.66 | 13.80 | 15.83 | 14.37 |
| castle | 12.45 | 14.55 | 12.70 | 14.77 | 13.90 |
| sashimi | 4.84 | 7.59 | 5.27 | 7.88 | 3.06 |
| sunset | 27.54 | 27.77 | 27.59 | 27.75 | 27.58 |
| archway | 11.92 | 14.82 | 12.17 | 15.00 | 14.12 |
| cathedral | 13.89 | 16.44 | 13.98 | 16.42 | 15.89 |
| cake (neg @C0) | −1.28 | 1.94 | −1.11 | 2.68 | −0.39 |
| lighthouse | 25.13 | 27.26 | 26.00 | 27.36 | 13.76 |
| **median** | **12.45** | **14.82** | 12.70 | **15.00** | 13.90 |

**Decision output: NO config crosses t70 on ANY scene** (best ceiling
anywhere = 27.77). The instrument's targets are unreachable for Ultra HDR
under EVERY available config — the ceiling is structural to the format
pipeline (filmic-tonemapped SDR base + gain-map reconstruction vs the PQ
ref under the PU judge), not to the knobs. Lever ranking:
- **gm_scale=1 is the only real lever**: +2.4 median, up to +5.3
  (tulips −5.4→−0.7, cake −1.3→+1.9 — both negatives rescued toward 0).
  Byte cost NOT measured here; a fleet-default change is a PROPOSAL only
  (user-gated).
- gm_quality 85→100: ≈ +0.2 (noise-level).
- **multi-channel: neutral-to-harmful on 9/9 (C4 ≤ C3 everywhere;
  lighthouse craters 27.4→13.8)** — the colored-highlight hypothesis is
  FALSIFIED; per-channel gains do not rescue the negative scenes. The
  lighthouse crater flags the multi-channel compute/apply path as
  worse-conditioned (encode roundtrip is now test-proven, so the loss is
  in map computation/reconstruction, not the container) — recorded, not
  pursued (the lever is dead for the ceiling question either way).

The zensim-target-loop story for jpeg-gainmap therefore stands as the
census wrote it: ceiling-bound at every config; the format serves
low/mid-fidelity HDR delivery, and t70+ HDR targets need a different
format (zenjxl / zenav1-svt reach 94+ on the same corpus).

