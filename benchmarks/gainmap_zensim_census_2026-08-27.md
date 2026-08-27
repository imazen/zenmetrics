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

(filled after the run)
