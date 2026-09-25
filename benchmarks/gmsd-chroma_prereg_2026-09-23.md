# gmsd-chroma Part A preregistration (committed 2026-09-24)

## Authorized peer rescore amendment (2026-09-24 10:35Z)

The user explicitly authorizes rescoring missing peer columns with
fast-ssim2 main on an available remote worker. The schema search did not
establish the required CPU columns. Freeze main at
`f011259a0cd2fb7f538e08398ba41c0ccd0dfcb4`, resolved through the GitHub
commit API before scoring. Use the public `compute_ssimulacra2` CPU API,
default runtime SIMD dispatch, `imgref` RGB8 sRGB ingress, one thread,
and the same frozen 1,539 TRAIN pairs and canonical RGB bytes as the other
arms. No GPU column is substituted. Retain source/archive, Cargo lock,
driver, binary, raw-input and output SHA-256 hashes. Join by exact pair
key with full-population coverage; no partial peer intersection. Statistics,
signs, bootstrap settings, input population and numeric gates are unchanged.
No human scores have been read. If the remote workers remain unreachable,
record that failure; this amendment does not authorize inventing scores.

Quarantined; no human label has been opened by this lane. Scope is the
2026-09-24 `GMSD_CHROMA_brief.md`, with DEVIN_COMMON and CODEX_NOTE.

## Definition and concrete caller

Implement the author's default **sum** MDSI, with one additive public entry
`gmsd::mdsi_rgb8(reference, distorted, width, height, stride_bytes) -> Result<f64>`.
The current caller is zenmetrics `score`/`score-pairs --metric mdsi`, using the
existing `cpu-gmsd` build feature. No multiplicative-mode API is proposed.
Existing GMSD arithmetic remains untouched. MDSI needs f64 colour conversion
and convolution; the rounded integer GMSD luma path cannot implement its
definition. Reuse the crate's band geometry and archmage tier pattern with
f64 row kernels and deterministic pooling.

The oracle is the author release at
<https://www.mathworks.com/matlabcentral/fileexchange/59809-mdsi-ref-dist-combmethod>:
archive SHA-256 `8b78597d180d5e250a928fbee7b5ffb8a714ce354e3e8640eec7703692db472b`,
MDSI.m `4dff198b28e57b39b20877c7bcb75af2c6278c95f9f321362a2046ee3e37351b`,
licence `0b717c37c30188b069cdf9cf19435a076f5d3b5f8f9bcb8f90e946f0e6949a2f`.
It stays outside the repo under `/var/tmp/gmsd-chroma/ref/` and is only an
oracle. The reference software's licence and header are recorded in the external archive. No reference executable enters scoring or training.

Freeze C1=140, C2=55, C3=550, alpha=0.6, the published L/H/M matrix, the
`max(1,round(min(w,h)/256))` box factor and the paper's deviation pooling
(q=o=1/4), all as stated by the paper. Wording about the reference software's
internals was removed from this copy at landing; the unredacted original and
its sha256 are in the external archive (see `gmsd-chroma_oracle_archive.pointer.md`).
MS-GMSD/MS-GMSDc remain optional pending author-code discovery; any
paper-derived implementation must be labelled as such and independently
checked with NumPy, never called author-software parity.

## Inputs, gates, and statistics

Before the first image-score run, freeze a label-free pair manifest with
pixel hashes, dimensions, source roles, paths, crop/stride variants and seed
20260924. Use at least 64 TRAIN pairs: at least 32 KADID TRAIN colour rows
(types 03/05/06/07/08/09 admitted by the canonical type descriptions),
16 CID22-train and 16 SafeSyn. Include native, odd, and sub-64 dimensions,
strided copies, identities, and negative-GCS stress fixtures. No CID22
human population, SELECT/TERMINAL KADID or KonFiG holdouts is admitted.

Existing pixel-only authorities to reuse, with SHA-256:

- `/var/tmp/gmsbank/calibration/pairs.tsv`:
  `f6f201193100c537a633ee81b09587554e40ac8359011f1d452850bc4ff870bb`.
- `/var/tmp/gmsbank/calibration/planes.tsv`:
  `aa49fbe114acb464c9d88316c28b7bd1e0b1ee8a17b6def9a742b6d4d034a6d3`.
- `/var/tmp/gmsbank/peer_gmsd/kadid_train.keys.tsv`:
  `0d573a4452d908c79c7128f62d3e2d3ddc02c9ef4a57c0ba732507b9dd00d398`.

Numeric gate, stated before scoring: f64 score relative error <=1e-9 for
nonzero reference scores, with absolute error <=1e-12 for zero; maps use
absolute <=1e-12 and relative <=1e-9 away from zero (|reference|>1e-12).
Report both maxima, including near-zero values separately. No f32 path is
planned. A C3=55 negative control must fail. Available CPU tiers and 1/8
threads must agree bitwise with scalar; unavailable architectures are
reported untested. Failed gates block a parity claim.

TRAIN colour reporting freezes the implementation before labels are read.
Use GMSD/MDSI/(MS-GMSDc if implemented)/stored fast-ssim2, KADID TRAIN colour
types, TID2013 colour types (all TRAIN), KonFiG originsplit TRAIN colour
diffusion. Use only `zen_stats.panel` for SROCC and reference-clustered
bootstrap 95% intervals, 2,000 replicates and seed 20260924. No tuning or
selection follows these numbers; ssim2's prior use of these sets is stated.
Freeze and hash the exact reporting population before any human-label read.

Speed uses zenbench interleaving, 64/256/1024/4096 square, 1 and 8 threads,
same RGB inputs, CPU dispatch enabled, no target-cpu=native. Report measured
cells and alpha+beta*pixels; contended execution is labelled CONTENDED.
All heavy work uses `~/tmp/devin/heavy`, all bulky output/cache paths are
under `/var/tmp/gmsd-chroma/`, and /home must retain at least 20 GiB free.

## Prerequisites and Part B

`master@origin` b50cde1f lacks gmsd. A local merge preserves it together
with existing optimized-GMSD tip 8fc65d91; only the overlapping changelog
insertions needed resolution, keeping both. Neither parent is rewritten.
No push or promotion is authorized.

Part B starts only after GMSBANK_CORRECTIONS_DONE.md exists and supplies its
tip. Its calibration/registry revision needs a separate committed prereg;
this document does not authorize starting B early or reading its labels.

## Amendment 1 (2026-09-24, before any image score or label read)

The KADID type-number list above had a transcription error. The dataset
owner at <https://database.mmsp-kn.de/kadid-10k-database.html> identifies
colour types **04,05,06,07,08** (03 is motion blur; 09 is JPEG2000).
Use 8 pairs per colour type, one per reference within type, then 16 CID22
TRAIN and 16 SafeSyn: 72 native TRAIN pairs before derived edge cases.
This corrects the intended population; it is not a result-driven change.

## Amendment 2 (2026-09-24, before any image score or label read)

The phrase "TID2013 colour types (all TRAIN)" above does not override the
source split. DATA_SPLITS.md section 8 assigns **12 reference images to
TID TRAIN**, 9 to SELECT, and 4 to TERMINAL. Use only the existing TID TRAIN
view, then filter colour distortion types. No full-TID human table is an
admitted input. KADID remains its 40-reference TRAIN view; KonFiG remains
originsplit TRAIN. Population manifests must enforce these roles before
reading labels. No reporting population or labels have been opened yet.

## Amendment 3 (2026-09-24, before any image score or label read)

**Amendment 2 is superseded.** It read the historical section-8 table without
the immediately following **section 8.1 user ruling**, which retires TID to
TRAIN-only: all 3,000 rows / 25 references, no TID evaluation surface.
The split compliance owner implements this ruling, and the September-19
joint-core plan and current GMSBANK source registry agree. The original
TID all-TRAIN admission was correct. Report colour types from all 25
references as TRAIN description, with SSIMULACRA2's in-sample caveat.
This correction precedes population construction and every label read;
no observations motivated it. KADID and KonFiG admissions are unchanged.

## Reporting population freeze (2026-09-24, before human-value read)

Use `/var/tmp/gmsd-chroma/colour_v2/population.json`, SHA-256
`13db0836bff1966a8593b89907752e3c7bdc9b9fc26116039280f33aadf91940`.
Its score-pairs TSV is
`615bfce812ec38a9c7082c7fb83ac77f0833513dca64d06a16af812510f5f837`.
There are 1,539 stimuli/pairs: KADID 200 per type 04–08, TID 125 per type
02/18/22/23, KonFiG 39 colour-diffusion rows. TID colour types follow the
dataset's noise-in-colour-components, saturation, dithered colour
quantization, and chromatic-aberration definitions (dataset paper:
<https://core.ac.uk/download/pdf/52995558.pdf>).

Type labels come from the source stimulus metadata, not the canonical
pixel path. The key bank can use one canonical path for identical decoded
content across different named stimuli. The first path-based freeze
`colour/population.json` (1,499) missed 40 KADID type-08 rows and is
**superseded, not used**. Its script and outputs are retained externally.
The corrected freeze projects only `pair_key`, `source_row_id`, `codec`,
and `knob` from TRAIN label containers; `human_score` is not requested or
decoded. It binds the complete containers by SHA-256 before values are
read. The report joins on source stimulus identity and pair key; predicted
values may be reused for pixel-identical stimuli without dropping rows.

## Paper-derived MS-GMSD conventions (2026-09-24, before measurement)

With no author software obtained, implement equations (4)–(11) of Zhang
et al. using an independent NumPy transcription as the oracle. This is a
declared paper variant, **not author-software parity**. Four scales start
at the original image. Freeze masking alpha=0.5; weights
0.096/0.596/0.289/0.019; beta1=0.01, beta2=0.32, beta3=15.

The paper leaves numerical conventions unspecified. Freeze RGB8 in its
0..255 gamma-encoded units; YIQ rows (0.299,0.587,0.114),
(0.595716,-0.274453,-0.321263), (0.211456,-0.522591,0.311135);
unrounded f64 planes; normalized 3x3 Prewitt with zero padding; c=170
(the GMSD reference's 8-bit stabilizer); iterative aligned 2x2 box
downsampling with replicated final odd row/column, ceil dimensions. Use
population deviation, evaluated by two centred passes in row order.
Evaluate joint I/Q RMSE only at scale 3 and fuse with the paper's logistic
weight, without clipping. Identity must be exactly zero.

Use the same 116-case oracle population including all 72 native TRAIN
pairs. Require relative score error <=1e-9 for nonzero oracle scores and
absolute <=1e-12 for zero; maps use the already declared MDSI map gates.
Test all available tiers and 1/8 threads for exact bits; wrong c=17 must
fail on at least one distorted input. This choice is made without labels,
scores, parameter sweeps or performance observations.
