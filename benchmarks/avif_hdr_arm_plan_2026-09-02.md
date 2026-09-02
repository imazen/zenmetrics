# AVIF high-bit-depth arm — design, feasibility, and the registered plan (2026-09-02)

**User decisions of record (2026-09-02):** the 2026-07-13 avif-HDR datagen halt is
**LIFTED**, and high-bit-depth AVIF encode testing is to happen.

**Status:** a **DESIGN + FEASIBILITY registration**. It declares nothing to the fleet
and launches nothing. A concurrent lane owns Stage-B (B-6) declarations; §6 sequences
this arm behind it.

**Registered BEFORE any cell of this arm runs**, per the sweep/calibration discipline
(`~/work/zen/CLAUDE.md`). Built on:

| doc | what this design takes from it |
|---|---|
| [`avif_doe_plan_2026-09-01.md`](avif_doe_plan_2026-09-01.md) | the DOE grammar: q ladders (§2.3), the pixel budget (§2.4), the deviation ladder, the K=32 corpus (§2.1) |
| [`avif_doe_stageA_2026-09-02.md`](avif_doe_stageA_2026-09-02.md) | the measured `bd10` main effect (§6), its three coverage gaps (§11.8), trigger B-3, fleet topology (§1.3) |
| [`avif_knob_dossier_2026-09-01.md`](avif_knob_dossier_2026-09-01.md) | the bit-depth axis (§8.1 #11), the port's depth envelope (§4.2), the `with_bit_depth` coercion (§605) |
| [`gainmap_zensim_census_2026-08-27.md`](gainmap_zensim_census_2026-08-27.md) | the HDR judge route, the frozen 27-cell HDR instrument |

---

## 0. TL;DR

1. **Three of the five named "high-bit-depth stills corpora" hold no pixels**, and one is
   not high-bit-depth. si-hdr / aic-hdr2025 / espl-live-hdr / hdr-vdc / rousselot-hdr are
   **64 K – 1.3 M each** on disk — labels, READMEs, MOS/JOD tables. ESPL-LIVE-HDR is
   **tone-mapped SDR PNGs** (Reinhard/Fattal/Durand/…) and licence-restricted to
   non-profit educational use; HDR-VDC is **video**. §1.
2. **The HBD pixels that exist are in two disjoint pools:** 30 true-HDR EXRs (UPIQ
   korshunov 20 + narwaria 10) and **76 16-bit PQ PNG HDR references** at 3000×4000
   (`imazen-26-png-v2`), plus a 1,140-file size grid. §1.2.
3. **10-bit AVIF encoding works today on two independent backends**, with non-ignored
   end-to-end tests. **12-bit does not** (`zenavif::EncodeBitDepth` has no `Twelve`);
   **aom-rs 10-bit does not** (`sweep/encode.rs:1228` hardcodes `bd: 8`). §2.
4. **The A1 `bd10` arm ran a genuine 10-bit encode pipeline on 8-bit source data**, and
   won: BD-rate **−1.02 %**, CI [−1.23, −0.36], 23/31 images, speed 4. It is also the
   **least-covered arm in the wave** — no s6 effect, no interactions, no transfer
   evidence — and fires trigger B-3. §2.4.
5. **The SDR sweep lane structurally cannot carry high-bit-depth *source* data.** Every
   AVIF cell funnels through `Rgb8Image { pixels: Vec<u8> }`. A separate, correctly-built
   HDR lane already exists and does carry 16-bit. §2.5.
6. **The brief's instrument premise is half right, and the half that is wrong is the
   half the fleet uses.** `sweep --hdr` / `score --hdr` are **f32 end-to-end — no 8-bit
   quantization anywhere**. But `score-pairs --hdr` / **jobexec** — the fleet path —
   puts ssim2, zensim, iwssim, butteraugli-CPU and cvvdp-CPU through a **u8 shell**.
   §3.1. This is the single most consequential fact in the document and it drives §4.
7. **⚠ A silent-corruption defect blocks safe execution and must be fixed first.** A PQ
   10-bit AVIF scored **without** `--hdr` is narrowed to 8 bits, relabelled sRGB, and
   scored — **no error, no warning**. PNG has a tripwire for exactly this; AVIF does
   not, and the second-line zenpixels guard does not fire. §3.2, **TODO-0**.
8. **Two things this lane measured that no prior doc records.** All 76 HDR references
   pass the `decode_hdr_ref` contract (16-bit, cICP transfer 16) — **76/76**, so T2 needs
   no conversion step. But they are **33 BT.709 / 43 Display-P3**, and **primaries are
   confounded with content class** (interiors 19/20 BT.709, nature 39/47 P3) — which
   would have silently turned every content conclusion into a content-or-gamut
   ambiguity. §1.2, and it drives a balance constraint (§4.3) and gate G0.5.
9. **Banding is reachable — but only on the PQ route, and never on the wired corpus.**
   The PQ `--hdr` route preserves 10 bits and unlocks BANDVIS on its 10-bit-calibrated
   PU constants. The 8-bit route is not merely blind to banding: requantization is a
   **measured false-positive generator**, so the sign can invert. And the encode-wired
   HDR corpus is gain-map-reconstructed **from 8-bit bases**, so the artifact is in the
   reference. §3.3–3.4.

---

## 1. Source audit — measured, not assumed

Measured 2026-09-02 with `du`, `find`, ImageMagick `identify`, and EXR header reads.

### 1.1 The named datasets

| dataset | on-disk | holds | stills? | HBD? | verdict |
|---|--:|---|---|---|---|
| `/mnt/v/datasets/si-hdr` | **64 K** | README + `experiment_results.csv` | — | — | **labels only** |
| `/mnt/v/datasets/aic-hdr2025` | **164 K** | clone of a README-only repo | — | — | **labels only** |
| `/mnt/v/datasets/espl-live-hdr` | **116 K** | README + MOS tables | — | — | **labels only; and not HBD** |
| `/mnt/v/datasets/hdr-vdc` | **104 K** | JOD CSV + README | **no — video** | — | out of scope |
| `/mnt/v/datasets/rousselot-hdr` | **1.3 M** | 2 `.xlsx` + ReadMe PDFs | — | — | **labels only** |

**ESPL-LIVE-HDR is not high-bit-depth.** Its README defines filenames as
`<Orientation>_<Actual Name>_<Processing Method>.PNG` where the method is a
**tone-mapping operator or multi-exposure fusion** (`ReinhardTMO`, `FattalTMO`,
`DurandTMO`, `WardHistAdjTMO`, `RamanTMO`, `PeceKautz`, `Paul`, `LA_Sig_11`, `GA`, and
two Photomatix effects). Those are LDR *outputs* of HDR processing. Licence: UT Austin,
free for research with citation, **non-profit educational use only** — independently
disqualifying for commercial-codec calibration.

**si-hdr is recoverable and is the best acquisition target.** 181 merged HDR scenes,
`reference/*.exr` at 1920×1280, plus RAW 14-bit stacks. Pixels are on Tower:
`/mnt/tower/input/datasets/si-hdr/reference.zip` (**1,491,596,813 B**),
`reconstructions.zip` (**12,778,645,072 B**), with `SHA256SUMS` beside them; provenance
in `zenpapers:datasets/SI-HDR.pointer.md`. Used once already — the 2026-07-29 transfer
study streamed them per-batch and deleted the pixels
(`/mnt/v/output/zensim/sihdr-transfer-2026-07-29/run_extraction.sh`, the template for a
disk-constrained ingest). Project page:
https://www.cl.cam.ac.uk/research/rainbow/projects/sihdr_benchmark/ — cite Hanji,
Mantiuk, Eilertsen, Hajisharif & Unger, SIGGRAPH '22.
**⚠ Tower returned `StaleNetworkFileHandle` on `/mnt/tower/` root on 2026-09-02** while
deeper paths resolved; remount before relying on it (gate G0.3).

**aic-hdr2025 may now be available and is the highest-value future acquisition.** It is
natively what this arm encodes: **5 source HDR images, Rec.2100 PQ, 10-bit RGB**, and
**100 compressed images across 4 codecs — including AVIF — at 5 bitrates**, with
**JND-scaled** scores (34,560 triplets, 151 participants, 95 % CI ≈ 0.27 JND). README
says release follows QoMEX 2025 (Madrid, 30 Sep – 2 Oct 2025), CC BY 4.0 indicated;
**that date has passed**, so a re-check is cheap (TODO-6). Five sources is too few to
stratify on, but as a **held-out human-labelled AVIF-HDR validation set it has no peer
in this workspace.** Context: https://cloudinary.com/labs/aic-3-and-hdr

### 1.2 The high-bit-depth pixels that DO exist

**Pool A — true-HDR EXR, 30 references.** `/mnt/v/datasets/upiq_extracted/`, 2.5 GB.

| sub-corpus | reference EXRs | all EXRs | reference geometry |
|---|--:|--:|---|
| `korshunov` | **20** | 260 | 944×1080 (i19: 944×872) |
| `narwaria` | **10** | 150 | 1920×1080 |
| `live`, `tid2013` | — | 0 (PNG, SDR) | not HDR |

Header-verified half-float, 3 channels, `LUMINANCE = RELATIVE` — **not** absolute cd/m²;
UPIQ's display model supplies the mapping. `narwaria/i01.exr` records
`FILE_NAME = i01.hdr` (converted from Radiance). **These are the only pixels in the
workspace whose smooth gradients were never 8-bit-quantized** — see §3.4. Labels:
`/mnt/v/datasets/upiq/upiq_subjective_scores.csv` (JOD).

**Pool B — 16-bit PQ PNG, 76 references (the encode-wired HDR corpus).**
`/mnt/v/output/imazen-26-png-v2/**/*.hdr.png` → `identify`: **3000×4000, depth 16**.

| category | n |
|---|--:|
| `1400-lilith-nature` | 47 |
| `1200-lilith-interiors` | 20 |
| `1000-lilith-photos-general` | 6 |
| `1600-lilith-food` | 3 |

- **Provenance (load-bearing):** built by `~/work/hdr-corpus-convert/` from **gain-map
  sources** (Ultra HDR JPEG, gain-map HEIC) per
  `RECONVERT-REPORT-2026-06-11.md` ("totals v1==v2 (2563 sdr + 76 hdr)", "verified
  consistent on all 43 gain-mapped"); PQ quantization by
  `zenpixels-convert::encode_pq16`. 24 filenames carry `_zfold7_`; s25u also present.
- **This reconciles the brief rather than contradicting it.** The imazen-26 HEIC
  *primaries* are 8-bit; the HDR here is the **reconstructed** signal (8-bit base ×
  gain map). Real HDR content, real extended range — but **it inherits the base's 8-bit
  quantization in smooth areas**, which is exactly the wrong property for banding. §3.4.
- **All 76 have a paired `.sdr.png`** (measured: `hdr=76 missing_sdr=0`).
- **Gate G0.2 pre-verified, 76/76 PASS.** Reading the PNG `IHDR` + `cICP` chunks
  directly: every file is **bit depth 16, colour type 2 (RGB), `cICP` transfer
  characteristics = 16 (PQ)**, matrix 0, full range. So the whole corpus is accepted by
  `decode_hdr_ref` as-is — no conversion step, no per-file triage.
- **⚠ But the corpus is NOT primaries-homogeneous, and primaries are confounded with
  content.** The same read gives **33 BT.709 (primaries = 1) / 43 Display-P3
  (primaries = 12)** — matching `HdrRef`'s own note that "1 and 12 both occur in the
  imazen-26-png-v2 corpus". Crossed against category:

  | category | BT.709 (1) | P3 (12) |
  |---|--:|--:|
  | `1200-lilith-interiors` | **19** | 1 |
  | `1400-lilith-nature` | 8 | **39** |
  | `1000-lilith-photos-general` | 4 | 2 |
  | `1600-lilith-food` | 2 | 1 |
  | **total** | **33** | **43** |

  Interiors are 95 % BT.709 and nature 83 % P3. **Content class and colour volume are
  therefore very nearly the same variable in this corpus**, so a naive content-only
  k-means would produce picks whose primaries are effectively determined by their
  content — and any "content effect" measured in T2 would be partly a gamut effect, with
  no way to separate them after the fact. This drives the stratification constraint in
  §4.3 and gate G0.5.
- Derived size grid: `/mnt/v/output/imazen-26-hdr-grid-2026-06-14/`, 1,140 files, 7.8 GB.
  **⚠ its `score_zensim` column is image-era-stamped and known MIXED** (43.57 vs 25.13 on
  the same bytes) — never join across eras; re-score, don't reuse.

**Pool C — the SDR DOE corpus.** K=32 at
`/mnt/v/output/avifsvt-subsample-2026-09-01/sources/` (native) and
`s3://codec-corpus/avif-doe-1024-2026-09-01/` (1024² crops). 8-bit. `bd10`'s corpus.

### 1.3 Suitability as encode references

| pool | suitable? | why |
|---|---|---|
| **C** (K=32 SDR) | **YES, today** | the paired 8-vs-10 read; already `bd10`'s corpus |
| **B** (76 PQ-PNG) | **YES, today** | the only format `decode_hdr_ref` accepts; owned content |
| **A** (30 EXR) | **not until TODO-2** | `decode_hdr_ref` refuses EXR by design; banding-valid |
| si-hdr (181 EXR) | **after acquisition + TODO-2** | 6× pool A, purpose-built, on Tower |
| aic-hdr2025 | **validation only** | 5 sources; native PQ 10-bit + human JND on AVIF |

---

## 2. Encode-path wiring verdict

Every claim read from source at the cited line.

### 2.1 Per-component

| component | 10-bit? | evidence | limit |
|---|---|---|---|
| **`zenavif` public API** | **YES (10 only)** | `src/encoder.rs:74-82` `EncodeBitDepth{Eight,Ten,Auto}`; `:1449 encode_rgb16`, `:1515 encode_rgba16` take `ImgRef<Rgb<u16>>`; `:1009-1024 resolve_bit_depth` | **no `Twelve` variant at all**; `encode_rgb16` **hardcodes `MatrixCoefficients::Identity`** (`:1481`,`:1547`) and ignores `color_model`/`chroma_subsampling`/`bit_depth` ⇒ 16-bit is **always GBR 4:4:4 at 10 bits** |
| **`zenrav1e` / `zenravif`** | **YES 8/10/12** | `ravif/src/av1encoder.rs:76-83`, `:1427 encode_raw_planes_10_bit`, `:1445 …_12_bit`; `zenrav1e/src/api/config/mod.rs:403-404` | reachable from zenavif only to 10 |
| **`zenavif-serialize`** | **YES 8/10/12** | `src/lib.rs:779-780` `high_bitdepth`/`twelve_bit`; `:788-790 pixi`; `:640-641` guard | — |
| **`zenav1-svt` port** ⚠ **IS cloned locally** (HEAD `30cf4b3d0`) | **YES, native u16 e2e** | `pipeline.rs:1152 try_encode_frame_420_hbd(y,u,v:&[u16])`, `:1248 try_encode_frame_hbd`, `:858 bd10_levels_native`, `:7359-7381 bd10_full_rd_supported` | **12-bit refused** (`:1172-1176`); **not byte-exact vs C at low presets** — see §2.3 |
| **— out-of-envelope configs** | **refused loudly** | `pipeline.rs:940-995` | refuses "8-bit-quantized levels under a 10-bit sequence header" |
| **`AvifEncoder::with_bit_depth`** | **coerces silently** | dossier §605: not 10/12 → **8, no error** | **hazard H-BD-3** |
| **`zenav1-aom`** | **port has `bd`, fleet does not** | port `crates/aom-encode/src/lib.rs:177 pub bd: u8`; **fleet: `sweep/encode.rs:1228` literal `bd: 8,`** | **out of scope** |
| **`rav1d-safe` decode** | **YES 10 and 12** | `src/lib.rs:509-510` → `BitDepth16`; `Cargo.toml:74` both default | — |
| **`zenavif` decode → pixels** | **YES — u16 out, no silent truncation** | `frame_convert.rs:153-154`; `:160-162` downscales **only** if `prefer_8bit`; `config.rs:68` `prefer_8bit: false` | truncation is opt-in |
| **`zenmetrics` HDR sweep — `Zenavif`** | **YES — 10-bit PQ 4:4:4** | `sweep/hdr.rs:213-214`, `:391-396` → `zenavif::encode_rgb16` | backend is **zenrav1e** |
| **`zenmetrics` HDR sweep — `Zenav1Svt`** | **YES — 10-bit BT.2020nc 4:2:0** | `sweep/hdr.rs:215-217`, `:501 to_yuv420_bd10`, `:577,585 .with_bit_depth(10)` | the port arm |

### 2.2 The preset 8/9 boundary is a producer change, not a speed change

`bd10_levels_native` (`pipeline.rs:879`) resolves 4:2:0 10-bit as
`preset >= 9 || bd10_full_rd_supported(...)` — **true at every preset, by two different
mechanisms**: the **full-RD funnel** at preset ≤ 8, a **level-only re-encode post-pass**
at preset ≥ 9. zenavif maps speed 1..=10 onto presets {0,1,3,4,6,7,9}, so **speeds 7–10
land on the post-pass and 1–6 on the funnel.**

> **H-BD-1 (registered).** A bd10 preset ladder spanning speed 6→7 crosses a producer
> boundary; an effect measured across it confounds "slower preset" with "different
> 10-bit level producer". **Mitigation:** report T1-a in two blocks and fit no
> BD-rate-vs-speed slope across the seam.

> **H-BD-2.** 10-bit **monochrome** is refused below preset 9 (`pipeline.rs:876-877`).
> Inert here (colour corpus); recorded so a grayscale arm does not rediscover it.

> **H-BD-3.** `AvifEncoder::with_bit_depth` **silently coerces** unknown depths to 8, so
> a typo'd depth yields a valid 8-bit encode labelled 10-bit. **Mitigation:** gate G3
> verifies depth from the **emitted bitstream**.

### 2.3 The port's 10-bit is not byte-identical to C at the presets this arm uses

`zenav1-svt/rust/STATUS.md:55-58`: bd10 64-aligned **p0..p8 = 21.5 %** non-flat
mismatch, partial-SB **26.3 %**; **p9..p13 clean**.

> **H-BD-4 (registered, and it scopes every number this arm produces).** The DOE's
> speeds 1–6 map to presets ≤ 8 — inside the non-parity band. So `bd10` measures **this
> port's 10-bit encoder**, not SVT-AV1 v4.2.0's. That is a legitimate and useful thing
> to measure (the port is what ships), but **no result from this arm may be stated as a
> property of SVT-AV1**, and any comparison against published SVT numbers is invalid.
> Cells at speeds 7–10 (preset 9) are in the clean band and may be compared.

### 2.4 What the A1 `bd10` arm actually did — the brief's central question

**It ran a genuine 10-bit encode pipeline on 8-bit source data, and it won.** The full
chain, every link read from source:

1. `zenmetrics-cli/src/decode.rs:16-28` — `pub struct Rgb8Image { pub pixels: Vec<u8>, … }`,
   "Decode `path` into 8-bit sRGB RGB". `decode_bytes_to_rgb8` funnels **every** input
   (its own comment lists "RGB8, RGBA8, gray, **16-bit**, …") through the RGB8 helper.
2. `PlannedConfig::encode_bytes(&self, source: &Rgb8Image)` — `sweep/plan.rs:69`.
3. `zenavif::encode_rgb8(...)` — `sweep/plan.rs:87`, `sweep/encode.rs:1008`. **`encode_rgb8`, not `encode_rgb16`.**
4. `bd10` token → `stratum.bit_depth = EncodeBitDepth::Ten` (`zenavif/src/sweep.rs:1987-1989`).
5. A1 is svt-rs (`avif_doe_stageA_2026-09-02.md:8`, `:868`) ⇒ `encoder_svt_rs::encode_rgb8_svt_rs`.
6. `effective_bit_depth(config, input_is_16bit=false)` = 10 → `Yuv420Planes::convert(…,10)`
   → `rgbx_to_yuv420_u16` → `try_encode_frame_420_hbd`.

So: **real u16 planes, real 10-bit `av1C`, the port's HBD entry point** — not a flag
passed through and dropped. But the luma carries exactly 8 bits of source information,
widened by the forward kernel. One nuance in zenavif's favour: the 8→10 path is **not**
naive bit replication — the f32 chroma recipe is quantized at the **output** depth, "so
an 8-bit source keeps its chroma-average fraction bits" (`encoder_svt_rs.rs:19-21`,
CHANGELOG `:590-592`), i.e. the 2×2 chroma box average genuinely gains sub-8-bit
precision. **Luma does not.**

**How to read the −1.02 %.** It is a valid measurement of *"encode an 8-bit source at
10-bit internal depth"* — the standard AVIF production trick, where the gain is the
codec's own transform/prediction/loop-filter precision. **It is NOT evidence that the
HBD *input* path works** and must never be cited as such: nothing in A1 touched
`encode_rgb16` or a u16 source buffer.

**Measured (§6, speed 4):** BD-rate **−1.02 %**, CI **[−1.23, −0.36]**, IQR 1.53,
**23/31 wins**, worst +5.9 %. Rank 8 of 15 by magnitude but **one of only two arms whose
CI excludes zero on the winning side**, with the tightest IQR among winners. It changed
every bitstream (**0 %** byte-identical, unlike the dead `tn0`/`scm3` at 288/288).

**Coverage — the worst-covered arm in the wave** (§11.8, verbatim): "`bd10` has no s6
main effect, no interaction coverage and no transfer evidence — it exists only as an A1
arm at s4." Absent from A2's 101 pairs and from AG; B-5 **NOT COMPUTABLE**; already on
the Stage-B list as `bd10@s4` under **B-3**.

### 2.5 Two lanes, and only one of them can carry HBD source

- **SDR sweep lane (the DOE):** the `Rgb8Image` funnel above sits upstream of every AVIF
  cell. **It structurally cannot carry a high-bit-depth source.** Feeding real 10-bit
  through it would need a new 16-bit source type in `sweep/plan.rs`.
- **HDR sweep lane (`hdr`/`hdr-svt` features):** genuinely 16-bit — `sweep/hdr.rs:113
  png_to_rgb16_pq`, `:78 rgb16: Vec<u16>` → `encode_rgb16` / `to_yuv420_bd10`. It already
  does this correctly.

**Consequence for the design:** T1 (paired 8-vs-10 on SDR) belongs on the DOE lane; T2
(HDR) belongs on the HDR lane. They are different harnesses and their numbers are not
interchangeable.

### 2.6 The HDR knob surface is two knobs wide

`sweep/hdr.rs` refuses unwired knobs — good design, and why the gap is visible:

| const | line | knobs |
|---|--:|---|
| `AVIF_HDR_KNOBS` | `:384` | `["lossless", "speed"]` |
| `SVT_HDR_KNOBS` | `:478` | `["preset", "qp"]` |

> "HDR sweep: zenavif knob '{unknown}' is not wired in HDR mode … **refusing to silently
> ignore it**" — `hdr.rs:406-409`

Two knobs per AVIF backend against the SDR DOE's seventeen. `tune`, QM, variance boost,
`sharpness`, `ac_bias`, `max_tx_size`, tiles and screen-content are unreachable in HDR
mode. **An HDR knob DOE is blocked by the harness, not the encoder** — the underlying
`EncodePipeline.hdr` fields are the same object the SDR path drives. TODO-3.

### 2.7 Source format for the HDR lane

`decode_hdr_ref` (`hdr.rs:97`) accepts **PQ-PNG only**:

> "HDR sweep references must be PQ PNGs (.png with cICP transfer 16); got .{ext}.
> **EXR / gain-map sources are score-path-only today**" — `hdr.rs:104-106`

`HdrRef` (`:76-92`) carries `rgb16` PQ codes, `cicp`, `nits`, and a **measured**
`display_peak_nits`. Pool A is one decoder branch from encodable — TODO-2.

---

## 3. Scoring — and the defect that must be fixed before any cell runs

### 3.1 The instrument forks three ways, and the fleet is on the weakest branch

**This corrects the brief's premise.** "The CPU metric path reduces HDR to u8-PU" is
**false for the CLI** and **true for the fleet**:

| route | ssim2 / zensim / iwssim / butter-CPU / cvvdp-CPU | cvvdp-GPU / butter-GPU | evidence |
|---|---|---|---|
| **`score --hdr` / `sweep --hdr`** (umbrella `HdrScorer`) | **f32 end-to-end — no 8-bit quantization anywhere.** ssim2/zensim get integrated PU21 on absolute-nits f32; iwssim gets f32 PU-luma | f32 linear planes | `zenmetrics-api/src/hdr.rs:348-380`, `:514-567`; `fast-ssim2/src/lib.rs:679`; `zensim/src/metric.rs:2624` |
| **`score-pairs --hdr` / `jobexec`** — **the fleet path** | **u8 shell** | f32 (faithful) | `hdr.rs:1091-1100`, quoted below |
| **SDR path (no `--hdr`)** | `round(v/257)` → 8-bit | — | `decode.rs:16-24`; `zenpixels-convert/src/convert_kernels.rs:1835` |

The fleet path's own fallback arm names the affected metrics — this is not an inference:

> "Everything else — **ssim2(-gpu), iwssim(-gpu), CPU butteraugli, plain zensim(-gpu)
> scoring: the PU21/PQ u8 shell** into the normal dispatch, exactly score-pairs'
> `to_sdr_rgb8` + `score_one_pair_maybe_hdr`." — `hdr.rs:1091-1094`, followed by
> `_ => run_metric(metric, reference.sdr_u8(), distorted.sdr_u8(), …)`

**The DOE scores with `ssim2,zensim` — both are in that list.** Only `butteraugli-gpu`
takes the faithful umbrella path there; `cvvdp` (CPU) is u8 but reference-peak-anchored;
`dssim` refuses HDR outright by design ("the u8 shell measured ~0.6 SROCC on UPIQ HDR —
omitted rather than shipped degraded"), which is the precedent for treating a u8-shelled
HDR number as degraded rather than merely noisy.

⚠ **Do not resolve this from `api/hdr.rs`'s comment** ("The u8 shell scored PU-u8
features with the SDR bake; no backend routes it any more"). That describes the
**umbrella `HdrScorer`** only. jobexec is a separate dispatch and still shells.

The PQ `--hdr` decode chain preserves all 10 bits: 10-bit planes →
`scale_pixels_to_u16` (`zenavif/src/convert.rs:91-99`, LSB replication `(v<<6)|(v>>4)`,
exact endpoint map) → `pq_slice_to_nits` (`hdr.rs:264-275`) → `Vec<f32>` cd/m². All 1024
code values survive into f32.

*(Stale comment to ignore: `main.rs:2354-2358` claims "GPU ssim2 → integrated PU21 … rest
→ u8 shell". The owner, `hdr_feeding` at `api/hdr.rs:361-375`, maps `Ssim2` and `Zensim`
to `IntegratedPuNits` on **every** backend. Trust `hdr_feeding`.)*

**Design consequence:** T2 must be scored through **`sweep --hdr`**, not through
jobexec, or its ssim2/zensim numbers are u8-shelled and §3.4's ceiling applies in full.
This is gate **G5** and TODO-4.

### 3.2 ⚠ Silent-corruption defect — TODO-0, fix before executing

**A PQ 10-bit AVIF scored without `--hdr` is narrowed to 8 bits, relabelled sRGB, and
scored — with no error and no warning.** Four links, each verified:

1. PNG has an explicit tripwire (`decode.rs:145-163`, refuses cICP PQ/HLG: "refusing to
   crush it through the 8-bit SDR decode path"). **AVIF has no equivalent**
   (`decode.rs:208-220`).
2. `ManagedAvifDecoder::decode` → `convert_to_image` (`frame_convert.rs:65-166`) **never
   calls `descriptor_with_cicp`** — only the strip/row-sink/grid paths do
   (`sink.rs:96/137/290`). zenavif documents it: "`Unknown` … is what the buffered path
   reports" (`convert.rs:44-48`).
3. So the buffer is `RGB16` with `TransferFunction::Unknown`
   (`zenpixels/src/descriptor.rs:938-943`), `is_hdr_to_sdr(Unknown, Srgb)` = **false**
   (`zenpixels-convert/src/convert.rs:126-133`), and the `HdrSourceRequiresPeak` refusal
   at `:561-566` never fires — despite `hdr-experimental` being enabled.
4. Transfer conversion is skipped when either side is Unknown (`:1696-1701`, "we
   preserve bytes as-is").

This is the zenmetrics#25 failure class ("scores look plausible, mean nothing") and it
falls squarely under the workspace's zero-tolerance rule for code paths that silently
produce wrong pixels. **It would poison this experiment specifically**, because T2
produces exactly the file type that triggers it. Worse, the behaviour is **inconsistent**:
a *grid-tiled* AVIF is tagged (`sink.rs:290`) and would be handled correctly, so the same
experiment could produce correct and corrupt cells depending on tiling.

### 3.3 What each track may claim

| track | reference | encode | compared at | instrument valid? |
|---|---|---|---|---|
| **T1** paired 8-vs-10, SDR sources | 8-bit PNG | 8- and 10-bit AVIF | 8-bit | **YES, fully** |
| **T2** HDR 10-bit RD, `sweep --hdr` | 16-bit PQ PNG | 10-bit PQ AVIF | **f32 nits** | **YES** for RD |
| **T2 if run through jobexec** | as above | as above | **u8 shell** | **degraded — do not** |
| **banding, any track** | — | — | — | **NO** — §3.4 |

**T1 carries no instrument limitation**, and this is worth stating because it is
counter-intuitive: the benefit of 10-bit encoding of 8-bit content is reduced **internal**
transform/quantization rounding, and it materializes as a better *8-bit* reconstruction
at the same rate. Reference and both decodes are 8-bit, so an 8-bit metric is the
**correct** instrument, not a compromised one. `bd10`'s −1.02 % is already honest on this
basis.

### 3.4 The banding ceiling — paste verbatim into any report from this arm

> **INSTRUMENT AND CORPUS LIMITATION (registered 2026-09-02).**
> On the SDR-paired track (T1) reference and both decodes are 8-bit, so the
> ssim2/zensim instrument is fully valid and the BD-rate numbers stand unqualified.
> On the HDR track (T2), **provided it is scored through `sweep --hdr`**, the path is
> f32 end-to-end and no 8-bit quantization occurs; if it is scored through
> `score-pairs --hdr` / jobexec, ssim2 and zensim are u8-shelled and the numbers order
> encodes only coarsely.
> **No result from this arm may be stated as a banding claim**, for two independent
> reasons, either sufficient alone:
> **(i) instrument** — on any 8-bit-quantized route, requantization does not merely hide
> the effect, it **manufactures the opposite one** (§3.5); and
> **(ii) corpus** — the encode-wired HDR references are **gain-map-reconstructed from
> 8-bit bases**, so their smooth gradients were 8-bit-quantized before reconstruction and
> the banding 10-bit prevents is **present in the reference**.
> A banding claim requires **both** a never-8-bit-quantized source (pool A EXR or
> si-hdr, i.e. TODO-2) **and** the PQ `--hdr` route (which does preserve 10 bits and does
> unlock BANDVIS on its 10-bit-calibrated PU constants). Until both hold, banding is
> **NOT MEASURED** — never a null, never a zero.
> **Do not cite HDR-VDP-2.2**: in-tree but unvalidated and unwired — its own header says
> "no number out of this crate should be published as an HDR-VDP-2 score"
> (`zenmetrics/crates/hdrvdp/src/lib.rs:18-49`, `publish = false`).

### 3.5 Is an 8-bit-quantized decode a valid banding instrument? — NO, and it inverts

Three mechanisms, two of them measured in-repo.

**1. The requantizer reconstructs the artifact it is measuring.** Banding is quantization
contouring. In an 8-bit encode the step is ≥1 8-bit code; in a 10-bit encode of the same
ramp it is 1 10-bit code = 0.25 8-bit codes. `round(v/257)` cannot represent a step
smaller than one 8-bit code, so it **re-imposes an 8-bit staircase on the 10-bit decode
at the same ramp-crossing pixels** where the 8-bit encode banded. In the case that
matters most — 8-bit bands at the 1-LSB level, 10-bit clean — the advantage is cancelled
*by the measurement*. Only gross banding (≥2–3 codes) survives; the near-threshold
regime, where "10-bit fixes it" is the interesting claim, is discarded.

**2. The detector's own constants are tied to the 8-bit lattice.** zensim's BANDVIS
half-points are defined as code steps: `BV_DELTA_LO_SDR = 0.00169` ≈ 0.5× the Y-plane
|∇| of a one-8-bit-code plateau step at mid-gray (`zensim/src/feature_v2.rs:373-374`);
`BV_DELTA_LO_PU = 0.00124` ≈ 0.5× one **10-bit PQ** code step at 100 cd/m²
(`:379-382`). Measured one-step magnitudes: SDR cbrt-Y 8-bit **0.003379**; PU-Y 10-bit PQ
0.001848–0.002726. **On the SDR route the smallest representable step already sits at 2×
the lower half-point** — a sub-8-bit-code improvement is below the band by construction.
On an SDR route `HL_BIN1/HL_BIN2` are structural zeros (`feature_v2.rs:7938-7948`).

**3. Requantization is a measured false-positive generator for this detector.** The
BANDVIS gate matrix records dithered destinations as a **MISS**: "dst-side dither **FIRES
rather than masks** (noise dither ratio 1.72, ordered 1.55): any ~1-code quantization
residual has dense mid-band curvature, and the flatness mask is REF-side by design"
(`zensim/benchmarks/append2_bandvis_gates_2026-07-27.md`). `round(v/257)` on a smooth
10-bit gradient produces exactly that residual. **So the instrument would report *more*
banding on the arm that has less — the sign inverts, it does not merely attenuate.**

Corroboration from the same calibration: "u8 fixture floor — 8-bit synthetic 'smooth'
ramps carry a ~0.3-code post-downscale curvature residual; **10-bit/f32 fixtures needed
for clean deband margins in future gates**."

Two secondary reasons: ssim2 is an SDR-designed structure/contrast metric with no
banding term; and T1's 8-bit *reference* cannot express the smoother gradient a 10-bit
encode produces, so on T1 a genuine banding improvement would register as **increased**
distortion. Banding is therefore excluded from T1's questions entirely.

---

## 4. The design

**T1 executes with zero new wiring. T2's baseline executes after TODO-0 (the defect) and
TODO-4 (route assertion); T2's knob block needs TODO-3.**

### 4.1 Common grammar (inherited unchanged)

- **q ladders:** control 29-point `1, 5,10,…,70, 72,74,…,98` (q 100 ≡ q 98, measured);
  knob ladder **9-point** `{5,15,25,35,45,60,76,90,96}`, a strict subset; probe
  `{15,45,90}`. Low-q density ≥ high-q density.
- **Pixel budget:** 1024×1024 content-aware crops at native resolution, never upscale,
  native below budget — with the as-built pass-through clause applied as implemented.
- **Comparison:** BD-rate vs the `deviations = 0` control at the same
  `(image, speed, q)` and size; A0R's dense 29-q ladder is the control.
- **CI:** 10,000-resample percentile bootstrap over images, seeded.
- **Sign:** negative BD-rate = fewer bits at matched quality = the arm wins.

### 4.2 Track 1 — HBD-SDR: close the `bd10` gaps (executable now)

Corpus pool C (K=32), backend `zenav1-svt` via zenavif. The registered discharge of
trigger **B-3** for `bd10`, plus the two gaps B-3 does not cover.

| block | shape | cells | question |
|---|---|--:|---|
| **T1-a** preset ladder | `bd10` × 6 non-s4 speeds × 9 q × 32 | **1,728** | does the win hold across the speed dial — and does it break at the producer seam (H-BD-1)? |
| **T1-b** interactions | `bd10` × 15 live single-deviation arms × 9 q × 32, at s6 | **4,320** | is 10-bit additive with the knobs, or does it substitute for the quantizer-domain ones? |
| **T1-c** s6 main effect | `bd10` @ s6 × 9 q × 32 | **288** | the missing second preset; makes B-5 computable for `bd10` |
| **T1-d** transfer gate | `bd10` × 3 q × 32, **native** | **96** | does −1.02 % survive native size, or is it a 1 MP crop artifact? |
| **T1 total** | | **6,432** | |

- **T1-b is the load-bearing block.** 10-bit and the QM/variance-boost arms all act on
  the quantizer; the wave measured them only in isolation from depth. If they substitute
  rather than add, a model treating `bit_depth` as independent will over-credit it.
- **T1-d is the cheapest high-value block.** Stage-A already found one reduced-size
  artifact with a **sign flip** (`tl1.0`: +0.65 % at 1024² → −0.12 % native) and an 8.6×
  shrink on `tl1.1` — "does this survive native size" is a demonstrated risk on this
  corpus, not hypothetical.
- **`hbd_mds` EXCLUDED by measurement.** The one knob that exists only under 10-bit is
  **inert here**: `bd10.rs:90` records "`--hbd-mds` is **NEVER consulted on the allintra
  path**", and `allintra_hbd_md` (`:93-104`) returns DUAL for every reachable preset
  regardless. Declaring it buys 1,728 cells of exact zeros. Recorded so the next lane
  does not re-buy it — the discipline Stage-A applied to `tn0`/`scm3`.
- **Excluded questions:** banding (§3.5); 10-bit *display*; 12-bit (no `Twelve` variant);
  4:4:4 (refused on the svt path); any claim about SVT-AV1 itself (H-BD-4).

### 4.3 Track 2 — HDR-10: the missing RD baseline

Corpus pool B (76 PQ-PNG), on the **HDR sweep lane**, scored through **`sweep --hdr`**.
There is no HDR RD baseline anywhere in the workspace — the gain-map census measured
target-hitting, not rate/quality — so this is the control every future HDR arm
differences against.

**Stratification — K = 16, and how to cluster HDR content honestly.** Two corrections to
the obvious approach:

- **zenanalyze's 102 picker features are 8-bit renditions by design** — "RGB16/RGBA16 —
  converted to RGB8 by taking the high byte"; for HDR "the analyzer measures the SDR
  rendition" (`zenanalyze/src/lib.rs:373-382`). And the DOE's `feat_*` columns are
  **zensim** features, not zenanalyze's (`sweep/feature_writer.rs:18`).
- So: **cluster on the paired `.sdr.png`** (all 76 present, measured), which is an
  aligned, documented proxy — same content, same converter, same capture, differing only
  in transfer/range, so the structure features clustering uses are the ones best
  preserved. What it cannot see is dynamic-range structure.
- **Mitigation, and it is cheap:** carry explicit HDR axes alongside the k-means label —
  `HdrRef.display_peak_nits` (already measured per reference at decode) plus
  zenanalyze's `tier_depth` scalars, the one part of zenanalyze that reads **source
  samples** rather than an RGB8 view: `EffectiveBitDepth`, `HdrHeadroomStops`,
  `HdrPresent` (`zenanalyze/src/tier_depth.rs`, `src/feature.rs:501/519/526`;
  `experimental`-gated). Stratify so the picks span the peak-luminance range by
  construction, not by luck.

K = 16 from 76 (order 12–24 per the brief), centroid-nearest members. Categories are
unbalanced (nature 47 of 76), so k-means is materially better than random, which would
return a nature-dominated set.

**Primaries balance constraint (registered, from the §1.2 confound).** Because primaries
are nearly determined by content in this corpus, the K=16 selection is **constrained to
carry at least 5 of each primaries value** (BT.709 and P3), taking the
centroid-next-nearest member within a cluster when the nearest would violate the
constraint, and recording every such substitution in the picks TSV. A fully crossed
primaries × content design is **impossible here and is not attempted** — there is exactly
1 P3 interior and 8 BT.709 nature images, so some cells cannot be filled. Consequently:

> **Registered analysis restriction.** No content-class conclusion may be drawn from T2
> without reporting the primaries cross-tab for the images behind it, and no
> primaries/gamut conclusion may be drawn without the content cross-tab. Where the two
> cannot be separated — which is most of this corpus — the finding is reported as
> **"content-or-gamut, not separable in this corpus"**, never attributed to one.

This is a limitation of the available HDR corpus, not of the design; si-hdr (TODO-7)
would supply a primaries-homogeneous alternative and is the registered way out.

| block | shape | cells | question |
|---|---|--:|---|
| **T2-a** HDR RD baseline, svt | `zenav1-svt` HDR × 7 presets × 29 q × 16 refs | **3,248** | the reference RD curve for 10-bit AVIF on HDR stills |
| **T2-b** backend contrast | `zenavif` (zenrav1e, PQ 4:4:4) × 3 speeds × 9 q × 16 | **432** | 4:2:0 BT.2020nc port vs 4:4:4 identity-PQ zenrav1e — two wired arms never compared |
| **T2 phase-1 total** | | **3,680** | |

Both use **only wired knobs** (`preset`/`qp`, `speed`), so phase 1 needs no encoder
change. **The HDR knob block is NOT declared here** — gated on TODO-3, registers
separately. Note T2-b compares two arms that differ in *three* ways at once (backend,
chroma, matrix), so it is a **contrast, not a controlled comparison** — reported as such.

### 4.4 Out of scope

- **`zenav1-aom`** — `sweep/encode.rs:1228` hardcodes `bd: 8`. Not a gap this lane closes.
- **12-bit** — `zenavif::EncodeBitDepth` has no `Twelve`; the svt port refuses it.
- **f32 SDR scoring** — `zensim` (`Srgb16Rgba`) and `fast-ssim2` (`ImgRef<[u16;3]>`)
  both accept 16-bit at the library level, but **no zenmetrics caller constructs one**.
  Noted as TODO-5, not used here.
- **Banding claims** — §3.4.
- **Any fleet declaration by this lane** — §6.

---

## 5. Gates (frozen at registration)

**G0 — source integrity, before any cell.**
- **G0.1** sha256 every reference fed to an encode; record counts (T1: 32, T2: 16) and
  the manifest path. A cell whose `inputs[0]` sha is not in the manifest is poison.
- **G0.2** T2 references must be **16-bit PNG, cICP transfer 16** — verified by reading
  the file, not the extension. **Pre-verified 2026-09-02: 76/76 PASS** (§1.2); re-run at
  execution to catch corpus drift.
- **G0.5 — primaries balance and disclosure.** The K=16 picks must carry ≥5 BT.709 and
  ≥5 P3, the picks TSV must record each pick's primaries, and the primaries × content
  cross-tab must be published **with the picks, before any encode**. Any T2 finding
  stated in content terms must carry that cross-tab (§4.3).
- **G0.3** if Tower is used, remount and re-verify `SHA256SUMS`; the 2026-09-02
  `StaleNetworkFileHandle` must be cleared, not worked around.
- **G0.4** T2's k-means picks, cluster sizes, `display_peak_nits` and `tier_depth`
  scalars written to a committed TSV **before** any encode.

**G1 — first-cell artifact gate.** One cell per track encodes, persists and scores
end-to-end before the block is declared; verify the blob in the run's `blobs/` prefix.
Fires **once, on the first cell**, per the ML-pipeline discipline.

**G2 — control identity.** T1 differences against A0R; assert shared
`(image, speed, q)` control cells are byte-identical to existing A0R blobs (Stage-A got
3,857/3,857). A drifted control invalidates every BD-rate.

**G3 — 10-bit decode-verify (the gate H-BD-3 exists for).** For every distinct 10-bit
cell shape, read back from the **stored blob** and assert `av1C` + sequence header carry
`high_bitdepth = 1`, `twelve_bit = 0`. **A request for depth 10 is not evidence of a
10-bit stream.** Also assert `bd10` cells are **not** byte-identical to their 8-bit
control (Stage-A: 0/288 identical; a regression to 100 % is the `tn0`/`scm3` failure).

**G4 — producer-seam declaration (H-BD-1).** T1-a reported in two blocks split at the
funnel/post-pass seam; no BD-rate-vs-speed slope fitted across it.

**G5 — route assertion (the defect gate, non-negotiable).** Every T2 scoring invocation
must **prove** it took the PQ `--hdr` route: assert the run used `sweep --hdr`, that the
reference decoded with `tc = 16`, and that the scorer reported the f32/PU feeding — not
the u8 shell. **Until TODO-0 lands, a mis-routed run returns plausible numbers with no
error**, so this gate cannot be satisfied by absence of failure; it requires positive
evidence in the run log. Any T2 cell without it is discarded, not reinterpreted.

**G6 — no silent knob acceptance.** Any HDR cell whose knob map contains an unwired key
must fail loud (already `hdr.rs:406`); the gate asserts it stays that way, because a
knob ignored in HDR but honoured in SDR poisons cross-mode joins.

**G7 — instrument statement.** §3.4's ceiling reproduced verbatim in any report carrying
a T2 number; H-BD-4 (port ≠ SVT-AV1) reproduced in any report carrying a T1 number.

### 5.1 Pre-registered analysis questions (frozen — nothing else may be reported as a finding)

1. **Q1** Does `bd10`'s −1.02 % hold at s6? *(T1-c; makes B-5 computable.)*
2. **Q2** Does it hold across the preset ladder, reported per producer block? *(T1-a, G4.)*
3. **Q3** Is `bd10` additive with the 15 live knobs, or does it substitute for the
   quantizer-domain ones (QM, variance boost)? *(T1-b — the primary question.)*
4. **Q4** Does the effect survive native resolution? *(T1-d; `tl1.0`'s sign flip is the precedent.)*
5. **Q5** What is the rate/quality curve of 10-bit AVIF on HDR stills? *(T2-a — a
   baseline, no PASS/FAIL bar, like the gain-map census.)*
6. **Q6** How do the two wired HDR AVIF arms differ? *(T2-b — a contrast, not a controlled comparison.)*

**Not questions of this arm:** banding; 10-bit display; aom; anything about SVT-AV1 itself.

---

## 6. Budget and sequencing

**Measured throughput:** 6,132 cells/h, three workers on r7900x (Stage-A §1.3), which
also records the reusable lesson — *on a box shared with an uncapped container, add
workers, not cores.* Tower capped to `cpuset 0-7` (household floor 8 cores free); dev on
`cpuset 24-30`.

| block | cells | est. wall @ 6,132/h |
|---|--:|--:|
| T1-a | 1,728 | 0.28 h |
| T1-b | 4,320 | 0.70 h |
| T1-c | 288 | 0.05 h |
| T1-d (native — slower per cell) | 96 | ~0.1 h |
| **T1 subtotal** | **6,432** | **~1.1 h** |
| T2-a | 3,248 | 0.53 h |
| T2-b | 432 | 0.07 h |
| **T2 phase-1 subtotal** | **3,680** | **~0.6 h** |
| **TOTAL** | **10,112** | **~1.7 h** |

Three caveats, stated not absorbed. (a) 6,132 cells/h is the 1024² budget rate; **T1-d
and all of T2 are larger** (native crops; 3000×4000 HDR sources), so T1-d is costed with
a hand margin. (b) **10-bit encode costs more time than 8-bit, and that cost is not
measured anywhere** — Stage-A's `bd10` "+5.9 %" is a *bitrate* column, and A4 (the timing
block) was never declared. **T1 must not be used for any speed claim**; A4's 6-point size
ladder is the owner and is a separate registration. (c) T2 scored through `sweep --hdr`
is a different harness from jobexec and its throughput is unmeasured.

**Sequencing (explicit).** The **B-6 wave (`acb3`, `shp3`, 27,840 cells ≈ 4.6 h) is
running now** under a concurrent lane. This arm is sized to **spare capacity** and
sequenced **AFTER B-6**: 10,112 cells is **36 % of B-6's size** and well inside the
60,000-cell Stage-B envelope, but the two must not contend for the same workers. **This
lane declares nothing.**

Recommended order — cheapest-discriminating first: **TODO-0 (the defect) → T1-c → T1-d →
T1-a → T1-b → T2-a → T2-b.** T1-c and T1-d together are 384 cells and answer whether the
effect is real at a second preset and at native size, which gates whether T1-b's 4,320
cells are worth buying at all.

---

## 7. Wiring gaps — one precise TODO each

This lane does not implement encoder internals. Sizes are estimates against the code read
above.

| # | what | repo · file | est. | why |
|---|---|---|--:|---|
| **TODO-0** ⚠ | **Add the PNG-style HDR tripwire to `decode_avif`** — or call `descriptor_with_cicp` on zenavif's buffered path so the existing `HdrSourceRequiresPeak` guard fires | `zenmetrics` · `crates/zenmetrics-cli/src/decode.rs:208-220`; and/or `zenavif` · `src/decoder_managed/frame_convert.rs:65-166` | ~30 LOC + test | **a silent-corruption defect (§3.2)** — a PQ 10-bit AVIF scored without `--hdr` yields plausible, meaningless numbers with no error, and behaves *differently* for tiled vs plain files. It would poison T2 specifically. **Fix before executing.** Zero-tolerance class |
| **TODO-1** | Correct `~/work/zen/CLAUDE.md`: `zenav1-svt` is listed "private; **NOT cloned locally**" — it **is** cloned at `/home/lilith/work/zen/zenav1-svt` (HEAD `30cf4b3d0`), consumed by zenavif as a **path** dep (`zenavif/Cargo.toml:136-152`) | zen workspace doc | 1 line | the crate index is the map sessions navigate by |
| **TODO-2** | Teach `decode_hdr_ref` to accept **EXR** (absolute-luminance linear f32) → `HdrRef` via the existing PQ16 quantizer, with an explicit display model for `LUMINANCE=RELATIVE` sources | `zenmetrics` · `sweep/hdr.rs:97-110`; quantizer exists as `zenpixels-convert::encode_pq16` | ~120 LOC + tests | **unlocks the only banding-valid sources** (30 UPIQ EXR now, 181 si-hdr after TODO-7). Without it §3.4's ceiling can never lift. The display model must be a recorded parameter — the si-hdr study set the "registered display model" precedent |
| **TODO-3** | Extend `AVIF_HDR_KNOBS` / `SVT_HDR_KNOBS` to the SDR DOE's knob set, mapping onto the same `EncodePipeline.hdr` fields | `zenmetrics` · `sweep/hdr.rs:384`, `:478`, `:386`, `:480` | ~200 LOC + tests | blocks the HDR **knob** DOE. Must preserve refuse-unknown-knobs (G6) and must handle SVT's **sticky `apply_tune_overrides`**, which only ever *sets* — reusing one pipeline leaks tune-3 state into later cells |
| **TODO-4** | Give `score-pairs --hdr` / jobexec the same f32 feeding the CLI umbrella has, for ssim2/zensim/iwssim/butter-CPU/cvvdp-CPU | `zenmetrics` · `crates/zenmetrics-cli/src/hdr.rs:1095-1100`, `:915-921` | ~80 LOC | the **fleet** path is the u8 shell (§3.1). Until fixed, T2 must be scored via `sweep --hdr` (G5), which constrains how T2 can be fleeted |
| **TODO-5** | Wire a 16-bit SDR source type so `zensim` (`Srgb16Rgba`) and `fast-ssim2` (`ImgRef<[u16;3]>`) can be driven at full depth from the CLI | `zenmetrics` · `sweep/plan.rs:69`, `metrics/` | ~150 LOC | both libraries already resolve 16 bits (`linear-srgb/src/scalar.rs:229-238`, a 65,536-entry LUT); only the CLI funnels to u8. Prerequisite for any depth-honest **SDR** measurement |
| **TODO-6** | Re-check **aic-hdr2025** availability (README: "after QoMEX 2025", 30 Sep–2 Oct 2025 — passed); if released, ingest **validation-only** | dataset acquisition | ~1 h | the only native PQ 10-bit sources **with human JND scores on AVIF** in reach |
| **TODO-7** | Stage **si-hdr** `reference.zip` (181 EXR, 1.49 GB, on Tower with `SHA256SUMS`) | dataset staging | ~1 h + disk | 6× pool A; pairs with TODO-2 |
| **TODO-8** | Correct the stale header in `bd10.rs`: "**UNWIRED** (add `pub mod bd10;` when integration starts)" while `lib.rs:15` already declares it | `zenav1-svt` · `rust/crates/svtav1-encoder/src/bd10.rs:3-5` | 3 lines | a reader who trusts it concludes 10-bit is unimplemented and abandons the arm — the exact wrong call, given `bd10` measured −1.02 % |

**TODO-0 gates T2.** Nothing blocks T1. TODO-3 blocks the HDR knob block; TODO-2 blocks
banding work permanently until it lands.

---

## 8. Where everything is

| artifact | path |
|---|---|
| HDR PQ-PNG references (76) | `/mnt/v/output/imazen-26-png-v2/**/*.hdr.png` |
| HDR size grid (1,140 files, 7.8 GB) | `/mnt/v/output/imazen-26-hdr-grid-2026-06-14/` ⚠ mixed zensim era |
| HDR corpus provenance | `/mnt/v/output/imazen-26-png-v2/RECONVERT-REPORT-2026-06-11.md` |
| True-HDR EXR (30 refs) | `/mnt/v/datasets/upiq_extracted/upiq_dataset/images/{korshunov,narwaria}/` |
| UPIQ subjective scores | `/mnt/v/datasets/upiq/upiq_subjective_scores.csv` |
| si-hdr pixels (unstaged) | `/mnt/tower/input/datasets/si-hdr/{reference,reconstructions}.zip` + `SHA256SUMS` |
| SDR DOE corpus (K=32) | `/mnt/v/output/avifsvt-subsample-2026-09-01/sources/`, `s3://codec-corpus/avif-doe-1024-2026-09-01/` |
| Stage-A scored dataset (49,120 × 18) | `/mnt/v/output/zensim-avifdoe/doe_scored_2026-09-02.parquet` |
| HDR sweep implementation | `zenmetrics/crates/zenmetrics-cli/src/sweep/hdr.rs` |
| HDR scoring feeding (the owner) | `zenmetrics/crates/zenmetrics-api/src/hdr.rs:348-380`, `:514-567` |
| 10-bit envelope + guards | `zenav1-svt/rust/crates/svtav1-encoder/src/pipeline.rs:858-995`, `:7359-7381` |
| 10-bit support module | `zenav1-svt/rust/crates/svtav1-encoder/src/bd10.rs` |
| port-vs-C bd10 parity status | `zenav1-svt/rust/STATUS.md:55-58` |
| HDR EXR corpus builder (score-path) | `zenmetrics/crates/zenhdr-corpus/` |
| external HDR read runner | `zensim/scripts/external_reads/run_external_reads.py` |

---

## 9. Corrections this document makes to the record

Recorded explicitly so they are greppable, per the docs-update discipline.

1. **`~/work/zen/CLAUDE.md`** lists `zenav1-svt` as "private; NOT cloned locally". It is
   cloned at `/home/lilith/work/zen/zenav1-svt`. (TODO-1.)
2. **`bd10.rs:3-5`** declares itself UNWIRED; `lib.rs:15` declares the module. (TODO-8.)
3. **`zenmetrics-cli/src/main.rs:2354-2358`** says "GPU ssim2 → integrated PU21 … rest →
   u8 shell". The owner (`hdr_feeding`, `api/hdr.rs:361-375`) maps `Ssim2` and `Zensim`
   to `IntegratedPuNits` on **every** backend. The comment is stale.
4. **The brief's source list** named si-hdr / aic-hdr2025 / espl-live-hdr as "true
   high-bit-depth stills corpora on disk". None holds pixels; espl-live-hdr is
   tone-mapped SDR and licence-restricted. The real on-disk pools are UPIQ EXR (30) and
   imazen-26 PQ-PNG (76). (§1.)
5. **The brief's "imazen-26 has no ≥10-bit sources worth using"** is true of the HEIC
   primaries and **not** of the corpus as built: 76 gain-map-reconstructed 16-bit PQ HDR
   references exist and are the only encode-wired HDR corpus. Their 8-bit-base
   provenance — not their bit depth — is what disqualifies them for banding. (§1.2, §3.4.)
6. **The brief's "the CPU metric path reduces HDR to u8-PU"** is **false for
   `score --hdr` / `sweep --hdr`** (f32 end-to-end, no quantization) and **true for
   `score-pairs --hdr` / jobexec**, which is the fleet path. The distinction is the
   difference between a valid and an invalid T2. (§3.1, TODO-4.)
7. **New measurement made by this lane:** the imazen-26 HDR corpus is **33 BT.709 / 43
   Display-P3**, and primaries are **confounded with content class** (interiors 19/20
   BT.709; nature 39/47 P3). No prior doc records this. It would have silently turned
   every content conclusion in an HDR arm into a content-or-gamut ambiguity. (§1.2,
   §4.3, G0.5.)
8. **Also verified by this lane:** all 76 HDR references pass the `decode_hdr_ref`
   contract (16-bit, `cICP` transfer 16) — **76/76**, so T2 needs no corpus conversion
   step. (§1.2, G0.2.)
