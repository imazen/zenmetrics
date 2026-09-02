# AVIF high-bit-depth arm — design, feasibility, and the registered plan (2026-09-02)

**User decisions of record (2026-09-02):** the 2026-07-13 avif-HDR datagen halt is
**LIFTED**, and high-bit-depth AVIF encode testing is to happen.

**Status:** a **DESIGN + FEASIBILITY registration**, now with an **EXECUTION RECORD**
appended as §10 (2026-09-02). §§1–9 are the registration as frozen; §10 records what
was declared, gated and run, and carries **two corrections to §4.2's counts** and the
**T2-a wiring blocker** found at execution time. Read §10 before citing any number or
cell count from §4.

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

**✅ RESOLVED 2026-09-02, commit `e9e2ef71`** (appended — the finding above stands as
written except for point 4, corrected below).

`decode_avif` now takes `ManagedAvifDecoder::decode_full` and **refuses** an AVIF whose
`ImageInfo.transfer_characteristics` is PQ (16) or HLG (18), naming the code, the
transfer and the real bit depth and pointing at `--hdr` — the same policy, and nearly the
same wording, as the PNG cICP tripwire, so one rule now covers both formats. `decode_full`
is the same decoder `decode_with` already selected here (the `AvifDecoder` sibling is
`unsafe-asm`-gated and never enabled) and the one `sweep::hdr::decode_avif_to_nits`
already drives, so **decoded pixels are unchanged**. Route (a) of the TODO was taken;
route (b) would have been a change to zenavif's public behaviour and was not needed.

*Measured pre-fix, on the then-current release binary:* `ref_64.avif` patched to `tc=16`
and `tc=18` both scored `ssim2=96.137450`, **bit-identical to the sRGB original** — the
signalling was ignored end to end. On genuine 10-bit PQ files
(`cosmos1650_yuv444_10bpc_p3pq.avif`, `colors_hdr_rec2020.avif`), `100.000000` against
themselves. Post-fix all four are a loud refusal, the 10-bit ones reporting `at 10-bit`.

**Correction to point 4 — the tiled/plain divergence does not exist on this route, and
the pre-fix behaviour was worse than recorded.** A real 1×5 grid AVIF
(`sofa_grid1x5_420`) patched to `tc=16` also scored **silently** pre-fix
(`ssim2=100.000000`). `sink.rs:290` is the *row-sink* grid path;
`decoder.rs::decode_grid`, which the buffered `decode`/`decode_full` uses, does not call
`descriptor_with_cicp` either. So **both** shapes were corrupt, not one — which makes the
fix simpler, since there was no correct-for-grid behaviour to preserve. `decode_full`
branches on `grid_config()` internally and returns an `ImageInfo` either way, so the
single guard site covers both by construction. Post-fix the grid file is refused.

**The guard is deliberately narrow — only 16 and 18.** Narrowing a 10-bit *SDR* AVIF to
8 bits is `decode.rs`'s documented contract and **T1/`bd10` depends on it**; BT.2020's SDR
transfers (14, 15) neighbour PQ/HLG in the CICP table. SDR non-regression was measured
pre-fix vs post-fix binary — identical output on 8-bit AVIF vs PNG, the `tc=1` variant,
grid-tiled SDR, and **genuine 10-bit SDR** AVIFs (`plum-blossom` profile0 10bpc 4:2:0 and
profile1 10bpc 4:4:4).

*Gates:* `crates/zenmetrics-cli/tests/avif_hdr_tripwire.rs` — `pq_transfer_is_refused`
and `hlg_transfer_is_refused` (both **FAIL** at the parent commit, with the recorded
message "expected a loud refusal, got a silent 64x64 decode (12288 bytes)"),
`committed_sdr_fixture_decodes`, `patching_transfer_alone_does_not_change_pixels`,
`refusal_is_scoped_to_hdr_transfers_only`; plus
`decode::tests::only_pq_and_hlg_are_refused` (exhaustive over all 256 transfer codes) and
`decode::tests::bt2020_sdr_transfers_are_not_hdr`. No public API change; no zenpixels or
zenavif change.

**What this does NOT do.** It refuses; it does not route. A T2 cell mis-sent to the SDR
path now fails loudly instead of returning a number, which is what **G5** needs to be
satisfiable — but G5 still requires positive evidence in the run log that the PQ `--hdr`
route was taken, because a refusal proves only that the wrong route was rejected.

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
  execution to catch corpus drift. **Owner: `scripts/hdr_corpus_precheck.py`** (committed
  with this plan) — exit 0 = G0.2 PASS, 1 = at least one file would be refused by
  `decode_hdr_ref`, 2 = nothing matched (an empty run must not pass silently). Verified
  against all three paths on real data: 76/76 on the HDR corpus, **0/41 on the `.sdr.png`
  renditions** (it correctly refuses 8-bit sRGB), exit 2 on an empty glob.
- **G0.5 — primaries balance and disclosure.** The K=16 picks must carry ≥5 BT.709 and
  ≥5 P3, the picks TSV must record each pick's primaries, and the primaries × content
  cross-tab must be published **with the picks, before any encode**. The same
  `scripts/hdr_corpus_precheck.py` prints that cross-tab and emits the per-file TSV
  (`--tsv`) that the picks table is built from. Any T2 finding
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
| **TODO-0** ✅ **DONE** `e9e2ef71` | **Add the PNG-style HDR tripwire to `decode_avif`** — or call `descriptor_with_cicp` on zenavif's buffered path so the existing `HdrSourceRequiresPeak` guard fires | `zenmetrics` · `crates/zenmetrics-cli/src/decode.rs:208-220`; and/or `zenavif` · `src/decoder_managed/frame_convert.rs:65-166` | ~30 LOC + test | **a silent-corruption defect (§3.2)** — a PQ 10-bit AVIF scored without `--hdr` yields plausible, meaningless numbers with no error, and behaves *differently* for tiled vs plain files. It would poison T2 specifically. **Fix before executing.** Zero-tolerance class. — **DONE 2026-09-02, `e9e2ef71`**: route (a), refuse-loudly, matching the PNG tripwire's policy. Gated by `tests/avif_hdr_tripwire.rs` (`pq_transfer_is_refused`, `hlg_transfer_is_refused` — both fail at the parent commit) + `decode::tests::only_pq_and_hlg_are_refused`. The tiled-vs-plain divergence was **measured not to exist** — both shapes were silent pre-fix; see §3.2's appended resolution |
| **TODO-1** | Correct `~/work/zen/CLAUDE.md`: `zenav1-svt` is listed "private; **NOT cloned locally**" — it **is** cloned at `/home/lilith/work/zen/zenav1-svt` (HEAD `30cf4b3d0`), consumed by zenavif as a **path** dep (`zenavif/Cargo.toml:136-152`) | zen workspace doc | 1 line | the crate index is the map sessions navigate by |
| **TODO-2** | Teach `decode_hdr_ref` to accept **EXR** (absolute-luminance linear f32) → `HdrRef` via the existing PQ16 quantizer, with an explicit display model for `LUMINANCE=RELATIVE` sources | `zenmetrics` · `sweep/hdr.rs:97-110`; quantizer exists as `zenpixels-convert::encode_pq16` | ~120 LOC + tests | **unlocks the only banding-valid sources** (30 UPIQ EXR now, 181 si-hdr after TODO-7). Without it §3.4's ceiling can never lift. The display model must be a recorded parameter — the si-hdr study set the "registered display model" precedent |
| **TODO-3** | Extend `AVIF_HDR_KNOBS` / `SVT_HDR_KNOBS` to the SDR DOE's knob set, mapping onto the same `EncodePipeline.hdr` fields | `zenmetrics` · `sweep/hdr.rs:384`, `:478`, `:386`, `:480` | ~200 LOC + tests | blocks the HDR **knob** DOE. Must preserve refuse-unknown-knobs (G6) and must handle SVT's **sticky `apply_tune_overrides`**, which only ever *sets* — reusing one pipeline leaks tune-3 state into later cells |
| **TODO-4** | Give `score-pairs --hdr` / jobexec the same f32 feeding the CLI umbrella has, for ssim2/zensim/iwssim/butter-CPU/cvvdp-CPU | `zenmetrics` · `crates/zenmetrics-cli/src/hdr.rs:1095-1100`, `:915-921` | ~80 LOC | the **fleet** path is the u8 shell (§3.1). Until fixed, T2 must be scored via `sweep --hdr` (G5), which constrains how T2 can be fleeted |
| **TODO-5** | Wire a 16-bit SDR source type so `zensim` (`Srgb16Rgba`) and `fast-ssim2` (`ImgRef<[u16;3]>`) can be driven at full depth from the CLI | `zenmetrics` · `sweep/plan.rs:69`, `metrics/` | ~150 LOC | both libraries already resolve 16 bits (`linear-srgb/src/scalar.rs:229-238`, a 65,536-entry LUT); only the CLI funnels to u8. Prerequisite for any depth-honest **SDR** measurement |
| **TODO-6** | Re-check **aic-hdr2025** availability (README: "after QoMEX 2025", 30 Sep–2 Oct 2025 — passed); if released, ingest **validation-only** | dataset acquisition | ~1 h | the only native PQ 10-bit sources **with human JND scores on AVIF** in reach |
| **TODO-7** | Stage **si-hdr** `reference.zip` (181 EXR, 1.49 GB, on Tower with `SHA256SUMS`) | dataset staging | ~1 h + disk | 6× pool A; pairs with TODO-2 |
| **TODO-8** | Correct the stale header in `bd10.rs`: "**UNWIRED** (add `pub mod bd10;` when integration starts)" while `lib.rs:15` already declares it | `zenav1-svt` · `rust/crates/svtav1-encoder/src/bd10.rs:3-5` | 3 lines | a reader who trusts it concludes 10-bit is unimplemented and abandons the arm — the exact wrong call, given `bd10` measured −1.02 % |

**TODO-0 gates T2** — **cleared 2026-09-02 (`e9e2ef71`)**; T2 is no longer blocked on the
defect, though **G5 still requires positive route evidence** (a refusal proves only that
the wrong route was rejected, not that the right one was taken). Nothing blocks T1. TODO-3 blocks the HDR knob block; TODO-2 blocks
banding work permanently until it lands.

---

## 8. Where everything is

| artifact | path |
|---|---|
| HDR PQ-PNG references (76) | `/mnt/v/output/imazen-26-png-v2/**/*.hdr.png` |
| HDR size grid (1,140 files, 7.8 GB) | `/mnt/v/output/imazen-26-hdr-grid-2026-06-14/` ⚠ mixed zensim era |
| HDR corpus provenance | `/mnt/v/output/imazen-26-png-v2/RECONVERT-REPORT-2026-06-11.md` |
| **G0.2 + G0.5 precheck (this plan's tool)** | `zenmetrics/scripts/hdr_corpus_precheck.py` |
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

---

## 10. EXECUTION RECORD — Track T1 declared, gated and staged; T2 blocked (2026-09-02)

Appended by the execution lane. Everything below is measured on this box unless
it names another. Nothing in §§1–9 is edited; where this section corrects a
number registered above it says so explicitly.

### 10.1 Preconditions

| precondition | state | evidence |
|---|---|---|
| **TODO-0** (the silent-narrowing defect) | **CLEARED** | `e9e2ef71` on `master@origin`, plan marked DONE at `cece471e`. Route (a), refuse-loudly. |
| **B-6 drained** | **held for, then cleared** | polled on a bounded 10-min cycle writing `~/tmp/hbdexec/b6_heartbeat.txt`; encode leg on r7900x, score leg (`avifdoe-svt-b6-sf-cpu-20260902`) on the dev box |

**A consequence of TODO-0's shape that matters for G5, recorded because it is
easy to get backwards:** the fix **refuses**, it does not route. A T2 cell
mis-sent to the SDR path now fails loudly instead of returning a number — which
is what makes G5 *satisfiable* — but a refusal proves only that the wrong route
was rejected. G5 still requires **positive** evidence that the right route was
taken.

### 10.2 What was built (zero deviation from §4.2's shape)

Three new zenavif plans (`zenavif` `bcd79789`), registered in the ONE `PLANS`
table so the "unknown plan" diagnostic names them:

| plan | block | shape |
|---|---|---|
| `svt_doe_t1_bd10_ladder` | T1-a + T1-c | 7 speeds × 1 default arm |
| `svt_doe_t1_bd10_knobs` | T1-b | 15 live arms at speed 6 |
| `svt_doe_t1_bd10_transfer` | T1-d | speed 4, native corpus |

**The mechanism, and it is why these are new plans rather than a flag.**
`svt_doe_main` carries `bit_depths = [Auto, Ten]`, so at `with_max_deviations(1)`
a 10-bit cell has **already spent its one deviation on depth** and can exist
only at the default speed. That — not an oversight — is why `bd10` lives solely
at s4 today, and §11.8's "no s6 main effect" is a structural consequence of the
axis layout. The T1 plans pin `bit_depths = [Ten]`, which makes `Ten` **index
0** and therefore **zero** deviations (`cross()`'s `idxs` array contains the
bit-depth index), so the speed or the knob becomes the isolated deviation.

`svt_doe_t1_live_knob_sets` **filters** `svt_doe_knob_sets` rather than
re-listing it (unlike `svt_doe_b6_knob_sets`, which needed different levels), so
a level added to the owner arrives automatically and the two cannot drift. It
removes exactly `tn0` and `scm3` — 17 − 2 = **15**, which is §4.2's "15 live
single-deviation arms": the **default plus 14 live knobs**, not 15 non-default
arms. (There are only 14 live non-default arms; the registered 4,320 = 15 × 9 ×
32 only closes with the default included.)

### 10.3 Gates

| gate | result |
|---|---|
| **G0.1** source integrity | 32 T1 sources, sha256 in the declared `source_sha`; 16 T2 refs sha256'd in the picks TSV |
| **G0.2** T2 refs are 16-bit cICP-16 PNG | **76/76 PASS**, re-run at execution via `scripts/hdr_corpus_precheck.py` (exit 0) |
| **G0.5** primaries balance + disclosure | full-corpus cross-tab reproduced exactly (33 BT.709 / 43 P3; interiors 19/1, nature 8/39); K=16 picks carry **6 BT.709 / 10 P3**, both ≥ 5 |
| **G1** first-cell artifact | **PASS** — `7/7 cells emitted; encode-fail=0 decode-fail=0 score-fail=0`, 7 distinct `.avif` blobs persisted and scored (ssim2 + zensim) |
| **G3** 10-bit decode-verify | **PASS** — see below |
| **G2** control identity | **satisfiable for all four blocks — verified before launch, not assumed** (below) |
| **G4** producer seam | declared: T1-a reported in two blocks split at speed 6/7 (preset 8 → 9); no BD-rate-vs-speed slope fitted across it |
| **G6** no silent knob acceptance | unchanged — `hdr.rs:406` still refuses unwired HDR knobs; T1 is the SDR lane and spells knobs through `svt_knobs`, which cannot express an unknown key |
| **G7** instrument statement | §3.4 and H-BD-4 reproduced in every report carrying a number from this arm |

**G2 — every T1 cell has a matched 8-bit control, checked at declare time.**
A BD-rate needs a `(image, speed, q)` partner at depth 8, and T1 declares no
control cells of its own by design (they exist already). Whether those partners
actually cover T1's speeds was **never checked by the plan**, and a gap would
only have surfaced at analysis — after the compute was spent. Measured from the
declared cell lists:

| T1 block | control | coverage |
|---|---|---|
| T1-a, T1-c | **A0R** (`avifdoe-svt-a0r-20260901`) | **all 7 speeds × 29 q × 32** = 6,496 cells; 928 per speed. The 9-point knob ladder is a **strict subset** of the 29-point grid, so every T1-a/c cell differences with **no interpolation and no anchor error** |
| T1-b | A0R, s6 default arm | the same s6 leg at the 9 shared q points |
| T1-d | **AG** (`avifdoe-svt-ag-20260901`), stratum `s4-svt-420` | **96 cells = 32 images × q{15,45,90} on the NATIVE corpus** — an exact shape match to T1-d |

So the arm is analyzable the moment it drains. Note T1-d's control is AG's
**native** leg, not A0R's crop leg: the two corpora share all 32 filenames and
13 carry genuinely different pixels, so differencing T1-d against A0R would
silently compare native encodes to crop references on 13/32 of the corpus —
the hazard `avifdoe_score_gapfill.sh`'s header already documents, and the
reason t1d carries its own `--refs-prefix` in the runbook.

**G3 in full — three independent reads, not one.** `avif_depth_verify`
(`crates/zenmetrics-cli/examples/avif_depth_verify.rs`, `c863fd30`) reads the
**av1C box** (`zenavif-parse`), the **AV1 sequence header**, and the **decoder's
`ImageInfo`** (`ManagedAvifDecoder::decode_full` — the same reader the PQ/HLG
tripwire uses). Disagreement is a FAIL on the blob's own evidence, with no
`--expect-depth` needed.

- Verified on **207 conformance vectors** (link-u 150 + libavif 57): zero depth
  mismatches, zero cross-read disagreements; all 17 PQ vectors read 10/10/10;
  profile-2 12bpc reads 12 with `twelve_bit = 1`.
- **Negative controls, each watched to fail** — a gate that has never failed is
  not known to work: 8-bit at `--expect-depth 10` → exit 1; 12-bit → exit 1;
  an **av1C patched to claim 8 while the sequence header still says 10** (the
  H-BD-3 shape) → exit 1 `reads DISAGREE`, *without* `--expect-depth`, which
  also proves the three reads are genuinely independent; `--control` on a
  byte-identical copy → exit 1; empty directory → exit 2.
- **On this arm's own first cell: 7/7 PASS**, all three reads agreeing at depth
  10, `high_bitdepth = 1`, `twelve_bit = 0`, profile 0, 4:2:0.
- **G3's byte-identity half: 0/7** bd10 blobs are byte-identical to the 8-bit
  control encoded from the same source at the same q (`svt_doe_transfer`,
  `max-deviations 0`), and that control itself reads **depth 8** on all three
  reads. Stage A measured 0/288; a regression to 100 % would be the
  `tn0`/`scm3` inertness failure.

### 10.4 Declaration

`avifdoe_declare.sh --track-t1`, through the canonical path:

| run | plan | q | corpus | cells |
|---|---|---|---|--:|
| `avifdoe-svt-t1ac-20260902` | `svt_doe_t1_bd10_ladder` | 9-point | budget | 2,016 |
| `avifdoe-svt-t1b-20260902` | `svt_doe_t1_bd10_knobs` | 9-point | budget | 4,320 |
| `avifdoe-svt-t1d-20260902` | `svt_doe_t1_bd10_transfer` | 3-point | **native** | 96 |

- **6,432 declared job ids — matching §4.2's block sum exactly.**
- `duplicates_merged 0 / invalid_skipped 0 / q_coarsenings 0 / dropped_axes 0 /
  over_budget false` on all three. Zero merges is **structurally correct** here
  (one knob axis, one pinned depth), not the red flag it is on a pairwise plan.
- **DETERMINISTIC**: two independent rounds gave sha-equal `_cells.jsonl`
  (`55715d26` / `aec2c512` / `40f57487`) and `plan.json` (`95e67de1` /
  `12ea6904` / `50d1cb5b`).
- Every declared cell carries the `bd10` token: 7 + 15 + 1 distinct cell shapes,
  0 missing.

**⚠ A COUNT CORRECTION TO §4.2, measured not assumed.** The distinct-cell count
is **6,087**, not the 6,144 a naive subtraction gives. Two overlaps:

1. **288** — the s6 knob-default stratum, shared by the ladder and knob runs
   (this is T1-c, which §4.2 counts three times: as its own block, inside T1-a's
   6-speed ladder, and as T1-b's default arm).
2. **57** — T1-d cells that are **also t1ac cells**. 19 of the 32 corpus images
   are sub-budget **passthroughs** whose native and 1024² files are
   byte-identical, so they share a `source_sha` and therefore a `CellId`. Stage A
   measured the same 19/13 split and *relies* on it (A0-native ≡ A0R on
   3,857/3,857 shared cells). 19 images × 3 probe-q = 57.

> **REGISTERED ANALYSIS RESTRICTION, and it is not cosmetic: T1-d's cross-size
> transfer gate has n = 13 IMAGES, NOT 32.** On the 19 passthroughs there is no
> size transfer to measure — the two sizes are the same pixels and the same
> encode. The 96 declared cells remain the right declaration (the 57 are
> already-done work and cost nothing), but any "does −1.02 % survive native
> size" statement must be made over the 13 genuinely-cropped images and must
> report that n. Q4 in §5.1 is answerable only at that n.

### 10.5 ⛔ TRACK T2 IS SPLIT: T2-b is executable, **T2-a is BLOCKED**

This is a wiring gap §2.6 did not record, found by reading the source at
execution time. §2.6 identified the HDR **knob** gap (TODO-3); this is
narrower and harder: **the `sweep --hdr` CLI cannot reach the svt backend at
all, even at its two wired knobs.**

- `validate_hdr_sweep` (`sweep/hdr.rs:134-141`) admits **only**
  `CodecKind::Zenjxl | CodecKind::Zenavif`; every other codec "errors loudly at
  sweep start". `HdrCodec::from_codec_kind` (`:256-263`) maps the same two.
- `zenav1-svt` is **not a `CodecKind` at all** — the `--codec` enum is
  zenpng/zenjpeg/zenwebp/zenavif/zenjxl/zengif/zentiff.
- `HdrCodec::Zenav1Svt` is reachable **only** through `HdrCodec::from_name`,
  whose sole caller is `jobexec.rs:1906` — the fleet path, whose ssim2/zensim
  scoring is the **u8 shell** that §3.1 and **G5** forbid for T2.
- In SDR the svt backend is reached as a `backend` **knob** on `zenavif`; the
  HDR path's `AVIF_HDR_KNOBS` is `["lossless", "speed"]` and refuses unknown
  keys by design (G6), so that door is shut too.

**So T2-a as registered — svt HDR encodes with f32 scoring — is unsatisfiable
today.** The two requirements are individually reachable and jointly are not:
`sweep --hdr` gives f32 but not svt; jobexec gives svt but not f32.

**This is exactly what TODO-4 is for**, and §4's "T2's baseline executes after
TODO-0 … and TODO-4" is therefore literally correct — TODO-4 is a T2-a
prerequisite, not merely an alternative to it.

**✅ TODO-4 LANDED MID-EXECUTION — `7051921a`, and it clears T2-a's blocker.**
A concurrent capability lane (`claude-bitdepth`) held a path-scoped marker
claiming `crates/zenmetrics-cli/src/hdr.rs` "fleet f32", and shipped it while
this section was being written: *"the fleet/score-pairs/batch HDR routes stop
u8-shelling ssim2, zensim, iwssim and CPU butteraugli."* `score-pairs --hdr`,
`batch --hdr` and jobexec's ScoreFile HDR arm now take the umbrella's validated
`hdr_feeding` recipe through a new `hdr::faithful_hdr_rows`; the u8 shell
survives only as a genuine fallback for a `(metric, backend)` the umbrella
cannot express. That lane measured the stake directly on a PQ ramp against its
own 8-bit quantization: **99.75 % of f32 samples differ where 5.82 % of u8
bytes do — the shell was collapsing 94.17 % of the difference**, which is
precisely the artifact a bit-depth arm exists to see.

**So the paragraph above is superseded: T2-a is no longer blocked on wiring.**
The fleet can now encode `zenav1-svt` HDR (jobexec, `HdrCodec::from_name`) *and*
score it faithfully, which is the combination that did not exist an hour ago.
Two consequences carry forward, both from that commit's own warnings:

- **⚠ It is a SCORING-ERA break for HDR cells.** Every stored `--hdr`
  ssim2/zensim/iwssim/butteraugli-CPU number from score-pairs, batch or jobexec
  was computed on the u8 shell and **must not be joined across `7051921a`**.
  T2, whenever it runs, is post-break by construction.
- **⚠ `--hdr-transfer` is now INERT** on these metrics — the faithful route
  takes absolute nits and PU21-encodes in-kernel, so there is no transfer shell
  to select. A T2 invocation that passes it is not selecting anything.

**T2 still did not run, and the remaining reasons are corpus and evidence, not
wiring.** Stated precisely so the next lane starts from facts:

1. **The 16 native references are not in the corpus store.** `codec-corpus/
   imazen-26-variants/hdr-grid-15scale@2026-08-23/` holds 1,140 `.hdr.png`
   (76 origins × 15 scales) but its ladder **tops out at 2304×3072** — there is
   no native rung. §4.3's pool B is the **native** 3000×4000 set, so T2 needs
   those 16 files staged (~0.6 GB) under their own prefix, or an explicit,
   registered decision to run at a ladder size instead. Do not silently
   substitute a 2304 rung for native: §1.2 already flags the derived size grid
   as **mixed-zensim-era**, and the plan's own T1-d precedent is that reduced
   size can flip a sign.
2. **The worker image must be rebuilt on ≥ `7051921a`.** The image this lane
   built and verified (`exec-avifhbd-t1-32e68a8f`) predates the fix, so it would
   score T2 through the u8 shell — silently, since the shell is still the
   documented fallback. This is the same class of trap as the stale-plan image,
   and it is worse here because it fails as a *plausible number*, not a panic.
3. **G5 needs positive evidence against the NEW route**, not the old one. The
   assertion to build is "this cell's ssim2/zensim came from
   `faithful_hdr_rows`", and the honest way to get it is a decode/route marker
   in the run log — not the absence of a refusal, and no longer "the run used
   `sweep --hdr`", since the fleet path is now equally faithful.

`hdrgrid_cells.py` is the right declaration template and already carries a
`zenav1-svt` arm at `{"preset": 6}`; T2-a is that grammar at 7 presets × 29 q ×
16 refs, T2-b the `zenavif` arm at 3 speeds × 9 q × 16.

**T2-b was NOT smoke-tested either, and the reason is the same lane.**
`sweep/hdr.rs` calls into `src/hdr.rs` (`decode_to_nits`, `HdrTransfer`,
`measured_display_peak_nits`), so any `sweep --hdr` run made from this working
tree right now would exercise another lane's **in-flight, uncommitted** edits to
the scoring path rather than `master`. A G5 route assertion produced that way
would be evidence about a tree that has never existed on `master` — worse than
no measurement, because it looks like one. **G5 is therefore UNSATISFIED for
both T2 blocks, by choice, and no T2 cell was declared or run.**

- **T2-a (3,248 cells): wiring UNBLOCKED by `7051921a`; still not declared or
  run** — pending the three items above (native refs staged, image rebuilt past
  the fix, G5 evidence designed against `faithful_hdr_rows`). **NOT MEASURED —
  never a null, never a zero.**
- **T2-b (432 cells): reachable two ways now** — via `sweep --hdr` (which
  `validate_hdr_sweep` admits for `zenavif`) or via the fleet, post-`7051921a`.
  Still not run, for reasons 1 and 2 above. Its corpus gates (G0.2, G0.4, G0.5)
  are **complete and committed**, so it is the cheaper of the two to start.
  Note §4.3's own caveat stands: T2-b differs from T2-a in **three** ways at
  once (backend, chroma, matrix), so it is a **contrast, not a controlled
  comparison** — and with T2-a blocked it currently has nothing to contrast
  *against*. Its standalone value is a zenavif-arm HDR RD baseline, which is
  still the first one in the workspace.

### 10.6 Two findings for the record

1. **`zenavif-parse` `read_iprp` drops the primary item's property set on
   two-`ipma` files.** `associations = read_ipma(&mut b)?` is a plain assignment
   inside the box loop, so a second `ipma` overwrites the first. Proven by an
   independent ISOBMFF walk of `plum-blossom-large.profile0.10bpc.yuv420
   .alpha-full.avif`: `pitm` = 1 **is** associated with `av1C` (ipco index 4),
   yet `av1_config()`, `spatial_extents()`, `color_info()` and
   `pixel_aspect_ratio()` all return `None`. It separates perfectly on alpha
   across the conformance corpus (64/64 FAILs have `alpha` in the name; 0/86
   PASSes do). **Inert for this arm** — its blobs are opaque, single-`ipma` and
   non-grid — so it is recorded, not fixed mid-arm. The distinct grid case is
   spec-correct (tiles carry `av1C`, not the derivation item).
2. **H-BD-3's stated MECHANISM looks stale at current `zenav1-svt` HEAD.**
   `svtav1/src/avif.rs:218` and `lib.rs:135` both read
   `{ self.bit_depth = depth; self }` — no coercion — with a doc comment saying
   it "deliberately does NOT coerce … It used to". §2.1 cites dossier §605 for
   the coercing behaviour; that line is worth re-reading. **Read from source,
   NOT measured** — the coercion was never exercised end to end here. **This
   does not retire G3**: the zenavif/zenrav1e arm reaches depth by a different
   path, whether the new refusal fires end-to-end is unverified, and G3's
   byte-identity half is untouched either way.
