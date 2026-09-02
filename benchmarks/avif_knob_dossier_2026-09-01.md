# AVIF encoder-knob dossier — `zenav1-svt` (SVT-AV1) and `zenav1-aom` (libaom)

**Date:** 2026-09-01 · **Lane:** research + source-verification only (no fleet
actions, no sweep declarations, no encoder changes).
**Purpose:** the knob reference a design-of-experiments lane plans AVIF
still-image sweeps against, feeding predictive models that pick knob combos from
(target quality, zenanalyze features) to optimize quality / size / speed.

**How to read this.** Every load-bearing number was read from the pinned
upstream source or the port source and carries a link. Where the docs and the
port source disagree, **the port source wins** and the contradiction is called
out inline. §9 (hazards) and §8 (shortlists) are the parts the DOE lane needs
first; §12 lists what is still UNVERIFIED.

---

## 0. TL;DR for the DoE lane

1. **Pins: SVT-AV1 `v4.2.0` and libaom `v3.14.1`.** The sibling
   `svtav1-v4.1.0/` directory is a decoy — see **H-1**.
2. **`tune` is a super-factor on both backends.** libaom's `tune=iq` forces 10
   other knobs; `zenav1-svt`'s `tune=3` forces 9. Make it the **outermost**
   axis or your factorial is aliased (**H-4**).
3. **The svt-rs speed dial has 10 levels but only 7 encodes** — speeds 7-10 are
   all preset 9, and presets 2/5/8 are unreachable. Sweep `preset` directly
   (**H-2**).
4. **aom-rs `encode_ms` is not an encoder time** — the arm runs the C oracle
   *and* the port and byte-compares them (**H-5**). Do not fit a speed model on
   it as-is.
5. **The live sweep drives `{backend, speed} × q` and nothing else** (§6), and
   hard-errors on unknown knobs — so every knob below is greenfield, and each
   one must be threaded explicitly.
6. **Biggest single exposure gap: `tune=iq` on `zenav1-aom`** — fully ported and
   byte-gated, but absent from the harness knob struct (§5.2, §10).

---

## 1. Version pins (evidence, not memory)

| Port | Upstream tracked | Pin evidence |
|---|---|---|
| **`zenav1-svt`** | **SVT-AV1 v4.2.0** | `/home/lilith/work/zen/zenav1-svt/PORTING.md:10` — "`reference/svt-av1/` the SVT-AV1 **v4.2.0** C fork (git SUBMODULE) — READ-ONLY reference + differential oracle". Submodule gitlink `3115c0c1b23e860dfd75c94f6740e0298182dd13`, url `https://github.com/imazen/zenav1-svt-c.git`, branch `imazen-parity` (`/home/lilith/work/zen/zenav1-svt/.gitmodules:1-4`). README:244-250 defines that branch as **v4.2.0 plus one patch**: an OFF-by-default `SVT_HDR_MODE` CMake option, default build = "mainline v4.2.0". |
| **`zenav1-aom`** | **libaom v3.14.1** | `/home/lilith/work/zen/zenav1-aom/README.md:6` — "validated against a pinned C libaom **v3.14.1** oracle (`03087864`)". Submodule `upstream/` gitlink `03087864cf4bea6abb0d28f95cf7843511413d8f`; `git -C upstream describe --tags` → **`v3.14.1`**; `upstream/CHANGELOG:1` → "2026-05-22 v3.14.1". |

**⚠ The v4.1.0 assumption is WRONG.** The sibling directory
`/home/lilith/work/zen/svtav1-v4.1.0` is *not* what `zenav1-svt` tracks. This
matters concretely — see hazard **H-1**: four CLI knobs and one tune mode exist
at v4.2.0 that do **not** exist at v4.1.0.

**Availability asymmetry, worth knowing before you plan reads.** The libaom
oracle is **checked out on disk** at the pin
(`/home/lilith/work/zen/zenav1-aom/upstream/`), so libaom claims below are read
from the exact pinned bytes. The SVT submodule is **not** checked out
(`reference/svt-av1/` is empty), so SVT upstream claims are read from the
**v4.2.0 tag on GitLab**, not from local C headers. Anything that would need
the C header specifically is marked UNVERIFIED in §12.

Permalink conventions used throughout:
[SVT-AV1 @ v4.2.0](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.2.0/) ·
[libaom @ 03087864 (googlesource)](https://aomedia.googlesource.com/aom/+/03087864cf4bea6abb0d28f95cf7843511413d8f/) ·
[libaom @ 03087864 (imazen mirror, clickable on GitHub)](https://github.com/imazen/libaom-mirror/tree/03087864cf4bea6abb0d28f95cf7843511413d8f) ·
[libavif v1.4.2](https://github.com/AOMediaCodec/libavif/tree/v1.4.2)


---

## 2. SVT-AV1 v4.2.0 — upstream knob table (still-image relevant)

Source for every row unless noted:
[`Docs/Parameters.md` @ v4.2.0](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.2.0/Docs/Parameters.md)
(fetched raw; ranges/defaults transcribed literally, not summarized).

| Config field | CLI | Range | Default | Still-image semantics (one line) | Interacts with | Version note |
|---|---|---|---|---|---|---|
| `EncoderMode` | `--preset` | `[-1-13]` | `8` | The speed/efficiency dial; sets sb-size (128 at ≤M6, 64 at ≥M7), filter-intra, wedge/diff/dist-weighted pred, global motion, SG/Wiener restoration. | everything | stable |
| `Avif` | `--avif` | `[0-1]` | `0` | **Still-picture coding optimizations** — improved coding efficiency + reduced memory. The single most AVIF-specific switch SVT has. | single-frame only | added ≤ v3.0.0; **serious bug until v4.0.0** ([issue 2245](https://gitlab.com/AOMediaCodec/SVT-AV1/-/issues/2245)) |
| `Tune` | `--tune` | `[0-5]` | `1` | `0`=VQ (video+still), `1`=PSNR (video+still), `2`=SSIM (video+still), **`3`=IQ (still image ONLY)**, `4`=MS-SSIM (video+still), `5`=VMAF (video only). | dominates RD; see H-4 | `[0-2]` at v3.0.0 → `[0-4]` at v4.0.0 (IQ, MS-SSIM) → **`[0-5]` at v4.2.0** (VMAF) |
| `CRF` | `--crf` | `[1-70]` | `35` | Quality dial, 0.25 increments. Docs: "similar to `--rc 0 --aq-mode 2 --qp x`". | **implies aq-mode 2** — H-3 | stable |
| `CQP` | `--cqp` | `[1-70]` | `35` | Quality dial, 0.25 increments. Docs: "similar to `--rc 0 --aq-mode 0 --qp x`". | **implies aq-mode 0** — H-3 | **NEW at v4.2.0** |
| `QP` | `--qp` | `[1-63]` | `35` | Initial QP level. | `--rc`, `--aq-mode` | stable |
| `RateControlMode` | `--rc` | `[0-2]` | `0` | `0`: CRF or CQP (CQP "if `--aq-mode` is 0"), `1`: VBR, `2`: CBR. Stills use `0`. | `--crf`/`--cqp`/`--aq-mode` | stable |
| `AdaptiveQuantization` | `--aq-mode` | `[0-2]` | `2` | `0`=off, `1`=variance via AV1 segments, `2`=deltaq pred efficiency. | the CRF/CQP alias — H-3 | stable |
| `EnableVarianceBoost` | `--enable-variance-boost` | `[0-1]` | `0` | Per-64×64 deltaq that *lowers* qindex on low-variance superblocks (clouds, skin, fog) — the "even visual energy" lever. | `--aq-mode` (both are deltaq) | added v2.1.0; improved v3.1.0 |
| `VarianceBoostStrength` | `--variance-boost-strength` | `[1-4]` | `2` | Boost curve: 1 mild / 2 gentle / 3 medium / 4 aggressive. Upstream: **"Strength 3 is best for still images"**. | requires `--enable-variance-boost 1` | v2.1.0+ |
| `VarianceOctile` | `--variance-octile` | `[1-8]` | `5` | How much of a superblock must be low-contrast to boost it. Upstream: "Recommended values are between 4 and 7." | requires `--enable-variance-boost 1` | v2.1.0+ |
| `EnableQM` | `--enable-qm` | `[0-1]` | `0` | Quantization matrices on/off. **OFF by default in SVT** (contrast: libaom turns them ON for images). | `--qm-min`/`--qm-max` | stable |
| `MinQmLevel` | `--qm-min` | `[0-15]` | `8` | Min QM flatness. | `--enable-qm` | stable |
| `MaxQmLevel` | `--qm-max` | `[0-15]` | `15` | Max QM flatness. | `--enable-qm` | stable |
| `MinChromaQmLevel` | `--chroma-qm-min` | `[0-15]` | `8` | Min chroma QM flatness. | `--enable-qm` | added v3.1.0 (from SVT-AV1-PSY) |
| `MaxChromaQmLevel` | `--chroma-qm-max` | `[0-15]` | `15` | Max chroma QM flatness. | `--enable-qm` | added v3.1.0 |
| `Sharpness` | `--sharpness` | `[-7-7]` | `0` | Bias toward decreased/increased sharpness (deblocking + RC). | DLF | added between v3.0.0 and v4.0.0 |
| `AcBias` | `--ac-bias` | `[0.0-8.0]` | `0.0` | RD bias toward high-frequency error — texture/grain retention. | `--tune` | added between v3.0.0 and v4.0.0 |
| `LuminanceQpBias` | `--luminance-qp-bias` | `[0-100]` | `0` | Frame QP adjust by average luma. | `--aq-mode` | added between v3.0.0 and v4.0.0 |
| `ScreenContentMode` | `--scm` | `[0-3]` | `2` | `0` none, `1` Block Copy + Palette forced, `2` content adaptive, `3` content adaptive (anti-alias aware). | gates IBC + palette | `3` added v4.0.0 |
| `EnableIntraBC` | `--enable-intrabc` | `[0-1]` | `1` | Intra Block Copy (preset-based when 1). | `--scm` | **NEW at v4.2.0** |
| `CDEFLevel` | `--enable-cdef` | `[0-1]` | `1` | Constrained Directional Enhancement Filter. | preset | stable |
| `EnableRestoration` | `--enable-restoration` | `[0-1]` | `1` | Loop restoration (Wiener / SG). Preset-gated: SG off ≥M4, Wiener off ≥M9. | preset | stable |
| `LoopFilterEnable` | `--enable-dlf` | `[0-2]` | `1` | Deblocking; `2` = slower, more accurate. | `--sharpness` | `2` documented v4.1 |
| `MaxTxSize` | `--max-tx-size` | `[32,64]` | `64` | Cap transform size. | preset | added between v3.0.0 and v4.0.0 |
| `FastDecode` | `--fast-decode` | `[0,2]` | `0` | Decode-targeted optimization levels; costs quality, buys decode speed. | preset, `--hierarchical-levels` | range 0-1 → 0-2 at v2.3.0 |
| `TileRow` / `TileCol` | `--tile-rows` / `--tile-columns` | `[0-6]` / `[0-4]` | `0` / `0` | log2 tile counts; enc+dec parallelism, costs efficiency and can show seams. | `--lp`, threads | stable |
| `LevelOfParallelism` | `--lp` | `[0, 6]` | `0` | Threads + picture buffers; `0` = auto from core count. | tiles | stable |
| `FilmGrain` | `--film-grain` | `[0-50]` | `0` | Grain synthesis denoise level. | `--film-grain-denoise`, `--adaptive-film-grain` | stable |
| `FilmGrainDenoise` | `--film-grain-denoise` | `[0-1]` | `0` | Denoise when grain is on. | `--film-grain` | stable |
| `AdaptiveFilmGrain` | `--adaptive-film-grain` | `[0-1]` | `1` | Vary grain blocksize by resolution. | `--film-grain` | added between v3.0.0 and v4.0.0 |
| `EncoderBitDepth` | `--input-depth` | `[8, 10]` | `8` | 8 or 10 bit. | `--profile`, `--hbd-mds` | stable |
| `HBDMDS` | `--hbd-mds` | `[-1-2]` | `-1` | 10-bit mode-decision: -1 preset default, 0 full 8b, 1 full 10b, 2 hybrid. **10-bit MD only works with 10-bit input.** | `--input-depth` | **NEW at v4.2.0** |
| `EncoderColorFormat` | `--color-format` | `[0-3]` | `1` | **Docs state: "only yuv420 is supported at this time".** | `--profile` | stable |
| `KeyFrameChromaQIndexOffset` | `--key-frame-chroma-qindex-offset` | `[-64-63]` | `0` | Chroma qindex offset on keyframes — for a still, *the* frame. | chroma q | stable |
| `ChromaU/V DC/AC QindexOffset` | `--chroma-{u,v}-{dc,ac}-qindex-offset` | `[-64-63]` | `0` | Per-plane, per-DC/AC chroma qindex offsets. | chroma q | stable |
| `ColorPrimaries` / `TransferCharacteristics` / `MatrixCoefficients` / `ColorRange` | `--color-primaries` / `--transfer-characteristics` / `--matrix-coefficients` / `--color-range` | `[0-12,22]` / `[0-22]` / `[0-14]` / `[0-1]` | `2` / `2` / `2` / `0` | CICP signalling only (no pixel effect). | container CICP | stable |

**Preset feature table** (what a preset actually changes) is in
[`Docs/CommonQuestions.md` § "What Presets Do"](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.2.0/Docs/CommonQuestions.md).
Still-relevant rows: **sb size 128 for presets 0–6, 64 for 7–10**; Filter-intra
ON ≤M6 / OFF ≥M7; wedge + diff-weighted + dist-weighted pred ON ≤M2 only;
global motion ON ≤M4; SG restoration ON ≤M3; Wiener restoration ON ≤M8;
**IBC and Palette listed ON at every preset but gated by screen-content
detection**. There is **no `--sb-size` / `--enable-palette` / `--lossless` CLI
knob at v4.2.0** — sb-size is preset-derived, palette rides `--scm`, and
`lossless` exists as an **API config field but not a documented CLI parameter**
(libavif sets `svt_config->lossless` directly — see §7.3).

---

## 3. libaom v3.14.1 — upstream knob table (still-image relevant)

Every row below was read from the **pinned source checked out on disk**
(`/home/lilith/work/zen/zenav1-aom/upstream/`, gitlink `03087864`). Links point
at the same bytes on the [imazen mirror](https://github.com/imazen/libaom-mirror/tree/03087864cf4bea6abb0d28f95cf7843511413d8f)
(clickable) — the [googlesource original](https://aomedia.googlesource.com/aom/+/03087864cf4bea6abb0d28f95cf7843511413d8f/) serves the same objects.

Defaults come from `default_extra_cfg` ([`av1/av1_cx_iface.c:253-407`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L253-L407));
ranges from the `RANGE_CHECK` block ([`:700-960`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L700-L960)).
**`AOM_USAGE_ALL_INTRA` (the AVIF still mode) changes four of these defaults** — see the split column.

| CLI / control | Range (at pin) | Default | ALL-INTRA default | Still-image semantics | Source |
|---|---|---|---|---|---|
| `--cpu-used` / `AOME_SET_CPUUSED` | **`0..9`** (non-realtime); `0..12` only for `AOM_USAGE_REALTIME` | `0` | same | The speed dial. `0..9` is exactly the range the sweep uses. | [`:775-776`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L775-L776) |
| `--cq-level` / `AOME_SET_CQ_LEVEL` | `0..63` | `10` | same | Quality dial in `AOM_Q` / `AOM_CQ` end-usage. **This is the AVIF quality knob.** | [`:802`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L802) |
| `--tune` / `AOME_SET_TUNING` | `AOM_TUNE_PSNR(0) .. AOM_TUNE_SSIMULACRA2(11)` | `AOM_TUNE_PSNR` | same | **`IQ`=10 and `SSIMULACRA2`=11 are the image tunes** — "meant for image encoding … Using these tuning modes for videos isn't recommended". Each silently forces 9–10 other knobs — **hazard H-4**. | [`aomcx.h:1740-1804`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/aom/aomcx.h#L1740-L1804), [`av1_cx_iface.c:892`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L892) |
| `--deltaq-mode` / `AV1E_SET_DELTAQ_MODE` | `0..6` | `DELTA_Q_OBJECTIVE(1)` | same | `0` none, `1` objective, `2` perceptual, `3` perceptual-AI (all-intra), `4` user-rating, `5` HDR, **`6` Variance Boost** (libaom's analogue of SVT's `--enable-variance-boost`). | [`encoder.h:153-162`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/encoder/encoder.h#L153-L162), [`:718`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L718) |
| `--enable-qm` / `AV1E_SET_ENABLE_QM` | `0..1` | `0` | `0` | Quantization matrices. | [`:290`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L290) |
| `--qm-min` / `--qm-max` | `0..15` (`NUM_QM_LEVELS=16`) | `5` / `9` (`DEFAULT_QM_FIRST/LAST`) | **`4` / `10`** (`DEFAULT_QM_*_ALLINTRA`) | QM flatness window. `tune=iq` forces **`2` / `10`** (`QM_FIRST/LAST_IQ_SSIMULACRA2`). | [`quant_common.h:37-42`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/common/quant_common.h#L37-L42) |
| `--sharpness` / `AOME_SET_SHARPNESS` | **`0..7`** (unsigned — *not* SVT's `-7..7`) | `0` | same | Bias toward block sharpness in coefficient RDO. `tune=iq/ssimulacra2` forces **7**. | [`:798`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L798) |
| `--enable-adaptive-sharpness` | `0..1` | `0` | same | Adjust loop-filter levels by QP. **`tune=iq` forces 1; `tune=ssimulacra2` does not** — the only knob that separates the two image tunes. | [`:799`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L799), [`:1971-1976`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L1971-L1976) |
| `--enable-cdef` / `AV1E_SET_ENABLE_CDEF` | **`0..3`** (`NONE/ALL/REFERENCE/ADAPTIVE`) | `1` (ALL) | **`0` (OFF)** — "CDEF has been found to blur images" | `3` = ADAPTIVE: off at qindex 0-32, reduced 33-220, full 221-255. `tune=iq` forces **3**. | [`pickcdef.h:22-29`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/encoder/pickcdef.h#L22-L29), [`:946`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L946), [`:3065-3078`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L3065-L3078) |
| `--enable-chroma-deltaq` / `AV1E_SET_ENABLE_CHROMA_DELTAQ` | `0..1` | `0` | same | Per-plane chroma deltaq. `tune=iq` forces 1. **Hard-errors with `--lossless=1`.** | [`:905-906`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L905-L906) |
| `--screen-detection-mode` | **`1..2`** (no 0) | `1` (STANDARD) | **`2`** (ANTIALIASING_AWARE) | Screen-content detection; gates palette + IntraBC. `tune=iq` forces 2. **Cost driver — hazard H-6.** | [`aomcx.h:1734-1736`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/aom/aomcx.h#L1734-L1736), [`:959`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L959) |
| `--dist-metric` | `AOM_DIST_METRIC_PSNR .. AOM_DIST_METRIC_QM_PSNR` | `PSNR` | same | In-block RD distortion metric (lower level than `--tune`, which acts across blocks). `tune=iq` forces `QM_PSNR`. | [`:894`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L894), [`aomcx.h:1806+`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/aom/aomcx.h#L1806) |
| `--enable-palette` | `0..1` | `1` | same | Palette prediction (screen content). | [`:359`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L359) |
| `--enable-intrabc` | `0..1` | `1` | same | Intra Block Copy. **Dominant cost on screen-detected frames** in the port. | [`:360`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L360) |
| `--enable-restoration` | `0..1` | `1` | same | Loop restoration. libavif turns it **off for alpha**. | [`:286`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L286) |
| `--sb-size` / `AV1E_SET_SUPERBLOCK_SIZE` | `64X64 .. DYNAMIC` | `DYNAMIC` | same | Superblock size. Unlike SVT, this **is** a real libaom knob. | [`:778-779`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L778-L779) |
| `--tile-columns` / `--tile-rows` | `0..6` each (log2) | `0` / `0` | same | Parallelism vs efficiency; can show seams. `--auto-tiles` (`0..1`, default `0`) derives them. | [`:787-788`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L787-L788) |
| `--row-mt` | `0..1` | `1` | same | Row-level multithreading. Affects **speed**, and can affect determinism of timing but not bytes. | [`:784`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L784) |
| `--lossless` / `AV1E_SET_LOSSLESS` | `0..1` | `0` | same | Coded-lossless. Conflicts with `enable-chroma-deltaq`. | [`:716`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L716) |
| `--arnr-maxframes` / `--arnr-strength` | `0..15` / `0..6` | `7` / `5` | same | Alt-ref temporal filtering. **INERT for a 1-frame still** (no alt-ref to filter) — see hazard H-7. | [`:800-801`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L800-L801) |
| `--enable-dnl-denoising` | `0..1` | `1` | same | Denoising in the noise-level path. | [`:365`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L365) |
| `AV1E_SET_SKIP_POSTPROC_FILTERING` | `0..1` | `0` | — | libavif sets **1** in ALL_INTRA (skips post-processing filters when the recon isn't needed). | [`libavif codec_aom.c:1058-1062`](https://github.com/AOMediaCodec/libavif/blob/v1.4.2/src/codec_aom.c#L1064-L1068) |

**Build-flag-gated tunes (NOT usable at the pin unless libaom was configured for them).**
`AOM_TUNE_VMAF_*` (4–7) require `-DCONFIG_TUNE_VMAF=1`; `AOM_TUNE_BUTTERAUGLI` (8) requires
`-DCONFIG_TUNE_BUTTERAUGLI=1` — the encoder emits an explicit "try to set
-DCONFIG_TUNE_… =1 at the time CMake is run" error otherwise
([`:875-890`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L875-L890)).
**`AOM_TUNE_IQ` and `AOM_TUNE_SSIMULACRA2` are NOT build-gated** — they are always available at v3.14.1
(guard macros `AOM_HAVE_TUNE_IQ` / `AOM_HAVE_TUNE_SSIMULACRA2` are defined unconditionally).
Neither VMAF nor Butteraugli tunes are reachable from the pure-Rust ports (§4–§5), so treat them as out of scope.

---

## 4. Port reality — `zenav1-svt`

**Three config tiers exist; only one is real.**

| Tier | Type | Status |
|---|---|---|
| DEAD SCAFFOLD | `svtav1::EncoderConfig` / `Encoder` | `rust/svtav1/src/lib.rs:67`, `:195`. Self-documented at `lib.rs:37-40`: "`Encoder::send_frame` **discards its input** and `Encoder::receive_packet` always returns `NotReady`. Do not build on them." Its `Yuv422`/`Yuv444` variants and 12-bit acceptance exist **only here and encode nothing**. |
| PRODUCT WRAPPER | `svtav1::avif::AvifEncoder` | `rust/svtav1/src/avif.rs:75`. Real, but only **4 encode-affecting axes** reach the encoder. |
| **THE REAL API** | `svtav1_encoder::pipeline::EncodePipeline` | `rust/crates/svtav1-encoder/src/pipeline.rs:25`. ~22 live knobs. **A DoE lane must drive this directly.** |

### 4.1 Exposed knobs (`EncodePipeline` + `EncodePipeline.hdr`)

Constructor `pipeline.rs:336`: `new(width, height, preset, rc_config, hierarchical_levels, intra_period)`. No `Default`. Fields `hdr`, `speed_config`, `rc_config`, `chroma_sample_position` are `pub` and assigned directly.

| port field | range | default | still-image semantics | defined / consumed |
|---|---|---|---|---|
| `preset` | **0..=13 silently CLAMPED**, effective **0..=9 for stills** | ctor arg | THE speed axis; ~40 consumption sites | `speed_config.rs:34-38` / `speed_config.rs:110` |
| `rc_config.qp` | **0..=63** (CLI domain, not qindex) | `35` | THE quantizer; also feeds SB-size rule + tune-IQ `max_tx_size` | `rate_control.rs:43`, dflt `:111` / `pipeline.rs:1490` |
| `rc_config.mode` | `Cqp`/`Crf`/`Vbr`/`Cbr` | `Crf` | **Cqp ≡ Crf byte-for-byte on a still** (verified across preset {0,8} × qp {20,40,55}, `rate_control.rs:7-16`). Vbr/Cbr broken — hazard **H-8** | `rate_control.rs:23`, dflt `:105` |
| `rc_config.extended_crf_qindex_offset` | **0..=3**, up to 28 at qp 63 | `0` | fractional CRF in quarter-qindex steps | `rate_control.rs:57` / `pipeline.rs:1744` |
| `bit_depth` | **8 or 10** (12 refused) | `8` | 10-bit 4:2:0 at every preset; 10-bit **mono** only at preset ≥ 9 | `pipeline.rs:60` / validated `:967`, `:7380` |
| `chroma_420` | bool | **`false` = MONOCHROME** | must be set `true` for colour — hazard **H-9** | `pipeline.rs:122`, dflt `:395` |
| `tile_rows_log2` / `tile_cols_log2` | log2, **clamped C-style, never rejected** | `0` / `0` | real tile grid | `pipeline.rs:192`, `:198` / `:2015-2016` |
| `sb_size_override` | `None` \| 64 \| 128 | `None` (preset-derived) | **SB128 is signalled but the 128 root is forced-SPLIT** — no genuine 128-level RD search (`pipeline.rs:437-449`). Sweeping it varies signalling + CDF context, not partition quality | `pipeline.rs:218` / `:1949` |
| `hdr.tune` | **0..=5** | `1` | `0` VQ, `1` PSNR, `2` SSIM, **`3` IQ (the still tune)**, `4` MS-SSIM, **`5` FILM_GRAIN — *not* VMAF** (contradiction **C-1**). Rewrites 9 fields: hazard **H-4** | `hdr_mode.rs:60` / `pipeline.rs:1387` |
| `hdr.enable_variance_boost` | bool | **`false`** | THE still-image AQ — per-SB delta-q | `hdr_mode.rs:86` / `pipeline.rs:1768` |
| `hdr.variance_boost_strength` | **1..=4, `debug_assert` ONLY** | `2` | boost magnitude; **3 and 4 saturate to the same plan** (`avif.rs:806`). Release-mode OOB panic: hazard **H-10** | `hdr_mode.rs:87` / `var_boost.rs:231` |
| `hdr.variance_octile` | **1..=8, `debug_assert` ONLY** | `5` | which variance octile represents the SB. Same panic hazard | `hdr_mode.rs:88` / `var_boost.rs:108` |
| `hdr.variance_boost_curve` | 0..=3 (3 = PQ, fork kernel only) | `0` | curve shape; caps deltaq at 80/120 | `hdr_mode.rs:90` / `pipeline.rs:1798` |
| `hdr.enable_qm` + `min/max_qm_level` + `min/max_chroma_qm_level` | bool; 0..=15 (15 = identity) | `false`; `8`/`15`; `8`/`15` | quantization matrices, luma + chroma envelopes | `hdr_mode.rs:105-109` / `pipeline.rs:2103-2131` |
| `hdr.sharpness` | `i8`, **clamped 0..=7 at use** (negatives floor to 0) | `0` | loop-filter sharpness, signalled in the FH | `hdr_mode.rs:92` / `pipeline.rs:2091` |
| `hdr.max_tx_size` | **{32, 64} ONLY — hard refusal otherwise** | `64` | 32 forbids 64×64 squares in the partition search | `hdr_mode.rs:100` / validated `pipeline.rs:892` |
| `hdr.screen_content_mode` | `None` (derive from preset) \| `Some(3)` (force detector) | `None` | forces palette/IntraBC detection at any preset | `hdr_mode.rs:104` / `pipeline.rs:1659-1662` |
| `hdr.ac_bias` | 0.0..=8.0 | `0.0` | psy-distortion weight; **live in mainline** (`pipeline.rs:2041` is ungated) | `hdr_mode.rs:80` / `:2041` |
| `hdr.mode` | `Mainline` \| `HdrFork` | `Mainline` | master gate for fork-only knobs (grain, `tx_bias`, `sharp_tx`, `cdef_scaling`, …) | `hdr_mode.rs:46` |
| `superres_denom` | **9..=16 (`panic!` outside)** | `None` | refused with loop-restoration (preset ≤ 6) and with 10-bit | `pipeline.rs:71` / `:1011` |
| `color_description` | CICP triple + range | `unspecified()` = 2/2/2 studio | SH CICP; **also drives fork chroma delta-q** | `entropy/obu.rs:20` / `pipeline.rs:2026` |
| `recon_output` | bool | `false` | off = **1.36-1.39× faster at p10/p13**, byte-inert at preset ≥ 7 | `pipeline.rs:113` / `:3209` |
| `thread_count` | 0 = auto | `0` | **byte-inert**; with 1 tile there is nothing to parallelise | `pipeline.rs:237` / `:2247` |

`AvifEncoder` reaches only **quality, speed, bit-depth, and (qm / variance-boost / lossless / subsampling / threads / CICP)** — **not** tune, sharpness, QM levels, octile, curve, `max_tx_size`, `screen_content_mode`, tiles, sb-size, superres, or fork mode.

### 4.2 REFUSED / INERT / DEAD — do not sweep

The repo maintains a generated inventory at `rust/docs/REFUSED-CONFIGS.md` (14 CAPABILITY + 22 CONTRACT refusals; `tools/refusal_inventory.sh --check` is a CI gate). The ones that matter here:

| knob | exact condition | file:line |
|---|---|---|
| **`aq_mode`** | `rc_config.aq_mode != 0` → *"aq_mode must be 0: C's aq-mode deltaq is TPL-gated and therefore **INERT for a single still** (`rc_aq.c:899`), so C's own default of 2 changes nothing there, while this port's non-zero aq_mode runs a homegrown frame-level VAQ/TPL qindex shift that is a port of nothing"* | `pipeline.rs:927-934` |
| `max_tx_size` | not 32 or 64 | `pipeline.rs:892-894` |
| `chroma_sample_position` | `> 2` (3 reserved) | `pipeline.rs:902-907` |
| bit depth | ∉ {8,10}; 10-bit mono needs preset ≥ 9 | `pipeline.rs:971-993` |
| **inter frames** | `!is_key` → "This encoder is still-image only" | `pipeline.rs:1462-1468` |
| superres | + loop restoration (preset ≤ 6), or + 10-bit | `pipeline.rs:1107-1121` |
| **AVIF 4:4:4 / 4:2:2** | `chroma_subsampling != Yuv420` → "only 4:2:0 chroma is implemented (and C v4.2.0 ships 420 only)" | `avif.rs:577-581` |
| AVIF lossless mono; quality > 99.2 mono | maps to QP 0 with no lossless arm | `avif.rs:567-576`, `:602-608` |
| **QP 0 (coded-lossless)** | six-arm envelope: refused under fork mode, monochrome, `bit_depth != 8`, inter, **`allow_screen_content_tools`**, superres | `pipeline.rs:1041-1090` |

**Structurally inert (settable, never change a byte):** `hdr.tf_strength`, `hdr.kf_tf_strength`, `hdr.qp_scale_compress_strength`, `hdr.noise_adaptive_filtering` (all **zero consumers crate-wide**); `rc_config.max_bitrate` / `buffer_size_ms` / `temporal_layers`; `speed_config.enable_temporal_filter` (gated `&& !is_key`, and inter is refused ⇒ unreachable on a still); `TfConfig` (only ever `default()`); `SbSizeInputs.{fast_decode,resize,rtc,sframe,variance_boost}` (built with `..Default::default()`).

**Dead modules** (consumer counts verified by reachability grep): `segmentation.rs` (56 KB, **0** consumers — the five FH writer sites still emit hardcoded `segmentation_enabled = 0`); `multipass.rs` (**0**, self-documented "dead everywhere, not just dormant"); `perceptual.rs` (1 call site, inside the refused `aq_mode != 0` branch ⇒ unreachable); `film_grain.rs` (1 site, `let _grain_params = …` — discarded; the live grain path is `noise_gen.rs`, fork-gated).

**Derived, not settable:** `palette_level` / `intrabc_level` (`sc_detect.rs:359/365`), chroma delta-q, CDEF search (preset-gated), restoration (preset-derived), deblock levels. Diagnostic env override only: `SVTAV1_SC_TOOLS=nopalette|noibc|none` (`pipeline.rs:7562-7568`).

### 4.3 PORT-MISSING (upstream v4.2.0 has it, the port does not)

| upstream knob | port status | wiring cost |
|---|---|---|
| `--enable-cdef` | SH bit **hardcoded `true`** (`entropy/obu.rs:754`); search preset-gated, no config field | **Shallow** — add field + gate the search predicate (was pruned as a dead field in `99d0f606b`) |
| `--enable-restoration` | preset-derived only (`speed_config.rs:153`) | **Shallow** — `seq_tools_for_preset` already computes it; add an override |
| `--enable-palette` | derived from `sc_detect` | **Shallow** — a diagnostic env override already exists |
| `--fast-decode` | field exists (`sb128_geom.rs:63`) but `SbSizeInputs` is built with `..Default::default()` | **Trivial for SB-size**; deep for its CDEF/LR effects (`cdef.rs:212`: only the `fast_decode == 0` branch was transcribed) |
| `--aq-mode 1` (segmentation VAQ) | `segmentation.rs` is capability-complete and C-parity-tested but **unwired** | **Deep** — 5 FH writer sites + config surface |
| `--tune 5` = VMAF | slot 5 is `TUNE_FILM_GRAIN` (fork enum); `tune.rs:22-23` states mainline's VMAF is not modelled | Deep (needs the metric) |
| `--enable-tf`, `--enable-overlays`, `--scd`, `--startup-qp-offset`, `--luminance-qp-bias`, `--enable-dlf`, `--enable-mfmv`, `--irefresh-type`, `--lookahead`, `--pred-struct` | **zero occurrences** in the port | mostly multi-frame/video — irrelevant to stills **except `--enable-dlf`** and `--luminance-qp-bias` |

**Worth exposing for this DoE, in priority order:** `--enable-cdef`, `--enable-restoration`, `--enable-palette` (all shallow, all directly relevant to still quality-at-size), then `--fast-decode`'s SB-size half.

---

## 5. Port reality — `zenav1-aom`

**The knob surface is layered, and the two layers are not the same set.**

| Layer | Type | What it is |
|---|---|---|
| **HARNESS KNOBS** | `aom_bench::ToggleKnobs` — `crates/aom-bench/src/lib.rs:479-667` | The ~30-field struct the differential harness (and therefore the live zenmetrics sweep) drives. Each field is documented with the upstream CLI flag it models and its C default. |
| **CRATE CONFIG** | `aom_encode::TuneKnobs` (`crates/aom-encode/src/lib.rs:116-129`), `QmCtx` (`:137-151`), `SpeedFeatures`, `PickFrameCfg`, `PackCfg` | The port's internal config types. **Some ported features live only here** and have no `ToggleKnobs` field — notably the whole tune-IQ family. |

### 5.1 `ToggleKnobs` — exposed, with C defaults (all from `crates/aom-bench/src/lib.rs:479-667`)

| port field | models | range | default | still-image note |
|---|---|---|---|---|
| `enable_rect_partitions` / `enable_ab_partitions` / `enable_1to4_partitions` | `--enable-{rect,ab,1to4}-partitions` | bool | `true` ×3 | partition-arm search effort — direct speed/quality trade |
| `min_partition_size_px` / `max_partition_size_px` | `--min/max-partition-size` | {4,8,16,32,64,128} | `4` / `128` | caps the partition tree; a strong speed lever |
| `enable_intra_edge_filter` / `enable_filter_intra` | seq-header bits (`encoder.c:646/647`) | bool | `true` | port asserts the C seq header agrees |
| `enable_smooth_intra` / `enable_paeth_intra` / `enable_cfl_intra` / `enable_directional_intra` / `enable_diagonal_intra` / `enable_angle_delta` | the intra-mode toggles | bool | `true` ×6 | intra mode-set size — **the core still-image search surface** |
| `enable_tx64` / `enable_rect_tx` / `enable_flip_idtx` | tx-size / tx-type sets | bool | `true` ×3 | `enable_tx64=0` caps the largest tx at 32 |
| `use_intra_dct_only` / `use_intra_default_tx_only` | `--use-intra-dct-only`, `--use-intra-default-tx-only` | bool | `false` | aggressive tx-search shortcuts |
| `reduced_tx_type_set` | frame-header bit `reduced_tx_set_used` | bool | `false` | shrinks both search and signalling |
| `enable_tx_size_search` | `--enable-tx-size-search` | bool | `true` | **C FORBIDS combining `=0` with `enable_tx64=0`** (`encodeframe.c:2461` assert) |
| `cdf_update_mode` | `--cdf-update-mode` | 0..2 | `1` | **mode 2 ≡ mode 1 on a lone KEY frame** — the port's own note says "not swept" |
| `disable_trellis_quant` | `--disable-trellis-quant` | 0..3 | `3` | `0` FULL / `1` NONE / `2` final-pass / `3` default (≈ FULL on the intra envelope) |
| `enable_palette` | port's palette RD-search enable | bool | **`false`** | still requires the frame's `allow_screen_content_tools`, exactly as C |
| `enable_intrabc` | port's IntraBC DV-search enable | bool | **`false`** | dominant cost on screen frames (**H-6**) |
| `tune_content_screen` | `--tune-content=screen` (`AV1E_SET_TUNE_CONTENT`) | bool | `false` | screen tools + `allow_intrabc` **without running the detector** — the only way to force screen tools deterministically. KB-41 root #13 |
| `qm` | `--enable-qm --qm-min --qm-max` | `Option<(i32,i32)>` | **`None`** | derived levels are cross-checked against the bootstrap header; drive C with `EncodeCell::c_encode_qm` |
| `deltaq_mode2` / `deltaq_mode3` | `--deltaq-mode=2` (perceptual, wavelet AC) / `=3` (perceptual-AI, wiener variance) | bool | `false` | **PORT-side only — not emitted by `ToggleKnobs::c_ctrls`**; the C side must be driven separately |
| `delta_lf_mode` | `--delta-lf-mode=1` | bool | `false` | rides on a firing delta-q mode |
| `coeff_cost_upd_freq` / `mode_cost_upd_freq` | `--coeff/mode-cost-upd-freq` | 0..3 | `0` (COST_UPD_SB) | **⚠ PRESENT BUT INERT PORT-SIDE** — the field's own doc says "the PORT-side gate is **NOT wired yet**"; C ctrls are emitted but the port ignores them |
| `disable_tx_stats_prune` | (no CLI control) | bool | `false` | **anti-vacuity witness only, not a product knob** |

### 5.2 The tune-IQ family: ported, but NOT reachable from the sweep harness

`crates/aom-encode/tests/encoder_gate_tune_iq_e2e.rs` is a full-frame **byte-match**
gate for "the **tune=IQ / tune=SSIMULACRA2 family** (PARITY.md C4)". The pieces
exist in the port crates — `variance_boost` in `aom-encode/src/allintra_vis.rs`,
`adaptive_sharpness` in `aom-encode/src/lf_search.rs`, `chroma_deltaq` and
`tune_iq` in `aom-dsp/src/quant/build_quantizer.rs`, `dist_metric` in
`aom-dsp/src/txb/optimize.rs` and `aom-encode/src/speed_features.rs`,
`screen_detect.rs` for the detector — and the public surface is
`aom_encode::TuneKnobs`:

```rust
pub struct TuneKnobs {
    /// `--dist-metric=qm-psnr` … weights the trellis distortion by the forward QM
    pub use_qm_dist_metric: bool,
    /// tuning ∈ {AOM_TUNE_IQ, AOM_TUNE_SSIMULACRA2} — the trellis `rshift = 7` (vs 5) arm
    pub iq_tuning: bool,
}
```
(`crates/aom-encode/src/lib.rs:116-129`)

**But `ToggleKnobs` has no tune field** (grep count 0 for `AOM_TUNE_IQ`,
`tune_iq`, `ssimulacra2`, `sharpness`, `chroma_deltaq`, `adaptive_sharpness`,
`dist_metric`, `variance_boost` anywhere under `crates/aom-bench/src/`), so the
harness the sweep drives cannot request it. **This is the single highest-value
port-exposure gap for this DoE** — see §10.

**Independent confirmation of hazard H-4 from the port's own test:** the gate
drives the reference with *"`ref_encode_av1_kf_tune` — **tuning first, explicit
overrides after, aomenc CLI ordering**"* (`encoder_gate_tune_iq_e2e.rs:4-5`).
The port authors hit the same ordering trap and encoded the resolution in the
harness.

### 5.3 The byte-identity constraint is the real limit on this arm

Every aom-rs cell must produce a frame-OBU payload **byte-identical to the C
oracle** (`zenmetrics .../sweep/encode.rs:1286-1295`). So a knob is only
sweepable here if **both** sides can be driven to the same config: `ToggleKnobs`
carries a `c_ctrls()` emitter for most fields, but `qm` (use
`EncodeCell::c_encode_qm`), `deltaq_mode2`, `deltaq_mode3`,
`disable_tx_stats_prune` and `tune_content_screen` are **port-side only** and
need their C counterpart driven explicitly. Adding a knob without its C half
turns every cell into a divergence error, not a measurement.

---

## 6. Harness reality — what the live sweep actually drives today

This matters because it defines the delta the DoE lane has to build.

**The entire knob surface in flight today is `{backend, speed}` × `q`.**
`knob_tuple_json = {"backend":"svt-rs"|"aom-rs","speed":S}`
(`zenmetrics/scripts/jobsys/avifsvt_cells.py:23-25`), and the executor accepts
nothing else — the aom-rs path **hard-errors on any unknown knob**:

> `zenavif aom-rs backend: knob '{unknown}' is not wired (supported: backend, speed); refusing to silently ignore it`
> — `zenmetrics/crates/zenmetrics-cli/src/sweep/encode.rs:1181-1187`

(That refusal is good design and should be preserved — it is why no knob has
ever been silently dropped. Every knob the DoE lane adds must be threaded
through `avif_config_from_knobs` / `encode_avif_aom_rs` explicitly.)

| axis | svt-rs | aom-rs |
|---|---|---|
| speed | **1..=10**, via `zenavif` `Av1Backend::SvtRs` → `speed_to_svt_preset` → **7 distinct presets** (hazard **H-2**) | **0..=9**, raw `--cpu-used`, refused above 9 (`encode.rs:1189-1191`) — exactly libaom's non-realtime range |
| q | 30-point grid: `1` + step 5 over `5..70` + step 2 over `72..100` (`avifsvt_cells.py:33-36`) | same grid → `aom_rs_cq_level(q)` |
| chroma | 4:2:0 **implied by the backend**, not a knob (`encode.rs:1046-1051`) | 4:2:0 hardcoded (`encode.rs:1194`, `ss_x/ss_y = 1`) |
| usage | — | `usage: 2` = `AOM_USAGE_ALL_INTRA`, commented "the avifenc still-image mode" (`encode.rs:1209`) |
| bit depth | 8 | `bd: 8` (`encode.rs:1212`) |
| CICP | zenavif default | **hardcoded** Bt709/sRGB/Bt601/full (`encode.rs:1322-1327`) |
| tune / qm / variance-boost / sharpness / cdef / deltaq / tiles / sb-size | **none wired** | **none wired** |

**The aom-rs arm is a differential harness, not a plain encoder** — see hazard
**H-5**. It runs the C oracle *and* the port on every cell and refuses to emit
bytes unless the frame-OBU payloads match
(`encode.rs:1237-1295`). Its port-side tool toggles are not free either: they are
**mirrored from the oracle's own screen detection**, not set independently —
`ToggleKnobs { enable_palette: screen, enable_intrabc: screen, ..default() }`
(`encode.rs:1281-1284`), because "driving the port with tools the oracle never
enabled (or vice versa) is a harness mismatch, not a port divergence".

**Consequence for the DoE lane:** on the aom-rs arm, any knob you add must be
applied to **both** the oracle config and the port's `ToggleKnobs`, or every cell
will fail the byte-identity gate. That is a real constraint on which libaom knobs
are cheaply sweepable there today — and an argument for sweeping the *SVT* arm
first, where the port encodes alone.

---

## 7. Upstream/ecosystem recommended starting combos for still images

Labelled by whose recommendation it is. **libavif's view is the ecosystem
reference, not an upstream-encoder statement.**

### 7.1 libaom — upstream's own recommendation, expressed in code

libaom does not publish an "AVIF settings" doc; it encodes the recommendation
**as defaults**, in two layers:

1. **`AOM_USAGE_ALL_INTRA` changes four defaults** ([`av1_cx_iface.c:3055-3078`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L3055-L3078)):
   `enable-cdef` 1→**0** ("CDEF has been found to blur images"), `screen-detection-mode` 1→**2**,
   `qm-min` 5→**4**, `qm-max` 9→**10**. The source comment is explicit that these are *defaults*:
   *"if users set --enable-cdef, --screen-detection-mode, --qm-min, or --qm-max … the encoder still respects it."*
2. **`tune=iq` is the packaged image recipe** ([`aomcx.h:1751-1766`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/aom/aomcx.h#L1751-L1766), implemented at [`av1_cx_iface.c:1938-1978`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L1938-L1978)) —
   `--enable-qm=1 --qm-min=2 --qm-max=10 --sharpness=7 --dist-metric=qm-psnr --enable-cdef=3 --enable-chroma-deltaq=1 --deltaq-mode=6 --screen-detection-mode=2`, plus `--enable-adaptive-sharpness=1` for IQ only.

**⇒ The libaom still baseline is:** `usage=all-intra`, `end-usage=q`, `cq-level=<dial>`, `cpu-used=<dial>`, `tune=iq`, `kf-max-dist=0`, everything else default.

### 7.2 SVT-AV1 — upstream's own recommendation

- **`--avif 1`** — the only switch whose stated purpose is stills: *"Enable still-picture coding optimizations for improved coding efficiency and reduced memory usage"* ([Parameters.md](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.2.0/Docs/Parameters.md)).
- **`--tune 3` (IQ)** — the only tune upstream marks **"still image only"** (same doc).
- **`--variance-boost-strength 3`** — [Appendix-Variance-Boost.md](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.2.0/Docs/Appendix-Variance-Boost.md): *"Strength 3 is best for **still images** or videos with a balanced mix of very high-contrast and low-contrast scenes."* Octile: *"Recommended values are between 4 and 7"* (default 5).
- **`--tune 0` (VQ)** for subjective sharpness — [CommonQuestions.md § Bang-For-Buck](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.2.0/Docs/CommonQuestions.md): *"The use of subjective mode (`--tune=0`) often results in an image with greater sharpness."*
- **Decode-side levers** (same doc, § Improving Decoding Performance): `--fast-decode 1`, higher CRF, `--tile-columns 2`, avoid synthetic grain, higher presets, 8-bit over 10-bit — each *"has the potential to reduce the image quality."*

**⇒ The SVT still baseline is:** `--avif 1 --tune 3 --crf <dial> --preset <dial>`, with `--enable-variance-boost 1 --variance-boost-strength 3` already implied by tune 3.

### 7.3 libavif v1.4.2 — what a real AVIF encoder actually sets

**libaom backend** ([`src/codec_aom.c`](https://github.com/AOMediaCodec/libavif/blob/v1.4.2/src/codec_aom.c)):
`AOM_USAGE_ALL_INTRA` when still ([`:707-713`](https://github.com/AOMediaCodec/libavif/blob/v1.4.2/src/codec_aom.c#L707-L713)) ⇒ `rc_end_usage = AOM_Q` ([`:827-829`](https://github.com/AOMediaCodec/libavif/blob/v1.4.2/src/codec_aom.c#L827-L829));
`kf_mode = AOM_KF_DISABLED`, `kf_max_dist = 0` ([`:910-912`](https://github.com/AOMediaCodec/libavif/blob/v1.4.2/src/codec_aom.c#L910-L912));
`AOME_SET_CQ_LEVEL` ([`:987`](https://github.com/AOMediaCodec/libavif/blob/v1.4.2/src/codec_aom.c#L987));
`AV1E_SET_SKIP_POSTPROC_FILTERING = 1` in all-intra ([`:1064-1068`](https://github.com/AOMediaCodec/libavif/blob/v1.4.2/src/codec_aom.c#L1064-L1068));
**default tune for colour = `iq` on libaom ≥ 3.13.0**, `psnr` for alpha ([`:737-786`](https://github.com/AOMediaCodec/libavif/blob/v1.4.2/src/codec_aom.c#L737-L786) — "AOM_TUNE_IQ has been tuned for the YCbCr family … favored for its low perceptual distortion");
`AV1E_SET_ENABLE_RESTORATION = 0` for **alpha only** ([`:1083`](https://github.com/AOMediaCodec/libavif/blob/v1.4.2/src/codec_aom.c#L1083));
default `--speed 6` ([`apps/avifenc.c:213`](https://github.com/AOMediaCodec/libavif/blob/v1.4.2/apps/avifenc.c#L213)).
Only seven knobs are passed through as codec-specific options: `aq-mode`, `cq-level`, `enable-chroma-deltaq`, `sharpness`, `tune`, `film-grain-test`, `film-grain-table` ([`:467-483`](https://github.com/AOMediaCodec/libavif/blob/v1.4.2/src/codec_aom.c#L467-L483)).

**SVT backend** ([`src/codec_svt.c`](https://github.com/AOMediaCodec/libavif/blob/v1.4.2/src/codec_svt.c)):
`aq_mode = 2` on SVT ≥ 4.0.0 ([`:173-177`](https://github.com/AOMediaCodec/libavif/blob/v1.4.2/src/codec_svt.c#L173-L177));
`rate_control_mode = 0` with the comment *"CRF because enable_adaptive_quantization is 2"* ([`:186`](https://github.com/AOMediaCodec/libavif/blob/v1.4.2/src/codec_svt.c#L186));
2-pass disabled; `force_key_frames = true` ([`:219`](https://github.com/AOMediaCodec/libavif/blob/v1.4.2/src/codec_svt.c#L219));
`lossless` on SVT ≥ 3.0.0 ([`:247`](https://github.com/AOMediaCodec/libavif/blob/v1.4.2/src/codec_svt.c#L247));
**`avif = 1` only on SVT ≥ 4.0.0**, with the comment *"Although the `avif` option was added in v3.0.0, it had a serious bug that was not fixed until v4.0.0"* ([`:250-253`](https://github.com/AOMediaCodec/libavif/blob/v1.4.2/src/codec_svt.c#L250-L253), [SVT issue 2245](https://gitlab.com/AOMediaCodec/SVT-AV1/-/issues/2245)) — **our v4.2.0 pin is past the fix**.

**⇒ Cross-backend observation the DoE lane should internalize.** libavif drives SVT with **`aq_mode = 2`** — precisely the value `zenav1-svt` **refuses** (`pipeline.rs:927`). The port's refusal note argues that for a *single still* C's aq-mode deltaq is TPL-gated and therefore inert anyway, so libavif's setting may be a no-op in the same situation. **That equivalence is asserted by the port, not measured here** — see UNVERIFIED **U-2**.

---

## 8. Ranked shortlists — what the DoE lane should design around

Ranking criterion: **expected effect on quality-at-size-at-speed for a single
SDR 4:2:0 8-bit still, given what the port actually implements.** A knob that is
refused, inert, or unreachable is excluded no matter how important it is
upstream. "Partners" = factors it aliases or interacts with, which the design
must handle (nest, block, or hold fixed).

### 8.1 `zenav1-svt` — 12 knobs

| # | knob | why it matters (mechanism) | upstream guidance | interaction partners |
|---|---|---|---|---|
| 1 | **`preset` 0..=9** (drive `EncodePipeline`, not the speed dial) | The master speed/efficiency dial: sets sb-size (128 ≤M6 / 64 ≥M7), filter-intra (off ≥M7), SG restoration (off ≥M4), Wiener (off ≥M9), and ~40 search predicates. Dominates encode time. | *"One should use the lowest preset that is tolerable"* — [CommonQuestions § What Presets Do](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.2.0/Docs/CommonQuestions.md) | everything; **H-2** (dial non-injective), sb-size, restoration |
| 2 | **`rc_config.qp` 0..=63** | THE rate/quality dial. Also feeds the SB-size rule and tune-IQ's `max_tx_size` switch at qp 45. | `--crf` is *"The quality/filesize tradeoff is controlled by the `crf` parameter"* — [CommonQuestions § Bang-For-Buck](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.2.0/Docs/CommonQuestions.md) | qp 0 = lossless stratum (**H-6**); `max_tx_size` under tune 3 |
| 3 | **`hdr.tune` ∈ {0,1,2,3}** — **outermost axis** | Changes the RD objective, and for 3/4 rewrites 9 other fields. Tune 3 = IQ is the only mode upstream marks still-image-only. | *"3 = IQ (**still image only**)"*; *"`--tune=0` often results in an image with greater sharpness"* — [Parameters.md](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.2.0/Docs/Parameters.md), [CommonQuestions](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.2.0/Docs/CommonQuestions.md) | **H-4** — aliases qm*, sharpness, variance_boost*, max_tx_size, scm |
| 4 | **`hdr.enable_variance_boost` + `strength` (1..=3 distinct) + `octile` (4..=7)** | Per-64×64 delta-q that lowers qindex on low-variance superblocks — "smarter allocation of bits", visibly evens quality across contrast. The main perceptual lever that is *not* the quantizer. | *"**Strength 3 is best for still images**"*; *"Recommended values [octile] are between 4 and 7"* — [Appendix-Variance-Boost.md](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.2.0/Docs/Appendix-Variance-Boost.md) | forced ON by tune 3/4 (**H-4**); clamp in harness (**H-10**); strength 3≡4 |
| 5 | **`hdr.enable_qm` + `min/max_qm_level`** | Quantization matrices reshape the quantizer across frequencies — the lever libaom's image tune leans on hardest. **Off by default in SVT**, so it is unexplored territory here. | SVT ships it OFF (`--enable-qm` default 0); libaom's image tunes force it ON with a narrow window (2-10) | tune 3/4 forces on at 4/10; chroma QM levels |
| 6 | **`hdr.sharpness` 0..=7** | Deblocking-filter sharpness + RC bias, **signalled in the frame header**. Both backends' image tunes force it to 7 — strong prior that it matters for stills. | forced to 7 by `tune=iq` on *both* encoders | `--enable-dlf`; **H-14** (scale differs from libaom) |
| 7 | **`hdr.screen_content_mode` ∈ {None, Some(3)}** | Forces the anti-alias-aware detector at any preset, turning on palette + IntraBC. Decisive on text/UI/screenshot content, which is a large slice of a web corpus. | `--scm 3` added v4.0.0 "for better screen content detection" — [CHANGELOG](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.2.0/CHANGELOG.md) | **H-6** (cost + lossless refusal); content-stratify |
| 8 | **`hdr.ac_bias` 0.0..=8.0** | RD bias toward high-frequency error — texture and grain retention. Live in mainline, default 0.0, so entirely unexplored. | *"helps with texture preservation and film grain retention"* — [Parameters.md](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.2.0/Docs/Parameters.md) | tune (both shape RD) |
| 9 | **`hdr.max_tx_size` ∈ {32, 64}** | Forbids 64×64 square transforms; changes the partition/transform search. Tune-IQ picks it *by qp* (32 at qp ≤ 45), which is upstream telling you the optimum is quality-dependent. | tune-IQ's own qp-conditional choice | qp (the switch is at qp 45); tune |
| 10 | **`tile_cols_log2` / `tile_rows_log2`** | Encode+decode parallelism at an efficiency cost; the standard speed lever that is not preset. | *"can lead to large improvements in encoding and decoding performance… Tiling can lead to visible artifacts"* — [CommonQuestions § Improving Decoding Performance](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.2.0/Docs/CommonQuestions.md) | image dimensions (tiles need width); `thread_count` |
| 11 | **`bit_depth` ∈ {8, 10}** | 10-bit reduces banding in smooth gradients at a bitrate cost — a real still-image quality axis, not just an HDR one. | *"Use 8-bit color depth instead of 10"* listed as a decode-speed lever — same doc | preset (10-bit mono needs ≥ 9); profile |
| 12 | **`sb_size_override` ∈ {None, 64, 128}** | Superblock size. **Lower confidence:** see **H-11** — SB128 is signalled but its RD search is unported, so this measures signalling/CDF effects only. | preset table: sb 128 at ≤M6, 64 at ≥M7 | preset (which already sets it); **H-11** |

**Excluded and why:** `aq_mode` (refused, `pipeline.rs:927`), `rc_mode` Vbr/Cbr (**H-8**),
CRF-vs-CQP (**H-3**, one factor), `tf_strength`/`kf_tf_strength`/`qp_scale_compress_strength`/
`noise_adaptive_filtering` (zero consumers), `thread_count` (byte-inert), 4:4:4 / 4:2:2 /
12-bit (refused), everything multi-frame, and `tune=5` (FILM_GRAIN in the fork enum, and the
fork-gated `noise_gen` path is the only live grain code — see contradiction **C-1**).

### 8.2 `zenav1-aom` — 13 knobs

Ranked for the same criterion. **Note the two-tier reachability**: (H) = a
`ToggleKnobs` field the sweep harness can drive today; (C) = ported in the
crates but needs a harness field added first.

| # | knob | tier | why it matters (mechanism) | upstream guidance | interaction partners |
|---|---|---|---|---|---|
| 1 | **`--cpu-used` 0..=9** | H (`speed`) | The master speed dial; range is exactly libaom's non-realtime bound. Drives every speed feature. | `RANGE_CHECK(cpu_used, 0, usage==REALTIME ? 12 : 9)` — [`:775-776`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L775-L776) | everything; **H-6** (screen cost at 0-1) |
| 2 | **`--cq-level` 0..=63** | H (`q`) | THE quality dial under `AOM_Q` end-usage, which all-intra selects. | [`:802`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L802); libavif uses `AOME_SET_CQ_LEVEL` | qm level derivation; cdef-adaptive thresholds |
| 3 | **`tune` ∈ {psnr, ssim, iq, ssimulacra2}** — **outermost axis** | **C** | libaom's packaged image recipe; forces 9-10 knobs. `iq` is libavif's default for colour on ≥3.13.0. **Almost certainly the largest single quality win available on this backend.** | *"AOM_TUNE_IQ and AOM_TUNE_SSIMULACRA2 are meant for image encoding"* — [`aomcx.h:1751-1766`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/aom/aomcx.h#L1751-L1766) | **H-4** — aliases qm, sharpness, dist-metric, cdef, chroma-deltaq, deltaq, screen-detection |
| 4 | **`--enable-qm` + `--qm-min`/`--qm-max`** | H (`qm: Option<(min,max)>`) | Reshapes the quantizer across frequencies. Three distinct upstream windows exist — default `5/9`, all-intra `4/10`, tune-IQ `2/10` — which is upstream saying the optimum moves with the use case. | [`quant_common.h:37-42`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/common/quant_common.h#L37-L42) | tune (forces 2/10); `dist-metric=qm-psnr` weights the trellis by the same matrix |
| 5 | **`--deltaq-mode` ∈ {0,1,2,3,6}** | H for 2/3; **C** for 6 | Per-SB qindex modulation. `3` = perceptual-AI is *the* all-intra mode; **`6` = Variance Boost** is the direct analogue of SVT's headline still-image lever and what `tune=iq` selects. | [`encoder.h:153-162`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/encoder/encoder.h#L153-L162) | tune (forces 6); `delta_lf_mode` rides on it |
| 6 | **`--enable-cdef` ∈ {0,1,3}** | **C** | All-intra defaults it **OFF** because "CDEF has been found to blur images"; tune-IQ instead picks **3 = ADAPTIVE** (off at qindex ≤32, reduced 33-220, full 221-255). Two upstream opinions in one release — worth measuring. | [`:3065-3078`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L3065-L3078), [`pickcdef.h:22-29`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/encoder/pickcdef.h#L22-L29) | qindex (mode 3 is qindex-conditional); tune |
| 7 | **`--sharpness` 0..=7** | **C** | Bias toward block sharpness in coefficient RDO. Both encoders' image tunes force **7** — the strongest cross-backend prior in this dossier. | forced to 7 by `handle_tuning` — [`:1951`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L1951) | `enable_adaptive_sharpness`; **H-14** (different scale from SVT) |
| 8 | **`--disable-trellis-quant` ∈ {0,1,2,3}** | H | Coefficient-level RDO. A classic size-at-quality lever with a large, well-understood speed cost, and one of the few pure *effort* knobs that is not `cpu-used`. | `init_rd_sf` (`speed_features.c:2479-2498`) per the port's own citation | `dist-metric` (tune-IQ changes the trellis distortion **and** its `rshift`) |
| 9 | **partition envelope: `min/max_partition_size_px` + `enable_{rect,ab,1to4}_partitions`** | H | The partition search is the dominant cost term and directly sets the size/quality frontier — and unlike `cpu-used` it moves one dimension at a time, which is what a model wants. | libaom CLI defaults 4 / 128, all three arms on | `cpu-used` (which already prunes these); `sb-size` |
| 10 | **intra mode-set toggles** (`smooth`, `paeth`, `cfl`, `directional`, `diagonal`, `angle_delta`, `filter_intra`, `intra_edge_filter`) | H | For a still, intra IS the encoder. Eight orthogonal on/off arms, each a clean effort/quality contrast — the richest cheap factor block on either backend. | all default ON upstream | `cpu-used`; each other (mode-set overlap) |
| 11 | **`--enable-palette` / `--enable-intrabc` / `--tune-content=screen`** | H | Screen-content tools. `tune_content_screen` is the only way to force them **deterministically** (bypasses the detector), which is what makes them designable factors instead of content lottery. | `av1_set_screen_content_options` (`encoder.c:2449-2455`) per the port's citation | **H-6** (cost + content stratification) |
| 12 | **`--enable-tx64` / `--enable-rect-tx` / `--enable-flip-idtx` / `reduced_tx_type_set` / `enable_tx_size_search`** | H | Transform search surface. | — | **`enable_tx_size_search=0` + `enable_tx64=0` is FORBIDDEN by C** (`encodeframe.c:2461` assert) — exclude that cell |
| 13 | **`--enable-chroma-deltaq`** | **C** | Lets the encoder adjust chroma quality for subsampling. Forced on by tune-IQ. | [`:1964-1966`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L1964-L1966) | **H-13** — hard-errors with `--lossless=1` |

**Excluded and why:** `--arnr-*` (no alt-ref on a still, **H-7**), VMAF/Butteraugli
tunes (build-flag gated and not ported, **H-12**), `coeff_cost_upd_freq` /
`mode_cost_upd_freq` (**present but port-side inert**, per the field's own doc),
`cdf_update_mode=2` (≡ 1 on a lone KEY frame), `disable_tx_stats_prune`
(anti-vacuity witness, not a product knob), CICP/colour (harness-pinned, **H-15**),
`--row-mt` / threads (speed-only, and confounded by **H-5**).

---

## 9. Hazards — read this before designing any factor grid

Ordered by how much damage they do to a sweep. **H-1…H-5 invalidate results
silently**; the rest cost cells or mislead a model.

### H-1 — the version trap: `svtav1-v4.1.0` is not the pin
`/home/lilith/work/zen/svtav1-v4.1.0` is a **different, older tree** than what
`zenav1-svt` tracks (v4.2.0). Five things exist at v4.2.0 and **not** at v4.1.0
(diffed from the parameter tables at both tags): **`--cqp`, `--enable-intrabc`,
`--enable-kf-tf`, `--hbd-mds`, and `--tune 5`** (`--tune` was `[0-4]` at v4.1.0,
`[0-5]` at v4.2.0). Anything built or documented against the v4.1.0 directory
will silently lack them.
Sources: [v4.2.0 Parameters.md](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.2.0/Docs/Parameters.md) ·
[v4.1.0 Parameters.md](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.1.0/Docs/Parameters.md) ·
[v4.2.0 CHANGELOG](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.2.0/CHANGELOG.md) ("New CLI options: `--cqp`, `--enable-intrabc`, `--hbd-mds`, `--enable-kf-tf`").

### H-2 — **the svt-rs speed dial is NOT injective: 10 levels → 7 encodes**
`speed_to_svt_preset(speed) = min(((speed-1)*13 + 4) / 9, 9)` — present
identically in **both** layers:
`zenavif/src/encoder_svt_rs.rs:219-222` and `zenav1-svt/rust/svtav1/src/avif.rs:295-304`.
Evaluated:

| speed | 1 | 2 | 3 | 4 | 5 | 6 | **7** | **8** | **9** | **10** |
|---|---|---|---|---|---|---|---|---|---|---|
| preset | 0 | 1 | 3 | 4 | 6 | 7 | **9** | **9** | **9** | **9** |

**Speeds 7, 8, 9, 10 are ONE cell.** Presets 2, 5, 8 are unreachable through
the dial. So a "dense 10-level speed axis" for svt-rs delivers **7 distinct
settings, with the top level 4× over-weighted** — a speed/size model fit on it
will see four identical-byte points at different measured times and read that as
pure noise at the fast end.
The `.min(9)` is deliberate and correct: `avif.rs:295-299` explains C remaps
every all-intra preset above M9 down to M9 (`enc_handle.c:4416-4419`), so
"preset 13" on a still **is** an M9 encode.
**⚠ Two docs contradict the code and lose:** `zenmetrics/benchmarks/avif_sweep_subsample_2026-09-01.md:151`
("SVT preset 0..=13 linear internally") and `zenavif/src/encode_plan.rs:158`
("speed linearly to SVT presets 0..=13"), plus `zenav1-svt`'s own `avif.rs:14`
and `:78` ("Mapped to svtav1 presets 0-13"). **Port source wins.**
*DoE action:* sweep **preset directly on `EncodePipeline` (0..=9)**, not the
speed dial; or use speeds {1,2,3,4,5,6,7} and drop 8-10 as duplicates.

### H-3 — CRF vs CQP is an aq-mode alias, not a second quality knob
Upstream: `--crf` ≡ `--rc 0 --aq-mode 2 --qp x`; `--cqp` ≡ `--rc 0 --aq-mode 0 --qp x`
([Parameters.md rows 87-88](https://gitlab.com/AOMediaCodec/SVT-AV1/-/blob/v4.2.0/Docs/Parameters.md)).
They differ **only** in aq-mode. Since `zenav1-svt` refuses `aq_mode != 0`, and
independently verifies **`Cqp ≡ Crf` byte-for-byte on a still**
(`rate_control.rs:7-16`), CRF and CQP are **one factor** in this port.
*DoE action:* do not cross `rc_mode` with `qp`. One quantizer axis.

### H-4 — **`tune` is a super-factor on BOTH backends: it silently rewrites 9-10 other knobs**
This is the single biggest threat to a factorial design, and it is symmetric.

| | forces |
|---|---|
| **libaom `tune=iq`(10) / `tune=ssimulacra2`(11)** | `enable-qm=1`, `qm-min=2`, `qm-max=10`, `sharpness=7`, `dist-metric=qm-psnr`, `enable-cdef=3`, `enable-chroma-deltaq=1`, `deltaq-mode=6`, `screen-detection-mode=2`; **IQ only:** `enable-adaptive-sharpness=1` ([`av1_cx_iface.c:1938-1978`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L1938-L1978)) |
| **`zenav1-svt` `tune=3`(IQ) / `tune=4`(MS-SSIM)** | `enable_qm=true`, qm 4/10 **luma and chroma**, `sharpness=7`, `enable_variance_boost=true`, strength `3`, curve `2`; **IQ only:** `max_tx_size = if qp<=45 {32} else {64}`, `screen_content_mode=Some(3)` (`hdr_mode.rs:357-374`, called unconditionally at `pipeline.rs:1387`) |

Crossing `tune` with `{qm*, sharpness, variance_boost*, deltaq-mode, cdef, chroma-deltaq, max_tx_size, screen-detection}` **aliases those factors into `tune`** and silently collapses cells.

Two further wrinkles, each its own trap:
- **libaom: ORDER decides.** `handle_tuning()` runs at the moment tune is set — from `AOME_SET_TUNING` ([`:1980-1988`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L1980-L1988)) *and* from the `aom_codec_set_option()` string path ([`:4477-4480`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L4477-L4480)). Set `sharpness` **before** `tune=iq` and it is overwritten; set it **after** and it wins. **The harness's option ordering silently changes the encode.** (Contrast the ALL_INTRA defaults, applied in `encoder_init` *before* any control call, so a later control always wins.)
- **SVT: the mutation is STICKY.** `apply_tune_overrides(&mut self)` only ever *sets*. Reusing one `EncodePipeline` and moving `tune` 3 → 0 leaves `enable_qm`, `sharpness=7`, `variance_boost=true`, strength 3, curve 2 **still set from the IQ cell**. `pipeline.hdr` also reads back different values than were written.
  *DoE action:* **construct a fresh `EncodePipeline` per cell**, and make `tune` the OUTERMOST axis on both backends.

### H-5 — **the aom-rs arm's `encode_ms` is not a port speed measurement**
`zenmetrics/crates/zenmetrics-cli/src/sweep/encode.rs:1170-1330`. Every aom-rs
cell (a) runs the **C libaom oracle** (`cell.c_encode_defaults()`, `:1237-1240`),
(b) runs the Rust port (`cell.port_encode_with`, `:1284`), (c) **requires the two
frame-OBU payloads to be byte-identical** and errors out otherwise (`:1286-1295`),
then (d) splices and muxes. The `Instant` starts at `:1196`, before all of it.
**So `encode_ms` ≈ colour-convert + C encode + Rust encode + verification + mux** —
it is a *verification-harness* time, not an encoder time, and it is **not
comparable to the svt-rs arm**, which times one encoder (`:1005-1013`).
*DoE action:* a speed model must not be fit on aom-rs `encode_ms` as-is. Either
time the port call alone or model the arms separately with the composition
declared.

### H-6 — screen-content detection is a content-dependent cost and correctness cliff
- **Cost:** on aom-rs the port's IntraBC DV search dominates on screen-detected
  frames; the harness mirrors the oracle's detection into
  `ToggleKnobs{enable_palette: screen, enable_intrabc: screen}` (`encode.rs:1281-1284`)
  and **DEFERS** screen-detected frames above `ZEN_AOMRS_MAX_SCREEN_MP` (default 16 MP,
  `:1266-1279`). The measured sweep already attributes **88% of aom-rs cost to speeds 0-1**
  and names screen-content detection as the driver
  (`zenmetrics/benchmarks/avif_sweep_subsample_2026-09-01.md:238,255`).
- **Correctness:** on `zenav1-svt`, QP 0 (coded-lossless) is **refused when
  `allow_screen_content_tools` is set** (`pipeline.rs:1077`) — and that flag is
  *derived from frame statistics* (`sc_detect.rs:467`), not from config. **A lossless
  cell succeeds on photographic content and fails on screenshot-like content at
  identical config.**
*DoE action:* stratify by screen-detected vs not, using the detector's own verdict —
never assume a config-only cell is homogeneous.

### H-7 — knobs that are inert in still-picture mode (do not spend cells)
- **libaom `--arnr-maxframes` / `--arnr-strength`** — alt-ref temporal filtering; there is no alt-ref for a single still.
- **SVT `--enable-tf` / `--enable-kf-tf` / `--tf-strength`** — in `zenav1-svt`, `speed_config.enable_temporal_filter` is gated `&& !is_key` and inter frames are refused (`pipeline.rs:1462`), so it is *unreachable*; `hdr.tf_strength` and `hdr.kf_tf_strength` have **zero consumers**.
- **SVT `--aq-mode`** — the port's refusal note argues C's aq-mode deltaq is TPL-gated and inert on a single still (`rc_aq.c:899`).
- **SVT `thread_count`** — byte-inert; with one tile there is nothing to parallelise.
- **SVT multi-frame family** (`--lookahead`, `--pred-struct`, `--hierarchical-levels`, `--scd`, `--irefresh-type`, `--enable-overlays`, S-frames, superres non-KF denominators) — structurally meaningless for one frame.

### H-8 — SVT VBR/CBR silently ignores both `qp` and `target_bitrate` on a still
`assign_picture_qp` (`rate_control.rs:271-293`) takes `state.qp` as the base in
the Vbr/Cbr arm. On frame 0, `total_frames == 0` ⇒ `delta == 0`, and
`temporal_layer == 0` ⇒ `layer_delta == 0`, so the QP used is
`RcState::default().qp` = **30**, regardless of what was configured. **It does not
error.** `AvifEncoder` dodges this by hardcoding `RcMode::Cqp` (`avif.rs:318`); a
direct pipeline caller does not.
*DoE action:* pin `RcMode::Cqp` (or `Crf`); never sweep Vbr/Cbr.

### H-9 — defaults that are traps
- **`EncodePipeline.chroma_420` defaults to `false` = MONOCHROME** (`pipeline.rs:395`). A pipeline built without setting it encodes greyscale, and `encode_frame_420` then refuses.
- **`AvifEncoder::with_bit_depth` silently coerces**: anything not 10 or 12 → **8**, no error (`avif.rs:208-212`); 12 is accepted here and only refused at encode time (`avif.rs:589`).
- **`AvifEncoder` returns raw AV1 OBUs, not an AVIF file** (`avif.rs:363-365`, `:429-433`) — no ISOBMFF/`meta`/`ispe`/`colr`/alpha. `lib.rs:23-24` calling it "a complete AVIF file" is stale. `encode_to_av1_obu` is **monochrome-only** despite the general name.
- **`encode_y8` pads to a multiple of 64 with edge replication** (`avif.rs:390-406`) while reporting the caller's true dimensions (`:419-420`) — **a size sweep on `encode_y8` is not measuring the size it thinks.** `encode_yuv420` has no such gap.

### H-10 — two release-mode panics on `pub` fields guarded only by `debug_assert`
`var_boost.rs:121` is `const STRENGTHS: [f64; 5]`, indexed `STRENGTHS[strength as usize]`
(`:129`, `:231`) behind `debug_assert!((1..=4).contains(&strength))` (`:101`, `:200`) —
**which vanishes in release**. Same shape for `variance_octile`:
`mid_idx = octile * SUBBLOCKS_IN_OCTILE - 1` (`var_boost.rs:108`) underflows at 0 and
indexes OOB above 8. `knob_config_error` checks neither. `AvifEncoder` clamps strength
(`avif.rs:248`); a pipeline caller is unguarded.
*DoE action:* clamp `variance_boost_strength` to 1..=4 and `variance_octile` to 1..=8
**in the harness**. Also note strength **3 and 4 saturate to the same plan** (`avif.rs:806`),
so that is 3 distinct levels, not 4.

### H-11 — SB128 is signalled but not searched (SVT)
`sb128_encode_supported` returns unconditional `true` (`pipeline.rs:441-449`), but
`:437-440` records that a genuine 128-level NONE/HORZ/VERT RD search is **unported** —
the 128 root is forced-SPLIT. Sweeping `sb_size_override` varies signalling and CDF
context, **not partition quality**; do not read a size delta there as a partition result.

### H-12 — build-flag-gated libaom tunes
`AOM_TUNE_VMAF_*` (4-7) need `-DCONFIG_TUNE_VMAF=1`; `AOM_TUNE_BUTTERAUGLI` (8) needs
`-DCONFIG_TUNE_BUTTERAUGLI=1` ([`:875-890`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L875-L890)).
**`AOM_TUNE_IQ`(10) and `AOM_TUNE_SSIMULACRA2`(11) are NOT gated** and are always
available at the pin. Neither VMAF nor Butteraugli is reachable from the pure-Rust
port (`#![forbid(unsafe_code)]`, no metric deps) — treat both as out of scope.

### H-13 — `--enable-chroma-deltaq=1` hard-errors with `--lossless=1` (libaom)
*"Only --enable_chroma_deltaq=0 can be used with --lossless=1."*
([`:905-906`](https://github.com/imazen/libaom-mirror/blob/03087864cf4bea6abb0d28f95cf7843511413d8f/av1/av1_cx_iface.c#L905-L906)).
Since `tune=iq` **forces** `enable-chroma-deltaq=1`, the combination
`tune=iq` + `lossless=1` is a **guaranteed failure** unless chroma-deltaq is re-set
to 0 *after* the tune. Exclude that corner or order the options deliberately.

### H-14 — the two backends' `sharpness` are different scales
SVT: `[-7..7]`, default 0. libaom: **`0..7` unsigned**, default 0. A shared
"sharpness" column across backends is **not a common factor**; and the port's SVT
`sharpness` is an `i8` clamped to `0..=7` at use (`pipeline.rs:2091`), so negatives
floor to 0 rather than doing anything.

### H-15 — CICP / colour knobs are pinned by the harness, not free
The aom-rs path hardcodes Bt709 / sRGB / Bt601-matrix / full-range in the muxer
(`encode.rs:1322-1327`) and feeds `zenyuv` Bt601-full (`:1200`). Colour/CICP is
therefore **not a sweepable axis** in the current harness on that arm, and any
metric computed against a differently-tagged reference inherits that convention.

---

## 10. PORT-MISSING knobs worth exposing, in priority order

Each row: what it buys the DoE, and how deep the wiring looks based on what is
already in the tree.

| # | backend | knob | why it is worth it | difficulty |
|---|---|---|---|---|
| 1 | **aom** | **`tune` (IQ / SSIMULACRA2) as a `ToggleKnobs` field** | The whole family is already ported and byte-gated (`encoder_gate_tune_iq_e2e.rs`, PARITY.md C4) and `aom_encode::TuneKnobs` is public — it simply is not reachable from the harness the sweep drives. It is also libavif's *default* for colour, so today's aom-rs cells are measuring a configuration the ecosystem does not ship. | **Shallow-to-medium**: add `tune` to `ToggleKnobs`, thread into `TuneKnobs` + the QM/sharpness/cdef/chroma-deltaq/deltaq settings, and drive the C side with tuning-first ordering (the e2e gate already shows how). The byte-identity gate will police it. |
| 2 | **aom** | `--sharpness`, `--enable-cdef`, `--enable-chroma-deltaq`, `--deltaq-mode=6` as first-class fields | These are the individually-addressable pieces of the tune-IQ bundle. Exposing them *separately* is what lets the DoE de-alias hazard **H-4** and learn which component actually pays — impossible if `tune` is the only handle. | **Medium** — same plumbing as #1, plus a `c_ctrls` id for each so both sides can be driven. |
| 3 | **svt** | `--enable-cdef` | SH bit is **hardcoded `true`** (`entropy/obu.rs:754`); the search is preset-gated with no override. libaom's all-intra default disables CDEF outright for images, so this is a directly testable cross-backend hypothesis SVT currently cannot answer. | **Shallow** — add a config field and gate the search predicate. (It was pruned as a dead field in `99d0f606b`, so the name is free.) |
| 4 | **svt** | `--enable-restoration` | Preset-derived only (`speed_config.rs:153`). libavif disables restoration for alpha; its value for colour stills at high quality is untested here. | **Shallow** — `seq_tools_for_preset` already computes it; add an override. |
| 5 | **svt** | `--enable-palette` | Derived from `sc_detect`; a diagnostic env override (`SVTAV1_SC_TOOLS`) already exists, so the plumbing is proven — it just is not a config field. | **Shallow.** |
| 6 | **aom** | `coeff_cost_upd_freq` / `mode_cost_upd_freq` **port-side gate** | Currently present-but-inert: C ctrls are emitted, the port ignores them. Any sweep over them today produces *false nulls* — worse than not having the knob. | **Medium** — the field's own doc carries a precise HANDOFF describing the required `pack.rs` split. |
| 7 | **svt** | `--fast-decode` (SB-size half) | The field exists (`sb128_geom.rs:63`) but `SbSizeInputs` is built with `..Default::default()`, so it is permanently false. Decode speed is a real product axis. | **Trivial for SB-size**; deep for its CDEF/LR effects (`cdef.rs:212` notes only the `fast_decode == 0` branch was transcribed). |
| 8 | **svt** | `--aq-mode 1` (segmentation VAQ) | `segmentation.rs` is 56 KB, capability-complete and C-parity-tested, but unwired — the five FH writer sites emit hardcoded `segmentation_enabled = 0`. | **Deep** — 5 writer sites + a config surface. Low priority: variance boost already covers the still-image AQ need. |

**Not worth exposing:** SVT `--tune 5` (VMAF) — needs the metric; VMAF/Butteraugli
tunes on aom — build-flag gated upstream and unported; anything multi-frame.

---

## 11. Where the port contradicts upstream docs (port source wins)

| # | contradiction | resolution |
|---|---|---|
| **C-1** | Upstream v4.2.0 `--tune 5` = **VMAF**. `zenav1-svt` `tune.rs:27-32` defines slot 5 as **`TUNE_FILM_GRAIN`**, with the comment *"Fork tune values (definitions.h:1919, fork enum — slot 5 is FILM_GRAIN in the fork; mainline v4.2's slot 5 is VMAF, not modeled)"*. | **The port models the fork enum.** `tune=5` in this port is not upstream's `tune=5`. Do not sweep it as VMAF; treat slot 5 as unsupported. |
| **C-2** | `zenmetrics/benchmarks/avif_sweep_subsample_2026-09-01.md:151` and `zenavif/src/encode_plan.rs:158` both say the svt-rs speed dial maps "linearly to SVT presets 0..=13"; `zenav1-svt`'s own `avif.rs:14` and `:78` say "Mapped to svtav1 presets 0-13". | **False.** `.min(9)` in `speed_to_svt_preset` caps at preset 9, and integer division skips 2, 5, 8 — **7 distinct presets, speeds 7-10 identical** (hazard **H-2**). The `.min(9)` is *correct* (C's all-intra M9 remap); the prose is stale. |
| **C-3** | `zenav1-svt` `lib.rs:23-24` calls `AvifEncoder`'s output *"a complete AVIF file"*. | **False.** Every entry point returns **raw AV1 OBUs** (`avif.rs:363-365`, `:429-433`) — no ISOBMFF/`meta`/`ispe`/`colr`/alpha. (zenmetrics muxes separately with `zenavif_serialize`.) Also `encode_to_av1_obu` is **monochrome-only** despite the general name. |
| **C-4** | `zenav1-svt` README `:126-127` and the bd10 error strings at `pipeline.rs:1173`/`:1270` require *"64-aligned dims"*. | **Stale.** `pipeline.rs:869-875` records *"NO GEOMETRY TERM (2026-08-04) … Both are gated per-CELL below rather than by dimension"*. The predicate no longer enforces it; the messages and README were not updated. |
| **C-5** | `--color-format` documents `[0-3]` (yuv400/420/422/444) at v4.2.0, and `zenav1-svt`'s `svtav1::EncoderConfig` offers `Yuv422`/`Yuv444` and 12-bit. | Upstream's own row says *"only yuv420 is supported at this time"*, and in the port those enum variants live on the **dead scaffold** (`lib.rs:94-96`) which encodes nothing; the real path refuses anything but 4:2:0 (`avif.rs:577-581`). **4:2:0 8/10-bit is the entire supported space.** |

---

## 12. UNVERIFIED — do not build on these without checking

| # | claim | why it is unverified | how to settle it |
|---|---|---|---|
| **U-1** | Every SVT **upstream** range/default in §2. | Read from `Docs/Parameters.md` at the v4.2.0 tag, **not** from the C header — `reference/svt-av1/` is not checked out on this machine. Upstream docs have been wrong before (the v4.0.1 changelog itself lists *"Fixed tf-strength's default value in the help output"*). | `git submodule update --init reference/svt-av1` and read `API/EbSvtAv1Enc.h` + `Source/App/EncApp/EbAppConfig.c`. |
| **U-2** | That libavif's `svt_config->aq_mode = 2` is a no-op for a single still, i.e. that `zenav1-svt` refusing `aq_mode != 0` costs nothing versus what the ecosystem ships. | This is the **port's own argument** (`pipeline.rs:927-934`, citing `rc_aq.c:899` as TPL-gated) — asserted, not measured here, and the cited C line could not be read (U-1). | Encode the same still with C SVT-AV1 v4.2.0 at `--aq-mode 0` vs `--aq-mode 2` and diff the bytes. If they differ, the refusal is a real capability gap, not a no-op. |
| **U-3** | The svt-rs `speed_to_svt_preset` table's *effect* — that speeds 7-10 produce byte-identical output. | The **mapping** is verified (arithmetic on the literal source, two independent copies). That equal presets ⇒ equal bytes also needs the rest of the config to be equal, which it is in the current harness — but this was not executed. | Encode one image at speeds 7,8,9,10 and compare sha256. Cheap, and it directly validates or kills 30% of the existing svt-rs speed axis. |
| **U-4** | The composition of aom-rs `encode_ms` (**H-5**). | Read from the code path (`Instant` at `encode.rs:1196`, oracle at `:1237`, port at `:1284`, verify at `:1286`), not measured. The *ratio* of C-oracle to port time is unknown. | Instrument the two calls separately, or compare against a port-only encode. Required before any speed model uses that column. |
| **U-5** | Whether SVT `--sharpness`'s upstream domain is genuinely signed `[-7..7]`. | The doc says `[-7-7]`; the port's field is `i8` but **clamped `0..=7` at use** (`pipeline.rs:2091`), so negatives are indistinguishable from 0. Cannot confirm upstream's intent without the C header (U-1). | Same as U-1; then decide whether the port's clamp is a bug. |
| **U-6** | The `zenav1-aom` `ToggleKnobs` defaults transcribed in §5.1 are the port's, and each doc comment names the upstream C default. **I did not independently verify each stated C default against `upstream/`.** | Transcription risk on ~30 fields; the ones I *did* check against upstream (`enable_palette` 1, `enable_intrabc` 1, `min/max_partition_size` 4/128, `enable_tpl_model` 1) agree. Note `ToggleKnobs`'s **own** defaults for `enable_palette`/`enable_intrabc` are `false`, deliberately unlike C's `1`. | Cross-read each against `upstream/av1/av1_cx_iface.c` `default_extra_cfg`. |
| **U-7** | Nothing in this dossier was executed. **No encode was run, no byte was compared, no timing was taken.** | This lane is research + source-verification only, by directive. | Everything above is a hypothesis with a source; the DoE lane's first job is the smoke tests named in U-2, U-3, U-4. |

---

## 13. Provenance

Upstream artifacts were fetched raw (not summarized) and parsed literally:
`Docs/Parameters.md` at tags **v4.2.0, v4.1.0, v4.0.0, v3.0.0** (the cross-tag
knob diff behind **H-1** and the version-note column), `CHANGELOG.md` at v4.2.0,
and the appendices (Variance-Boost, CommonQuestions, Palette, Intra-Block-Copy,
Rate-Control, CDEF, Restoration, Film-Grain). libaom was read from the **pinned
submodule on disk** at `03087864` (= v3.14.1). libavif was read at **v1.4.2**.
Port claims cite absolute paths + line numbers in the working trees at the
commits noted in §1.

Repos were claimed with `.workongoing` (`claude-knobdocs`) for the read;
`zenavif` and `zenmetrics` carried a concurrent `claude-permretrofit` marker and
were therefore read **strictly read-only**, with no marker written and nothing
modified. **⚠ That lane is auditing and retrofitting the AVIF sweep's knob
permutations right now** — the harness facts in §6 are a snapshot of
`zenmetrics e91a03b4` (this commit's parent) and may move under you. Re-check §6
against the tree before designing on it.
