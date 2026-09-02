# Bit-depth capability matrix — encode, decode, score (2026-09-02)

**Lane:** `claude-bitdepth`, the CAPABILITY layer under the HBD arm. Companion to
[`avif_hdr_arm_plan_2026-09-02.md`](avif_hdr_arm_plan_2026-09-02.md) (which owns the
ARM: corpus, T1/T2 declarations, gates G0–G5). That doc asks "what can this arm run?";
this one answers "what is actually wired, and what silently isn't".

**Every cell is VERIFIED (read from source at the cited line, and where marked
MEASURED, run) or BLOCKED with the reason.** Nothing here is inferred from a
config field's existence — the whole point of the lane is that a `bd` field
being present is not evidence that a depth is honoured.

---

## 0. The three defects this lane closed, in one line each

| # | defect | where | fix |
|---|---|---|---|
| **b** | `AvifEncoder::with_bit_depth` mapped every depth but 10/12 to **8, silently** | `zenav1-svt` `rust/svtav1/src/avif.rs:207` | store verbatim; the existing encode-time guard now refuses it as a typed error — `6fe01232` |
| **a** | the sweep hardcoded `bd: 8` **and** an 8-bit container depth, so zenav1-aom's real HBD path was unreachable | `zenmetrics` `crates/zenmetrics-cli/src/sweep/encode.rs` | a `bd` knob (8/10/12), refused BY NAME outside the byte-verified speed band, with bit-replication promotion and a bitstream read-back gate — `0155c165` |
| **d** | the FLEET scored HDR through a u8 shell that erases **94.17 %** of a 10-bit-vs-8-bit difference | `zenmetrics` `crates/zenmetrics-cli/src/hdr.rs`, `src/main.rs` | route `score-pairs` / `batch` / jobexec through the umbrella's validated f32 feeding — `7051921a` |

---

## 1. ENCODE

Columns are the depth the **coded stream** carries. "source" is what the harness can
feed: every SDR sweep cell funnels through `Rgb8Image { pixels: Vec<u8> }`, so an
8-bit source at a 10-bit coded depth is the standard AVIF production trick, not a
high-bit-depth *input* path.

| backend | 8 | 10 | 12 | evidence |
|---|---|---|---|---|
| **zenav1-aom** (sweep, direct) | **VERIFIED** | **VERIFIED**, speeds {0,7,8,9} only | **VERIFIED**, speeds {0,7,8,9} only | MEASURED: `sweep::encode::aom_rs_depth_tests::emitted_bitstream_carries_the_requested_depth` encodes 64×64 at 8/10/12 (speed 9) and reads `av1C` back — three pairwise-different blobs, each declaring its requested depth |
| — outside {0,7,8,9} | n/a | **BLOCKED, refused by name** | **BLOCKED, refused by name** | zenav1-aom pins `--cpu-used` 1..=6 DIVERGENT vs the C libaom v3.14.1 oracle for bd10 **and** bd12 (`config_permutations.rs` `b10_64`; enforced by the un-ignored `speed_envelope_stock_map_is_pinned`; coverage entry `HBD_OPEN`, luma-borne, unlocalized) |
| **zenavif → zenrav1e**, 8-bit source | **VERIFIED** | **VERIFIED** | **BLOCKED at zenavif** | ravif supports it: `ravif/ravif/src/av1encoder.rs:79` `BitDepth::Twelve`, `:1105-1111` / `:1191-1199` / `:1231` → `:1445 encode_raw_planes_12_bit`. zenavif cannot express it: `EncodeBitDepth` (`zenavif/src/encoder.rs:73-82`) has no `Twelve` — see §3 |
| **zenavif → zenrav1e**, 16-bit source | **BLOCKED — silently 10** | VERIFIED | BLOCKED | `encode_rgb16` (`zenavif/src/encoder.rs:1449`) hardcodes `scale_from_u16(.., 10)` + `encode_raw_planes_10_bit` + `MatrixCoefficients::Identity` and **never reads `config.bit_depth`**. Same in `encode_rgba16`. **NEW DEFECT — see §2** |
| **zenavif → svt-rs**, 8-bit source | **VERIFIED** | **VERIFIED** | **BLOCKED, refused** | `zenav1-svt` refuses any depth but 8\|10 at encode (`rust/svtav1/src/avif.rs` `validate_inert_knobs`; `rust/crates/svtav1-encoder/src/pipeline.rs:1172-1176`), matching C v4.2.0's own init check (`svt_av1_verify_settings`, `enc_settings.c:460`). **This is upstream's envelope, not a port gap** |
| — 10-bit, presets < 9, monochrome | | **BLOCKED, refused** | | `pipeline.rs:876-877`; zenavif refuses early (`encoder_svt_rs.rs:509`) |
| — 10-bit at speeds 1..6 | | VERIFIED but **not byte-identical to C** | | `zenav1-svt/rust/STATUS.md:55-58`: bd10 64-aligned p0..p8 **21.5 %** non-flat mismatch, partial-SB **26.3 %**; p9..p13 clean. Measures *the port's* 10-bit, never SVT-AV1's (hazard H-BD-4) |

**Container/muxer** (`zenavif-serialize`): 8/10/12 all **VERIFIED** —
`src/lib.rs:779-780` `high_bitdepth`/`twelve_bit`, `:788-790` `pixi`, `:640-641`
guard; `try_to_vec` rejects any `depth_bits` but 8/10/12.

## 2. ⚠ NEW DEFECT — `encode_rgb16` / `encode_rgba16` ignore `config.bit_depth`

Found while verifying §3, **not fixed here**, registered so it is not rediscovered.

`zenavif/src/encoder.rs:1449 encode_rgb16` and `:1515 encode_rgba16` scale every
sample with `scale_from_u16(p.*, 10)` and call `encode_raw_planes_10_bit`. They read
`config.color_model`, `config.chroma_subsampling` and `config.bit_depth` **not at
all** — 16-bit input is *always* GBR 4:4:4 at 10 bits. So:

- `EncoderConfig { bit_depth: Eight, .. }` + a 16-bit buffer → a **10-bit** file, no
  error, no warning. The generic zencodec path reaches this for any `Rgb16`/`Rgba16`
  input (`zenavif/src/codec/encoder.rs:225`, `:255`, `:399`, `:411`, `:525`).
- This is the **same defect class as §0's (b)** — a depth request silently coerced —
  living in zenavif today, on a published path.

**Why it was not fixed in this lane:** the honest fix is to *honour* the request, not
to refuse it (`bit_depth: Eight` on a 16-bit buffer is a legitimate "narrow this to an
8-bit AVIF" ask that the generic codec path exists to serve), and honouring it means a
narrowing route plus a 12-bit route plus the Identity/GBR interaction, in a **published
crate (zenavif 0.1.7)** with a pinned `tests/identity_roundtrip.rs`, while a
concurrent lane was working in that repo. Refusing instead would remove a real
capability. Registered rather than half-done.

**It does not affect the HBD arm's T2**, which wants exactly the 10 bits this path
hardcodes (`zenmetrics` `sweep/hdr.rs:391-396`).

## 3. `EncodeBitDepth::Twelve` — verified out of scope, with the upstream facts

The brief asked for the variant **only where a backend genuinely supports it end to
end**. Measured, that condition holds for exactly one of three combinations:

| combination | 12-bit? | fact |
|---|---|---|
| zenrav1e, 8-bit source | **YES** | `av1encoder.rs:79`, `:1105-1111`, `:1191-1199`, `:1231`, `:1445` |
| zenav1-svt | **NO — refused upstream** | C v4.2.0 rejects any depth but 8/10 at encoder init; the port matches |
| zenavif `encode_rgb16` | **NO — would be silently 10** | §2 |

So adding the variant today would ship a knob that is honest on one entry point and
**lies on another** — precisely the defect this lane was sent to remove. It would also
be a **semver break**: `EncodeBitDepth` is not `#[non_exhaustive]`
(`zenavif/src/encoder.rs:72-74`), so a new variant breaks every downstream exhaustive
match — on a published 0.1.7 crate that is a leading-non-zero-digit bump to 0.2.0,
which the workspace rules require to be real, unavoidable and user-approved.

**The change set, if it is ever wanted** (registered, not started):

1. `EncodeBitDepth::Twelve` + `#[non_exhaustive]` on the enum in the same break.
2. `resolve_bit_depth`: `Twelve => ravif::BitDepth::Twelve` (`encoder.rs:1009`).
3. `effective_bit_depth` (`encoder_svt_rs.rs:477`): `Twelve => 12`, and
   `reject_out_of_envelope_depth` must refuse it EARLY with the C-envelope reason,
   rather than letting the pipeline refuse it deep.
4. §2 first — otherwise `Twelve` is silently 10 on the 16-bit entries.
5. A read-back gate per backend, from the bitstream (§4), never from the request.

## 4. DECODE

| | 8 | 10 | 12 | evidence |
|---|---|---|---|---|
| `rav1d-safe` | VERIFIED | **VERIFIED** | **VERIFIED** | `src/lib.rs:509-510` → `BitDepth16`; `Cargo.toml:74` both default |
| `zenavif` → pixels | VERIFIED | **VERIFIED, u16 out** | VERIFIED | `frame_convert.rs:153-154`; `:160-162` narrows **only** if `prefer_8bit`, and `config.rs:68` defaults it false — truncation is opt-in |
| `zenav1-aom` | VERIFIED (169 vectors) | VERIFIED (66 vectors) | **PARTIAL** | zero conformance vectors carry 12-bit (`README.md:65`); covered only by port-generated C-oracle gates |

**Depth read-back is a solved problem — use the existing tool, do not re-derive.**
`zenmetrics` `crates/zenmetrics-cli/examples/avif_depth_verify.rs` (gate G3) reads a
stored blob's depth by three independent routes — `av1C` box, AV1 sequence header,
decoder `ImageInfo` — and fails when they disagree. The aom-rs read-back test added by
this lane calls the same R1 owner (`zenavif_parse::AvifParser::av1_config()`) rather
than parsing boxes itself.

## 5. SCORE — the depth-honesty of each route

| route | ssim2 / zensim / iwssim / butter-CPU | evidence |
|---|---|---|
| `score --hdr` | **f32 end to end** (always was) | `main.rs` `cmd_score` tries `score_via_hdr_scorer` for every metric |
| `sweep --hdr` | **f32 end to end** (always was) | umbrella `HdrScorer` |
| `batch --hdr` | **f32 — FIXED `7051921a`** | was butteraugli-gpu only |
| `score-pairs --hdr` | **f32 — FIXED `7051921a`** | was cvvdp-gpu + butteraugli-gpu only |
| **jobexec (the fleet)** | **f32 — FIXED `7051921a`** | `hdr::score_hdr_pair_per_score_pairs` |
| SDR (no `--hdr`) | 8-bit by contract | `decode.rs:16-24` — unchanged, and structurally unreachable from the fix |
| **zensim FEATURE sidecar** | **STILL the v1 PU21 u8-shell regime** | REGISTERED, not changed: trained models consume those vectors, so the regime is a DATA decision. The scalar score now follows the feature path's regime for that invocation, so a sidecar row can never disagree with itself |

### Why it mattered — MEASURED, not argued

`crates/zenmetrics-cli/tests/hdr_depth_sensitivity.rs` builds a 192×192 PQ ramp from
10-bit codes and its own 8-bit quantization — real banding, constructed in code space
rather than tuned:

| | value |
|---|---|
| f32 samples that differ | **110 315 / 110 592 (99.75 %)** |
| u8-shell bytes that differ | **6 434 / 110 592 (5.82 %)** |
| **difference the shell erases** | **94.17 %** |

Pre-fix, all four metrics returned values **bit-identical** to the shell's. Post-fix,
deviation from the identity score, shell → f32:

| metric | u8 shell | faithful f32 | ratio |
|---|--:|--:|--:|
| ssim2 | 4.022 | 6.631 | **1.65×** |
| zensim | 5.564 | 6.711 | **1.21×** |
| iwssim | 7.258e-4 | 9.359e-4 | **1.29×** |
| butteraugli | 0.3858 | 1.1697 | **3.03×** |

### Cost, MEASURED

Paired arms in ONE binary, release, min of 7, on a loaded box — read the ratios, not
the absolutes. Sizes per the sweep discipline (tiny → large):

| size | ssim2 shell → f32 | zensim shell → f32 |
|---|---|---|
| 64² | 0.223 → 0.560 ms (2.51×) | 0.258 → 0.346 ms (1.34×) |
| 256² | 4.319 → 10.089 ms (2.34×) | 1.249 → 2.096 ms (1.68×) |
| 1024² | 76.704 → 171.067 ms (2.23×) | 11.701 → 28.986 ms (2.48×) |
| 2048² | 372.358 → 724.223 ms (1.94×) | 81.501 → 142.131 ms (1.74×) |

**≈2× per pair, at every size.** That is the price of being able to see banding.

### Two consequences that must not be rediscovered

- **⚠ SCORING-ERA BREAK.** Every stored `--hdr` ssim2/zensim/iwssim/butteraugli-CPU
  number from `score-pairs`, `batch` or jobexec before `7051921a` was computed on the
  u8 shell. **Do not join across it.**
- **⚠ `--hdr-transfer` is now INERT** on those metrics: the faithful route takes
  absolute nits and PU21-encodes in-kernel, so there is no transfer shell to select.
  Pinned by `hdr_pair_parity::hdr_transfer_is_inert_on_the_faithful_route`; the shell's
  own transfer sensitivity is still pinned, re-pointed at the shell primitives.

## 6. What "least sufficient bit depth" still needs

| # | need | state |
|---|---|---|
| 1 | 10/12-bit **encode** reachable as a knob | **DONE** for zenav1-aom (§1). No plan vocabulary declares `bd` yet — a wave must put it in its grid |
| 2 | depth verified from the **bitstream** | **DONE** — `avif_depth_verify` (G3) + the aom-rs read-back gate |
| 3 | scoring that can **see banding** | **DONE** for the scalar metrics on every route (§5) |
| 4 | a high-bit-depth **SOURCE** through the SDR sweep lane | **BLOCKED** — the `Rgb8Image` funnel (plan §2.5); every depth cell today is 8-bit content at a deeper coded depth |
| 5 | 16-bit SDR source for zensim + fast-ssim2 | **BLOCKED** — plan TODO-5; both libraries already resolve 16 bits, only the CLI funnels to u8 |
| 6 | banding-VALID references | **BLOCKED** — plan §3.4: the wired HDR corpus is gain-map-reconstructed from 8-bit bases; the only never-8-bit-quantized pixels are the 30 UPIQ EXRs (plan TODO-2) |
| 7 | zensim HDR **feature** vectors at f32 | **REGISTERED** (§5) — a data decision, not a routing one |
| 8 | `EncodeBitDepth::Twelve` | **REGISTERED** (§3) — blocked on §2 and a semver break |

Items 4–6 mean a depth *picker* cannot be trained honestly yet: the encode and the
score are now depth-honest, the **corpus** is not.
# PENDING APPEND — zenmetrics/benchmarks/bitdepth_capability_matrix_2026-09-02.md

**Why this is here and not in the file:** at 2026-09-02T16:35Z the zenmetrics
`.workongoing` marker read `claude-hbdexec T2: image rebuild from clean master,
G5 discriminator, declare` (15 min stale) AND `jj status` showed a foreign
uncommitted change to `benchmarks/avif_hdr_arm_plan_2026-09-02.md` in `@`.
Because jj snapshots the whole working copy, any commit made in that repo would
have swept that lane's in-flight file into it. Blocked on the concurrent lane,
not on the content. **Apply this append verbatim when the marker is free.**

Two edits, both additive.

---

## EDIT 1 — append to §2, after the "Why it was not fixed in this lane" paragraph

**✅ FIXED 2026-09-02, zenavif `cc17cb86` (on `origin/main`).** The fix HONOURS
the request rather than refusing it, as this section proposed:
`EncodeBitDepth::Eight` on a 16-bit buffer now takes a narrowing route to an
8-bit coded stream. `Ten` and `Auto` are untouched — `Auto` keeps its documented
"16-bit input → 10-bit AV1" contract, and its output is **byte-identical**
pre/post (rgb16 + rgba16 × {Auto, Ten}, four files, sha256 unchanged, built from
the parent commit's sources and diffed).

Mechanism, for the record: `encode_raw_planes_*` takes the coded depth as an
*argument*, which overrides the encoder's own `with_bit_depth`.
`build_ravif_encoder` had already computed the right answer via
`resolve_bit_depth(config.bit_depth, true)` — it was discarded one line later.
The svt-rs backend never had the bug (`encoder_svt_rs::effective_bit_depth`);
this levels the zenrav1e path with it.

Narrowing uses zenavif's existing owner `convert::scale_from_u16(v, 8)` (wrapped
as `convert::narrow_to_u8` for the `[u8; 3]` plane API) — no new conversion rule
was invented. It is the exact inverse of the widening rule (`scale_to_u16`, LSB
replication), so 8-bit content promoted to 16 bits round-trips to the original
bytes, and it matches the decode side's `downscale_to_8bit` ("high byte of each
channel"). Half-up rounding was rejected on MEASUREMENT: it leaves the u8 domain
at `0xFFFF` (→ 256) and corrupts the 8→16→8 round-trip for **128 of 256 bytes**.

Gate: zenavif `tests/bit_depth_request.rs` — 5 tests, depth read back from the
**bitstream** via `zenavif::detect::probe` (the R1 owner `zenavif_parse`), never
from the request, per §4's rule. 3 of the 5 fail before the fix with
`left: 10, right: 8`. Plus `convert::narrow_16_to_8`, 3 unit tests, one
exhaustive over the whole u16 domain. No public API change (`cargo public-api`
diff empty across 1,834 items).

**No stored data is affected.** Verified by grep, not assumed: nothing outside
zenavif calls `zenavif::encode_rgb16`/`encode_rgba16` (the one hit in
`sweep/hdr.rs:391` is a comment; `:926` is `zenpng::encode_rgb16`, a different
crate), no `EncodeBitDepth::Eight` appears anywhere in `zenmetrics/crates/`, and
the sweep funnels through `Rgb8Image`. So no sweep cell ever reached the
defective path, consistent with this section's own "does not affect the HBD
arm's T2".

---

## EDIT 2 — replace the §6 item-8 row

| 8 | `EncodeBitDepth::Twelve` | **REGISTERED** (§3) — §2 is now CLEARED (fixed 2026-09-02, zenavif `cc17cb86`); the remaining blocker is the semver break alone |

Rationale: §3's change set lists "§2 first — otherwise `Twelve` is silently 10 on
the 16-bit entries" as prerequisite 4. That prerequisite is met. `Twelve` was
NOT added: `EncodeBitDepth` is not `#[non_exhaustive]`, so the variant is a
0.1.7 → 0.2.0 break the workspace rules require to be real, unavoidable and
user-approved. It is queued in zenavif's CHANGELOG under
`QUEUED BREAKING CHANGES`, paired with `#[non_exhaustive]` so both ship in one
break, and carrying §3's upstream facts (zenrav1e supports 12 end to end;
zenav1-svt refuses it at init, matching C v4.2.0, so
`reject_out_of_envelope_depth` must refuse early with the C-envelope reason).

---

# APPEND 2026-09-02 (imazen-only lane) — the THREE-BACKEND TRUTH TABLE

Written under USER RULE **"IMAZEN-ONLY IMAGING/CODEC SOFTWARE"**
(`~/work/zen/CLAUDE.md`, 2026-09-02). It was minted after a session wrongly
declared zenav1-aom *"validation-only, not a backend"*; this section is the
ground truth that claim should have been checked against. Everything below is
read from SOURCE on the refs named, not from docs — and §7.4 lists four places
where the repo's own docs disagree with its source.

## 7.1 The table

zenavif carries **two distinct backend enums**, which is the first thing that
makes this easy to get wrong:

- `Av1Backend` (**encode**) — `zenavif/src/encoder.rs:186` — `{ Zenravif, Svtav1
  (deprecated, never shipped), SvtRs }`
- `DecodeBackend` (**decode**) — `zenavif/src/decode_av1.rs:491` — `{ Rav1dSafe,
  AomRs, Rav1dFfi }`

zenavif `main` = `origin/main` = `8f203c1f`. **`default = ["avx512"]` only — no
backend feature is default, and `encode` itself is not default.**

| backend | op | ref | feature | default? | seam (file:line) | wiring state | tests |
|---|---|---|---|---|---|---|---|
| **zenrav1e** | ENCODE | `main` | `encode` | no | `src/encoder.rs:1067` `build_ravif_encoder`; default arm of `encode_rgb8_once` (`:1332`) | **fully wired** — `Av1Backend::Zenravif` is `#[default]`; every public encode entry falls through to it | `encode_roundtrip.rs`, `encode_contracts.rs`, `target_quality.rs`, `two_pass.rs` |
| zenrav1e | DECODE | — | — | — | **no seam** | encode-only crate | — |
| **zenav1-svt** | **ENCODE** | `main` | `encode-svt-rs` | no | `src/encoder_svt_rs.rs` (~1300 lines); entries `:814` rgb8, `:965` rgba8, `:1043` rgb16, `:1093` rgba16, `:1147` gray8; dispatched from `src/encoder.rs:1317,1359,1420,1463,1558` | **fully wired, real work** — muxes AVIF in-crate via `zenavif-serialize`; no `todo!()`; out-of-envelope configs get TYPED refusals (`validation.rs:253-330`, `encoder_svt_rs.rs:349,504`); feature off → `Error::Unsupported` (`encoder.rs:1322`) | `tests/svt_rs_backend.rs` (**24**), `cross_backend_decode.rs` svt arms, `validation.rs:384` |
| zenav1-svt | DECODE | — | — | — | **no seam anywhere** | SVT-AV1 is encoder-only upstream | — |
| **zenav1-aom** | **DECODE** | `main` | `aom-backend` | no | raw-OBU `src/decode_av1.rs:513/539` → `:548` `AomRs` arm → `:819`; **container** `src/decoder_managed/aom.rs` (`decode_full_aom:44`, `decode_grid_aom:75`), selected by `config.rs:159`, dispatched `decoder_managed/decoder.rs:203,:248`; animation `animation.rs:211/235`; gain-map `decode_av1.rs:58` | **fully wired to the public API** — stills, alpha, grid, animation, gain-map, 8/10/12-bit, mono. Only row-sink streaming refused (`sink.rs:163`, honest `Unsupported`) | `tests/product_aom_backend.rs` (**8**, incl. `animations_`/`grids_decode_identically_across_backends`), `cross_backend_decode.rs` (**7**) |
| **zenav1-aom** | **ENCODE** | **none — no ref, no branch** | — | — | **NO SEAM EXISTS** | **absent from zenavif entirely** | — |
| *rav1d-safe* (incumbent) | DECODE | `main` | none (non-optional dep) | **yes** | `decode_av1.rs:546/:557`; `decoder_managed/decoder.rs:101` | fully wired; `DecodeBackend::Rav1dSafe` is the config default (`config.rs:69`) | most of the suite |
| *rav1d FFI* (C+asm) | DECODE | `main` | `unsafe-asm` | no | `decode_av1.rs:550` → `decoder.rs:503` | **benchmark arm only** — explicitly NOT accepted by `DecoderConfig::decode_backend` (`config.rs:157`) | `examples/decode_4way_bench.rs` |

## 7.2 What PR #31 actually landed — and a date correction

**Merge commit `a311b427a2f6f35b6d03644ae0f678e9c080bad6`**, `svtav1-rs-backend`
→ `main`, *"feat: AV1 backend seams — svtav1-rs encode + zenav1-aom decode,
unified ssim2 targeting, in-house YUV kernels"*.

> ⚠ **Merged 2026-08-11T22:52Z, NOT 2026-07-13.** 2026-07-13 is `f5c51d9`, the
> *branch* commit that introduced the svt encode seam; the aom decode seam
> followed in `4f18d29` (2026-07-17) and both sat on the branch ~4 weeks. Any
> statement dating these seams "on main in mid-July" is off by a month. (The
> imazen-only rule's own parenthetical carries the 07-13 date; the rule's
> substance — that the backends exist and predate the claim — is unaffected.)

Verified present in the merge commit's tree: the svt encode seam
(`src/encoder_svt_rs.rs` + `Av1Backend::SvtRs`), the aom decode seam
(`DecodeBackend` + `aom-backend`), the in-house YUV kernel family (P1–P8),
backend-generic ssim2 target search, cross-backend byte-identity gates, and
`docs/BACKEND_SUPPORT_MATRIX.md`.

## 7.3 The aom-ENCODE answer: no seam exists, on any ref, in any state

Four independent lines of evidence:

1. **The encode enum has no aom variant on any ref.** `Av1Backend`'s body was
   dumped from `main`, `origin/svtav1-rs-backend`,
   `origin/preserve/2026-07-25-svtav1-rs-backend`,
   `origin/backup/svtav1-rs-backend-pre-rebase-2026-07-23`,
   `origin/abandoned/spike-av1-backends-2026-05-23`, `origin/svtav1`,
   `origin/cooptloop`, `origin/feat/gainmap-decode`. Every ref is
   `{Zenravif, Svtav1}` or `{Zenravif, Svtav1, SvtRs}`. **Never an `Aom`.**
2. **`git grep` across every local + remote ref** for
   `aom_encode|aom-encode|encoder_aom|Av1Backend::Aom|zenav1-aom-encode` returns
   exactly one hit: `Cargo.toml:448`, a **doc comment** — and itself a typo
   (`Av1Backend::AomRs`; the function takes `DecodeBackend`). That one string is
   why a naive grep for "aom encode" looks like a hit.
3. **The encoder crate is never linked.** zenavif depends only on
   `aom-decode = { package = "zenav1-aom-decode", path = "../zenav1-aom/crates/aom-decode" }`
   (`Cargo.toml:105`). The sibling `aom-encode` crate appears in **no manifest on
   any ref**, so no aom encoder symbol is reachable.
4. **The only "aom encode" in the repo is the C reference.**
   `scripts/rd_gap/aom_cell.sh` / `aom_only.sh` shell `aomenc`/`aomdec`
   (`AOMENC="${AOMENC:?set AOMENC to a libaom aomenc build}"`) for RD-gap
   baselines — libaom-the-C-project, not zenav1-aom. **Under the new rule these
   two scripts are themselves a foreign-tool reach; they are RD-gap baselining,
   not tuning-data generation, but they are named here so the next reader does
   not mistake them for a port seam.**

**`abandoned/spike-av1-backends-2026-05-23` is not an aom spike.** Single commit
`669de3c`; it adds `src/backend.rs` (a `pub trait Av1DecoderBackend`) and
`src/backend_ffmpeg.rs` behind `backend-ffmpeg` — an **ffmpeg shell-out hardware
DECODE** probe (VA-API/D3D11VA/DXVA2/CUDA) to price HW decode. Nothing to
salvage for encode.

### TODO — `PREREQ-AOM-STANDALONE` (owner: **zenav1-aom**, not this repo)

Today the ONLY route to zenav1-aom's encoder is `zenmetrics`'s sweep calling
`aom_bench` / `aom-encode` **directly**, bypassing zenavif. That route is
C-bound by construction: every `aom-bench` `port_encode*` takes a C-encoded
`bootstrap` by signature (`zenav1-aom/crates/aom-bench/src/lib.rs:1150,:1176`),
the screen-content decision is read out of the C stream, and the emitted AVIF
splices the port payload into the **oracle's** OBU frame
(`zenmetrics/crates/zenmetrics-cli/src/sweep/encode.rs:1307-1380`). `aom-bench`
is the port's **differential-validation harness**, not a product encoder API.

So two things are needed, in order, and **neither is this lane's to build**:

1. **`PREREQ-AOM-STANDALONE`** — zenav1-aom exposes an encode entry point that
   derives its own sequence + frame header (no `bootstrap: &[u8]`), emitting a
   complete AV1 temporal unit. Owner: zenav1-aom.
2. **A zenavif `Av1Backend::AomRs` encode seam** — a new variant + `encoder_aom.rs`
   + an optional `aom-encode` dep + a feature gate. `src/encoder_svt_rs.rs` is
   the exact template (it is the second encode backend and muxes in-crate via
   `zenavif-serialize`). Owner: zenavif. **Blocked on (1)**; without it a seam
   would just relocate the C dependency.

Until (1) lands, the §1 aom-rs ENCODE rows above describe a **C-bootstrapped**
encoder, and per the imazen-only rule their cells are **port-parity evidence,
not tuning data**. Registered in
`benchmarks/avif_doe_plan_2026-09-01.md` §16.

## 7.4 Corrected basis for the §1 "BLOCKED, refused by name" rows

§1's aom-rs sub-row justifies the bd>8 speed refusal as *"DIVERGENT vs the C
libaom v3.14.1 oracle"*. That is a true fact but the **wrong basis** to state
under the imazen-only rule, and it misreads the port. Corrected, from the port's
own record:

- zenav1-aom does **not** class bd10/bd12 × `--cpu-used` 1..=6 as broken. Its
  coverage queue files that band under tier **T4 — "measured, pinned,
  unlocalized (byte divergences, NO REFUSAL)"** (`zenav1-aom/CLAUDE.md` T4 table,
  row `HBD_OPEN` / `b10_64`; the pinned set is
  `crates/aom-bench/tests/s4cov_qm_axis.rs:380`). The port **encodes** there.
- The refusal is forced **one level up, by the harness**: with the header spliced
  from C, a payload that is not byte-identical to C has no stream to emit **at
  any depth**. It is a harness limit, not a correctness verdict on the port.
- C-parity is therefore **metadata**, never the admission criterion. Pinned by
  `sweep::encode::aom_rs_depth_tests::hbd_outside_the_byte_verified_speed_band_is_refused_by_name`,
  which now asserts the message cites the port-side pin file, names the bootstrap
  constraint, and carries the disclaimer.

## 7.5 Doc-vs-source discrepancies found while building this table

Recorded, not fixed — three of the four are in zenavif, which this lane read
read-only.

1. **⚠ `zenavif/src/decoder_managed/aom.rs:25-27` is stale and contradicts the
   code beneath it.** It reads *"Scope: non-grid stills; grid images and
   animation return honest `Unsupported`…"*. **False**: `decode_grid_aom` is
   implemented at `:75` and called at `:47`; animation is implemented at
   `animation.rs:235` and wired at `:211`; both are pinned by passing tests
   (`product_aom_backend.rs:163`, `:209`). `docs/BACKEND_SUPPORT_MATRIX.md` is
   the accurate account; the module doc is the wrong one. **This is very likely
   the origin of the "validation-only, not a backend" mistake** — it is the
   comment a reader hits first.
2. **⚠ PR #31 merge date** — 2026-08-11, not 2026-07-13 (§7.2).
3. **⚠ `aom-backend` is absent from zenavif's README feature table**
   (`README.md:286-296` lists `encode`, `encode-asm`, `encode-threading`,
   `encode-imazen`, `encode-svt-rs`, `unsafe-asm`, `zencodec` — not
   `aom-backend`), despite it being a wired, public, product-path decode backend.
   Neither README mentions it at all.
4. **⚠ `docs/BACKEND_SUPPORT_MATRIX.md`'s header block is stale** — self-dates
   *"as of 2026-07-23 (branch `svtav1-rs-backend`)"* with rev pins
   (`zenav1-svt 3e25f52b`, `rav1d-safe f9458f43`, `zenav1-aom 7b972e50`), but its
   body has been updated since (cites a 2026-08-29 change and open issue #42) and
   `main`'s real deps differ: `rav1d-safe` is `66f58fa6`, and zenav1-svt +
   zenav1-aom are **path deps, not rev pins**. Body trustworthy, header not.
5. **Trap, not a discrepancy:** zenavif's `Cargo.toml` carries **path deps** for
   both zenav1-svt (`:175`) and zenav1-aom (`:105`), each with an in-file note to
   return to a git-rev pin before landing. A clone without the siblings present
   cannot resolve `encode-svt-rs` or `aom-backend`.

---

# APPEND 2026-09-02 — NON-IMAZEN TOOL SWEEP of the campaign pipeline

Bounded to the campaign paths named in this lane's brief: `zenmetrics/scripts/`,
`zenmetrics/crates/zenmetrics-cli/src/sweep/`, and `zensim/scripts/`. Searched
for magick/convert/identify/mogrify/montage, heif-*, exiftool/exiv2,
ffmpeg/ffprobe, avifenc/avifdec, cjxl/djxl, cwebp/dwebp, cjpeg/djpeg/jpegtran,
aomenc/aomdec, x264/x265, vips, and the Python imaging stacks (PIL, cv2,
imageio, skimage, torchvision.io).

**31 hits: 12 LIVE-IN-PIPELINE · 8 LIVE-BUT-PERIPHERAL · 5 DEAD · 6
MENTION-ONLY.** The Rust sweep dir is **clean** — `sweep/encode.rs` contains no
`Command` at all, and `distort.rs` spawns only the caller-supplied
`--distort-cmd` (default `python3 -m kadis_distort.serve`, imazen).

## 8.1 Fixed in this pass

| file:line | was | now |
|---|---|---|
| `zensim/scripts/run_full_eval.sh:153-170` | `magick`/`convert` for **both** M3 axes — `-filter Mitchell -resize NxN` and `-quality Q` (ImageMagick's libjpeg) — plus a **graceful skip** (`M3_SIZES=(576)` + a warning) when neither binary was found | `zensim-bench/examples/m3_fixture_gen.rs` — **zenpng** decode/encode + **zenresize** `Filter::Mitchell` + **zenjpeg** encode. The skip is now a **fatal exit 3** with build instructions |
| `zenmetrics/scripts/sweep/CLAUDE.md:259-265` | *"libjxl is the authority… always test with `djxl` directly"* — a committed instruction to reach for a C decoder, in the sweep runbook | scoped: the spec is the authority, differential libjxl comparison is **port-repo triage only**, never a sweep-side admission criterion |

**The `run_full_eval.sh` hit was the most serious one found.** M3a is a
first-class **model-selection** input (`docs/WAVE_PLAYBOOK.md` step 6; the
`freeze_check --select` tie-break is `balanced_composite + 0.15·M3a`), so a
foreign JPEG encoder was inside the loop that decides which zensim model ships.
The graceful skip compounded it: with no `magick` on the box the size axis
silently collapsed to 576-only, so **M3a changed meaning without failing**.

**⚠ ERA HAZARD — MEASURED, not asserted.** zenjpeg's `q` is not
ImageMagick-libjpeg's `q`. On `city_384` at identical nominal quality:

| q | ImageMagick-era bytes | zenjpeg bytes | ratio |
|--:|--:|--:|--:|
| 20 | 11,852 | 10,752 | **0.907** |
| 50 | 20,513 | 16,307 | **0.795** |
| 75 | 30,113 | 24,897 | **0.827** |

The Mitchell downscales differ too (`city_384.png` 241,630 B ImageMagick-era vs
241,828 B zenresize; not byte-identical at either size). So a fixture made by
the new owner is a **different rate point**, and an M3a measured on it is not
comparable to any M3a in the record.

Therefore the 48 ImageMagick-era fixtures under
`/mnt/v/output/zensim/diffmap-coherence-2026-07-18/` are **left alone** and the
default `$FIX` still points at them. The rewired loop only fills a **missing**
file. **VERIFIED:** with the generator deliberately absent, the loop needs zero
regenerations (full coverage) and the fixture-set digest is unchanged — **no
published M3/M3a number moves on this commit.** Regenerating means pointing
`ZENSIM_M3_FIXTURES` at a **new era-stamped directory**, same discipline as
`2026-05-15-full-features` vs `2026-08-30-full-features-372`, and expecting the
whole axis to re-base.

## 8.2 Registered TODOs — owner named, not built here

| # | file:line | tool | imazen owner | size |
|--:|---|---|---|---|
| T1 | `zenmetrics/scripts/jobsys/avifdoe_build_budget_corpus.py:375-376` | PIL open/convert/crop/save | **zencodecs** decode + **zenpng** encode | SMALL code, **BIG blast radius** — it builds 13 of the 32 DoE references and `.convert("RGB")` silently normalises depth/palette. Changing it **changes every `crop_sha256` in `crop_manifest_2026-09-01.tsv`**, and the wave is LIVE. Do it at an era boundary, never mid-wave |
| T2 | `zensim/scripts/pu21_single_model_exp.sh:28` | `magick src -strip dst` | **zencodecs** decode + **zenpng** | SMALL. Decodes CID22 **distorted** jpg/webp — a foreign decoder on codec bitstreams under test |
| T3 | `zensim/scripts/hdr/srgb_to_pq_png.py:36,46` | `cv2.imread` | **zencodecs** decode (the hand-rolled PQ numpy beside it is separately owned by **zenpixels-convert**/**zentone**) | SMALL for the decode; feeds the HDR 944 route |
| T4 | `zensim/scripts/v_next/rd_probe_2026-07-18.sh:79` | `mogrify -format png *.ppm` | **zenbitmaps** (PNM) + **zenpng** | SMALL — converts zenjpeg's own PPM output |
| T5 | `zensim/scripts/squintly/crop_holdout_check.py:58` | PIL | **zencodecs** decode | SMALL, but it is a **contamination GATE** — the rule names gates explicitly |
| T6 | 4 dimension-only probes: `zenmetrics/scripts/provenance/index_sources.py:35` (`identify`), `provenance/index_corpus.py:54`, `picker/gen_dense_corpus.py:41`, `zensim/scripts/squintly/mine_adjudication_stimuli.py:385` (PIL) | ImageMagick / PIL | one shared **zencodecs header probe** | SMALL each. `index_sources.py` additionally swallows failures into `(0,0)` — silently wrong provenance rows |
| T7 | `zenmetrics/scripts/jobsys/gpu_util_harness.sh:66,190` | `convert -resize`, `identify` | **zenresize** + the T6 probe | SMALL; perf-harness fixture only, no model data |
| T8 | `zensim/scripts/corpus/synthetic/synth_nonphoto.py:60` | PIL + matplotlib **rasterisation** | **none exists** | **BIG.** This rasterises synthetic non-photo sources (text/charts/line-art) INTO the training corpus. Imazen has no rasteriser; needs a real decision, not a swap |
| T9 | `zenavif/scripts/rd_gap/{aom_cell,aom_only}.sh` | `aomenc`/`aomdec` binaries | — | RD-gap **baselining**, not tuning data. Named so nobody mistakes it for a zenav1-aom seam (§7.3) |

**T1–T7 share one missing piece**, and it is the single highest-leverage build
in this list: a Rust entry point Python and shell can call. A
`zenmetrics image {probe,decode,resize,encode}` subcommand covers every one of
them, and would have covered `run_full_eval.sh` too (which needed its own
example precisely because no such thing exists). Registered, not built.

## 8.3 Judged compliant — recorded so the reading is explicit, not assumed

- `zenmetrics/scripts/cvvdp_goldens/*.py` and `scripts/sweep/pycvvdp_worker.py`
  (PIL + a pinned `cvvdp==0.5.4` venv) are **differential port validation for
  `cvvdp-gpu`, inside the repo that owns it** — the rule's explicit carve-out.
  `main.rs:205` already documents the worker as the parity counterpart to
  `score-pairs`. **Flagged for confirmation rather than assumed.**
- `zensim/scripts/external_reads/asrun/{avthdr,hdrvdc,sihdr}/` (ffmpeg/ffprobe,
  cv2) are **frozen archival provenance copies** — their own README says *"Do
  not extend them; extend the runner"*, and the live runner's `--from-stored`
  path does no video decode. **But note the trap:** they are the only recorded
  route to re-extract those HDR-video reads, and imazen has **no demuxer and no
  HEVC/VVC video decoder**, so re-running that study is currently impossible
  under the rule. Registered here rather than discovered mid-wave.
- `zensim/scripts/canonical_corpus/make_dhash_montages.py` renders
  **human-review** montages (its `montage()` is a local PIL function, not
  ImageMagick). Review rendering, not measurement.
- **False positives ruled out:** `probes/phone_rd_probe.sh:5-6,24-25` — `CJXL`
  and `DJXL` there point at `jxl-encoder/…/cjxl-rs` and
  `zenjxl-decoder-cli`, both **imazen** binaries with confusing variable names.
  Also `identify`/`montage` as ordinary English in
  `avifdoe_stagea_gates.py:231`, `run_with_error_trap.sh:121`, and two
  `canonical_corpus` scripts.
