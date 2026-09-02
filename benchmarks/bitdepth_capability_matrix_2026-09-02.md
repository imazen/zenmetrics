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
