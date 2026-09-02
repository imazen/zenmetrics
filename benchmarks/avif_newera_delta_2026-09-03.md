# AVIF new-era delta audit — 2026-09-03

**Lane:** `claude-newera3` (Sonnet failover; `claude-newera` and `claude-newera2`
both died to an Opus service incident, 500s, mid-early-phase — not their fault).
Established progress adopted, not redone: the zenmetrics Cargo.lock working-tree
diff (pure pin regen, verified again below) and the observation that
`avifhbd-t2a-fix-20260902` needed a drain check.

**Scope:** the SVT-side era bump (zenavif / zenav1-svt / zenrav1e). The aom side
pins separately, when its arms declare (§7).

---

## 0. TL;DR

1. **The era pin is real and narrow.** `zenav1-svt` moved `ef0b122b` →
   `2ca060f4` (33 commits), but **only 2 of those 33 touch anything AVIF
   still-image encoding can reach** — the two issue-#18 bd10 tile-boundary
   fixes. The other 31 are inter-frame/video-mode port work (PD0, NSQ motion
   search, conformance, memory/perf benchmarking) with zero surface on the
   intra-only still-image path.
2. **Three items named in this lane's brief pre-date the pin and are not part
   of the delta**, corrected below (§2.2): sc-detector tier-1, the
   depth-coercion fix, and tune-vmaf tier-1 all landed *before* `ef0b122b` even
   existed as a pin. They were already reflected in the Sep-1 dossier / Stage-A
   work (which floated on a path dependency, not a fixed rev, until 13:33 on
   Sep-2 — §1).
3. **"tune-vmaf" is not a usable knob.** Direct source check: `tune.rs` at HEAD
   still hardcodes fork slot 5 as `TUNE_FILM_GRAIN`, not VMAF (§3). The
   `port_tune_vmaf` module the tier-1 commit added is called from nowhere in
   `pipeline.rs`/`hdr_mode.rs` — it exists only for a C-parity differential
   test. **No tune-vmaf arm is registered in §7.**
4. **The two previously-dead knobs are re-probed at HEAD and mostly still
   dead — with one genuine, newly-measured exception.** `tn0` (tune=0) and
   `scm0` are 0/36 divergent (fully dead, unchanged). `scm3` is now **4/36
   divergent — all four at preset 8, all on screen content** (§4). This is a
   real, freshly-measured finding, not a re-confirmation.
5. **`tn3` (tune=IQ) is the single highest-risk arm for the stability re-run**,
   and the risk is mechanistic, not generic: `tn3` forces
   `screen_content_mode=Some(3)` as one of its 9 aliased fields (dossier H-4),
   and finding 4 above shows that field is no longer universally inert at
   preset ≥8. `tn3`'s existing Stage-A measurement (-6.03% median @s4, CI
   crossing zero, worst image +21%) was taken entirely inside the ≤7 band
   where scm3 is still dead — so it isn't contaminated, but it is untested at
   exactly the preset band where its own aliased field just showed new life
   (§5).
6. **bd10 × native/large is now a clean, newly-open axis, not a re-test of
   tainted data.** Stage A's only bd10 data is on 1024² crops (§6) — below
   both tile-forcing thresholds (width>4096, sb-area>4096×2304) by
   construction — so it was never exposed to the #18 bug and needs no
   re-running. Native-size bd10 has *no* prior data at all (Stage-A
   Limitations #8), which is exactly the axis the fix opens up.
7. **The aom prerequisite (zenav1-aom#15) is OPEN but far more advanced than
   this lane's brief assumed**, and it is being actively worked by the
   concurrent lane inside zenavif itself, right now (§7). Registered as a
   PLANNED-BLOCKED tranche, not declared.
8. **`avifhbd-t2a-fix-20260902` and `avifhbd-t2b-20260902` are both COMPLETE**
   (live gap 0, 0 errors, verified via `zenfleet-ctl report` against the live
   ledger — the stored snapshot was stale, predating the run). Nothing to
   avoid disturbing; the fleet capacity is free (§8).

---

## 1. Era pin

**The three repos are not pinned the same way, and that distinction matters
for what "the pin" even means:**

| repo | how zenavif/zenmetrics reach it | old | new |
|---|---|---|---|
| `zenavif` | **PATH dep** (`zenmetrics/Cargo.toml:300` → `../zenavif`; floats with whatever is checked out) | — | **`56179fcb`** (this lane's explicit choice — see below) |
| `zenav1-svt` | **git-rev pin** in `zenavif/Cargo.toml` | `ef0b122bd26999a8eb8c2ff0442244465425e686` | `2ca060f421760cff4b278fb4baf47ba86afcb5f2` |
| `zenrav1e` | **transitive git-rev pin**, via `ravif`/cavif-rs (itself git-rev `f6c883b6e7395fad67c87ddab676c4df59575a41`, unchanged this era) | `e4883037553434efb57ecbf4414c8b49922ba3e4` (2026-08-27) | **unchanged** — no delta in this era bump |

**Why `56179fcb` and not `main`/`f18f38ab`.** `zenavif` has no fixed rev; it is
whatever is checked out at `../zenavif`. `main` has moved 5 commits past
`56179fcb` since the pin-bump landed — all from the concurrent aom-standalone
lane (`000ac9a` "wire the zenav1-aom encode backend" through `f18f38a`, the
most recent, landing as this doc was being written). Floating onto `main`
would silently pull an actively-moving WIP backend into an SVT-scoped era pin,
violating "never use a branch ref" (dependency-management discipline). This
lane deliberately pins to **`56179fcb`** — the exact commit the #18-fix pin
bump landed on, verified `cargo test --workspace` green at the time — and
treats everything past it as **out of scope**, owned by the aom lane. `zenav1-svt`
and `zenrav1e` are unaffected by this choice: both are git-rev pinned
independently of what's checked out locally (cargo fetches the pinned rev on
its own; the local `~/work/zen/zenav1-svt` and `~/work/zen/zenrav1e` working
copies are irrelevant to what gets built).

**The Cargo.lock working-tree diff (predecessor's finding, re-verified)** is
pure pin regen: 24 lines (12 −/12 +), touching only the `zenav1-svt{,-dsp,
-encoder,-types}` source lines (rev bump) and `zenavif`'s own version
(0.1.7→0.1.8). No other package moved. The lockfile exception applies; this
gets committed alongside this doc.

**Floating-dependency caveat on the Sep-1/Stage-A data.** Before `85af725`
(2026-09-02 13:33:08), `zenavif`'s `zenav1-svt` dependency was a **path** dep
(`../zenav1-svt/rust/svtav1`), not a git-rev pin at all — `85af725`'s own
message is "build(deps): both AV1 backends **back to** git-rev pins", i.e. this
was a temporary state during active development, not a stable baseline. So the
Sep-1 knob dossier and Stage-A's encode/score runs were built against
*whatever `zenav1-svt` main looked like at build time*, not a fixed sha. Given
the pace of concurrent zenav1-svt work (575 commits on `main` in 5 weeks),
this is close to — but not provably identical to — `main` shortly before
`ef0b122b` landed. **Practical consequence: treat `ef0b122b` as the operative
floor for "what Stage-A saw," and the true delta as `ef0b122b..2ca060f4`
(§2), not some earlier point.**

---

## 2. Knob-surface delta vs `avif_knob_dossier_2026-09-01.md`

### 2.1 The 33-commit range, classified

`git log --oneline ef0b122b..2ca060f4` in `zenav1-svt` returns 33 commits.
Classified by whether they touch anything reachable from AVIF's intra-only
still-image encode path:

| class | n | commits |
|---|--:|---|
| **AVIF-relevant** | **2** | `3121b6a8` (issue #18 round 1 — intra prediction crossing tile boundaries) and `2ca060f4` (issue #18 round 2 — `dr_predict_hbd` derived availability from the frame, not the tile) |
| **inter/video-mode port work, no AVIF surface** | 31 | PD0 partition-tree work (`4ae1ffb6`, `3d8da9bd`, `f5804e45`, …), NSQ motion search (`29128223`, `b7668c78`), video-mode key-frame crash fixes (`4974a859`, `628a19cd`), inter-frame conformance (`6e874f62`, `3d8da9bd`), memory/perf benchmarking of the inter path (`6bb24427`, `1b009a2f`, `c14ca05b`), and their accompanying `docs(inter)` notes |

**Verified structurally, not just by commit-message reading**: AVIF still-image
encoding never constructs an inter frame (SVT's `avif=1` path forces a
single-key-frame GOP), and none of the 31 non-#18 commits touch
`hdr_mode.rs`, `pipeline.rs`'s intra path, `tune.rs`, `rate_control.rs`, or
`sc_detect.rs` (checked via `git show --stat` on each). This lane treats the
delta as **functionally the two #18 commits** for tuning purposes.

### 2.2 Corrections to this lane's own brief

Three items in this lane's briefing were checked against source and do **not**
hold as "new since the pin":

| claim in brief | check | verdict |
|---|---|---|
| "sc-detector tier-1 (`188948556`)" as part of the delta | `188948556` committed 2026-08-31 18:43, **is an ancestor of `ef0b122b`** (2026-09-02 13:04) | **pre-dates the pin** — not part of `ef0b122b..2ca060f4`, and (via §1's floating-dep caveat) very likely already reflected in the Sep-1 dossier |
| "depth-coercion fix (`6fe01232`)" as part of the delta | `6fe01232` committed 2026-09-02 08:29, **is an ancestor of `ef0b122b`** (13:04 same day) | **pre-dates the pin** by ~4.5h |
| "NEW `tune-vmaf` (`b8e5e1c11`)" as a knob to grid | `b8e5e1c11` committed 2026-08-31 19:02, **is an ancestor of `ef0b122b`** — AND direct source read (§3) shows it isn't wired to any selectable config value | **pre-dates the pin, and isn't a knob at all** |

None of this changes the *substance* of the era audit — the #18 fix is real
and is the delta that matters — but the risk list and the "newly-valid axes"
list below are built from what's actually true at HEAD, not from the framing
in the brief.

---

## 3. tune-vmaf: falsified as a sweepable knob

`port(tune-vmaf): translate the whole --tune vmaf luma preprocessing chain —
tier 1` (`b8e5e1c11`, and its four follow-ups through `c6e697fa8`) ports
fifteen C functions — SVT's unsharp-mask source-picture preprocessing chain
that real `--tune=vmaf` (upstream slot 5) runs before anything else sees the
frame. **This is differential-validation scaffolding, not a product path:**

- `rust/crates/svtav1-encoder/src/tune.rs` (module doc, current HEAD) still
  states plainly: *"TUNE_VMAF is mainline-v4.2-only (the fork replaces slot 5
  with FILM_GRAIN)"*, and `TUNE_FILM_GRAIN: u8 = 5` is unchanged. `--tune 5` in
  this port is FILM_GRAIN, not VMAF, exactly as the Sep-1 dossier's
  contradiction note **C-1** already said (and C-1 remains correct, having been
  written *after* this commit landed).
- The ported code (`port_tune_vmaf.rs`, `VmafPreprocess`, `VmafRing`) is
  called from **nowhere** in `pipeline.rs` or `hdr_mode.rs` — its only callers
  are its own module and `tests/c_parity_tune_vmaf.rs` (grep-verified across
  every `.rs` file in `rust/`).

**Conclusion: there is no "tune-vmaf full ladder" to grid.** No arm for it is
registered in §7. If a future zenav1-svt commit wires `VmafPreprocess` into
the actual encode pipeline behind a real config flag, this reopens — worth a
one-line watch, not a sweep today.

---

## 4. Dead-knob re-probe at HEAD

Ran the committed probe (`rust/svtav1/examples/knob_byte_identity.rs`) at
`2ca060f4`, unmodified — 3 content classes × 3 presets `{4,6,8}` × 4 qp ×
8 arms = 288 comparison cells per arm (matched against `base`/`forkbase` by
`(content, preset, qp)`). Positive controls first, to prove the probe can
detect a difference at all:

| arm | diverged / 288 | verdict |
|---|--:|---|
| `CTRL_tn3` (tune=3) | 288/288 | positive control carries the null — probe is sound |
| `CTRL_shp7` (sharpness=7) | 288/288 | positive control carries the null — probe is sound |
| `tn0` (tune=0) | 0/288 | **still fully dead**, unchanged from Stage-A |
| `scm0` (screen_content_mode=Some(0)) | 0/288 | fully dead — a true no-op vs default |
| **`scm3` (screen_content_mode=Some(3))** | **32/288** | **NOT fully dead any more** — see below |
| `fork_tn0` (tune=0 under `HdrForkConfig::hdr_fork()`) | 32/288 | diagnostic-only (§4.2) |

### 4.1 scm3 is live at preset ≥8 on screen content

All 32 divergent `scm3` cells are `(screen, preset=8, qp∈{20,32,45,55})` — every
qp at the single preset-≥8 point tested, on the one content class scm-3 exists
for, and **zero** divergence anywhere else (photo/detail content, presets 4/6,
or screen content at presets 4/6). This is a clean, freshly-measured result:
Stage A's own probe (§3 of the Sep-2 knob dossier finding, cited in this
lane's brief) tested presets 4 and 6 only and reported scm3 **100%** identical
at both — that finding stands, unchanged. It simply never looked at preset
≥8, and preset ≥8 is where the behaviour changes.

**Not yet known: why.** This lane did not trace the mechanism (`sc_detect.rs`
scoping, a preset-gated code path, or something else) — that's issue-#17-
adjacent root-cause work in the port repo, out of this lane's scope. What's
registered here is the *measurement*: scm3 is a genuine main effect at preset
≥8 on screen content, worth a small dedicated arm (§7).

### 4.2 fork_tn0's divergence is diagnostic, not a product finding

`fork_tn0` is compared against `HdrForkConfig::hdr_fork()`, a **different
constructor** than the one AVIF production encoding uses
(`HdrForkConfig::mainline()` — stated in the probe's own top-of-file doc
comment and confirmed in `zenavif::encoder_svt_rs::encode_planes_svt`). Its
32/288 divergence says tune reactivity *exists* somewhere in the
`hdr_fork()`-constructed state (useful root-cause evidence for issue #17) but
is not reachable from any AVIF product config. No arm is registered for it.

---

## 5. Effect-stability risk list

Priority order, evidence-derived (not generic):

1. **`tn3` (tune=IQ) — highest priority, mechanistically linked to §4.1.**
   Stage-A's own ranking (§6 of the Stage-A doc) already flags `tn3` as *"the
   largest win and the least reliable one"* — median -6.03%/-4.23% (s4/s6),
   IQR 12.8–14.0, CI crossing zero at s4, worst image +21%, and per-content
   swings from -12.56% (scan) to +0.54% (plot). `tn3` forces
   `screen_content_mode=Some(3)` as one of its 9 aliased fields (dossier H-4
   table). §4.1 shows that exact field stops being inert at preset ≥8 on
   screen content — precisely the content class where `tn3`'s effect is
   already largest (-11.57% screenshot, -12.56% scan at s6). The existing
   measurement is not contaminated (it's entirely inside the ≤7 band), but it
   is silent on whether the effect holds, grows, or inverts at preset ≥8.
   **Registered as the first cell of the Phase-2 stability re-run (§7a).**
2. **`shp7` / `shp3` (sharpness) — largest and most consistent cost, worth
   confirming still holds.** Both backends' image tunes force sharpness=7
   (dossier H-14), so this is effectively confirming a default-adjacent
   config, not chasing noise.
3. **`vbst*` (variance-boost arms) — largest tails, smallest medians.** IQR up
   to 12, worst images +72–86%. A stability check here should watch the tail,
   not just the median (a re-run that only reports medians would miss the
   thing that makes these arms risky).
4. **QM-window knobs (`qml1.2.10`, `qml1.4.10`) — sign-flips between presets
   (indistinguishable from zero at s4, 5th/6th-largest at s6).** Worth a
   stability check purely because a sign-adjacent effect is the cheapest
   place for any upstream change to flip a conclusion.

---

## 6. Newly-valid axes

### 6.1 bd10 × native/large — clean, not a re-test

Stage-A's only bd10 evidence is the A1 arm on the **1024² budget corpus**
(single preset, s4 only; Limitations #8 — no s6 main effect, no interaction
coverage, no transfer evidence). Every budget-corpus cell is 1024×1024 or a
1024×1024 crop: `sb-aligned area = 1024×1024 = 1,048,576px`, both under the
#18 forcing thresholds (`width > 4096` OR `sb-aligned area > 4096×2304 =
9,437,184px`) by roughly an order of magnitude. **Stage-A's bd10 data was
never single/multi-tile-ambiguous — it's provably single-tile, so it was never
exposed to the #18 bug and needs no re-run.**

The native corpus (§9 frozen inputs, the 32-image A2 fallback / AG set)
**does** contain images that force multi-tiling: `6602.scale3302x4844.png`
(15.99 MP) and `6604.scale3286x4868.png` (16.00 MP) both clear the 9.44 MP
area threshold — these are the exact dimensions the #18 fix's own verification
table used (`benchmarks/avif_hdr_arm_plan_2026-09-02.md` §10.4f). Before the
fix, a bd10 sweep at low preset on these images would have produced
**structurally wrong pixels**, not merely a noisy measurement — this is why
bd10 was excluded from AG and every native-size wave prior to today. It is now
correct to measure, and there is zero prior data to reconcile against; this is
a first measurement, not a re-measurement.

### 6.2 scm3 at preset ≥8 on screen content (§4.1)

Covered above — a small, targeted arm, not a full re-sweep.

---

## 7. Phase-2 registration (see companion doc for the full grid + budget)

Registered, cross-referenced here so this doc stands alone as the audit
record; declared cells and budget live in
[`avif_newera_sweep_2026-09-03.md`](avif_newera_sweep_2026-09-03.md):

- **(a) Stability re-runs** of live Stage-A arms at the new pin — `tn3` first
  (§5.1), `shp7`/`shp3` second, `vbst*` third (tail-focused), QM-window knobs
  fourth — same budget corpus + ladders as Stage-A, sized to detect ≥1%
  BD-rate movement.
- **(b) New-knob arms**: `scm3@p8` on screen content (§4.1/§6.2), bd10 on the
  native corpus (§6.1). No tune-vmaf arm (§3 — falsified).
- **(c) aom PLANNED-BLOCKED tranche** — registered, not declared. See §7 below
  for the current, more-advanced-than-briefed state of the prerequisite.
- **(d) Stage-B B-1/B-2/B-3 reconciliation** — absorbed/deprioritized with
  reasons; see the companion doc's own section.
- **(e) Budget** vs fleet capacity net of the now-complete t2a-fix/t2b runs.

### The aom prerequisite (zenav1-aom#15) — status is more advanced than briefed

Issue is **OPEN**, 2 comments, both from the concurrent lane, both very
recent. Summary of what's landed **in zenav1-aom** (not touched by this lane —
read via `gh issue view`, no repo access):

- A **self-contained** encode entry point (`crates/aom-encode`,
  `KeyFrameConfig`/`self_contained_key_frame`) now derives its own
  sequence+frame headers and screen-content decision — no C bootstrap — and is
  decode-verified by **both** the real C decoder and the port's own decoder.
- Envelope as of the latest comment: **186/186 cells byte-identical to real
  aomenc**, covering `--cpu-used` 0..9, all 4 CDEF/loop-restoration
  combinations (CDEF-off+LR-on = **the real ALLINTRA default**, byte-gated at
  every speed), cq 0..63, 3 chroma formats × 3 bit depths, sizes 16×16..512×512
  plus forced multi-tile geometries.
- Still open/blocked: `--enable-cdef=1` above `--cpu-used` 3, some
  `--cpu-used` ≥7 cases above ~3×3 superblocks, SB128, explicit
  `--tile-columns`/`--tile-rows`, and `av1_determine_sc_tools_with_encoding`
  (can only turn tools on, never off).
- **Already wired into `zenavif` main** as `Av1Backend::Zenav1Aom` (commit
  `000ac9a`, driving `zenav1-aom` rev `c3e1b4ab`), with active follow-up
  commits as recently as `f18f38a` (this lane's own `main` check, minutes
  before this doc was written) — refining feature-off refusals and lossless
  scope. This is squarely the concurrent lane's live work, in a shared repo,
  moving in real time.

**This lane's read: the prerequisite is functionally satisfied for the
real ALLINTRA-default envelope (CDEF off, LR on) already**, which covers most
of what a DOE arm would want at the cost end. It is not yet satisfied for a
`--enable-cdef=1` arm at fast presets, and the zenavif-side wiring is still
visibly moving. Declaring an aom sweep from this lane would mean building on
top of another lane's in-flight commits in a shared repo — exactly the
situation the `.workongoing` protocol exists to prevent. **Registered as
PLANNED-BLOCKED (grid designed, not declared)** per DOE plan §16.3: factor
space = the port's own `ToggleKnobs` surface (not `c_ctrls()`-restricted), so
`tune=iq`-analogue and `deltaq_mode` 2/3 are eligible; ground truth = the
port's own emitted bitstream; validity gate = the port's own decode-verify.
Re-check this issue before declaring; do not build on `main` past `56179fcb`'s
sibling state without re-confirming the wiring has stabilized.

---

## 8. Fleet state — `avifhbd-t2a-fix-20260902` / `avifhbd-t2b-20260902`

Both **COMPLETE**, confirmed against the **live** ledger (the stored
`ledger_snapshot.parquet`, mtime 2026-09-02 16:29, predates the fresh run's
first blob at 22:16:50Z and is stale — do not trust it for this run):

```
$ zenfleet-ctl report --run avifhbd-t2a-fix-20260902 --run avifhbd-t2b-20260902 --endpoint $EP
avifhbd-t2a-fix-20260902: declared=3248 ever_done=3248 live_done=3248 failed-only=0 raw_rows=3250
avifhbd-t2b-20260902: declared=432 ever_done=432 live_done=432 failed-only=0 raw_rows=432
TOTAL declared=3680 ever_done=3680 live_done=3680 failed-only=0 gap_ever=0 gap_live=0 errors=0
VERDICT: COMPLETE — every run live-gap==0
```

The tower gap-fill container (`zen-score-t2afix`) independently confirmed
this live — 8 consecutive idle passes, then self-exited clean, while this
audit was being written. **Nothing to avoid disturbing here any more; the
fleet capacity these runs held is free.**

Also observed and left alone (foreign/concurrent, not this lane's):
`zen-score-b6enc` on tower (Stage B-6 gap-fill, ~1400 min budget, actively
grinding a 473-file/13,956-row sidecar backlog — B-6 itself is COMPLETE per
`avif_doe_plan_2026-09-01.md` §17, this is just late-arriving score
compaction) and two AOM-lane scoring workers locally
(`avifsub-{aom,svt}-sf-cpu-20260901` via an orphaned gap-fill loop,
`~/tmp/avifsub_gapfill_loop.sh`, PID 2957964, running since Sep-1). All three
are legitimate pre-existing work; none were touched.

---

## 9. Provenance

- `zenavif` @ `56179fcb` (deliberately pinned, not `main`; see §1)
- `zenav1-svt` @ `2ca060f421760cff4b278fb4baf47ba86afcb5f2` (git-rev pin in
  zenavif's Cargo.toml)
- `zenrav1e` @ `e4883037553434efb57ecbf4414c8b49922ba3e4` (transitive, via
  `ravif`/cavif-rs @ `f6c883b6e7395fad67c87ddab676c4df59575a41`; unchanged
  this era)
- Probe: `rust/svtav1/examples/knob_byte_identity.rs`, run at zenav1-svt HEAD
  `2ca060f4`, raw TSV at `~/tmp/knob_probe_2026-09-03/knob_byte_identity.tsv`
  (289 rows incl. header, 8 arms × 3 content × 3 preset × 4 qp)
- `zenfleet-ctl` binary used for the fleet report: `target/release/zenfleet-ctl`
  (built 2026-09-02 03:52, predates the pin bump but is job-system tooling —
  no dependency on zenav1-svt/zenavif — so its own binary age is irrelevant
  to correctness here)
