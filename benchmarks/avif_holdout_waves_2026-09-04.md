# AVIF autotune — two data-gap waves: odd-origin holdout + zenrav1e native arm

**Status: PRE-REGISTERED, NOT LAUNCHED — PARKED 2026-09-04.** Coordinator
directive, same session: the AV1 backend (`zenav1-svt`/`zenrav1e`/`zenav1-aom`)
is under active concurrent development and about to get ~2× faster. Every knob
code, timing number, and fleet-image tag below is a snapshot of the pre-rework
state and MUST be re-verified before this is declared — see §5. **Nothing in
this document has been encoded, declared, or launched.** No fleet job exists,
no worker was started, no image was built, no file in `zenavif`/`zenav1-svt`/
`zenav1-aom`/`zenpipe` was touched while designing this.

This is the design for the top two items in
[`avif_autotune_v1_2026-09-04.md`](avif_autotune_v1_2026-09-04.md) §5 ("what
the numbers say to buy next"): a true held-out AVIF corpus (item 1) and a
zenrav1e native-size arm (item 2, the direct fix for the failed backend head —
see the contract's §0 row and the record's §4 "Backend pick — FAILS").
Consumer contract: [`avif_autotune_contract_2026-09-04.md`](avif_autotune_contract_2026-09-04.md).
Fleet conventions this design follows: [`avif_stageB_remainder_2026-09-03.md`](avif_stageB_remainder_2026-09-03.md)
§4.5/§4.5c/§4.5d (worker topology) and §4.9/§6 (gates, execution record).

---

## 0. Why these two, and why now they wait

Both waves attack the two structural gaps the training-view record measured,
not guessed:

- **No canonical holdout exists.** All 32 subsample refs (both the budget and
  native corpora under `avif-autotune-2026-09-04`) were k-means-selected under
  `--parity 0` (even-origin), so `origin_split.split_of` returns `train` for
  every one of them — the canonical `{1,3,5}`/`{7,9}` buckets are structurally
  empty (record §3). `eval8` is a leg-side stand-in carved from train-origin
  content, not a real holdout, and the record says so explicitly.
- **The backend head fails**, 54.0% agreement vs. a 67.7% always-svt baseline,
  because the entire zenrav1e arm (`brsdr`) ran at 1024² budget size only —
  zero native-size zenrav1e coverage exists anywhere in the corpus (record §4,
  contract §0 row 2).

Both fixes are pure encode/data work with a pre-existing design (this doc) and
no code changes needed in the picker/trainer. They were about to be declared
when the coordinator's stop landed. **Parking them mid-design is cheap and
correct**: the backend rework will change per-cell timing (making any β
re-derivation done now stale on arrival) and may touch the knob surface these
grids reference, so re-verifying at resume is mandatory regardless of how much
polish went in before the stop.

---

## 1. WAVE 1 — odd-origin holdout (design, not yet built)

**Goal:** a K-means-selected, ODD-origin (never-train) imazen26 subsample,
built with the *exact* method and tooling that produced the existing 32
even-origin refs, then encoded over the training view's `core` SVT config set
plus the zenrav1e arm, at budget (1024²) size, on the registered 9-point q
ladder. This becomes the canonical `eval8` replacement — a true generalization
holdout instead of a leg-side one.

### 1.1 Selection — no code change needed

`scripts/imazen26_recluster_even.py` **already has `--parity {0,1}`** (0=even/
train default, 1=odd/holdout) — the "there may be an odd-side path" in the
brief turned out to be true; no `--parity odd` flag needs adding. It also
already carries `--crop-label` (added 2026-09-01 for the even build) and
`--out-model` (additive, unused by the even build). Exact mechanical
complement of the command that built the 32 even refs
(`avif_sweep_subsample_2026-09-01.md` §1):

```sh
uv run --with scikit-learn --with pyarrow --with numpy python3 scripts/imazen26_recluster_even.py \
  --parquet /mnt/v/output/imazen-26-features/imazen26_features_2026-06-13.parquet \
  --select-k 20 --parity 1 --crop-label full --seed 0 \
  --out-manifest /mnt/v/output/avifsvt-subsample-oddholdout-2026-09-04/reps_K20_full_odd_2026-09-04.tsv
```

- **K = 20**, chosen as the midpoint of the brief's 16-24 range (population is
  large enough: the even build's own log line put the whole-native-image count
  at 1,978 even+odd combined against 1,082 even, so the odd pool is ~896 —
  comfortably >> K).
- **`--parity 1` pools BOTH canonical digit buckets** (`{1,3,5}` validate +
  `{7,9}` test) into one holdout, because the script's parity filter is binary
  (even/odd), not the finer 5-way split `origin_split.py` uses elsewhere. This
  is the natural mechanical complement of what already exists, but it is a
  **design choice flagged for confirmation, not decided unilaterally here**:
  whether a picker holdout should keep validate/test undifferentiated (fine
  for "is this generalizing at all") or split them (needed if this corpus will
  ever gate a final ship decision) is the resuming lane's or coordinator's
  call.
- Expect the same log-line verification the even build used: `ODD-id reps
  (MUST be... )` — for parity=1 the check inverts to **EVEN-id reps MUST be 0**
  (the script always reports the count for the *other* parity; read the
  printed count against whichever parity was requested, don't assume the
  literal string).
- Seed, K-source, and z-scoring are otherwise byte-for-byte the same recipe —
  same excluded GEOM features, same standardization, same centroid-nearest
  member rule.

### 1.2 Corpus build — G-CROP applies, unchanged tool

`scripts/jobsys/avifdoe_build_budget_corpus.py` (the same builder that made
`avif-doe-1024-2026-09-01`) takes the K=20 manifest, crops each whole native
image to a 1024² window, and **re-extracts imazen26 features from the crop**
(G-CROP) — the builder already hard-errors non-zero (`*** G-CROP FAILED ***` /
`*** G-CROP NOT SATISFIED ***`) rather than silently accepting an unchecked
crop, per `avif_doe_plan_2026-09-01.md` §11.3/§12.1. No changes needed to this
tool; point it at the new odd manifest and a fresh output dir
(`avif-doe-1024-oddholdout-2026-09-04` suggested, unbuilt).

### 1.3 Encode grid

Reuses the training view's exact `core` SVT config set (48 named configs,
listed verbatim in `_MANIFEST.json`'s `core_view.config_names` at
`/mnt/v/zen/avif-autotune-2026-09-04/_MANIFEST.json` — e.g. `svt-s1`..`svt-s7`,
every `qml{1.2.10,1.4.10,1.8.15}`/`shp{3,7}`/`mtx32`/`tl{1.0,1.1}`/`tn3`
combination the training view actually used) plus the zenrav1e speed arm
(`rav-s1`..`rav-s10`, the same 10-speed dial `brsdr` used), both at BUDGET
(1024²) size, on the registered 9-point ladder already used for `brnat`/`brsdr`
(`Q_LADDER9="5,15,25,35,45,60,76,90,96"` in `avifdoe_declare.sh`).

| arm | configs | q points | refs | cells |
|---|--:|--:|--:|--:|
| SVT `core` | 48 | 9 | 20 | **8,640** |
| zenrav1e arm | 10 | 9 | 20 | **1,800** |
| **WAVE 1 total** | 58 | 9 | 20 | **10,440** |

**Reconciling with the brief's own shorthand.** "16-24 refs × 48 cells ≈
800–1,200 cells" computes to 960 at K=20/48-configs-only — that is the
**(ref × config) build-point count**, not the fleet-declared **cell** count
this repo's own convention uses everywhere else (`(image, config, q)` triple —
e.g. `brnat` = 26 strata × 9 q × 32 images = 7,488). Once the 9-q ladder is
applied, the real total is **10,440**, ~9× the shorthand. Both numbers are
recorded here rather than silently picking one, per this doc family's own
"corrected in place, not quietly replaced" convention (stageB remainder §4.3/
§4.8 did the same when its own estimates were wrong).

**Rough cost ceiling (estimate, not measured — this corpus doesn't exist
yet):**
- SVT: using `brnat`'s native blended rate (1.90 s/cell) as a deliberately
  conservative ceiling (budget-size cells are cheaper than native — mean
  budget MP is ~1.16 vs native's ~5.05): 8,640 × 1.9s ≈ **4.6 CPU-h ceiling**.
- zenrav1e: scaling `brsdr`'s own measured blended rate (23.9 CPU-h for 32
  refs × 29 q × 10 speeds) down to 20 refs × 9 q: 23.9 × (20/32) × (9/29) ≈
  **4.6 CPU-h**.
- **WAVE 1 rough ceiling: ~9.2 CPU-h.** Low-to-moderate confidence — scaled
  from measured rates on the *even* corpus; the odd K-means pick will draw a
  different content mix (record: content class swings cost 3.4×), so this is a
  ceiling to sanity-check G-RATE against, not a number to plan around.

### 1.4 Gates (mirrors stageB remainder §4, adapted)

| id | gate |
|---|---|
| G-SMOKE-1 | recluster log line confirms 20 reps, 0 wrong-parity leaks |
| G-CROP | budget-corpus builder exits 0; every ref's crop-feature recheck recorded (cluster-shift noted, not fatal) — same bar as the even build's `133b93d80c15581f…`/`2 of 13 changed cluster` result |
| G-DEDUP | `avifdoe_declare.sh`'s plan audit (`duplicates_merged`, `invalid_skipped`) inspected before declaring — a knob grid with 0 duplicates merged is a red flag on this corpus too |
| G-FIRSTCELL | one real cell of EACH arm (svt core, zenrav1e) encodes inside the fleet image against a real source, **including an av1C box read-back** confirming the expected chroma subsampling for that backend (the task brief's explicit addition to the standard gate — catches a corpus/plan mismatch the way `avif-doe-1024-2026-09-01`'s empty-sources bug did) |
| G-RATE | after ~500 cells, realised CPU-s/cell projects the wave inside a stated ceiling (§1.3) or the de-scope ladder below fires |
| G-SCORE | ssim2 + zensim declared **at launch**, not after |

**De-scope ladder, pre-registered, fires on G-RATE only (never re-ordered
after seeing results, per the stageB precedent):**
1. Drop the zenrav1e arm's speeds 1-2 (cheapest cut per stageB §4.3's own
   finding: speeds 1-2 carry ~73% of zenrav1e's cost for 20% of its cells).
2. Drop to K=16 refs (bottom of the registered range) rather than shrinking
   the config set — the whole point of this wave is holdout breadth across
   content classes, so cutting refs before cutting configs preserves fewer-
   but-complete per-ref RD curves.
3. Never cut: the SVT `core` set (that's the whole training-consistency point)
   or the q ladder below 9 points (matching `brsdr`'s own floor).

---

## 2. WAVE 2 — zenrav1e native-size arm (design, not yet built)

**Goal:** the direct fix for the failed backend head — zenrav1e coverage at
native size on the SAME 32 even refs already in the training view, so the
backend comparison is no longer confounded with a size extrapolation.

### 2.1 Grid

Same shape as `brsdr` (`{"backend":["zenravif"],"speed":[1..10]}`
knob-grid, `Q_LADDER9`), pointed at the **native** corpus
(`/mnt/v/output/avifsvt-subsample-2026-09-01/sources`, `NATIVE_SOURCES` in
`avifdoe_declare.sh`) instead of budget — mechanically the same shift `brnat`
made relative to A2's budget-side pairwise plan.

| arm | configs | q points | refs | cells |
|---|--:|--:|--:|--:|
| zenrav1e native | 10 | 9 | 32 | **2,880** |

2,880 = 10 × 9 × 32, matching the record's own §5 item-2 estimate exactly.

**§2.3 upgrades this to a 420+444 double grid (5,760 cells total) — both
facts that originally gated that upgrade are now confirmed; see §2.3.**

### 2.2 CPU-h — re-derivation NOT completed before the stop; using the
record's own number as-is

The task brief asked to re-derive the ~19 CPU-h figure from the speed
instrument's (`avif_speed_instrument_2026-09-03.md`) per-source β fits and trim
if it exceeded ~20 CPU-h. **That re-derivation was not done** — the stop
directive landed before this lane opened that document. Two reasons this is
not chased further right now rather than treated as a silent gap:

1. **The number is already under the stated cap** (~19 < ~20), so even a
   successful re-derivation would not have changed the grid in §2.1 — only
   the confidence label on the estimate.
2. **The backend is being reworked for ~2× speed.** Any β fit against the
   current binary is stale the moment the rework lands. Spending effort
   deriving precision on a number that is about to move is lower value than
   flagging it for resume-time re-measurement, when the new β will actually
   be the one that matters. This is a "the input is about to change" reason,
   distinct from — and not an instance of — deferring due to context budget.

**At resume:** open `avif_speed_instrument_2026-09-03.md`, pull the per-source
α+β(MP) fits for zenrav1e (note: pooled fits were flagged
`linear_model_failed` on 20/20 arms with β spreading up to 24.3× — per-source
fits are the ones to use, not a pooled mean), multiply by the 32 native refs'
actual MP (mean ~5.05 MP per the record), and confirm the total against the
~20 CPU-h cap. Trim by dropping speeds 1-2 first (same rationale as Wave 1's
de-scope step 1) if it's over.

### 2.3 Chroma — UPDATE: both facts now confirmed via `44ff01e6`, upgrade the
design to 420+444, but still NOT declared

This lane was told not to touch zenavif directly, but the two gating facts
from the original design turned out to already be answered by a commit this
lane had *already fetched* (`git show 44ff01e6`, read for its filename while
finalizing this doc — no zenavif files touched, no new zenavif investigation
opened):

- **The zenrav1e-420 knob exists and works.** `44ff01e6`'s own commit body
  reports a `br420` arm — zenrav1e at 4:2:0 — with **2,880/2,880 cells
  scored**, era-inert (2,880/2,880 byte-identical to a control) and
  scorer-inert (max |delta ssim2| = 0.0 on 2,880 shared bitstreams). That is
  direct proof of the knob's existence and correctness, stronger than
  checking a commit hash.
- **The chroma verdict is in, at max strength.** `avif_chroma_split_2026-09-04.md`
  (confirmed as the real path via `git show 44ff01e6 --stat`, not yet read in
  full) — **CHROMA, k11=11/11** against the pre-registered k11>=9 rule: every
  one of the 16 pre-registered svt-4:2:0-fails-90 references is *also* a
  reach-90 failure for zenrav1e at 4:2:0. The commit's own framing: chroma is
  "not a bytes-efficiency lever... it is a hard reach CEILING," and with
  chroma held constant, backend alone moves BD-rate +10.19% svt-favouring
  (CI [+0.25, +19.15]).

**Both of §2.3's original gates pass.** The design upgrades to: run zenrav1e
natively at **both** 420 and 444 — 2x the grid in §2.1, **5,760 cells** total
— so backend and chroma are separable at native size, matching the task
brief's original ask in full rather than deferring the 420 half.

**Still not declared, and not upgraded to a launch by this update.** Reading a
commit message already in hand to correct this doc's own claims is a
documentation fix, not new investigation or a green light to proceed — the
coordinator's stop covers declaring/launching, which this section does not
do. At resume: read `avif_chroma_split_2026-09-04.md` in full (this lane has
only read the commit headline + body, not the analysis doc itself) before
declaring, in case its detailed results change the grid (e.g. a q-range or
size caveat this summary doesn't carry), then declare the 5,760-cell 420+444
grid per §2.1's shape doubled.

### 2.4 Gates

Same table as §1.4, minus G-CROP (native refs are the existing whole images,
no crop step) and G-DEDUP (single knob-grid arm, no stratum filter to audit).
G-FIRSTCELL's av1C read-back is exactly what separates the 444-only launch
from the 444+420 launch: confirm the encoded box reports the requested
subsampling for BOTH arms if the 420 follow-up is in scope, not just one.

---

## 3. Fleet conventions both waves inherit (no deviation registered)

From `avif_stageB_remainder_2026-09-03.md` §4.5c/§4.5d, restated here so a
resuming session doesn't have to re-derive it:

- **Narrow per-worker cpusets (1 core each), `ZEN_CORE_OVERSUBSCRIBE=2`
  (the launcher default — do not disable it).** Wide cpusets lose per-encode
  cache locality; that cost was mis-attributed to oversubscription once
  already (§4.5 → corrected in §4.5d) — don't re-make that mistake.
- **`ZEN_CHUNK_WALL_SEC=300`** (bounded claim → regular ledger flushes,
  bounded loss window on a killed pass) **+ `ZEN_PASS_TIMEOUT=7200`** (so a
  300s-bounded chunk of zenrav1e-class cells — 3.7×-64.8× slower per cell than
  svt — still finishes inside the pass) **+ `ZEN_LONG_LIVED=1`** for any worker
  trailing a live run that isn't expected to finish in one pass.
- **Any manifest change (de-scope, gap-fill, re-declaration) requires
  restarting long-lived workers** — `ZEN_LONG_LIVED=1` pins the manifest fetch
  outside the pass loop.
- **`zenfleet-ctl gap` against the run's own manifest is the ONLY progress
  signal.** Blob counts undercount (content-addressed dedup — `brnat` finished
  at 7,488 cells / 5,817 distinct blobs) and score-blob counts are chunked
  (12 cell-records per blob in the stageB run). A monitor built on either will
  false-alarm on a finished run exactly like stageB's did.
- **Fleet image: reuse the newest tag that carries the chroma knob — do not
  build a new package name.** Last known tag in this doc family:
  `ghcr.io/imazen/zenfleet-worker:exec-avifhbd-eradelta-e015344f` (stageB's
  pin). The chroma-split lane may have published a newer tag as part of
  landing `44ff01e6` — **check its record at resume**, don't assume the stageB
  pin is still current.
- **Scoring (ssim2 + zensim) declared at the same time as the encode run**,
  not after — the standing rule this doc family calls out by name every time
  (stageB §0 bullet 5, repeated because three prior waves got this wrong).
- **Two-stage smoke → scale.** Dry-run cell counts must equal the arithmetic
  in §1.3/§2.1 before declaring; one real cell of each arm before scaling to
  the full grid (G-FIRSTCELL).
- **Topology, observe-before-load:** check `docker ps` + `uptime` on every
  target box before assuming it's free, exactly as stageB's r7900x reservation
  and tower B-6 restart-loop discoveries required. r7900x specifically is
  reserved for the resume lane's S1c timing instrument as of park time — its
  `.workongoing` marker in this repo says *"watching for S1c COMPLETE marker on
  r7900x"* — **do not put a worker there until that marker is gone or the
  resuming session confirms S1c has released it.**

---

## 4. What was NOT done, honestly

- No fleet job declared, no worker started, no image built, no encode run.
- No code changes: `imazen26_recluster_even.py` needed none (parity=1 already
  works); `avifdoe_declare.sh` would need a new mode (`--odd-holdout` or
  similar, following the exact `--stage-b-remainder`/`--chroma-split` pattern
  of reusing `declare_block_filtered`/`declare_block_knobgrid`) to actually
  emit Wave 1's combined svt-core + zenrav1e-arm grid in one script — that
  mode was designed above (§1.3) but not written.
- WAVE 2's CPU-h was not re-derived from the speed instrument's raw β fits
  (§2.2) — used the record's own pre-computed ~19 CPU-h / 2,880-cell figure.
- zenavif's `f15bb3a5` zenrav1e-420 knob claim: not independently checked against that exact commit, but indirectly confirmed working via `44ff01e6`'s `br420` arm (2,880/2,880 scored, era+scorer inert) — see updated §2.3.
- The chroma-split verdict (`44ff01e6`) headline + commit body were read (informing the §2.3 update above); the full analysis doc `avif_chroma_split_2026-09-04.md` was not opened.
- No `DATA_PROVENANCE.md` or `DATA_SPLITS.md` entry was added — both describe
  artifacts that don't exist yet; adding them now would be speculative. Add
  them when a wave actually launches and produces a manifest with real
  sha256s (follow the `avif-autotune-2026-09-04` row in `zensim/docs/
  DATA_SPLITS.md` as the template for Wave 1's eventual entry).

## 5. Resume checklist

Before declaring either wave:

1. Re-read this doc's §1-§3 against current `zenavif`/`zenav1-svt`/
   `zenrav1e` state — the coordinator's stated reason for the pause is a ~2×
   backend speedup in flight; knob codes, default configs, and timing all may
   have moved.
2. Confirm the fleet image tag (§3) — don't reuse a stale one blindly.
3. Confirm r7900x availability (§3) and current household/tower load
   (observe-before-load).
4. Re-run G-SMOKE-1/G-DEDUP dry-runs fresh — don't trust this doc's cell-count
   arithmetic without a dry-run confirming it against whatever the corpus and
   plan look like at resume time.
5. Read `avif_chroma_split_2026-09-04.md` in FULL (only the headline + commit body are read as of this park) before declaring Wave 2's 420+444 grid -- confirm no q-range/size caveat narrows it.
6. Re-derive Wave 2's CPU-h from the (by-then-current) speed instrument before
   declaring, per §2.2.
7. Declare scoring (ssim2 + zensim) in the same action as declaring encodes.

---

*Design-only pass. Claude-Session: https://claude.ai/code/session_01P4Ns1rhyf7XZmwDcTfsUud*
