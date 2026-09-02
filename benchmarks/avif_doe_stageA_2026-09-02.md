# AVIF knob DOE — Stage A: the wave, and what it measured

**Date:** 2026-09-02 · **Design of record:**
[`avif_doe_plan_2026-09-01.md`](avif_doe_plan_2026-09-01.md) (§7.1 analysis,
§7.2 Stage-B triggers, §3.8 transfer gate, §3.9 bytes model, §12.4 degeneracy
correction) · **Lane:** wave completion + Stage-A analysis.

This document is the completion record for the svt-rs half of the AVIF knob DOE
and the pre-registered Stage-A read of it. It ends with a mechanical Stage-B
trigger list and a budget; **it does not declare Stage B** — that decision is
the coordinator's.

---

## 0. TL;DR

1. **The wave is COMPLETE.** All four svt-rs encode runs reached live-gap 0 with
   **zero failed cells**: A1 6,912, A2 33,984, A0R 6,496, AG 1,728 — 49,120
   cells, and every one of them scored.
2. **The A1 "480 failed-only" gap in the plan's §13.8 never existed.** A1's
   ledger shows the run finished at **04:47:21Z**, ~57 minutes before that
   snapshot was taken, with 6,912/6,912 live-done and **zero post-fix
   `encoder_panic`**. There is no poison to annotate and no port-bug lead here.
3. **Two of the seventeen knob arms are dead.** `tune=0` and
   `screen_content_mode=Some(3)` produce **byte-identical bitstreams to the
   default on 288 of 288 cells, at both presets** — while the harness's own
   resolved-state fingerprint says they are different configurations. That is a
   knob plumbed to the fingerprint but not to the bitstream. It consumed
   **8,972 cells** of this wave and must not consume any of Stage B's. Filed
   with the port program as
   [imazen/zenav1-svt#17](https://github.com/imazen/zenav1-svt/issues/17)
   (2026-09-02).
4. **The plan's §3.2 cell arithmetic is impossible under its own isolation
   rule.** §3.2 registers "17 arms × all 7 effective presets" = 34,272 cells,
   but `--max-deviations 1` means a non-default preset *is* the one permitted
   deviation, so a cell cannot carry both. zenavif's own test asserts the real
   design: **24 strata**. The declared 6,912 is correct; the doc is wrong. The
   consequence is load-bearing: **knob main effects exist at two presets (s4 and
   s6), not seven**, so §7.1(3)'s "per speed" and trigger **B-5** are evaluable
   across exactly one preset pair.
5. **The transfer gate certifies only 2 of 16 knobs for reduced-size
   screening** — `mtx32` and `qml1.8.15` PASS; 7 hold direction but not
   magnitude; **`acb3` and `shp3` genuinely fail** and fire B-6; 3 are
   unmeasurable and 2 are the inert pair. It runs on **11 references** at
   **speed 4 only** (§12.4 + §4.2), so it is thin — but its null check is
   perfect (0 violations on all 19 passthroughs).
6. **Tiling's bitrate cost is a reduced-size ARTEFACT.** `tl1.0` goes **+0.65%
   at 1024² to −0.12% at native** (sign flip) and `tl1.1` shrinks **8.6×**,
   with the only two significant T3 sign tests in the table (p = 0.012 and
   0.001). Tile overhead is roughly fixed per tile, so it is material on a 1 MP
   crop and negligible on a 12-16 MP frame. §6's "tiling costs bits on every
   image" **does not carry to native size**.
7. **Honouring every Stage-B trigger costs 447,636 cells against a registered
   60,000 envelope — 7.5×.** The trigger list is mechanical and published;
   prioritisation is a decision and **no Stage-B wave is declared here**.
8. **Five integrity gates passed**, none of them assumed: the AG corpus
   identity (3 independent proofs), A0-native ≡ A0R on **3,857/3,857** shared
   passthrough cells, A0R ≡ the in-run controls on **2,304/2,304**, BD-rate
   bit-parity against zenavif's own implementation, and the transfer gate's own
   passthrough null.

---

## 1. Wave completion

### 1.1 Final state

All counts from `zenfleet-ctl report` (live-gap semantics) against the LAN store.

| run | grid | declared | live_done | failed-only | verdict |
|---|---|--:|--:|--:|---|
| `avifdoe-svt-a1-20260901` | main effects, budget | 6,912 | 6,912 | 0 | COMPLETE |
| `avifdoe-svt-a2-20260901` | pairwise, budget | 33,984 | 33,984 | 0 | COMPLETE |
| `avifdoe-svt-a0r-20260901` | same-size control, budget | 6,496 | 6,496 | 0 | COMPLETE |
| `avifdoe-svt-ag-20260901` | transfer gate, **native** | 1,728 | 1,728 | 0 | COMPLETE |
| **total** | | **49,120** | **49,120** | **0** | |

**A0R and AG had no encode workers at all** when this lane opened — and A0R is
the control arm every knob effect is differenced against, so it was the critical
path. Both were launched here; neither produced a single failed cell.

### 1.2 The A1 gap that had already closed

The plan's §13.8 recorded `A1 ... live_done 6,432, gap 480, failed-only`. That
was a **stale read**, not a real gap. Reduced from A1's own 501 ledger files
(26,208 rows):

| era | status | error_class | rows |
|---|---|---|--:|
| pre-fix | `done` | — | **480** |
| pre-fix | `failed` | `encoder_panic` | 6,432 |
| pre-fix | `poison` | `encoder_panic` | 6,432 |
| post-fix | `failed` | `worker_lost` | 6,432 ← the §11.7 `requeue-pardon` rows |
| post-fix | `done` | — | **6,432** |

- **Zero post-fix `encoder_panic`.** Every panic predates the image fix, exactly
  as §11.7's `--before` scoping predicted, and the pardon did not launder a
  live failure.
- **480 + 6,432 = 6,912**, a clean partition: the 480 are the cells that
  succeeded before the stale image was replaced, the 6,432 are the pardoned
  remainder.
- Last ledger write **04:47:21Z**; the §13.8 snapshot that reported a 480 gap
  was taken at 05:44Z. The likely cause is a listing undercount on the object
  store — the same caveat the subsample doc already records for `s5cmd` — and it
  is worth knowing that `report` can under-read a *finished* run.

**So: no cells were requeued, no poison was annotated, and A1 contributes no
port-bug lead.** The only structural encoder finding in this wave is §3 below,
and it is not a failure — it is a silence.

### 1.3 What was launched, and the launcher gap it exposed

`scripts/jobsys/lan_score_launch.sh` — the canonical LAN launcher — forwarded
`ZEN_CORPUS_PREFIX` but **not** `ZEN_CORPUS_BUCKET`, so `jobexec`
(`jobexec.rs:400`) fell back to the *run* bucket when resolving
`cell.image_path`. Every corpus living in its own read-only bucket — the normal
case, `codec-corpus` — was therefore unreachable through it, which is why the
earlier encode arms were launched by hand. Fixed additively (`e9a64c60`); the
two launches below are its live verification.

| host | worker | run | cap | note |
|---|---|---|---|---|
| r7900x | `doe-a0r`, `doe-a0r2`, `doe-a0r3` | A0R | 16g/8g/8g | one worker was CPU-starved by the co-resident uncapped aom container (100% vs 1001%) with ~13 cores idle; **three workers took A0R from 1,803 to 6,132 cells/h (3.4×)** |
| tower | `doe-ag` | AG | cpuset 0-7, shares 256, 12g | leaves 10 of 32 cores free — the household floor is 8 |
| dev | `doe-ag-local` | AG | cpuset 24-30, 16g | added only after the score run reached gap 0 and its workers went idle; peak RSS 1.3 GiB against the cap |

The scale-up is the reusable lesson: **on a box shared with an uncapped
container, one zenfleet worker does not expand to fill the idle cores — add
workers, not cores.** The lease-based claim makes that safe by construction.

### 1.4 A recurring score declaration re-does its own work, and the cause is row order

The DOE score run finished with **`declared=4,128` and `ever_done=16,476`** — a
**4.0× multiplier** on job completions for a fixed body of work, with no
failures anywhere (`rescore tax 1.01x`, `errors=0`). The waste is not retries.

**Measured cause:** `zenfleet-ctl pairs`, run twice against a *frozen* ledger,
emits **the same set of rows in a different order** (verified on AG: `same SET
of encode_sha? YES`, byte-identical file? no). `LedgerView` stores rows in a
`HashMap<JobId, _>`, so `view.rows()` yields Rust's per-process randomized hash
order. `declare-scorefiles` then turns that order into job identity, and every
5-minute gap-fill round re-declared the whole run as fresh jobs while workers
redid cells that were already scored.

It is a *waste* bug, not a correctness bug — every blob is valid and the ledger
converges, because identity is content-addressed — but it multiplies the cost of
exactly the pattern the job system recommends (a recurring, idempotent
declaration loop).

> **FIXED 2026-09-02** in zenmetrics `08215e84`, with two corrections to the
> paragraph above that changed the decision to land it.
>
> **1. The mechanism is chunk MEMBERSHIP, not input order.** `JobId::of` sorts
> and dedups its inputs (`ids.rs:69-73`, "so input order can't change the
> identity"), so member order *within* a chunk cannot move an id. What moves it
> is which members share a chunk: `declare-scorefiles` cuts each ref's member
> list into contiguous `--chunk` slices, so a permutation re-cuts membership —
> but **only when a ref has more members than `--chunk`**. This run was
> maximally exposed: 32 refs × 203 members at `--chunk 12`, i.e. ~17 chunks per
> ref, every one re-cut every round. A run whose refs fit inside one chunk was
> never affected. That distinction is what decides whether any *other* run is
> exposed, and it is now pinned by the test
> `only_refs_larger_than_the_chunk_can_remint`. It also explains why `declared`
> stayed pinned at exactly 4,128 while the identities rotated: a permutation
> changes which members pair up, never how many chunks fall out.
>
> **2. "Not applied here" was the wrong call, because the run was not settling —
> it was compounding.** The hold assumed the churn was a bounded historical
> cost. It was live. Over rounds 37-40 the encode side sat frozen at 49,120 DONE
> cells and `declared` stayed pinned at 4,128, yet score blobs climbed
> **25,818 → 26,251 → 26,973 → 27,639** — the same 4,128 jobs re-minted under
> new identities every five minutes, with four workers re-scoring finished
> cells. Re-measured at the moment the fix landed: `ever_done=29,664` against
> `declared=4,128` — the multiplier had gone **4.0× → 7.19×**, i.e. ~7 full
> passes over one pass of work, and was still rising. (Score blobs track
> distinct completed `job_id`s about 1:1 — 29,608 against `ever_done` 29,664 —
> which is why the blob count rose in step with the churn rather than with
> coverage.) **Such a run never settles**: its gap closes each round and the
> next round re-mints it, which is why `report` could say
> `VERDICT: COMPLETE — live-gap==0` throughout. Holding the fix did not avoid
> churn; it extended it. There was also no mid-flight hazard to weigh: the
> encode ledgers are untouched by this change (encode job ids do not come from
> pairs ordering), and the score run's ids were already rotating every round —
> landing a stable sort simply made the next rotation the last one.
>
> The sort key leads with the emitted cell identity and ends with `job_id` as
> the tie-break — unique by construction, being the `LedgerView` map key — so
> the order is total. Verified live: three separate `pairs` processes over a
> frozen 6,496-row ledger produced byte-identical `.tsv` and `.parquet`
> (sha `148165c7…`), and two `declare-scorefiles` runs produced byte-identical
> manifests (sha `2ce81289…`). Both gate tests were confirmed to fail with the
> sort defeated.

---

## 2. The AG corpus identity — proven three ways

A1/A2/A0R encode the **1024-budget crop** corpus; AG encodes the **native** one.
The two corpora share **all 32 filenames**, so pointing AG's scoring at the crop
prefix would silently compare native encodes against crop references. This is
the §12.3 collision, and it is the reason the previous lane refused to declare AG
scoring at all. It is now settled by measurement, not by reading the plan:

| # | layer | result |
|--:|---|---|
| 1 | **corpus** — sha256 of all 32 files in each prefix | **19 byte-identical** passthroughs, **13 genuinely different** crops |
| 2 | **manifest** — declared `inputs[0]` vs the file sha | AG == **native** sha **32/32**; A0R == **crop** sha **32/32** |
| 3 | **runtime** — an AG blob for a *cropped* reference (`6006`) | `ispe` = **2320×3408** (native), not 1024×1024 (crop) |

Layer 3 is the one that matters: layers 1-2 prove what was *declared*, layer 3
proves what the worker *fetched*. Cross-check: AG's declared sha equals the crop
sha on exactly **19/32** — precisely the passthroughs, where the two are the same
bytes.

Scoring was then declared with `--refs-prefix
s3://codec-corpus/avif-subsample-2026-09-01/` and verified live: **AG's pairs
rows resolve 100% under the native prefix, A0R's 100% under the crop prefix.**
The wiring is committed in `scripts/jobsys/avifdoe_score_gapfill.sh` with the
proof in its header, so the next reader does not have to re-derive it.

---

## 3. THE FINDING: two knob arms never reach the bitstream

This is the wave's most actionable result and it does not depend on any modelling
choice — it is byte equality.

### 3.1 What was measured

For every single-deviation arm, the fraction of its `(image, q)` cells whose
encoded bitstream is **byte-identical to the default arm** at the same
`(image, speed, q)`:

| knob | meaning | identical @ s4 | identical @ s6 |
|---|---|--:|--:|
| **`scm3`** | `screen_content_mode = Some(3)` | **288/288 = 100%** | **288/288 = 100%** |
| **`tn0`** | `tune = 0` (VQ) | **288/288 = 100%** | **288/288 = 100%** |
| `mtx32` | `max_tx_size = 32` | 72/288 = 25.0% | 79/288 = 27.4% |
| `qml1.8.15` | QM window (8,15) | 32/288 = 11.1% | 32/288 = 11.1% |
| `acb1` | `ac_bias = 1.0` | 17/288 = 5.9% | 37/288 = 12.9% |
| `acb3` | `ac_bias = 3.0` | 16/288 = 5.6% | 24/288 = 8.3% |
| `shp3` / `shp7` | `sharpness` 3 / 7 | 5.6% / 5.2% | 0% / 0% |
| `bd10`, `qml1.2.10`, `qml1.4.10`, `tl1.0`, `tl1.1`, `tn3`, `vbst×3` | — | 0% | 0% |

`scm3` and `tn0` are not weak effects. They are **no effect at all**, on every
image, at every quality point, at both presets.

### 3.2 The knob reaches the fingerprint but not the encoder

The harness's own resolved-state fingerprint — the hash that is supposed to make
two configurations distinguishable — **does** separate them:

| arm | fp @ q5 | fp @ q45 | fp @ q96 |
|---|---|---|---|
| `s4-svt-420` (default) | `39121e954d30bfe5` | `525f02196fc98147` | `27ef653437e59dc7` |
| `s4-svt-420-tn0` | `ee1082fec17e38e5` | `98b717369c47c847` | `c19e9bac140044c7` |
| `s4-svt-420-scm3` | `edf9fd1ddcd7f88b` | `66969d4a0b2866c1` | `bc0f079c597a1b41` |
| `s4-svt-420-tn3` | `cd9ed696474abe9a` | `a3567e1386ad8a08` | `d3624754e847ea88` |

So the DOE believed it was varying something. **Minimal repro** — image
`7004.scale1024x1024.png`, q 45, speed 4, plan `svt_doe_main`, corpus
`avif-doe-1024-2026-09-01`:

| arm | `encode_sha` | bytes |
|---|---|--:|
| `s4-svt-420` | `e07ecbc4f28e4339…` | 52,306 |
| `s4-svt-420-tn0` | `e07ecbc4f28e4339…` | 52,306 ← **same bitstream** |
| `s4-svt-420-scm3` | `e07ecbc4f28e4339…` | 52,306 ← **same bitstream** |
| `s4-svt-420-tn3` | `24cfb87a676e5b69…` | 53,709 ← tune *does* work at 3 |

**`tune` is partly wired: `tune=3` changes the bitstream, `tune=0` does not.**
That narrows the lead considerably — it is not "tune is ignored", it is "the
tune-0 path resolves to the same encoder state as the default tune-1 path". For
`screen_content_mode`, `Some(3)` is a candidate out-of-range value (the plan's
own hazard H-10 documents clamping of out-of-band `pub` fields).

**Not diagnosed further and deliberately not fixed here** — zenavif is another
repo and another lane's subject. The repro above is the deliverable for the AV1
port program, and is now filed as
[imazen/zenav1-svt#17](https://github.com/imazen/zenav1-svt/issues/17) with one
lead read from the consumer side: `zenavif/src/expert.rs:258-266` documents that
only tunes 3/4 rewrite other config fields (via the port's
`apply_tune_overrides`), which would make tune 0 and the default tune 1 resolve
to identical encoder state unless something downstream reads `tune` directly.
The issue also notes that the existing parity test
`resolved_matches_the_port_tune_overrides` compares **config to config**, so it
cannot catch a field that resolves correctly and is then never consumed.

### 3.3 The blast radius, and why 27 A2 "interactions" are not interactions

Because the two arms are exact no-ops, **every pair containing one of them is a
byte-identical alias of the other knob's single arm.** Verified, not inferred —
all 27 such A2 pairs are identical on **288/288** cells each, and `tn0-scm3` is
an alias of the bare control:

```
s6-svt-420-tn0-mtx32      ==  s6-svt-420-mtx32       288/288
s6-svt-420-scm3-qml1.2.10 ==  s6-svt-420-qml1.2.10   288/288
s6-svt-420-tn0-scm3       ==  s6-svt-420             288/288
   … 27 of 27 pairs, all exact
```

| block | cells on an inert arm | share of block |
|---|--:|--:|
| A1 | 576 | 8.3% |
| A2 | 8,352 | 24.6% |
| AG | 44 | 2.5% |
| **total** | **8,972** | **21.8% of A1+A2+AG** |

Of A2's 118 declared strata, **29 (24.6%) carry no information**: 2 singles that
equal the control and 27 pairs that equal a single. The interaction analysis
below therefore reports on **74 informative pairs**, and the Stage-B trigger
evaluation excludes inert knobs by construction rather than letting them
silently score a residual of exactly zero.

This is not wasted measurement — "these two knobs do nothing" is a genuine and
useful result. It is wasted *repetition* if Stage B re-buys it.

---

## 4. Registered vs observed

| block | registered (§) | declared | what the data actually is | status |
|---|---|--:|---|---|
| **A0R** | §3.1 — 7 presets × 29 q × 32 img, budget | 6,496 | exactly that; labels `s1..s7-svt-420` | **as registered** |
| **A1** | §3.2 — "17 arms × all 7 effective presets × 9 q × 32" = **34,272** | **6,912** | 24 strata × 9 q × 32 = 1 default + 16 knob arms + 6 non-default speeds + 1 bit-depth, **all knob arms at speed 4** | **doc arithmetic wrong; data correct** |
| **A2** | §3.4 — 126 pairs + singles × speed 6 × 9 q × 32 | 33,984 | 118 strata × 9 q × 32 = 1 control + 16 singles + **101** pairs (`bd10` absent) | as registered, post-merge |
| **AG** | §3.8 — 17 arms × speeds **{4,6}** × 3 q × 32 = 3,264 | **1,728** | 18 strata × 3 q × 32 = 16 knob arms **at s4 only** (no `bd10`) + s4 control + a bare s6 | **speed axis collapsed** |
| **A3** (aom main effects) | §3.5 — 23 arms | — | **never declared** | ABSENT |
| **A4** (timing block) | §3.6 — 6-point size ladder | — | **never declared** | ABSENT |

### 4.1 The A1 discrepancy is the plan's error, not the wave's

§3.2 asks for 17 knob arms crossed with all 7 presets. That is **not expressible
under `--max-deviations 1`**: speed 4 is index 0, so a non-default preset *is*
the one permitted deviation and cannot be combined with a knob deviation.
zenavif's own test states the real design in a comment and asserts it:

```rust
// svt_doe_main_at_one_deviation_is_isolated_main_effects  (zenavif src/sweep.rs)
// speed 4 and bit-depth Auto are index 0, so: 1 default + 16
// knob arms + 6 speeds + 1 bit-depth = 24 strata at one q.
assert_eq!(plan.cells.len(), 24, …);
```

24 × 9 × 32 = **6,912** — the declared count, to the cell. So the declaration is
right and §3.2's "34,272" (and the 12.1 CPU-h / "speed 1 alone is 8.75" cost
model built on it) describes a grid that could never have been declared.

**Consequences, all load-bearing:**

- §7.1(3)'s main effects "per speed" exist at **two** presets — s4 from A1 and
  s6 from A2's singles — not seven.
- **B-5** (main effect inverts sign across the preset ladder) is evaluable
  across exactly **one preset pair**, s4 vs s6, for the 16 knobs present in
  both. `bd10` appears at s4 only, so B-5 is **NOT COMPUTABLE** for it.
- §3.3's stated reason for folding A1b away — "presets 0 and 1 at 9 q instead of
  3" — did not happen: no knob arm exists at preset 0 or 1. A1b's scientific
  purpose (does a knob invert at the slow end?) is **still open**, and is now a
  Stage-B question rather than something A1 answered.
- §7.3's de-scope ladder ("A1 → drop speed 1", "→ drop speed 3") would have shed
  *bare preset controls*, not knob×preset cells.

### 4.2 AG's speed axis collapsed the same way

`svt_doe_transfer` sets `speeds = vec![4, 6]`, but under `--max-deviations 1` the
s6 cells spend their deviation on the preset, so the only s6 stratum that
survives is the **bare control**. The gate therefore runs at **speed 4 only**,
and `bd10` — which AG also drops — has **no transfer evidence at all**.

### 4.3 Both absent blocks matter, and one cannot be recovered from this data

- **A3 (aom-rs main effects) was never declared.** Every knob result in this
  document is **svt-rs only**. There is no aom axis to report and none can be
  derived; the "top knobs per backend" ask has exactly one backend.
- **A4 (the timing block) was never declared, and speed is not recoverable
  here.** `encode_ms` is not persisted by the fleet path — confirmed against the
  ledger schema (`job_id, image_path, codec, q, knob_tuple_json, output_sha,
  status, error_class, attempts, ts, worker, provider, kind_json`) and the score
  blobs (`kind, image_path, codec, encode_sha, metric, score, scores, runtime` /
  `regime, features`). Neither carries a duration, and `ts` is a ledger-write
  time on concurrently-chunked work. **No speed model is reported and none is
  estimated** — §3.9 already registered that the size pair buys the bytes model
  everything and the speed model nothing.

---

## 5. Method, and the gates that make it trustworthy

### 5.1 Instruments

| step | owner | note |
|---|---|---|
| score-blob → tidy table | `scripts/jobsys/avifdoe_harvest.py` | joins on `encode_sha`; computes no statistics |
| RD / BD-rate / main effects / interactions | `scripts/jobsys/avifdoe_stagea_analyze.py` | §7.1 |
| transfer gate / bytes model / triggers | `scripts/jobsys/avifdoe_stagea_gates.py` | §3.8, §3.9, §7.2 |
| rank statistics | **`zenstats`** via `zensim/scripts/lib/zen_stats.py` | nothing correlation-shaped is hand-rolled |
| BD-rate | mirrors `zenavif/scripts/rd_gap/bd_arm.py` | **parity-gated, see 5.2** |

**Rows are heterogeneous and the code keys on that.** The score run changed
metric sets mid-flight (§13.5), so every row carries `metrics_present`:

| `metrics_present` | rows |
|---|--:|
| `ssim2 \| zensim_features` (2-metric, post-boundary) | majority |
| `butteraugli \| butteraugli_max \| butteraugli_pnorm3 \| ssim2 \| zensim_features` | the pre-boundary remainder |

**`ssim2` is the only corpus-wide scalar quality response**, exactly as §7.1
designates it. `zensim` is emitted as a 720-wide **feature vector**
(`kind:"feature"`, `regime:"v2-ab"`) and carries **no scalar**, so it is not a
second response without running a model over it. `butteraugli` covers only the
pre-boundary pairs and is **not** a corpus-wide column — it is not used for any
ranking here.

**Cells, not bitstreams, are the row set.** Encoding is content-addressed, so
the 42k cells collapse to ~28k distinct blobs; keying rows on `encode_sha` would
drop one of every byte-identical pair and destroy precisely the inertness signal
of §3. The harvester iterates cells and attaches scores to them.

### 5.2 Four gates, all passed, none assumed

| gate | what it rules out | result |
|---|---|---|
| **BD-rate parity** | a private BD-rate that quietly disagrees with the house one | identical to `zenavif/scripts/rd_gap/bd_arm.py` on 200 random ladders, **max \|Δ\| = 0.0 exactly** |
| **A0-native ≡ A0R on passthroughs** | the two size legs differing in *configuration* rather than in pixels — which would confound every BD-rate and the whole bytes model | **3,857 / 3,857 shared `(image, speed, q)` cells byte-identical, 0 differ** |
| **A0R ≡ in-run controls** | cross-run differencing being invalid | **2,304 / 2,304 byte-identical, 0 differ** |
| **AG passthrough null** | the transfer gate measuring a pipeline artefact instead of a size effect | on the 19 passthroughs the native and budget legs are the same encode; **0 residual violations** |

The second and third are what license the whole differencing scheme: A0R can
serve as the control for A1 and A2 arms because it is provably the *same
encoder configuration*, and the bytes model can attribute native-vs-budget
differences to pixel count because nothing else differs.

### 5.3 Exclusions, stated rather than silent

**Two references are excluded from BD-rate with cause: `6006` and `6018`.** Both
are crops of scanned patent pages that landed on near-blank regions. Their rate
does not respond to quality at all and their `ssim2` saturates:

| reference | bytes q5 → q96 | ratio | ssim2 q5 → q96 |
|---|---|--:|---|
| `6006.scale2320x3408` | 6,486 → 7,045 | **1.1×** | 71.9 → **100.0** |
| `6018.scale2320x3408` | 29,215 → 29,387 | **1.0×** | 74.4 → **100.0** |

A Pareto frontier over such a ladder collapses below the 4-point minimum, so no
BD-rate is defined. This is a property of the content, not a pipeline fault, and
it is the same crop-degeneracy family §12.2 already flagged.

**No other reference is dropped.** An earlier pass appeared to lose `9654` and
`1634` as well; that was a stale score-blob snapshot, and both have complete,
well-behaved ladders once the sync caught up. Recording it because it is the
failure mode most likely to recur: **a partial sync looks exactly like
degenerate content.**

### 5.4 Content classes are small, and the class verdicts need reading with care

The 32 references carry 12 fine content classes, sized 6, 5, 4, 3, 3, 3, 2, 2,
1, 1, 1, 1. A median over n = 1 is one observation. Results are therefore
reported over **five coarse families** — `photo` (7), `plot` (6),
`screenshot` (5), `ai-gen` (9), `scan` (5) — with the fine class kept in the
per-image table. The plan's own "11/32 screenshot+plot" split (§2.1) is exactly
`plot` + `screenshot` here.

**The G-CROP `feature_recheck` verdict must be read beside `parent_z_dist`, and
three crops are flagged wherever a class-keyed result uses them** (features
sha `133b93d8…`):

| reference | verdict | `parent_z_dist` | why it is flagged |
|---|---|--:|---|
| `1442.scale4000x3000` | **SHIFTED** 0→25 | **24.41** | genuinely changed cluster |
| `1634.scale3000x4000` | **SHIFTED** 25→10 | 4.58 | genuinely changed cluster |
| `6604.scale3286x4868` | "preserved" | **67.72** | *trivially* preserved — it stayed only because nothing was nearer; the largest z in the corpus |
| `6602.scale3302x4844` | "preserved" | 16.15 | same shape, less extreme |

Four of the twelve fine classes have a single member, so "preserved" is
trivially true for them too. **Class medians in §8 are directional evidence, not
estimates**, and any B-3 trigger resting on a class with n < 3 is marked
PROVISIONAL in the trigger table.

---

## 6. Main effects — svt-rs, two presets

**Sign convention: NEGATIVE BD-rate = the arm needs FEWER bits at matched
quality = the arm WINS.** Effect size is BD-rate against the `deviations = 0`
control at the same `(image, speed)` and the same size (§7.1(2)). The control
is **A0R's dense 29-q same-size ladder** — what A0R was built for (§3.1) — used
for **100% of the 4,102 arm cells**; the arms' own 9-q `deviations = 0` stratum
is reported as a robustness arm in §6.2. `tn0` and `scm3` are omitted: their
BD-rate is exactly 0 by construction (§3), not by measurement.

CI is a 10,000-resample percentile bootstrap over images (seeded). `wins / n`
counts images needing fewer bits. `worst image` is the arm's largest per-image
loss — the column that matters for a knob you would enable by default.

**Speed 4 — all 15 live knob arms, ranked by |median BD-rate|**

| rank | knob | median BD-rate | 95% CI | IQR | wins / n | worst image |
|--:|---|--:|---|--:|--:|--:|
| 1 | `shp7` | **+6.52%** | [+3.19, +7.01] | 7.20 | 3 / 30 | +12.3% |
| 2 | `tn3` | **-6.03%** | [-11.05, +0.27] | 14.04 | 20 / 30 | +21.2% |
| 3 | `tl1.1` | **+3.81%** | [+2.35, +6.57] | 6.22 | 1 / 31 | +57.1% |
| 4 | `vbst1.3.5` | **+2.36%** | [-0.82, +4.68] | 11.99 | 13 / 30 | +85.7% |
| 5 | `tl1.0` | **+2.34%** | [+1.36, +3.65] | 3.08 | 2 / 30 | +9.1% |
| 6 | `vbst1.2.5` | **+1.87%** | [-1.70, +3.55] | 7.23 | 13 / 30 | +72.4% |
| 7 | `shp3` | **+1.83%** | [+0.55, +2.16] | 2.12 | 5 / 30 | +4.5% |
| 8 | `bd10` | **-1.02%** | [-1.23, -0.36] | 1.53 | 23 / 31 | +5.9% |
| 9 | `vbst1.3.7` | **-0.66%** | [-1.87, +0.18] | 4.36 | 20 / 30 | +12.0% |
| 10 | `acb3` | **+0.54%** | [-0.02, +0.87] | 1.55 | 10 / 30 | +3.1% |
| 11 | `qml1.2.10` | **+0.51%** | [-0.76, +1.13] | 2.41 | 13 / 30 | +8.1% |
| 12 | `acb1` | **+0.42%** | [+0.13, +0.74] | 0.97 | 7 / 30 | +2.6% |
| 13 | `mtx32` | **+0.42%** | [-0.31, +1.96] | 2.93 | 12 / 30 | +18.9% |
| 14 | `qml1.4.10` | **+0.22%** | [-0.75, +0.89] | 2.48 | 14 / 30 | +6.0% |
| 15 | `qml1.8.15` | **+0.09%** | [-0.22, +0.71] | 1.36 | 13 / 30 | +3.0% |

**Speed 6 — all 14 live knob arms, ranked by |median BD-rate|**

| rank | knob | median BD-rate | 95% CI | IQR | wins / n | worst image |
|--:|---|--:|---|--:|--:|--:|
| 1 | `shp7` | **+7.73%** | [+6.60, +10.03] | 6.63 | 4 / 31 | +14.4% |
| 2 | `tn3` | **-4.23%** | [-10.84, -1.88] | 12.82 | 22 / 31 | +20.4% |
| 3 | `tl1.1` | **+2.44%** | [+1.27, +2.86] | 2.51 | 0 / 31 | +7.3% |
| 4 | `shp3` | **+2.04%** | [+1.51, +3.01] | 2.39 | 5 / 31 | +8.3% |
| 5 | `qml1.2.10` | **-1.89%** | [-4.00, -1.18] | 4.16 | 23 / 31 | +7.9% |
| 6 | `qml1.4.10` | **-1.79%** | [-3.27, -0.97] | 3.88 | 23 / 31 | +5.8% |
| 7 | `tl1.0` | **+1.45%** | [+0.92, +1.80] | 1.42 | 2 / 31 | +5.9% |
| 8 | `vbst1.3.7` | **-1.27%** | [-2.13, +0.10] | 3.40 | 19 / 30 | +13.4% |
| 9 | `vbst1.3.5` | **+0.70%** | [-0.54, +4.26] | 9.46 | 12 / 30 | +52.4% |
| 10 | `mtx32` | **+0.34%** | [-0.02, +0.87] | 1.66 | 12 / 31 | +21.3% |
| 11 | `vbst1.2.5` | **+0.28%** | [-1.38, +2.84] | 7.88 | 13 / 30 | +45.1% |
| 12 | `acb3` | **+0.20%** | [+0.08, +0.43] | 0.63 | 8 / 31 | +2.5% |
| 13 | `qml1.8.15` | **-0.19%** | [-0.45, +0.27] | 1.24 | 16 / 31 | +1.8% |
| 14 | `acb1` | **+0.18%** | [-0.04, +0.47] | 0.79 | 11 / 31 | +2.5% |
### 6.1 What the ranking says

- **`shp7` (sharpness 7) is the most expensive knob in the set and the most
  consistently expensive** — the largest effect at speed 6, second at speed 4,
  CI clear of zero at both, losing on 27-28 of 31 images. Both backends' image
  tunes force sharpness 7 (§3.2 #4), so this is a direct cost measurement of a
  setting the upstream defaults reach for.
- **`tn3` (tune = IQ) is the largest win and the least reliable one** — the best
  median at both presets, but IQR ~13-14 and a worst image of **+21%**. It wins
  on about two thirds of images and loses badly on the rest. That is a
  content-conditional rule, not a default; at speed 4 its CI even crosses zero.
- **Tiling costs bits at 1024², monotonically in tile count** — `tl1.0` then
  `tl1.1`, positive on 29-31 of 31 images at both presets, tight CI. **But see
  §8.3: this is the one effect the transfer gate shows to be a reduced-size
  artefact.**
- **The QM window knobs separate by preset.** `qml1.2.10` / `qml1.4.10` are
  indistinguishable from zero at speed 4 (CI spans it) and become the 5th/6th
  largest effects at speed 6, both winning on 23 of 31 images.
- **`ac_bias` is now explored, and it is nearly inert.** `acb1` / `acb3` sit at
  +0.2 to +0.5% with the smallest IQRs in the table, despite §3.2 #6 calling the
  axis "entirely unexplored". It is explored: at these levels it does almost
  nothing.
- **The variance-boost arms have small medians and enormous tails** — IQR up to
  12 and worst images of **+72% to +86%**. A median near zero here means "helps
  as often as it hurts", not "does nothing", and the tail is where the risk is.

### 6.2 The control choice does not move the top of the ranking

Recomputing every effect against the arms' own 9-q control instead of A0R's
dense ladder moves the medians by **0.386 pp (median), 0.986 pp (p90), 1.574 pp
(max)** across 33 (speed, knob) cells. **The top three are identical in both
arms at both presets** — `shp7`, `tn3`, `tl1.1` — while ranks 4-5 shuffle. So
the headline ordering is robust to the control; the mid-table ordering is not,
and no conclusion below rank 3 rests on rank alone.

### 6.3 Per content family

**Speed 6 — median BD-rate by content family** (n in parentheses; read across a row, and remember these are medians of 4-9 images)

| knob | photo | plot | screenshot | ai-gen | scan | spread |
|---|--:|--:|--:|--:|--:|--:|
| `vbst1.3.5` | -0.54 (7) | +0.28 (6) | +3.01 (5) | +5.09 (9) | -8.33 (3) | 13.43 |
| `tn3` | -2.20 (7) | +0.54 (6) | -11.57 (5) | -7.07 (9) | -12.56 (4) | 13.10 |
| `vbst1.2.5` | -2.25 (7) | +0.28 (6) | +1.50 (5) | +3.69 (9) | -8.57 (3) | 12.26 |
| `shp7` | +8.88 (7) | -0.13 (6) | +4.20 (5) | +11.09 (9) | +7.25 (4) | 11.23 |
| `qml1.2.10` | -1.77 (7) | +1.20 (6) | -2.52 (5) | -2.25 (9) | -2.77 (4) | 3.97 |
| `qml1.4.10` | -1.41 (7) | +1.29 (6) | -2.62 (5) | -1.88 (9) | -2.28 (4) | 3.91 |
| `mtx32` | +1.42 (7) | +0.05 (6) | -0.01 (5) | +0.87 (9) | -2.28 (4) | 3.70 |
| `shp3` | +2.40 (7) | -0.14 (6) | +0.86 (5) | +3.17 (9) | +3.07 (4) | 3.31 |
| `vbst1.3.7` | -1.37 (7) | +0.28 (6) | -2.26 (5) | -1.83 (9) | -0.04 (3) | 2.54 |
| `qml1.8.15` | -0.35 (7) | +1.19 (6) | -0.91 (5) | -0.19 (9) | -0.11 (4) | 2.10 |
| `tl1.1` | +2.54 (7) | +1.32 (6) | +1.72 (5) | +3.27 (9) | +2.90 (4) | 1.95 |
| `tl1.0` | +1.46 (7) | +0.73 (6) | +1.06 (5) | +1.80 (9) | +1.49 (4) | 1.08 |
| `acb1` | +0.59 (7) | +0.09 (6) | -0.20 (5) | +0.18 (9) | +0.15 (4) | 0.79 |
| `acb3` | +0.57 (7) | +0.14 (6) | +0.18 (5) | +0.13 (9) | +0.05 (4) | 0.53 |
**The `plot` family is nearly inert to the whole knob set.** `shp7` costs +8.9%
on photos, +11.1% on ai-gen and +7.3% on scans — and **−0.13% on plots**. Same
story for both variance-boost arms and `shp3`. Only the QM knobs and tiling move
plots at all, and the QM knobs move them in the **opposite direction** to every
other family.

The mechanism is measured, not guessed: **plot content has a compressed rate
dynamic range.** Median bytes ratio from q5 to q96, per family:

| family | median q96/q5 bytes ratio |
|---|--:|
| `plot` | **4.5×** |
| `screenshot` | 12.5× |
| everything else | **55.6×** |

A knob that shifts the rate-quality tradeoff has little room to act on a curve
that spans 4.5× in rate; two of the six plots do not exceed ssim2 54 at q96 at
all. **`scan` is the opposite extreme** and the most knob-sensitive family
(`tn3` −12.6%, both variance-boost arms −8.3 to −8.6%) — on n = 3-4, so
directional evidence only.

---

## 7. Interactions — speed 6

Interaction residual = observed pair BD-rate − (main effect of `k1` + main
effect of `k2`), per image (§7.1(4)). Of A2's 101 declared pairs, **27 are
byte-identical aliases of a single arm** (§3.3) and carry a structural zero;
this is the remaining **74 informative pairs**.

**Interactions at speed 6 — top 12 of 74 informative pairs, by mean |residual|**

| k1 | k2 | median resid | mean abs resid | images with abs ≥1% | max abs |
|---|---|--:|--:|--:|--:|
| `qml1.2.10` | `shp7` | -5.47% | 4.95% | 90% | 9.8% |
| `qml1.4.10` | `shp7` | -5.14% | 4.74% | 87% | 9.0% |
| `tn3` | `vbst1.3.7` | +0.63% | 4.53% | 73% | 15.0% |
| `vbst1.3.5` | `shp7` | +0.26% | 1.83% | 50% | 10.2% |
| `vbst1.2.5` | `shp7` | +0.16% | 1.63% | 50% | 8.2% |
| `qml1.2.10` | `shp3` | -1.64% | 1.62% | 68% | 3.9% |
| `qml1.4.10` | `shp3` | -1.51% | 1.53% | 68% | 3.5% |
| `vbst1.3.5` | `mtx32` | -0.09% | 1.23% | 40% | 8.7% |
| `vbst1.2.5` | `mtx32` | -0.08% | 1.22% | 43% | 7.6% |
| `vbst1.3.5` | `qml1.4.10` | -0.09% | 1.10% | 47% | 4.5% |
| `vbst1.3.7` | `shp7` | +0.32% | 1.09% | 33% | 6.1% |
| `qml1.8.15` | `shp7` | -0.77% | 1.08% | 35% | 4.1% |
### 7.1 Two families, and only one has a direction

**QM × sharpness is a strong, consistent, SYNERGISTIC interaction.**
`(qml1.2.10, shp7)` and `(qml1.4.10, shp7)` sit near **−5.2 to −5.5%** residual
on **87-90%** of images. `shp7` alone costs about +7.7% and `qml1.2.10` alone
saves about −1.9%, so the additive prediction is roughly +5.8% while the
observed pair is close to neutral — **enabling QM largely cancels the sharpness
penalty.** Both act on the quantisation path, so this is mechanistically
unsurprising and directly actionable: a tuner must not treat sharpness and QM as
separable, and §6.1's "sharpness is expensive" holds **only when QM is off**.
The same pattern repeats one level down with `shp3` (−1.5 to −1.6% on 68%).

**Variance-boost × anything is high-dispersion with no direction.**
`(tn3, vbst1.3.7)`, `(vbst*, shp7)`, `(vbst*, mtx32)`, `(vbst*, qml*)` all show
median residuals near zero with |residual| ≥ 1% on 33-73% of images and maxima
to 15%. An additive model will be wrong on a third to three quarters of images
for these pairs, in a direction it cannot predict — the signature of a
content-conditional interaction, matching variance-boost's main-effect shape
(§6.1: small median, enormous tail).

---

## 8. The cross-size transfer gate — the wave's own check on itself

A1 and A2 measure knobs at a 1024² pixel budget. §3.8 pre-registered the gate
that says whether a knob's effect there points the same way as at native size,
and it **blocks**: an arm that fails is not screened at reduced size.

Bars unchanged from §3.8: **T1** sign agreement ≥ 0.80 (counting only images
whose |BD-rate| at native ≥ 0.5%), **T2** Spearman of per-image BD-rates
native-vs-budget ≥ 0.70, **T3** median |BD_budget − BD_native| ≤ 1.0% with no
systematic sign (binomial p ≥ 0.05). Per §12.4 the tests run over the **13
cropped references only** — for the other 19 "budget" and "native" are the same
encode — less the two degenerate ladders of §5.3, giving **n = 11**. Per §4.2
the gate runs at **speed 4 only**, and `bd10` has no native leg at all.

| knob | n crop | n eff | T1 sign-agree (bar .80) | T2 SROCC (bar .70) | T3 med abs resid (bar 1.0%) | T3 binom p | verdict |
|---|--:|--:|--:|--:|--:|--:|---|
| `acb1` | 11 | 2 | 1.00 | 0.364 | 0.34 | 1.000 | NOT-MEASURED (only 2 of 11 cropped refs move >=0.5% at native, need 3) |
| `acb3` | 11 | 4 | 0.25 | 0.064 | 0.73 | 0.227 | FAIL-T1 (not screenable at budget) |
| `mtx32` | 11 | 5 | 1.00 | 0.891 | 0.40 | 1.000 | PASS |
| `qml1.2.10` | 11 | 11 | 1.00 | 0.682 | 0.88 | 1.000 | PARTIAL (direction holds; magnitude/rank flagged) |
| `qml1.4.10` | 11 | 11 | 1.00 | 0.627 | 0.90 | 0.549 | PARTIAL (direction holds; magnitude/rank flagged) |
| `qml1.8.15` | 11 | 11 | 0.91 | 0.736 | 0.34 | 0.549 | PASS |
| `scm3` | 11 | 0 | — | — | — | — | INERT (byte-identical to control; nothing to transfer) |
| `shp3` | 11 | 8 | 0.62 | 0.536 | 0.89 | 0.549 | FAIL-T1 (not screenable at budget) |
| `shp7` | 11 | 10 | 0.90 | 0.700 | 1.50 | 1.000 | PARTIAL (direction holds; magnitude/rank flagged) |
| `tl1.0` | 11 | 2 | 0.50 | 0.309 | 0.56 | 0.012 | NOT-MEASURED (only 2 of 11 cropped refs move >=0.5% at native, need 3) |
| `tl1.1` | 11 | 2 | 0.50 | 0.645 | 0.62 | 0.001 | NOT-MEASURED (only 2 of 11 cropped refs move >=0.5% at native, need 3) |
| `tn0` | 11 | 0 | — | — | — | — | INERT (byte-identical to control; nothing to transfer) |
| `tn3` | 11 | 11 | 0.91 | 0.827 | 4.34 | 1.000 | PARTIAL (direction holds; magnitude/rank flagged) |
| `vbst1.2.5` | 11 | 10 | 1.00 | 0.446 | 3.76 | 1.000 | PARTIAL (direction holds; magnitude/rank flagged) |
| `vbst1.3.5` | 11 | 10 | 1.00 | 0.536 | 3.51 | 1.000 | PARTIAL (direction holds; magnitude/rank flagged) |
| `vbst1.3.7` | 11 | 10 | 0.90 | 0.373 | 4.01 | 0.549 | PARTIAL (direction holds; magnitude/rank flagged) |
### 8.1 Verdict

| outcome | n | knobs |
|---|--:|---|
| **PASS** — screened at budget, Stage-B may stay at budget | **2** | `mtx32`, `qml1.8.15` |
| **PARTIAL** — direction holds, magnitude/rank flagged | **7** | `qml1.2.10`, `qml1.4.10`, `shp7`, `tn3`, `vbst1.2.5`, `vbst1.3.5`, `vbst1.3.7` |
| **FAIL-T1** — NOT screenable at reduced size | **2** | `acb3`, `shp3` |
| **NOT-MEASURED** — too few refs move ≥0.5% at native | **3** | `acb1`, `tl1.0`, `tl1.1` |
| **INERT** — nothing to transfer | **2** | `scm3`, `tn0` |

**The null check passes cleanly.** On all 19 passthrough references the native
and budget legs are the same encode and the residual is exactly zero on every
one — **0 identity violations**. The gate is measuring a size effect, not a
pipeline artefact.

**What this licenses, and what it does not.** Only `mtx32` and `qml1.8.15` are
certified for reduced-size screening. The 7 PARTIAL arms keep their **direction**
— which is what the Stage-B triggers key on — with magnitude size-conditional;
`tn3` and the variance-boost arms miss T3 by a wide margin (median residuals
3.5-4.3%), so **their effect SIZES at 1024² should not be quoted as native
numbers.** `acb3` and `shp3` genuinely fail direction and fire **B-6**.

### 8.2 The bar that could not be failed, and the one that could not be reached

§12.4 established that T3's median over 32 references could not fail because 19
of them are identity pairs. Restated over n = 11 it is a real test, and it
**does** fail four arms. Two further construction problems survive:

- **T1's denominator is the effect itself.** It counts only references moving
  ≥ 0.5% at native, so an arm whose effect *vanishes* at native reads as
  NOT-MEASURED rather than "does not transfer". That is precisely what happened
  to tiling — see §8.3, where the finding is real and the gate's own wording
  hides it.
- **`bd10` has no transfer evidence at all**, because §4.2's collapsed speed
  axis dropped it from AG.

### 8.3 Tiling's bitrate cost is a REDUCED-SIZE ARTEFACT

The strongest size-dependent result in the wave, and the gate found it by its
T3 sign test rather than its headline verdict. Median BD-rate over the same 11
cropped references, same 3-q ladder, budget leg vs native leg:

| knob | median BD @ 1024² budget | median BD @ native | T3 binomial p |
|---|--:|--:|--:|
| **`tl1.0`** | **+0.65%** | **−0.12%** | **0.012** |
| **`tl1.1`** | **+0.94%** | **+0.11%** | **0.00098** |
| `shp7` | +3.79% | +2.99% | 1.000 |
| `shp3` | +0.50% | +0.41% | 0.549 |
| `tn3` | −15.53% | −13.95% | 1.000 |
| `qml1.8.15` | −1.38% | −2.02% | 0.549 |
| `vbst1.3.5` | −9.29% | −7.99% | 1.000 |
| `mtx32` | +0.03% | −0.07% | 1.000 |
| `acb3` | +0.07% | −0.43% | 0.227 |

`tl1.0` **changes sign** and `tl1.1` shrinks **8.6×**. The binomial p-values —
0.012 and 0.001, the only two significant ones in the table — say the residual
has a systematic sign, i.e. the budget leg *consistently overstates* the tiling
cost. The mechanism is straightforward: tile overhead is roughly fixed per tile,
so it is material on a 1 MP crop and negligible on a 12-16 MP native frame.

**Consequence: §6.1's "tiling costs bits on every image" is a statement about
1024² only and must not be carried to native size.** Every other knob in the
table transfers with the same sign and within ~1.3× on magnitude.
---

## 9. The bytes model — §3.9's decomposition is not identifiable here

§3.9 registered `total = α + β·pixels` as **free from the fleet**: run the
control at both sizes and every `(image, speed, q)` cell yields two pixel
counts. Two things are wrong with that, one of them fatal.

**First, the same degeneracy §12.4 found in AG applies here and was not carried
across.** The 19 passthrough references have *identical* pixel counts in both
legs, so they give **zero leverage** on the intercept. The fit rests on the 13
cropped references — and per-image it is a two-point fit of two unknowns, so it
is exactly determined: no residual, no goodness-of-fit, no test.

**Second, and fatally: a crop is a different image, not a smaller one.** The
intercept is supposed to be container overhead — a few hundred bytes, constant
in quality. Measured, it is neither:

| | |
|---|---|
| fits (13 cropped refs × 7 speeds × 29 q) | **2,639** |
| median α over all fits | **7,463 bytes** (IQR −1,518 … 38,439) |
| α at q = 1 → α at q = 98 (medians) | **731 → 59,176 bytes**, an **81×** climb |
| **SROCC(α, q) across 91 (image, speed) groups** | **median 0.943** (min 0.014) |
| within-image α range (positive-α groups) | median **56×**, max **238×** |
| fits with α < 0 | **781 of 2,639** |

**A container header does not scale with quality.** α tracks q almost perfectly
because it is absorbing the content difference between a 1024² window and the
whole 3000×4000 frame, not a fixed cost — and a third of the fits put it below
zero, which a header cannot be. The one place it behaves is the degenerate
near-blank crop `6006`, whose content term nearly vanishes: at speed 4 its α
sits between **515 and 611 bytes across the entire 29-point ladder** (a 1.2×
range, SROCC 0.357). That figure and the **731-byte median at q = 1** are the
only honest indications in this data of what an AVIF container actually costs —
**roughly 0.5-0.7 KB**. Suggestive, not a fit.

**Conclusion: §3.9's "free" decomposition does not hold on a crop-based size
pair.** Getting α requires the *same content* at multiple sizes — a downscale
ladder — which §2.4 deliberately rejected to preserve native HF content. That
was the right call for the transfer gate's purpose and it costs the bytes model
entirely. The two goals are in tension and the plan did not notice.

---

## 10. Stage-B triggers — the mechanical list, and it does not fit

Evaluated exactly as §7.2 defines them, with inert knobs excluded by
construction (§3.3) and every B-3 firing carrying its class sizes (§5.4).

| trigger | n | cells if honoured | keys |
|---|--:|--:|---|
| **B-1** | 17 | 236,640 | `qml1.2.10@s6`, `qml1.4.10@s6`, `shp3@s4`, `shp3@s6`, `shp7@s4`, `shp7@s6`, `tl1.0@s4`, `tl1.1@s4`, `tl1.1@s6`, `tn3@s4`, `tn3@s6`, `vbst1.2.5@s4`, `vbst1.2.5@s6`, `vbst1.3.5@s4`, `vbst1.3.5@s6`, `vbst1.3.7@s4`, `vbst1.3.7@s6` |
| **B-2** | 23 | 59,616 | `(qml1.2.10,shp3)@s6`, `(qml1.2.10,shp7)@s6`, `(qml1.4.10,shp3)@s6`, `(qml1.4.10,shp7)@s6`, `(qml1.8.15,shp3)@s6`, `(qml1.8.15,shp7)@s6`, `(shp3,mtx32)@s6`, `(shp7,mtx32)@s6`, `(tn3,tl1.1)@s6`, `(tn3,vbst1.3.7)@s6`, `(vbst1.2.5,mtx32)@s6`, `(vbst1.2.5,qml1.2.10)@s6`, `(vbst1.2.5,qml1.4.10)@s6`, `(vbst1.2.5,shp3)@s6`, `(vbst1.2.5,shp7)@s6`, `(vbst1.3.5,mtx32)@s6`, `(vbst1.3.5,qml1.2.10)@s6`, `(vbst1.3.5,qml1.4.10)@s6`, `(vbst1.3.5,shp3)@s6`, `(vbst1.3.5,shp7)@s6`, `(vbst1.3.7,qml1.2.10)@s6`, `(vbst1.3.7,qml1.4.10)@s6`, `(vbst1.3.7,shp7)@s6` |
| **B-3** | 13 | 123,540 | `bd10@s4`, `mtx32@s4`, `mtx32@s6`, `qml1.2.10@s4`, `qml1.2.10@s6`, `qml1.4.10@s4`, `qml1.4.10@s6`, `tn3@s4`, `vbst1.2.5@s4`, `vbst1.2.5@s6`, `vbst1.3.5@s4`, `vbst1.3.5@s6`, `vbst1.3.7@s4` |
| **B-6** | 2 | 27,840 | `acb3`, `shp3` |
| **total** | **55** | **447,636** | vs a registered envelope of **60,000** = **7.5x** |

Median-BD-rate agreement between the two control arms over 33 (speed, knob) cells: median |Δ| **0.386 pp**, max **1.574 pp**.
**B-4 is NOT EVALUABLE** — it keys on the A4 speed fit's held-out MAPE, and A4
was never declared (§4.3). **B-5 does not fire**: no knob has |median| ≥ 1% at
both presets with opposite signs. That is a genuine null, but a narrow one — it
covers the single preset pair s4-vs-s6 that §4.1 leaves, and the slow end
(presets 0-1), where §3.3 expected inversions, is untested.

### 10.1 The budget does not fit, by 7.5×

| | cells |
|---|--:|
| every trigger honoured | **447,636** |
| §7.2's registered envelope | **60,000** |
| **overrun** | **7.5×** |

Costing follows §7.2's own follow-up definitions: B-1 = 5 levels × 29 q × 32
images × 3 speeds = 13,920 cells per knob; B-2 = 3×3 levels × 9 q × 32 images =
2,592 per pair; B-5/B-6 = B-1's grid at 2 presets / at native; B-3 = B-1's grid
restricted to the triggering classes' images, the cheapest reading of
"content-stratified dense".

**Prioritisation is a decision, not a computation, and it is the coordinator's.**
This lane does not declare Stage B. What the data supports, offered as input:

1. **Do not re-buy the inert knobs.** `tn0` and `scm3` consumed 8,972 cells of
   Stage A and are excluded from every trigger above; they should stay excluded
   until the port question in §3 is answered.
2. **The two B-6 arms are the cheapest high-value cells** — `acb3` and `shp3`
   at 27,840 cells total, and they are the only arms *proven* not screenable at
   reduced size, so leaving them at budget is known to be wrong.
3. **B-2's QM × sharpness cluster is the strongest measured structure** in the
   wave (§7.1) and its six pairs are ~15,552 cells — small, and the result is
   already directional rather than exploratory.
4. **B-1's variance-boost arms are the most expensive and the least certain.**
   Six of the 17 B-1 triggers are `vbst*`, and they trigger on IQR, not median —
   the follow-up would be characterising a tail, which a 5-level dense grid on
   32 images is not obviously the right instrument for.
5. **Anything keyed on `tl1.0` / `tl1.1` should be re-scoped to native size
   first** (§8.3), since the effect that triggered them is largely a
   reduced-size artefact.

---

## 11. Limitations — what this wave cannot tell you

1. **One backend.** A3 was never declared, so every number here is **svt-rs**.
   There is no aom-rs knob evidence in this wave and none can be derived from it.
2. **No speed axis, and none estimated.** `encode_ms` is not persisted by the
   fleet path (verified against both the ledger and the score-blob schemas), and
   A4 — the instrument that would have measured it — was never declared. §7.1(5)
   and trigger **B-4** are consequently **NOT EVALUABLE**. No `ms/MP` figure
   appears in this document, with or without an intercept.
3. **Two presets, not seven.** §4.1: `--max-deviations 1` makes a knob×preset
   cell inexpressible, so main effects exist at s4 and s6 only. The slow end
   (presets 0 and 1) — where SG restoration, Wiener, filter-intra and wedge
   prediction turn on, and where §3.3 expected inversions — is **untested**.
4. **Reduced-size screening is not certified.** §7 — 0 of 16 knobs pass the
   transfer gate, and 12 are not even evaluable at the n the 3-q ladder leaves.
   Every main effect and interaction here is measured at 1024² and is
   **provisional with respect to native size**.
5. **The bytes decomposition does not identify a container overhead.** §9 — a
   crop is a different image, not a smaller one, so the 2-point fit's intercept
   absorbs content and tracks quality (SROCC(α, q) = 0.99) instead of standing
   still. §3.9's "free from the fleet" claim does not hold.
6. **Content classes are small.** 12 fine classes over 32 references, four of
   them singletons; the coarse families run n = 5-9. Class medians are
   directional evidence, not estimates, and three crops carry a suspect class
   assignment (§5.4).
7. **`ssim2` is the only quality response.** `zensim` came back as a feature
   vector with no scalar and butteraugli covers only the pre-boundary pairs, so
   there is no second opinion on any BD-rate here. A knob that games ssim2
   specifically would not be caught by this wave — the standing veto protocol
   needs a second metric, and this data cannot supply one without running a
   model over the stored 720-wide vectors.
8. **Two references are excluded** (`6006`, `6018`) for degenerate ladders, and
   `bd10` has no s6 main effect, no interaction coverage and no transfer
   evidence — it exists only as an A1 arm at s4.

---

## 12. Where everything is

| artifact | path |
|---|---|
| scored dataset (49,120 × 18) | `/mnt/v/output/zensim-avifdoe/doe_scored_2026-09-02.parquet` |
| primary analysis (A0R dense control) | `/mnt/v/output/zensim-avifdoe/stagea_a0r/` |
| robustness arm (in-run 9-q control) | `/mnt/v/output/zensim-avifdoe/stagea_inrun/` |
| pointer + shas | `benchmarks/avif_doe_stageA_2026-09-02.pointer.md` |
| harvester / analyzer / gates | `scripts/jobsys/avifdoe_{harvest,stagea_analyze,stagea_gates}.py` |
| recurring score declaration | `scripts/jobsys/avifdoe_score_gapfill.sh` |

Per-directory tables: `main_effects.tsv`, `main_effects_by_class.tsv`,
`bd_per_image.tsv`, `interactions.tsv`, `interactions_per_image.tsv`,
`arm_byte_identity.tsv`, `ag_transfer_gate.tsv`, `ag_identity_violations.tsv`,
`bytes_alpha_beta.tsv`, `stage_b_triggers.tsv`, `stage_b_budget.json`.
