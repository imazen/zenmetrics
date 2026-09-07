# REV2 RECALCULATION — the `JobKind::Feature` executor and the LAN wave

**Pre-registered 2026-09-06.** Gates below were written **before** the code and
are never edited to match a result. A gate that fails is reported failed, not
re-scoped.

**User directive (verbatim, 2026-09-05):** *"we should fix arithmetic defects
aggressively before shipping, and perhaps change feature definitions and
formulas to make monotone linear models better. **we have a fleet to
recalculate**"*.

**What this lane owns:** the *fleet capability* to recompute every feature table
at a named formula revision, and the wave that does it. It does **not** own the
arithmetic (zensim's rev2 lane) or the choice of F4 arm (the R6 lane).

Source of truth for WHAT to recompute:
[zensim `benchmarks/feature_rev2_2026-09-05.md` §3](../../zensim/benchmarks/feature_rev2_2026-09-05.md)
— "Recalculation manifest — REGISTERED, NOT RUN".
Plan it executes: zensim `docs/PLAN_FEATURE_REV2_2026-09-05.md`.

---

## 0. The blocking fact, and what is NOT blocked by it

**A rev2 wave cannot be declared yet.** The F4 arm (`clamp` / `lorentz` / `c1`)
is undecided: zensim's record §1.6 measured that the cheap ladder proxy
**cannot discriminate** between the three arms (all four arms give identical
monotonicity on both control-passing ladders, because the ladders never enter
the regime where the arms differ), so R6's monotone-linear fit on real corpora
must run first. The R6 lane hands over an **arm token**; this lane polls
`benchmarks/feature_rev2_2026-09-05.md` §R6 for it on a long cadence and does
other work meanwhile.

**What is NOT blocked:** the executor, the image, and the *correctness gate*.
Revision 1 is byte-reproducible today, the local zensim checkout is at
`4fbd8ff8` — **exactly the `build_commit` the postC 372 root records** — and
the fleet's output at rev1 must be bit-exact to that root on the same pixels.
That is a stronger gate than anything rev2 can offer, because a stored answer
already exists to compare against. So: build and gate at **rev1** now, re-declare
at **rev2** when the token arrives.

---

## 1. Why `JobKind::Feature` gets an executor instead of `ScoreFile` + a metric name

`JobKind::Feature { regime }` exists in `crates/zenfleet-core/src/job.rs:175`,
has a `profile()` arm (`:333`, `CpuHeavy` / `GroupBy::SourceSha` /
`CheapRegenerable`), is named by the worker (`crates/zenfleet-worker/src/lib.rs:2108`)
and is in the worker's **default warm-eligible set** (`:3688` test,
`warm_kinds_from_env`) — but `crates/zenmetrics-cli/src/jobexec.rs::run_one_job`
special-cases only `score_file` and `diffmap`, so a `feature` job falls through
to `unhandled job kind` (`:1983`).

Feature tables are produced today through `ScoreFile` + a `zensim-foldapp2*`
metric name. That works and has shipped four waves. It is nonetheless the wrong
vehicle for a **revision-scoped recalculation**, for three reasons that are
properties of the wire, not preferences:

1. **JSON rows do not scale to 944 f64 columns.** The ScoreFile feature arm
   emits `{"kind":"feature", …,"features":[…944 floats…]}` — one JSON row per
   pair. At 5.74 M bigcodec rows that is the dominant cost of the wave, in
   bytes on the wire and in parse time on writeback. Parquet+zstd is the
   format every consumer already reads.
2. **A metric name cannot carry a formula revision.** `zensim-foldapp2` under
   `ZENSIM_FORMULA_REV=1` and under `=2` are *different work with different
   output bytes* — but they serialize to the identical `JobKind::ScoreFile`,
   so they collide on one content-addressed `JobId`. The ledger would call a
   rev2 cell "already done" because a rev1 cell with the same id exists. This
   is the exact failure the `hdr` flag was added to `Encode`/`ScoreFile` to
   avoid, and the fix is the same shape.
3. **`feature_set_id` has an owner and a wire contract** (zensim
   `docs/FEATURE_SET_IDS.md`, user directive 2026-09-05: *feature sets are
   NAMED, not counted*). A recalculation whose outputs do not carry
   `<compute>@w<layout>/<era>#<slots-hash8>` re-creates the "944 has named
   seven different feature sets" problem at fleet scale.

Per the zenfleet mandate — *"if the job system lacks a capability, BUILD IT
INTO zenfleet/zenmetrics — never bypass it"* — the executor is written.

---

## 2. Scope, in priority order

Priority follows the user's stated product order and the manifest's own
"pixels present?" column. **Features only — this lane declares no encode and no
score job.**

| # | wave | rows | why this order |
|---|---|--:|---|
| 1 | **rev1 gate** — postC 372 corpora, first chunk | ~600 | the correctness gate; nothing else launches until it is bit-exact |
| 2 | 372 postC root, 9 re-extractable corpora | ~71 k | smallest, fully local, has a stored answer for every row |
| 3 | 944 canonical legs (`ext944-era2r4`) | 149 k | pairs TSVs verified present, all 11 legs |
| 4 | instruments (dial grid, corruption grid, dial anchor, ladder) | 2–5 k each | cheap; gates depend on them |
| 5 | safesyn training leg | 111 k | bitstreams present (47 GB); must decode in-process |
| 6 | KADIS-700k | 700 k | distorted are **R2-only** → scoped temp creds |
| 7 | bigcodec | 5.74 M | no local pixels; **TEST views of the 7 picker datasets first** |

**6 corpora in the 372 root are byte-COPIES and cannot be rebuilt on this box**
(aic4, nonphoto, imazen26, sdr25, hfnlproxy, hf_nearlossless) — a zero delta
there is an identity, not evidence. They are excluded from every count above
and must be excluded from every claim.

---

## 3. Pre-registered gates

### G-EXEC — the executor is correct before it is fast

* **G-EXEC.1 (unit).** A 4-pair fixture runs end-to-end through
  `run_feature_job` and yields a table with the declared row count and column
  count. Fails without the executor.
* **G-EXEC.2 (idempotence).** The same job run twice yields **byte-identical**
  output bytes.
* **G-EXEC.3 (ledger convergence).** Two partial passes over one manifest
  converge: pass 1 interrupted after k cells, pass 2 completes, and the union
  equals a single full pass — no duplicate rows, no missing rows.
* **G-EXEC.4 (fail-loud).** A missing extractor capability, an undecodable
  input, or a regime the build cannot serve **exits non-zero with a classified
  error**; it never emits a short or zero-filled row.
* **G-EXEC.5 (revision is in the id).** A `Feature` job with `revision: None`
  serializes **byte-identically** to the pre-revision schema (so no existing
  content-addressed id moves), and `revision: Some("2")` yields a **different**
  `JobId`. Asserted, not asserted-by-inspection.

### G-BITEXACT — the rev1 correctness gate (THE gate)

> Fleet output at `ZENSIM_FORMULA_REV=1`, regime 372, on the **same pixels**,
> must be **bit-identical** to the stored postC root
> `/mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC/`.

Compared as `to_bits()` on every `f0..f371` cell of a chunk, not as a
tolerance. **Any** non-zero delta halts the wave and is reported, not rounded.

Two known, pre-declared divergence risks, both to be *measured* rather than
assumed away:

* **Sub-64 padding.** zensim's `extract_features_372col` passes RAW dims to
  `compute_zensim_with_config` (which reflect-pads internally since
  `f9fac41e`); zenmetrics' `extract_features_regime` reflect-pads **externally**
  to 64 and then passes the padded dims. For every image ≥64×64 the external
  pad is a no-op and the two agree by construction. Below 64 they may not. The
  eval corpora are all ≥64; the gate asserts that, and a corpus that is not is
  refused rather than silently routed.
* **Two producers, one root.** The postC root was built by **two** extractors:
  `zensim-validate --extract-only` (cid22/kadid/tid/pipal) and
  `extract_features_372col` (konjnd/aic3/csiq/live). The gate must cover at
  least one corpus from **each** producer, or it proves half a claim.

### G-IMAGE — bake everything

* **G-IMAGE.1.** The image contains a **precompiled** `zenmetrics` binary
  `COPY`'d in. No `cargo`/`apt`/`pip` at build-time inside the Dockerfile and
  none at boot.
* **G-IMAGE.2.** `fleet-entrypoint.sh` verifies every baked tool and **fails
  loud** on a missing one.
* **G-IMAGE.3.** The image is a **new TAG on the canonical
  `ghcr.io/imazen/zenfleet-worker` package** — never a new package name.
* **G-IMAGE.4.** `TMPDIR` points at a bind-mounted **disk-backed** path; the
  container never writes to its own `/tmp`.
* **G-IMAGE.5.** The image records the zensim `build_commit` it was built
  from, readable without running a job.

### G-WAVE — first-cell gate before scale

* **G-WAVE.1.** After the **first** chunk of a wave completes, and **before**
  any second box is launched: the output table is fetched and verified —
  manifest present, `feature_set_id` present, revision present, decoder era per
  format present, row count equals the declared chunk size, column count equals
  the regime width.
* **G-WAVE.2.** That first chunk passes **G-BITEXACT** against the locally
  extracted rows for the same pairs.
* **G-WAVE.3.** A wave whose first chunk fails either gate is **stopped**; the
  runner is fixed and the image rebuilt before another box-minute is spent.

### G-FLEET — machine safety

* Observe before load on **every** node (`docker ps` for a live `zen*` worker,
  `uptime`) — a slow box is a busy box.
* `node-2` / `node-3` only when idle and only via the sanctioned
  PXE flip; never left flipped.
* Tower is **Docker-only**; nothing installed on the host OS.
* Narrow cpusets leaving ≥8 cores free on tower; memory caps; low `cpu-shares`.
* Local heavy work goes through `~/work/zen/scripts/run-heavy`.
* **No paid cloud.** LAN + tower only.

### G-PERSIST — nothing computed is thrown away

Every wave output is mirrored to the LAN store **and** Tower with a sha
manifest before any cleanup, per the ML-pipeline discipline. `_MANIFEST.json`
carries `build_commit`, `feature_set_id`, `formula_revision`, per-format
decoder era, per-file sha256, and row counts.

---

## 4. The executor's contract (pre-registered, so a later reader can check it)

**Input.** `JobKind::Feature { regime, revision }`; `cell.image_path` = the
reference; `inputs[]` = the distorted side, one entry per pair.

**Input resolution is ONE ordered rule**, so every corpus in the manifest is
declarable without a second pipeline:

1. an existing **local file** → used in place (no network, no temp);
2. `s3://…` → fetched in-process;
3. a **64-hex sha** → the variant index (tar byte-range / direct-object /
   pre-extracted local dir), i.e. the existing `fetch_variant` path;
4. otherwise → resolved **relative to the corpus prefix**, like the reference.

**Revision.** `revision` is threaded to zensim through `ZENSIM_FORMULA_REV`,
which zensim reads **once per process** into a `OnceLock`. Consequence, and it
is load-bearing: under the worker's **warm** executor pool (`feature` is
warm-eligible by default) one child process is reused across jobs, so a child
that has already read the variable **cannot** serve the other revision. The
executor therefore **refuses** a job whose revision differs from the one its
process resolved, rather than silently emitting the wrong era. A wave is
single-revision by construction; two revisions are two waves.

**Output.** A columnar chunk, not JSON rows — decided against the wire cost in
§1.1. Exact container recorded in the implementation note once the worker's
stdout contract is read; the requirement is that it is (a) self-describing,
(b) content-addressed by the job system unchanged, and (c) readable by the
existing writeback tooling.

**Every emitted table stamps**: `feature_set_id`, `formula_revision`, the
extractor `build_commit`, the regime/width, and the **decoder era per format**.
The last one is not optional: MEASURED 2026-09-04, re-decoding safesyn through
current imazen codecs shifts shipped B's dial by mean **−3.658** points against
an extractor-era defect of **−4.98/−5.86** — decoder era is **73 %** of
extractor era, a confound of comparable size. A table that does not record
which decoders read its pixels cannot be compared with one that does.

---

## 5. What would make this lane WRONG

Recorded now so it cannot be rationalized later:

* If G-BITEXACT fails at rev1, the executor is not a drop-in replacement for
  the stored root's producer, and **no rev2 number it produces is comparable to
  any stored number**. That is a stop, not a footnote.
* If the warm-pool revision refusal (§4) is not implemented, a mixed-revision
  wave silently produces a table with two eras in it — the exact defect class
  the 372 stored-root incident cost this project a month over.
* If the 6 byte-COPY corpora are counted in any "N corpora recalculated"
  claim, that claim is false.
* If a wave is declared before the R6 arm token arrives, its rows are at an
  arm nobody chose.

---

## 6. AMENDMENTS (each dated, each with its reason)

A pre-registered plan that is edited to match a result is worthless. These are
recorded as amendments, with what changed the decision, so a later reader can
see the difference between "chosen before" and "learned during".

### A1 — 2026-09-06: the output is JSON-lines, NOT a columnar chunk

§4 pre-registered "a columnar chunk, not JSON rows", and §1.1 argued it from
wire cost at 944 f64 × 5.74 M rows. **That argument is still true and the
decision is still reversed**, for a reason that outranks it and that I did not
know when I wrote §1:

`scripts/jobsys/writeback_scores.py` is the harvester for every job-system
blob, it is JSONL-shaped, and it **already dispatches on `{"kind":"feature"}`**
(`writeback_scores.py:71`), keyed `(basename(image_path), basename(encode_sha))`
— the exact row shape the `ScoreFile` feature arm emits today. The worker
itself stores stdout as opaque bytes (`zenfleet-worker/src/lib.rs:1070-1073`,
`store.put(&bytes)`), so a Parquet blob would have worked mechanically — but it
would have needed a **second harvester**, which is a duplicate implementation
of a thing that exists, and the no-duplication rule is not negotiable for a
wire cost.

So: the executor emits JSONL, and the Parquet conversion stays where it already
lives, in `writeback_scores.py`. **The wire cost is real and is not
hand-waved**: it is now a known, named cost of the bigcodec leg (§2 wave 7),
and if it bites, the fix is to extend the harvester to accept both encodings —
one owner, two input shapes — not to grow a parallel one.

*Changed by reading, before any measurement. No result influenced it.*

### A2 — 2026-09-06: the revision is a LAUNCH-level pin, not a per-job setting

§4 pre-registered that the executor "refuses a job whose revision differs from
the one its process resolved", implemented by setting `ZENSIM_FORMULA_REV`
in-process on first use. **`zenmetrics` is `#![forbid(unsafe_code)]` and
`std::env::set_var` is `unsafe` under the 2024 edition**, so that shape does
not compile.

The rewrite is stronger than the original, not a workaround: the launcher (or
the image) exports `ZENSIM_FORMULA_REV`, and the executor **verifies** that
every job's pinned revision equals the environment it is actually running in,
refusing loudly otherwise. That pin is visible in `docker inspect`, survives a
warm-child recycle, and cannot race the first zensim call — none of which was
true of the in-process version. The refusal the plan asked for is intact; only
who sets the value moved.

*Changed by the compiler, not by a result.*

### A3 — 2026-09-06: `feature_set_id` is DECLARED and echoed, never re-derived

§4 pre-registered "every emitted table stamps `feature_set_id`". The executor
does stamp it — but it **reads it from the declare** (it rides in
`cell.knob_tuple_json`) rather than computing it. `zensim::feature_set_id` is
that id's owner, and its era token is a registry concept the zensim lane owns;
a second derivation inside the executor is exactly the "944 has named seven
different feature sets" problem the naming directive exists to stop. Declared
and echoed keeps one owner and still makes every row self-describing.

### A4 — 2026-09-06: a capability token was added that the plan did not ask for

Not a change of mind — a defect found while reading. `JobKind::Feature`'s
`required_capabilities()` returned `[]`, and `filter_by_exec_capabilities`
returns early when every job's `requires` is empty. So **a stale executor image
with no Feature arm would happily claim the cells, exit non-zero, and poison
every one of them** — `encoder_panic` is DETERMINISTIC, and the reconciler
poisons on the FIRST failure. That is not a stalled wave, it is a destroyed
one, and it is the measured shape of the 2026-08-30 incident where an image
without `avif-aom` burned 312 freshly-pardoned cells in 28 s because `requires`
named only a token every executor had.

`Feature` now requires `feature-jobs` (a new no-op cargo feature, advertised by
`zenmetrics capabilities`) and, when a revision is pinned, `feature-rev`.
Deliberately NOT `cpu-metrics`: every CPU executor advertises that, which is
the same under-claim in a new costume.

---

## 7. RESULTS as of 2026-09-06 — what is proven, what is blocked

### 7.1 G-BITEXACT: PASSED, on both producers

Executor output at revision 1, regime 372, vs the stored postC root
(`build_commit 4fbd8ff8` — the commit this build's zensim is at), compared as
`to_bits()` with row alignment verified on `ref_basename` first:

| corpus | stored table's producer | rows | cells | differ |
|---|---|--:|--:|--:|
| csiq | `extract_features_372col` | 866 | 322,152 | **0** |
| tid | `zensim-validate --extract-only` | 3,000 | 1,116,000 | **0** |
| kadid | `zensim-validate --extract-only` | 10,125 | 3,766,500 | **0** |
| | | **13,991** | **5,204,652** | **0** |

The two-producer requirement §3 pre-registered turned out to matter more than
expected: the producers differ in **decoder crate** (`zen_decode` → the imazen
codecs vs `image::open`) **and** in zensim entry point
(`compute_zensim_with_config` vs `compute_zensim_with_ref_and_config`). The
executor is a *third* combination — imazen decoders + the cached-ref path — and
reproduces both bit-exactly. Independently corroborated by R6's own C1 control
(zensim `benchmarks/f4_arm_decision_2026-09-05.md` §1).

### 7.2 ⛔ Blocker found by the gate: `live` cannot be re-extracted by the fleet

MEASURED: the stored LIVE table was built from **`.bmp`** — its `ref_basename`
is `bikes.bmp`, and `live_r2_pairs_png.tsv` misaligns against it, so the PNG
list is NOT what produced it. zensim's `zen_decode` handles BMP/PNM/farbfeld
via `zenbitmaps`; **`zenmetrics-cli/src/decode.rs` has no BMP arm**, so the
executor fails loud ("could not detect image format"). That is G-EXEC.4 working,
not a silent wrong answer. 779 of the 42,470 re-extractable rows (1.8 %).
Fix, registered not run: add a `zenbitmaps` BMP arm to `decode.rs`. LIVE's
stored row order is also not its pairs-TSV order, so it needs a key-based join,
not the positional compare the other three corpora use.

### 7.3 ⛔ A precondition §2 did not state: the 372 corpora are NOT on the LAN store

The 372 root's pixels live at `/mnt/v/dataset*` — **local WSL paths no fleet
node can see**. A declared 372 manifest (908 jobs / 13,991 pairs / 136
references, produced and verified) is runnable *locally* today and on the fleet
only after the corpora are staged to the LAN store under a corpus prefix. That
staging is a prerequisite of wave 2, and it was not in the manifest's
"pixels present?" column, which answers a different question (can this box
rebuild it) than the fleet's (can a worker reach it).

Waves 5–7 (safesyn / KADIS / bigcodec) do not have this problem: their pixels
are already object-addressable.

### 7.4 The wave is BLOCKED on the arm token, and correctly so

R6 has published no F4 arm — `benchmarks/f4_arm_decision_2026-09-05.md` ends at
its §2 correction with no verdict section, and `scripts/r6_decide.py` (the
reader that applies the pre-registered rule) has not produced one. Declaring a
rev2 wave now would put every row at an arm nobody chose.

**Resume command**, once the token lands (`<ARM>` ∈ `c1` / `lorentz` / `clamp`,
`<REV>` the revision token the zensim lane assigns it):

```bash
# 1. rebuild the executor against a zensim carrying the selector, and turn on
#    the `feature-rev` capability so a revision pin is claimable at all:
cargo build --release --target x86_64-unknown-linux-musl -p zenmetrics-cli \
  --bin zenmetrics --no-default-features \
  --features sweep,png,jpeg,webp,avif,jxl,cpu-metrics,feature-jobs,feature-rev,hdr-gainmap

# 2. re-run the gate at rev1 FIRST (the new build must still be bit-exact):
python3 scripts/jobsys/rev2_bitexact_gate.py \
  --pairs /mnt/v/dataset/csiq/csiq_pairs.tsv \
  --stored /mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC/csiq_features_372col_2026-07-18.parquet \
  --exec target/x86_64-unknown-linux-musl/release/zenmetrics --regime 372

# 3. declare the rev2 wave (a DIFFERENT job id than the rev1 one, by design):
target/release/zenfleet-ctl declare-features \
  --pairs <pairs.tsv> --out <manifest.json> --regime 372 --revision <REV> --chunk 16

# 4. launch with the revision pinned in the ENVIRONMENT (it is a launch-level
#    pin — the executor refuses a job whose revision disagrees with its env):
ZENSIM_FORMULA_REV=<REV> ZEN_CORPUS_BUCKET=<bucket> ZEN_CORPUS_PREFIX=<prefix> \
  bash scripts/jobsys/lan_score_launch.sh <host> <job-set> feat cpu <image tag>
```

Poll for the token at zensim `benchmarks/f4_arm_decision_2026-09-05.md` — the
verdict section, not the §2 correction.

---

## 7.5 The two §7.2/§7.3 blockers are CLOSED (2026-09-06, ops lane)

Both prerequisites §7 registered as blocking wave 2 are done. Neither
required the arm token; both are independent of the F4 decision.

### §7.3 (LAN reachability): the 372 corpora ARE on the LAN store

All 8 re-extractable corpora (cid22, kadid, tid, csiq, live, aic3, konjnd,
pipal — 43,870 rows, ~25 GB) are synced to
`s3://codec-corpus/eval372-rev2-2026-09-06/<corpus>/`. Full record + per-corpus
manifests: `docs/rev2_lan_stage.pointer.md` +
`/mnt/v/zen/zensim-training/rev2-lan-stage-2026-09-06/`.

**§4's "otherwise → resolved relative to the corpus prefix" contingency did
not need building.** `resolve_source` / `resolve_feature_input`
(`crates/zenmetrics-cli/src/jobexec.rs`) already fetch `s3://…` in-process for
BOTH the reference and every distorted input — this was evidently built
*for* the recalculation (the doc comments on both functions name this exact
manifest) but had not yet been proven against real staged data. It now has:
`scripts/jobsys/verify_lan_stage_reachability.sh <corpus>` declares and runs
one Feature job against the staged `s3://` pairs TSV and the SAME pairs
through their local-path originals, and diffs the feature vectors
`to_bits()`. Run against csiq, live, pipal, konjnd: **0 cells differ, every
time.** No code change to the executor was needed or made.

Cross-node read: `r7900x` fetched 3 randomly-picked files by their manifest
sha256 over the LAN store — 3/3 matched.

**PIPAL is staged as a 23,200-pair SUPERSET, not a verified match to the
postC root's 21,800.** Every PIPAL reference's label file has 116 valid
(ref,dist) pairs (verified: `load_pipal` in `zensim-validate/src/main.rs`
returns the un-truncated list, and `--max-images` — default 0 — is not
passed by `build_eval372_root.sh`), while the stored table has exactly 109
per reference, uniformly across all 200 refs. No selection rule for the
missing 7-per-ref was found. All pixels for the full 23,200 are staged
(nothing is missing for wave 2 to draw on); reconciling the exact historical
21,800-row cut is registered, not resolved, here.

### §7.2 (LIVE blocker): BMP arm shipped — G-BITEXACT PASSES

`crates/zenmetrics-cli/src/decode.rs` gained a `bmp` feature
(`zenbitmaps::decode_bmp`, funnelled through a new `bmp_layout_to_rgb8`
covering all 10 named `PixelLayout` variants) — magic-byte sniff (`BM`),
extension fallback, and the `decode_bytes_to_rgb8` dispatch arm, matching
zensim's own `zen_decode.rs` choice of decoder (`zenbitmaps`, imazen-only).
4 unit tests round-trip real BMP bytes (24-bit RGB, wrong-extension
override, 32-bit alpha-drop, every `PixelLayout` variant) produced by
`zenbitmaps`' own encoder — no hand-rolled BMP bytes, no third-party imaging
crate.

**§7.2's own framing of the fix was half right.** It correctly identified
LIVE's stored row order as NOT matching `live_r2_pairs.tsv`'s file order,
and said this "needs a key-based join." MEASURED: it does not — a **stable
sort of the pairs by `basename(ref_path)`** (Python's `sorted(key=…)`,
matching Rust's stable `sort_by` at
`zensim-bench/examples/extract_features_372col.rs:216`, `rows.sort_by(|a,b|
a.0.cmp(&b.0))`, which the "pairs-tsv" loader path runs after extraction)
reproduces the stored table's exact row order — grouping AND within-group
order — on all 779 rows, verified against both `ref_basename` and
`human_score`. No join, no key beyond the reference's own filename.

Gate re-run with the fixed binary
(`--features sweep,png,jpeg,webp,avif,jxl,cpu-metrics,feature-jobs,feature-rev,bmp`)
and the stable-sorted pairs TSV:

```
pairs: 779
declared 58 feature jobs
extracted 779 rows
G-EXEC.2 idempotence: OK (job 0 byte-identical on re-run)
row alignment: OK (ref_basename matches on all 779 compared rows)

=== G-BITEXACT ===
compared 289788 cells (779 rows x 372 features)
  basic   0/121524 differ
  peaks   0/56088 differ
  masked  0/56088 differ
  iw      0/56088 differ
RESULT: BIT-EXACT — 0 of 289788 cells differ
```

**§7.1's G-BITEXACT table now covers all 4 re-extractable non-byte-copy
producers this root has** (csiq/tid/kadid already passed; live now passes
too) — 779 + 13,991 = 14,770 rows, 5,494,440 cells, 0 differ.

Neither fix required or made any arithmetic change — `zensim` is untouched;
this is entirely fleet-side (decode dispatch) and data-side (staging + a
row-order finding).
