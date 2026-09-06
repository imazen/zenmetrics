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
* Kids' PCs (node-2 / node-3) only when idle and only via the sanctioned
  PXE flip; never left flipped.
* Tower is **Docker-only**; nothing installed on the Unraid host.
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
