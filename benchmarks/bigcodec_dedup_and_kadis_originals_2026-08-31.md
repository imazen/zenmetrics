# Bigcodec box-tar deduplication + KADIS-700k originals — measured (2026-08-31)

Two retirement questions, answered by measurement rather than by reading the
manifests. Companion to `benchmarks/r2_lan_mirror_2026-08-30.md` (§2 corpus
structure, §7 the R2 deletion of `canonical/2026-06-27/`).

Read-only pass: nothing was written to R2, the LAN store or the tower. All byte
reads were **ranged GETs into the box tars** driven off the mirrored
`variant_index.tsv` files — the same fetch path a tarrange consumer uses — so no
tar was ever downloaded whole.

---

## Q1 — are the 61 bigcodec box tars deduplicated?

**Short answer: no, and 22.6 % of their payload is byte-identical duplicate
content.** Measured duplicate mass **52,268,279,641 B = 48.68 GiB**, 95 % CI
**[48.30, 49.03] GiB**. The corpus stores every member verbatim; there is no
content addressing anywhere in the chain.

### 1.1 Corpus shape (exact, from the indexes)

| | value |
|---|--:|
| box tars | **61** across 6 runs |
| tar bytes | **235,474,984,960** (219.30 GiB) |
| members | **5,742,669** |
| member payload | **231,078,618,620** (215.21 GiB) |
| tar header + padding overhead | **4,396,366,340** (4.09 GiB) = 765.6 B/member |
| distinct source renditions | **4,497** |

Per-family payload reconciles **exactly** with the R2 `encodes/` byte totals
recorded in the mirror doc §7.1 — `zenjpeg_lossy` 46,311,414,326,
`zenjxl_lossless` 58,419,699,144, `zenpng_lossless` 13,691,411,533, and
`zenwebp` lossy+lossless 24,187,758,994 + 6,954,651,422 = 31,142,410,416. That
cross-check is what licenses using the indexes as the population frame.

### 1.2 (a) The indexes carry NO content hash

`variant_index.tsv` is **4 tab-separated columns**:

```
<member_name>  <offset_data>  <size>  <member_name>
```

Column 4 is a verbatim repeat of column 1. Confirmed by reading the head of all
six families and against the producer,
`scripts/jobsys/index_tar_byterange.py:41`:

```python
idx.append("%s\t%d\t%d\t%s" % (dm, m.offset_data, m.size, dm))
```

So there is no hash to count distinct-vs-total with. **The trailing 16-hex token
in a member name is NOT a content hash** — the naming scheme is
`<src_stem>_<src_path_hash16>_<codec>_q<q>_<knob_hash16>.<ext>`
(`crates/zenmetrics-cli/src/sweep/run.rs:1600`), so the trailing hash identifies
the **knob tuple**, i.e. the encode *parameters*. Two members with different
knob hashes are different cells by construction, whatever their bytes turn out
to be. Duplication therefore had to be measured.

### 1.3 (a) Method — length-grouping first, then sampled sha256

Equal length is a necessary condition for byte-identity, so members were grouped
by length before any byte was fetched. Two nested frames:

| frame | members in a shared class | upper bound on duplicate bytes |
|---|--:|--:|
| global `(ext, size)` | 5,453,928 (94.97 %) | 125,429,598,597 (116.82 GiB, 54.28 %) |
| within-rendition `(family, rendition, size)` | 1,651,371 (28.76 %) | **54,633,329,231** (50.88 GiB, 23.64 %) |
| cross-rendition, same `(ext, size)` | 361,761 classes | 70,796,060,773 (65.93 GiB, 30.64 %) |

The global frame is loose and misleading on its own. Its single largest class is
87 members of exactly 2,409,592 B — but they are 87 **different** source images,
all `scale896x896` zenpng `q0` under one knob hash, at 896·896·3 + PNG overhead:
an effectively-stored PNG whose size is a function of dimensions, not content.
Equal size there is a coincidence of geometry. That is exactly why the two
frames are sampled separately.

**Stratum A — within-rendition.** Groups sampled **PPS with replacement**,
weight `w_g = (n_g − 1)·size_g` (the group's potential duplicate bytes),
400 draws per family (2,400 draws, 2,241 distinct groups). **Every member of
every sampled group was fetched by byte range and sha256'd** — 2.21 GiB read, no
sub-sampling, so `d_g` (distinct contents) is exact per sampled group. The
Hansen-Hurwitz estimator collapses to `Ŷ = W · mean(r_g)` with
`r_g = (n_g − d_g)/(n_g − 1) ∈ [0,1]`; CI by 4,000-resample bootstrap over the
draws.

**Stratum B — cross-rendition.** Classes sampled PPS by
`(R_c − 1)·size_c` where `R_c` = distinct renditions in the class, one
representative member per rendition (capped at 6), 500 draws / 494 classes /
**2,310 members hashed**.

### 1.4 (a) Result

| family | dataset | payload B | bound B | realized dup frac (95 % CI) | est dup B | % of family |
|---|---|--:|--:|---|--:|--:|
| `zavif` | zenavif_lossy | 49,530,478,971 | 1,364,343,348 | 0.7717 [0.7308, 0.8125] | 1,052,818,284 | 2.13 |
| `zjpeg` | zenjpeg_lossy | 46,311,414,326 | 693,360,238 | **0.0000** [0, 0] | **0** | **0.00** |
| `zjxll` | zenjxl_lossy (vardct) | 31,983,204,230 | 6,986,641,636 | 0.8748 [0.8527, 0.8956] | 6,111,978,219 | 19.11 |
| `zjxlm` | zenjxl_lossless (modular) | 58,419,699,144 | 37,327,986,383 | **0.99969** [0.99906, 1.0] | 37,316,321,387 | **63.88** |
| `zpng` | zenpng_lossless | 13,691,411,533 | 425,258,068 | 0.9775 [0.9625, 0.9900] | 415,689,761 | 3.04 |
| `zwebp` | zenwebp lossy+lossless | 31,142,410,416 | 7,835,739,558 | 0.9408 [0.9191, 0.9608] | 7,371,471,989 | 23.67 |
| **total** | | **231,078,618,620** | **54,633,329,231** | | **52,268,279,641** | **22.62** |

95 % CI on the total: **48.30 – 49.03 GiB**.

**Stratum B: 0 collisions in 2,310 hashed members across 494 classes.** With
zero successes the bootstrap CI is degenerate, so the honest statement is the
rule-of-three bound: 95 % upper bound on the mean fraction ≈ 3/500 = 0.006, i.e.
cross-rendition duplication is **≤ ~0.40 GiB** and is most plausibly zero. No
two different source images produced the same encoded bytes, and no two formats
did either.

So essentially all duplication is **within one rendition, within one codec**.

### 1.5 (a) Mechanism — a knob grid that collapses, content-dependently

In **479,653 of 528,937** within-rendition multi-member groups (**90.7 %**) the
quality is constant and only the **knob tuple** differs. The sweep encoded the
same `(rendition, q)` under several knob tuples that turn out to emit
byte-identical bitstreams.

The rate tracks how much the knobs can actually move the bitstream:

- `zjxlm` (JXL lossless modular, `q` is always 0): 60 knob variants per
  rendition, **63.9 %** of its bytes duplicate — the single biggest pool.
- `zwebp`: 219 variants per rendition, 23.7 %.
- `zjpeg`: **0 %** — every zenjpeg knob changes the bitstream. Its 693 MB of
  equal-length candidates are all false positives.

**The redundancy is content-dependent, not a fixed property of the grid.**
Hashing every member of 25 randomly chosen `zjxlm` renditions (1,500 members)
gave 540 distinct contents (**64.0 %** duplicate members) but **20 distinct
`(q,knob) → content` partitions across the 25 renditions** — the most common
covering only 4 of 25. Same shape for `zwebp`: 10 renditions, 2,190 members,
1,439 distinct (34.3 % duplicate), **8 distinct partitions**. Consequence: you
**cannot** recover these bytes by pruning knobs out of the sweep plan, because
which knobs collapse depends on the image. Only content-addressed storage
captures it.

### 1.6 (b) Name/key-level duplication — exact, and it is ZERO

Cheap and exact from the indexes alone:

| check | result |
|---|--:|
| total members across all 61 indexes | 5,742,669 |
| **distinct member names** | **5,742,669** |
| names occurring more than once (any tar, any run) | **0** |

No cell is stored twice under the same identity — not within a tar, not across
tars in a run, not across runs. The six runs are disjoint by construction and
the box split inside each run is a clean partition.

At the **rendition** level the answer inverts, and it is by design:

| | value |
|---|--:|
| distinct renditions | 4,497 |
| renditions present in all 6 codec families | **4,497 (100 %)** |
| renditions in fewer than 6 families | 0 |

Every source rendition is encoded by every codec — this is a cross-codec picker
corpus, so full cross-run rendition reuse is the point. The 4,497 figure matches
`codec-corpus/clean-picker-corpus-2026-06-26/` (4,497 objects) exactly. But
rendition reuse is **not** byte duplication: stratum B found 0 byte collisions
across renditions or formats.

### 1.7 (c) The practical answer

**Is the corpus stored deduplicated today? No.** Every member sits verbatim in a
tar; nothing in the chain (tar, index, LAN store, R2) is content-addressed.

| quantity | bytes | GiB |
|---|--:|--:|
| tar footprint, one copy | 235,474,984,960 | 219.30 |
| member payload | 231,078,618,620 | 215.21 |
| **genuinely unique content** | **178,810,338,979** | **166.53** |
| **duplicate content (removable, zero information loss)** | **52,268,279,641** | **48.68** |
| tar header/padding | 4,396,366,340 | 4.09 |

**Dedup opportunity = 48.68 GiB per copy** — 22.2 % of the tar footprint, at
zero information loss, since every removed byte is provably identical to a byte
that remains.

**And the corpus is currently stored in three physical copies** (mirror doc
§7.6): R2 `jxl-lossy/runs/*/variants/`, the LAN SeaweedFS store, and the
tower's plain tree `/mnt/disk1/coefficient/zen924/tars/` (verified this pass:
235,474,984,976 B including a 16 B `mirror.log`). So the **aggregate** dedup
opportunity across what is stored today is **~146 GiB**, and copy count is a
bigger lever than dedup on any single copy. (A fourth grouping — the 15
`variants.<split>.tar`, 141.30 GiB of the same members re-tarred — was already
deleted from R2 on 2026-08-30.)

**How much is re-derivable?** Structurally, all of it: the 5,742,669 encodes
derive from `originals/` (1.03 GiB, 3 tars, the 4,497 renditions) plus the codec
and the knob grid. Two caveats keep that from being a licence to delete:

1. Re-encoding 5.74 M cells is a large compute job, not a restore.
2. **Every score and feature vector in the canonical parquets was computed on
   these exact bytes.** A re-encode that is not bit-exact — an encoder version
   moved, a default changed — silently decouples the labels from the pixels.
   That is the same trap already measured on KADIS (Q2 below), and it is why
   `originals/` was given the off-site slot in mirror doc §7.7.

Ranked, then: dropping a *copy* is the largest single lever (219.30 GiB per
copy retired, and three copies exist today); content-addressed dedup is worth
48.68 GiB per surviving copy and is safe by construction; re-deriving from
`originals/` is a rebuild, not a restore, and carries a label-validity risk.

---

## Q2 — how many KADIS originals are there?

**140,000 pristine reference images. The originals exist, in two byte-identical
copies. The 197.43 GiB of distorted PNGs are NOT faithfully regenerable.**

### 2.1 The count — verified four ways

| source | rows | distinct `source_id` | distinct `source_filename` |
|---|--:|--:|--:|
| `kadis700k_canonical_gpu_2026-07-01.parquet` | 700,000 | **140,000** (0…139,999) | **140,000** |
| `kadis700k_canonical_2026-06-30.parquet` | 700,000 | **140,000** | **140,000** |
| `kadis700k_924.parquet` (kadis-924-2026-07-27) | 699,999 | **140,000** | **140,000** |
| `/mnt/v/datasets/kadis700k/refs/` (actual files) | — | — | **140,000 files** |

The two canonical parquets' filename sets are **identical**; the 924 rescore's
set is identical to both (0 either-way difference); and the refs directory
listing matches the parquet filename set **exactly — 0 missing, 0 extra, in both
directions**. The documented shape (140k × 1 dist type × 5 levels = 700k) holds.
Read with pyarrow.

### 2.2 Where the originals live

| location | objects | bytes | note |
|---|--:|--:|---|
| `/mnt/v/datasets/kadis700k/refs/` | 140,000 | **44,632,022,736** (41.57 GiB) | dev box |
| `s3://zentrain/kadis-700k/refs/` | 140,000 | **44,632,022,736** | R2, off-site |
| tower | — | — | **absent** (only `output/kadis700k/canonical/`, 5 files) |
| LAN store | — | — | **absent** (only `zentrain/kadis-700k/canonical/`) |

Count and total bytes agree to the byte between local and R2, and 3/3 sha256
spot checks (`zwinger-1541696.png`, `121-cc-774921.png`, `19-199047.png`) are
identical between the two. So there are **two independent copies, one off-site**
— the originals are not a durability gap.

Upstream, KADIS-700k's 140,000 pristine references are published by the VQA
Group at Universität Konstanz (Lin, Hosu, Saupe) and described as freely
available to the research community
(https://database.mmsp-kn.de/kadid-10k-database.html), so a third recovery path
exists in principle. Credit and citation: `~/work/kadis-distort/docs/DATASET.md`.

### 2.3 The distorted PNGs — measured, and the verdict

`s3://zentrain/kadis-700k-gpu/distorted/` re-listed this pass with the `aws`
lister: **699,999 objects, 211,990,306,109 B = 197.43 GiB** — matching mirror
doc §2 exactly. **R2-only**: not on the tower, not on the LAN store, not on
`/mnt/v`.

**Verdict: they are NOT regenerable, and "regenerable from links" is true only
in the sense that the links point at these very objects.** Two independent
reasons:

1. **Regeneration does not reproduce the pixels — measured 2026-07-24.**
   `kadis_distort.io.rng_for(fname, dt, lv)` is not the seed the 2026-06-30 /
   2026-07-01 generator used, so stochastic distortion types come out
   **mean |Δ| ≈ 9.8** different (verified on dist_type 23 `color_block` sev 5,
   same 512×384, not identical). The mechanism is visible in the code today:
   `kadis_distort/serve.py` carries **two** seeding schemes — v1 content-hash
   `_seed(ref_hash, dt, level)` and v2 name-based `io.seed_for(ref_name, dt,
   level)` — and picks v2 whenever `ref_name` is present. No matching regen has
   been demonstrated.
2. **Even a bit-exact regen would not restore validity.** All 7 metric scores
   and every feature vector (372 / 720 / 924 / 944) in the canonical parquets
   were computed *on these pixels*, and the `negrich` corruption-head subset is
   *defined* as `score_zensim_gpu < 0` on them. Pixels that differ at all
   silently decouple the features from their labels.

⚠ **Doc conflict, flagged not fixed:** `~/work/kadis-distort/docs/DATASET.md`
still says the distorted PNGs "are cheap to regenerate deterministically from
`(source, dist_type, level)`". That sentence describes the earlier 2026-06-30
generate-discard variant and was falsified by the 2026-07-24 measurement above.
It belongs to the kadis-distort repo, so it is recorded here rather than edited.

**What retiring the 197.43 GiB would actually cost.** Not the existing work: the
372, 720, 924 and 944 feature regimes have all been extracted and are stored
(`kadis-720-2026-07-24`, `kadis-924-2026-07-27`, `kadis-944-2026-08-01`, all
triple-mirrored). What is lost is the ability to extract a **future** regime —
which the zensim roadmap explicitly anticipates ("Coming later: additional HDR
features"). That is the whole decision: 197.43 GiB of R2 (~$3.18/month at
$0.015/GB-month) against the option to rescore 700k labeled cells at a new
feature width without re-labelling them. A middle path exists — the 167,034-cell
`negrich` severe subset is the part that carries the corruption-head labels, and
keeping only its PNGs would retain 167,034 of 699,999 cells (23.9 %).

---

## Method notes and confidence

- **Population frame**: all 61 `variant_index.tsv` (53 from
  `s3://zentrain/jobs/bf-*`, 8 from
  `s3://zentrain/_archive/bigcodec-2026-08-30/indexes/bf-zjpeg-t*`), 895 MB,
  5,742,669 rows. Box→run mapping verified by reading the 512-byte tar header
  immediately preceding `offset_data` for box-0 of each family and confirming
  the name matches, plus the format magic of the member itself (`ftypavif`,
  `ffd8ff`, `ff0a`, `89504e47`, `RIFF….WEBP`) — 6/6.
- **Byte reads**: SigV4-signed ranged `GET` against R2 with connection reuse,
  24 threads, `nice -n19 ionice -c3`. ~2.5 GiB fetched in total. R2 was used
  rather than the LAN store deliberately: the tower is a live media server and
  R2 egress is free.
- **Exactness**: within a sampled group, every member was hashed — no
  sub-sampling — so `d_g` is exact and the only uncertainty is which groups were
  drawn. That uncertainty is what the bootstrap CI covers.
- **Confidence**: high on Q1's total (the CI is ±0.7 % and the estimator is
  unbiased under PPS); high on Q1(b) and the corpus shape (exact counts, and
  per-family payload reconciles to the byte with the independently-recorded R2
  `encodes/` totals); high on Q2's 140,000 (four agreeing sources, one of them
  the files themselves). The cross-rendition zero rests on 2,310 hashed members
  and is reported as a rule-of-three bound rather than as a proof.
- Working artifacts (not committed): `~/tmp/dedup2026/` — `idx/` (the 61
  indexes), `all_members.tsv`, `sampleFAM.pkl`, `sampleA.log`, `sampleX.log`.
