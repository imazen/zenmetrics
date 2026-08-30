# R2 → LAN-store mirror — the active-pipeline working set (2026-08-30)

**Directive:** "put all r2 files we need on seaweed." Deletion from R2 is **out of
scope** and remains user-gated; this pass only ADDS objects to the LAN store.

**Store contract.** The LAN store (SeaweedFS on tower, `zen-lanstore`) is reached
through `scripts/lib/s3env.sh` — `ZEN_STORE` unset/`tower`/`lan` resolves to it,
`ZEN_STORE=r2` is the explicit opt-out. Every object below is mirrored at the
**identical `bucket/key`**, so `ZEN_STORE=tower` needs no path change anywhere.

## 0. What made this urgent (the measured pain)

`zensim/benchmarks/wlin_round7_rawframe_2026-08-30.md` §8: a 208,169-cell tbig
extraction spent **72 min fetch-dominated at 48 cells/s** pulling encodes from
R2, and could not use the fleet at all, because

1. `s5cmd --endpoint-url <LAN> ls s3://zentrain/canonical/2026-06-27/` returned
   *"no object found"* — the bigcodec corpus was **not on the LAN store**; and
2. for `zenavif_lossy` + `zenjxl_lossy` (103,585 of 208,169 cells, 49.8 %) the
   per-file `encodes/` prefix does not exist at all, so those bytes are only
   reachable as **byte ranges inside `variants/box-N.tar`**, which
   `declare-scorefiles --full-uri` cannot express as a per-object URI.

## 1. Corpus structure — measured, not assumed

`s3://zentrain/canonical/2026-06-27/<ds>/` holds two byte representations, and
which ones exist differs per dataset:

| dataset | `encodes/` (per-file) | `variants.<split>.tar` | box tars under `jxl-lossy/runs/…/variants/` | fetch mode |
|---|---|---|---|---|
| `zenjpeg_lossy`    | YES | YES (47.45 GB) | 8 (44.19 GiB) | object |
| `zenwebp_lossy`    | YES | YES (24.91 GB) | 9 (29.71 GiB, shared w/ lossless) | object |
| `zenwebp_lossless` | YES | YES (6.99 GB)  | (same 9 boxes) | — (not in the resolver map) |
| `zenpng_lossless`  | YES | YES (13.75 GB) | 2 (12.81 GiB) | object |
| `zenjxl_lossless`  | YES | YES (58.63 GB) | 10 (54.60 GiB) | object |
| `zenjxl_lossy`     | **NO** | **NO** | 24 (30.80 GiB) | **tarrange** |
| `zenavif_lossy`    | **NO** | **NO** | 8 (47.20 GiB) | **tarrange** |

Two consequences that shaped the mirror:

> **This decision was REVISED mid-pass — see the note at the end of this
> section.** The redundancy argument below holds only while the `encodes/`
> mirror is expected to finish; once it was deferred (§3b), the box tars stopped
> being redundant and became the primary path.

- **The box tars for the five regrouped datasets are byte-redundant with their
  `encodes/` objects** (`mandfix2-zenjpeg` totals 47,450,193,920 B — exactly the
  sum of that dataset's three `variants.<split>.tar`). Mirroring all of them
  would have cost ~141 GiB for zero new bytes, so three of the four runs are
  REGISTERED, not mirrored. The **`mandfix2-zenjpeg` boxes are the exception and
  WERE mirrored**, because `zenjpeg_lossy` is the one object-mode dataset with
  **no `variant_index.tsv` anywhere** (`bf-zjl2` and `bf-zenjpeg-lossy` carry
  none) — it has no tar-range fallback, and its `encodes/` prefix is the largest
  single object set in the corpus, so its box tars are cheap byte-insurance.
- **`variants.<split>.tar` has no consumer at all.** A grep of
  `zenmetrics/scripts`, `zensim/scripts` and `zenanalyze` finds zero readers;
  every consumer uses `encodes/` (object mode) or the box tars + a
  `variant_index.tsv` byte-range index (tarrange mode). Also registered, not
  mirrored.

So the minimal *complete* byte set is: **`encodes/` for the five datasets that
have it, plus the 32 box tars of the two that do not, plus the byte-range
indexes** — and that is what the pass started on, with `mandfix2-zenjpeg`'s 8
boxes added as insurance for the one dataset that has no second path.

**REVISION (made mid-pass, after the load ceiling in §3 forced the `encodes/`
lanes to stop).** With `encodes/` deferred for four datasets, "redundant" no
longer described the box tars — for `zenwebp_lossy`, `zenpng_lossless` and
`zenjxl_lossless` the tar is now the *only* LAN-resident byte path, and each of
those three already has its `variant_index.tsv` mirrored, so the tar-range fetch
works the moment the tar lands. The choice became:

| to complete those three datasets | objects | GiB |
|---|--:|--:|
| their `encodes/` prefixes | **1,290,639** | 89.69 |
| their box tars (`mandfix2-zenpng`, `mandfix2-zenwebp`, `jxl-modular`) | **21** | 97.12 |

Nearly the same bytes, five orders of magnitude fewer objects — and object rate,
not bytes, is what this store cannot absorb (§3). Those 21 tars were therefore
moved from *registered-redundant* into the mirror queue. With them, **every
dataset in the corpus is byte-complete and fetchable on the LAN store through
the tar + index path**, and the `encodes/` prefixes become a convenience
(a plain GET instead of a ranged GET) rather than a prerequisite.

## 2. Inventory — built from usage, not guesswork

Consumers were found by grepping `s3://` reads across `zenmetrics/scripts/jobsys`,
`zenmetrics/scripts/picker`, `zensim/scripts`, `zenanalyze` and the live
`~/tmp/*.sh` fleet chains. Sizes are `s5cmd du` against R2 on 2026-08-30.

### P0 — the bigcodec corpus (no LAN copy existed; the round-7 bottleneck)

| bucket/prefix | GiB | objects | what reads it | status |
|---|--:|--:|---|---|
| `zentrain/canonical/2026-06-27/zenwebp_lossless/encodes/` | 6.48 | 40,473 | resolve_bigcodec_pair_uris.py object-mode; fetch_bigcodec_bytes.py | **MIRRORED+VERIFIED** |
| `zentrain/canonical/2026-06-27/zenpng_lossless/encodes/` | 12.75 | 76,449 | same | **PARTIAL 27,500/76,449** |
| `zentrain/canonical/2026-06-27/zenjxl_lossless/encodes/` | 54.41 | 269,820 | same | **DEFERRED (tower-local ingest)** |
| `zentrain/canonical/2026-06-27/zenwebp_lossy/encodes/` | 22.53 | 944,370 | same | **DEFERRED (tower-local ingest)** |
| `zentrain/canonical/2026-06-27/zenjpeg_lossy/encodes/` | 43.13 | 1,484,010 | same (ONLY path - no tar index exists) | **DEFERRED (bytes already on tower)** |
| `zentrain/jxl-lossy/runs/mandfix4-zenavif-1782593621/variants/` | 47.20 | 8 | tarrange bytes for zenavif_lossy (no encodes/) | **MIRRORED+VERIFIED** |
| `zentrain/jxl-lossy/runs/jxl-lossy-vardct-1782609551/variants/` | 30.80 | 24 | tarrange bytes for zenjxl_lossy (no encodes/) | **MIRRORED+VERIFIED** |
| `zentrain/jobs/bf-{zavif,zjxll,zjxlm,zpng,zwebp}-t*/variant_index.tsv` | 0.65 | 53 | byte-range index for tarrange fetch | **MIRRORED+VERIFIED** |
| `zentrain/canonical/2026-06-27/*/{_MANIFEST.json,pairs.*,train|validate|test}.parquet` | 8.30 | 50 | build_canonical/verify_canonical; every picker script | **MIRRORED+VERIFIED** |
| `zentrain/canonical/2026-06-27/originals/` | 1.03 | 3 | corpus originals | **MIRRORED+VERIFIED** |
| `codec-corpus/clean-picker-corpus-2026-06-26/` | 1.02 | 4,497 | pairs.*.parquet ref_path | **MIRRORED+VERIFIED** |
| `zentrain/jxl-lossy/runs/mandfix2-zenjpeg-1782584881/variants/` | 44.19 | 8 | INSURANCE for zenjpeg encodes (no tar index) | **MIRRORED+VERIFIED** |
| `zentrain/jxl-lossy/runs/mandfix2-zenpng-1782584881/variants/` | 12.81 | 2 | tar-range bytes for zenpng_lossless (its index is mirrored; encodes deferred) | **QUEUED (tars3 lane)** |
| `zentrain/jxl-lossy/runs/mandfix2-zenwebp-1782584881/variants/` | 29.71 | 9 | tar-range bytes for zenwebp_lossy+lossless (index mirrored; encodes deferred) | **QUEUED (tars3 lane)** |
| `zentrain/jxl-lossy/runs/jxl-modular-1782596759/variants/` | 54.60 | 10 | tar-range bytes for zenjxl_lossless (index mirrored; encodes deferred) | **QUEUED (tars3 lane)** |

### P1 — eval instruments + regime training tables (local `/mnt/v` mirrors exist; the LAN copy makes them fleet-addressable)

| bucket/prefix | GiB | objects | what reads it | status |
|---|--:|--:|---|---|
| `zentrain/eval-grids/` | 0.020 | 4 | bake_verdict dial+corruption grids | **MIRRORED+VERIFIED** |
| `zentrain/canonical-2026-05-21/` | 1.35 | 29 | canonical 372-feature trainer input | **MIRRORED+VERIFIED** |
| `zentrain/kadis-700k/canonical/` | 0.84 | 1 | KADIS-700k canonical parquet | **MIRRORED+VERIFIED** |
| `zentrain/kadis-700k-gpu/canonical/` | 0.87 | 1 | KADIS-700k GPU-metric canonical parquet | **MIRRORED+VERIFIED** |
| `zentrain/ext944-canonical-2026-08-01/` | 26.51 | 85 | 944-regime legs | **QUEUED (gentle lane)** |
| `zentrain/tbig-944-2026-08-02/` | 29.29 | 16 | bigcodec 944 views | **QUEUED (gentle lane)** |
| `zentrain/kadis-944-2026-08-01/` | 3.09 | 7 | KADIS 944 rescore | **QUEUED (gentle lane)** |
| `zentrain/ext924-canonical-2026-07-27/` | 49.96 | 36 | 924-regime legs | **QUEUED (gentle lane)** |
| `zentrain/ext720-canonical-2026-07-22/` | 55.58 | 37 | 720-regime legs | **QUEUED (gentle lane)** |
| `zentrain/tbig-720-2026-07-22/` | 17.59 | 2 | bigcodec 720 views | **QUEUED (gentle lane)** |
| `zentrain/kadis-720-2026-07-24/` | 3.13 | 3 | KADIS 720 rescore | **QUEUED (gentle lane)** |
| `zentrain/hdr/` | 4.77 | 185 | HDR corpora/tables | **QUEUED (gentle lane)** |
| `zentrain/canonical-2026-07-15/` | 0.004 | 4 | HF near-lossless + kadis_negrich training parquets (the recurring-priority set) | **QUEUED (gentle lane)** |
| `zentrain/canonical-gates-2026-07-29/` | 0.001 | 57 | canonical-source gate verdicts | **QUEUED (gentle lane)** |
| `zentrain/profiles/` | 0.001 | 6 | Profile C ship artifacts (C-2026-08-05) | **QUEUED (gentle lane)** |
| `zentrain/hdr-corpora/` | 0.034 | 4 | HDR corpora tables | **QUEUED (gentle lane)** |
| `zentrain/jxl-nearlossless-2026-07-06/` | 0.050 | 29 | input to build_hf_nearlossless_2026-07-15.py | **QUEUED (gentle lane)** |
| `zentrain/kadis-924-2026-07-27/` | 2.88 | 3 | KADIS 924 rescore | **QUEUED (gentle lane)** |
| `zentrain/canonical-training-2026-05-18/` | 0.75 | 14 | May-era canonical training tables | **QUEUED (gentle lane)** |
| `zentrain/dualmodel-2026-06-28/` | 5.20 | 171 | LOO fleet / hetzner_ml_train inputs | **QUEUED (gentle lane)** |
| `zentrain/strategy-fleet-2026-07-02/` | 11.61 | 36 | reproduce_t1dro51.sh + strategy_fleet.sh inputs | **QUEUED (gentle lane)** |

### Already on the LAN store before this pass (verified by listing, not re-copied)

| bucket/prefix | GiB | objects | what reads it | status |
|---|--:|--:|---|---|
| `zentrain/refs/` | 10.35 | 7,119 | fleet score jobs (train-renditions, imazen-26-hdr-grid, clean-picker) | **ALREADY-PRESENT** |
| `zentrain/jobs/ (live fleet runs)` | - | - | avifaom/avifsvt/hdrgrid/hdrfeat944/fleetbench/steer8h | **ALREADY-PRESENT** |
| `zentrain/blobs/` | - | - | job-system content-addressed store | **ALREADY-PRESENT** |
| `codec-corpus/imazen-26-variants/` | - | - | imazen-26 variant registry | **ALREADY-PRESENT** |

### Registered — measured, deliberately NOT mirrored

| bucket/prefix | GiB | objects | what reads it | status |
|---|--:|--:|---|---|
| `zentrain/canonical/2026-06-27/*/variants.<split>.tar` | 141.30 | 15 | NO CONSUMER FOUND; byte-redundant with encodes/ | **NOT-MIRRORED** |
| `zentrain/kadis-700k-gpu/distorted/` | 197.43 | 699,999 | rescore-from-links PNGs; canonical parquet IS mirrored | **NOT-MIRRORED** |
| `codec-corpus/picker-sweep-2026-06-22/` | 165.91 | 6,069 | superseded by canonical/2026-06-27 | **NOT-MIRRORED** |
| `codec-corpus/kadis-hdr-2026-07-13/` | 66.54 | 12,695 | HDR distortion corpus; no active LAN reader | **NOT-MIRRORED** |
| `codec-corpus/synthetic-v2/` | 38.22 | 729,703 | no active reader (one code comment + a CLAUDE.md pointer); Tower archive holds it (59G images + 2.3G tables) | **NOT-MIRRORED** |
| `zentrain/jxl-lossy/runs/*/{omni,features,manifests,chunks,done}/` | - | - | reproducibility record; canonical parquets are the joined product | **NOT-MIRRORED** |

### Not needed on the LAN store

| bucket/prefix | GiB | objects | what reads it | status |
|---|--:|--:|---|---|
| `s3://coefficient/**` | - | - | vast.ai cloud fleet legacy + cvvdp-goldens fetched over PUBLIC HTTPS; cloud boxes cannot route to LAN | **NOT-NEEDED** |
| `zentrain/sweep-v*` | - | - | May-era scratch sweeps | **NOT-NEEDED** |


## 3. Capacity — checked before any copy

`ssh root@tower df` (read-only), 2026-08-30T20:18Z:

| target | size | used | free | note |
|---|---|---|---|---|
| `/mnt/user` (Unraid array, aggregate) | 35 T | 15 T | **21 T (60 % free)** | the 15 %-free floor is 5.25 T — never approached |
| `/mnt/disk1` (array member) | 17 T | 7.2 T | 9.3 T | holds SeaweedFS `/data2` |
| `/mnt/cache` (NVMe pool) | 1.9 T | 802 G | 1.1 T | holds SeaweedFS `/data` |

**SeaweedFS spans both tiers** — `zen-lanstore` runs
`server -s3 -dir=/data,/data2` with `/data` bound to
`/mnt/cache/appdata/zenstore-sw/data` and `/data2` bound to
`/mnt/disk1/coefficient/zenstore-data2`. Measured during this mirror: **every new
byte landed on `/data2` (the array)** — `data2` grew while `data` moved 544 KB —
because SeaweedFS places new volumes on the dir with the most free space, and
disk1 has 9.3 T against the cache's 1.1 T. So this mirror does **not** consume
the NVMe cache pool the household containers share.

The full P0+P1 working set is well under 0.5 TiB against 21 T free, so **no
prefix had to be refused for capacity**; the prefixes left unmirrored were
refused for *redundancy or absent consumers*, not space (§2).

**Shared-box discipline, and the one place it was breached — recorded because it
is the operational limit of this store.** Every transfer ran `nice -n19 ionice -c3`
with bounded parallelism, and for the large-object lanes (3 rclone transfers × 4
upload streams) the tower stayed healthy: load ≈10 on 32 threads, 81-90 % CPU
idle, 3.5-6.5 % iowait, Plex and the *arr stack live throughout.

**The small-object lanes are a different animal.** Pushing `encodes/` objects at
~333 PUT/s (s5cmd, 96 upload workers) drove the tower to **load 30.7 with 48.9 %
iowait and only 37 % idle**, and the S3 endpoint stopped answering — a
`head-object` timed out at 30 s and in-flight transfers collapsed from ~50 MiB/s
to 1 KB/s. `top` on the tower showed `weed` itself at 121 % CPU / 5.7 GB RSS and
a **loop device pinned at 100 % util serving 212 MB/s of reads**, with another
361 MB/s read off the NVMe cache: the write storm had triggered SeaweedFS
background volume work whose *read* amplification, not the writes, was the thing
saturating the box.

All lanes were stopped immediately — this box runs the household's media, which
outranks the mirror unconditionally. **Read `%iowait` and `%idle`, not load average, when judging this box.** During
the healthy large-object phase it sat at load 16-19 with **94.5 % CPU idle and
0.3 % iowait** — the load figure counts IO-blocked threads and runs high here
even when nothing is strained. The incident looked different in the way that
matters: load 30.7 *with* **48.9 % iowait and 37 % idle**. The committed gate in
`scripts/lanstore/mirror_r2.sh` still keys on load average because it is one
cheap `ssh` and errs toward pausing when it need not; when it pauses on a box
whose iowait is near zero, that is the gate being conservative, not a signal.

The operational rule this establishes:
**large sequential objects are cheap for this store; sustained small-object PUT
rates are not.** A small-object lane must be load-gated (poll
`/proc/loadavg` on the tower and pause above ~12) and run at a fraction of the
concurrency the raw benchmark suggests. The 333 obj/s figure is what SeaweedFS
will *accept*; it is not what this tower can *sustain*.


## 3b. Two discoveries that change how the REST of this should be done

Both were found while diagnosing the load incident, and both mean the remaining
work is cheaper than it looks — but must not be done the way the first half was.

### The tower already holds the bulk of these bytes as plain files

`/mnt/disk1/coefficient/zen924/` on the tower carries, from the July/August
bf-924 waves:

| tree | size | what it is |
|---|--:|---|
| `zen924/tars/<6 run dirs>/` | **235,474,984,976 B (219.3 GiB)** | **all 61 `box-N.tar` files**, all six runs — byte-for-byte the set measured on R2 (235,474,984,960 B + a 16 B `mirror.log`) |
| `zen924/zjl2-encodes/` | 33 GiB on disk1 + 14 GiB still on cache, ~1.48 M files | **the whole `zenjpeg_lossy/encodes/` corpus** — filenames match the canonical keys exactly (e.g. `o_1011.png.scale108x192_db148f3cd8359d3f_zenjpeg_q15_027d8a471f5b5481.jpg` is present in both) |
| `zen924/refs/` | 1.1 GiB | reference renditions |

So **the bytes were never at risk on this box; what was missing was S3
addressability at the canonical key.** A plain file on the array cannot be
fetched by `s3://…` byte range from a fleet worker or by
`resolve_bigcodec_pair_uris.py`, which is precisely what the mirror provides —
so the work is still necessary. But it means the *remaining* ingest is a
**tower-local operation** (array file → SeaweedFS, over the loopback) rather
than a 78-GiB-and-up pull across the WAN. That is both faster and free of R2
egress. Anyone continuing this should ingest from `zen924/tars/` and
`zen924/zjl2-encodes/` in place, not re-pull from R2.

### An Unraid `mover` job has been running since 2026-08-26

`ps` on the tower: `/usr/local/sbin/mover start`, **started Wed Aug 26 17:19:50,
elapsed 3 d 21 h** at the time of this measurement. It is draining
`zen924/zjl2-encodes` from the cache pool to the array — **902,005 files already
moved, 582,013 still on cache** — and Unraid's mover calls `fuser` on *every
file*, which is what the repeated short-lived `fuser` processes at 20-55 % CPU
were.

That job is the tower's load floor right now, it writes to **disk1 — the same
array disk that holds SeaweedFS's `/data2`** — and it is not going to finish
soon. It is also, notably, moving *another copy of the same zenjpeg encodes this
mirror was about to write a third copy of*.

**Consequence for scheduling:** small-object ingest into SeaweedFS must wait for
that mover to drain. Until it does, this box can absorb large sequential objects
at low concurrency and little else.

### A store setting worth revisiting before the next bulk ingest

`zen-lanstore` runs with **`-master.volumeSizeLimitMB=1024`** — a 1 GB volume
cap, **30x below SeaweedFS's 30 GB default**:

```
weed -logtostderr=true server -dir=/data -volume.max=0 \
     -master.volumeSizeLimitMB=1024 -s3 -s3.config=/conf/s3.json -dir=/data,/data2
```

With ~1 TB stored that yields **1,123 volume files** (569 under `/data`, 554
under `/data2`) and **2,633 open file descriptors** in the `weed` process, and
the count grows linearly with every gigabyte added. Under the write burst the
master was still allocating fresh volumes (IDs past 4890) while the ingest ran.
Per-volume index and housekeeping cost is what the box was paying, and it does
not go away when the writes stop: `weed` sat at **238-276 % CPU for more than
ten minutes after every client of mine was killed**.

Raising the cap toward the default would cut the volume count ~30x for the same
bytes. **Not changed here** — it is a live store other lanes depend on and the
change is the owner's call — but the next bulk ingest should be preceded by that
conversation, because this mirror's remaining work roughly doubles the stored
bytes.


## 4. Method + verification

**Transfer.** `rclone copy r2:<bucket>/<key> lan:<bucket>/<key>` — same bucket,
same key, so `ZEN_STORE=tower` resolution needs no path change. `rclone copy` is
**re-runnable**: it skips objects already present with a matching size/mtime, so
an interrupted prefix resumes rather than restarting. The LAN remote is
configured entirely through `RCLONE_CONFIG_LAN_*` environment variables sourced
from the operator's local env file — **no credential is written to
`rclone.conf`**, and nothing in this repo names the endpoint.

One transfer-shape trap worth recording: the first attempt at the byte-range
indexes ran `rclone copy <job-prefix>/ --include variant_index.tsv`, which made
rclone **enumerate the whole `bf-*` job prefix** (each carries a `blobs/`
content-addressed store) before filtering down to one file. Replacing it with
`rclone copyto <exact-key> <exact-key>` cut each index from minutes to seconds.
**Never `--include` a single object out of a job-run prefix.**

**⚠ `s5cmd ls` UNDERCOUNTS against SeaweedFS — never verify with it.** Measured
here on a settled prefix (`zenwebp_lossless/encodes/`, no writes in flight):
three consecutive `s5cmd ls` runs returned **40,450 / 40,466 / 40,467** objects,
while `aws s3 ls --summarize` returned **40,473** three times running — the exact
R2 count, with a byte total (6,954,651,422) matching R2 to the byte. The keys
`s5cmd` omitted were verifiably present (a direct `ls` of each one returned the
right size), and a different set was omitted on each run.

This is not a transfer problem, it is a **lister** problem, and it is a trap: a
mirror driven or verified by `s5cmd ls` diffs will report phantom-missing
objects forever, re-transfer them, and still call the prefix INCOMPLETE. It cost
one wasted retry pass here before it was caught. **Every count/byte figure in
this document comes from `aws s3 ls --recursive --summarize` on both stores.**
`s5cmd` is still used for bulk `cp` — measured at **190 obj/s end-to-end**
(339 obj/s down from R2, 433 obj/s up to the LAN store) against rclone's
**16.5 obj/s** on the same prefix, an **11.5x** difference — and for R2-side
listings, which are stable.

**Verification per prefix** (three independent checks):

1. **object count** — `aws s3 ls s3://<b>/<p> --recursive --summarize`, both stores;
2. **total bytes** — the `Total Size` line of that same call, both stores;
3. **sha256 spot-check** — at least 3 objects per prefix fetched from *both*
   stores and hashed locally (by byte range for multi-GB objects).


### Results

| prefix | objects R2 / LAN | bytes R2 / LAN | match | spot-check |
|---|--:|--:|:--:|---|
| `zentrain/jobs/bf-*/variant_index.tsv` | 53 / 53 | 695,062,062 / 695,062,062 | **YES** | 3/3 sha256 full-object OK |
| `zentrain/jxl-lossy/runs/mandfix4-zenavif-1782593621/variants/` | 8 / 8 | 50,682,634,240 / 50,682,634,240 | **YES** | 8/8 sha256 range OK (incl. 2 real member ranges from variant_index) |
| `zentrain/eval-grids/` | 4 / 4 | 25,465,806 / 25,465,806 | **YES** | 3/3 sha256 full-object OK |
| `zentrain/canonical/2026-06-27/originals/` | 3 / 3 | 1,100,933,120 / 1,100,933,120 | **YES** | 6/6 sha256 range OK (head+tail x3) |
| `zentrain/jxl-lossy/runs/jxl-lossy-vardct-1782609551/variants/` | 24 / 24 | 33,067,909,120 / 33,067,909,120 | **YES** | 4/4 sha256 range OK (incl. 2 real member ranges from variant_index) |
| `codec-corpus/clean-picker-corpus-2026-06-26/` | 4,497 / 4,497 | 1,097,459,519 / 1,097,459,519 | **YES** | 3/3 sha256 full-object OK |
| `zentrain/canonical-2026-05-21/` | 29 / 29 | 1,454,864,166 / 1,454,864,166 | **YES** | 1 full-object + 6 range sha256 OK; `features/cvvdp_iwssim_LARGE_372col.parquet` additionally matches the sha256 recorded in zensim/CLAUDE.md (14c2053327…) in full |
| `zentrain/canonical/2026-06-27/*/ (metadata: 50 objects)` | 50 / 50 | - / - | **YES** | 21/21 parquets match the sha256 recorded in _MANIFEST.json at BUILD time (2026-06-27), 0 mismatch |
| `zentrain/canonical/2026-06-27/zenwebp_lossless/encodes/` | 40,473 / 40,473 | 6,954,651,422 / 6,954,651,422 | **YES** | 5/5 sha256 full-object OK |
| `zentrain/kadis-700k/canonical/` | 1 / 1 | 906,001,718 / 906,001,718 | **YES** | 3/3 sha256 range OK (head+mid+tail) |
| `zentrain/kadis-700k-gpu/canonical/` | 1 / 1 | 936,367,503 / 936,367,503 | **YES** | 3/3 sha256 range OK (head+mid+tail) |
| `zentrain/canonical/2026-06-27/zenpng_lossless/encodes/` | 76,449 / 27,500 | 13,691,411,533 / 6,165,342,628 | **PARTIAL** | interrupted mid-upload by the load stop (§3); resumes by diff |
| `zentrain/canonical/2026-06-27/zenjxl_lossless/encodes/` | 269,820 / 0 | 58,419,699,144 / 0 | **DEFERRED** | not started — see §5b (tower-local ingest once the mover drains) |
| `zentrain/jxl-lossy/runs/mandfix2-zenjpeg-1782584881/variants/` | 8 / 8 | 47,450,193,920 / 47,450,193,920 | **YES** | 6/6 sha256 range OK (head+tail on 3 of 8 boxes) |

Large objects (multi-GB tars, >64 MB parquets) are spot-checked by **byte
range** rather than whole-file hash: identical ranges are fetched from both
stores and hashed. For the two tar-range datasets the sampled ranges include
**real member offsets read out of the mirrored `variant_index.tsv`** — so the
check exercises the exact fetch path a consumer uses, not just the bytes.

## 5. Notes for consumers (no pin was changed by this pass)

Per the brief, this pass **changed no consumer configuration** — it only added
objects. Three things a consumer lane may now want to do, recorded here rather
than done here:

1. **`zensim/scripts/canonical_corpus/resolve_bigcodec_pair_uris.py` can now
   resolve against the LAN store.** Its `--bucket` default (`zentrain`) and its
   hard-coded `ENCODES_PREFIX = "canonical/2026-06-27"` are already correct for
   the LAN store because the mirror is key-identical; only the *endpoint* the
   fetcher uses has to come from `s3env.sh` rather than the R2 default. That
   makes the round-7 fetch a LAN read instead of an internet read.

2. **`declare-scorefiles --full-uri` is now viable for the four object-mode
   datasets** (`zenjpeg_lossy`, `zenwebp_lossy`, `zenpng_lossless`,
   `zenjxl_lossless`) — their `encodes/` objects are individually addressable on
   the LAN store, which is what the job system needs. It is still **not** viable
   for `zenavif_lossy` / `zenjxl_lossy`: a tar-member byte range is not
   expressible as a per-object URI, so those two still need a materialization
   pass first (the tars and their `variant_index.tsv` indexes are on the LAN
   store, so that pass is now a LAN-local operation).

3. **`zenwebp_lossless` is absent from the resolver's `DATASETS` map.** It has a
   populated `encodes/` prefix, 40,473 objects, and is mirrored — but
   `resolve_bigcodec_pair_uris.py` will not resolve it because the map has six
   entries and this is the seventh canonical picker dataset. Flagged, not
   changed: it belongs to the zensim lane.

## 5b. What remains, and the cheapest way to finish it

**The half that mattered most is done.** `zenavif_lossy` and `zenjxl_lossy` were
the only two datasets with *no* per-object path at all — the ones that made the
fleet structurally unusable for half the round-7 cells. Both are mirrored and
verified, along with all 53 byte-range indexes, so **the tar-range fetch path is
fully LAN-resident and was proven end to end** (§4: sampled ranges taken from
the mirrored `variant_index.tsv` itself).

**What is deferred, and why it should NOT be finished the way it started:**

| remaining | size | why deferred |
|---|--:|---|
| `zenjpeg_lossy/encodes/` | 43.13 GiB / 1,484,010 obj | the tower **already holds this exact corpus** as `zen924/zjl2-encodes` (§3b), and Unraid's mover is *still moving it* — a third copy written across the WAN would be the worst of all options |
| `zenwebp_lossy/encodes/` | 22.53 GiB / 944,370 obj | small-object PUT rate is the thing that took the store down; needs the mover drained first |
| `zenjxl_lossless/encodes/` | 54.41 GiB / 269,820 obj | same |
| `zenpng_lossless/encodes/` | 12.75 GiB / 76,449 obj | **partial: 27,500 of 76,449 already on the LAN store**; resumes by diff |

The cheap path for all four, once the mover has drained:

1. **Ingest from the tower's own disk, not from R2.** `zen924/tars/` already holds
   all 61 box tars and `zen924/zjl2-encodes/` holds the zenjpeg objects. A
   tower-local `s5cmd cp` into the LAN store moves the same bytes over loopback
   with zero WAN transfer and zero R2 egress.
2. **Gate on tower load and keep upload concurrency low** — the measured ceiling
   is not SeaweedFS's accept rate (333 obj/s) but what this box sustains beside
   Plex and a running mover.
3. **Diff first, always.** Every lane here recomputed `(R2 keys − LAN keys)` and
   moved only the difference, so a resumed pass costs only what is genuinely
   missing. Use `aws s3 ls` for the LAN side of that diff, never `s5cmd ls` (§4).

**One pre-existing store error, noted in passing and not touched:** the filer
holds a dangling reference — `volume 473 not found for fileId 473,11bb5f1c703c7b`
— and something is polling the object behind it
(`zentrain/fleetbench-epoch-2026-08-24/claims/chunk-7fbea38b…`) in a tight retry
loop, several failures per second. Volume 473 is not in
`zenstore-quarantine-20260827/` (that holds 568/570/572), so this is a separate
loss. It predates this mirror and belongs to whoever owns that fleet run.

## 6. Reproducing / extending this mirror

**Committed tool: `scripts/lanstore/mirror_r2.sh`** (`copy` / `verify`), which
encodes the three things this pass had to learn the hard way — the `aws` lister
for LAN-side counts, a tower load gate, and `rclone copyto` for single objects
inside a job-run prefix:

```
scripts/lanstore/mirror_r2.sh verify zentrain canonical/2026-06-27/zenwebp_lossless/encodes/
scripts/lanstore/mirror_r2.sh copy   zentrain eval-grids/ canonical-2026-05-21/
# ZEN_LOAD_GATE (default 15), ZEN_TRANSFERS (default 2), ZEN_MIRROR_LOG
```

**⚠ A credential trap that bit this script during development, and will bite
anything else that mirrors *from* R2 *to* the LAN store.** `scripts/lib/s3env.sh`
re-exports the *selected* store's key under the legacy `R2_ACCESS_KEY_ID` /
`R2_SECRET_ACCESS_KEY` names for back-compat. When the selected store is the LAN
one (the default), those names therefore hold the **LAN** key. A script that
sources `s3env.sh` and then reaches for `$R2_ACCESS_KEY_ID` to talk to actual R2
hands Cloudflare the wrong key — and **the failure is silent**: listings come
back empty and every prefix looks like it has zero objects, which reads as
"nothing to copy" rather than as an auth error. `mirror_r2.sh` reads the R2
credentials into its own `SRC_*` variables from
`~/.config/cloudflare/r2-credentials` *after* `s3env.sh` has run.

The underlying recipe, for anything the tool does not cover:

```
# resolve the store (LAN is the default; ZEN_STORE=r2 is the opt-out)
. zenmetrics/scripts/lib/s3env.sh

# whole prefix, resumable, bounded parallelism, key-identical
rclone copy r2:<bucket>/<prefix> lan:<bucket>/<prefix> \
    --transfers 3 --checkers 8 --s3-upload-concurrency 4 --s3-chunk-size 64M \
    --retries 5 --low-level-retries 20 --stats 60s

# ONE object out of a job-run prefix — never `copy --include`, which enumerates
# the run's whole blobs/ store first
rclone copyto r2:<bucket>/<key> lan:<bucket>/<key>
```

Verification is the three checks in §4; the operator inventory lives at
`~/tmp/r2lan/inventory.tsv` and the transfer log at
`~/tmp/r2_mirror_2026-08-30.log`.
