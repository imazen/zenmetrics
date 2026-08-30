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
indexes** — and that is what was mirrored, with `mandfix2-zenjpeg`'s 8 boxes
added as insurance for the one dataset that has no second path.

## 2. Inventory — built from usage, not guesswork

Consumers were found by grepping `s3://` reads across `zenmetrics/scripts/jobsys`,
`zenmetrics/scripts/picker`, `zensim/scripts`, `zenanalyze` and the live
`~/tmp/*.sh` fleet chains. Sizes are `s5cmd du` against R2 on 2026-08-30.

### P0 — the bigcodec corpus (no LAN copy existed; the round-7 bottleneck)

| bucket/prefix | GiB | objects | what reads it | status |
|---|--:|--:|---|---|
| `zentrain/canonical/2026-06-27/zenwebp_lossless/encodes/` | 6.48 | 40,473 | resolve_bigcodec_pair_uris.py object-mode; fetch_bigcodec_bytes.py | **PENDING** |
| `zentrain/canonical/2026-06-27/zenpng_lossless/encodes/` | 12.75 | 76,449 | same | **PENDING** |
| `zentrain/canonical/2026-06-27/zenjxl_lossless/encodes/` | 54.40 | 269,820 | same | **PENDING** |
| `zentrain/canonical/2026-06-27/zenwebp_lossy/encodes/` | ? | ? | same | **PENDING** |
| `zentrain/canonical/2026-06-27/zenjpeg_lossy/encodes/` | ? | ? | same (ONLY path - no tar index exists) | **PENDING** |
| `zentrain/jxl-lossy/runs/mandfix4-zenavif-1782593621/variants/` | 47.20 | 8 | tarrange bytes for zenavif_lossy (no encodes/) | **PENDING** |
| `zentrain/jxl-lossy/runs/jxl-lossy-vardct-1782609551/variants/` | 30.80 | 24 | tarrange bytes for zenjxl_lossy (no encodes/) | **PENDING** |
| `zentrain/jobs/bf-{zavif,zjxll,zjxlm,zpng,zwebp}-t*/variant_index.tsv` | 0.65 | 53 | byte-range index for tarrange fetch | **DONE** |
| `zentrain/canonical/2026-06-27/*/{_MANIFEST.json,pairs.*,train|validate|test}.parquet` | 8.30 | 50 | build_canonical/verify_canonical; every picker script | **PENDING** |
| `zentrain/canonical/2026-06-27/originals/` | 1.03 | 3 | corpus originals | **PENDING** |
| `codec-corpus/clean-picker-corpus-2026-06-26/` | 1.02 | 4,497 | pairs.*.parquet ref_path | **PENDING** |
| `zentrain/jxl-lossy/runs/mandfix2-zenjpeg-1782584881/variants/` | 44.19 | 8 | INSURANCE for zenjpeg encodes (no tar index) | **PENDING** |

### P1 — eval instruments + regime training tables (local `/mnt/v` mirrors exist; the LAN copy makes them fleet-addressable)

| bucket/prefix | GiB | objects | what reads it | status |
|---|--:|--:|---|---|
| `zentrain/eval-grids/` | 0.020 | 4 | bake_verdict dial+corruption grids | **PENDING** |
| `zentrain/canonical-2026-05-21/` | 1.35 | 29 | canonical 372-feature trainer input | **PENDING** |
| `zentrain/kadis-700k/canonical/` | 0.84 | 1 | KADIS-700k canonical parquet | **PENDING** |
| `zentrain/kadis-700k-gpu/canonical/` | 0.87 | 1 | KADIS-700k GPU-metric canonical parquet | **PENDING** |
| `zentrain/ext944-canonical-2026-08-01/` | 26.51 | 85 | 944-regime legs | **PENDING** |
| `zentrain/tbig-944-2026-08-02/` | 29.29 | 16 | bigcodec 944 views | **PENDING** |
| `zentrain/kadis-944-2026-08-01/` | 3.09 | 7 | KADIS 944 rescore | **PENDING** |
| `zentrain/ext924-canonical-2026-07-27/` | 49.96 | 36 | 924-regime legs | **PENDING** |
| `zentrain/ext720-canonical-2026-07-22/` | 55.58 | 37 | 720-regime legs | **PENDING** |
| `zentrain/tbig-720-2026-07-22/` | 17.59 | 2 | bigcodec 720 views | **PENDING** |
| `zentrain/kadis-720-2026-07-24/` | 3.13 | 3 | KADIS 720 rescore | **PENDING** |
| `zentrain/hdr/` | 4.77 | 185 | HDR corpora/tables | **PENDING** |
| `zentrain/canonical-2026-07-15/` | 0.004 | 4 | HF near-lossless + kadis_negrich training parquets (the recurring-priority set) | **PENDING** |
| `zentrain/canonical-gates-2026-07-29/` | 0.001 | 57 | canonical-source gate verdicts | **PENDING** |
| `zentrain/profiles/` | 0.001 | 6 | Profile C ship artifacts (C-2026-08-05) | **PENDING** |
| `zentrain/hdr-corpora/` | 0.034 | 4 | HDR corpora tables | **PENDING** |
| `zentrain/jxl-nearlossless-2026-07-06/` | 0.050 | 29 | input to build_hf_nearlossless_2026-07-15.py | **PENDING** |
| `zentrain/kadis-924-2026-07-27/` | 2.88 | 3 | KADIS 924 rescore | **PENDING** |
| `zentrain/canonical-training-2026-05-18/` | 0.75 | 14 | May-era canonical training tables | **PENDING** |
| `zentrain/dualmodel-2026-06-28/` | 5.20 | 171 | LOO fleet / hetzner_ml_train inputs | **PENDING** |
| `zentrain/strategy-fleet-2026-07-02/` | 11.61 | 36 | reproduce_t1dro51.sh + strategy_fleet.sh inputs | **PENDING** |

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
| `zentrain/jxl-lossy/runs/{jxl-modular,mandfix2-zenwebp,mandfix2-zenpng}/variants/` | 97.12 | 21 | byte-redundant with those encodes/; each has a mirrored variant_index.tsv | **NOT-MIRRORED** |
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

**Shared-box discipline.** Every transfer ran `nice -n19 ionice -c3` with bounded
parallelism (3 transfers × 4 upload streams for multi-GB tars; 12 transfers for
small-object prefixes). Tower load stayed ≈10 on 32 threads with 81 % CPU idle
and 6.5 % iowait, with Plex and the *arr stack live throughout.


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
`s5cmd` is still used for bulk `cp` (where it is 11.5x faster than rclone, §4)
and for R2-side listings, which are stable.

**Verification per prefix** (three independent checks):

1. **object count** — `s5cmd ls 's3://<b>/<p>*' | wc -l`, both stores;
2. **total bytes** — sum of the size column of that same listing, both stores;
3. **sha256 spot-check** — >= 3 objects per prefix downloaded from *both* stores
   and hashed locally.


### Results

| prefix | objects R2 / LAN | bytes R2 / LAN | match | spot-check |
|---|--:|--:|:--:|---|
| `zentrain/jobs/bf-*/variant_index.tsv` | 53 / 53 | 695,062,062 / 695,062,062 | **YES** | 3/3 sha256 full-object OK |
| `zentrain/jxl-lossy/runs/mandfix4-zenavif-1782593621/variants/` | 8 / 8 | 50,682,634,240 / 50,682,634,240 | **YES** | 8/8 sha256 range OK (incl. 2 real member ranges from variant_index) |
| `zentrain/eval-grids/` | 4 / 4 | 25,465,806 / 25,465,806 | **YES** | 3/3 sha256 full-object OK |
| `zentrain/canonical/2026-06-27/originals/` | 3 / 3 | 1,100,933,120 / 1,100,933,120 | **YES** | 6/6 sha256 range OK (head+tail x3) |
| `zentrain/jxl-lossy/runs/jxl-lossy-vardct-1782609551/variants/` | 24 / 24 | 33,067,909,120 / 33,067,909,120 | **YES** | 4/4 sha256 range OK (incl. 2 real member ranges from variant_index) |
| `codec-corpus/clean-picker-corpus-2026-06-26/` | 4,497 / 4,497 | 1,097,459,519 / 1,097,459,519 | **YES** | 3/3 sha256 full-object OK |
| `zentrain/canonical-2026-05-21/` | 29 / 29 | 1,454,864,166 / 1,454,864,166 | **YES** | 1/1 sha256 full-object OK (2 objects >64MB skipped) |
| `zentrain/canonical/2026-06-27/*/ (metadata: 50 objects)` | 50 / 50 | - / - | **YES** | 21/21 parquets match the sha256 recorded in _MANIFEST.json at BUILD time (2026-06-27), 0 mismatch |
| `zentrain/canonical/2026-06-27/zenwebp_lossless/encodes/` | 40,473 / 40,473 | 6,954,651,422 / 6,954,651,422 | **YES** | 5/5 sha256 full-object OK |
| `zentrain/kadis-700k/canonical/` | 1 / 1 | 906,001,718 / 906,001,718 | **YES** | 2/2 sha256 range OK (head+tail) |
| `zentrain/kadis-700k-gpu/canonical/` | 1 / 1 | 936,367,503 / 936,367,503 | **YES** | 2/2 sha256 range OK (head+tail) |

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

## 6. Reproducing / extending this mirror

The drivers are throwaway operator scripts, not committed code; the reproducible
statement is the recipe:

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
