# Fleet plan: 924-feature backfill on the household nodes (2026-07-27)

**Mission for the coordinating session:** backfill the streaming-only
folded+append **924-feature** extraction (zensim ≥ `0b3d16b0`, regime
`Folded720Append`) across ALL canonical datasets — the 11 local legs,
KADIS-700k, and **bigcodec/tbig (5.74M rows — explicitly in scope per user
directive)** — running entirely on the household fleet. No paid cloud.

**Read first, in order:** `~/work/zen/DATA_PROVENANCE.md` (canonical data
index) · `zenmetrics/NODES.md` (node runbook — MANDATORY before touching
any node) · `zenmetrics/docs/RUNNING_JOBS.md` + `docs/PLAN_SWEEPS.md` (the
job system — NEVER hand-roll fleet orchestration) ·
`zensim/STREAMING_FOLDAPP_PLAN_2026-07-26.md` +
`zensim/benchmarks/v2_append_block_2026-07-26.md` +
`streaming_foldapp_gates_2026-07-26.md` (what the extractor is; ~65 ms/pair
1-thread at ~1 MP, 221 MB peak at 12 MP, O(width) memory) ·
`zenpapers/docs/zensim-bake-subset-plans-2026-07-26.md` (what the backfill
unblocks — P12/P11/P13 run the moment wave 1 lands).

## Ground truth: where the data is (audited 2026-07-27 from wsl)

| Need | Local (/mnt/v, wsl-only) | Tower NFS (`/mnt/tower` = `tower:/mnt/user/coefficient`, 2.7 TB free) | R2 |
|---|---|---|---|
| 11 local-leg sources+pairs (149,195 rows) | ✓ (`/mnt/v/dataset/*`, pairs per `ext720-canonical-2026-07-22/_MANIFEST.json` provenance) | mirrors of 720 OUTPUTS only (`output/zensim-ext720-canonical-2026-07-22/`) | ✓ outputs |
| KADIS-700k distorted PNGs (700k) | ✗ (parquets only) | ✗ (parquets only — **no `distorted/`**) | ✓ `s3://zentrain/kadis-700k-gpu/distorted/<chunk>/*.png` (rescore-from-links; NEVER regenerate — RNG diverges, see provenance) |
| KADIS refs (140k pristine) | ✗ | check `v-datasets-archives-2026-07-22/` + `input/` | ✓ |
| bigcodec encoded-variant blobs (5.74M cells) | ✗ (`canonical-picker-2026-06-27/` = 8.9 GB parquets/pairs, NOT blobs) | ✗ (`tbig-720-2026-07-22/` = feature parquet only) | ✓ (canonical dirs; keyed `encode_sha` == `encoded_filename`) |
| bigcodec reference images | partial (`/mnt/v/output/canonical-picker-2026-06-27/` README/pairs point at R2 URLs) | `canonical-picker-2026-07-01-zensimA/` (verify contents) | ✓ |

**Answer to "is tower set up with all the files on NFS": no.** Tower holds
the 720-era OUTPUT mirrors; it lacks every raw INPUT the fleet waves need
(kadis distorted, bigcodec blobs, kadis refs unverified). The established,
proven data plane for node workers is **R2-ledger pulls** (the tbig-720
extraction itself ran this way — `fleet_blob_fetch_720.py`), not NFS; R2
egress is free, so home-fleet pulls cost only bandwidth. Only wsl mounts
tower NFS today; other nodes are not set up for it and should not need it.
Optional: stage kadis distorted (~hundreds of GB) onto tower for LAN-local
reads IF home bandwidth becomes the bottleneck — decide from wave-3 smoke
measurements, not upfront (tower has 2.7 TB free; array is 93% — check
`df` before writing, never fill it).

## Node roster + allocation (from NODES.md — re-read it; kids have priority)

wsl (operator + 28C worker) · lianli (always-Ubuntu worker; **currently
running claude-opus-coherent089's training grid — negotiate**) · tower
(`zen720-basement` docker worker, capped: `--cpuset-cpus` leave ≥8 free,
`--cpu-shares=256`, `--memory=40g`; media has priority; Docker-ONLY on the
host) · jason + ian (dual-boot kids' PCs — idle windows only, flip via
`fleet-pxe to-ubuntu <mac>`, ALWAYS return to Windows) · mac (M4 Pro,
idle-only launchd worker). NODES.md says "our 6 household nodes"; the user
says 7 — reconcile the roster with the user on session start (a 7th box
may be un-runbooked; the PXE path can enroll it).

**Concurrency ground rules:** another session (claude-opus-coherent089) is
actively training on lianli/jason + local evals on wsl. `.workongoing`
markers per repo (check and claim as SEPARATE tool calls; refresh ≤2 min);
one heavy job per box; `nice -n19 ionice -c3` everything local; do NOT
enroll a node that session is using — coordinate via the marker text +
`zenfleet` pool state (`docker ps` / worker units) before enrolling.

## Work items

**W0 — extractor into the job system (the only real integration work).**
The job system must gain the 924 extraction as a first-class job kind:
bump zenmetrics' zensim dep to ≥ `0b3d16b0` (path/git patch per the
unpublished-dep rule), plumb the `Folded720Append` streaming entry through
the sweep/jobexec feature-output path (DATA_PROVENANCE notes the old
`feature_output` plumbing gap — close it properly this time), and rebuild
the worker image per the BAKE-EVERYTHING rules: binary precompiled, no
apt/pip at boot, **existing canonical package name + a new TAG** (e.g.
`zenfleet-worker:exec-v28-zensim924`) — never a new package. Two-stage
launch: 1-box smoke, VERIFY artifacts/rows landed in R2 (`aws s3 ls`)
before scaling — the W44-PHASE4-S1 lesson.

**W1 — local legs (149,195 rows) on wsl.** Can run before/parallel to W0
(pure local, driver already exists): `v2_ab_extract` `ZENSIM_AB_MODE=
foldapp` per corpus pairs TSV (provenance in the ext720 manifest), rayon
multithread, ~10–30 min wall — but ONLY when coherent089's local eval is
done (one heavy job per box). Output: `/mnt/v/zen/zensim-training/
ext924-canonical-2026-07-27/` — one parquet per corpus, schema
`ref_basename, human_score, f0..f923`, ZSTD, NaN-free; `_MANIFEST.json`
with build_commit + sha256 + pairs provenance + role (adapt
`zensim/scripts/canonical_corpus/promote_ext720_canonical.py`); mirror R2
`s3://zentrain/ext924-canonical-2026-07-27/` + Tower
`/mnt/tower/output/zensim-ext924-canonical-2026-07-27/` (sha-spot-verify);
new DATA_PROVENANCE section. Then immediately kick P12 (residual-boost) +
P11 (auto-decorrelation) from the subset-plans doc — they only need wave 1.

**W2 — KADIS-700k (700k rows + negrich).** Job-system run,
rescore-from-links: workers pull `distorted/<chunk>/*.png` + refs from R2,
extract 924, write per-chunk sidecars to R2; converge via ledger
(declare→gap→reconcile). Output `kadis-924-<date>/kadis700k_924.parquet`
(699,999 × 939: f0..f923 + the 7 metric targets + source_id + ref_id) +
negrich sibling; local + R2 + Tower + provenance, mirroring the
`kadis-720-2026-07-24` shape. ~13 core-hours of extraction; wall dominated
by blob fetch — measure on the smoke box.

**W3 — bigcodec/tbig (5,742,660 cells — IN SCOPE, the big one).**
Job-system run over the canonical-picker corpus: fetch encoded blob by
`encode_sha`, decode, pair with its reference, extract 924, sidecar
keyed `encode_sha` — the exact `fleet_blob_fetch_720.py` /
`tbig_join_720.py` shape, re-pointed at the streaming extractor. ~104
core-hours extraction + decode + fetch; across wsl+lianli+tower+idle
windows ≈ 1–3 days wall. Then the views join (7 datasets × 3 splits, EXACT
`encoded_filename` match — reuse `tbig_join_720.py`, expect match_rate
1.0000) → `tbig_924_full.parquet` + views + `_JOIN_REPORT` + Tower mirror
+ provenance. Carry forward the documented hqdedup caveat (train views =
pre-dedup digits).

**W4 — promote + unblock training.** Update DATA_PROVENANCE (★ section),
verify counts against the 720-era row counts exactly, then hand off to the
training session: P12/P11 instruments → P13 duels → P1/P2/P5 per the
subset-plans execution order.

## Hard rules (pointers, all MANDATORY)

Regime purity: 924 rows are the folded-streaming regime — NEVER
column-mix into ext-720/v1 parquets (padded-width divergence + zeroed
pools; provenance "Regime rule"). Sidecar-first, content-addressed keys
(`encode_sha`), every `_MANIFEST.json` carries `build_commit`. Persist
everything a worker computes (encoded bytes already exist — this is
score-side, so sidecars only). Fleet: zenfleet job system ONLY (no
hand-rolled chunkers); ONE ghcr package, tags for variants; workers
verify baked tools at boot and fail loud. Nodes: kids' machines idle-only
and returned to Windows; tower Docker-only, capped, array never filled;
mac idle-only. Every long command tees to `~/tmp/*.log`; progress
heartbeats streamed and tailed. R2 creds: mint scoped tokens per
`~/work/claudehints/topics/r2-credentials.md` even for household boxes.
Commit bench/measurement docs to the repo per run.

## Acceptance gates

- G-W1: 11 parquets, row counts == ext720 counts exactly, f0..f923
  NaN-free, manifests + 3 mirrors sha-verified, DATA_PROVENANCE updated.
- G-W2: 699,999 rows + negrich 167,034; targets carried over byte-equal
  from the GPU canonical; ledger converged (gap==0).
- G-W3: 5,742,660 sidecar rows, join match_rate 1.0000 on all 21 splits,
  cross-pool dedupe + error-row accounting reported like the 720 run.
- G-ALL: two-stage smoke before every scale-out; no artifact-less burn;
  every wave's doc committed before the next wave starts.
