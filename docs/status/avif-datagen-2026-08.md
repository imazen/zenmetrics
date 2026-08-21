# SDR AVIF datagen on the household fleet — living status

**Living doc.** Updated by the campaign's driver loop while the run is in flight. The
terminal record is zensim campaign **appendix Z** (registered `ab22e07a`, 2026-08-06) plus
the corpus `_MANIFEST.json`.

- **Owner:** `claude-avifgen` lane.
- **Started:** 2026-08-06.
- **Mission:** the canonical fresh-pin SDR AVIF corpus — 1,455 renditions ×
  planner-audited knob/q grid = **564,300 encode cells** at zenavif `66e3c417`, GPU
  perceptual metrics + 944-regime features, content-addressed persistence.
- **Mandate:** USER DIRECTIVE 2026-08-06 lifts the 2026-07-13 avif-datagen halt: *"you can
  fleet up avif datagen, remember to use gpu machines for metrics and funnel encodes from
  the whole fleet, since they can be slow - specialize the gpu machines. have an agent
  handle it and edit zenmetrics repo as needed."*
- **Orchestration:** the zenfleet job system only (workspace mandate).

---

## Current state

| | |
|---|---|
| **Phase** | **CLOSED 2026-08-21 (LAN-only close-out).** Campaign terminal record: zensim appendix **Z.R** (`fb731572`). Rescue union 8,623/8,623; verdict RESCUE-WINS (3/2,000); G-Z5 re-gate **PASSED 0.999313**; views train_944 (459,780) + eval8_944 (104,520) emitted + manifested; mirrors /mnt/v + Tower (sha-verified), NO R2 copies. Corpus `_MANIFEST.json` sha `1cc50769…e6224397`. WAVE-12 DATA GATE: **OPEN**. |
| **Grid** | frozen in appendix Z (main 472,680 + probe 90,720 + huge 900; monsters >16MP excluded) |
| **zenavif pin** | `66e3c417b43ff950323d402824aeb1ecbbc7f683` (path-dep sibling; source tree == commit tree) |
| **Images** | CPU `ghcr.io/imazen/zenfleet-worker:exec-avifgen-66e3c417`; GPU `…:exec-gpu-avifgen-66e3c417` (both built locally from the pinned checkouts, pushed) |
| **Ledger fill** | see incidents log tail / `~/tmp/avifgen/progress.log` on the operator |
| **First-cell encode gate (G-Z2)** | **PASSED 2026-08-06 ~14:45** (sha match + decode-back + byte-stable) |
| **First-cell score gate (G-Z3)** | **PASSED 2026-08-06 ~15:00** on node-2 (800/800 finite rows, both butteraugli variants, refusal probe rc=101) |

### Milestones

- [x] Recon: budget doc, job-system caps, June estate precedent, zenavif state
- [x] Appendix Z pre-registered + pushed (`ab22e07a`) BEFORE launch
- [x] Planner dry-runs audited (dropped-axis + alias-merge + invalid-strata reports read)
- [x] Executor images rebuilt at the pin (CPU + GPU tags) + pushed
- [x] Encode run declared (564,300 jobs); scoped creds minted (run+refs prefixes, 7-day TTL)
- [x] **G-Z2 first-cell encode gate PASSED** (blob on R2 + ledger sha match + decode-back + byte-stability)
- [x] Fleet scaled: node-2, node-3, i265, ryzen5800xt, lianli (systemd units) + tower (Docker, cpuset 0-23, shares 256, mem 40g)
- [ ] Encode ledger 50%
- [ ] Encode ledger ≥99.5% (G-Z4)
- [x] ScoreFile GPU smoke declared + **G-Z3 gate PASSED on node-2** (full declare follows encode drain)
- [ ] lianli joins GPU queue; CPU queue (cvvdp + zensim-foldapp2 944, tier-matched) declared
- [ ] Score ledgers ≥99.5%; write-back parquets built
- [ ] `_MANIFEST.json` + orientation gate (G-Z5) + DATA_PROVENANCE/DATA_SPLITS entries + mirrors

## Fleet roles (per the user directive)

| Node | Role |
|---|---|
| `node-2` (RTX 3070 8 GB) | **GPU metric scoring** (ZENMETRICS_REQUIRE_GPU=1); encodes only until metric work is queued |
| `lianli` (RTX 2080 8 GB) | **GPU metric scoring** (same gate); encodes only until metric work is queued |
| `node-3`, `i265`, `ryzen5800xt` | encode + CPU metric queue (cvvdp, 944 features tier-matched) |
| `tower` | encode, Docker-only, cpuset-capped, media-first, observe-before-load |
| `wsl` | operator — declares, gates, harvests; does not measure |

## Coordination with the hdr-corpus lane

- Their **B1** (HDR encode flag) is not needed here (SDR path is the built-in default).
- Their **B2** diffmap-executor build is NOT duplicated here: encodes are persisted
  content-addressed, so diffmaps stay computable; a Diffmap follow-up declare is registered
  in appendix Z §Z.6.3 and will run when the executor exists (theirs, or built here if
  their lane stays idle — coordinated in this doc first).
- Their **B5** (AVIF arm user-halted) is superseded for **SDR** by the 2026-08-06 directive
  above; the HDR AVIF arm remains their call.
- Their **B6** fix (`ZENMETRICS_REQUIRE_GPU`, `5f6f06f4`) is adopted as gate G-Z3.

## Incidents / decisions log

| When (UTC) | What |
|---|---|
| 2026-08-06 ~13:37 | Claimed `zensim` + `zenmetrics` (append-style markers, own jj workspaces). zenavif inspected READ-ONLY: stale 08-01 marker, `M Cargo.lock` only, conflicted `hdr-mdcv-st2086-fix` bookmark — reported in Z.0, untouched. |
| 2026-08-06 ~13:50 | Corpus pixel estate measured from rendition names: 1,489 MP; the 57 >8MP renditions carry 78% of pixels; 27 >16MP monsters (to 101.8 MP) exceed the 8 GB-card VRAM budget (~348 MiB/MP ssim2-gpu) → excluded, recorded. |
| 2026-08-06 ~14:00 | Planner dry-runs at the pin: main 12×30 (budget dropped bit-depth-10 from rd_core, kept speeds×qm×subsampling), probe 27×30 (51,360 aliases merged, 1,664 invalid skipped), huge 1×30. Totals frozen into appendix Z. |
| 2026-08-06 ~14:05 | Appendix Z pushed (`ab22e07a`, merge-base-verified on zensim origin/main). |
| 2026-08-06 ~14:07 | Encode run `jobs/avifgen-enc-20260806` declared (564,300 jobs, manifest 283 MB). CPU image `ghcr.io/imazen/zenfleet-worker:exec-avifgen-66e3c417` built from the pinned checkouts + pushed. Local probe through the production jobexec path: SDR s4 ~1.1–1.7 s/MP, α ~0.17 s, RSS ~18 MB/MP (16 MP cell = 226 MB) — encode is 2–3× cheaper than the HDR-derived bound; no memory-admission risk. |
| 2026-08-06 ~14:30 | **G-Z2 incident (caught by the gate, before scale-up):** the two-bucket corpus credential (`ZEN_CORPUS_AWS_*`) is NOT threaded into the in-process objstore source GET — documented gap at `jobexec.rs:202` ("not yet threaded into objstore"). Every cell of the first 21-min gate window failed the corpus GET, **misclassified `encoder_panic`** (a second defect: source-fetch failures land in the encoder's error class), deterministic → poison. 359 all-FAILED sidecars (143,600 cells) verified by content + pre-restart timestamp, deleted; cells re-opened. Fix adopted: refs re-hosted at `s3://zentrain/refs/train-renditions-2026-06-14/` (single-bucket, the pattern every prior wave used); one scoped RW cred (run prefix + refs prefix). The objstore corpus-cred gap + the error-class mislabel are recorded here for the next zenfleet maintenance pass. |
| 2026-08-06 ~14:45 | **G-Z2 PASSED**: blob sha256 == ledger `output_sha`; `ftypavif` valid, decode-back + ssim2 sane (79.9 @ q50 tiny); **byte-stable** — container fleet encode == bare local re-encode, byte-identical; ledger rows carry full cell identity + worker/provider provenance. Milestone ping 1 sent. |
| 2026-08-06 ~14:50 | Fleet scaled: node-3/i265/ryzen5800xt/node-2/lianli via `homefleet zenmetrics/scripts/jobsys/avifgen_encode_bringup.sh` (unit + env + pinned image, all `active`); tower `zen-avifgen` container (cpuset 0-23, cpu-shares 256, mem 40g; array 95% full — nothing written to the array). ryzen was restart-looping a drained 720-era pool (done=0 cycles) — repurposed. GPU image `:exec-gpu-avifgen-66e3c417` built + pushed. `pairs_from_encode_ledger.py` added: the Encode→ScoreFile bridge (ledger → full-URI pairs parquet for `declare_direct_objects.py` ZEN_FULL_URI=1 + the writeback join table). Note for G-Z3: jobexec ScoreFile JSONL rows carry no `runtime` column — the gate therefore rests on `ZENMETRICS_REQUIRE_GPU=1` (CPU rung structurally removed) + an on-node hidden-GPU refusal probe, not on per-row provenance. lianli lacks the nvidia docker runtime (node-2 has it) — `gpu-setup.sh` queued before lianli joins the score queue. |
| 2026-08-06 ~15:00 | **G-Z3 PASSED on node-2** (run in parallel with the encode drain, on a 400-pair smoke cut from the partial ledger via the new bridge): 34 ScoreFile jobs drained in ONE 17 s pass — 800/800 rows, 0 errors, 0 non-finite, `butteraugli_max_gpu`+`butteraugli_pnorm3_gpu` on all 400, 388 unique blob inputs (12 cells content-dedup'd — writeback fans shared-sha scores to all matching cells by design). 7.2 MP pair ≈ 2.9 s/metric on the 3070 ≈ the 5070-measured 0.40 s/MP — no household-card derating penalty at this size. **Refusal probe**: same job, GPU hidden, `ZENMETRICS_REQUIRE_GPU=1` → rc=101, 0 rows (cudarc load panic) — the CPU rung is structurally unreachable; a CPU number can never land under a GPU column. Gate note: the entrypoint's loud boot-check keys on `ZEN_REQUIRE_GPU` (which exports the metrics var) — both are now set in the score env. lianli GPU container runtime installed (`gpu-setup.sh`; driver was already present, no reboot needed). |
| 2026-08-06 ~20:10 | **WAKE-UP AUDIT after a 4 h orphaned-waiter window (supervisor nudge; own miss, recorded honestly):** the exact-count drain waiter died silently at 15:55Z (the evidence-free bespoke-waiter class the latency discipline warns about) and the estimator's 16:50Z "DRAINED" was FALSE (ledger FILES × 400 ≫ cells). True state at 19:58Z: **encode 343,462/564,300 DONE (61%)**; the 5 CPU boxes + tower worked the whole time (2 h passes, chunk-durable), but **node-2 idled ~3 h** cycling the drained 34-job smoke queue because the full score declare was notification-gated. **Root cause of the tail slowdown found and FIXED in zenmetrics (`92432e37`): single-run workers never passed `--ledger-in`** — empty reconcile views made the gap ALL cells every pass, dedup fell to claim leases, and the ledger shows the measured tax: **1,237,361 rows for 343,465 distinct DONE (3.6×)**. Fix = the POOL-mode snapshot contract in single-run mode + `compact_ledgers.py` snapshot uploaded + both images rebuilt/pushed/rolled (node-3 pass 1 confirms `snap=24281335`). **GPU scoring started 20:01Z** on node-2 over the done 61% (27,989 ScoreFile jobs / 329,123 unique pairs). Endgame is now a session-independent daemon (`~/tmp/avifgen/endgame_daemon.sh`): snapshot refresh loop → exact-count drain detection (99.5 % floor honored) → final sf-gpu re-declare + sf-cpu declare → box flips (lianli→sf-gpu, CPU boxes+tower→sf-cpu) → score-phase snapshots → `SCORE_FINAL.done` sentinel. Revised ETAs: encode drain ~1–3 h; GPU queue remains the bound, ≈33 h from 20:00Z (node-2 solo until drain, then 2-way). |
| 2026-08-06 ~20:54 | **(epochshard lane, FYI — no action needed, nothing about this campaign changes):** epoch-sharded claiming landed in zenfleet (`zenfleet-core::epoch` + worker `--claim-mode epoch-sharded` + sim gates) — the level BELOW your `92432e37` snapshot fix: it removes the claim contention itself (deterministic rendezvous-hash shards per wall-clock epoch; leases only at ownership seams). **avifgen finishes on the current fixed lease mechanism** — the mode is opt-in, default stays lease, and the worker images you just rolled are untouched (the new flags ride into the image at the next natural rebuild; no mid-campaign rebuild is needed or wanted). |

## Per-box attribution (from the LEDGER — user-directed audit, 2026-08-06 ~21:30Z snapshot, encode run)

Whole-campaign encode-ledger scan (1,518,977 rows; 508,239 distinct DONE at scan time).
"effort" = all rows written (includes the pre-fix empty-view re-work era); "credit" =
latest-wins distinct-DONE attribution. credit/effort < 1 quantifies the re-work tax per box.

| worker | threads | rows (effort) | distinct-done credit | credit/effort | credit per thread·h (~6.5 h) |
|---|---|---|---|---|---|
| lianli | 24 | 318,522 | 135,246 | 0.42 | ~0.87 k |
| i265 | 20 | 323,204 | 97,361 | 0.30 | ~0.75 k |
| ryzen5800xt | 32 | 256,480 | 91,195 | 0.36 | ~0.44 k |
| tower (capped 24) | 24 | 184,534 | 90,402 | 0.49 | ~0.58 k |
| wsl-gate (operator) | ~28 | 310,800 | 65,516 | 0.21 | — (see note) |
| node-3 | 12 | 119,712 | 26,292 | 0.22 | ~0.34 k |
| node-2 | 16 | 5,725 | 2,227 | 0.39 | — (encoded ~10 min, then score queue / idle window) |

Findings, honestly stated:
- **Spread was real but uneven.** Five boxes + tower carried the run; per-thread credit spans
  ~2.5× (lianli 0.87k → node-3 0.34k). node-3 (6C/12T, the weakest box) under-contributed
  per-thread; ryzen5800xt's low per-thread number reflects SMT-heavy 32T on 16C. No box was
  image-stale; all consumed the same pinned image.
- **The re-work tax hit everyone** (credit/effort 0.21–0.49 before the `92432e37` fix; the
  fleet-wide 3.6× row inflation is the same fact viewed from the other side).
- **wsl-gate never self-stopped**: `ZEN_MAX_MIN` is enforced by POOL mode only — single-run
  mode ignores it, so the "25-min gate" container worked ~5–6 h on the operator box
  (unbudgeted heavy job; it exited on its own before 21:30Z). Recorded as an entrypoint
  follow-up: honor ZEN_MAX_MIN in single-run mode too.
- **node-2's near-zero share is the known miss** (idle on the drained smoke queue ~3 h),
  already recorded in the wake-up audit above.
- Ledger `ts` is pass-start-quantized (2 h passes → 2 h buckets), so per-hour throughput
  comes from the operator progress log, not row timestamps.

## Restart/resume invariants (user-directed verification, 2026-08-06)

**VERIFIED live:** `node-3` was deliberately restarted mid-run at 21:28Z — its first pass
fetched and consumed the done-set snapshot (`snap=33818862` in the pass-1 heartbeat) and
claimed only undone work. The campaign is kill-and-resume-safe at any point:

- **What survives a kill (all on R2):** content-addressed encode blobs (`blobs/<sha256>` —
  re-encoding a done cell reproduces the identical sha, so even redundant work is
  convergent); ledger chunk sidecars (written the moment each ~300 s chunk finishes);
  `ledger_snapshot.parquet` (refreshed ~every 15 min by the endgame daemon); manifests;
  claims (lease-expire on their own).
- **Loss bound on any worker kill:** only the in-flight chunk cells not yet sidecar'd —
  they lease-expire and are redone by any worker. Zero data loss; bounded re-work.
- **Snapshot staleness bound:** a restarted worker's view misses at most the cells done
  since the last snapshot upload (≤ ~15 min of fleet output); those are claim-gated in the
  meantime, so the residual re-work window is minutes, not hours.
- **What a fresh operator session re-arms:** the endgame daemon
  (`~/tmp/avifgen/endgame_daemon.sh`, nohup; idempotent — sentinel files gate each phase) +
  a monitor on the sentinels `~/tmp/avifgen/{ENCODE_FINAL,SCORE_FINAL}.done` and
  `ENCODE_STALLED.flag`. Everything else (declares, creds recipe, bring-up scripts,
  writeback) is committed: zenmetrics `scripts/jobsys/` + homefleet
  `zenmetrics/scripts/jobsys/avifgen_*.sh`; env templates in `~/tmp/avifgen/worker.env.*`
  (re-mintable from `~/.config/cloudflare/r2-credentials` in one curl).
- **Gap closed this session:** before `92432e37`, a restarted single-run worker re-encoded
  from an EMPTY view (only claim leases stood between it and full re-work) — that is the
  gap the snapshot fix + daemon refresh loop closes; it is now verified behaviour, not a
  "should".

## Node change + epoch-sharding decision (2026-08-06 ~21:35Z)

- **USER DIRECTIVE executed: `node-3` → Windows default; `node-2` stays an Ubuntu worker.**
  Drained (unit stopped+disabled), PXE flag cleared, rebooted, **verified in Windows**
  (`Microsoft Windows [Version 10.0.26200.8875]`). Standing-rule supersession recorded in
  `NODES.md` (this commit) + homefleet. Rate impact: node-3 carried ~3% of post-fix encode
  throughput — drain ETA moves minutes, not hours. sf-cpu wave capacity drops from 4+tower
  to 3+tower boxes → wave estimate ~4–7 h (was ~3–6 h); not the campaign bound (GPU queue is).
- **Epoch-sharded claiming (zenfleet `88559a45`): NOT adopted for this campaign — stated
  call.** Adoption needs an image ≥`ad1dd3a0`; my images predate it and the sf-cpu wave does
  NOT otherwise need a rebuild (the ledger-in fix is already rolled). Rebuilding now would
  move the baked zensim/feature pin mid-campaign for a 4-worker wave whose claim contention
  is negligible post-snapshot. First live use belongs to a wave that builds fresh images
  anyway.
| 2026-08-06 ~22:05 | **(epochshard lane, FYI):** weighted sharding landed on top of epoch-sharded claiming — `fleet/handicaps.toml` now carries REGISTERED per-box speed multipliers seeded from YOUR attribution audit (`0f9fa073`): the `encode.zenavif` column is the per-box distinct-done/active-hour credit normalized to lianli=1.0 (i265 0.72, ryzen5800xt 0.67, tower 0.67, node-3 0.19; node-2 0.64 low-confidence 10-min sample). Available-not-adopted for this campaign, per your own recorded decision — nothing changes for the running waves (mode still opt-in, default lease, no image rebuild needed or wanted mid-campaign). |
| 2026-08-07 ~03:20 | **ENCODE CLOSED AT 100% OF UNIQUE JOBS — and the stall was an accounting discovery, not lost work.** The daemon's stall flag fired at 562,860 "of 564,300": diagnosis showed the manifest's 564,300 declared cells collapse to **562,860 unique content-addressed jobs** — **4 byte-identical rendition pairs** (8134↔8136, 8210↔8212, 8312↔8314, 8380↔8382 — different origin numbers, identical bytes; 360 main-leg cells each = 1,440 duplicate preimages) which the job system correctly deduplicated at declare time. **Split-leakage audit: all 4 pairs are train/train — zero leakage** (`duplicate_renditions.json`). True residue was **3 cells** (transient `upload_fail`, poisoned at the attempt cap): re-encoded locally through the pinned jobexec (deterministic bytes), blobs content-addressed into the run, one schema-exact DONE sidecar appended (`pass-wsl-manual-gapfill-1.parquet`) — final bridge reads **562,860/562,860 DONE, 0 skipped**. The cell table fans the 4 duplicate names back out: **pairs_final_fanned.parquet = exactly 564,300 cell rows** (every declared cell present; duplicate names share `encode_sha` by construction). Also fixed: my daemon's 99.5% floor constant was typo'd (563,479 vs the correct 561,479) — the hold was the typo, the fleet had actually finished. |
| 2026-08-07 ~03:30 | **SCORE PHASE FULLY LIVE.** Final declares over 534,464 unique (ref, blob) pairs: `avifgen-sf-gpu-20260806` (ssim2-gpu + butteraugli max+pnorm3) and `avifgen-sf-cpu-20260806` (cvvdp native CPU + zensim-foldapp2 944 features) — **45,129 ScoreFile jobs each**. Fleet: node-2 + lianli on the GPU queue (both with `ZEN_REQUIRE_GPU=1`+`ZENMETRICS_REQUIRE_GPU=1`, envs verified); i265 + ryzen5800xt + tower on the CPU queue (AVX2-tier-pure for the 944 features — lianli/AVX-512 excluded from feature extraction by construction). Daemon relaunched into its score-watch loop (snapshots for both runs every ~15 min); monitor re-armed on `SCORE_FINAL.done`. 29,836 cells share bytes with another cell (564,300 − 534,464) — scored once, fanned to all owners at writeback. |

## GPU profiling pass + the OOM storm (user-directed, 2026-08-07 ~05:30-06:30Z)

**dmon (60 s, both boxes, workers live):** lianli sm 32-82% avg ~57%, FB swinging 2.3-7.7 GB;
node-2 sm alternating 85→0→49→0 with FB dropping to 17 MB — **host-stall dominated, worst on
node-2**: the chunked claim path runs each ScoreFile job as a FRESH process (by design, for the
memory bound), so every job pays CUDA context + JIT + scorer-cache init; `ZEN_PERSISTENT_EXEC=1`
is a no-op under chunking (the serial `ZEN_CHUNK_WALL_SEC=0` path is what honors it). nsys was
SKIPPED — the queue was near-drained and the dmon + ledger evidence already identified the
binding defect below; api_sum profiling belongs to the next wave with the levers under test.

**THE REAL FINDING — VRAM OOM storm, 10,291 jobs (23% of pairs):** ledger scan showed 10,291
sf-gpu jobs FAILED (`encoder_panic` mislabel again); manual re-run captured the true error:
`CUDA_ERROR_OUT_OF_MEMORY`. Mechanism: **BoxBudget admission has no VRAM dimension** — unhinted
ScoreFile jobs (512 MB default) admitted 10-16 concurrent GPU processes per 8 GB card.
`ZENMETRICS_REQUIRE_GPU=1` then refused each starved process LOUDLY (correct! no silent CPU
numbers — B6 did its job); the scheduler was the defect. node-2 failed 6,961 / lianli 3,330.
Also recorded: my progressive re-declare created a second job-id generation (chunk boundaries
differ), so ~60% of pairs were scored under earlier-generation ids — wasteful but convergent
(content-addressed rows dedup at writeback).

**RECOVERY (live):** definitive pair coverage computed from the score BLOBS (458,316/534,464
pairs both-metrics-complete; 76,148 missing incl. 8,815 single-metric partials, concentrated in
365 mostly-big sources) → retry declared as **8,580 jobs, CHUNK=9** (new chunk boundaries escape
the poisoned ids) **with ResourceHints (10 GiB / 5 threads)** via the new
`ZEN_SCOREFILE_HINT_MEM_GB/_THREADS` support in `declare_direct_objects.py` — hints bound
admission to ≤2 concurrent GPU jobs/box. **Verified live on lianli: exactly 2 jobexec
processes, FB stable 5.2 GB, sm 23-62%.** Fleet follow-up registered: BoxBudget needs a real
VRAM admission dimension; ScoreFile cost estimates need pixel-aware sizing.

**sf-cpu COMPLETE, zero failures: 45,129/45,129** (cvvdp + zensim-foldapp2 944, AVX2-tier-pure).
Attribution: tower 19,799 / ryzen5800xt 18,224 / i265 7,106 — a MEASURED same-window basis for
the `metric` handicap rows, and it confirms the seed's own warning: the memory-bound metric
ranking differs sharply from encode (i265 0.72-seeded → 0.36 measured; tower 0.67 → 1.00
fleet-max; ryzen 0.67 → 0.92). `fleet/handicaps.toml` metric rows updated with citations
(registry sanity tests green). `gpu_metric` derivation stays TODO **deliberately**: the only
concurrent GPU window so far is OOM-contaminated (done-rates reflect failure storms, not card
speed); derive from the clean hinted-retry window instead.

**Revised ETA: the 30 h GPU bound was wrong by an order of magnitude** — the corpus is
tiny-pair-dominated and the cards run near-5070 parity: sf-gpu reached 85.7% pair coverage in
~2 h despite the OOM storm. Retry wave ≈ 2-5 h at the safe concurrency → campaign close
expected within the day, writeback + manifest + mirrors to follow.
| 2026-08-08 ~14:44Z | Rescue re-declared as the UNION (8,623 jobs: all pairs of the 99 affected renditions + the 2,000-pair stratified verification sample), proportional per-MP hints (2–17.2 GiB; zenmetrics `1406f59c`). Both GPU boxes draining; operator daemon watching the count. |

## THE 13-DAY STALL + REVIVAL (2026-08-08 22:00Z → 2026-08-21) — the recurring silent-waiter class, again

**Honest record: this lane sat dead for 13 days on a silently-failed waiter.** Chronology,
every claim measured from the salvaged ledger/journals on 2026-08-21:

1. **The daemon died lying quietly.** `rescue_daemon.sh` logged healthy counts to 21:47Z
   (4,429/8,623), then four consecutive `done=fail` lines 21:57–22:13Z and stopped. Root
   cause MEASURED at revival: its `timeout 560` on `compact_ledgers.py` — the R2
   ledger-read crossed 560 s as sidecars grew past ~10k files (measured 768 s on
   2026-08-21); the kill left no `distinct_done=` line and the daemon's `tail -1` masked
   ALL stderr, so "fail" carried zero evidence. **This is exactly the evidence-free
   bespoke-waiter class the latency discipline bans** — same class as the 2026-08-06
   4 h orphaned-waiter incident in this very campaign. The owning session then died in
   harness restarts ~22:00Z with no surviving monitor.
2. **The fleet did NOT fail — the workers drained on for 2.5 more days.** lianli
   (now `r7900x`, then carrying the RTX 2080) worked the queue until the scoped cred
   expired **2026-08-11 09:37Z** (last sidecar 09:36Z — to the minute). node-2 had
   already dropped out 2026-08-08 via repeated `status=137` SIGKILL worker deaths.
   Terminal state reached: **8,361/8,623 union jobs DONE (97.0%), residue 262**.
3. **13 days of loud-but-unheard failure.** node-2's `zen-worker.service` crash-looped
   on the expired cred (`SignatureDoesNotMatch`, exit 4, restart counter 246 by
   revival) — the fail-loud design worked; nobody was listening. Meanwhile the fleet
   hosts were renamed (`lianli`→`r7900x`, node-2 heartbeats `i134`), the RTX 2080 left
   r7900x (GTX 1060 6GB now sits there, unenrolled), and the 2026-08-10 s3env
   LAN-default flip made a bare `compact_ledgers.py` resolve to the LAN store where
   this run does not exist (FileNotFoundError) — the salvage read needed `ZEN_STORE=r2`.
4. **Revival (2026-08-21):** truth audit via the canonical `zenfleet-ctl ids` join
   (my hand JSON JobId reconstruction matched 0/8,623 — use the owner); cred re-minted
   (7-day scoped); snapshot refreshed (the stale 08-08 snapshot would have re-done
   ~4,000 jobs); node-2 restarted and closed **90 jobs, 0 failures** before the
   LAN-only directive landed. Replacement daemon staged with stderr capture + a
   3-strike STALLED flag (`~/tmp/avifgen/rescue_daemon2.sh`) — the done=fail class is
   now structurally impossible to repeat silently.
5. **USER DIRECTIVE (mid-revival): LAN-only — R2 is read-only salvage, one pass.**
   Executed: ONE rate-limited `--size-only` salvage sync — 10,306 ledger sidecars
   (82 MB), 8,590 score-row blobs (60 MB → `/mnt/v/zen/writeback-rescue/blobs/`),
   1,545 residue encode blobs (379 MB → `/mnt/v/zen/avifgen-residue-inputs/blobs/`).
   Nothing touches R2 after this pass — no declares, snapshots, mirrors, or heartbeats.
6. **The baked-image blocker, reported + resolved parity-safe.** The pinned GPU image's
   binaries (built 2026-08-06) predate BOTH allow_http fixes (worker s3io `55f8a339`;
   executor objstore in `e7e04994`), so it cannot talk to the plain-http LAN store.
   Resolution kept the metric pin exact: the pinned workspace tree (`1406f59c`)
   already carried the worker fix; the ONE-line executor objstore hunk was applied to
   the pinned tree; both binaries rebuilt from it (separate cargo invocations per the
   JobId preserve_order known-bug) and bind-mounted over the pinned image on node-2 —
   metric code untouched by construction.
7. **Residue re-declared on the LAN store** as a 172-job gap manifest
   (`jobs/avifgen-sf-gpu-rescue2-lan-20260821`; inputs mirrored 1:1 — 1,545 encode
   blobs + 22 refs). First-cell gate PASSED: sidecar + 30 score blobs on the LAN store
   within 3.5 min of container start (patched http GET + PUT proven end-to-end).
8. **Sample verdict (registered AC.R1 decision rule): RESCUE-WINS.** Final (post-drain)
   run: 2,000/2,000 sample pairs checked against stored scores: **3 mismatches, rate
   0.0015** (< the 0.005 escalation threshold; the 3 are additional VRAM-era corruption
   outside the exact-saturation census, cured by the same rescue overlay). No full re-score.
9. **LAN residue drain: COMPLETE in ~9 minutes.** 172/172 jobs done, zero failures, on
   node-2 alone (the "hard tail" was an artifact of the storm-era conditions, not the
   pairs). **JobId drift found and contained:** the pinned-tree (1406f59c) worker build
   computes DIFFERENT JobIds than the Aug-6 fleet binaries + main-repo ctl — the
   JobId::of rev-fragility class the Known Bugs section warns about, observed live
   across revs (not just feature unification). Verified harmless here: pinned-ctl ids
   match the LAN ledger 172/172 and the gap ids are exactly the union residue, so union
   coverage composes to **8,623/8,623**; the writeback joins on (image_path, encode_sha)
   and never on job ids. Do NOT mix worker/ctl builds from different revs on one run.
10. **Close-out chain completed 2026-08-21 ~06:1xZ:** rebuild (74,241 cells patched
    rescue-wins; pre-rescue kept as `scores.pre-rescue.bak.parquet`) → post-rebuild
    census = 3 cells, all rescue-RE-VERIFIED pixel-identical flat-content (rendition
    7048, s4-noqm, q 1/5/30 — legit, not corruption) → G-Z5 re-gate **PASSED 0.999313**
    (per-column: ssim2 0.9975, cvvdp 0.9979, butter_max 0.9950, butter_p3 0.9954) →
    views + view manifest → final corpus `_MANIFEST.json` (sha `1cc50769…e6224397`) →
    Tower mirrors sha-verified. Worker container removed; stale units disabled on both
    GPU boxes; `/opt/zen-lan/` on node-2 keeps the patched pinned binaries + env for
    reference.

| 2026-08-08 ~09:30 | **G-Z5 CAUGHT SILENT GPU-SCORE CORRUPTION — manifest correctly delayed.** The registered default-stratum ssim2 bar missed (0.9876 vs 0.99) and diagnosis of the violators found the real cause: **VRAM-pressure-era garbage** — med/large ladders where ssim2_gpu reads impossible flat 100.0 + butteraugli 0.0 at q≤70 while the SAME pairs' cvvdp (CPU queue) rises perfectly 8.03→10.0 JOD (which also proves pair association + table joins correct). Census: **2,169 impossible-class cells across 99 renditions, 100% med/large** — the pre-hint OOM-window population; exact saturation is a lower bound. cvvdp + 944 features CLEAN. **Rescue running:** all 40,466 pairs of the 99 affected renditions + a 2,000-pair stratified verification sample = 5,573 hinted jobs (`avifgen-sf-gpu-rescue-20260808`), both GPU boxes. Decision rule pre-stated: sample-clean ⇒ patch affected cells (rescue rows win) + re-gate; sample-dirty ⇒ full GPU re-score. Also recorded: the **training-view split is structurally train-only** — every origin in `train_renditions_2026-06-14` ends 0/2/4/6/8 (it IS the June train-split rendition set), so validate/test views are empty by the canonical rule; wave-12 sources val/test elsewhere. Lean columnar writeback landed the 564,300-cell tables (scores 8 cols / features 944; the stock per-row-dict join OOM'd at 53.8 GB uncapped — my run-heavy violation, owned; rebuilt under a 24G cgroup cap at 8.5 GB peak, 1,045 s). |
