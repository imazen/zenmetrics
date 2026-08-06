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
| **Phase** | ENCODE FLEET LIVE (6 nodes) — run `jobs/avifgen-enc-20260806` |
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
