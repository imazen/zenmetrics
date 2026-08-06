# HDR multi-codec corpus build — living status

**Living doc.** Updated continuously by the build's driver/harvest loops while the
run is in flight; it is not a terminal report. The terminal record is the zensim
campaign's registered appendix S plus the corpus `_MANIFEST.json`.

- **Owner:** `claude-hdrcorpus` lane.
- **Started:** 2026-08-05.
- **Mission:** build the HDR phase-2 corpus — HDR sources × multiple codecs ×
  a sweep-discipline grid, with cvvdp-mix targets, scored fleet-parallel.
- **Why:** the existing HDR training leg (`hdr_v3mix` @944) is **single-codec**
  (zenjxl only). The zensim campaign registers this as gap **Q-G6**, and models
  trained on it are codec-blind on HDR. Multi-codec breadth is the named lever.
- **Orchestration:** the zenfleet job system only (workspace mandate — no
  hand-rolled fleet scripts).

---

## Current state

| | |
|---|---|
| **Phase** | grid pre-registered; **blocked on zenfleet HDR extensions** |
| **Blocking on** | B1 (no HDR encode job), B2 (artifact persistence), B6 (fleet cannot guarantee GPU scoring) |
| **Ledger fill** | n/a (not declared yet) |
| **First-cell R2 persistence gate** | NOT YET RUN — hard gate before any fleet scale-up |

### Milestones

- [x] Standing-rule change recorded (node-2/node-3 permanently Ubuntu)
- [x] node-2 + node-3 PXE-flipped to Ubuntu and verified up
- [x] Fleet capability surveyed; blockers identified and registered
- [x] cvvdp GPU-vs-CPU verified; budget doc corrected in place
- [x] Appendix S grid pre-registered
- [ ] zenfleet extended for HDR multi-codec encode + score (**B1/B2/B6**)
- [ ] Workers enrolled on all encode nodes
- [ ] **First-cell artifact-persistence gate PASSED** (encoded bytes + diffmaps + all metric variants confirmed on R2)
- [ ] Fleet scaled to full grid
- [ ] 50% ledger fill
- [ ] Corpus complete + `_MANIFEST.json` + orientation gate + provenance registered

## Blockers (registered as campaign appendix S §S.6)

| id | blocker | consequence |
|---|---|---|
| **B1** | zenfleet has no HDR **encode** job — `JobKind::Encode` carries no HDR flag; HDR encode exists only behind `zenmetrics sweep --hdr` and the jobexec encode path is hardcoded SDR | must extend zenfleet (not bypass) before any encode cell runs |
| **B2** | artifact persistence unmet on the scoring path — a `Metric` job **unlinks** the encode after decoding it, and `JobKind::Diffmap` has no executor | two-stage `Encode`→`ScoreFile` fixes encoded bytes; diffmaps still need an executor |
| **B3** | no `zenav1-svt` codec registered in zenmetrics (encoder itself is ready and byte-gated at 10-bit) | integration gap, not a codec gap |
| **B4** | JPEG-gainmap is decode-only; no gain-map encoder in any sweep path | arm cannot be built until an encoder arm is added |
| **B5** | AVIF arm is **user-halted** pending a zenavif settle-check; checked 2026-08-05 and zenavif is **not settled** (stale marker, uncommitted lockfile, conflicted HDR bookmark) | arm stays out; needs user confirmation, not a workaround |
| **B6** | **the fleet cannot guarantee GPU scoring** — jobexec hardcodes the `auto` GPU runtime at four sites with no override, and `auto`'s CPU rung records a CPU number under the GPU column name at exit 0 with no log | "scored on GPU" is unfalsifiable until fixed; blocking for the metric half |
| **B7** | executor image must carry the `hdr` (and `png`) features | image rebuild, canonical name + new tag |

---

## Node roster and roles

Neutral IDs only (per repo privacy policy); identifying detail lives in the
private fleet repo.

| Node | Cores | RAM | GPU | Role in this build |
|---|---|---|---|---|
| `node-2` | 16 | 30 G | RTX 3070 8 GB | **GPU metric scoring** + encode |
| `lianli` | 24 | 29 G | RTX 2080 8 GB | **GPU metric scoring** + encode |
| `node-3` | 12 | 41 G | GTX 1660 Ti 6 GB | encode (GPU is spare capacity, not a directed scoring role) |
| `i265` | 20 | 30 G | none | encode |
| `ryzen5800xt` | — | — | — | encode (pending probe) |
| `tower` | 32 T | 62 G | none | encode only if it fits its rules: Docker-only, capped, media-first, observe-before-load |
| `wsl` | 28 | 58 G | — | operator — drives the fleet, does not measure |

**GPU metric scoring is directed to `node-2` and `lianli`** (user directive).

### Design consequence: the GPU metric queue is the binding constraint

The budget lane's headline finding is that **encode wall time is trivial**
(~3.4 min fleet-wide across the 76 HDR sources) while **the GPU metric queue is
the real cost** (~11.5 h). So the grid is sized against GPU-metric throughput on
`node-2` + `lianli`, **not** against encode capacity. Adding encode nodes buys
almost nothing; the levers that matter are metric-queue depth, per-cell metric
count, and total cell count.

---

## Source estate

- **Path:** `/mnt/v/output/imazen-26-hdr-grid-2026-06-14/`
- **1,140 HDR reference PNGs = 76 origins × 15 aspect-preserving scales**, 7.8 GB.
- Scale ladder spans ~`96x128` (tiny, ~12 kpx) to `3072x2304` (~7 MP) — a
  log-spaced size axis that already satisfies the sweep discipline's size
  requirement including the tiny bucket, so **no new resampling is needed**.
- Content classes by origin: nature 47, interiors 20, general 6, food 3.
- These are the same origins the existing single-codec HDR leg was built from,
  which keeps the new multi-codec corpus joinable to it.

---

## Incidents / decisions log

| When (UTC) | What |
|---|---|
| 2026-08-05 ~04:41 | Claimed `zensim` / `zenmetrics` / `homefleet`. |
| 2026-08-05 ~04:43 | Standing-rule change recorded: node-2/node-3 permanently Ubuntu; flip-back rule revoked. Public record neutral-ID-only. |
| 2026-08-05 ~04:44 | Both nodes found in their **other OS** with **no worker flags set** — i.e. the pre-directive default. Set both to boot the Ubuntu worker (procedure in the private fleet repo). |
| 2026-08-05 ~04:44 | The flip helper's own reboot step failed on both boxes ("couldn't reach…") — the **documented** gotcha for a box sitting in its other OS. Used the documented admin-account reboot fallback. Cleared known-hosts entries **before** polling, per the changed-key gotcha that has previously cost ~10 min of false "box never came up". |
| 2026-08-05 ~04:46 | Both nodes up in Ubuntu in ~75 s, GPUs enumerated, driver 595.84. |
| 2026-08-05 ~04:47 | Noted: the pre-existing `zensim/scripts/hdr/hdr_score_fleet.sh` is a **hand-rolled** fleet scorer predating the zenfleet mandate. Read as the HDR scoring contract, **not** extended — this build goes through zenfleet. |
| 2026-08-05 ~04:51 | **Boot-order misdiagnosis corrected.** A report that node-3's firmware had regressed to other-OS-first was investigated and not reproduced: both boxes read PXE first and both showed `BootCurrent` = the PXE entry. The real cause was that every per-box worker flag was cleared — PXE runs, finds no flag, chainloads the other OS by design. Verified with a full reboot cycle on node-3 (boot ID changed, still PXE, returned to Ubuntu unattended). **No firmware change was needed or made.** |
| 2026-08-05 ~04:52 | Yank-rule detection applied per the new user directive: node-3 had **no interactive session** (only a `manager`-class session and the operator's own SSH; no display server) ⇒ yanked. Detection method + observed state recorded in `NODES.md`. |
| 2026-08-05 ~04:54 | **cvvdp GPU claim verified.** The budget doc's `cvvdp-gpu == cvvdp-CPU` row was challenged as possibly CPU-vs-CPU. Re-measured with `nvidia-smi` sampled through each run: cvvdp-gpu allocates real device memory and shows nonzero utilization at every size ⇒ **the row is genuinely GPU and stands**. Peak GPU utilization is only 2-10%, which *measures* the CPU-prep-bound mechanism the row asserted. |
| 2026-08-05 ~04:54 | **Memory ceiling answered:** ~222 MiB/MP (cvvdp-gpu) and ~348 MiB/MP (ssim2-gpu) ⇒ ~1.6 GB and ~2.5 GB at the 7.08 MP top tier. **Every ladder size fits the 8 GB fleet cards**; no size tier is excluded from GPU scoring. |
| 2026-08-05 ~04:55 | **Silent-fallback defect confirmed (B6).** With the GPU hidden, explicit `--gpu-runtime cuda` exits 1 and refuses ("explicit backend requests never fall back"), but the **default** `auto` exits 0 and emits the **GPU column name** from a CPU computation. A code audit found jobexec hardcodes `auto` at four sites with no override, so the explicit-flag mitigation **does not reach the fleet path**. Budget doc annotated in place; raw probe data committed. |
| 2026-08-05 ~04:58 | Appendix S pre-registered in the zensim campaign (grid, persistence contract, gates, and all seven blockers stated as measured facts). |
